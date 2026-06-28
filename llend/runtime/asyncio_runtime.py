"""AsyncioRuntime — v0 in-process agent execution backend.

Built on plain ``asyncio``: no LangGraph, no Redis, no Celery.
Agents are asyncio tasks exchanging ``Message`` objects via ``asyncio.Queue``.
The Orchestrator is the hub — all messages route through it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from llend.runtime.base import AgentRuntime
from llend.runtime.checkpoint import Checkpoint
from llend.runtime.lifecycle import (
    AgentState,
    AgentType,
    is_alive,
    transition,
)
from llend.runtime.message import AgentErrorCode, Message, MsgType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal handle
# ---------------------------------------------------------------------------


@dataclass
class _AgentHandle:
    """Runtime-internal bookkeeping for one agent instance."""

    instance_id: str
    agent_type: str
    state: AgentState = AgentState.INIT
    queue: asyncio.Queue[Message] = field(default_factory=asyncio.Queue)
    context: dict = field(default_factory=dict)
    spawned_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Background tasks
    main_task: asyncio.Task[None] | None = None
    heartbeat_task: asyncio.Task[None] | None = None

    # Interrupt machinery — set when the agent is paused
    interrupt_future: asyncio.Future[str] | None = None


# ---------------------------------------------------------------------------
# AsyncioRuntime
# ---------------------------------------------------------------------------


class AsyncioRuntime(AgentRuntime):
    """v0 in-process runtime using ``asyncio.Queue`` for message routing.

    Parameters
    ----------
    heartbeat_interval:
        Seconds between ``agent.heartbeat`` pings from each agent.
        Default 30 s per Spec 001.
    checkpoint_ttl:
        Default TTL (seconds) for interrupt checkpoints.  Default 86 400 (24 h).
    """

    def __init__(
        self,
        heartbeat_interval: float = 30.0,
        checkpoint_ttl: int = 86400,
    ) -> None:
        self._agents: dict[str, _AgentHandle] = {}
        self._orchestrator_id: str | None = None
        self._session_id: UUID | None = None
        self._heartbeat_interval = heartbeat_interval
        self._checkpoint_ttl = checkpoint_ttl
        self._closed = False

    # ------------------------------------------------------------------
    # Public API (AgentRuntime contract)
    # ------------------------------------------------------------------

    async def spawn(self, agent_type: str, context: dict) -> str:
        """Create and start a new agent instance.  Returns the instance_id."""
        self._guard_closed()

        instance_id = f"{agent_type}-{uuid4().hex[:8]}"
        handle = _AgentHandle(instance_id=instance_id, agent_type=agent_type, context=context)

        if agent_type == AgentType.ORCHESTRATOR.value:
            self._orchestrator_id = instance_id
            # The orchestrator defines the session
            self._session_id = UUID(hex=uuid4().hex)

        # INIT → RUNNING
        handle.state = transition(AgentState.INIT, AgentState.RUNNING)
        self._agents[instance_id] = handle

        # Start periodic heartbeat
        handle.heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(handle),
            name=f"heartbeat-{instance_id}",
        )

        logger.info("spawned agent instance_id=%s type=%s", instance_id, agent_type)
        return instance_id

    async def send(self, message: Message) -> None:
        """Route *message* to its recipient's inbox queue."""
        self._guard_closed()

        # Drop expired messages and notify sender
        if message.is_expired:
            await self._handle_expired_message(message)
            return

        instance_id = self._resolve_recipient(message)
        if instance_id is None:
            logger.warning(
                "message dropped — no route for recipient=%s instance=%s msg_type=%s",
                message.recipient,
                message.recipient_instance,
                message.msg_type.value,
            )
            return

        handle = self._agents[instance_id]
        await handle.queue.put(message)

    async def interrupt(self, instance_id: str, prompt: str, options: list[str]) -> str:
        """Pause *instance_id*, ask human, block until response.  Returns chosen option."""
        self._guard_closed()
        handle = self._get_handle(instance_id)

        # Persist checkpoint in memory
        checkpoint = Checkpoint(
            interrupt_id=uuid4(),
            session_id=self._session_id or uuid4(),
            agent_instance=instance_id,
            agent_type=handle.agent_type,
            task_context=handle.context,
            interrupt_message=prompt,
            interrupt_options=options,
            ttl_seconds=self._checkpoint_ttl,
        )
        handle._last_checkpoint = checkpoint

        # Transition to INTERRUPT
        handle.state = transition(handle.state, AgentState.INTERRUPT)

        # Create a future that will be resolved by resolve_interrupt()
        handle.interrupt_future = asyncio.get_event_loop().create_future()

        logger.info("interrupt raised instance_id=%s prompt=%r", instance_id, prompt[:120])

        # Block until human responds (or future is cancelled)
        try:
            decision = await handle.interrupt_future
        finally:
            handle.interrupt_future = None

        # Read checkpoint from memory
        updated = getattr(handle, "_last_checkpoint", None)
        human_note = updated.human_response if updated else None

        handle.context["interrupt_decision"] = decision
        handle.context["interrupt_note"] = human_note

        # Inject INTERRUPT_RESPONSE into the agent's inbox
        response_msg = Message(
            session_id=checkpoint.session_id,
            sender="runtime",
            sender_instance="runtime",
            recipient=handle.agent_type,
            recipient_instance=instance_id,
            msg_type=MsgType.INTERRUPT_RESPONSE,
            payload={"decision": decision, "human_note": human_note},
            parent_id=checkpoint.interrupt_id,
        )
        await handle.queue.put(response_msg)

        # Resume agent
        handle.state = transition(handle.state, AgentState.RUNNING)
        logger.info("interrupt resolved instance_id=%s decision=%s", instance_id, decision)

        return decision

    async def kill(self, instance_id: str) -> None:
        """Terminate *instance_id*.  Idempotent — calling on a dead agent is a no-op."""
        handle = self._agents.get(instance_id)
        if handle is None or handle.state == AgentState.DEAD:
            return  # already dead or never existed — idempotent

        # Cancel any pending interrupt future
        if handle.interrupt_future is not None and not handle.interrupt_future.done():
            handle.interrupt_future.cancel("agent killed")

        # Cancel and await background tasks so they clean up properly
        for task in (handle.main_task, handle.heartbeat_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        handle.state = AgentState.DEAD
        logger.info("killed agent instance_id=%s", instance_id)

    async def shutdown(self) -> None:
        """Kill all agents and mark the runtime closed."""
        for instance_id in list(self._agents.keys()):
            await self.kill(instance_id)

        self._agents.clear()
        self._closed = True
        logger.info("runtime shutdown complete")

    # ------------------------------------------------------------------
    # v0-specific helpers (not on the ABC — for tests / external callers)
    # ------------------------------------------------------------------

    async def resolve_interrupt(
        self, instance_id: str, decision: str, note: str | None = None
    ) -> None:
        """Resolve a pending interrupt on *instance_id*.

        This is the callback that a human-input channel (CLI, web hook, …)
        calls to feed a decision back into the runtime.
        """
        handle = self._get_handle(instance_id)

        if handle.interrupt_future is None:
            raise RuntimeError(f"Agent {instance_id} has no pending interrupt")

        if handle.interrupt_future.done():
            raise RuntimeError(f"Agent {instance_id} interrupt already resolved")

        # Update the in-memory checkpoint
        cp = getattr(handle, "_last_checkpoint", None)
        if cp is not None:
            cp.human_response = note or decision
            cp.resolved_at = datetime.now(UTC)

        handle.interrupt_future.set_result(decision)

    @property
    def agent_count(self) -> int:
        """Return the number of tracked agent instances (all states)."""
        return len(self._agents)

    @property
    def is_closed(self) -> bool:
        """Return True after ``shutdown()`` has been called."""
        return self._closed

    @property
    def session_id(self) -> UUID | None:
        """Return the current session id, or None before the orchestrator spawns."""
        return self._session_id

    def get_handle_state(self, instance_id: str) -> AgentState:
        """Return the current lifecycle state of *instance_id*."""
        return self._get_handle(instance_id).state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _guard_closed(self) -> None:
        """Raise RuntimeError if the runtime has been shut down."""
        if self._closed:
            raise RuntimeError("AsyncioRuntime is closed — cannot accept new work")

    def _get_handle(self, instance_id: str) -> _AgentHandle:
        """Return the handle for *instance_id*, or raise KeyError."""
        handle = self._agents.get(instance_id)
        if handle is None:
            raise KeyError(f"No agent with instance_id={instance_id!r}")
        return handle

    def _resolve_recipient(self, message: Message) -> str | None:
        """Map a message's routing fields to a concrete instance_id."""
        # Exact instance match
        if message.recipient_instance is not None:
            return (
                message.recipient_instance if message.recipient_instance in self._agents else None
            )

        # Orchestrator is the default hub
        if message.recipient == AgentType.ORCHESTRATOR.value:
            return self._orchestrator_id

        # Find first agent of the given type (v0: single responder, …)
        for handle in self._agents.values():
            if handle.agent_type == message.recipient:
                return handle.instance_id

        return None

    async def _handle_expired_message(self, message: Message) -> None:
        """Send an AGENT_ERROR back to the sender about an expired message."""
        error_msg = Message(
            session_id=message.session_id,
            sender="runtime",
            sender_instance="runtime",
            recipient=message.sender,
            recipient_instance=message.sender_instance,
            msg_type=MsgType.AGENT_ERROR,
            payload={
                "error_code": AgentErrorCode.TIMEOUT.value,
                "detail": f"Message {message.id} expired unread",
                "recoverable": True,
            },
            parent_id=message.id,
        )
        await self.send(error_msg)
        logger.info("message %s expired — sending error back to %s", message.id, message.sender)

    async def _heartbeat_loop(self, handle: _AgentHandle) -> None:
        """Periodically send AGENT_HEARTBEAT to the orchestrator while alive."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)

            # Exit if agent is no longer alive or runtime was shut down
            if not is_alive(handle.state) or self._closed:
                return

            orchestrator_id = self._orchestrator_id
            if orchestrator_id is None or orchestrator_id not in self._agents:
                continue

            heartbeat = Message(
                session_id=self._session_id or uuid4(),
                sender=handle.agent_type,
                sender_instance=handle.instance_id,
                recipient=AgentType.ORCHESTRATOR.value,
                msg_type=MsgType.AGENT_HEARTBEAT,
                payload={},
            )
            orch_handle = self._agents[orchestrator_id]
            await orch_handle.queue.put(heartbeat)
