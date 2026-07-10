"""AsyncioRuntime — lightweight in-process agent execution backend.

Built on plain ``asyncio``: no LangGraph, no Redis, no Celery.
Agents are asyncio tasks exchanging ``Message`` objects via ``asyncio.Queue``.
The Orchestrator is the hub (§2.4) — all messages route through it.

This is the **lightweight alternative** to ``LangGraphRuntime`` (§3.1).
Use it for simple use cases where you don't need LangGraph's state graph
and built-in checkpointing.

Spec reference
==============
- **§3.1** — lightweight alternative backend, same ABC contract
- **§3.2** — implements ``AgentRuntime`` (spawn, send, interrupt, kill, shutdown)
- **§3.3** — agent lifecycle state machine via ``lifecycle.transition()``
- **§3.4** — checkpoint persistence (disk), notification, TTL enforcement
- **§3.5** — concurrency via ``asyncio.Queue`` + per-agent heartbeat tasks
- **§2.3** — message expiry handling in ``_handle_expired_message``
- **§2.4** — routing rule in ``_resolve_recipient`` (Orchestrator is the hub)
- **§2.2** — ``agent.heartbeat`` every 30s idle (§3.5)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from llend.runtime.base import AgentRuntime, MessageHandler
from llend.runtime.checkpoint import (
    Checkpoint,
    InterruptTimeoutError,
)
from llend.runtime.lifecycle import (
    AgentState,
    AgentType,
    is_alive,
    transition,
)
from llend.runtime.message import AgentErrorCode, Message, MsgType
from llend.runtime.notifications import (
    ConsoleNotificationChannel,
    NotificationChannel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal handle  —  §3.3 per-agent bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _AgentHandle:
    """Runtime-internal bookkeeping for one agent instance.  §3.3.

    Tracks the agent's lifecycle state, inbox queue (§3.5), background
    tasks (heartbeat per §2.2 / §3.5), and interrupt machinery (§3.4).
    """

    instance_id: str
    agent_type: str
    state: AgentState = AgentState.INIT
    queue: asyncio.Queue[Message] = field(default_factory=asyncio.Queue)
    context: dict = field(default_factory=dict)
    spawned_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Background tasks (§3.5 concurrency)
    main_task: asyncio.Task[None] | None = None
    heartbeat_task: asyncio.Task[None] | None = None       # §2.2 agent.heartbeat

    # Interrupt machinery — set when agent is paused (§3.4 step 4)
    interrupt_future: asyncio.Future[str] | None = None

    # Message handler — registered by agent via register_handler() (Spec 003 §5.1)
    handler: MessageHandler | None = None


# ---------------------------------------------------------------------------
# AsyncioRuntime  —  Spec 001 §3.1 (lightweight alternative)
# ---------------------------------------------------------------------------


class AsyncioRuntime(AgentRuntime):
    """Lightweight in-process runtime using ``asyncio.Queue`` for message routing.  §3.1.

    Parameters
    ----------
    heartbeat_interval:
        Seconds between ``agent.heartbeat`` pings from each agent.
        Default 30 s per §2.2 / §3.5.
    checkpoint_ttl:
        Default TTL (seconds) for interrupt checkpoints.  §3.4 — default 86 400 (24 h).
        Per §3.4 Q1, configurable per interrupt via ``Checkpoint.ttl_seconds``.
    notification_channel:
        Where to send human notifications on interrupt.  §3.4 step 3.
        Defaults to ``ConsoleNotificationChannel`` (stdout).
    data_dir:
        Base directory for checkpoint persistence.  §3.4 — ``~/.llend``.
    ttl_check_interval:
        Seconds between TTL expiry scans.  §3.4 ¶2 — default 10 s.
    """

    def __init__(
        self,
        heartbeat_interval: float = 30.0,          # §2.2 / §3.5: 30s default
        checkpoint_ttl: int = 86400,               # §3.4: 24h default
        notification_channel: NotificationChannel | None = None,  # §3.4 step 3
        data_dir: Path | None = None,              # §3.4: ~/.llend
        ttl_check_interval: float = 10.0,          # §3.4 ¶2: TTL scan interval
    ) -> None:
        self._agents: dict[str, _AgentHandle] = {}
        self._orchestrator_id: str | None = None   # §2.4: Orchestrator is the hub
        self._session_id: UUID | None = None       # §2.1: session_id
        self._heartbeat_interval = heartbeat_interval
        self._checkpoint_ttl = checkpoint_ttl
        self._notification_channel = notification_channel or ConsoleNotificationChannel()
        self._data_dir = Path(data_dir) if data_dir else Path.home() / ".llend"
        self._ttl_check_interval = ttl_check_interval
        self._closed = False

        # Agent factory registry  §5.2 (Spec 005)
        self._agent_factories: dict[str, Callable[..., Any]] = {}

        # Active interrupt checkpoints keyed by instance_id (§3.4)
        self._checkpoints: dict[str, Checkpoint] = {}

        # Background TTL monitor (§3.4 ¶2)
        self._ttl_monitor_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public API — AgentRuntime contract  §3.2
    # ------------------------------------------------------------------

    async def spawn(self, agent_type: str, context: dict) -> str:
        """Create and start a new agent instance.  Returns the instance_id.  §3.2.

        Follows the lifecycle diagram in §3.3: INIT → RUNNING.
        Starts the heartbeat loop (§2.2 / §3.5).
        """
        self._guard_closed()

        instance_id = f"{agent_type}-{uuid4().hex[:8]}"
        handle = _AgentHandle(instance_id=instance_id, agent_type=agent_type, context=context)

        if agent_type == AgentType.ORCHESTRATOR.value:
            self._orchestrator_id = instance_id
            # The orchestrator defines the session (§2.1)
            self._session_id = UUID(hex=uuid4().hex)

        # §3.3: INIT → RUNNING
        handle.state = transition(AgentState.INIT, AgentState.RUNNING)
        self._agents[instance_id] = handle

        # §2.2: Start periodic heartbeat (agent.heartbeat, idle > 30s)
        handle.heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(handle),
            name=f"heartbeat-{instance_id}",
        )

        # §3.4: Ensure TTL monitor is running
        self._ensure_ttl_monitor()

        # §5.2 (Spec 005): If a factory is registered for this agent type,
        # launch the agent's processing loop as a background task.
        if agent_type in self._agent_factories:
            handle.main_task = asyncio.create_task(
                self._agent_factories[agent_type](handle, context),
                name=f"agent-{instance_id}",
            )

        logger.info("spawned agent instance_id=%s type=%s", instance_id, agent_type)
        return instance_id

    def register_agent_type(self, agent_type: str, factory: Callable[..., Any]) -> None:
        """Register a factory that creates an agent's processing task.  Spec 005 §5.2.

        When ``spawn(agent_type, context)`` is called, the runtime creates the
        agent handle (queue, lifecycle state, heartbeat) and then launches the
        factory as a background ``asyncio.Task``::

            factory(handle, context)

        The factory receives the ``_AgentHandle`` (so it has access to the
        inbox queue) and the spawn ``context`` dict.  It is expected to run
        until the agent completes, errors, or is killed.
        """
        self._agent_factories[agent_type] = factory

    async def send(self, message: Message) -> None:
        """Route *message* to its recipient's inbox queue.  §3.2, §2.4.

        §2.3: Expired messages are dropped — the sender receives
        ``agent.error(TIMEOUT)`` via ``_handle_expired_message``.
        §2.4: Routing follows the Orchestrator-is-hub rule.
        """
        self._guard_closed()

        # §2.3: Drop expired messages and notify sender
        if message.is_expired:
            await self._handle_expired_message(message)
            return

        # §2.4: Resolve recipient via routing table
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

        # Spec 003 §5.1: Fire registered handler (fire-and-forget to avoid reentrancy)
        if handle.handler is not None:
            asyncio.create_task(
                handle.handler(message),
                name=f"handler-{instance_id}",
            )

    async def interrupt(self, instance_id: str, prompt: str, options: list[str]) -> str:
        """Pause *instance_id*, ask human, block until response.  §3.2, §3.4.

        Follows the 5-step interrupt flow from §3.4 ¶1:
        1. Freezes agent state → creates ``Checkpoint``
        2. Saves checkpoint to disk (§3.4 ¶2)
        3. Notifies human via configured channel (§3.4 step 3)
        4. Blocks the agent on ``asyncio.Future`` (§3.4 step 4)
        5. On response: injects ``interrupt.response``, resumes (§3.4 step 5)

        Raises ``InterruptTimeoutError`` if TTL expires (§3.4 ¶2).
        """
        self._guard_closed()
        handle = self._get_handle(instance_id)

        # §3.4 step 1: Freeze agent state
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
        # §3.4 step 2: Save to disk
        checkpoint.save(self._data_dir)
        self._checkpoints[instance_id] = checkpoint

        # §3.3: Transition to INTERRUPT
        handle.state = transition(handle.state, AgentState.INTERRUPT)

        # §3.4 step 3: Notify human
        await self._notification_channel.notify_interrupt(checkpoint)

        # §3.4 step 4: Block the agent (not the whole runtime — §3.5)
        handle.interrupt_future = asyncio.get_event_loop().create_future()

        logger.info(
            "interrupt raised instance_id=%s interrupt_id=%s prompt=%r",
            instance_id,
            checkpoint.interrupt_id,
            prompt[:120],
        )

        # Block until human responds (or TTL expires via §3.4 ¶2)
        try:
            decision = await handle.interrupt_future
        except InterruptTimeoutError:
            raise
        finally:
            handle.interrupt_future = None

        # §3.4 step 5: On response — update checkpoint, inject interrupt.response
        updated = self._checkpoints.pop(instance_id, checkpoint)
        human_note = updated.human_response if updated else None

        if not updated.is_resolved:
            updated.human_response = human_note or decision
            updated.resolved_at = datetime.now(UTC)
        updated.save(self._data_dir)

        handle.context["interrupt_decision"] = decision
        handle.context["interrupt_note"] = human_note

        # §3.4 step 5: inject INTERRUPT_RESPONSE into agent's inbox (§2.2)
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

        # §3.3: Resume agent — INTERRUPT → RUNNING
        handle.state = transition(handle.state, AgentState.RUNNING)
        logger.info("interrupt resolved instance_id=%s decision=%s", instance_id, decision)

        return decision

    async def register_handler(
        self, instance_id: str, handler: MessageHandler
    ) -> None:
        """Register a message handler for *instance_id*.  Spec 003 §5.1.

        Replaces any previously registered handler.  The handler is invoked
        via ``asyncio.create_task`` (fire-and-forget) so the runtime's
        ``send()`` is never blocked by handler logic.
        """
        self._guard_closed()
        handle = self._get_handle(instance_id)
        handle.handler = handler

    async def kill(self, instance_id: str) -> None:
        """Terminate *instance_id*.  Idempotent — calling on a dead agent is a no-op.  §3.2.

        §3.3: any state → DEAD.
        Cleans up pending interrupt futures and checkpoint files (§3.4).
        """
        handle = self._agents.get(instance_id)
        if handle is None or handle.state == AgentState.DEAD:
            return  # §3.2: idempotent

        # Cancel any pending interrupt future (§3.4)
        if handle.interrupt_future is not None and not handle.interrupt_future.done():
            handle.interrupt_future.cancel("agent killed")

        # Clean up checkpoint file (§3.4)
        cp = self._checkpoints.pop(instance_id, None)
        if cp is not None:
            try:
                cp.delete(self._data_dir)
            except OSError:
                logger.debug("failed to delete checkpoint file for %s", instance_id)

        # Cancel and await background tasks so they clean up properly
        for task in (handle.main_task, handle.heartbeat_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # §3.3: → DEAD
        handle.state = AgentState.DEAD
        logger.info("killed agent instance_id=%s", instance_id)

    async def shutdown(self) -> None:
        """Kill all agents and mark the runtime closed.

        Stops the TTL monitor first (§3.4), then kills all agents (§3.2).
        """
        # Stop TTL monitor first (§3.4 ¶2)
        if self._ttl_monitor_task is not None and not self._ttl_monitor_task.done():
            self._ttl_monitor_task.cancel()
            try:
                await self._ttl_monitor_task
            except asyncio.CancelledError:
                pass
            self._ttl_monitor_task = None

        for instance_id in list(self._agents.keys()):
            await self.kill(instance_id)

        self._agents.clear()
        self._checkpoints.clear()
        self._closed = True
        logger.info("runtime shutdown complete")

    # ------------------------------------------------------------------
    # v0-specific helpers (not on the ABC — for tests / external callers)
    # ------------------------------------------------------------------

    async def resolve_interrupt(
        self, instance_id: str, decision: str, note: str | None = None
    ) -> None:
        """Resolve a pending interrupt on *instance_id*.  §3.4 step 5.

        This is the callback that a human-input channel (CLI, web hook, …)
        calls to feed a decision back into the runtime.  Updates the
        checkpoint on disk and unblocks the ``interrupt()`` future.
        """
        handle = self._get_handle(instance_id)

        if handle.interrupt_future is None:
            raise RuntimeError(f"Agent {instance_id} has no pending interrupt")

        if handle.interrupt_future.done():
            raise RuntimeError(f"Agent {instance_id} interrupt already resolved")

        # Update the checkpoint on disk (§3.4 step 5)
        cp = self._checkpoints.get(instance_id)
        if cp is not None:
            cp.human_response = note or decision
            cp.resolved_at = datetime.now(UTC)
            cp.save(self._data_dir)

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
        """Return the current session id (§2.1), or None before the orchestrator spawns."""
        return self._session_id

    def get_handle_state(self, instance_id: str) -> AgentState:
        """Return the current lifecycle state of *instance_id* (§3.3)."""
        return self._get_handle(instance_id).state

    def get_checkpoint(self, instance_id: str) -> Checkpoint | None:
        """Return the active checkpoint for *instance_id* (§3.4), if any."""
        return self._checkpoints.get(instance_id)

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

    # ------------------------------------------------------------------
    # Routing  —  Spec 001 §2.4
    # ------------------------------------------------------------------

    def _resolve_recipient(self, message: Message) -> str | None:
        """Map a message's routing fields to a concrete instance_id.  §2.4.

        Resolution order (§2.4):
        1. Exact ``recipient_instance`` match
        2. ``recipient == "orchestrator"`` → route to the hub
        3. First agent of matching type (v0: single responder, …)
        """
        # 1. Exact instance match (§2.1: recipient_instance)
        if message.recipient_instance is not None:
            return (
                message.recipient_instance if message.recipient_instance in self._agents else None
            )

        # 2. Orchestrator is the default hub (§2.4)
        if message.recipient == AgentType.ORCHESTRATOR.value:
            return self._orchestrator_id

        # 3. Find first agent of the given type (§2.4: no peer-to-peer)
        for handle in self._agents.values():
            if handle.agent_type == message.recipient:
                return handle.instance_id

        return None

    # ------------------------------------------------------------------
    # Message expiry  —  Spec 001 §2.3
    # ------------------------------------------------------------------

    async def _handle_expired_message(self, message: Message) -> None:
        """Send an AGENT_ERROR back to the sender about an expired message.  §2.3.

        §2.3 step 1: log the expiry
        §2.3 step 2: send ``agent.error(TIMEOUT, recoverable=True)`` back to sender
        §2.3 step 3: the sender decides — retry / escalate
        """
        error_msg = Message(
            session_id=message.session_id,
            sender="runtime",
            sender_instance="runtime",
            recipient=message.sender,
            recipient_instance=message.sender_instance,
            msg_type=MsgType.AGENT_ERROR,
            payload={
                "error_code": AgentErrorCode.TIMEOUT.value,    # §2.2.1: TIMEOUT
                "detail": f"Message {message.id} expired unread",
                "recoverable": True,                           # §2.3 step 3: sender can retry
            },
            parent_id=message.id,
        )
        await self.send(error_msg)
        logger.info("message %s expired — sending error back to %s", message.id, message.sender)

    # ------------------------------------------------------------------
    # Heartbeat  —  Spec 001 §2.2 / §3.5
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self, handle: _AgentHandle) -> None:
        """Periodically send ``agent.heartbeat`` to the orchestrator while alive.  §2.2.

        §2.2: "Still alive (if idle > 30s)" — default 30s interval.
        §3.5: runs as background task per agent, independent of message processing.
        """
        while True:
            await asyncio.sleep(self._heartbeat_interval)

            # Exit if agent is no longer alive (§3.3) or runtime was shut down
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
                msg_type=MsgType.AGENT_HEARTBEAT,       # §2.2: agent.heartbeat
                payload={},
            )
            orch_handle = self._agents[orchestrator_id]
            await orch_handle.queue.put(heartbeat)

    # ------------------------------------------------------------------
    # TTL enforcement  —  Spec 001 §3.4 ¶2
    # ------------------------------------------------------------------

    def _ensure_ttl_monitor(self) -> None:
        """Start the TTL monitor background task if not already running.  §3.4 ¶2.

        Lazily started on first ``spawn()`` to avoid unnecessary background
        work when no agents exist.
        """
        if self._ttl_monitor_task is None or self._ttl_monitor_task.done():
            self._ttl_monitor_task = asyncio.create_task(
                self._ttl_monitor_loop(), name="ttl-monitor"
            )

    async def _ttl_monitor_loop(self) -> None:
        """Periodically scan active checkpoints and auto-terminate expired ones.  §3.4 ¶2.

        §3.4 ¶2: "If human doesn't respond in TTL (default 24h), interrupt
        times out → ERROR state → Orchestrator decides: retry / skip / escalate."
        """
        while not self._closed:
            try:
                await asyncio.sleep(self._ttl_check_interval)
                await self._enforce_ttl()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("TTL monitor iteration failed")

    async def _enforce_ttl(self) -> None:
        """Check all active checkpoints (§3.4) and terminate any that have expired."""
        for instance_id, cp in list(self._checkpoints.items()):
            if cp.is_expired and not cp.is_resolved:
                await self._terminate_expired_interrupt(instance_id)

    async def _terminate_expired_interrupt(self, instance_id: str) -> None:
        """Force-terminate an interrupt whose TTL has expired.  §3.4 ¶2.

        Steps:
        1. Mark checkpoint resolved with ``__timeout__`` sentinel and save to disk
        2. Notify human channel (§3.4 step 3 — timeout variant)
        3. Transition agent INTERRUPT → ERROR (§3.3)
        4. Raise ``InterruptTimeoutError`` via the future → ``interrupt()`` raises
        5. Send ``agent.error(INTERRUPT_TIMEOUT)`` to the Orchestrator (§2.2)
        """
        handle = self._agents.get(instance_id)
        cp = self._checkpoints.pop(instance_id, None)

        if cp is None:
            return

        # 1. Save timeout to disk (§3.4)
        cp.human_response = "__timeout__"
        cp.resolved_at = datetime.now(UTC)
        try:
            cp.save(self._data_dir)
        except OSError:
            logger.debug("failed to save timeout checkpoint for %s", instance_id)

        # 2. Notify human (§3.4 step 3)
        await self._notification_channel.notify_interrupt_timeout(cp)

        # 3. Transition agent to ERROR (§3.3: INTERRUPT → ERROR)
        if handle is not None and handle.state == AgentState.INTERRUPT:
            handle.state = transition(handle.state, AgentState.ERROR)

            # 4. Cancel the interrupt future (§3.4 ¶2)
            if handle.interrupt_future is not None and not handle.interrupt_future.done():
                handle.interrupt_future.set_exception(
                    InterruptTimeoutError(cp.interrupt_id, cp.ttl_seconds)
                )

            # 5. Send error to orchestrator (§2.2: agent.error → Orch)
            orch_id = self._orchestrator_id
            if orch_id is not None:
                error_msg = Message(
                    session_id=cp.session_id,
                    sender="runtime",
                    sender_instance="runtime",
                    recipient=AgentType.ORCHESTRATOR.value,
                    recipient_instance=orch_id,
                    msg_type=MsgType.AGENT_ERROR,
                    payload={
                        "error_code": AgentErrorCode.INTERRUPT_TIMEOUT.value,  # §2.2.1
                        "detail": (
                            f"Interrupt {cp.interrupt_id} for {instance_id} "
                            f"timed out after {cp.ttl_seconds}s"
                        ),
                        "recoverable": False,           # §3.4: terminated, not retryable
                        "interrupt_id": str(cp.interrupt_id),
                    },
                    parent_id=cp.interrupt_id,           # §2.5: reply chain
                )
                await self.send(error_msg)

        logger.info(
            "interrupt timeout terminated instance_id=%s interrupt_id=%s",
            instance_id,
            cp.interrupt_id,
        )
