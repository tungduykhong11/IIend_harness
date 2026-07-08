"""LangGraphRuntime — AgentRuntime implementation backed by LangGraph.  Spec 001 §3.1.

Uses LangGraph for:
- State management (``StateGraph`` with typed state, §3.1)
- Checkpoint persistence (``InMemorySaver`` — §3.4; production: ``SqliteSaver``)
- Human-in-the-loop (LangGraph ``interrupt()`` — §3.4)
- Graph execution (``ainvoke`` / ``astream`` — §3.5)

Keeps the same ``AgentRuntime`` ABC contract (§3.2) so all upstream code
(skills, orchestrator, tool bridge) works without changes.

This is the **v0 primary backend** (§3.1): LangGraph's state graph is a
natural fit for the agent lifecycle (§3.3) and built-in ``interrupt()``
maps directly to our HITL requirement (§3.4).

Spec reference
==============
- **§3.1** — v0 primary backend (LangGraph state graph + checkpointing)
- **§3.2** — implements ``AgentRuntime`` (spawn, send, interrupt, kill, shutdown)
- **§3.3** — agent lifecycle via ``lifecycle.transition()`` + LangGraph graph per agent
- **§3.4** — checkpoint persistence (disk), notification, TTL enforcement
- **§3.5** — concurrency via LangGraph threads + asyncio tasks
- **§2.3** — message expiry in ``_handle_expired_message``
- **§2.4** — routing rule in ``_resolve_recipient`` (Orchestrator is the hub)
"""

from __future__ import annotations

import asyncio
import logging
import operator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from llend.runtime.base import AgentRuntime, MessageHandler
from llend.runtime.checkpoint import (
    Checkpoint,
    InterruptTimeoutError,
)
from llend.runtime.lifecycle import AgentState, AgentType, transition
from llend.runtime.message import AgentErrorCode, Message, MsgType
from llend.runtime.notifications import (
    ConsoleNotificationChannel,
    NotificationChannel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph state schema  —  AgentGraphState (§3.1)
# ---------------------------------------------------------------------------


class AgentGraphState(TypedDict):
    """Shared state for a single agent's LangGraph thread.  §3.1.

    - ``messages``: accumulated via LangGraph's ``operator.add`` reducer
    - ``context``: agent metadata (instance_id, agent_type, task context)
    - ``pending_interrupt``: when set, ``_interrupt_check_node`` triggers ``interrupt()`` (§3.4)
    - ``interrupt_decision``: human's choice, set after resume (§3.4 step 5)
    """

    messages: Annotated[list[Message], operator.add]
    context: dict
    pending_interrupt: dict | None   # {prompt, options} — triggers LangGraph interrupt()
    interrupt_decision: str | None   # human's choice after resume


# ---------------------------------------------------------------------------
# Internal handle  —  §3.3 per-agent bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _AgentHandle:
    """Runtime-internal bookkeeping for one agent instance.  §3.3.

    Like ``AsyncioRuntime._AgentHandle`` but uses LangGraph's compiled
    graph + config instead of an ``asyncio.Queue`` inbox.
    """

    instance_id: str
    agent_type: str
    state: AgentState = AgentState.INIT
    context: dict = field(default_factory=dict)
    graph: CompiledStateGraph | None = None      # §3.1: per-agent StateGraph
    config: dict | None = None                   # LangGraph config with thread_id
    run_task: asyncio.Task[None] | None = None   # §3.5: graph execution task
    interrupt_future: asyncio.Future[str] | None = None  # §3.4 step 4
    spawned_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Message handler — registered by agent via register_handler() (Spec 003 §5.1)
    handler: MessageHandler | None = None


# Avoid circular import — CompiledStateGraph is the return type of .compile()
from langgraph.graph.state import CompiledStateGraph  # noqa: E402

# ---------------------------------------------------------------------------
# Graph node functions  —  §3.1 (module-level so they can be pickled)
# ---------------------------------------------------------------------------


def _interrupt_check_node(state: AgentGraphState) -> dict:
    """Check for a pending interrupt request and pause if one exists.  §3.4.

    LangGraph's ``interrupt()`` pauses graph execution until a
    ``Command(resume=…)`` arrives.  This is the core HITL primitive (§3.4).
    """
    pending = state.get("pending_interrupt")
    if pending is None:
        return {}
    # §3.4: LangGraph interrupt() — pauses, waits for Command(resume=...)
    decision = interrupt(pending)
    return {"interrupt_decision": decision, "pending_interrupt": None}


def _agent_process_node(state: AgentGraphState) -> dict:
    """Main agent logic — process the most recent message.  §2.2 dispatch.

    **v0 placeholder**: echoes back any message so the message-routing
    contract can be tested end-to-end.  Real agent logic will be injected
    by the Orchestrator when it dispatches a task (§2.2 ``task.dispatch``).
    """
    messages = state.get("messages", [])
    context = state.get("context", {})

    if not messages:
        return {}

    last_msg = messages[-1]
    response = Message(
        session_id=last_msg.session_id,
        sender=context.get("agent_type", "executor"),
        sender_instance=context.get("instance_id", "unknown"),
        recipient=AgentType.ORCHESTRATOR.value,
        msg_type=MsgType.TASK_RESULT,                # §2.2: Executor → Orch
        payload={"echo": last_msg.payload, "status": "processed"},
        parent_id=last_msg.id,                       # §2.5: reply chain
    )
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Agent graph builder  —  §3.1 StateGraph
# ---------------------------------------------------------------------------


def _build_agent_graph(checkpointer: InMemorySaver) -> CompiledStateGraph:
    """Build and compile the per-agent StateGraph.  §3.1.

    Graph topology:
    START → interrupt_check (§3.4) → process (§2.2) → END
    """
    builder = StateGraph(AgentGraphState)
    builder.add_node("interrupt_check", _interrupt_check_node)  # §3.4
    builder.add_node("process", _agent_process_node)            # §2.2
    builder.add_edge(START, "interrupt_check")
    builder.add_edge("interrupt_check", "process")
    builder.add_edge("process", END)
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# LangGraphRuntime  —  Spec 001 §3.1 (v0 primary backend)
# ---------------------------------------------------------------------------


class LangGraphRuntime(AgentRuntime):
    """v0 LangGraph-based agent execution backend.  §3.1.

    Parameters
    ----------
    checkpoint_ttl:
        Default TTL (seconds) for interrupt checkpoints.  §3.4 — default 86 400 (24 h).
    notification_channel:
        Where to send human notifications on interrupt.  §3.4 step 3.
    data_dir:
        Base directory for checkpoint persistence.  §3.4 — ``~/.llend``.
    ttl_check_interval:
        Seconds between TTL expiry scans.  §3.4 ¶2 — default 10 s.
    """

    def __init__(
        self,
        checkpoint_ttl: int = 86400,               # §3.4: 24h default
        notification_channel: NotificationChannel | None = None,  # §3.4 step 3
        data_dir: Path | None = None,              # §3.4: ~/.llend
        ttl_check_interval: float = 10.0,          # §3.4 ¶2: TTL scan
    ) -> None:
        self._agents: dict[str, _AgentHandle] = {}
        self._orchestrator_id: str | None = None   # §2.4: Orchestrator is the hub
        self._session_id: UUID | None = None       # §2.1: session_id
        self._checkpointer = InMemorySaver()       # §3.4: LangGraph checkpointer
        self._checkpoint_ttl = checkpoint_ttl
        self._notification_channel = notification_channel or ConsoleNotificationChannel()
        self._data_dir = Path(data_dir) if data_dir else Path.home() / ".llend"
        self._ttl_check_interval = ttl_check_interval
        self._closed = False

        # Active interrupt checkpoints keyed by instance_id (§3.4)
        self._checkpoints: dict[str, Checkpoint] = {}

        # Background TTL monitor (§3.4 ¶2)
        self._ttl_monitor_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public API — AgentRuntime contract  §3.2
    # ------------------------------------------------------------------

    async def spawn(self, agent_type: str, context: dict) -> str:
        """Create a new agent backed by a LangGraph thread.  §3.2, §3.1.

        §3.3: INIT → RUNNING.
        Each agent gets its own compiled ``StateGraph`` with an isolated
        LangGraph thread (thread_id = instance_id).
        """
        self._guard_closed()

        instance_id = f"{agent_type}-{uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": instance_id}}

        # §3.1: Build a fresh compiled graph for this agent
        graph = _build_agent_graph(self._checkpointer)

        handle = _AgentHandle(
            instance_id=instance_id,
            agent_type=agent_type,
            context=context,
            graph=graph,
            config=config,
        )

        if agent_type == AgentType.ORCHESTRATOR.value:
            self._orchestrator_id = instance_id
            self._session_id = UUID(hex=uuid4().hex)

        # §3.3: INIT → RUNNING
        handle.state = transition(AgentState.INIT, AgentState.RUNNING)
        self._agents[instance_id] = handle

        # Seed the agent's state with initial context
        seed_state: AgentGraphState = {
            "messages": [],
            "context": {**context, "instance_id": instance_id, "agent_type": agent_type},
            "pending_interrupt": None,
            "interrupt_decision": None,
        }
        await graph.ainvoke(seed_state, config)

        # §3.4: Ensure TTL monitor is running
        self._ensure_ttl_monitor()

        logger.info("spawned agent instance_id=%s type=%s", instance_id, agent_type)
        return instance_id

    async def send(self, message: Message) -> None:
        """Route *message* by appending it to the recipient's LangGraph state.  §3.2, §2.4.

        §2.3: Expired messages are dropped — sender receives ``agent.error(TIMEOUT)``.
        §2.4: Routing follows Orchestrator-is-hub rule.
        The message is pushed into the agent's LangGraph state and the graph
        is re-invoked to process it (§3.1).
        """
        self._guard_closed()

        # §2.3: Drop expired messages
        if message.is_expired:
            await self._handle_expired_message(message)
            return

        # §2.4: Resolve recipient
        instance_id = self._resolve_recipient(message)
        if instance_id is None:
            logger.warning(
                "message dropped — no route for recipient=%s instance=%s",
                message.recipient,
                message.recipient_instance,
            )
            return

        handle = self._get_handle(instance_id)
        # Push the message into the agent's state and re-invoke the graph (§3.1)
        update = {"messages": [message]}
        await handle.graph.aupdate_state(handle.config, update)
        # Kick the agent to process the new message (§3.5: async graph execution)
        handle.run_task = asyncio.create_task(
            self._run_agent(handle), name=f"run-{instance_id}"
        )

        # Spec 003 §5.1: Fire registered handler (fire-and-forget)
        if handle.handler is not None:
            asyncio.create_task(
                handle.handler(message),
                name=f"handler-{instance_id}",
            )

    async def interrupt(self, instance_id: str, prompt: str, options: list[str]) -> str:
        """Pause *instance_id*, ask human, return chosen option.  §3.2, §3.4.

        Follows the 5-step interrupt flow from §3.4 ¶1:
        1. Freeze agent state → ``Checkpoint``
        2. Save checkpoint to disk
        3. Notify human channel
        4. Block via LangGraph ``interrupt()`` + ``asyncio.Future`` bridge
        5. On response: update checkpoint, inject ``interrupt.response``, resume

        Unlike ``AsyncioRuntime``, the blocking uses LangGraph's native
        ``interrupt()`` inside ``_interrupt_check_node``, bridged to an
        ``asyncio.Future`` so the event loop doesn't deadlock.

        Raises ``InterruptTimeoutError`` if TTL expires (§3.4 ¶2).
        """
        self._guard_closed()
        handle = self._get_handle(instance_id)

        interrupt_payload = {"prompt": prompt, "options": options}

        # §3.4 step 1-2: Create and persist checkpoint
        cp = Checkpoint(
            interrupt_id=uuid4(),
            session_id=self._session_id or uuid4(),
            agent_instance=instance_id,
            agent_type=handle.agent_type,
            task_context=handle.context,
            interrupt_message=prompt,
            interrupt_options=options,
            ttl_seconds=self._checkpoint_ttl,
        )
        cp.save(self._data_dir)
        self._checkpoints[instance_id] = cp

        # §3.4 step 3: Notify human
        await self._notification_channel.notify_interrupt(cp)

        # §3.4 step 4: Create future + launch LangGraph graph (will hit interrupt())
        handle.interrupt_future = asyncio.get_event_loop().create_future()

        async def _invoke_with_interrupt() -> None:
            # Push the interrupt payload as an update and run the graph.
            # The _interrupt_check_node will see pending_interrupt and call
            # LangGraph's interrupt() which pauses execution (§3.4).
            try:
                await handle.graph.ainvoke(
                    Command(update={"pending_interrupt": interrupt_payload}),
                    handle.config,
                )
            except Exception as exc:
                if not handle.interrupt_future.done():
                    handle.interrupt_future.set_exception(exc)

        handle.run_task = asyncio.create_task(_invoke_with_interrupt())
        handle.state = transition(handle.state, AgentState.INTERRUPT)  # §3.3

        logger.info(
            "interrupt raised instance_id=%s interrupt_id=%s prompt=%r",
            instance_id,
            cp.interrupt_id,
            prompt[:120],
        )

        # Block until human responds (or TTL expires via §3.4 ¶2)
        try:
            decision = await handle.interrupt_future
        except InterruptTimeoutError:
            raise
        finally:
            handle.interrupt_future = None

        # §3.4 step 5: Update checkpoint on disk, resume agent
        updated = self._checkpoints.pop(instance_id, cp)
        if not updated.is_resolved:
            updated.human_response = decision
            updated.resolved_at = datetime.now(UTC)
        updated.save(self._data_dir)

        handle.state = transition(handle.state, AgentState.RUNNING)  # §3.3: resume
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
        """Terminate *instance_id*.  Idempotent.  §3.2.

        §3.3: any state → DEAD.
        Cancels LangGraph run task and cleans up checkpoint file (§3.4).
        """
        handle = self._agents.get(instance_id)
        if handle is None or handle.state == AgentState.DEAD:
            return

        if handle.interrupt_future is not None and not handle.interrupt_future.done():
            handle.interrupt_future.cancel("agent killed")

        # Clean up checkpoint file (§3.4)
        cp = self._checkpoints.pop(instance_id, None)
        if cp is not None:
            try:
                cp.delete(self._data_dir)
            except OSError:
                logger.debug("failed to delete checkpoint file for %s", instance_id)

        if handle.run_task is not None and not handle.run_task.done():
            handle.run_task.cancel()
            try:
                await handle.run_task
            except asyncio.CancelledError:
                pass

        handle.state = AgentState.DEAD              # §3.3: → DEAD
        logger.info("killed agent instance_id=%s", instance_id)

    async def shutdown(self) -> None:
        """Kill all agents and mark runtime closed.

        Stops the TTL monitor first (§3.4), then kills all agents (§3.2).
        """
        # Stop TTL monitor (§3.4 ¶2)
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
    # v0-specific helpers (not on the ABC)
    # ------------------------------------------------------------------

    async def resolve_interrupt(
        self, instance_id: str, decision: str, note: str | None = None
    ) -> None:
        """Resolve a pending interrupt on *instance_id*.  §3.4 step 5.

        Sends ``Command(resume=decision)`` to LangGraph which unblocks
        the ``interrupt()`` call inside ``_interrupt_check_node`` (§3.4),
        then sets the ``asyncio.Future`` result to unblock ``interrupt()``.
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

        # Resume the LangGraph execution — unblocks interrupt() in
        # _interrupt_check_node (§3.4)
        await handle.graph.ainvoke(Command(resume=decision), handle.config)

        handle.interrupt_future.set_result(decision)

    @property
    def agent_count(self) -> int:
        """Return the number of tracked agent instances."""
        return len(self._agents)

    @property
    def is_closed(self) -> bool:
        """Return True after ``shutdown()``."""
        return self._closed

    @property
    def session_id(self) -> UUID | None:
        """Return the current session id (§2.1)."""
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
        if self._closed:
            raise RuntimeError("LangGraphRuntime is closed — cannot accept new work")

    def _get_handle(self, instance_id: str) -> _AgentHandle:
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
        3. First agent of matching type
        """
        if message.recipient_instance is not None:
            return (
                message.recipient_instance if message.recipient_instance in self._agents else None
            )
        if message.recipient == AgentType.ORCHESTRATOR.value:
            return self._orchestrator_id
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

        Pushes directly to the sender's LangGraph state instead of calling
        ``send()`` to avoid re-entrant expiry checks.
        """
        error_msg = Message(
            session_id=message.session_id,
            sender="runtime",
            sender_instance="runtime",
            recipient=message.sender,
            recipient_instance=message.sender_instance,
            msg_type=MsgType.AGENT_ERROR,
            payload={
                "error_code": AgentErrorCode.TIMEOUT.value,    # §2.2.1
                "detail": f"Message {message.id} expired unread",
                "recoverable": True,                           # §2.3 step 3
            },
            parent_id=message.id,
        )
        # Push directly to the sender's state — avoid recursive send()
        sender_id = self._resolve_recipient(error_msg)
        if sender_id is not None:
            handle = self._get_handle(sender_id)
            await handle.graph.aupdate_state(handle.config, {"messages": [error_msg]})
        logger.info("message %s expired — sending error back to %s", message.id, message.sender)

    # ------------------------------------------------------------------
    # Graph execution  —  §3.1, §3.5
    # ------------------------------------------------------------------

    async def _run_agent(self, handle: _AgentHandle) -> None:
        """Invoke the agent's LangGraph graph asynchronously.  §3.1, §3.5.

        The graph runs until it completes, hits an ``interrupt()`` call
        (§3.4), or is cancelled.  After completion the agent transitions
        to COMPLETE (§3.3).
        """
        try:
            result = await handle.graph.ainvoke(None, handle.config)
            # §3.4: LangGraph surfaces __interrupt__ in the result when paused
            if "__interrupt__" in (result or {}):
                # Graph is paused — the interrupt_future is already
                # being awaited by the interrupt() method.
                return
            # §3.3: graceful completion
            handle.state = transition(handle.state, AgentState.COMPLETE)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("agent %s crashed", handle.instance_id)
            # §3.3: unhandled exception → ERROR
            handle.state = transition(handle.state, AgentState.ERROR)

    # ------------------------------------------------------------------
    # TTL enforcement  —  Spec 001 §3.4 ¶2
    # ------------------------------------------------------------------

    def _ensure_ttl_monitor(self) -> None:
        """Start the TTL monitor background task if not already running.  §3.4 ¶2."""
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
        """Check all active checkpoints (§3.4) and terminate expired ones."""
        for instance_id, cp in list(self._checkpoints.items()):
            if cp.is_expired and not cp.is_resolved:
                await self._terminate_expired_interrupt(instance_id)

    async def _terminate_expired_interrupt(self, instance_id: str) -> None:
        """Force-terminate an interrupt whose TTL has expired.  §3.4 ¶2.

        Steps:
        1. Mark checkpoint resolved with ``__timeout__`` and save to disk
        2. Notify human channel
        3. Transition agent INTERRUPT → ERROR (§3.3)
        4. Cancel LangGraph run task + raise ``InterruptTimeoutError`` via future
        5. Send ``agent.error(INTERRUPT_TIMEOUT)`` to Orchestrator (§2.2)
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

            # 4. Cancel LangGraph run task (graph paused on interrupt())
            if handle.run_task is not None and not handle.run_task.done():
                handle.run_task.cancel()
                try:
                    await handle.run_task
                except asyncio.CancelledError:
                    pass

            # 4 (cont). Raise InterruptTimeoutError via future (§3.4 ¶2)
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
                        "recoverable": False,
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
