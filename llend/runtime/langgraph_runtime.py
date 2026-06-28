"""LangGraphRuntime — AgentRuntime implementation backed by LangGraph.

Uses LangGraph for:
- State management (StateGraph with typed state)
- Checkpoint persistence (SqliteSaver / InMemorySaver)
- Human-in-the-loop (langgraph interrupt())
- Graph execution (ainvoke / astream)

Keeps the same ``AgentRuntime`` ABC contract so all upstream code
(skills, orchestrator, tool bridge) works without changes.
"""

from __future__ import annotations

import asyncio
import logging
import operator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from llend.runtime.base import AgentRuntime
from llend.runtime.lifecycle import AgentState, AgentType, transition
from llend.runtime.message import AgentErrorCode, Message, MsgType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph state schema
# ---------------------------------------------------------------------------


class AgentGraphState(TypedDict):
    """Shared state for a single agent's LangGraph thread."""

    messages: Annotated[list[Message], operator.add]
    context: dict
    pending_interrupt: dict | None  # {prompt, options} — set to trigger interrupt()
    interrupt_decision: str | None  # human's choice, set after resume


# ---------------------------------------------------------------------------
# Internal handle (similar to AsyncioRuntime's _AgentHandle)
# ---------------------------------------------------------------------------


@dataclass
class _AgentHandle:
    """Runtime-internal bookkeeping for one agent instance."""

    instance_id: str
    agent_type: str
    state: AgentState = AgentState.INIT
    context: dict = field(default_factory=dict)
    graph: CompiledStateGraph | None = None
    config: dict | None = None  # LangGraph config with thread_id
    run_task: asyncio.Task[None] | None = None
    interrupt_future: asyncio.Future[str] | None = None
    spawned_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# Avoid circular import — CompiledStateGraph is the return type of .compile()
from langgraph.graph.state import CompiledStateGraph  # noqa: E402

# ---------------------------------------------------------------------------
# Graph node functions (module-level so they can be pickled if needed)
# ---------------------------------------------------------------------------


def _interrupt_check_node(state: AgentGraphState) -> dict:
    """Check for a pending interrupt request and pause if one exists.

    Called at the top of every agent invocation.  If ``pending_interrupt``
    is set the node calls LangGraph's ``interrupt()`` which pauses
    execution until a ``Command(resume=…)`` arrives.
    """
    pending = state.get("pending_interrupt")
    if pending is None:
        return {}
    decision = interrupt(pending)
    return {"interrupt_decision": decision, "pending_interrupt": None}


def _agent_process_node(state: AgentGraphState) -> dict:
    """Main agent logic — process the most recent message.

    In v0 this is a placeholder.  The real agent logic will be injected
    by the Orchestrator when it dispatches a task.  For now the node
    simply echoes back any message it receives so the message-routing
    contract can be tested end-to-end.
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
        msg_type=MsgType.TASK_RESULT,
        payload={"echo": last_msg.payload, "status": "processed"},
        parent_id=last_msg.id,
    )
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Agent graph builder
# ---------------------------------------------------------------------------


def _build_agent_graph(checkpointer: InMemorySaver) -> CompiledStateGraph:
    """Build and compile the per-agent StateGraph."""
    builder = StateGraph(AgentGraphState)
    builder.add_node("interrupt_check", _interrupt_check_node)
    builder.add_node("process", _agent_process_node)
    builder.add_edge(START, "interrupt_check")
    builder.add_edge("interrupt_check", "process")
    builder.add_edge("process", END)
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# LangGraphRuntime
# ---------------------------------------------------------------------------


class LangGraphRuntime(AgentRuntime):
    """v0 LangGraph-based agent execution backend.

    Parameters
    ----------
    checkpoint_ttl:
        Default TTL (seconds) for interrupt checkpoints.  Default 86 400 (24 h).
    """

    def __init__(self, checkpoint_ttl: int = 86400) -> None:
        self._agents: dict[str, _AgentHandle] = {}
        self._orchestrator_id: str | None = None
        self._session_id: UUID | None = None
        self._checkpointer = InMemorySaver()
        self._checkpoint_ttl = checkpoint_ttl
        self._closed = False

    # ------------------------------------------------------------------
    # Public API (AgentRuntime contract)
    # ------------------------------------------------------------------

    async def spawn(self, agent_type: str, context: dict) -> str:
        """Create a new agent backed by a LangGraph thread."""
        self._guard_closed()

        instance_id = f"{agent_type}-{uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": instance_id}}

        # Build a fresh compiled graph for this agent
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

        handle.state = transition(AgentState.INIT, AgentState.RUNNING)
        self._agents[instance_id] = handle

        # Seed the agent's state
        seed_state: AgentGraphState = {
            "messages": [],
            "context": {**context, "instance_id": instance_id, "agent_type": agent_type},
            "pending_interrupt": None,
            "interrupt_decision": None,
        }
        await graph.ainvoke(seed_state, config)

        logger.info("spawned agent instance_id=%s type=%s", instance_id, agent_type)
        return instance_id

    async def send(self, message: Message) -> None:
        """Route *message* by appending it to the recipient's LangGraph state."""
        self._guard_closed()

        if message.is_expired:
            await self._handle_expired_message(message)
            return

        instance_id = self._resolve_recipient(message)
        if instance_id is None:
            logger.warning(
                "message dropped — no route for recipient=%s instance=%s",
                message.recipient,
                message.recipient_instance,
            )
            return

        handle = self._get_handle(instance_id)
        # Push the message into the agent's state and re-invoke the graph
        update = {"messages": [message]}
        await handle.graph.aupdate_state(handle.config, update)
        # Kick the agent to process the new message
        handle.run_task = asyncio.create_task(self._run_agent(handle), name=f"run-{instance_id}")

    async def interrupt(self, instance_id: str, prompt: str, options: list[str]) -> str:
        """Pause *instance_id*, ask human, return chosen option.

        Sets ``pending_interrupt`` in the agent's state and invokes the
        graph with a ``Command``.  The graph's ``_interrupt_check_node``
        hits LangGraph ``interrupt()`` which blocks the ``ainvoke`` until
        a matching ``Command(resume=…)`` arrives from ``resolve_interrupt()``.

        Because LangGraph's ``interrupt()`` blocks the event loop, we run
        the graph invocation in a background task and bridge it to the
        caller via an ``asyncio.Future``.
        """
        self._guard_closed()
        handle = self._get_handle(instance_id)

        interrupt_payload = {"prompt": prompt, "options": options}

        # Create a future that resolve_interrupt() will set
        handle.interrupt_future = asyncio.get_event_loop().create_future()

        async def _invoke_with_interrupt() -> None:
            # Push the interrupt payload as an update and run the graph.
            # The _interrupt_check_node will see pending_interrupt and call
            # langgraph's interrupt(), which pauses execution.
            try:
                await handle.graph.ainvoke(
                    Command(update={"pending_interrupt": interrupt_payload}),
                    handle.config,
                )
            except Exception as exc:
                if not handle.interrupt_future.done():
                    handle.interrupt_future.set_exception(exc)

        handle.run_task = asyncio.create_task(_invoke_with_interrupt())
        handle.state = transition(handle.state, AgentState.INTERRUPT)

        logger.info("interrupt raised instance_id=%s prompt=%r", instance_id, prompt[:120])

        try:
            decision = await handle.interrupt_future
        finally:
            handle.interrupt_future = None

        handle.state = transition(handle.state, AgentState.RUNNING)
        logger.info("interrupt resolved instance_id=%s decision=%s", instance_id, decision)
        return decision

    async def kill(self, instance_id: str) -> None:
        """Terminate *instance_id*.  Idempotent."""
        handle = self._agents.get(instance_id)
        if handle is None or handle.state == AgentState.DEAD:
            return

        if handle.interrupt_future is not None and not handle.interrupt_future.done():
            handle.interrupt_future.cancel("agent killed")

        if handle.run_task is not None and not handle.run_task.done():
            handle.run_task.cancel()
            try:
                await handle.run_task
            except asyncio.CancelledError:
                pass

        handle.state = AgentState.DEAD
        logger.info("killed agent instance_id=%s", instance_id)

    async def shutdown(self) -> None:
        """Kill all agents and mark runtime closed."""
        for instance_id in list(self._agents.keys()):
            await self.kill(instance_id)
        self._agents.clear()
        self._closed = True
        logger.info("runtime shutdown complete")

    # ------------------------------------------------------------------
    # v0-specific helpers (not on the ABC)
    # ------------------------------------------------------------------

    async def resolve_interrupt(
        self, instance_id: str, decision: str, note: str | None = None
    ) -> None:
        """Resolve a pending interrupt on *instance_id*.

        Sends ``Command(resume=decision)`` to LangGraph which unblocks
        the ``interrupt()`` call inside ``_interrupt_check_node``.
        """
        handle = self._get_handle(instance_id)

        if handle.interrupt_future is None:
            raise RuntimeError(f"Agent {instance_id} has no pending interrupt")
        if handle.interrupt_future.done():
            raise RuntimeError(f"Agent {instance_id} interrupt already resolved")

        # Resume the LangGraph execution — this unblocks the ainvoke()
        # that is waiting on interrupt() inside _interrupt_check_node
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
        """Return the current session id."""
        return self._session_id

    def get_handle_state(self, instance_id: str) -> AgentState:
        """Return the current lifecycle state of *instance_id*."""
        return self._get_handle(instance_id).state

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

    def _resolve_recipient(self, message: Message) -> str | None:
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

    async def _handle_expired_message(self, message: Message) -> None:
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
        # Push directly to the sender's state — avoid recursive send()
        sender_id = self._resolve_recipient(error_msg)
        if sender_id is not None:
            handle = self._get_handle(sender_id)
            await handle.graph.aupdate_state(handle.config, {"messages": [error_msg]})
        logger.info("message %s expired — sending error back to %s", message.id, message.sender)

    async def _run_agent(self, handle: _AgentHandle) -> None:
        """Invoke the agent's LangGraph graph asynchronously.

        The graph runs until it either completes, hits an ``interrupt()``
        call, or is cancelled.  After completion the agent transitions to
        COMPLETE (or stays RUNNING if more messages arrive).
        """
        try:
            result = await handle.graph.ainvoke(None, handle.config)
            # Check if the graph was interrupted (LangGraph surfaces
            # __interrupt__ in the result when paused)
            if "__interrupt__" in (result or {}):
                # Graph is paused — the interrupt_future is already
                # being awaited by the interrupt() method.
                return
            handle.state = transition(handle.state, AgentState.COMPLETE)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("agent %s crashed", handle.instance_id)
            handle.state = transition(handle.state, AgentState.ERROR)
