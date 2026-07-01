"""Agent lifecycle state machine.  Spec 001 §3.3.

Defines the six states every agent instance moves through during its lifetime
and the rules governing valid transitions between them.

Spec reference
==============
- **§3.3** — states: INIT → RUNNING → (INTERRUPT)* → COMPLETE / ERROR → DEAD
  Transition table, predicates, and the ``AgentInstance`` tracking record are
  all defined here.
"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# States  —  Spec 001 §3.3 table
# ---------------------------------------------------------------------------


class AgentState(StrEnum):
    """Six states in the agent lifecycle.  Spec 001 §3.3.

    +------------+-----------------------------------------------+
    | State      | Meaning                                       |
    +============+===============================================+
    | INIT       | Agent created, not yet running                |
    | RUNNING    | Processing a task or answering a question     |
    | INTERRUPT  | Paused, waiting for human (§3.4)              |
    | COMPLETE   | Task / session finished                       |
    | ERROR      | Crashed or timed out                          |
    | DEAD       | Terminal — agent killed or naturally ended    |
    +------------+-----------------------------------------------+
    """

    INIT = "init"
    RUNNING = "running"
    INTERRUPT = "interrupt"
    COMPLETE = "complete"
    ERROR = "error"
    DEAD = "dead"


class AgentType(StrEnum):
    """Well-known agent roles in the harness topology.

    Spec 001 §1 (Agent Topology table):
    - ORCHESTRATOR: "sếp" — receives requests, plans, dispatches, synthesizes
    - EXECUTOR: "làm" — constructive, completes one task, stateless
    - REVIEWER: "kiểm" — adversarial, verifies Executor output, stateless
    - RESPONDER: conversational Q&A — Spec 003
    """

    ORCHESTRATOR = "orchestrator"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    RESPONDER = "responder"


# ---------------------------------------------------------------------------
# State transition table  —  Spec 001 §3.3 (immutable)
# ---------------------------------------------------------------------------

_ALLOWED_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    # §3.3 table row "INIT"
    AgentState.INIT: frozenset({AgentState.RUNNING}),
    # §3.3 table row "RUNNING" — can go to INTERRUPT, COMPLETE, ERROR, or DEAD
    AgentState.RUNNING: frozenset(
        {AgentState.INTERRUPT, AgentState.COMPLETE, AgentState.ERROR, AgentState.DEAD}
    ),
    # §3.3 table row "INTERRUPT" — resume → RUNNING, timeout → ERROR, kill → DEAD
    AgentState.INTERRUPT: frozenset({AgentState.RUNNING, AgentState.ERROR, AgentState.DEAD}),
    # §3.3 table row "COMPLETE"
    AgentState.COMPLETE: frozenset({AgentState.DEAD}),
    # §3.3 table row "ERROR"
    AgentState.ERROR: frozenset({AgentState.DEAD}),
    # §3.3 table row "DEAD" — terminal, no way out
    AgentState.DEAD: frozenset(),
}

_TERMINAL_STATES: frozenset[AgentState] = frozenset({AgentState.DEAD})
_ALIVE_STATES: frozenset[AgentState] = frozenset(
    {AgentState.INIT, AgentState.RUNNING, AgentState.INTERRUPT}
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def validate_transition(current: AgentState, target: AgentState) -> bool:
    """Return True if *current → target* is a legal transition (§3.3 table)."""
    return target in _ALLOWED_TRANSITIONS.get(current, frozenset())


def transition(current: AgentState, target: AgentState) -> AgentState:
    """Perform a state transition (§3.3), returning the new state.

    Raises ValueError if the transition is not in the allowed table.
    """
    if not validate_transition(current, target):
        raise ValueError(f"Cannot transition from {current.value!r} to {target.value!r}")
    return target


def is_terminal(state: AgentState) -> bool:
    """Return True if *state* is a terminal state — DEAD (§3.3)."""
    return state in _TERMINAL_STATES


def is_alive(state: AgentState) -> bool:
    """Return True if the agent is still active — INIT, RUNNING, or INTERRUPT (§3.3)."""
    return state in _ALIVE_STATES


# ---------------------------------------------------------------------------
# Agent instance tracking  —  Spec 001 §3.3
# ---------------------------------------------------------------------------


class AgentInstance(BaseModel):
    """Lightweight tracking record for one agent instance.  Spec 001 §3.3.

    Used by runtimes to track spawn time and lifecycle state.
    """

    instance_id: str
    agent_type: AgentType
    state: AgentState = AgentState.INIT
    spawned_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stopped_at: datetime | None = None
