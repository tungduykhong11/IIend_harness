"""Agent lifecycle state machine.

Defines the states every agent instance moves through during its lifetime
and the rules governing valid transitions between them.
"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class AgentState(StrEnum):
    """States in the agent lifecycle."""

    INIT = "init"  # agent created, not yet running
    RUNNING = "running"  # processing a task or answering a question
    INTERRUPT = "interrupt"  # paused, waiting for human
    COMPLETE = "complete"  # task / session finished
    ERROR = "error"  # crashed or timed out
    DEAD = "dead"  # terminal — agent has been killed or naturally ended


class AgentType(StrEnum):
    """Well-known agent roles in the harness."""

    ORCHESTRATOR = "orchestrator"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    RESPONDER = "responder"


# ---------------------------------------------------------------------------
# State transition table (immutable)
# ---------------------------------------------------------------------------

_ALLOWED_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.INIT: frozenset({AgentState.RUNNING}),
    AgentState.RUNNING: frozenset(
        {AgentState.INTERRUPT, AgentState.COMPLETE, AgentState.ERROR, AgentState.DEAD}
    ),
    AgentState.INTERRUPT: frozenset({AgentState.RUNNING, AgentState.ERROR, AgentState.DEAD}),
    AgentState.COMPLETE: frozenset({AgentState.DEAD}),
    AgentState.ERROR: frozenset({AgentState.DEAD}),
    AgentState.DEAD: frozenset(),  # terminal — no way out
}

_TERMINAL_STATES: frozenset[AgentState] = frozenset({AgentState.DEAD})
_ALIVE_STATES: frozenset[AgentState] = frozenset(
    {AgentState.INIT, AgentState.RUNNING, AgentState.INTERRUPT}
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def validate_transition(current: AgentState, target: AgentState) -> bool:
    """Return True if *current → target* is a legal transition, False otherwise."""
    return target in _ALLOWED_TRANSITIONS.get(current, frozenset())


def transition(current: AgentState, target: AgentState) -> AgentState:
    """Perform a state transition, returning the new state.

    Raises ValueError if the transition is not allowed.
    """
    if not validate_transition(current, target):
        raise ValueError(f"Cannot transition from {current.value!r} to {target.value!r}")
    return target


def is_terminal(state: AgentState) -> bool:
    """Return True if *state* is a terminal state (DEAD)."""
    return state in _TERMINAL_STATES


def is_alive(state: AgentState) -> bool:
    """Return True if the agent is still active (INIT, RUNNING, or INTERRUPT)."""
    return state in _ALIVE_STATES


# ---------------------------------------------------------------------------
# Agent instance tracking
# ---------------------------------------------------------------------------


class AgentInstance(BaseModel):
    """Lightweight tracking record for one agent instance."""

    instance_id: str
    agent_type: AgentType
    state: AgentState = AgentState.INIT
    spawned_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stopped_at: datetime | None = None
