"""Checkpoint model for interrupt / human-in-the-loop.

When an agent raises an interrupt the runtime freezes its state into a
``Checkpoint``.  With LangGraph the persistence is handled by the
LangGraph checkpointer (``SqliteSaver``, ``InMemorySaver``, etc.) —
the ``Checkpoint`` model carries interrupt-specific metadata through
the graph state.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Checkpoint model
# ---------------------------------------------------------------------------


class Checkpoint(BaseModel):
    """A frozen snapshot of an agent blocked on a human decision."""

    interrupt_id: UUID
    session_id: UUID
    agent_instance: str
    agent_type: str  # "executor" | "reviewer" | "responder"
    agent_state: str = "INTERRUPT"  # always INTERRUPT when checkpointed

    reply_chain: list[UUID] = Field(default_factory=list)
    task_context: dict[str, Any] = Field(default_factory=dict)

    interrupt_message: str
    interrupt_options: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ttl_seconds: int = 86400  # 24 h default, configurable per interrupt

    human_response: str | None = None
    resolved_at: datetime | None = None

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def is_expired(self) -> bool:
        """True when the TTL has elapsed since *created_at*."""
        elapsed = (datetime.now(UTC) - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds

    @property
    def is_resolved(self) -> bool:
        """True after the human has responded."""
        return self.human_response is not None

    @property
    def age_seconds(self) -> float:
        """Seconds since this checkpoint was created."""
        return (datetime.now(UTC) - self.created_at).total_seconds()
