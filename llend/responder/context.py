"""Session context models consumed by the Responder agent.

Every ``respond.query`` carries a ``SessionContext`` so the Responder knows
what happened before this question and can give informed answers.

Spec references
===============
- **§5.1** → ``SessionContext`` — the full context passed per query
- **§5.2** → ``ConversationTurn`` — a single Q&A pair in the session
- **§5.3** → ``TaskResultSummary`` — condensed task output for Responder awareness
- **§5.4** → ``SessionContext.trim_history()`` — keep conversation bounded (50 turns)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from llend.runtime.message import TaskStatus

if TYPE_CHECKING:
    from llend.responder.memory import UserProfile


class ConversationTurn(BaseModel):
    """A single turn in the conversation between user and Responder.  §5.2."""

    role: Literal["user", "responder"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TaskResultSummary(BaseModel):
    """Condensed summary of a completed task, for Responder context.  §5.3.

    The Orchestrator produces one of these after every task completes.
    It is a 1–3 sentence summary with key metrics — enough for the
    Responder to reference without reading the full task output.
    """

    task_id: UUID  # §5.3 — UUID identifier for the task
    skill_name: str
    status: TaskStatus
    summary: str = ""  # 1–3 sentence summary of findings  §5.3
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: list[str] = Field(default_factory=list)


class SessionContext(BaseModel):
    """The full session context passed to the Responder with each query.  §5.1.

    Accumulates over the session lifetime: conversation history, completed
    task results, the active task (if any), and the user's persistent profile.
    """

    session_goal: str
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    task_results: list[TaskResultSummary] = Field(default_factory=list)
    active_task: TaskResultSummary | None = None  # §5.1 — running task summary, or None
    user_profile: UserProfile | None = None  # §5.1, §9.1 — loaded from ~/.llend/

    # --- limits ---

    _max_history_turns: int = 50

    def trim_history(self, max_turns: int | None = None) -> None:
        """Drop the oldest conversation turns when over the limit.  §5.4."""
        limit = max_turns if max_turns is not None else self._max_history_turns
        if len(self.conversation_history) > limit:
            self.conversation_history = self.conversation_history[-limit:]

    def add_turn(self, role: Literal["user", "responder"], content: str) -> None:
        """Append a conversation turn and trim if needed.  §5.2."""
        self.conversation_history.append(
            ConversationTurn(role=role, content=content)
        )
        self.trim_history()
