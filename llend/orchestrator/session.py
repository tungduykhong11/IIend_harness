"""Session lifecycle — state tracking, start, complete, and artifact management.

The Orchestrator owns the session from creation to completion.  This module
defines the ``SessionState`` model (§7.3) and the ``SessionManager`` that
orchestrates the startup and shutdown sequences (§11.1–§11.4).

Spec references
===============
- **§7.3** → ``SessionState`` model (accumulated over session lifetime)
- **§11.1** → Session start — load profile, spawn Responder, ready for input
- **§11.2** → During session — classify → route loop
- **§11.3** → Session complete — cancel tasks, synthesize, save artifacts, kill Responder
- **§11.4** → Final synthesis prompt (implemented in ``summarizer.py``)
- **§13.2** → UserProfile — loaded at start, updated on complete
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from llend.registry.pipeline import ExecutionPlan
from llend.responder.context import ConversationTurn, TaskResultSummary
from llend.responder.memory import UserProfile
from llend.runtime.message import Artifact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SessionState  —  §7.3
# ---------------------------------------------------------------------------


class SessionState(BaseModel):
    """Orchestrator's in-memory session state.  §7.3.

    Accumulates over the session lifetime: completed tasks, conversation
    history, warnings, and generated artifacts.
    """

    session_id: UUID = Field(default_factory=uuid4)
    session_goal: str = ""
    plan: ExecutionPlan | None = None
    completed_tasks: list[TaskResultSummary] = Field(default_factory=list)
    active_task: TaskResultSummary | None = None
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    accumulated_warnings: list[str] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)

    # Lifecycle tracking
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    def add_task_result(self, result: TaskResultSummary) -> None:
        """Record a completed task.  §7.3."""
        self.completed_tasks.append(result)
        for path in result.artifact_paths:
            self.artifacts.append(
                Artifact(
                    name=result.skill_name,
                    path=path,
                    type=Path(path).suffix.lstrip(".") or "other",
                    description=result.summary,
                )
            )

    def add_conversation_turn(self, role: str, content: str) -> None:
        """Append a conversation turn.  §7.3."""
        self.conversation_history.append(
            ConversationTurn(role=role, content=content)  # type: ignore[arg-type]
        )

    def add_warning(self, warning: str) -> None:
        """Accumulate a warning.  §7.3."""
        self.accumulated_warnings.append(warning)

    @property
    def skipped_tasks(self) -> list[str]:
        """Return names of tasks in the plan that were NOT completed.  §11.4."""
        if self.plan is None:
            return []
        completed_names = {ts.skill_name for ts in self.completed_tasks}
        return [ts.skill_name for ts in self.plan.skills if ts.skill_name not in completed_names]

    @property
    def artifact_paths(self) -> list[str]:
        """All artifact file paths.  §11.4."""
        return [a.path for a in self.artifacts]


# ---------------------------------------------------------------------------
# SessionManager  —  §11
# ---------------------------------------------------------------------------


class SessionManager:
    """Manage the full session lifecycle.  §11.1–§11.4.

    Parameters
    ----------
    session_goal:
        The user's declared goal for this session.
    output_dir:
        Where to save artifacts (relative to cwd, or absolute).  §13.1.
    profile_path:
        Path to the user profile JSON file.  §13.2.
    """

    def __init__(
        self,
        session_goal: str = "",
        output_dir: str = "output",
        profile_path: Path | None = None,
    ) -> None:
        self.state = SessionState(session_goal=session_goal)
        self._output_dir = Path(output_dir)
        self._profile_path = profile_path or UserProfile._default_path()

        # Loaded at start  §11.1
        self._user_profile: UserProfile | None = None

    # ------------------------------------------------------------------
    # §11.1 — Start
    # ------------------------------------------------------------------

    def start(self) -> UserProfile:
        """Initialise the session.  §11.1.

        - Loads ``UserProfile`` from disk
        - Creates the output directory
        - Returns the loaded profile (so Orchestrator can pass it to Responder)

        Called once at session start.
        """
        self._user_profile = UserProfile.load(self._profile_path)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Session started: %s (goal=%r)",
            self.state.session_id,
            self.state.session_goal,
        )
        return self._user_profile

    # ------------------------------------------------------------------
    # §11.3 — Complete
    # ------------------------------------------------------------------

    def complete(
        self,
        final_synthesis: str,
        updated_profile: UserProfile | None = None,
    ) -> Path:
        """Finalise the session.  §11.3.

        - Saves the final synthesis to ``output/synthesis.md``
        - Saves artifacts list to ``output/artifacts.json``
        - Updates and persists the user profile (§13.2)
        - Marks the session as completed

        Returns the output directory path.
        """
        self.state.completed_at = datetime.now(UTC)

        # Save final synthesis
        synthesis_path = self._output_dir / "synthesis.md"
        synthesis_path.write_text(final_synthesis, encoding="utf-8")
        logger.info("Synthesis saved to %s", synthesis_path)

        # Save artifacts manifest
        artifacts_data = {
            "session_id": str(self.state.session_id),
            "session_goal": self.state.session_goal,
            "completed_at": self.state.completed_at.isoformat(),
            "artifacts": [a.model_dump() for a in self.state.artifacts],
        }
        import json
        artifacts_path = self._output_dir / "artifacts.json"
        artifacts_path.write_text(
            json.dumps(artifacts_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Artifacts manifest saved to %s", artifacts_path)

        # Update user profile  §13.2
        if updated_profile is not None:
            updated_profile.save(self._profile_path)
        elif self._user_profile is not None:
            self._user_profile.save(self._profile_path)

        logger.info(
            "Session complete: %s — %d tasks, %d artifacts",
            self.state.session_id,
            len(self.state.completed_tasks),
            len(self.state.artifacts),
        )
        return self._output_dir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_plan(self, plan: ExecutionPlan) -> None:
        """Record the execution plan for the current request.  §5.1."""
        self.state.plan = plan

    def set_active_task(self, task: TaskResultSummary | None) -> None:
        """Update the active task indicator.  §7.3."""
        self.state.active_task = task
