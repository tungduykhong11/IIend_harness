"""Progress reporting — human-visible status updates during task execution.

While tasks execute, the Orchestrator emits progress events that the UI
channel displays.  These are NOT formal ``msg_type`` values (§12.2) —
they are internal events forwarded to the human.

Spec references
===============
- **§12.1** → What the human sees (example output)
- **§12.2** → ``ProgressEvent`` model
- **§12.3** → In-progress status ("Đang làm gì đấy?")
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from llend.registry.pipeline import ExecutionPlan, TaskSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ProgressEvent  —  §12.2
# ---------------------------------------------------------------------------


class ProgressEvent(BaseModel):
    """A single progress update emitted by the Orchestrator.  §12.2.

    These are NOT ``MsgType`` values — they are internal events forwarded
    to the UI channel for human consumption.
    """

    level: Literal["info", "task_start", "task_complete", "warning", "error"]
    message: str
    task_id: UUID | None = None
    step: tuple[int, int] | None = None  # (current, total)


# ---------------------------------------------------------------------------
# ProgressReporter
# ---------------------------------------------------------------------------


class ProgressReporter:
    """Formats and emits progress updates during execution.  §12.1–§12.3.

    Parameters
    ----------
    on_event:
        Optional async callback called for every ``ProgressEvent``.
        If unset, events are only logged.
    """

    def __init__(
        self,
        on_event: "Callable[[ProgressEvent], Awaitable[None]] | None" = None,
    ) -> None:
        self._on_event = on_event

    async def emit(self, event: ProgressEvent) -> None:
        """Emit a progress event.  §12.2."""
        logger.info(
            "[%s]%s %s",
            event.level,
            f" [{event.step[0]}/{event.step[1]}]" if event.step else "",
            event.message,
        )
        if self._on_event is not None:
            result = self._on_event(event)
            # Support both sync callbacks (return None) and async callbacks
            if asyncio.iscoroutine(result):
                await result

    # -- Convenience emitters -----------------------------------------------

    async def plan_start(self, plan: ExecutionPlan) -> None:
        """Emit the plan overview.  §12.1."""
        skill_names = [ts.skill_name for ts in plan.skills]
        path = " → ".join(skill_names)
        await self.emit(ProgressEvent(
            level="info",
            message=f"[PLAN] {path} ({len(plan.skills)} task(s))",
        ))

    async def task_start(self, task_spec: TaskSpec, task_id: UUID, total: int) -> None:
        """Emit a task-start event.  §12.1."""
        await self.emit(ProgressEvent(
            level="task_start",
            message=f"[...] {task_spec.skill_name}...",
            task_id=task_id,
            step=(task_spec.step, total),
        ))

    async def task_complete(
        self, task_spec: TaskSpec, task_id: UUID, total: int, summary: str
    ) -> None:
        """Emit a task-complete event.  §12.1."""
        await self.emit(ProgressEvent(
            level="task_complete",
            message=f"[OK] {summary}",
            task_id=task_id,
            step=(task_spec.step, total),
        ))

    async def task_warning(self, task_spec: TaskSpec, message: str) -> None:
        """Emit a warning about a task.  §12.1."""
        await self.emit(ProgressEvent(
            level="warning",
            message=f"[WARN] [{task_spec.skill_name}] {message}",
        ))

    async def error(self, message: str) -> None:
        """Emit an error event.  §12.1."""
        await self.emit(ProgressEvent(
            level="error",
            message=f"[ERROR] {message}",
        ))

    async def session_complete(self) -> None:
        """Emit session-complete event.  §12.1."""
        await self.emit(ProgressEvent(
            level="info",
            message="[DONE] Session complete.",
        ))


# ---------------------------------------------------------------------------
# In-progress status  —  §12.3
# ---------------------------------------------------------------------------


def format_plan_progress(
    plan: ExecutionPlan,
    completed: set[str],
    active: str | None = None,
    *,
    completed_details: dict[str, str] | None = None,
    active_detail: str | None = None,
) -> str:
    """Format a human-readable progress table.  §12.3.

    Produces the "Đang làm gì đấy?" output::

        [PLAN] Progress:
          [OK] data_provider — completed (500 listings)
          [...] analyze_pricing — running (elapsed: 45s)
          [--] write_report — waiting

    Parameters
    ----------
    completed_details:
        Optional dict mapping skill_name → detail text for completed tasks
        (e.g. ``{"data_provider": "500 listings"}``).
    active_detail:
        Optional detail string for the active task
        (e.g. ``"elapsed: 45s"``).
    """
    lines: list[str] = ["[PLAN] Progress:"]
    details = completed_details or {}
    for ts in plan.skills:
        if ts.skill_name in completed:
            detail = details.get(ts.skill_name, "")
            suffix = f" — completed ({detail})" if detail else " — completed"
            lines.append(f"  [OK] {ts.skill_name}{suffix}")
        elif ts.skill_name == active:
            suffix = f" — running ({active_detail})" if active_detail else " — running"
            lines.append(f"  [...] {ts.skill_name}{suffix}")
        else:
            lines.append(f"  [--] {ts.skill_name} — waiting")
    return "\n".join(lines)
