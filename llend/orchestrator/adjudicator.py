"""Verdict adjudication — the Orchestrator's decision after each Reviewer verdict.

After a Reviewer returns ``task.verdict``, the Orchestrator must decide:
next task, retry (with improved spec), skip, or abort the entire session.
This module encapsulates that decision logic.

Spec references
===============
- **§4.3** → Adjudication logic (flowchart: PASS / PASS_WITH_WARNINGS / FAIL)
- **§4.4** → Max retries per enforcement level table
- **§4.5** → Reviewer prompt construction (adversarial verification)
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from llend.runtime.message import ReviewIssue, TaskStatus, Verdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & models
# ---------------------------------------------------------------------------


class AdjudicationAction(StrEnum):
    """What the Orchestrator should do after adjudicating a verdict.  §4.3."""

    NEXT_TASK = "next_task"
    RETRY = "retry"
    SKIP_TASK = "skip_task"
    ABORT_SESSION = "abort_session"


class AdjudicationResult(BaseModel):
    """The Orchestrator's decision after reviewing a verdict.  §4.3."""

    action: AdjudicationAction
    reason: str = ""
    improved_task_spec: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Retry limits  —  §4.4 table
# ---------------------------------------------------------------------------

_RETRY_LIMITS: dict[str, int] = {
    "mandatory": 5,
    "strict": 3,
    "suggested": 1,
}

_ON_EXHAUSTION: dict[str, AdjudicationAction] = {
    "mandatory": AdjudicationAction.ABORT_SESSION,
    "strict": AdjudicationAction.SKIP_TASK,
    "suggested": AdjudicationAction.SKIP_TASK,
}


# ---------------------------------------------------------------------------
# Adjudication
# ---------------------------------------------------------------------------


def adjudicate(
    verdict: Verdict,
    retry_count: int,
    enforcement: str,
    *,
    reviewer_issues: list[ReviewIssue] | None = None,
    max_retries_override: dict[str, int] | None = None,
) -> AdjudicationResult:
    """Decide the next action after a Reviewer verdict.  §4.3.

    Parameters
    ----------
    verdict:
        The Reviewer's ``task.verdict`` — PASS, PASS_WITH_WARNINGS, or FAIL.
    retry_count:
        How many times this task has already been retried (0-indexed).
    enforcement:
        The skill's enforcement level: ``"mandatory"``, ``"strict"``, or ``"suggested"``.
    reviewer_issues:
        The issues the Reviewer found (used to build an improved task_spec on retry).
    max_retries_override:
        Optional per-enforcement overrides (e.g. from ``OrchestratorConfig``).

    Returns
    -------
    AdjudicationResult
        The action the Orchestrator should take next.
    """
    limits = {**_RETRY_LIMITS, **(max_retries_override or {})}
    max_retries = limits.get(enforcement, 1)

    issues = reviewer_issues or []

    if verdict == Verdict.PASS:
        return AdjudicationResult(
            action=AdjudicationAction.NEXT_TASK,
            reason="Reviewer passed — output meets spec.",
        )

    if verdict == Verdict.PASS_WITH_WARNINGS:
        warning_msgs = [f"[{i.severity}] {i.field}: {i.message}" for i in issues]
        return AdjudicationResult(
            action=AdjudicationAction.NEXT_TASK,
            reason=f"Passed with {len(warning_msgs)} warning(s).",
            warnings=warning_msgs,
        )

    # Verdict.FAIL — §4.3 adjudication logic
    if retry_count < max_retries:
        improved_spec = _build_improved_task_spec(issues) if issues else None
        return AdjudicationResult(
            action=AdjudicationAction.RETRY,
            reason=(
                f"Reviewer failed — retry {retry_count + 1}/{max_retries} "
                f"(enforcement={enforcement})."
            ),
            improved_task_spec=improved_spec,
        )

    # Retries exhausted — §4.4
    exhaustion_action = _ON_EXHAUSTION.get(enforcement, AdjudicationAction.SKIP_TASK)
    return AdjudicationResult(
        action=exhaustion_action,
        reason=(
            f"Retries exhausted ({retry_count}/{max_retries}, "
            f"enforcement={enforcement}). "
            f"Action: {exhaustion_action.value}."
        ),
    )


def _build_improved_task_spec(issues: list[ReviewIssue]) -> dict[str, Any]:
    """Build an improved task spec that incorporates Reviewer feedback.  §4.3.

    The improved spec is included in the retry ``task.dispatch`` so the next
    Executor knows what went wrong and can correct it.
    """
    critical = [i for i in issues if i.severity == "critical"]
    important = [i for i in issues if i.severity == "important"]
    minor = [i for i in issues if i.severity == "minor"]

    feedback_parts: list[str] = []
    if critical:
        feedback_parts.append("CRITICAL issues to fix:")
        feedback_parts.extend(f"  - [{i.field}] {i.message}" for i in critical)
    if important:
        feedback_parts.append("Important issues to fix:")
        feedback_parts.extend(f"  - [{i.field}] {i.message}" for i in important)
    if minor:
        feedback_parts.append("Minor issues (address if possible):")
        feedback_parts.extend(f"  - [{i.field}] {i.message}" for i in minor)

    return {
        "reviewer_feedback": "\n".join(feedback_parts),
        "previous_issues": [i.model_dump() for i in issues],
    }
