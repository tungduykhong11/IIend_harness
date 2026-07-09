"""Error recovery — exponential backoff, crash handling, graceful degradation.

When agents crash, LLM APIs fail, or interrupts time out, the Orchestrator
decides what to do next.  This module implements the error-response table
from Spec 004 §10.1.

Spec references
===============
- **§10.1** → Error types & responses table
- **§10.2** → Exponential backoff for LLM errors (code sketch verbatim)
- **§10.3** → Graceful degradation — skip non-mandatory tasks, note failures
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from llend.registry.pipeline import ExecutionPlan, TaskSpec
from llend.runtime.message import AgentErrorCode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exponential backoff  —  §10.2 (verbatim code sketch adapted)
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Raised when an LLM API call fails after all retries."""


async def with_llm_retry(
    callable: Callable[[], Awaitable[Any]],
    max_retries: int = 4,
    base_delay: float = 1.0,
) -> Any:
    """Wrap an LLM call with exponential backoff.  §10.2.

    Retries up to *max_retries* times, doubling the delay each time
    (1s → 2s → 4s → 8s).  Raises ``LLMError`` on exhaustion.
    """
    for attempt in range(max_retries):
        try:
            return await callable()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise LLMError(
                    f"LLM call failed after {max_retries} attempts: {exc}"
                ) from exc
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "LLM error, retrying in %.1fs (attempt %d/%d)",
                delay, attempt + 1, max_retries,
            )
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Error recovery decisions  —  §10.1 table
# ---------------------------------------------------------------------------


class RecoveryAction(BaseModel):
    """What to do after an error is detected.  §10.1."""

    action: str  # "retry" | "respawn" | "auto_pass" | "escalate" | "skip" | "restart"
    max_attempts: int = 2
    reason: str = ""


# Map error codes → recovery actions  §10.1
_ERROR_RECOVERY: dict[AgentErrorCode, RecoveryAction] = {
    AgentErrorCode.CRASH: RecoveryAction(
        action="respawn",
        max_attempts=2,
        reason="Re-spawn agent (max 2). On exhaustion → mark task FAILED.",
    ),
    AgentErrorCode.LLM_ERROR: RecoveryAction(
        action="retry",
        max_attempts=4,
        reason="Exponential backoff (1s, 2s, 4s, 8s). On exhaustion → escalate to human.",
    ),
    AgentErrorCode.TOOL_ERROR: RecoveryAction(
        action="review_partial",
        max_attempts=1,
        reason="Executor handles internally (ActionDispatcher retry). If Executor reports failure → Reviewer evaluates partial output.",
    ),
    AgentErrorCode.VALIDATION_ERROR: RecoveryAction(
        action="retry",
        max_attempts=3,
        reason="Retry with schema feedback (see §4.2 step 5).",
    ),
    AgentErrorCode.TIMEOUT: RecoveryAction(
        action="respawn",
        max_attempts=2,
        reason="Re-spawn Executor (max 2). On exhaustion → mark task FAILED.",
    ),
    AgentErrorCode.INTERRUPT_TIMEOUT: RecoveryAction(
        action="skip",
        max_attempts=1,
        reason="Mark task as INCOMPLETE. Continue plan if possible. Notify human.",
    ),
    AgentErrorCode.UNKNOWN: RecoveryAction(
        action="respawn",
        max_attempts=2,
        reason="Re-spawn Executor (max 2). On exhaustion → mark task FAILED.",
    ),
}


def get_recovery_action(error_code: AgentErrorCode) -> RecoveryAction:
    """Return the recovery action for a given error code.  §10.1."""
    return _ERROR_RECOVERY.get(
        error_code,
        RecoveryAction(action="respawn", max_attempts=2, reason="Unknown error — re-spawn max 2."),
    )


# ---------------------------------------------------------------------------
# Orchestrator crash recovery  —  §10.1 row 7
# ---------------------------------------------------------------------------


async def handle_orchestrator_crash(
    checkpoint_dir: str = "~/.llend/checkpoints",
    session_id: str | None = None,
) -> dict[str, object] | None:
    """Attempt to restart the Orchestrator from the last checkpoint.  §10.1 row 7.

    Spec: "Runtime detects. Attempt restart from last checkpoint (Spec 001 §3.4).
    Session state recovered from disk."

    Returns the recovered session state dict if a checkpoint exists, or ``None``
    if no recovery is possible.  The caller (Runtime) should re-create the
    Orchestrator with the recovered state.
    """
    import json
    from pathlib import Path

    ckpt_dir = Path(checkpoint_dir).expanduser()
    if not ckpt_dir.exists():
        return None

    # Find the most recent checkpoint file for this session
    checkpoint_files = sorted(
        ckpt_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not checkpoint_files:
        return None

    for ckpt_file in checkpoint_files:
        try:
            data = json.loads(ckpt_file.read_text(encoding="utf-8"))
            if session_id is None or data.get("session_id") == session_id:
                return data
        except (json.JSONDecodeError, OSError):
            continue

    return None


# ---------------------------------------------------------------------------
# Crash handlers  —  §10.1
# ---------------------------------------------------------------------------


def handle_executor_crash(
    attempt: int,
    max_attempts: int = 2,
) -> RecoveryAction:
    """Decide what to do when an Executor crashes.  §10.1 row 1.

    Re-spawn up to *max_attempts* times.  On exhaustion → mark task FAILED.
    """
    if attempt < max_attempts:
        return RecoveryAction(
            action="respawn",
            max_attempts=max_attempts - attempt,
            reason=f"Executor crashed — re-spawning (attempt {attempt + 1}/{max_attempts}).",
        )
    return RecoveryAction(
        action="skip",
        max_attempts=0,
        reason=f"Executor crashed {max_attempts} times — marking task FAILED.",
    )


def handle_reviewer_crash(
    attempt: int,
    max_attempts: int = 2,
) -> RecoveryAction:
    """Decide what to do when a Reviewer crashes.  §10.1 row 2.

    Re-spawn up to *max_attempts* times.  On exhaustion → auto-pass with warning.
    """
    if attempt < max_attempts:
        return RecoveryAction(
            action="respawn",
            max_attempts=max_attempts - attempt,
            reason=f"Reviewer crashed — re-spawning (attempt {attempt + 1}/{max_attempts}).",
        )
    return RecoveryAction(
        action="auto_pass",
        max_attempts=0,
        reason=f"Reviewer crashed {max_attempts} times — auto-passing with warning.",
    )


# ---------------------------------------------------------------------------
# Graceful degradation  —  §10.3
# ---------------------------------------------------------------------------


def should_skip_downstream(
    failed_skill_name: str,
    plan: ExecutionPlan,
) -> list[str]:
    """Return downstream tasks that depend on a failed task.  §10.3.

    If a non-mandatory task fails, any downstream task that lists it in
    ``input_from`` is also skipped.  This prevents cascading failures.
    """
    affected: list[str] = []
    for ts in plan.skills:
        if ts.skill_name == failed_skill_name:
            continue
        if ts.input_from and failed_skill_name in ts.input_from:
            affected.append(ts.skill_name)
    return affected
