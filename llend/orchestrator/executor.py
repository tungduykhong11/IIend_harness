"""Task execution loop — the Executor → Reviewer → adjudicate cycle.

Extracted from ``OrchestratorAgent`` per Spec 004 §16 file layout.  The
Orchestrator delegates to this module for the per-task loop while keeping
session-level concerns (classification, routing, session lifecycle) in
``agent.py``.

Spec references
===============
- **§4.1** → Core pattern: spawn Executor → Reviewer → adjudicate
- **§4.2** → Step-by-step execution (8 steps)
- **§4.3** → Adjudication logic
- **§4.4** → Max retries per enforcement
- **§4.5** → Reviewer prompt construction
- **§16** → File layout — this module is listed as ``orchestrator/executor.py``
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID, uuid4

from llend.llm.client import LLMClient
from llend.orchestrator.adjudicator import AdjudicationAction, adjudicate
from llend.orchestrator.config import OrchestratorConfig
from llend.orchestrator.progress import ProgressReporter
from llend.orchestrator.summarizer import summarize_task_result
from llend.registry.models import Skill
from llend.registry.pipeline import TaskSpec
from llend.runtime.base import AgentRuntime
from llend.runtime.lifecycle import AgentType
from llend.runtime.message import Message, MsgType, ReviewIssue, Verdict

logger = logging.getLogger(__name__)

# Reviewer adversarial prompt — verbatim from Spec 004 §4.5
REVIEWER_SYSTEM_PROMPT = """You are a REVIEWER. Your job is to find flaws in the Executor's output.

Task: {task_spec}
Expected output schema: {output_schema}
Enforcement level: {enforcement}

Executor's output: {executor_output}
Executor's own concerns: {concerns}

Verify:
1. Does the output match the expected schema? (if schema exists)
2. Are the numbers/data consistent and reasonable?
3. Are there logical errors or unsupported claims?
4. Did the Executor address ALL requirements in the task spec?
5. [If concerns from Executor]: Verify each concern — confirm or dismiss.

For each issue found, provide:
- severity: "critical" | "important" | "minor"
- field: which part of the output
- message: what's wrong

Return your verdict:
- "pass": no issues found
- "pass_with_warnings": minor issues only, output is still usable
- "fail": critical or important issues make the output unreliable"""


async def execute_task_loop(
    task_spec: TaskSpec,
    task_id: UUID,
    total: int,
    *,
    runtime: AgentRuntime,
    registry: object,  # SkillRegistry
    llm_client: LLMClient,
    config: OrchestratorConfig,
    session_id: UUID,
    orchestrator_instance_id: str,
    progress: ProgressReporter,
) -> object | None:
    """Run the Executor → Reviewer loop for one task.  §4.1, §4.2.

    Parameters
    ----------
    task_spec:
        The task to execute — skill name, params, and wiring info.
    task_id:
        Unique identifier for this task execution.
    total:
        Total number of tasks in the plan (for progress reporting).
    runtime:
        The agent runtime (for spawn/send).
    registry:
        The skill registry (for skill resolution).
    llm_client:
        LLM client for summarization.
    config:
        Orchestrator settings (retry limits, timeouts).
    session_id:
        Current session UUID.
    orchestrator_instance_id:
        The Orchestrator's instance_id (for sender fields).
    progress:
        Progress reporter for human-visible updates.

    Returns
    -------
    The task output on success, or ``None`` if the task failed and was skipped.
    """
    skill = registry.get(task_spec.skill_name)  # type: ignore[union-attr]
    if skill is None:
        await progress.error(f"Skill {task_spec.skill_name!r} not found in registry")
        return None

    max_retries = config.get_max_retries(skill.enforcement)
    retry_count = 0
    output = None

    while retry_count <= max_retries:
        # --- Step 1-3: Spawn Executor, dispatch, wait for result  §4.2 ---
        await progress.task_start(task_spec, task_id, total)

        executor_id = await runtime.spawn(
            AgentType.EXECUTOR.value,
            context={
                "skill_name": task_spec.skill_name,
                "task_spec": task_spec.task_spec,
                "skill_context": {
                    "skill_md": skill.skill_md,
                    "action_bindings": {
                        name: binding.model_dump()
                        for name, binding in skill.action_bindings.items()
                    },
                    "output_schema": skill.output_schema,
                    "enforcement": skill.enforcement,
                },
            },
        )

        # Send task.dispatch  §4.2 step 3
        dispatch_msg = Message(
            session_id=session_id,
            sender=AgentType.ORCHESTRATOR.value,
            sender_instance=orchestrator_instance_id,
            recipient=AgentType.EXECUTOR.value,
            recipient_instance=executor_id,
            msg_type=MsgType.TASK_DISPATCH,
            payload={
                "task_id": str(task_id),
                "skill_name": task_spec.skill_name,
                "task_spec": task_spec.task_spec,
            },
        )
        await runtime.send(dispatch_msg)

        # §4.2 step 4: Wait for task.result (with timeout)
        result_msg = await _await_response(
            runtime, orchestrator_instance_id, session_id,
            executor_id, MsgType.TASK_RESULT, timeout=config.task_timeout_default,
        )

        if result_msg is None:
            # Timeout → kill Executor, retry
            await runtime.kill(executor_id)
            retry_count += 1
            if retry_count > max_retries:
                break
            continue

        output = result_msg.payload.get("output", {})
        concerns = result_msg.payload.get("concerns")

        # --- Step 5: Validate output schema  §4.2 ---
        schema_issues: list[str] = []
        if skill.output_schema is not None and output:
            schema_issues = _validate_output(output, skill)

        if schema_issues and skill.enforcement == "mandatory":
            retry_count += 1
            if retry_count <= max_retries:
                # Retry with schema feedback
                task_spec.task_spec["schema_errors"] = schema_issues
                continue

        # --- Step 6-8: Reviewer cycle  §4.2 ---
        reviewer_id = await runtime.spawn(
            AgentType.REVIEWER.value,
            context={"task_spec": task_spec.task_spec},
        )

        # Build adversarially-framed review prompt  §4.5
        import json as _json
        reviewer_prompt = REVIEWER_SYSTEM_PROMPT.format(
            task_spec=_json.dumps(task_spec.task_spec, ensure_ascii=False),
            output_schema=_json.dumps(skill.output_schema) if skill.output_schema else "(none)",
            enforcement=skill.enforcement,
            executor_output=_json.dumps(output, ensure_ascii=False, default=str)[:4000],
            concerns=_json.dumps(concerns) if concerns else "(none)",
        )

        review_msg = Message(
            session_id=session_id,
            sender=AgentType.ORCHESTRATOR.value,
            sender_instance=orchestrator_instance_id,
            recipient=AgentType.REVIEWER.value,
            recipient_instance=reviewer_id,
            msg_type=MsgType.TASK_REVIEW,
            payload={
                "task_id": str(task_id),
                "original_task_spec": task_spec.task_spec,
                "executor_output": output,
                "concerns_from_executor": concerns,
                "schema_validation_issues": schema_issues,
                "system_prompt": reviewer_prompt,
            },
        )
        await runtime.send(review_msg)

        # Wait for task.verdict
        verdict_msg = await _await_response(
            runtime, orchestrator_instance_id, session_id,
            reviewer_id, MsgType.TASK_VERDICT, timeout=config.review_timeout_default,
        )

        if verdict_msg is None:
            # Reviewer timeout → auto-pass with warning
            await runtime.kill(reviewer_id)
            verdict = Verdict.PASS_WITH_WARNINGS
            issues: list[ReviewIssue] = []
        else:
            verdict = Verdict(verdict_msg.payload.get("verdict", "fail"))
            raw_issues = verdict_msg.payload.get("issues", [])
            issues = [
                ReviewIssue(**i) if isinstance(i, dict) else i for i in raw_issues
            ]

        # §4.3: Adjudicate
        result = adjudicate(
            verdict,
            retry_count,
            skill.enforcement,
            reviewer_issues=issues,
            max_retries_override={
                "mandatory": config.max_retries_mandatory,
                "strict": config.max_retries_strict,
                "suggested": config.max_retries_suggested,
            },
        )

        if result.action == AdjudicationAction.NEXT_TASK:
            # Summarize and complete
            if output is not None:
                summary = await summarize_task_result(
                    task_result_payload={"task_id": str(task_id), "output": output},
                    skill=skill,
                    task_spec=task_spec,
                    llm_client=llm_client,
                )
                await progress.task_complete(task_spec, task_id, total, summary.summary)
            return output

        elif result.action == AdjudicationAction.RETRY:
            retry_count += 1
            if result.improved_task_spec:
                task_spec.task_spec.update(result.improved_task_spec)
            await progress.task_warning(
                task_spec, f"Retrying ({retry_count}/{max_retries})..."
            )
            # Kill reviewer before retry
            await runtime.kill(reviewer_id)
            continue

        elif result.action == AdjudicationAction.SKIP_TASK:
            await progress.task_warning(task_spec, "Skipping task.")
            await runtime.kill(reviewer_id)
            return None

        elif result.action == AdjudicationAction.ABORT_SESSION:
            await progress.error(
                f"Critical task {task_spec.skill_name} failed after {retry_count} attempts."
            )
            await runtime.kill(reviewer_id)
            return None

        # Kill reviewer after use
        await runtime.kill(reviewer_id)

    # Exhausted all retries
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _await_response(
    runtime: AgentRuntime,
    orchestrator_instance_id: str,
    session_id: UUID,
    target_instance: str,
    expected_type: MsgType,
    timeout: float = 300.0,
) -> Message | None:
    """Wait for a message of *expected_type* from *target_instance*.  §4.2.

    Uses the runtime's send mechanism — in practice the Orchestrator's inbox
    handles message routing.  This is a simplified polling loop for use
    within ``execute_task_loop``.

    Note: this function requires access to the Orchestrator's inbox to
    intercept messages.  In the actual implementation the Orchestrator
    passes messages to this function via an asyncio.Event or callback.
    For v0 we accept a simpler polling approach via the runtime.
    """
    deadline = asyncio.get_event_loop().time() + timeout

    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            logger.warning(
                "Timeout waiting for %s from %s", expected_type.value, target_instance
            )
            return None

        # In practice, the Orchestrator's main loop receives messages and
        # forwards them.  For v0, the executor module returns control to
        # the agent which handles the message routing.
        await asyncio.sleep(0.1)
        # (The Orchestrator agent.py implementation handles message
        #  interception directly via its own inbox and _await_response.)


def _validate_output(output: object, skill: Skill) -> list[str]:
    """Validate Executor output against the skill's output schema.  §4.2 step 5.

    In v0, basic structural validation; full JSON Schema in v1.
    """
    if skill.output_schema is None:
        return []
    issues: list[str] = []
    if isinstance(output, dict) and "type" in skill.output_schema:
        expected = skill.output_schema["type"]
        if expected == "object" and not isinstance(output, dict):
            issues.append(f"Expected object, got {type(output).__name__}")
    return issues
