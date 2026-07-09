"""Context summarization — distilling task results and synthesising final output.

After every task completes, the Orchestrator produces a ``TaskResultSummary``
for the Responder's consumption.  At session end, it generates a human-readable
synthesis of everything that happened.

Spec references
===============
- **§7.1** → Why summarise: Responder context + session memory
- **§7.2** → ``summarize_task_result()`` — LLM prompt (verbatim)
- **§7.3** → ``SessionState`` accumulation model
- **§11.4** → ``synthesize_session()`` — final synthesis prompt (verbatim)
- **§17** → Model selection: cheap for summaries, capable for synthesis
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from llend.llm.client import LLMClient
from llend.registry.models import Skill
from llend.registry.pipeline import TaskSpec
from llend.responder.context import ConversationTurn, TaskResultSummary
from llend.runtime.message import TaskStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task-result summarization  —  §7.2
# ---------------------------------------------------------------------------

TASK_SUMMARY_PROMPT = """Summarize this task execution result in 1-3 sentences.
Extract up to 5 key numeric metrics.

Skill: {skill_name} — {skill_description}
Task params: {task_params}
Output: {output_snippet}

Respond with JSON:
{{
  "summary": "...",
  "key_metrics": {{"metric_name": value, ...}},
  "notable_findings": ["..."]  // optional, empty list if none
}}"""


async def summarize_task_result(
    task_result_payload: dict[str, Any],
    skill: Skill,
    task_spec: TaskSpec,
    llm_client: LLMClient,
) -> TaskResultSummary:
    """Generate a 1–3 sentence summary + key metrics from a task result.  §7.2.

    Uses a cheap LLM (Haiku, per §17).  The output is a ``TaskResultSummary``
    that the Responder can reference in follow-up answers.
    """
    output_raw = json.dumps(task_result_payload.get("output", {}))
    output_snippet = output_raw[:2000]

    prompt = TASK_SUMMARY_PROMPT.format(
        skill_name=skill.name,
        skill_description=skill.description,
        task_params=json.dumps(task_spec.task_spec),
        output_snippet=output_snippet,
    )

    try:
        raw = await llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        data = json.loads(raw)
        return TaskResultSummary(
            task_id=UUID(task_result_payload.get("task_id", str(UUID(int=0)))),
            skill_name=skill.name,
            status=TaskStatus(task_result_payload.get("status", "done")),
            summary=data.get("summary", f"Completed {skill.name}."),
            key_metrics=data.get("key_metrics", {}),
            artifact_paths=task_result_payload.get("artifacts", []),
        )
    except Exception:
        logger.exception("Task summarization failed for %s", skill.name)
        return TaskResultSummary(
            task_id=UUID(task_result_payload.get("task_id", str(UUID(int=0)))),
            skill_name=skill.name,
            status=TaskStatus(task_result_payload.get("status", "done")),
            summary=f"Completed {skill.name}.",
            key_metrics={},
            artifact_paths=task_result_payload.get("artifacts", []),
        )


# ---------------------------------------------------------------------------
# Final session synthesis  —  §11.4
# ---------------------------------------------------------------------------

SESSION_SYNTHESIS_PROMPT = """You are a session synthesizer. Given the completed tasks and conversation,
write a clear summary of what was accomplished.

Session goal: {session_goal}

Completed tasks:
{task_summaries}

Conversation highlights:
{conversation_highlights}

Write a summary that:
1. Answers the original session goal directly
2. Highlights the most important findings
3. Mentions any limitations or skipped tasks
4. Suggests next steps if applicable
5. Lists all generated artifact files"""


async def synthesize_session(
    session_goal: str,
    completed_tasks: list[TaskResultSummary],
    conversation_history: list[ConversationTurn],
    llm_client: LLMClient,
    *,
    skipped_tasks: list[str] | None = None,
    artifacts: list[str] | None = None,
) -> str:
    """Generate a human-readable summary of the entire session.  §11.4.

    Uses a capable LLM (Sonnet, per §17) because the synthesis is user-facing
    and should be well-written.
    """
    # Build task summaries block
    task_lines: list[str] = []
    for ts in completed_tasks:
        task_lines.append(f"- {ts.skill_name}: {ts.summary}")
        if ts.key_metrics:
            metrics_str = ", ".join(f"{k}={v}" for k, v in ts.key_metrics.items())
            task_lines.append(f"  Key metrics: {metrics_str}")

    task_summaries = "\n".join(task_lines) if task_lines else "(no tasks completed)"

    # Build conversation highlights (last 5 turns)  §11.4
    last_turns = conversation_history[-5:]
    conv_lines: list[str] = []
    for turn in last_turns:
        role_label = "User" if turn.role == "user" else "Responder"
        # Truncate long content
        content = turn.content[:200] + "..." if len(turn.content) > 200 else turn.content
        conv_lines.append(f"- {role_label}: {content}")

    conversation_highlights = "\n".join(conv_lines) if conv_lines else "(no conversation)"

    prompt = SESSION_SYNTHESIS_PROMPT.format(
        session_goal=session_goal,
        task_summaries=task_summaries,
        conversation_highlights=conversation_highlights,
    )

    if skipped_tasks:
        prompt += f"\n\nSkipped tasks (could not complete): {', '.join(skipped_tasks)}"
    if artifacts:
        prompt += f"\n\nGenerated artifacts:\n" + "\n".join(f"- {a}" for a in artifacts)

    try:
        return await llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:
        logger.exception("Session synthesis failed")
        # Fallback: simple concatenation
        parts: list[str] = [f"Session: {session_goal}\n"]
        for ts in completed_tasks:
            parts.append(f"- {ts.skill_name}: {ts.summary}")
        if skipped_tasks:
            parts.append(f"\nNote: Could not complete: {', '.join(skipped_tasks)}")
        return "\n".join(parts)
