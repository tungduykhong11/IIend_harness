"""Responder persona definitions and system prompts.

Spec references
===============
- **§7.1** → ``Persona`` enum (AUTO, ANALYST, ADVISOR, FRIEND)
- **§7.2** → ``PERSONA_SYSTEM_PROMPTS`` — per-persona system prompt snippets
- **§7.3** → ``build_system_prompt()`` — merges persona + user profile + session context
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llend.responder.context import SessionContext
    from llend.responder.memory import UserProfile


class Persona(StrEnum):
    """Responder's tone of voice.  §7.1.

    Three explicit personas + an auto-detect mode.  The Responder selects
    a persona at session start (via ``UserProfile.persona_preference``) and
    can switch mid-session if the Orchestrator passes a ``persona`` field on
    ``respond.query``.
    """

    AUTO = "auto"          # Detect best persona from question type (default)
    ANALYST = "analyst"    # Data-driven, precise, cites numbers  §7.2.1
    ADVISOR = "advisor"    # Practical recommendations, pros/cons   §7.2.2
    FRIEND = "friend"      # Casual, simple explanations           §7.2.3


PERSONA_SYSTEM_PROMPTS: dict[Persona, str] = {
    Persona.AUTO: (
        "You are a helpful, versatile AI assistant. "
        "Adapt your tone and style to match the user's question: "
        "be analytical for data questions, practical for advice, and friendly for casual chat. "
        "Always respond in the same language as the user's query."
    ),
    Persona.ANALYST: (
        "You are a sharp data analyst. "
        "Be precise, quantitative, and evidence-based in every response. "
        "Cite specific numbers, metrics, and data points from the available task results. "
        "Never speculate without data — if the data is insufficient, say so clearly. "
        "Use structured formats (tables, bullet points) when comparing values. "
        "Always respond in the same language as the user's query."
    ),
    Persona.ADVISOR: (
        "You are a practical advisor focused on helping the user make good decisions. "
        "Weigh pros and cons for every recommendation. "
        "Consider the user's budget, preferences, and constraints. "
        "Give actionable, concrete advice — not abstract principles. "
        "When there are trade-offs, explain them clearly and suggest a balanced choice. "
        "Always respond in the same language as the user's query."
    ),
    Persona.FRIEND: (
        "You are a friendly, conversational assistant. "
        "Use casual, everyday language that anyone can understand. "
        "Keep explanations simple and jargon-free. "
        "Be warm, empathetic, and encouraging. "
        "Use analogies and examples to make complex ideas accessible. "
        "Always respond in the same language as the user's query."
    ),
}


def build_system_prompt(
    persona: Persona,
    user_profile: "UserProfile | None" = None,
    session_context: "SessionContext | None" = None,
    language: str = "en",
) -> str:
    """Build a complete system prompt by merging persona, user profile, and session context.  §7.3.

    Parameters
    ----------
    persona:
        The persona whose base system prompt to use.
    user_profile:
        Optional — user preferences that personalize the prompt.  §9.1.
    session_context:
        Optional — current session state (goal, completed tasks, conversation).  §5.1.
    language:
        Detected language code (e.g. ``"vi"``, ``"en"``, ``"zh"``).  §14 Q2.
        Injected into the prompt as an explicit instruction.
    """
    parts: list[str] = [PERSONA_SYSTEM_PROMPTS[persona]]

    # Explicit language instruction  §14 Q2
    lang_names: dict[str, str] = {
        "vi": "Vietnamese (Tiếng Việt)",
        "en": "English",
        "zh": "Chinese (中文)",
    }
    lang_name = lang_names.get(language, language)
    parts.append(f"IMPORTANT: Respond in {lang_name}. Match the user's language exactly.")

    # User profile context  §9.1
    if user_profile is not None:
        profile_lines: list[str] = []
        if user_profile.preferred_platforms:
            profile_lines.append(
                f"User prefers these platforms: {', '.join(user_profile.preferred_platforms)}."
            )
        if user_profile.favorite_categories:
            profile_lines.append(
                f"User is interested in: {', '.join(user_profile.favorite_categories)}."
            )
        if user_profile.budget_conscious:
            profile_lines.append("User is budget-conscious — highlight good deals and value.")
        if user_profile.custom_notes:
            for key, value in user_profile.custom_notes.items():
                profile_lines.append(f"User note ({key}): {value}")

        if profile_lines:
            parts.append("\n## User Profile\n" + "\n".join(profile_lines))

    # Session context  §5.1
    if session_context is not None:
        ctx_lines: list[str] = [f"Session goal: {session_context.session_goal}"]

        if session_context.task_results:
            ctx_lines.append("\nCompleted tasks so far:")
            for ts in session_context.task_results[-5:]:  # last 5 only for brevity
                ctx_lines.append(f"- {ts.skill_name}: {ts.summary}")
                if ts.key_metrics:
                    metrics_str = ", ".join(
                        f"{k}={v}" for k, v in ts.key_metrics.items()
                    )
                    ctx_lines.append(f"  Metrics: {metrics_str}")

        if session_context.active_task:
            ctx_lines.append(
                f"\nCurrently running task: {session_context.active_task}"
            )

        parts.append("\n## Session Context\n" + "\n".join(ctx_lines))

    return "\n\n".join(parts)
