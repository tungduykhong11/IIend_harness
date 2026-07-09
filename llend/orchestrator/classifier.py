"""Message classification — the Orchestrator's first decision point.

Every human message is classified into one of four categories so the
Orchestrator knows how to route it.  Classification uses a cheap LLM
(Haiku) with a fixed prompt per Spec 003 §3.3.

Spec references
===============
- **§3.1** → The classification problem — 4 categories with example messages
- **§3.2** → Classification logic diagram
- **§3.3** → Classification prompt (reproduced verbatim below)
- **§3.4** → Routing table — category → message type mapping
- **§17** → Decision: cheap LLM (Haiku) for classification
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum

from pydantic import BaseModel, Field

from llend.llm.client import LLMClient
from llend.runtime.message import MsgType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & models
# ---------------------------------------------------------------------------


class MessageCategory(StrEnum):
    """The four categories a human message can be classified into.  §3.1 table."""

    TASK = "task"
    CONVERSATIONAL = "conversational"
    SESSION_END = "session_end"
    CONTROL = "control"


class ClassificationResult(BaseModel):
    """Output of the classification LLM call.  §3.3."""

    category: MessageCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Routing table  —  §3.4
# ---------------------------------------------------------------------------

ROUTING_TABLE: dict[MessageCategory, MsgType | None] = {
    MessageCategory.TASK: MsgType.TASK_DISPATCH,
    MessageCategory.CONVERSATIONAL: MsgType.RESPOND_QUERY,
    MessageCategory.SESSION_END: MsgType.SESSION_COMPLETE,
    MessageCategory.CONTROL: None,  # Handled internally — no single MsgType
}

# ---------------------------------------------------------------------------
# Classification prompt  —  §3.3 (verbatim)
# ---------------------------------------------------------------------------

CLASSIFICATION_PROMPT = """You are a message classifier for an AI agent harness.

Given a user message, classify it into exactly ONE category:
- "task": The user wants an ACTION performed (crawl, analyze, export, search, compare).
           These map to skills in the registry.
- "conversational": The user wants an OPINION, EXPLANATION, ADVICE, or FOLLOW-UP question.
                    These do NOT require running a skill.
- "session_end": The user is saying goodbye or indicating the session is done.
- "control": The user wants to control the session itself (cancel, pause, status).

User message: "{message}"

Respond with JSON: {{"category": "...", "confidence": 0.0-1.0, "reasoning": "..."}}"""


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


async def classify_message(
    message: str,
    llm_client: LLMClient,
    *,
    model: str | None = None,
) -> ClassificationResult:
    """Classify a human message into one of four categories.  §3.2.

    Uses a cheap LLM (default: Haiku, per §17) with the prompt from §3.3.
    The response is parsed as JSON and validated into a ``ClassificationResult``.

    If the LLM call fails or returns unparseable output, defaults to
    ``conversational`` as the safest fallback.
    """
    prompt = CLASSIFICATION_PROMPT.format(message=message)

    try:
        raw = await llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
        )
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        data = json.loads(raw)
        return ClassificationResult(
            category=MessageCategory(data.get("category", "conversational")),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=str(data.get("reasoning", "")),
        )
    except Exception:
        logger.exception("Classification failed — defaulting to conversational")
        return ClassificationResult(
            category=MessageCategory.CONVERSATIONAL,
            confidence=0.0,
            reasoning="Classification failed — falling back to conversational.",
        )


def route_message(category: MessageCategory) -> MsgType:
    """Return the ``MsgType`` that corresponds to *category*.  §3.4.

    ``session_end`` and ``control`` both map to ``SESSION_COMPLETE``; the
    Orchestrator's ``_main_loop`` distinguishes them by checking
    ``category`` directly before routing.
    """
    return ROUTING_TABLE.get(category, MsgType.RESPOND_QUERY)
