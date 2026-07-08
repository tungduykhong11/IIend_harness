"""Streaming chunk assembly utilities for the Responder agent.

The Responder streams its LLM output as a series of ``respond.reply`` messages,
each carrying a text chunk.  The Orchestrator reassembles them for the human.
A non-streaming fallback sends the complete answer in a single message.

Spec references
===============
- **§8.1** → ``make_reply_chunk()`` — streaming chunk payload (chunk_index, stream=True, done=False)
- **§8.2** → ``make_final_reply()`` — non-streaming single-message payload
- **§8.3** → ``make_error_reply()`` — error response (stream=False, done=True, error=...)
- **§8.4** → ``reassemble_chunks()`` — Orchestrator-side chunk assembly by chunk_index
- **§3 ¶2** → ``respond.reply`` payload: answer, advice?, follow_up_suggestions?, confidence
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def make_reply_chunk(
    query_id: str,
    chunk_index: int,
    chunk_content: str,
    *,
    done: bool = False,
    final_answer: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Build the payload dict for a streaming ``respond.reply`` chunk.  §8.1.

    Parameters
    ----------
    query_id:
        The ``Message.id`` of the original ``respond.query`` this reply belongs to.
    chunk_index:
        0-based ordering index.
    chunk_content:
        The text delta for this chunk.
    done:
        ``True`` for the final chunk of the stream.
    final_answer:
        The complete assembled answer — only meaningful when *done* is ``True``.
    confidence:
        Confidence score 0.0–1.0 — only meaningful when *done* is ``True``.
    """
    payload: dict[str, Any] = {
        "query_id": query_id,
        "chunk_index": chunk_index,
        "chunk_content": chunk_content,
        "stream": True,
        "done": done,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if done:
        payload["final_answer"] = final_answer or ""
        payload["confidence"] = confidence if confidence is not None else 0.0
    return payload


def make_final_reply(
    query_id: str,
    answer: str,
    confidence: float = 1.0,
    *,
    advice: str | None = None,
    follow_up_suggestions: list[str] | None = None,
) -> dict[str, Any]:
    """Build the payload for a non-streaming ``respond.reply``.  §8.2, §3 ¶2.

    Matches the spec's non-streaming payload shape:
    ``{answer, advice?, follow_up_suggestions?, confidence}``.
    """
    payload: dict[str, Any] = {
        "query_id": query_id,
        "answer": answer,
        "stream": False,
        "done": True,
        "confidence": confidence,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if advice:
        payload["advice"] = advice
    if follow_up_suggestions:
        payload["follow_up_suggestions"] = follow_up_suggestions
    return payload


def make_error_reply(query_id: str, error: str) -> dict[str, Any]:
    """Build the payload for an error ``respond.reply``.  §8.3."""
    return {
        "query_id": query_id,
        "answer": "",
        "stream": False,
        "done": True,
        "confidence": 0.0,
        "error": error,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def reassemble_chunks(chunks: list[dict[str, Any]]) -> str:
    """Sort chunks by ``chunk_index`` and concatenate ``chunk_content`` fields.  §8.4.

    Parameters
    ----------
    chunks:
        Raw payload dicts from a series of ``respond.reply`` messages.

    Returns
    -------
    str
        The reassembled full text.
    """
    sorted_chunks = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
    return "".join(c.get("chunk_content", "") for c in sorted_chunks)
