"""ReviewerAgent — the adversarial verifier.  Spec 005 §3.

The Reviewer is spawned fresh after each Executor completes.  It receives
``task.review`` with a fully constructed adversarial system prompt (built
by the Orchestrator per Spec 004 §4.5), performs a single LLM call, and
returns ``task.verdict``.

This is a **thin wrapper** (~50 lines): the Orchestrator does all the
heavy lifting (prompt construction, verdict adjudication).  The Reviewer
just calls the LLM and parses the JSON verdict.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from typing import Any
from uuid import UUID

from llend.llm.client import LLMClient
from llend.runtime.lifecycle import AgentType
from llend.runtime.message import (
    AgentErrorCode,
    Message,
    MsgType,
    Verdict,
)

logger = logging.getLogger(__name__)


class ReviewerAgent:
    """Adversarial verifier — receives ``task.review``, returns ``task.verdict``.  §3.

    Parameters
    ----------
    runtime:
        The runtime that spawned this agent.
    instance_id:
        Unique instance id from ``runtime.spawn()``.
    session_id:
        Current session id.
    llm_client:
        Any ``LLMClient`` implementation.
    """

    def __init__(
        self,
        runtime: Any,  # AgentRuntime
        instance_id: str,
        session_id: UUID,
        llm_client: LLMClient,
    ) -> None:
        self._runtime = runtime
        self._instance_id = instance_id
        self._session_id = session_id
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Lifecycle  §3
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Register handler and wait for ``task.review``.

        Processes exactly ONE review and then exits.  The agent is
        disposable — fresh context per task (§3).
        """
        await self._runtime.register_handler(self._instance_id, self._handler)

        handle = self._runtime._agents.get(self._instance_id)
        if handle is None:
            logger.error("Reviewer %s: no handle found after spawn", self._instance_id)
            return

        try:
            msg = await handle.queue.get()
            if msg.msg_type != MsgType.TASK_REVIEW:
                logger.warning(
                    "Reviewer %s: expected task.review, got %s",
                    self._instance_id, msg.msg_type.value,
                )
                await self._send_error(
                    AgentErrorCode.UNKNOWN,
                    f"Unexpected message type: {msg.msg_type.value}",
                )
                return

            await self._review(msg)

        except asyncio.CancelledError:
            logger.info("Reviewer %s cancelled", self._instance_id)
        except Exception:
            logger.exception("Reviewer %s crashed", self._instance_id)
            await self._send_error(
                AgentErrorCode.CRASH,
                "Reviewer crashed — see logs for traceback",
            )

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def _handler(self, message: Message) -> None:
        """No-op — the agent blocks on handle.queue.get() directly."""

    # ------------------------------------------------------------------
    # Review logic  §3
    # ------------------------------------------------------------------

    async def _review(self, msg: Message) -> None:
        """Call the LLM with the pre-built adversarial prompt.  §3.

        The Orchestrator constructed the full ``system_prompt`` from the
        template in Spec 004 §4.5 and placed it in the ``task.review``
        payload (§3 — canonical payload: ``{task_id, task_spec,
        executor_output, system_prompt, concerns?, schema_validation_issues?}``).
        """
        payload = msg.payload
        task_id = payload.get("task_id", "")
        system_prompt = payload.get("system_prompt", "")

        if not system_prompt:
            logger.error("Reviewer %s: no system_prompt in task.review payload", self._instance_id)
            await self._send_error(
                AgentErrorCode.VALIDATION_ERROR,
                "Missing system_prompt in task.review payload",
            )
            return

        try:
            response_text = await self._llm.generate(
                messages=[{"role": "user", "content": "Review carefully."}],
                system=system_prompt,
            )
            verdict_data = self._parse_verdict(response_text)
        except Exception as exc:
            logger.exception("Reviewer %s: LLM call failed", self._instance_id)
            await self._send_error(AgentErrorCode.LLM_ERROR, str(exc))
            return

        await self._send_verdict(task_id, verdict_data)

    # ------------------------------------------------------------------
    # Verdict parsing  §3
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_verdict(text: str) -> dict[str, Any]:
        """Parse the LLM's JSON verdict output.  §3.

        Returns a dict with ``verdict``, ``issues``, and ``confidence`` keys.
        Falls back to a fail verdict if parsing fails.
        """
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            data = _json.loads(text)
            if isinstance(data, dict):
                verdict_raw = data.get("verdict", "fail")
                # Validate verdict value
                try:
                    Verdict(verdict_raw)
                except ValueError:
                    verdict_raw = "fail"
                return {
                    "verdict": verdict_raw,
                    "issues": data.get("issues", []),
                    "confidence": float(data.get("confidence", 0.0)),
                }
        except (_json.JSONDecodeError, TypeError, ValueError):
            pass

        # Fallback — treat unparseable as fail
        return {
            "verdict": "fail",
            "issues": [{
                "severity": "critical",
                "field": "output",
                "message": "Reviewer could not parse LLM response as valid JSON verdict",
            }],
            "confidence": 0.0,
        }

    # ------------------------------------------------------------------
    # Verdict / error dispatch  §3
    # ------------------------------------------------------------------

    async def _send_verdict(self, task_id: str, verdict_data: dict[str, Any]) -> None:
        """Send ``task.verdict`` to the Orchestrator.  §3."""
        msg = Message(
            session_id=self._session_id,
            sender=AgentType.REVIEWER.value,
            sender_instance=self._instance_id,
            recipient=AgentType.ORCHESTRATOR.value,
            msg_type=MsgType.TASK_VERDICT,
            payload={
                "task_id": task_id,
                "verdict": verdict_data.get("verdict", "fail"),
                "issues": verdict_data.get("issues", []),
                "confidence": verdict_data.get("confidence", 0.0),
            },
        )
        await self._runtime.send(msg)
        logger.info(
            "Reviewer %s: verdict sent verdict=%s confidence=%.2f",
            self._instance_id,
            verdict_data.get("verdict"),
            verdict_data.get("confidence", 0.0),
        )

    async def _send_error(self, code: AgentErrorCode, detail: str) -> None:
        """Send ``agent.error`` to the Orchestrator.  §3."""
        msg = Message(
            session_id=self._session_id,
            sender=AgentType.REVIEWER.value,
            sender_instance=self._instance_id,
            recipient=AgentType.ORCHESTRATOR.value,
            msg_type=MsgType.AGENT_ERROR,
            payload={
                "error_code": code.value,
                "detail": detail,
            },
        )
        await self._runtime.send(msg)
        logger.info(
            "Reviewer %s: agent.error sent code=%s", self._instance_id, code.value,
        )
