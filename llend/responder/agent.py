"""ResponderAgent — long-lived conversational Q&A agent.

The Responder is spawned once per session and reused for every conversational
turn.  It holds an LLM client, maintains conversation history, supports four
personas, streams replies chunk-by-chunk, and can request tool execution
through the Orchestrator when it needs more data to answer a question.

Spec references
===============
- **§2.1** → ``ResponderAgent`` — role, mindset, lifespan (peer to Executor/Reviewer)
- **§2.4** → ``_detect_language()`` — auto-detect and match user's language
- **§4.1** → Lifecycle — spawn via ``start()``, shutdown via ``shutdown()``
- **§4.2** → IDLE substate — ``_inbox`` wait when no query is pending
- **§5.3** → LLM integration — ``LLMClient`` injected at construction
- **§6.1** → Tool request flow — ``respond.request_tool`` → await → ``respond.tool_result``
- **§6.2** → Tool matching by ``tool_call_id`` (the LLM's native tool-use identifier)
- **§8.1** → Streaming — multiple ``respond.reply`` chunks with ``chunk_index`` ordering
- **§8.2** → Non-streaming fallback — single ``respond.reply`` with ``stream=False``
- **§8.3** → Error handling — ``agent.error`` sent on crash alongside error reply
- **§3 ¶1** → ``register_handler()`` on runtime — fire-and-forget callback per message
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any
from uuid import UUID

from llend.llm.client import LLMClient
from llend.responder.context import SessionContext, TaskResultSummary
from llend.responder.memory import UserProfile
from llend.responder.persona import Persona, build_system_prompt
from llend.responder.stream import make_error_reply, make_final_reply, make_reply_chunk
from llend.runtime.base import AgentRuntime
from llend.runtime.lifecycle import AgentType
from llend.runtime.message import AgentErrorCode, Message, MsgType

logger = logging.getLogger(__name__)


class ResponderAgent:
    """Long-lived conversational agent.  §2.1, §4.1.

    Spawned once per session.  Receives ``respond.query`` messages from the
    Orchestrator, streams ``respond.reply`` chunks back, and optionally
    requests tool execution via ``respond.request_tool`` when more data is
    needed.  §6.1.
    """

    # ------------------------------------------------------------------
    # constructor  §5.3
    # ------------------------------------------------------------------

    def __init__(
        self,
        runtime: AgentRuntime,
        instance_id: str,
        session_id: UUID,
        llm_client: LLMClient,
        *,
        persona: Persona = Persona.AUTO,
        registry: Any = None,  # SkillRegistry | None — for tool listing  §6.1
        profile_path: Path | None = None,
    ) -> None:
        self._runtime = runtime
        self._instance_id = instance_id
        self._session_id = session_id
        self._llm = llm_client
        self._persona = persona
        self._registry = registry
        self._user_profile = UserProfile.load(profile_path)  # §9.2
        self._profile_path = profile_path or UserProfile._default_path()

        # Session state  §5.1
        self._session = SessionContext(session_goal="")
        self._inbox: asyncio.Queue[Message] = asyncio.Queue()  # §4.2 IDLE substate

        # Processing task
        self._processing_task: asyncio.Task[None] | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def session_context(self) -> SessionContext:
        return self._session

    @property
    def persona(self) -> Persona:
        return self._persona

    @persona.setter
    def persona(self, value: Persona) -> None:
        self._persona = value

    # ------------------------------------------------------------------
    # lifecycle  §4.1
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Register the message handler and begin the processing loop.  §4.1.

        Calls ``runtime.register_handler()`` (§3 ¶1) so the runtime fires
        ``_message_handler`` on every delivered message.
        """
        if self._closed:
            raise RuntimeError("ResponderAgent is closed and cannot be restarted.")

        await self._runtime.register_handler(self._instance_id, self._message_handler)
        self._processing_task = asyncio.create_task(
            self._processing_loop(),
            name=f"responder-loop-{self._instance_id}",
        )
        logger.info(
            "ResponderAgent started instance_id=%s persona=%s",
            self._instance_id,
            self._persona.value,
        )

    async def shutdown(self) -> None:
        """Cancel the processing loop and mark as closed.  §4.1.

        After ``shutdown()`` the Responder rejects further messages.
        """
        self._closed = True
        if self._processing_task is not None:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None
        logger.info("ResponderAgent shutdown instance_id=%s", self._instance_id)

    # ------------------------------------------------------------------
    # message handler  §3 ¶1  (called by runtime via register_handler)
    # ------------------------------------------------------------------

    async def _message_handler(self, message: Message) -> None:
        """Fire-and-forget callback registered with the runtime.  §3 ¶1.

        Pushes the message into the serialized inbox for single-threaded
        processing.  The runtime invokes this via ``asyncio.create_task``
        so ``send()`` is never blocked.
        """
        try:
            self._inbox.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning("Responder inbox full — dropping message id=%s", message.id)

    # ------------------------------------------------------------------
    # processing loop  §4.2
    # ------------------------------------------------------------------

    async def _processing_loop(self) -> None:
        """Single-threaded loop: reads one message at a time from the inbox.  §4.2.

        IDLE substate is implicit — when the inbox is empty the loop sits
        at ``_inbox.get()`` waiting for the next ``respond.query``.
        """
        while not self._closed:
            try:
                msg = await self._inbox.get()
            except asyncio.CancelledError:
                break

            try:
                if msg.msg_type == MsgType.RESPOND_QUERY:
                    await self._process_query(msg)
                elif msg.msg_type == MsgType.RESPOND_TOOL_RESULT:
                    logger.warning(
                        "responder received unexpected tool_result without active query"
                    )
                else:
                    logger.debug("responder ignoring msg_type=%s", msg.msg_type.value)
            except Exception:
                logger.exception("Error processing message id=%s", msg.id)

    # ------------------------------------------------------------------
    # query processing  §2.1, §8.1
    # ------------------------------------------------------------------

    async def _process_query(self, query_msg: Message) -> None:
        """Handle a single ``respond.query`` — the main entry point.  §2.1.

        1. Extract question + optional persona override.
        2. Update session context from Orchestrator-supplied payload.
        3. Detect language  §2.4.
        4. Build system prompt (persona + profile + context + language).
        5. Stream LLM response → send ``respond.reply`` chunks  §8.1.
        6. Handle tool calls if the LLM requests them  §6.1.
        7. Update conversation history  §5.2.
        """
        payload = query_msg.payload
        question: str = payload.get("question", "")
        if not question:
            await self._send_reply(query_msg, make_error_reply(str(query_msg.id), "Empty question received"))
            return

        # Update session context from Orchestrator-supplied payload  §5.1
        if "session_context" in payload:
            ctx_data = payload["session_context"]
            if isinstance(ctx_data, dict):
                if "task_results" in ctx_data:
                    self._session.task_results = [
                        TaskResultSummary(**tr) if isinstance(tr, dict) else tr
                        for tr in ctx_data["task_results"]
                    ]
                if "session_goal" in ctx_data:
                    self._session.session_goal = ctx_data["session_goal"]
                if "active_task" in ctx_data:
                    self._session.active_task = ctx_data["active_task"]

        # Persona override for this turn  §7.1
        persona_str = payload.get("persona")
        persona = Persona(persona_str) if persona_str else self._persona

        # Detect language  §2.4, §14 Q2
        language = self._detect_language(question)

        # Build system prompt  §7.3
        system = build_system_prompt(
            persona, self._user_profile, self._session, language=language
        )

        # Build message list for LLM
        messages = self._build_messages(question)

        # Track in-flight tool requests: request_id → tool_call_id  §6.1
        pending_tools: dict[str, str] = {}

        # Stream LLM response  §8.1
        all_text: list[str] = []
        chunk_index = 0

        try:
            async for event in self._llm.stream_generate(messages, system=system):
                if event.type == "text":
                    text = event.text_delta or ""
                    all_text.append(text)
                    await self._send_reply(
                        query_msg,
                        make_reply_chunk(str(query_msg.id), chunk_index, text),
                    )
                    chunk_index += 1

                elif event.type == "tool_call":
                    # Build the request message  §6.1
                    request_msg = self._build_message(
                        recipient="orchestrator",
                        msg_type=MsgType.RESPOND_REQUEST_TOOL,
                        payload={
                            "reason": f"Responder needs {event.tool_name} to answer the question",
                            "suggested_skill": event.tool_name,
                            "tool_params": event.tool_input or {},
                        },
                        parent_id=query_msg.id,
                    )
                    # Track mapping: request_id → LLM tool_call_id  §6.2
                    request_id = str(request_msg.id)
                    pending_tools[request_id] = event.tool_call_id or ""
                    await self._runtime.send(request_msg)

                    # Wait for tool result matching request_id  §6.1 step 6
                    tool_result = await self._await_tool_result(request_id=request_id)
                    if tool_result is not None:
                        # Append tool result to messages — LLM continues streaming  §6.1
                        result_data = tool_result.payload.get("result", {})
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": event.tool_call_id,
                                    "content": str(result_data),
                                }
                            ],
                        })
                        continue
                    else:
                        # Tool timeout  §6.1 — error reply + agent.error  §8.3
                        await self._send_error(
                            query_msg,
                            f"Tool '{event.tool_name}' timed out",
                            error_code=AgentErrorCode.TIMEOUT,
                        )
                        return

                elif event.type == "done":
                    break

                elif event.type == "error":
                    await self._send_error(
                        query_msg,
                        event.error_message or "LLM error",
                        error_code=AgentErrorCode.LLM_ERROR,
                    )
                    return

        except Exception as exc:
            logger.exception("Error during LLM streaming for query %s", query_msg.id)
            await self._send_error(
                query_msg, str(exc), error_code=AgentErrorCode.LLM_ERROR
            )
            return

        # Send final chunk  §8.1
        final_answer = "".join(all_text)
        await self._send_reply(
            query_msg,
            make_reply_chunk(
                str(query_msg.id),
                chunk_index,
                "",
                done=True,
                final_answer=final_answer,
                confidence=0.85,
            ),
        )

        # Update conversation history  §5.2
        self._session.add_turn("user", question)
        self._session.add_turn("responder", final_answer)

    # ------------------------------------------------------------------
    # tool result waiting  §6.2
    # ------------------------------------------------------------------

    async def _await_tool_result(
        self,
        request_id: str,
        timeout: float = 120.0,
    ) -> Message | None:
        """Block until a ``respond.tool_result`` with matching *request_id* arrives.  §6.2.

        The spec (§6.1) uses ``request_id`` — the UUID of the
        ``respond.request_tool`` message — as the matching key.
        Re-queues any ``respond.query`` messages that arrive while waiting.
        """
        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.warning("tool result timeout request_id=%s", request_id)
                return None

            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.warning("tool result timeout request_id=%s", request_id)
                return None

            if msg.msg_type == MsgType.RESPOND_TOOL_RESULT:
                result_request_id = str(msg.payload.get("request_id", ""))
                if result_request_id == request_id:
                    return msg
                logger.warning(
                    "unexpected tool result for %s, expected %s",
                    result_request_id,
                    request_id,
                )

            elif msg.msg_type == MsgType.RESPOND_QUERY:
                # Re-queue — process after current query finishes  §6.2
                self._inbox.put_nowait(msg)
            else:
                logger.debug("dropping msg_type=%s during tool wait", msg.msg_type.value)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_messages(self, current_query: str) -> list[dict[str, Any]]:
        """Convert conversation history to LLM message format.  §5.2."""
        messages: list[dict[str, Any]] = []
        for turn in self._session.conversation_history:
            role = "user" if turn.role == "user" else "assistant"
            messages.append({"role": role, "content": turn.content})
        messages.append({"role": "user", "content": current_query})
        return messages

    async def _send_reply(self, query_msg: Message, payload: dict[str, Any]) -> None:
        """Send a ``respond.reply`` message back through the runtime.  §3 ¶2."""
        await self._send_message(
            recipient="orchestrator",
            msg_type=MsgType.RESPOND_REPLY,
            payload=payload,
            parent_id=query_msg.id,
        )

    async def _send_error(
        self,
        query_msg: Message,
        error: str,
        error_code: AgentErrorCode = AgentErrorCode.UNKNOWN,
    ) -> None:
        """Send an error ``respond.reply`` AND an ``agent.error`` message.  §8.3, §3 ¶3."""
        # Error reply to Orchestrator (user-visible)
        await self._send_reply(query_msg, make_error_reply(str(query_msg.id), error))
        # Formal agent error for lifecycle tracking  §3 ¶3
        await self._send_message(
            recipient="orchestrator",
            msg_type=MsgType.AGENT_ERROR,
            payload={
                "error_code": error_code.value,
                "detail": error,
                "query_id": str(query_msg.id),
            },
            parent_id=query_msg.id,
        )

    def _build_message(
        self,
        recipient: str,
        msg_type: MsgType,
        payload: dict[str, Any],
        parent_id: UUID | None = None,
    ) -> Message:
        """Build a ``Message`` envelope without sending it.

        Used when the caller needs the message's ``id`` before dispatching
        (e.g. to track ``request_id`` for tool-result matching  §6.2).
        """
        return Message(
            session_id=self._session_id,
            sender=AgentType.RESPONDER.value,
            sender_instance=self._instance_id,
            recipient=recipient,
            msg_type=msg_type,
            payload=payload,
            parent_id=parent_id,
        )

    async def _send_message(
        self,
        recipient: str,
        msg_type: MsgType,
        payload: dict[str, Any],
        parent_id: UUID | None = None,
    ) -> None:
        """Build and send a message through the runtime in one step."""
        msg = self._build_message(recipient, msg_type, payload, parent_id)
        await self._runtime.send(msg)

    @staticmethod
    def _detect_language(text: str) -> str:
        """Heuristic language detection based on character counting.  §2.4.

        Returns one of ``"vi"``, ``"zh"``, ``"en"``.
        Used to inject an explicit language instruction into the system prompt  §7.3.
        """
        # Vietnamese tone-marked characters (both cases)  §2.4
        viet_pattern = re.compile(
            r"[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩị"
            r"òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"
            r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊ"
            r"ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]"
        )
        cjk_pattern = re.compile(r"[一-鿿㐀-䶿豈-﫿]")

        viet_count = len(viet_pattern.findall(text))
        cjk_count = len(cjk_pattern.findall(text))
        latin_count = len(re.findall(r"[a-zA-Z]", text))

        if cjk_count > max(viet_count, latin_count):
            return "zh"
        if viet_count > 0:
            return "vi"
        return "en"
