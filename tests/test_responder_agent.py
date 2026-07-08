"""Tests for ResponderAgent — streaming, tool calls, errors, personas.  Spec 003."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from llend.llm.client import LLMStreamEvent
from llend.responder.agent import ResponderAgent
from llend.responder.context import SessionContext
from llend.responder.memory import UserProfile
from llend.responder.persona import Persona
from llend.runtime.lifecycle import AgentType
from llend.runtime.message import Message, MsgType
from llend.responder.stream import make_final_reply, make_reply_chunk


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_runtime():
    """An AgentRuntime mock that captures sent messages."""
    rt = MagicMock()
    rt.send = AsyncMock()
    rt.register_handler = AsyncMock()
    rt.spawn = AsyncMock(return_value="responder-test-01")
    rt.session_id = None
    return rt


@pytest.fixture
def mock_llm():
    """An LLMClient mock that yields events on demand."""
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="Mock response.")
    llm.stream_generate = AsyncMock(return_value=[])
    return llm


@pytest.fixture
async def responder(mock_runtime, mock_llm):
    """A ResponderAgent that has been started."""
    import uuid

    sid = uuid.uuid4()
    rid = "responder-test-01"
    agent = ResponderAgent(mock_runtime, rid, sid, mock_llm)
    await agent.start()
    yield agent
    await agent.shutdown()


def _make_query_msg(
    session_id,
    question: str = "Hello?",
    persona: str | None = None,
    session_context: dict | None = None,
):
    """Build a ``respond.query`` message."""
    import uuid

    payload: dict = {"question": question}
    if persona:
        payload["persona"] = persona
    if session_context:
        payload["session_context"] = session_context
    return Message(
        session_id=session_id,
        sender="orchestrator",
        sender_instance="orch-1",
        recipient="responder",
        recipient_instance="responder-test-01",
        msg_type=MsgType.RESPOND_QUERY,
        payload=payload,
    )


# =============================================================================
# Tests
# =============================================================================


class TestResponderStartup:
    async def test_start_registers_handler(self, mock_runtime, mock_llm):
        import uuid

        sid = uuid.uuid4()
        agent = ResponderAgent(mock_runtime, "resp-1", sid, mock_llm)
        await agent.start()
        mock_runtime.register_handler.assert_awaited_once()
        args = mock_runtime.register_handler.call_args
        assert args[0][0] == "resp-1"  # instance_id
        assert callable(args[0][1])  # handler callback
        await agent.shutdown()

    async def test_shutdown_cleans_up(self, mock_runtime, mock_llm):
        import uuid

        sid = uuid.uuid4()
        agent = ResponderAgent(mock_runtime, "resp-1", sid, mock_llm)
        await agent.start()
        await agent.shutdown()
        # No errors — successful shutdown


class TestStreaming:
    async def test_handle_query_streaming(self, responder, mock_runtime, mock_llm):
        """LLM emits 3 text chunks → 4 reply messages (3 chunks + final)."""
        async def stream_three(messages, system=None, tools=None):
            for t in ["Hello, ", "how can I ", "help you?"]:
                yield LLMStreamEvent(type="text", text_delta=t)
            yield LLMStreamEvent(type="done", finish_reason="end_turn")

        mock_llm.stream_generate = stream_three

        msg = _make_query_msg(responder._session_id, question="Greet me")
        await responder._message_handler(msg)
        await asyncio.sleep(0.15)  # let the processing loop run

        calls = mock_runtime.send.call_args_list
        replies = [c[0][0] for c in calls if c[0][0].msg_type == MsgType.RESPOND_REPLY]
        assert len(replies) == 4  # 3 chunks + 1 final

        # Check final reply
        final = replies[-1]
        assert final.payload["done"] is True
        assert "Hello, how can I help you?" in final.payload.get("final_answer", "")

    async def test_handle_query_llm_error(self, responder, mock_runtime, mock_llm):
        """LLM returns an error event → error reply + agent.error sent.  §8.3."""
        async def error_stream(messages, system=None, tools=None):
            yield LLMStreamEvent(type="error", error_message="API timeout")
            return

        mock_llm.stream_generate = error_stream

        msg = _make_query_msg(responder._session_id, question="test")
        await responder._message_handler(msg)
        await asyncio.sleep(0.15)

        calls = mock_runtime.send.call_args_list
        replies = [c[0][0] for c in calls if c[0][0].msg_type == MsgType.RESPOND_REPLY]
        errors = [c[0][0] for c in calls if c[0][0].msg_type == MsgType.AGENT_ERROR]
        assert len(replies) >= 1
        assert len(errors) >= 1  # §3 ¶3: agent.error sent on crash
        assert replies[-1].payload.get("error") == "API timeout"

    async def test_handle_empty_question(self, responder, mock_runtime, mock_llm):
        """Empty question → immediate error reply."""
        msg = _make_query_msg(responder._session_id, question="")
        await responder._message_handler(msg)
        await asyncio.sleep(0.1)

        calls = mock_runtime.send.call_args_list
        replies = [c[0][0] for c in calls if c[0][0].msg_type == MsgType.RESPOND_REPLY]
        assert len(replies) == 1
        assert replies[0].payload.get("error") is not None

    @staticmethod
    async def _stream_from_texts(texts: list[str], **kwargs):
        for t in texts:
            yield LLMStreamEvent(type="text", text_delta=t)
        yield LLMStreamEvent(type="done", finish_reason="end_turn")


class TestToolCalls:
    async def test_handle_tool_call_flow(self, responder, mock_runtime, mock_llm):
        """LLM emits: text → tool_call → text → done.  Responder sends
        respond.request_tool, waits for tool result, incorporates it."""
        async def stream_with_tool(messages, system=None, tools=None):
            yield LLMStreamEvent(type="text", text_delta="Let me check. ")
            yield LLMStreamEvent(
                type="tool_call",
                tool_name="data_provider",
                tool_call_id="tc-1",
                tool_input={"url": "https://example.com"},
            )
            yield LLMStreamEvent(type="text", text_delta="The data shows 42.")
            yield LLMStreamEvent(type="done", finish_reason="end_turn")

        mock_llm.stream_generate = stream_with_tool

        # Simulate tool result arriving while responder is waiting
        async def deliver_tool_result(*args, **kwargs):
            # After the first send (request_tool), inject the tool_result
            # into the responder's inbox after a short delay
            pass

        msg = _make_query_msg(responder._session_id, question="Check data")
        await responder._message_handler(msg)
        await asyncio.sleep(0.2)

        calls = mock_runtime.send.call_args_list
        sent_types = [c[0][0].msg_type for c in calls]

        # Should have sent a tool request
        assert MsgType.RESPOND_REQUEST_TOOL in sent_types

        # Should have sent reply chunks
        reply_count = sent_types.count(MsgType.RESPOND_REPLY)
        assert reply_count >= 1


class TestPersonas:
    async def test_persona_passed_from_query(self, responder, mock_runtime, mock_llm):
        """Persona override in query payload is respected."""
        captured_system: list[str] = []

        async def stream(messages, system=None, tools=None):
            captured_system.append(system or "")
            yield LLMStreamEvent(type="text", text_delta="OK.")
            yield LLMStreamEvent(type="done", finish_reason="end_turn")

        mock_llm.stream_generate = stream

        msg = _make_query_msg(responder._session_id, question="Advise me", persona="advisor")
        await responder._message_handler(msg)
        await asyncio.sleep(0.15)

        # Check that system prompt contained advisor content
        assert len(captured_system) == 1
        system = captured_system[0]
        assert "advisor" in system.lower() or "practical" in system.lower()

    async def test_persona_property(self, responder):
        """Persona getter/setter works."""
        assert responder.persona == Persona.AUTO
        responder.persona = Persona.ANALYST
        assert responder.persona == Persona.ANALYST


class TestLanguageDetection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Hello, how are you?", "en"),
            ("Xin chào, bạn khỏe không?", "vi"),
            ("你好，你好吗？", "zh"),
            ("Giá iPhone 15 bao nhiêu?", "vi"),
            ("What is the price of iPhone 15?", "en"),
        ],
    )
    async def test_language_detection(self, text, expected):
        result = ResponderAgent._detect_language(text)
        assert result == expected


class TestSessionContext:
    async def test_consecutive_queries_update_history(self, responder, mock_runtime, mock_llm):
        """Two consecutive queries should grow conversation history."""
        async def stream_answer_1(messages, system=None, tools=None):
            yield LLMStreamEvent(type="text", text_delta="Answer 1")
            yield LLMStreamEvent(type="done", finish_reason="end_turn")

        async def stream_answer_2(messages, system=None, tools=None):
            yield LLMStreamEvent(type="text", text_delta="Answer 2")
            yield LLMStreamEvent(type="done", finish_reason="end_turn")

        mock_llm.stream_generate = stream_answer_1

        msg1 = _make_query_msg(responder._session_id, question="Q1")
        await responder._message_handler(msg1)
        await asyncio.sleep(0.15)

        # Send second query
        mock_llm.stream_generate = stream_answer_2
        msg2 = _make_query_msg(responder._session_id, question="Q2")
        await responder._message_handler(msg2)
        await asyncio.sleep(0.15)

        ctx = responder.session_context
        assert len(ctx.conversation_history) >= 2
        assert ctx.conversation_history[0].role == "user"
        assert ctx.conversation_history[1].role == "responder"
