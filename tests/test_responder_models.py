"""Tests for responder Pydantic models — context, persona, memory, stream.  Spec 003."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from llend.responder.context import ConversationTurn, SessionContext, TaskResultSummary
from llend.responder.memory import UserProfile
from llend.responder.persona import PERSONA_SYSTEM_PROMPTS, Persona, build_system_prompt
from llend.responder.stream import (
    make_error_reply,
    make_final_reply,
    make_reply_chunk,
    reassemble_chunks,
)
from llend.runtime.message import TaskStatus


# =============================================================================
# ConversationTurn §3.2
# =============================================================================


class TestConversationTurn:
    def test_create_user_turn(self) -> None:
        turn = ConversationTurn(role="user", content="Giá iPhone 15 bao nhiêu?")
        assert turn.role == "user"
        assert turn.content == "Giá iPhone 15 bao nhiêu?"
        assert isinstance(turn.timestamp, datetime)

    def test_create_responder_turn(self) -> None:
        turn = ConversationTurn(role="responder", content="Median price is $325.")
        assert turn.role == "responder"

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValueError):  # Pydantic validation
            ConversationTurn(role="invalid", content="test")  # type: ignore[arg-type]


# =============================================================================
# TaskResultSummary §3.3
# =============================================================================


class TestTaskResultSummary:
    def test_create_minimal(self) -> None:
        tid = __import__("uuid").uuid4()
        ts = TaskResultSummary(
            task_id=tid,
            skill_name="data_provider",
            status=TaskStatus.DONE,
        )
        assert ts.task_id == tid
        assert ts.skill_name == "data_provider"
        assert ts.status == TaskStatus.DONE
        assert ts.summary == ""
        assert ts.key_metrics == {}
        assert ts.artifact_paths == []

    def test_create_with_metrics(self) -> None:
        tid = __import__("uuid").uuid4()
        ts = TaskResultSummary(
            task_id=tid,
            skill_name="analyze_pricing",
            status=TaskStatus.DONE,
            summary="Found 500 listings, median $325.",
            key_metrics={"median": 325, "count": 500},
            artifact_paths=["output/report.xlsx"],
        )
        assert ts.key_metrics == {"median": 325, "count": 500}
        assert ts.artifact_paths == ["output/report.xlsx"]


# =============================================================================
# SessionContext §3.1
# =============================================================================


class TestSessionContext:
    def test_create_defaults(self) -> None:
        ctx = SessionContext(session_goal="Analyze iPhone 15 pricing")
        assert ctx.session_goal == "Analyze iPhone 15 pricing"
        assert ctx.conversation_history == []
        assert ctx.task_results == []
        assert ctx.active_task is None

    def test_add_conversation_turn(self) -> None:
        ctx = SessionContext(session_goal="test")
        ctx.add_turn("user", "Hello")
        ctx.add_turn("responder", "Hi there!")
        assert len(ctx.conversation_history) == 2
        assert ctx.conversation_history[0].role == "user"
        assert ctx.conversation_history[1].role == "responder"

    def test_trim_history_overflow(self) -> None:
        ctx = SessionContext(session_goal="test")
        # Add 52 turns — expect trimmed to 50
        for i in range(52):
            ctx.add_turn("user", f"message {i}")
        assert len(ctx.conversation_history) == 50
        # Oldest should be dropped: messages 0 and 1
        assert ctx.conversation_history[0].content == "message 2"

    def test_trim_history_custom_max(self) -> None:
        ctx = SessionContext(session_goal="test")
        for i in range(20):
            ctx.add_turn("user", f"msg {i}")
        ctx.trim_history(max_turns=10)
        assert len(ctx.conversation_history) == 10
        assert ctx.conversation_history[0].content == "msg 10"


# =============================================================================
# UserProfile §9.2 — persistence
# =============================================================================


class TestUserProfile:
    def test_create_defaults(self) -> None:
        p = UserProfile()
        assert p.preferred_platforms == []
        assert p.budget_conscious is False
        assert p.favorite_categories == []
        assert p.persona_preference == Persona.AUTO
        assert p.custom_notes == {}
        assert p.last_updated is not None  # §9.1

    def test_load_missing_file(self) -> None:
        p = UserProfile.load(Path("/nonexistent/profile_xyz.json"))
        assert p == UserProfile()  # defaults

    def test_roundtrip_json(self, tmp_path: Path) -> None:
        p = UserProfile(
            preferred_platforms=["ebay", "amazon"],
            budget_conscious=True,
            favorite_categories=["electronics"],
            persona_preference=Persona.ADVISOR,
            custom_notes={"avoid": "new sellers"},
        )
        saved = p.save(tmp_path / "user_profile.json")
        loaded = UserProfile.load(saved)
        assert loaded.preferred_platforms == ["ebay", "amazon"]
        assert loaded.budget_conscious is True
        assert loaded.favorite_categories == ["electronics"]
        assert loaded.persona_preference == Persona.ADVISOR
        assert loaded.custom_notes == {"avoid": "new sellers"}

    def test_load_corrupt_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "corrupt.json"
        bad.write_text("not valid json {{{")
        p = UserProfile.load(bad)
        assert p == UserProfile()  # fallback to defaults


# =============================================================================
# Persona §4
# =============================================================================


class TestPersona:
    def test_persona_values(self) -> None:
        assert Persona.AUTO == "auto"
        assert Persona.ANALYST == "analyst"
        assert Persona.ADVISOR == "advisor"
        assert Persona.FRIEND == "friend"

    def test_persona_prompts_exist(self) -> None:
        for persona in Persona:
            assert persona in PERSONA_SYSTEM_PROMPTS
            assert len(PERSONA_SYSTEM_PROMPTS[persona]) > 20


class TestBuildSystemPrompt:
    def test_basic_prompt(self) -> None:
        prompt = build_system_prompt(Persona.ANALYST)
        assert "data analyst" in prompt.lower()
        assert "respond in the same language" in prompt.lower()

    def test_with_user_profile(self) -> None:
        profile = UserProfile(
            preferred_platforms=["ebay"],
            budget_conscious=True,
            favorite_categories=["electronics"],
        )
        prompt = build_system_prompt(Persona.ADVISOR, user_profile=profile)
        assert "ebay" in prompt.lower()
        assert "budget-conscious" in prompt.lower()
        assert "electronics" in prompt

    def test_with_session_context(self) -> None:
        tid = __import__("uuid").uuid4()
        ctx = SessionContext(
            session_goal="Analyze iPhone pricing",
            task_results=[
                TaskResultSummary(
                    task_id=tid,
                    skill_name="data_provider",
                    status=TaskStatus.DONE,
                    summary="Crawled 500 listings",
                    key_metrics={"count": 500},
                ),
            ],
        )
        prompt = build_system_prompt(Persona.AUTO, session_context=ctx)
        assert "Analyze iPhone pricing" in prompt
        assert "data_provider" in prompt
        assert "count=500" in prompt

    def test_language_instruction_always_present(self) -> None:
        for persona in Persona:
            prompt = build_system_prompt(persona)
            assert "same language" in prompt.lower()

    def test_language_parameter_injected(self) -> None:
        prompt = build_system_prompt(Persona.AUTO, language="vi")
        assert "Vietnamese" in prompt
        prompt_en = build_system_prompt(Persona.AUTO, language="en")
        assert "English" in prompt_en


# =============================================================================
# Stream utilities §8
# =============================================================================


class TestStreamUtilities:
    def test_make_reply_chunk(self) -> None:
        payload = make_reply_chunk("q-1", 0, "Hello ")
        assert payload["query_id"] == "q-1"
        assert payload["chunk_index"] == 0
        assert payload["chunk_content"] == "Hello "
        assert payload["stream"] is True
        assert payload["done"] is False
        assert "final_answer" not in payload

    def test_make_reply_chunk_final(self) -> None:
        payload = make_reply_chunk("q-1", 2, "", done=True, final_answer="Hello World", confidence=0.95)
        assert payload["done"] is True
        assert payload["final_answer"] == "Hello World"
        assert payload["confidence"] == 0.95

    def test_make_final_reply(self) -> None:
        payload = make_final_reply("q-1", "Complete answer", confidence=0.85)
        assert payload["stream"] is False
        assert payload["done"] is True
        assert payload["answer"] == "Complete answer"
        assert payload["confidence"] == 0.85

    def test_make_final_reply_with_advice(self) -> None:
        payload = make_final_reply(
            "q-1", "Buy it", confidence=0.9,
            advice="Check seller rating first",
            follow_up_suggestions=["Compare with Amazon", "Look at refurbished options"],
        )
        assert payload["advice"] == "Check seller rating first"
        assert len(payload["follow_up_suggestions"]) == 2

    def test_make_error_reply(self) -> None:
        payload = make_error_reply("q-1", "LLM timeout")
        assert payload["error"] == "LLM timeout"
        assert payload["done"] is True
        assert payload["confidence"] == 0.0

    def test_reassemble_chunks(self) -> None:
        chunks = [
            {"chunk_index": 1, "chunk_content": " World"},
            {"chunk_index": 0, "chunk_content": "Hello"},
        ]
        result = reassemble_chunks(chunks)
        assert result == "Hello World"

    def test_reassemble_empty(self) -> None:
        assert reassemble_chunks([]) == ""
