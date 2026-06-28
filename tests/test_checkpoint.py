"""Tests for the Checkpoint model."""

import uuid
from datetime import UTC, datetime, timedelta

from llend.runtime.checkpoint import Checkpoint

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _session_id():
    return uuid.uuid4()


def _fresh_checkpoint(session_id):
    return Checkpoint(
        interrupt_id=uuid.uuid4(),
        session_id=session_id,
        agent_instance="executor-task1-run1",
        agent_type="executor",
        task_context={"task_id": "task1", "skill": "data_provider"},
        interrupt_message="15k rows — analyze all or filter?",
        interrupt_options=["A: all 15k", "B: latest 1k", "C: sample 2k"],
    )


# ---------------------------------------------------------------------------
# Checkpoint model
# ---------------------------------------------------------------------------


class TestCheckpointModel:
    def test_defaults(self):
        c = _fresh_checkpoint(_session_id())
        assert c.ttl_seconds == 86400
        assert c.human_response is None
        assert c.resolved_at is None
        assert c.agent_state == "INTERRUPT"
        assert isinstance(c.reply_chain, list)
        assert c.reply_chain == []

    def test_not_expired(self):
        c = _fresh_checkpoint(_session_id())
        assert not c.is_expired

    def test_expired(self):
        c = Checkpoint(
            interrupt_id=uuid.uuid4(),
            session_id=_session_id(),
            agent_instance="e-1",
            agent_type="executor",
            interrupt_message="q",
            interrupt_options=["yes", "no"],
            created_at=datetime.now(UTC) - timedelta(seconds=10),
            ttl_seconds=5,
        )
        assert c.is_expired

    def test_not_resolved_by_default(self):
        c = _fresh_checkpoint(_session_id())
        assert not c.is_resolved

    def test_resolved_after_response(self):
        c = _fresh_checkpoint(_session_id())
        c.human_response = "B"
        c.resolved_at = datetime.now(UTC)
        assert c.is_resolved

    def test_age_seconds(self):
        c = Checkpoint(
            interrupt_id=uuid.uuid4(),
            session_id=_session_id(),
            agent_instance="e-1",
            agent_type="executor",
            interrupt_message="q",
            interrupt_options=["yes"],
            created_at=datetime.now(UTC) - timedelta(seconds=10),
        )
        assert c.age_seconds >= 10

    def test_roundtrip_json(self):
        """model_dump_json → model_validate_json preserves all fields."""
        c = _fresh_checkpoint(_session_id())
        json_str = c.model_dump_json()
        restored = Checkpoint.model_validate_json(json_str)
        assert restored.interrupt_id == c.interrupt_id
        assert restored.session_id == c.session_id
        assert restored.interrupt_message == c.interrupt_message
        assert restored.interrupt_options == c.interrupt_options
        assert restored.ttl_seconds == c.ttl_seconds
