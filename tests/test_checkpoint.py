"""Tests for the checkpoint persistence layer."""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from llend_harness.runtime.checkpoint import (
    Checkpoint,
    _base_path,
    _checkpoint_path,
    cleanup_expired_checkpoints,
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    """Redirect ``Path.home()`` to a temp directory for isolated test files."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def session_id():
    return uuid.uuid4()


@pytest.fixture
def fresh_checkpoint(session_id):
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
    def test_defaults(self, fresh_checkpoint):
        c = fresh_checkpoint
        assert c.ttl_seconds == 86400
        assert c.human_response is None
        assert c.resolved_at is None
        assert c.agent_state == "INTERRUPT"
        assert isinstance(c.reply_chain, list)
        assert c.reply_chain == []

    def test_not_expired(self, fresh_checkpoint):
        assert not fresh_checkpoint.is_expired

    def test_expired(self, session_id):
        c = Checkpoint(
            interrupt_id=uuid.uuid4(),
            session_id=session_id,
            agent_instance="e-1",
            agent_type="executor",
            interrupt_message="q",
            interrupt_options=["yes", "no"],
            created_at=datetime.now(UTC) - timedelta(seconds=10),
            ttl_seconds=5,
        )
        assert c.is_expired

    def test_not_resolved_by_default(self, fresh_checkpoint):
        assert not fresh_checkpoint.is_resolved

    def test_resolved_after_response(self, fresh_checkpoint):
        fresh_checkpoint.human_response = "B"
        fresh_checkpoint.resolved_at = datetime.now(UTC)
        assert fresh_checkpoint.is_resolved

    def test_age_seconds(self, session_id):
        c = Checkpoint(
            interrupt_id=uuid.uuid4(),
            session_id=session_id,
            agent_instance="e-1",
            agent_type="executor",
            interrupt_message="q",
            interrupt_options=["yes"],
            created_at=datetime.now(UTC) - timedelta(seconds=10),
        )
        assert c.age_seconds >= 10


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


class TestPaths:
    def test_base_path(self, tmp_home, session_id):
        expected = tmp_home / ".llend" / "sessions" / str(session_id) / "checkpoints"
        assert _base_path(session_id) == expected

    def test_checkpoint_path(self, tmp_home, session_id):
        intr_id = uuid.uuid4()
        expected = (
            tmp_home / ".llend" / "sessions" / str(session_id) / "checkpoints" / f"{intr_id}.json"
        )
        assert _checkpoint_path(intr_id, session_id) == expected


# ---------------------------------------------------------------------------
# Save / Load / Delete
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_home, fresh_checkpoint):
        path = save_checkpoint(fresh_checkpoint)
        assert path.exists()
        assert path.suffix == ".json"

        loaded = load_checkpoint(fresh_checkpoint.interrupt_id, fresh_checkpoint.session_id)
        assert loaded is not None
        assert loaded.interrupt_id == fresh_checkpoint.interrupt_id
        assert loaded.session_id == fresh_checkpoint.session_id
        assert loaded.agent_instance == fresh_checkpoint.agent_instance
        assert loaded.agent_type == fresh_checkpoint.agent_type
        assert loaded.interrupt_message == fresh_checkpoint.interrupt_message
        assert loaded.interrupt_options == fresh_checkpoint.interrupt_options
        assert loaded.task_context == fresh_checkpoint.task_context
        assert loaded.ttl_seconds == fresh_checkpoint.ttl_seconds

    def test_load_missing(self, tmp_home, session_id):
        assert load_checkpoint(uuid.uuid4(), session_id) is None

    def test_load_expired_returns_none(self, tmp_home, session_id):
        """Expired checkpoints are cleaned up on load and return None."""
        c = Checkpoint(
            interrupt_id=uuid.uuid4(),
            session_id=session_id,
            agent_instance="e-1",
            agent_type="executor",
            interrupt_message="q",
            interrupt_options=["a"],
            created_at=datetime.now(UTC) - timedelta(seconds=100),
            ttl_seconds=5,
        )
        path = save_checkpoint(c)
        assert path.exists()

        result = load_checkpoint(c.interrupt_id, session_id)
        assert result is None
        assert not path.exists()  # cleaned up

    def test_delete_existing(self, tmp_home, fresh_checkpoint):
        save_checkpoint(fresh_checkpoint)
        assert delete_checkpoint(fresh_checkpoint.interrupt_id, fresh_checkpoint.session_id)

    def test_delete_missing(self, tmp_home, session_id):
        assert not delete_checkpoint(uuid.uuid4(), session_id)

    def test_delete_removes_file(self, tmp_home, fresh_checkpoint):
        path = save_checkpoint(fresh_checkpoint)
        assert path.exists()
        delete_checkpoint(fresh_checkpoint.interrupt_id, fresh_checkpoint.session_id)
        assert not path.exists()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_removes_expired(self, tmp_home, session_id):
        """Only checkpoints older than max_age_seconds are removed."""
        # Fresh checkpoint
        fresh = Checkpoint(
            interrupt_id=uuid.uuid4(),
            session_id=session_id,
            agent_instance="e-1",
            agent_type="executor",
            interrupt_message="q",
            interrupt_options=["a"],
            ttl_seconds=86400,
        )
        save_checkpoint(fresh)

        # Old checkpoint (mock by writing directly with old mtime)
        old = Checkpoint(
            interrupt_id=uuid.uuid4(),
            session_id=session_id,
            agent_instance="e-2",
            agent_type="executor",
            interrupt_message="old q",
            interrupt_options=["b"],
            ttl_seconds=86400,
        )
        old_path = save_checkpoint(old)

        # Make it "old" by setting mtime far in the past
        old_ts = (datetime.now(UTC) - timedelta(days=2)).timestamp()
        old_path.touch()  # updates mtime to now (not what we want)
        import os

        os.utime(old_path, (old_ts, old_ts))

        removed = cleanup_expired_checkpoints(session_id, max_age_seconds=86400)
        assert removed == 1

    def test_cleanup_non_existent_session(self, session_id):
        assert cleanup_expired_checkpoints(session_id) == 0
