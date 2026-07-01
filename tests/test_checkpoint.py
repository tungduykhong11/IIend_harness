"""Tests for the Checkpoint model — properties + disk persistence (§3.4)."""

import json
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from llend.runtime.checkpoint import (
    Checkpoint,
    InterruptTimeoutError,
    set_default_base_dir,
)


# ---------------------------------------------------------------------------
# Helpers
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
# Checkpoint model — core properties
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


# ---------------------------------------------------------------------------
# InterruptTimeoutError
# ---------------------------------------------------------------------------


class TestInterruptTimeoutError:
    def test_contains_interrupt_id_and_ttl(self):
        iid = uuid.uuid4()
        exc = InterruptTimeoutError(iid, 42)
        assert exc.interrupt_id == iid
        assert exc.ttl_seconds == 42
        assert str(iid) in str(exc)
        assert "42" in str(exc)

    def test_can_be_caught_like_normal_exception(self):
        with pytest.raises(InterruptTimeoutError):
            raise InterruptTimeoutError(uuid.uuid4(), 10)


# ---------------------------------------------------------------------------
# Disk persistence (Spec §3.4)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_base_dir():
    """Temporary base dir isolated from real ~/.llend."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        old = set_default_base_dir.__wrapped__ if hasattr(set_default_base_dir, "__wrapped__") else None  # no-op guard
        set_default_base_dir(path)
        yield path
        # Restore default
        set_default_base_dir(Path.home() / ".llend")


class TestDiskPersistence:
    """Tests for Checkpoint.save / .load / .delete / .list_for_session."""

    def test_save_creates_file(self, tmp_base_dir):
        sid = _session_id()
        c = _fresh_checkpoint(sid)
        path = c.save()
        assert path.exists()
        assert path.suffix == ".json"
        assert str(sid) in str(path)
        assert str(c.interrupt_id) in path.name

    def test_save_writes_valid_json(self, tmp_base_dir):
        c = _fresh_checkpoint(_session_id())
        path = c.save()
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["interrupt_message"] == c.interrupt_message
        assert raw["interrupt_options"] == c.interrupt_options

    def test_load_returns_identical_checkpoint(self, tmp_base_dir):
        sid = _session_id()
        c = _fresh_checkpoint(sid)
        c.save()
        loaded = Checkpoint.load(sid, c.interrupt_id)
        assert loaded is not None
        assert loaded.interrupt_id == c.interrupt_id
        assert loaded.session_id == sid
        assert loaded.interrupt_message == c.interrupt_message
        assert loaded.agent_instance == c.agent_instance
        assert loaded.ttl_seconds == c.ttl_seconds

    def test_load_nonexistent_returns_none(self, tmp_base_dir):
        loaded = Checkpoint.load(_session_id(), uuid.uuid4())
        assert loaded is None

    def test_delete_removes_file(self, tmp_base_dir):
        c = _fresh_checkpoint(_session_id())
        path = c.save()
        assert path.exists()
        c.delete()
        assert not path.exists()

    def test_delete_nonexistent_no_error(self, tmp_base_dir):
        c = _fresh_checkpoint(_session_id())
        # Delete without saving — should not crash
        c.delete()

    def test_save_overwrites_existing(self, tmp_base_dir):
        c = _fresh_checkpoint(_session_id())
        c.save()
        c.human_response = "updated"
        c.resolved_at = datetime.now(UTC)
        c.save()  # overwrite

        loaded = Checkpoint.load(c.session_id, c.interrupt_id)
        assert loaded is not None
        assert loaded.human_response == "updated"

    def test_list_for_session_returns_all_ids(self, tmp_base_dir):
        sid = _session_id()
        a = _fresh_checkpoint(sid)
        b = _fresh_checkpoint(sid)
        a.save()
        b.save()

        ids = Checkpoint.list_for_session(sid)
        assert len(ids) == 2
        assert a.interrupt_id in ids
        assert b.interrupt_id in ids

    def test_list_for_session_empty(self, tmp_base_dir):
        ids = Checkpoint.list_for_session(_session_id())
        assert ids == []

    def test_list_for_session_nonexistent_dir(self, tmp_base_dir):
        ids = Checkpoint.list_for_session(uuid.uuid4())
        assert ids == []

    def test_disk_path_uses_custom_base_dir(self, tmp_base_dir):
        c = _fresh_checkpoint(_session_id())
        custom = Path("/tmp/test-llend")
        path = c.disk_path(custom)
        assert str(custom) in str(path)
        assert str(c.session_id) in str(path)
        assert str(c.interrupt_id) in path.name
