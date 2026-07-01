"""Tests for notification channels (Spec §3.4)."""

import uuid
from datetime import UTC, datetime

import pytest

from llend.runtime.checkpoint import Checkpoint
from llend.runtime.notifications import (
    ConsoleNotificationChannel,
    MultiChannel,
    NotificationChannel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _checkpoint(**overrides) -> Checkpoint:
    defaults = dict(
        interrupt_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        agent_instance="executor-task1-run1",
        agent_type="executor",
        interrupt_message="Continue?",
        interrupt_options=["yes", "no"],
    )
    defaults.update(overrides)
    return Checkpoint(**defaults)


# ---------------------------------------------------------------------------
# Stub channel for testing MultiChannel
# ---------------------------------------------------------------------------


class _RecordingChannel(NotificationChannel):
    """Records all calls for test inspection."""

    def __init__(self):
        self.interrupt_calls: list[Checkpoint] = []
        self.timeout_calls: list[Checkpoint] = []
        self._should_fail_on_interrupt = False
        self._should_fail_on_timeout = False

    async def notify_interrupt(self, checkpoint: Checkpoint) -> None:
        if self._should_fail_on_interrupt:
            raise RuntimeError("simulated failure")
        self.interrupt_calls.append(checkpoint)

    async def notify_interrupt_timeout(self, checkpoint: Checkpoint) -> None:
        if self._should_fail_on_timeout:
            raise RuntimeError("simulated failure")
        self.timeout_calls.append(checkpoint)


# ---------------------------------------------------------------------------
# ConsoleNotificationChannel
# ---------------------------------------------------------------------------


class TestConsoleNotificationChannel:
    async def test_notify_interrupt_prints_checkpoint_info(self, capsys):
        cp = _checkpoint(
            interrupt_message="15k rows — what to do?",
            interrupt_options=["A: all", "B: sample"],
        )
        ch = ConsoleNotificationChannel()
        await ch.notify_interrupt(cp)

        captured = capsys.readouterr().out
        assert "INTERRUPT" in captured
        assert "15k rows — what to do?" in captured
        assert "A: all" in captured
        assert "B: sample" in captured
        assert cp.agent_instance in captured

    async def test_notify_interrupt_timeout_prints_warning(self, capsys):
        cp = _checkpoint(ttl_seconds=60)
        ch = ConsoleNotificationChannel()
        await ch.notify_interrupt_timeout(cp)

        captured = capsys.readouterr().out
        assert "INTERRUPT TIMEOUT" in captured
        assert str(cp.interrupt_id) in captured
        assert "60" in captured


# ---------------------------------------------------------------------------
# MultiChannel
# ---------------------------------------------------------------------------


class TestMultiChannel:
    async def test_fans_out_to_all_channels(self):
        a = _RecordingChannel()
        b = _RecordingChannel()
        multi = MultiChannel(a, b)

        cp = _checkpoint()
        await multi.notify_interrupt(cp)

        assert len(a.interrupt_calls) == 1
        assert len(b.interrupt_calls) == 1
        assert a.interrupt_calls[0].interrupt_id == cp.interrupt_id

    async def test_notify_interrupt_timeout_fans_out(self):
        a = _RecordingChannel()
        b = _RecordingChannel()
        multi = MultiChannel(a, b)

        cp = _checkpoint()
        await multi.notify_interrupt_timeout(cp)

        assert len(a.timeout_calls) == 1
        assert len(b.timeout_calls) == 1

    async def test_one_channel_fails_others_still_called(self):
        a = _RecordingChannel()
        a._should_fail_on_interrupt = True
        b = _RecordingChannel()
        multi = MultiChannel(a, b)

        cp = _checkpoint()
        await multi.notify_interrupt(cp)  # should not raise

        assert len(a.interrupt_calls) == 0  # failed
        assert len(b.interrupt_calls) == 1  # still called

    async def test_one_channel_fails_timeout_others_still_called(self):
        a = _RecordingChannel()
        a._should_fail_on_timeout = True
        b = _RecordingChannel()
        multi = MultiChannel(a, b)

        cp = _checkpoint()
        await multi.notify_interrupt_timeout(cp)  # should not raise

        assert len(a.timeout_calls) == 0
        assert len(b.timeout_calls) == 1

    async def test_empty_multi_channel_no_crash(self):
        multi = MultiChannel()
        cp = _checkpoint()
        await multi.notify_interrupt(cp)  # should not raise
        await multi.notify_interrupt_timeout(cp)  # should not raise
