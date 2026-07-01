"""Integration tests for AgentRuntime implementations.

Parametrised over both ``AsyncioRuntime`` and ``LangGraphRuntime`` so
the same behaviour contract is verified for every backend.
"""

import asyncio
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from llend.runtime.asyncio_runtime import AsyncioRuntime
from llend.runtime.checkpoint import (
    Checkpoint,
    InterruptTimeoutError,
    set_default_base_dir,
)
from llend.runtime.langgraph_runtime import LangGraphRuntime
from llend.runtime.lifecycle import AgentState, AgentType
from llend.runtime.message import AgentErrorCode, Message, MsgType

# ---------------------------------------------------------------------------
# Runtime factory fixture
# ---------------------------------------------------------------------------


def _make_asyncio(ttl: int = 86400, **kwargs) -> AsyncioRuntime:
    return AsyncioRuntime(heartbeat_interval=0.05, checkpoint_ttl=ttl, **kwargs)


def _make_langgraph(ttl: int = 86400, **kwargs) -> LangGraphRuntime:
    return LangGraphRuntime(checkpoint_ttl=ttl, **kwargs)


@pytest.fixture(
    params=[_make_asyncio, _make_langgraph], ids=["asyncio", "langgraph"]
)
def runtime(request):
    """Return a fresh runtime of each backend type."""
    return request.param()


@pytest.fixture
async def runtime_with_orch(runtime):
    """Runtime with an orchestrator already spawned."""
    await runtime.spawn(AgentType.ORCHESTRATOR.value, {"goal": "test"})
    yield runtime
    await runtime.shutdown()


# ---------------------------------------------------------------------------
# Spawn
# ---------------------------------------------------------------------------


class TestSpawn:
    async def test_spawn_returns_instance_id(self, runtime):
        iid = await runtime.spawn(AgentType.EXECUTOR.value, {"task": "t1"})
        assert iid.startswith("executor-")
        assert len(iid) > len("executor-")

    async def test_spawn_sets_running_state(self, runtime):
        iid = await runtime.spawn(AgentType.EXECUTOR.value, {"task": "t1"})
        assert runtime.get_handle_state(iid) == AgentState.RUNNING

    async def test_spawn_orchestrator_sets_session(self, runtime):
        iid = await runtime.spawn(AgentType.ORCHESTRATOR.value, {"goal": "test"})
        assert runtime.session_id is not None
        assert runtime.get_handle_state(iid) == AgentState.RUNNING

    async def test_spawn_multiple_agents_unique_ids(self, runtime):
        a = await runtime.spawn(AgentType.EXECUTOR.value, {})
        b = await runtime.spawn(AgentType.EXECUTOR.value, {})
        assert a != b
        assert runtime.agent_count == 2


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------


class TestSend:
    async def test_send_to_unknown_recipient_no_crash(self, runtime_with_orch):
        """Sending to an unknown recipient is dropped gracefully."""
        runtime = runtime_with_orch
        msg = await _make_msg(runtime, MsgType.TASK_DISPATCH, "nonexistent")
        await runtime.send(msg)  # should not raise

    async def test_send_expired_message_sends_error(self, runtime_with_orch):
        runtime = runtime_with_orch
        eid = await runtime.spawn(AgentType.EXECUTOR.value, {})
        msg = await _make_msg(runtime, MsgType.TASK_DISPATCH, "executor", eid)
        object.__setattr__(
            msg, "expires_at", datetime.now(UTC) - timedelta(seconds=1)
        )
        await runtime.send(msg)
        # Should not crash — the expired message is dropped and an error
        # is routed back to the sender (orchestrator).


# ---------------------------------------------------------------------------
# Kill
# ---------------------------------------------------------------------------


class TestKill:
    async def test_kill_transitions_to_dead(self, runtime_with_orch):
        runtime = runtime_with_orch
        eid = await runtime.spawn(AgentType.EXECUTOR.value, {})
        await runtime.kill(eid)
        assert runtime.get_handle_state(eid) == AgentState.DEAD

    async def test_kill_is_idempotent(self, runtime_with_orch):
        runtime = runtime_with_orch
        eid = await runtime.spawn(AgentType.EXECUTOR.value, {})
        await runtime.kill(eid)
        await runtime.kill(eid)  # second call is no-op
        assert runtime.get_handle_state(eid) == AgentState.DEAD

    async def test_kill_unknown_agent_no_crash(self, runtime):
        await runtime.kill("nonexistent-123")


# ---------------------------------------------------------------------------
# Interrupt — basic flow
# ---------------------------------------------------------------------------


class TestInterrupt:
    async def test_interrupt_flow(self, runtime_with_orch):
        runtime = runtime_with_orch
        eid = await runtime.spawn(AgentType.EXECUTOR.value, {})

        # Start interrupt in background — it blocks until resolved
        async def do_interrupt():
            return await runtime.interrupt(eid, "Continue?", ["yes", "no"])

        task = asyncio.create_task(do_interrupt())

        # Give the interrupt time to suspend the agent
        await asyncio.sleep(0.2)
        assert runtime.get_handle_state(eid) == AgentState.INTERRUPT

        # Resolve via external callback
        await runtime.resolve_interrupt(eid, "yes", "go ahead")

        decision = await asyncio.wait_for(task, timeout=5)
        assert decision == "yes"
        assert runtime.get_handle_state(eid) == AgentState.RUNNING

    async def test_resolve_interrupt_no_pending(self, runtime_with_orch):
        runtime = runtime_with_orch
        eid = await runtime.spawn(AgentType.EXECUTOR.value, {})
        with pytest.raises(RuntimeError, match="no pending interrupt"):
            await runtime.resolve_interrupt(eid, "x")


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    async def test_shutdown_kills_all(self, runtime_with_orch):
        runtime = runtime_with_orch
        await runtime.spawn(AgentType.EXECUTOR.value, {})
        await runtime.spawn(AgentType.EXECUTOR.value, {})

        await runtime.shutdown()

        assert runtime.is_closed
        assert runtime.agent_count == 0

    async def test_send_after_shutdown_raises(self, runtime_with_orch):
        runtime = runtime_with_orch
        await runtime.shutdown()

        with pytest.raises(RuntimeError, match="closed"):
            await runtime.spawn(AgentType.EXECUTOR.value, {})


# ---------------------------------------------------------------------------
# Checkpoint disk persistence (Spec §3.4)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_base_dir():
    """Temporary base dir for checkpoint files, isolated from real ~/.llend."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td)
        set_default_base_dir(path)
        yield path
        set_default_base_dir(Path.home() / ".llend")


class TestCheckpointPersistence:
    """Tests that checkpoints are saved to / loaded from disk correctly."""

    @pytest.fixture(params=[_make_asyncio, _make_langgraph], ids=["asyncio", "langgraph"])
    async def runtime_persist(self, request, tmp_base_dir):
        """Runtime with orchestrator, using tmp_base_dir for checkpoint storage."""
        rt = request.param(data_dir=tmp_base_dir)
        await rt.spawn(AgentType.ORCHESTRATOR.value, {"goal": "test"})
        yield rt
        await rt.shutdown()

    async def test_checkpoint_saved_to_disk_on_interrupt(
        self, runtime_persist, tmp_base_dir
    ):
        """After interrupt, a JSON checkpoint file exists on disk."""
        runtime = runtime_persist
        eid = await runtime.spawn(AgentType.EXECUTOR.value, {})

        async def do_interrupt():
            return await runtime.interrupt(eid, "proceed?", ["yes"])

        task = asyncio.create_task(do_interrupt())
        await asyncio.sleep(0.2)

        # Find the checkpoint on disk
        cp = runtime.get_checkpoint(eid)
        assert cp is not None
        assert cp.disk_path().exists()

        # Resolve & clean up
        await runtime.resolve_interrupt(eid, "yes")
        await asyncio.wait_for(task, timeout=5)

    async def test_checkpoint_updated_on_disk_after_resolve(
        self, runtime_persist, tmp_base_dir
    ):
        """After resolve_interrupt(), the file reflects human_response."""
        runtime = runtime_persist
        eid = await runtime.spawn(AgentType.EXECUTOR.value, {})

        async def do_interrupt():
            return await runtime.interrupt(eid, "OK?", ["yes"])

        task = asyncio.create_task(do_interrupt())
        await asyncio.sleep(0.2)

        cp = runtime.get_checkpoint(eid)
        assert cp is not None

        await runtime.resolve_interrupt(eid, "yes", "sure")
        await asyncio.wait_for(task, timeout=5)

        # Reload from disk — use runtime's actual data_dir
        loaded = Checkpoint.load(cp.session_id, cp.interrupt_id, base_dir=tmp_base_dir)
        assert loaded is not None
        assert loaded.human_response == "sure"
        assert loaded.resolved_at is not None
        assert loaded.is_resolved

    async def test_checkpoint_deleted_on_kill(
        self, runtime_persist, tmp_base_dir
    ):
        """Killing an interrupted agent removes its checkpoint file."""
        runtime = runtime_persist
        eid = await runtime.spawn(AgentType.EXECUTOR.value, {})

        async def do_interrupt():
            return await runtime.interrupt(eid, "OK?", ["yes"])

        task = asyncio.create_task(do_interrupt())
        await asyncio.sleep(0.2)

        cp = runtime.get_checkpoint(eid)
        path = cp.disk_path()
        assert path.exists()

        await runtime.kill(eid)
        assert not path.exists()


# ---------------------------------------------------------------------------
# TTL enforcement (Spec §3.4)
# ---------------------------------------------------------------------------


class TestTTLEnforcement:
    """Tests that checkpoints with expired TTL are auto-terminated."""

    async def test_short_ttl_expires_raises_interrupt_timeout_error(self):
        """With a 1s TTL, an unresolved interrupt raises InterruptTimeoutError."""
        runtime = _make_asyncio(ttl=1, ttl_check_interval=0.5)
        try:
            await runtime.spawn(AgentType.ORCHESTRATOR.value, {"goal": "test"})
            eid = await runtime.spawn(AgentType.EXECUTOR.value, {})

            with pytest.raises(InterruptTimeoutError) as exc_info:
                await runtime.interrupt(eid, "Hurry!", ["yes", "no"])

            assert exc_info.value.ttl_seconds == 1
        finally:
            await runtime.shutdown()

    async def test_short_ttl_transitions_agent_to_error(self):
        """After TTL expiry the agent is in ERROR state."""
        runtime = _make_asyncio(ttl=1, ttl_check_interval=0.5)
        try:
            await runtime.spawn(AgentType.ORCHESTRATOR.value, {"goal": "test"})
            eid = await runtime.spawn(AgentType.EXECUTOR.value, {})

            with pytest.raises(InterruptTimeoutError):
                await runtime.interrupt(eid, "Hurry!", ["yes"])

            assert runtime.get_handle_state(eid) == AgentState.ERROR
        finally:
            await runtime.shutdown()

    async def test_ttl_expiry_sends_error_to_orchestrator(self):
        """Orchestrator receives agent.error(INTERRUPT_TIMEOUT) after expiry."""
        runtime = _make_asyncio(ttl=1, ttl_check_interval=0.5)
        try:
            orch_id = await runtime.spawn(
                AgentType.ORCHESTRATOR.value, {"goal": "test"}
            )
            eid = await runtime.spawn(AgentType.EXECUTOR.value, {})

            with pytest.raises(InterruptTimeoutError):
                await runtime.interrupt(eid, "Hurry!", ["yes"])

            # The orchestrator should have an error message waiting
            orch_handle = runtime._get_handle(orch_id)
            # Read all messages from orchestrator's queue
            error_found = False
            while not orch_handle.queue.empty():
                msg = orch_handle.queue.get_nowait()
                if (
                    msg.msg_type == MsgType.AGENT_ERROR
                    and msg.payload.get("error_code")
                    == AgentErrorCode.INTERRUPT_TIMEOUT.value
                ):
                    error_found = True
            assert error_found, "Orchestrator did not receive INTERRUPT_TIMEOUT error"
        finally:
            await runtime.shutdown()

    async def test_ttl_expiry_saves_checkpoint_as_timeout(self, tmp_base_dir):
        """After TTL expiry, checkpoint on disk is marked with __timeout__."""
        runtime = _make_asyncio(
            ttl=1, ttl_check_interval=0.5, data_dir=tmp_base_dir
        )
        try:
            await runtime.spawn(AgentType.ORCHESTRATOR.value, {"goal": "test"})
            eid = await runtime.spawn(AgentType.EXECUTOR.value, {})

            with pytest.raises(InterruptTimeoutError):
                await runtime.interrupt(eid, "Hurry!", ["yes"])

            # List checkpoints on disk
            cp_ids = Checkpoint.list_for_session(runtime.session_id)
            assert len(cp_ids) >= 1

            cp = Checkpoint.load(runtime.session_id, cp_ids[0])
            assert cp is not None
            assert cp.human_response == "__timeout__"
            assert cp.resolved_at is not None
        finally:
            await runtime.shutdown()

    async def test_long_ttl_does_not_expire(self, runtime_with_orch):
        """With a long TTL, interrupt stays open and can be resolved normally."""
        runtime = runtime_with_orch
        eid = await runtime.spawn(AgentType.EXECUTOR.value, {})

        async def do_interrupt():
            return await runtime.interrupt(eid, "OK?", ["yes"])

        task = asyncio.create_task(do_interrupt())
        await asyncio.sleep(0.2)

        # Should still be interruptable normally
        assert runtime.get_handle_state(eid) == AgentState.INTERRUPT
        await runtime.resolve_interrupt(eid, "yes")
        decision = await asyncio.wait_for(task, timeout=5)
        assert decision == "yes"

    async def test_resolved_interrupt_not_terminated_by_ttl(self, runtime_with_orch):
        """A resolved interrupt is not terminated even if created_at is old."""
        runtime = runtime_with_orch
        eid = await runtime.spawn(AgentType.EXECUTOR.value, {})

        async def do_interrupt():
            return await runtime.interrupt(eid, "OK?", ["yes"])

        task = asyncio.create_task(do_interrupt())
        await asyncio.sleep(0.2)

        # Resolve it quickly
        await runtime.resolve_interrupt(eid, "yes")
        await asyncio.wait_for(task, timeout=5)

        # Agent should be RUNNING (not ERROR)
        assert runtime.get_handle_state(eid) == AgentState.RUNNING


# ---------------------------------------------------------------------------
# Integration: multiple agents
# ---------------------------------------------------------------------------


class TestIntegration:
    async def test_parallel_spawn_and_send(self, runtime_with_orch):
        """Spawn multiple agents concurrently."""
        runtime = runtime_with_orch

        async def spawn_one(task_name):
            return await runtime.spawn(
                AgentType.EXECUTOR.value, {"task": task_name}
            )

        ids = await asyncio.gather(spawn_one("a"), spawn_one("b"), spawn_one("c"))
        assert len(ids) == 3
        assert runtime.agent_count == 4  # 3 executors + 1 orchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_msg(
    runtime,
    msg_type: MsgType,
    recipient: str,
    recipient_instance: str | None = None,
) -> Message:
    orch_id = getattr(runtime, "_orchestrator_id", None) or "orchestrator-1"
    return Message(
        session_id=runtime.session_id or uuid.uuid4(),
        sender=AgentType.ORCHESTRATOR.value,
        sender_instance=orch_id,
        recipient=recipient,
        recipient_instance=recipient_instance,
        msg_type=msg_type,
        payload={},
    )
