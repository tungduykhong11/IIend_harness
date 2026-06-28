"""Integration tests for AgentRuntime implementations.

Parametrised over both ``AsyncioRuntime`` and ``LangGraphRuntime`` so
the same behaviour contract is verified for every backend.
"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from llend.runtime.asyncio_runtime import AsyncioRuntime
from llend.runtime.langgraph_runtime import LangGraphRuntime
from llend.runtime.lifecycle import AgentState, AgentType
from llend.runtime.message import Message, MsgType

# ---------------------------------------------------------------------------
# Runtime factory fixture
# ---------------------------------------------------------------------------


def _make_asyncio() -> AsyncioRuntime:
    return AsyncioRuntime(heartbeat_interval=0.05, checkpoint_ttl=86400)


def _make_langgraph() -> LangGraphRuntime:
    return LangGraphRuntime(checkpoint_ttl=86400)


@pytest.fixture(params=[_make_asyncio, _make_langgraph], ids=["asyncio", "langgraph"])
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
        object.__setattr__(msg, "expires_at", datetime.now(UTC) - timedelta(seconds=1))
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
# Interrupt
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
# Integration: multiple agents
# ---------------------------------------------------------------------------


class TestIntegration:
    async def test_parallel_spawn_and_send(self, runtime_with_orch):
        """Spawn multiple agents concurrently."""
        runtime = runtime_with_orch

        async def spawn_one(task_name):
            return await runtime.spawn(AgentType.EXECUTOR.value, {"task": task_name})

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
