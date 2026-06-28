"""Integration tests for AsyncioRuntime — spawn, send, interrupt, kill, shutdown."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from llend_harness.runtime.asyncio_runtime import AsyncioRuntime
from llend_harness.runtime.lifecycle import AgentState, AgentType
from llend_harness.runtime.message import Message, MsgType


@pytest.fixture
def runtime():
    """Return a fresh, un-started runtime."""
    return AsyncioRuntime(heartbeat_interval=0.05, checkpoint_ttl=86400)


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
    async def test_send_delivers_to_recipient(self, runtime_with_orch):
        runtime = runtime_with_orch
        eid = await runtime.spawn(AgentType.EXECUTOR.value, {})

        msg = await _make_msg(runtime, MsgType.TASK_DISPATCH, "executor", eid)
        await runtime.send(msg)

        # Message should be in the executor's queue
        handle = runtime._get_handle(eid)
        received = await asyncio.wait_for(handle.queue.get(), timeout=1)
        assert received.id == msg.id
        assert received.msg_type == MsgType.TASK_DISPATCH

    async def test_send_to_orchestrator_by_role(self, runtime_with_orch):
        """Sending with recipient='orchestrator' routes to the orchestrator."""
        runtime = runtime_with_orch
        msg = await _make_msg(runtime, MsgType.TASK_RESULT, "orchestrator")
        await runtime.send(msg)

        # Should be in orchestrator's queue
        orch_handle = runtime._get_handle(runtime._orchestrator_id)
        received = await asyncio.wait_for(orch_handle.queue.get(), timeout=1)
        assert received.msg_type == MsgType.TASK_RESULT

    async def test_send_expired_message_sends_error(self, runtime_with_orch):
        runtime = runtime_with_orch

        eid = await runtime.spawn(AgentType.EXECUTOR.value, {})
        msg = await _make_msg(runtime, MsgType.TASK_DISPATCH, "executor", eid)

        # Force expiry
        object.__setattr__(msg, "expires_at", datetime.now(UTC) - timedelta(seconds=1))
        await runtime.send(msg)

        # Sender (orchestrator) should get an AGENT_ERROR
        orch_handle = runtime._get_handle(runtime._orchestrator_id)
        error_msg = await asyncio.wait_for(orch_handle.queue.get(), timeout=1)
        assert error_msg.msg_type == MsgType.AGENT_ERROR

    async def test_send_to_unknown_recipient_no_crash(self, runtime_with_orch):
        """Sending to an unknown recipient is dropped gracefully."""
        runtime = runtime_with_orch
        msg = await _make_msg(runtime, MsgType.TASK_DISPATCH, "nonexistent")
        # Should not raise
        await runtime.send(msg)


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
        """Killing a non-existent agent should be a no-op."""
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
        await asyncio.sleep(0.1)
        assert runtime.get_handle_state(eid) == AgentState.INTERRUPT

        # Resolve via external callback
        await runtime.resolve_interrupt(eid, "yes", "go ahead")

        decision = await asyncio.wait_for(task, timeout=2)
        assert decision == "yes"
        assert runtime.get_handle_state(eid) == AgentState.RUNNING

        # Agent should have received INTERRUPT_RESPONSE in its inbox
        handle = runtime._get_handle(eid)
        resp = await asyncio.wait_for(handle.queue.get(), timeout=1)
        assert resp.msg_type == MsgType.INTERRUPT_RESPONSE
        assert resp.payload["decision"] == "yes"

    async def test_resolve_interrupt_no_pending(self, runtime_with_orch):
        """resolve_interrupt on an agent with no pending interrupt raises."""
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
# Heartbeat
# ---------------------------------------------------------------------------


class TestHeartbeat:
    async def test_heartbeat_sent(self, runtime):
        """After spawn, heartbeats are delivered to the orchestrator."""
        # Short heartbeat interval for test
        runtime._heartbeat_interval = 0.05
        oid = await runtime.spawn(AgentType.ORCHESTRATOR.value, {})
        eid = await runtime.spawn(AgentType.EXECUTOR.value, {})

        # Wait for at least one heartbeat cycle
        await asyncio.sleep(0.15)

        orch_handle = runtime._get_handle(oid)
        # Drain queue — should contain at least one heartbeat
        heartbeats = []
        while not orch_handle.queue.empty():
            try:
                msg = orch_handle.queue.get_nowait()
                if msg.msg_type == MsgType.AGENT_HEARTBEAT:
                    heartbeats.append(msg)
            except asyncio.QueueEmpty:
                break

        assert len(heartbeats) >= 1
        # Any heartbeat from the executor is sufficient
        exec_hb = [h for h in heartbeats if h.sender_instance == eid]
        assert len(exec_hb) >= 1
        assert exec_hb[0].payload == {}


# ---------------------------------------------------------------------------
# Integration: multiple agents message routing
# ---------------------------------------------------------------------------


class TestIntegration:
    async def test_multiple_executors_receive_own_messages(self, runtime_with_orch):
        runtime = runtime_with_orch
        e1 = await runtime.spawn(AgentType.EXECUTOR.value, {"task": "scrape-ebay"})
        e2 = await runtime.spawn(AgentType.EXECUTOR.value, {"task": "scrape-amazon"})

        m1 = await _make_msg(runtime, MsgType.TASK_DISPATCH, "executor", e1)
        m2 = await _make_msg(runtime, MsgType.TASK_DISPATCH, "executor", e2)

        await runtime.send(m1)
        await runtime.send(m2)

        h1 = runtime._get_handle(e1)
        h2 = runtime._get_handle(e2)

        r1 = await asyncio.wait_for(h1.queue.get(), timeout=1)
        r2 = await asyncio.wait_for(h2.queue.get(), timeout=1)

        assert r1.id == m1.id
        assert r2.id == m2.id

    async def test_parallel_spawn_and_send(self, runtime_with_orch):
        """Spawn multiple agents concurrently and send to each."""
        runtime = runtime_with_orch

        async def spawn_and_send(task_name):
            eid = await runtime.spawn(AgentType.EXECUTOR.value, {"task": task_name})
            msg = await _make_msg(runtime, MsgType.TASK_DISPATCH, "executor", eid)
            await runtime.send(msg)
            return eid

        ids = await asyncio.gather(spawn_and_send("a"), spawn_and_send("b"), spawn_and_send("c"))
        assert len(ids) == 3
        assert runtime.agent_count == 4  # 3 executors + 1 orchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_msg(
    runtime: AsyncioRuntime,
    msg_type: MsgType,
    recipient: str,
    recipient_instance: str | None = None,
) -> Message:
    orch_id = runtime._orchestrator_id or "orchestrator-1"
    return Message(
        session_id=runtime.session_id or uuid.uuid4(),
        sender=AgentType.ORCHESTRATOR.value,
        sender_instance=orch_id,
        recipient=recipient,
        recipient_instance=recipient_instance,
        msg_type=msg_type,
        payload={},
    )
