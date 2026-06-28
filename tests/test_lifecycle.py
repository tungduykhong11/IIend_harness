"""Tests for the agent lifecycle state machine."""

from datetime import UTC

import pytest

from llend.runtime.lifecycle import (
    AgentInstance,
    AgentState,
    AgentType,
    is_alive,
    is_terminal,
    transition,
    validate_transition,
)


class TestValidTransitions:
    """Every legal edge in the state diagram."""

    def test_init_to_running(self):
        assert validate_transition(AgentState.INIT, AgentState.RUNNING)
        assert transition(AgentState.INIT, AgentState.RUNNING) == AgentState.RUNNING

    def test_running_to_interrupt(self):
        assert transition(AgentState.RUNNING, AgentState.INTERRUPT) == AgentState.INTERRUPT

    def test_running_to_complete(self):
        assert transition(AgentState.RUNNING, AgentState.COMPLETE) == AgentState.COMPLETE

    def test_running_to_error(self):
        assert transition(AgentState.RUNNING, AgentState.ERROR) == AgentState.ERROR

    def test_running_to_dead(self):
        assert transition(AgentState.RUNNING, AgentState.DEAD) == AgentState.DEAD

    def test_interrupt_to_running(self):
        assert transition(AgentState.INTERRUPT, AgentState.RUNNING) == AgentState.RUNNING

    def test_interrupt_to_error(self):
        assert transition(AgentState.INTERRUPT, AgentState.ERROR) == AgentState.ERROR

    def test_interrupt_to_dead(self):
        assert transition(AgentState.INTERRUPT, AgentState.DEAD) == AgentState.DEAD

    def test_complete_to_dead(self):
        assert transition(AgentState.COMPLETE, AgentState.DEAD) == AgentState.DEAD

    def test_error_to_dead(self):
        assert transition(AgentState.ERROR, AgentState.DEAD) == AgentState.DEAD


class TestInvalidTransitions:
    """Every cross-edge NOT in the state table raises ValueError."""

    @pytest.mark.parametrize(
        "current, target",
        [
            (AgentState.INIT, AgentState.INTERRUPT),
            (AgentState.INIT, AgentState.COMPLETE),
            (AgentState.INIT, AgentState.ERROR),
            (AgentState.INIT, AgentState.DEAD),
            (AgentState.RUNNING, AgentState.INIT),
            (AgentState.INTERRUPT, AgentState.INIT),
            (AgentState.INTERRUPT, AgentState.COMPLETE),
            (AgentState.COMPLETE, AgentState.INIT),
            (AgentState.COMPLETE, AgentState.RUNNING),
            (AgentState.COMPLETE, AgentState.INTERRUPT),
            (AgentState.COMPLETE, AgentState.ERROR),
            (AgentState.ERROR, AgentState.INIT),
            (AgentState.ERROR, AgentState.RUNNING),
            (AgentState.ERROR, AgentState.INTERRUPT),
            (AgentState.ERROR, AgentState.COMPLETE),
            (AgentState.DEAD, AgentState.INIT),
            (AgentState.DEAD, AgentState.RUNNING),
            (AgentState.DEAD, AgentState.INTERRUPT),
            (AgentState.DEAD, AgentState.COMPLETE),
            (AgentState.DEAD, AgentState.ERROR),
        ],
    )
    def test_invalid_transition_raises(self, current, target):
        assert not validate_transition(current, target)
        with pytest.raises(ValueError):
            transition(current, target)


class TestPredicates:
    def test_dead_is_terminal(self):
        assert is_terminal(AgentState.DEAD)

    def test_non_dead_not_terminal(self):
        for state in AgentState:
            if state != AgentState.DEAD:
                assert not is_terminal(state)

    def test_alive_states(self):
        assert is_alive(AgentState.INIT)
        assert is_alive(AgentState.RUNNING)
        assert is_alive(AgentState.INTERRUPT)

    def test_not_alive_states(self):
        assert not is_alive(AgentState.COMPLETE)
        assert not is_alive(AgentState.ERROR)
        assert not is_alive(AgentState.DEAD)


class TestAgentInstance:
    def test_default_state_is_init(self):
        inst = AgentInstance(instance_id="test-1", agent_type=AgentType.EXECUTOR)
        assert inst.state == AgentState.INIT
        assert inst.stopped_at is None

    def test_spawned_at_is_set(self):
        from datetime import datetime

        inst = AgentInstance(instance_id="test-2", agent_type=AgentType.ORCHESTRATOR)
        assert isinstance(inst.spawned_at, datetime)
        assert inst.spawned_at.tzinfo == UTC
