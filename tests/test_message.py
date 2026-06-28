"""Tests for the Message Protocol — envelope, enums, and auxiliary models."""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from llend_harness.runtime.message import (
    AgentErrorCode,
    Artifact,
    Message,
    MsgType,
    ReviewIssue,
    TaskStatus,
    Verdict,
)

# ---------------------------------------------------------------------------
# Message creation
# ---------------------------------------------------------------------------


def test_message_create_minimal():
    """Message with only required fields fills defaults correctly."""
    msg = Message(
        session_id=uuid4(),
        sender="orchestrator",
        sender_instance="orchestrator-1",
        recipient="executor",
        msg_type=MsgType.TASK_DISPATCH,
    )
    assert isinstance(msg.id, UUID)
    assert msg.recipient_instance is None
    assert msg.payload == {}
    assert msg.parent_id is None
    assert isinstance(msg.created_at, datetime)
    assert msg.expires_at is None


def test_message_create_full():
    """All fields set explicitly round-trip correctly."""
    sid = uuid4()
    mid = uuid4()
    parent = uuid4()
    now = datetime.now(UTC)
    expires = now + timedelta(minutes=5)

    msg = Message(
        id=mid,
        session_id=sid,
        sender="executor",
        sender_instance="executor-task1-run1",
        recipient="orchestrator",
        recipient_instance="orchestrator-1",
        msg_type=MsgType.TASK_RESULT,
        payload={"task_id": "t1", "status": "done"},
        parent_id=parent,
        created_at=now,
        expires_at=expires,
    )
    assert msg.id == mid
    assert msg.session_id == sid
    assert msg.sender == "executor"
    assert msg.sender_instance == "executor-task1-run1"
    assert msg.recipient == "orchestrator"
    assert msg.recipient_instance == "orchestrator-1"
    assert msg.msg_type == MsgType.TASK_RESULT
    assert msg.payload == {"task_id": "t1", "status": "done"}
    assert msg.parent_id == parent
    assert msg.expires_at == expires


# ---------------------------------------------------------------------------
# Message serialization
# ---------------------------------------------------------------------------


def test_message_roundtrip_json():
    """``model_dump_json`` → ``model_validate_json`` preserves all fields."""
    msg = Message(
        session_id=uuid4(),
        sender="reviewer",
        sender_instance="reviewer-task2-run1",
        recipient="orchestrator",
        msg_type=MsgType.TASK_VERDICT,
        payload={"verdict": "pass", "confidence": 0.95},
    )
    json_str = msg.model_dump_json()
    restored = Message.model_validate_json(json_str)

    assert restored.id == msg.id
    assert restored.session_id == msg.session_id
    assert restored.sender == msg.sender
    assert restored.sender_instance == msg.sender_instance
    assert restored.msg_type == msg.msg_type
    assert restored.payload == msg.payload
    # UUIDs and datetimes survive JSON roundtrip via Pydantic
    assert isinstance(restored.id, UUID)
    assert isinstance(restored.created_at, datetime)


# ---------------------------------------------------------------------------
# Message expiry
# ---------------------------------------------------------------------------


def test_message_not_expired_no_expiry():
    """A message without ``expires_at`` is never expired."""
    msg = Message(
        session_id=uuid4(),
        sender="orchestrator",
        sender_instance="orchestrator-1",
        recipient="executor",
        msg_type=MsgType.TASK_DISPATCH,
    )
    assert not msg.is_expired


def test_message_not_expired_future():
    """A message with a future ``expires_at`` is not expired."""
    msg = Message(
        session_id=uuid4(),
        sender="orchestrator",
        sender_instance="orchestrator-1",
        recipient="executor",
        msg_type=MsgType.TASK_DISPATCH,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    assert not msg.is_expired


def test_message_expired():
    """A message with a past ``expires_at`` is expired."""
    msg = Message(
        session_id=uuid4(),
        sender="orchestrator",
        sender_instance="orchestrator-1",
        recipient="executor",
        msg_type=MsgType.TASK_DISPATCH,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    assert msg.is_expired


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


def test_msg_type_values():
    """All MsgType members have the correct dot-notation string values."""
    assert MsgType.TASK_DISPATCH.value == "task.dispatch"
    assert MsgType.TASK_RESULT.value == "task.result"
    assert MsgType.TASK_REVIEW.value == "task.review"
    assert MsgType.TASK_VERDICT.value == "task.verdict"
    assert MsgType.INTERRUPT_RAISE.value == "interrupt.raise"
    assert MsgType.INTERRUPT_RESPONSE.value == "interrupt.response"
    assert MsgType.SESSION_START.value == "session.start"
    assert MsgType.SESSION_COMPLETE.value == "session.complete"
    assert MsgType.AGENT_ERROR.value == "agent.error"
    assert MsgType.AGENT_HEARTBEAT.value == "agent.heartbeat"


def test_task_status_values():
    assert TaskStatus.DONE.value == "done"
    assert TaskStatus.DONE_WITH_CONCERNS.value == "done_with_concerns"
    assert TaskStatus.PARTIAL.value == "partial"
    assert TaskStatus.ERROR.value == "error"


def test_verdict_values():
    assert Verdict.PASS.value == "pass"
    assert Verdict.PASS_WITH_WARNINGS.value == "pass_with_warnings"
    assert Verdict.FAIL.value == "fail"


def test_agent_error_code_values():
    assert AgentErrorCode.TIMEOUT.value == "timeout"
    assert AgentErrorCode.LLM_ERROR.value == "llm_error"
    assert AgentErrorCode.TOOL_ERROR.value == "tool_error"
    assert AgentErrorCode.VALIDATION_ERROR.value == "validation_error"
    assert AgentErrorCode.CRASH.value == "crash"
    assert AgentErrorCode.INTERRUPT_TIMEOUT.value == "interrupt_timeout"
    assert AgentErrorCode.UNKNOWN.value == "unknown"


# ---------------------------------------------------------------------------
# Auxiliary models
# ---------------------------------------------------------------------------


def test_review_issue_create():
    """ReviewIssue with each severity level."""
    for sev in ("critical", "important", "minor"):
        issue = ReviewIssue(severity=sev, field="price_median", message="Value seems off")
        assert issue.severity == sev
        assert issue.field == "price_median"
        assert issue.message == "Value seems off"


def test_artifact_with_description():
    a = Artifact(
        name="Report",
        path="output/report.xlsx",
        type="xlsx",
        description="Pricing analysis report",
    )
    assert a.name == "Report"
    assert a.description == "Pricing analysis report"


def test_artifact_without_description():
    a = Artifact(name="Dataset", path="output/dataset.csv", type="csv")
    assert a.description is None


# ---------------------------------------------------------------------------
# Payload & reply chain
# ---------------------------------------------------------------------------


def test_message_payload_arbitrary():
    """Payload holds nested structures."""
    msg = Message(
        session_id=uuid4(),
        sender="executor",
        sender_instance="executor-1",
        recipient="orchestrator",
        msg_type=MsgType.TASK_RESULT,
        payload={
            "listings": [{"title": "iPhone", "price": 500}],
            "count": 1,
            "active": True,
        },
    )
    assert msg.payload["listings"][0]["price"] == 500
    assert msg.payload["count"] == 1


def test_message_parent_id_chain():
    """Two messages linked via parent_id form a reply chain."""
    session = uuid4()
    m1 = Message(
        session_id=session,
        sender="orchestrator",
        sender_instance="orch-1",
        recipient="executor",
        msg_type=MsgType.TASK_DISPATCH,
    )
    m2 = Message(
        session_id=session,
        sender="executor",
        sender_instance="exec-1",
        recipient="orchestrator",
        msg_type=MsgType.TASK_RESULT,
        parent_id=m1.id,
    )
    assert m2.parent_id == m1.id
