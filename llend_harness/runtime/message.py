"""Message Protocol — the typed envelope and enums used by all agents to communicate.

Every message in the harness uses the same envelope (Message), routes through the
Orchestrator hub, and carries a typed msg_type with corresponding payload.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Message Type enum
# ---------------------------------------------------------------------------


class MsgType(StrEnum):
    """All message types in the llend_harness protocol."""

    TASK_DISPATCH = "task.dispatch"
    TASK_RESULT = "task.result"
    TASK_REVIEW = "task.review"
    TASK_VERDICT = "task.verdict"
    INTERRUPT_RAISE = "interrupt.raise"
    INTERRUPT_RESPONSE = "interrupt.response"
    SESSION_START = "session.start"
    SESSION_COMPLETE = "session.complete"
    AGENT_ERROR = "agent.error"
    AGENT_HEARTBEAT = "agent.heartbeat"
    # Spec 003 placeholders — Responder agent conversation messages
    RESPOND_QUERY = "respond.query"
    RESPOND_REPLY = "respond.reply"
    RESPOND_REQUEST_TOOL = "respond.request_tool"
    RESPOND_TOOL_RESULT = "respond.tool_result"


# ---------------------------------------------------------------------------
# Task / Review enums
# ---------------------------------------------------------------------------


class TaskStatus(StrEnum):
    """Executor's own assessment of its output."""

    DONE = "done"
    DONE_WITH_CONCERNS = "done_with_concerns"
    PARTIAL = "partial"
    ERROR = "error"


class Verdict(StrEnum):
    """Reviewer's verdict on an Executor's output."""

    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    FAIL = "fail"


class AgentErrorCode(StrEnum):
    """Standardised error codes for agent.error messages."""

    TIMEOUT = "timeout"
    LLM_ERROR = "llm_error"
    TOOL_ERROR = "tool_error"
    VALIDATION_ERROR = "validation_error"
    CRASH = "crash"
    INTERRUPT_TIMEOUT = "interrupt_timeout"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Auxiliary models
# ---------------------------------------------------------------------------


class ReviewIssue(BaseModel):
    """A single issue found by the Reviewer in an Executor's output."""

    severity: Literal["critical", "important", "minor"]
    field: str
    message: str


class Artifact(BaseModel):
    """A deliverable produced during a session — file relative to session output dir."""

    name: str
    path: str
    type: str  # "csv" | "xlsx" | "json" | "pdf" | "txt" | "other"
    description: str | None = None


# ---------------------------------------------------------------------------
# Message envelope
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """Universal message envelope for all inter-agent communication.

    Every message carries an id, session_id, sender/recipient routing info,
    a typed msg_type, and an opaque payload.  The Orchestrator is the hub:
    all messages route through it — there is no peer-to-peer.
    """

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    sender: str  # agent type: "orchestrator" | "executor" | "reviewer" | "responder"
    sender_instance: str  # instance id e.g. "orchestrator-1", "executor-task3-run2"
    recipient: str  # agent type or "orchestrator"
    recipient_instance: str | None = None  # None = any; specific = route to exact instance

    msg_type: MsgType
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_id: UUID | None = None  # reply chain for tracing

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = (
        None  # TTL; if unread by expiry → dropped + error back to sender
    )

    @property
    def is_expired(self) -> bool:
        """Check whether this message has passed its expiry time."""
        if self.expires_at is None:
            return False
        return datetime.now(UTC) >= self.expires_at
