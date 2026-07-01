"""Message Protocol — the typed envelope and enums used by all agents to communicate.

Every message in the harness uses the same envelope (``Message``), routes through the
Orchestrator hub, and carries a typed ``msg_type`` with corresponding payload.

Spec references
===============
- **§2.1 Envelope**      → ``Message`` (the only contract agents need to understand)
- **§2.2 Message Types** → ``MsgType`` enum (10 core + 4 Spec 003 placeholders)
- **§2.2.1 Enums**       → ``TaskStatus``, ``Verdict``, ``AgentErrorCode``, ``ReviewIssue``, ``Artifact``
- **§2.3 Expiry**        → ``Message.is_expired`` (checked by both runtimes before delivery)
- **§2.5 Reply Chains**  → ``Message.parent_id`` (links messages into trees for audit tracing)
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Message Type enum  —  Spec 001 §2.2
# ---------------------------------------------------------------------------


class MsgType(StrEnum):
    """All message types in the llend protocol.  Spec 001 §2.2."""

    # Core protocol — Spec 001 §2.2 table
    TASK_DISPATCH = "task.dispatch"          # Orch → Executor
    TASK_RESULT = "task.result"              # Executor → Orch
    TASK_REVIEW = "task.review"              # Orch → Reviewer
    TASK_VERDICT = "task.verdict"            # Reviewer → Orch
    INTERRUPT_RAISE = "interrupt.raise"      # Any → Orch  (§3.4)
    INTERRUPT_RESPONSE = "interrupt.response" # Orch → Agent (§3.4)
    SESSION_START = "session.start"          # Runtime → Orch
    SESSION_COMPLETE = "session.complete"    # Orch → Runtime
    AGENT_ERROR = "agent.error"              # Any → Orch  (§2.3, §3.4)
    AGENT_HEARTBEAT = "agent.heartbeat"      # Any → Orch  (§3.5, idle > 30s)

    # Spec 003 placeholders — Responder agent conversation messages
    RESPOND_QUERY = "respond.query"              # Spec 003
    RESPOND_REPLY = "respond.reply"              # Spec 003
    RESPOND_REQUEST_TOOL = "respond.request_tool" # Spec 003
    RESPOND_TOOL_RESULT = "respond.tool_result"   # Spec 003


# ---------------------------------------------------------------------------
# Task / Review enums  —  Spec 001 §2.2.1
# ---------------------------------------------------------------------------


class TaskStatus(StrEnum):
    """Executor's own assessment of its output.  Spec 001 §2.2.1."""

    DONE = "done"
    DONE_WITH_CONCERNS = "done_with_concerns"  # Executor flags own doubts
    PARTIAL = "partial"                        # Incomplete but useful
    ERROR = "error"                            # Execution failed


class Verdict(StrEnum):
    """Reviewer's verdict on an Executor's output.  Spec 001 §2.2.1."""

    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"  # Acceptable but noted
    FAIL = "fail"                               # Must re-do


class AgentErrorCode(StrEnum):
    """Standardised error codes for ``agent.error`` messages.  Spec 001 §2.2.1."""

    TIMEOUT = "timeout"                      # Agent exceeded time limit  (§2.3)
    LLM_ERROR = "llm_error"                  # LLM API error / rate limit
    TOOL_ERROR = "tool_error"                # Action/tool execution failed
    VALIDATION_ERROR = "validation_error"    # Output failed schema validation
    CRASH = "crash"                          # Unhandled exception
    INTERRUPT_TIMEOUT = "interrupt_timeout"  # Human didn't respond in TTL  (§3.4)
    UNKNOWN = "unknown"                      # Catch-all


# ---------------------------------------------------------------------------
# Auxiliary models  —  Spec 001 §2.2.1
# ---------------------------------------------------------------------------


class ReviewIssue(BaseModel):
    """A single issue found by the Reviewer in an Executor's output.

    Distinct from Spec 002's ``ValidationIssue`` (skill validation).
    Spec 001 §2.2.1.
    """

    severity: Literal["critical", "important", "minor"]
    field: str     # which part of the output has the issue
    message: str   # human-readable description


class Artifact(BaseModel):
    """A deliverable produced during a session.  Spec 001 §2.2.1.

    File paths are relative to the session output directory.
    """

    name: str           # human-readable label
    path: str           # path relative to session output directory
    type: str           # "csv" | "xlsx" | "json" | "pdf" | "txt" | "other"
    description: str | None = None


# ---------------------------------------------------------------------------
# Message envelope  —  Spec 001 §2.1
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """Universal message envelope for all inter-agent communication.  Spec 001 §2.1.

    Every message carries an id, session_id, sender/recipient routing info,
    a typed ``msg_type``, and an opaque payload.  The Orchestrator is the hub
    (§2.4): all messages route through it — there is no peer-to-peer.

    Fields (all from §2.1):
    - ``id``: unique message ID
    - ``session_id``: which session (Orchestrator lifetime)
    - ``sender``: agent type — "orchestrator" | "executor" | "reviewer" | "responder"
    - ``sender_instance``: instance id e.g. "orchestrator-1", "executor-task3-run2"
    - ``recipient``: agent type or "orchestrator" (Orchestrator is always the hub, §2.4)
    - ``recipient_instance``: None = any; specific = route to exact instance
    - ``msg_type``: see ``MsgType`` enum (§2.2)
    - ``payload``: type-specific content (§2.2 table)
    - ``parent_id``: reply chain for tracing (§2.5)
    - ``created_at``: timestamp
    - ``expires_at``: TTL; if unread by expiry → dropped + error back to sender (§2.3)
    """

    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    sender: str  # "orchestrator" | "executor" | "reviewer" | "responder"
    sender_instance: str  # e.g. "orchestrator-1", "executor-task3-run2"
    recipient: str  # agent type or "orchestrator"
    recipient_instance: str | None = None  # None = any; specific = route to exact instance

    msg_type: MsgType
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_id: UUID | None = None  # §2.5 reply chain

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None  # §2.3 TTL

    # ------------------------------------------------------------------
    # §2.3 Message Expiry
    # ------------------------------------------------------------------

    @property
    def is_expired(self) -> bool:
        """Check whether this message has passed its expiry time.  Spec 001 §2.3.

        Runtimes call this before delivery — expired messages are dropped
        and an ``agent.error(TIMEOUT)`` is sent back to the sender.
        """
        if self.expires_at is None:
            return False
        return datetime.now(UTC) >= self.expires_at
