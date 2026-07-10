# Spec 001: Message Protocol & Runtime Core

**Status:** Draft
**Date:** 2026-06-26
**Author:** Human + Claude

---

## 1. Scope

This spec defines:

- **Message Protocol** — how Orchestrator, Executor, Reviewer, Responder, and Interrupt nodes communicate. No direct method calls. Everything through a message bus.
- **Runtime Core** — the asyncio event loop that owns agent lifecycle (spawn, run, kill), message routing, and interrupt/checkpoint.

Out of scope: skill format, tool mapping, bootstrap, telemetry (those get their own specs).

---

## 2. Message Protocol

### 2.1 Envelope

Every message has the same envelope. This is the only contract agents need to understand:

```python
from pydantic import BaseModel, Field
from typing import Any
from uuid import UUID, uuid4
from datetime import UTC, datetime

class Message(BaseModel):
    id: UUID = Field(default_factory=uuid4)       # unique message ID
    session_id: UUID                              # which session (Orchestrator lifetime)
    sender: str                                   # agent type: "orchestrator" | "executor" | "reviewer" | "responder"
    sender_instance: str                          # instance id (orchestrator-1, executor-task3-run2)
    recipient: str                                # agent type or "orchestrator" (Orchestrator is always the hub)
    recipient_instance: str | None = None         # None = any; specific = route to exact instance

    msg_type: MsgType                             # see §2.2
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_id: UUID | None = None                 # reply chain (for tracing) — §2.5

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None            # TTL; if unread by expiry → dropped + error back to sender — §2.3

    @property
    def is_expired(self) -> bool: ...             # §2.3 convenience check
```

### 2.2 Message Types (msg_type)

| msg_type | Direction | Payload | Meaning |
|----------|-----------|---------|---------|
| `task.dispatch` | Orch → Executor | `{task_id, skill_name, task_spec, skill_context}` | Assign 1 task to a fresh Executor. `skill_context` contains skill definition, allowed actions, action bindings, and output schema — see Spec 002 §9. |
| `task.result` | Executor → Orch | `{task_id, status: TaskStatus, output, concerns?: str[]}` | Executor's final output. `status` uses `TaskStatus` enum. `concerns` is Executor's own doubts (optional). Output validated against skill's output schema if present (Spec 002 §9.2). |
| `task.review` | Orch → Reviewer | `{task_id, task_spec, executor_output, system_prompt, concerns?, schema_validation_issues?}` | Ask Reviewer to verify. `system_prompt` is the fully rendered adversarial review prompt (built by Orchestrator from template in Spec 004 §4.5). `concerns` are Executor's own doubts from `task.result` (if any). `schema_validation_issues` are from Orchestrator's output schema validation step (if any). Reviewer performs a single LLM call with this prompt and returns a verdict. |
| `task.verdict` | Reviewer → Orch | `{task_id, verdict: Verdict, issues: ReviewIssue[], confidence: float}` | `verdict` uses `Verdict` enum. `issues` is a list of `ReviewIssue` objects. `confidence` is 0.0–1.0. |
| `interrupt.raise` | Any → Orch | `{message, options: str[], context: {task_id, current_step, relevant_summary?}}` | Agent needs human judgment. `context` carries enough info for human to decide without reading full history. |
| `interrupt.response` | Orch → Agent | `{decision, human_note?}` | Human's answer fed back |
| `session.start` | Runtime → Orch | `{goal, params}` | New session initiated |
| `user.message` | Human → Orch | `{text: str}` | General user input during a session. Orchestrator classifies and routes per Spec 004 §3. Sent by CLI, Web UI, or any human-facing channel. |
| `session.complete` | Orch → Runtime | `{summary: str, artifacts: Artifact[]}` | Final result. `Artifact = {name, path, type, description?}` — file paths relative to session output dir. |
| `agent.error` | Any → Orch | `{error_code: AgentErrorCode, detail: str, recoverable: bool}` | Agent crashed / timed out / validation failed. See `AgentErrorCode` enum. |
| `agent.heartbeat` | Any → Orch | `{}` | Still alive (if idle > 30s) |

### 2.2.1 Supporting Enums

```python
from enum import StrEnum  # Python 3.11+

class MsgType(StrEnum):
    """All valid message types."""
    TASK_DISPATCH = "task.dispatch"
    TASK_RESULT = "task.result"
    TASK_REVIEW = "task.review"
    TASK_VERDICT = "task.verdict"
    INTERRUPT_RAISE = "interrupt.raise"
    INTERRUPT_RESPONSE = "interrupt.response"
    SESSION_START = "session.start"
    SESSION_COMPLETE = "session.complete"
    USER_MESSAGE = "user.message"                    # Spec 004 §3 (classification)
    AGENT_ERROR = "agent.error"
    AGENT_HEARTBEAT = "agent.heartbeat"
    RESPOND_QUERY = "respond.query"              # Spec 003
    RESPOND_REPLY = "respond.reply"              # Spec 003
    RESPOND_REQUEST_TOOL = "respond.request_tool" # Spec 003
    RESPOND_TOOL_RESULT = "respond.tool_result"   # Spec 003

class TaskStatus(StrEnum):
    """Executor's final status in task.result."""
    DONE = "done"
    DONE_WITH_CONCERNS = "done_with_concerns"  # Executor flags own doubts
    PARTIAL = "partial"                        # Incomplete but useful
    ERROR = "error"                            # Execution failed

class Verdict(StrEnum):
    """Reviewer's judgment in task.verdict."""
    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"  # Acceptable but noted
    FAIL = "fail"                               # Must re-do

class AgentErrorCode(StrEnum):
    """Error codes for agent.error messages."""
    TIMEOUT = "timeout"                    # Agent exceeded time limit
    LLM_ERROR = "llm_error"                # LLM API error / rate limit
    TOOL_ERROR = "tool_error"              # Action/tool execution failed
    VALIDATION_ERROR = "validation_error"  # Output failed schema validation
    CRASH = "crash"                        # Unhandled exception
    INTERRUPT_TIMEOUT = "interrupt_timeout" # Human didn't respond in TTL
    UNKNOWN = "unknown"                    # Catch-all

class ReviewIssue(BaseModel):
    """An issue found by Reviewer in task.verdict — distinct from Spec 002's ValidationIssue (skill validation)."""
    severity: Literal["critical", "important", "minor"]
    field: str          # which part of the output has the issue
    message: str        # human-readable description

class Artifact(BaseModel):
    """A file produced during the session."""
    name: str           # human-readable label
    path: str           # path relative to session output directory
    type: str           # "csv" | "xlsx" | "json" | "pdf" | "txt" | "other"
    description: str | None = None

class AgentType(StrEnum):
    """Well-known agent roles in the harness topology — §1 Agent Topology."""
    ORCHESTRATOR = "orchestrator"  # "sếp": nhận yêu cầu, lập plan, dispatch
    EXECUTOR = "executor"          # "làm": constructive, hoàn thành 1 task
    REVIEWER = "reviewer"          # "kiểm": adversarial, refute, bắt lỗi
    RESPONDER = "responder"        # conversational Q&A — Spec 003
```

### 2.3 Message Expiry

Messages with `expires_at < now()` are **dropped by the Runtime** before delivery. When a message expires unread:

1. Runtime logs the expiry with message `id` and `msg_type`
2. Runtime sends `agent.error(TIMEOUT, detail="Message {id} expired unread", recoverable=True)` back to `sender`
3. The sender decides: retry with new TTL, or escalate

This applies to ALL message types. For `interrupt.raise`, the interrupt checkpoint also has its own TTL (§3.4) — if BOTH expire, the interrupt is terminated.

### 2.4 Routing Rule

**Orchestrator is the hub.** All messages route through Orchestrator. No Executor talks directly to Reviewer. No peer-to-peer.

```
Human ────→ Orchestrator ──→ Reviewer
 (user.message)  │
                 │
 Executor ───────┤
                 │
                 ├──→ Responder          ← Spec 003
                 │       │
                 │       └──→ respond.request_tool → Orchestrator → Executor
                 │
                 ├──→ Interrupt (human)
                 │
                 └──→ Executor (re-spawn on review fail)
```

This is simpler than a full mesh and makes every decision traceable. It also means the Orchestrator always knows the full state of every in-flight task and conversation.

### 2.5 Reply Chains

`parent_id` links messages into trees. When a Reviewer issues a verdict, `parent_id` points to the `task.dispatch` that started the chain. This enables full audit tracing: "show me the entire lifecycle of Task 3".

---

## 3. Runtime Core

### 3.1 Runtime Backend

**v0 uses LangGraph** as the state management and execution engine. Rationale:

- **State graph is a natural fit.** The agent lifecycle (§3.3) is inherently a state machine with branching (RUNNING → INTERRUPT → RUNNING / ERROR). LangGraph's `StateGraph` + checkpointing (`InMemorySaver`) models this cleanly without reinventing the wheel.
- **Interrupt is built-in.** LangGraph's `interrupt()` primitive maps directly to our HITL requirement (§3.4) — pause execution, persist checkpoint, wait for human, resume. No need to hand-roll this on raw asyncio.
- **Less code.** LangGraph handles the plumbing (state transitions, checkpoint serialization, graph execution) so we focus on agent logic, not event-loop mechanics.
- **Replaceable backend preserved.** The `AgentRuntime` is an ABC. v0 ships `LangGraphRuntime`. The `AsyncioRuntime` (plain asyncio + Queue) is also kept as a lightweight alternative for simple use cases. v1 could add `RayRuntime` or `CeleryRuntime` without changing skill code.

### 3.2 AgentRuntime Interface

```python
class AgentRuntime(ABC):
    """Abstract agent execution backend.

    Owns agent lifecycle (§3.3), message routing (§2.4), and the
    interrupt / human-in-the-loop primitive (§3.4).
    """

    @abstractmethod
    async def spawn(self, agent_type: str, context: dict) -> str:
        """Create a new agent instance. Returns instance_id.

        Sets the agent's initial state to INIT then RUNNING (§3.3),
        and starts any background tasks (e.g. heartbeat, §2.2).
        """
        ...

    @abstractmethod
    async def send(self, message: Message) -> None:
        """Route a message to its recipient.

        Resolves ``recipient`` / ``recipient_instance`` (§2.4).  If the
        message has expired (§2.3) drops it and sends ``agent.error``
        back to the sender.
        """
        ...

    @abstractmethod
    async def interrupt(self, instance_id: str, prompt: str, options: list[str]) -> str:
        """Pause agent, ask human, return decision. Blocks until human responds.

        Saves a checkpoint to disk (§3.4), notifies the human channel,
        transitions the agent to INTERRUPT (§3.3), and blocks on an
        ``asyncio.Future`` until ``resolve_interrupt()`` is called.
        Raises ``InterruptTimeoutError`` if TTL expires (§3.4 ¶2).
        """
        ...

    @abstractmethod
    async def kill(self, instance_id: str) -> None:
        """Terminate an agent instance. Idempotent.

        §3.3: any state → DEAD.  Cancels pending interrupt futures and
        cleans up checkpoint files (§3.4).
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully stop all agents, cancel pending tasks, and release
        resources.  After ``shutdown()`` the runtime must reject further
        ``send()`` calls.
        """
        ...
```

### 3.3 Agent Lifecycle

```
                  spawn()
                     │
                     ▼
   ┌─────────────────────────────────────┐
   │  Agent Instance                     │
   │                                     │
   │  INIT → RUNNING → (INTERRUPT)*      │
   │              │         │            │
   │              │         └─ resume ──┘│
   │              │                      │
   │              ├─ COMPLETE (done)     │
   │              └─ ERROR (crashed)     │
   │                                     │
   │  kill() → DEAD (any state → DEAD)   │
   └─────────────────────────────────────┘
```

States:

| State | Meaning | Can transition to |
|-------|---------|-------------------|
| `INIT` | Agent created, not yet running | RUNNING |
| `RUNNING` | Processing a task (Executor/Reviewer) or answering a question (Responder) | INTERRUPT, COMPLETE, ERROR, DEAD |
| `INTERRUPT` | Paused, waiting for human | RUNNING (resume), ERROR (timeout), DEAD |
| `COMPLETE` | Task finished (Executor/Reviewer) or session ended (Responder/Orchestrator) | DEAD |
| `ERROR` | Crashed or timed out | DEAD |
| `DEAD` | Terminal | — |

### 3.4 Checkpoint (for Interrupt)

When an agent raises `interrupt.raise`, the runtime:

1. **Freezes** the agent's state (all messages in the reply chain, current task context)
2. **Saves** a checkpoint to disk (`~/.llend/sessions/{session_id}/checkpoints/{interrupt_id}.json`)
3. **Notifies** human via configured channel (Telegram, Discord, WebSocket)
4. **Blocks** the agent (not the whole runtime — other tasks continue)
5. **On response:** reloads checkpoint, injects `interrupt.response` into agent's inbox, resumes

If human doesn't respond in `TTL` (default 24h), interrupt times out → `ERROR` state → Orchestrator decides: retry / skip / escalate.

**Checkpoint file schema** (`~/.llend/sessions/{session_id}/checkpoints/{interrupt_id}.json`):

```python
class InterruptTimeoutError(Exception):
    """Raised inside ``runtime.interrupt()`` when TTL expires without human response.

    The caller (Executor / Reviewer / Orchestrator) catches this to decide:
    retry, skip, or escalate.
    """
    def __init__(self, interrupt_id: UUID, ttl_seconds: int) -> None: ...

class Checkpoint(BaseModel):
    """Saved agent state for interrupt/resume."""
    interrupt_id: UUID
    session_id: UUID
    agent_instance: str              # e.g. "executor-task1-run1"
    agent_type: str                  # "executor" | "reviewer" | "responder"
    agent_state: str = "INTERRUPT"   # always "INTERRUPT" when checkpointed
    reply_chain: list[UUID] = []     # all message IDs in the current task chain
    task_context: dict[str, Any] = {}  # current task_spec, partial results, etc.
    interrupt_message: str           # the human-facing question
    interrupt_options: list[str] = []  # choices presented to human
    created_at: datetime
    ttl_seconds: int = 86400         # 24h default, configurable per interrupt (Q1)
    human_response: str | None = None  # filled on resume
    resolved_at: datetime | None = None

    # Disk persistence methods (§3.4 ¶2)
    def save(self, base_dir: Path | None = None) -> Path: ...
    @classmethod
    def load(cls, session_id: UUID, interrupt_id: UUID, base_dir: Path | None = None) -> Checkpoint | None: ...
    def delete(self, base_dir: Path | None = None) -> None: ...
    @classmethod
    def list_for_session(cls, session_id: UUID, base_dir: Path | None = None) -> list[UUID]: ...
```

#### 3.4.1 Notification Channels

```python
class NotificationChannel(ABC):
    """Contract for a human-notification transport — §3.4 step 3."""

    @abstractmethod
    async def notify_interrupt(self, checkpoint: Checkpoint) -> None:
        """Called when an agent raises ``interrupt.raise``."""
        ...

    @abstractmethod
    async def notify_interrupt_timeout(self, checkpoint: Checkpoint) -> None:
        """Called when a checkpoint's TTL expires without a human response."""
        ...

class ConsoleNotificationChannel(NotificationChannel):
    """Print interrupt notifications to stdout — v0 default."""

class MultiChannel(NotificationChannel):
    """Fan-out to multiple ``NotificationChannel`` instances.

    A failure in one channel does **not** prevent subsequent channels
    from being called — errors are logged and swallowed.
    """
    def __init__(self, *channels: NotificationChannel) -> None: ...
```

Channels mentioned in spec but implemented later: Telegram, Discord, WebSocket (§3.4 step 3).

### 3.5 Concurrency Model

```
                    ┌─────────────┐
                    │   Asyncio   │
                    │ Event Loop  │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                  ▼
   Orchestrator        Executor           Reviewer
   (1 instance)        (0..N)             (0..N, one per task)
         │
         └── Responder (1 instance, Spec 003)
                           │
                     spawn → run → complete → kill
                     (sequential per task, but
                      tasks can be parallel if independent)
```

**Rule:** one Executor per task at a time. If a task fails review, kill the old Executor, spawn a new one. Two tasks CAN run concurrently if the Orchestrator determines they're independent (e.g., scrape eBay + scrape Amazon in parallel, then merge). Responder runs alongside Executors — conversations do not block task execution.

---

## 4. Walkthrough: Market Researcher Task 1

```
1. Human: "Phân tích thị trường iPhone 15 trên eBay"

2. Runtime spawns Orchestrator
   → Message(session.start, goal="Phân tích thị trường iPhone 15 trên eBay")

3. Orchestrator spawns Responder (Spec 003) — lives entire session for conversational Q&A

4. Orchestrator decomposes into plan:
   Task 1: data_provider (scrape eBay iPhone 15)
   Task 2: analyze_pricing
   Task 3: write_report

5. Orchestrator → Message(task.dispatch, skill="data_provider", ...)
   Runtime spawns Executor #1 (INIT → RUNNING)

6. Executor #1 crawls eBay, hits 15,000 listings
   → Message(interrupt.raise, "15k dòng, phân tích hết hay lọc? [A/B/C]")

7. Runtime: checkpoint → notify human → block Executor #1 (INTERRUPT)

8. Human: "Chọn B - 1,000 dòng mới nhất"
   → Runtime: checkpoint load → Message(interrupt.response, "B") → Executor #1 resumes

9. Executor #1 finishes → Message(task.result, status=DONE, output=clean_dataset)

10. Orchestrator → Message(task.review, spec=..., output=clean_dataset)
    Runtime spawns Reviewer #1

11. Reviewer #1 → Message(task.verdict, verdict=pass, issues=[])

12. Orchestrator marks Task 1 complete, proceeds to Task 2...
```

### 4.1 Walkthrough: Parallel Tasks

When tasks are independent, they run concurrently:

```
1. Human: "So sánh giá iPhone 15 trên eBay và Amazon"

2. Orchestrator decomposes:
   Task 1: data_provider (eBay)
   Task 2: data_provider (Amazon)
   Task 3: compare_pricing (depends on Task 1 + Task 2)

3. Orchestrator spawns Executor #1 (eBay) AND Executor #2 (Amazon) IN PARALLEL
   → Both execute simultaneously via asyncio.gather()

4. Executor #1 finishes → task.result(eBay dataset)
   Executor #2 finishes → task.result(Amazon dataset)

5. Both complete → Orchestrator dispatches Task 3 with both datasets as input

6. Executor #3 runs compare_pricing → task.result(comparison report)
```

**Rule for parallel dispatch:** Orchestrator checks `SkillPipeline.build_plan()` — tasks at the same depth with no interdependency are marked `parallelizable=True`. Orchestrator calls `asyncio.gather()` on all parallelizable tasks at the same depth before proceeding to the next depth.

---

## 5. File Layout

```
llend/
├── runtime/
│   ├── __init__.py
│   ├── base.py              # AgentRuntime ABC (§3.2)
│   ├── asyncio_runtime.py   # Lightweight asyncio backend (§3.1)
│   ├── langgraph_runtime.py # v0 primary backend — LangGraph state graph (§3.1)
│   ├── message.py           # Message envelope + enums (§2.1, §2.2, §2.2.1)
│   ├── lifecycle.py         # Agent states, spawn/kill transitions (§3.3)
│   ├── checkpoint.py        # Checkpoint model + disk persistence (§3.4)
│   └── notifications.py     # Human notification channels (§3.4 step 3)
├── ...
```

---

## 6. Open Questions

- **Q1:** Interrupt TTL — is 24h a sensible default? Should be configurable per interrupt? → **Configurable per interrupt** confirmed. `Checkpoint.ttl_seconds` overrides the 24h default.
- **Q2:** ~~Checkpoint format — JSON or pickle?~~ **Resolved:** JSON. Schema defined in §3.4. Debuggable, human-readable. Pickle only if performance demands it later.
- **Q3:** Should `asyncio.gather()` be the only parallelism primitive, or do we want a task queue (Redis/AMQP) from day 1? → **Start with asyncio only.** Parallel dispatch demonstrated in §4.1. Add queue when scaling demands it.
- **Q4:** ~~Message serialization — JSON over what transport?~~ **Resolved:** v0 = in-process Python object passing within the same asyncio event loop. No serialization, no network transport. Messages are plain Python `Message` objects passed via `asyncio.Queue` between agents. Checkpoints are the ONLY place where messages are serialized (to JSON on disk, §3.4). v1 may add Redis pub/sub for distributed agents.

---

*Next spec: 003 — Responder Agent & Conversation Module*
