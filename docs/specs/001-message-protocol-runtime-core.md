# Spec 001: Message Protocol & Runtime Core

**Status:** Draft
**Date:** 2026-06-26
**Author:** Human + Claude

---

## 1. Scope

This spec defines:

- **Message Protocol** — how Orchestrator, Executor, Reviewer, and Interrupt nodes communicate. No direct method calls. Everything through a message bus.
- **Runtime Core** — the asyncio event loop that owns agent lifecycle (spawn, run, kill), message routing, and interrupt/checkpoint.

Out of scope: skill format, tool mapping, bootstrap, telemetry (those get their own specs).

---

## 2. Message Protocol

### 2.1 Envelope

Every message has the same envelope. This is the only contract agents need to understand:

```python
from pydantic import BaseModel
from typing import Any, Optional
from uuid import UUID
from datetime import datetime

class Message(BaseModel):
    id: UUID                          # unique message ID
    session_id: UUID                  # which session (Orchestrator lifetime)
    sender: str                       # agent type: "orchestrator" | "executor" | "reviewer"
    sender_instance: str              # instance id (orchestrator-1, executor-task3-run2)
    recipient: str                    # agent type or "orchestrator" (Orchestrator is always the hub)
    recipient_instance: Optional[str] # None = any; specific = route to exact instance

    msg_type: str                     # see §2.2
    payload: dict[str, Any]           # type-specific content
    parent_id: Optional[UUID]         # reply chain (for tracing)

    created_at: datetime
    expires_at: Optional[datetime]    # TTL; if unread by expiry → dropped + error back to sender
```

### 2.2 Message Types (msg_type)

| msg_type | Direction | Payload | Meaning |
|----------|-----------|---------|---------|
| `task.dispatch` | Orch → Executor | `{task_id, skill_name, task_spec, context}` | Assign 1 task to a fresh Executor |
| `task.result` | Executor → Orch | `{task_id, status, output, concerns?}` | Executor's final output |
| `task.review` | Orch → Reviewer | `{task_id, task_spec, executor_output}` | Ask Reviewer to verify |
| `task.verdict` | Reviewer → Orch | `{task_id, verdict, issues[], confidence}` | pass / fail + issues found |
| `interrupt.raise` | Any → Orch | `{message, options[], context}` | Agent needs human judgment |
| `interrupt.response` | Orch → Agent | `{decision, human_note?}` | Human's answer fed back |
| `session.start` | Runtime → Orch | `{goal, params}` | New session initiated |
| `session.complete` | Orch → Runtime | `{summary, artifacts[]}` | Final result |
| `agent.error` | Any → Orch | `{error_code, detail, recoverable?}` | Agent crashed / timed out |
| `agent.heartbeat` | Any → Orch | `{}` | Still alive (if idle > 30s) |

### 2.3 Routing Rule

**Orchestrator is the hub.** All messages route through Orchestrator. No Executor talks directly to Reviewer. No peer-to-peer.

```
Executor ──→ Orchestrator ──→ Reviewer
                │
                ├──→ Interrupt (human)
                │
                └──→ Executor (re-spawn on review fail)
```

This is simpler than a full mesh and makes every decision traceable. It also means the Orchestrator always knows the full state of every in-flight task.

### 2.4 Reply Chains

`parent_id` links messages into trees. When a Reviewer issues a verdict, `parent_id` points to the `task.dispatch` that started the chain. This enables full audit tracing: "show me the entire lifecycle of Task 3".

---

## 3. Runtime Core

### 3.1 Event Loop

Plain `asyncio` — no LangGraph, no Celery. Rationale:

- Avoid framework lock-in. LangGraph's interrupt pattern inspired us, but their runtime is tied to their graph model.
- `asyncio` is Python's native concurrency model. Every LLM SDK (OpenAI, Anthropic) already supports it.
- **Future: replaceable backend.** The `AgentRuntime` is an ABC. v0 ships `AsyncioRuntime`. v1 could add `RayRuntime` or `CeleryRuntime` without changing skill code.

### 3.2 AgentRuntime Interface

```python
class AgentRuntime(ABC):
    """Abstract agent execution backend."""

    @abstractmethod
    async def spawn(self, agent_type: str, context: dict) -> str:
        """Create a new agent instance. Returns instance_id."""
        ...

    @abstractmethod
    async def send(self, message: Message) -> None:
        """Route a message to its recipient."""
        ...

    @abstractmethod
    async def interrupt(self, instance_id: str, prompt: str, options: list[str]) -> str:
        """Pause agent, ask human, return decision. Blocks until human responds."""
        ...

    @abstractmethod
    async def kill(self, instance_id: str) -> None:
        """Terminate an agent instance. Idempotent."""
        ...
```

### 3.3 Agent Lifecycle (AsyncioRuntime)

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
| `RUNNING` | Processing a task | INTERRUPT, COMPLETE, ERROR, DEAD |
| `INTERRUPT` | Paused, waiting for human | RUNNING (resume), ERROR (timeout), DEAD |
| `COMPLETE` | Task finished successfully | DEAD |
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

### 3.5 Concurrency Model

```
                    ┌─────────────┐
                    │   Asyncio   │
                    │ Event Loop  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        Orchestrator   Executor     Reviewer
        (1 instance)   (0..N)       (0..N, one per task)
                           │
                     spawn → run → complete → kill
                     (sequential per task, but
                      tasks can be parallel if independent)
```

**Rule:** one Executor per task at a time. If a task fails review, kill the old Executor, spawn a new one. Two tasks CAN run concurrently if the Orchestrator determines they're independent (e.g., scrape eBay + scrape Amazon in parallel, then merge).

---

## 4. Walkthrough: Market Researcher Task 1

```
1. Human: "Phân tích thị trường iPhone 15 trên eBay"

2. Runtime spawns Orchestrator
   → Message(session.start, goal="Phân tích thị trường iPhone 15 trên eBay")

3. Orchestrator decomposes into plan:
   Task 1: data_provider (scrape eBay iPhone 15)
   Task 2: analyze_pricing
   Task 3: write_report

4. Orchestrator → Message(task.dispatch, skill="data_provider", ...)
   Runtime spawns Executor #1 (INIT → RUNNING)

5. Executor #1 crawls eBay, hits 15,000 listings
   → Message(interrupt.raise, "15k dòng, phân tích hết hay lọc? [A/B/C]")

6. Runtime: checkpoint → notify human → block Executor #1 (INTERRUPT)

7. Human: "Chọn B - 1,000 dòng mới nhất"
   → Runtime: checkpoint load → Message(interrupt.response, "B") → Executor #1 resumes

8. Executor #1 finishes → Message(task.result, status=DONE, output=clean_dataset)

9. Orchestrator → Message(task.review, spec=..., output=clean_dataset)
   Runtime spawns Reviewer #1

10. Reviewer #1 → Message(task.verdict, verdict=pass, issues=[])

11. Orchestrator marks Task 1 complete, proceeds to Task 2...
```

---

## 5. File Layout

```
llend_harness/
├── runtime/
│   ├── __init__.py
│   ├── base.py          # AgentRuntime ABC
│   ├── asyncio_runtime.py  # v0 implementation
│   ├── message.py       # Message, msg_type enum, routing
│   ├── lifecycle.py     # Agent states, spawn/kill transitions
│   └── checkpoint.py    # Interrupt save/load
├── ...
```

---

## 6. Open Questions

- **Q1:** Interrupt TTL — is 24h a sensible default? Should be configurable per interrupt?
- **Q2:** Checkpoint format — JSON or pickle? JSON for now (debuggable), pickle later if needed.
- **Q3:** Should `asyncio.gather()` be the only parallelism primitive, or do we want a task queue (Redis/AMQP) from day 1? → **Start with asyncio only.** Add queue when scaling demands it.
- **Q4:** Message serialization — JSON over what transport? In-process for v0 (same event loop). Network transport (Redis pub/sub) for v1.

---

*Next spec: 002 — Skill Format & Registry*
