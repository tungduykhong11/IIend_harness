# Spec 003: Responder Agent & Conversation Module

**Status:** Draft
**Date:** 2026-06-27
**Author:** Human + Claude
**Depends on:** [Spec 001 — Message Protocol & Runtime Core](./001-message-protocol-runtime-core.md), [Spec 002 — Skill Format & Registry](./002-skill-format-registry.md)

---

## 1. Motivation

Spec 001 defines a 3-agent topology: **Orchestrator** (plan + dispatch) → **Executor** (do) → **Reviewer** (check). This works for task execution, but it leaves a gap:

> After Executor crawls eBay and finds a desk for $150 with a 12-month installment policy... what does the user do next? Ask follow-up questions. Get advice. Have a conversation.

Currently, once the task pipeline completes, the session ends with `session.complete`. There is no agent whose job is to **talk to the human naturally** — answer questions, give recommendations, explain findings, suggest next actions.

This spec defines a 4th agent type: **Responder** — a conversational agent that lives for the entire session, answers questions in natural language, and can request additional tools when it needs more data.

---

## 2. Responder Agent

### 2.1 What It Is

| | Orchestrator | Executor | Reviewer | **Responder (NEW)** |
|---|---|---|---|---|
| **Role** | Plan, dispatch, adjudicate | Execute 1 task | Adversarially verify | Converse, advise, explain |
| **Mindset** | Manager | Constructor | Skeptic | Advisor, conversationalist |
| **Lifespan** | Entire session | Per task | Per task | **Entire session** |
| **Input** | Session goal | task.dispatch | task.review | respond.query (+ full session context) |
| **Output** | Execution plan | task.result | task.verdict | respond.reply (+ optionally respond.request_tool) |
| **Stateless?** | Holds session state | Stateless | Stateless | **Stateful** — maintains conversation history |

### 2.2 Updated Agent Topology

```
                         Orchestrator
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    Executor ──► Reviewer    │              Responder
    (do)        (check)      │              (converse)
                             │
                    Interrupt (human)
```

Responder sits as a peer to the Executor/Reviewer pipeline. It does NOT go through the do→check loop — its output is conversational, not a task artifact.

### 2.3 When Orchestrator Routes to Responder

Orchestrator classifies incoming human messages:

| Human says | Classification | Routes to |
|------------|---------------|-----------|
| "Phân tích giá iPhone 15 trên eBay" | Task request | Executor (via SkillPipeline) |
| "Giá $325 thì có nên mua không?" | Conversational question | **Responder** |
| "Giải thích tại sao median khác mean?" | Conversational question | **Responder** |
| "Crawl thêm Amazon để so sánh đi" | Task request | Executor |
| "Cảm ơn, tạm biệt" | Session end signal | Orchestrator → `session.complete` |

**Classification rule:** If the message asks for a new *action* (crawl, analyze, export) → task. If it asks for *opinion, explanation, advice, or follow-up* → route to Responder. Orchestrator uses an LLM call (cheap, fast model) for classification.

---

## 3. Message Types (New)

Spec 001 defines 10 `msg_type` values. Spec 003 adds 4 more:

| msg_type | Direction | Payload | Meaning |
|----------|-----------|---------|---------|
| `respond.query` | Orch → Responder | `{question: str, session_context: SessionContext, persona?: Persona}` | Ask Responder to answer a conversational question. Optional `persona` overrides default for this turn. |
| `respond.reply` | Responder → Orch | `{answer: str, advice?: str, follow_up_suggestions?: str[], confidence: float}` | Responder's answer |
| `respond.request_tool` | Responder → Orch | `{reason: str, suggested_skill: str, tool_params: dict}` | Responder wants more data to answer properly |
| `respond.tool_result` | Orch → Responder | `{request_id: UUID, result: dict}` | Tool result fed back to Responder to continue answering |

Additionally, Responder uses these existing message types:
- `interrupt.raise` — when Responder needs human clarification before answering
- `agent.error` — Responder crashes or times out
- `agent.heartbeat` — keep-alive

### 3.1 Supporting Enums & Models (Extend Spec 001)

```python
# Add to Spec 001's enums (§2.2.1)

class MsgType(str, Enum):
    # ... existing ...
    RESPOND_QUERY = "respond.query"
    RESPOND_REPLY = "respond.reply"
    RESPOND_REQUEST_TOOL = "respond.request_tool"
    RESPOND_TOOL_RESULT = "respond.tool_result"

class Persona(str, Enum):
    """Responder's tone of voice."""
    AUTO = "auto"        # Detect best persona from question type
    ANALYST = "analyst"  # Data-driven, precise, cites numbers
    ADVISOR = "advisor"  # Practical recommendations, pros/cons
    FRIEND = "friend"    # Casual, simple explanations
```

```python
# New models for respond payloads

class SessionContext(BaseModel):
    """What Responder sees when answering a question."""
    session_goal: str                          # original session objective
    conversation_history: list[ConversationTurn]  # past Q&A in this session
    task_results: list[TaskResultSummary]         # completed task outputs (summarized)
    active_task: TaskResultSummary | None = None  # currently running task (if any)
    user_profile: UserProfile | None = None       # preferences from past sessions

class ConversationTurn(BaseModel):
    """One turn in the conversation."""
    role: Literal["user", "responder"]
    content: str
    timestamp: datetime

class TaskResultSummary(BaseModel):
    """Summarized task result — not raw data, but key insights."""
    task_id: UUID
    skill_name: str
    status: TaskStatus
    summary: str                   # 1-3 sentence summary of what was found
    key_metrics: dict[str, Any]    # extracted numbers (e.g., {"median_price": 325})
    artifact_paths: list[str]      # paths to full output files if needed

class UserProfile(BaseModel):
    """Preferences learned across sessions."""
    preferred_platforms: list[str] = []       # ["ebay", "amazon"]
    budget_conscious: bool = False
    favorite_categories: list[str] = []       # ["electronics", "furniture"]
    persona_preference: Persona = Persona.AUTO
    custom_notes: dict[str, str] = {}         # free-form notes
```

---

## 4. Responder Lifecycle

### 4.1 Spawn & Reuse

Unlike Executor/Reviewer (spawned fresh per task), Responder is spawned **once per session** and reused:

```
Session Start
     │
     ▼
Orchestrator INIT
     │
     ├── Spawn Responder #1 (INIT → RUNNING)
     │   └── Responder stays alive entire session
     │
     ├── User asks task → Executor pipeline
     │
     ├── User asks question → respond.query → Responder answers → respond.reply
     │
     ├── User asks another question → respond.query → Responder answers
     │   └── (same Responder instance, accumulated conversation history)
     │
     └── Session ends → kill Responder (RUNNING → DEAD)
```

**Rationale:** Conversation requires continuity. Re-spawning a new Responder per question would lose:
- What was discussed earlier in the session
- Context from previous answers
- The relationship/style that developed with the user

### 4.2 Lifecycle States

Responder uses the same states as other agents (Spec 001 §3.3):

```
INIT → RUNNING → (INTERRUPT)* → COMPLETE/ERROR → DEAD
```

Key difference from Executor: Responder transitions between RUNNING and IDLE (internal substate). When no question is pending, Responder is IDLE — still RUNNING, just waiting for the next `respond.query`.

---

## 5. Session Context

### 5.1 What Responder Sees

When Orchestrator sends `respond.query`, the payload includes `SessionContext` — everything Responder needs to answer intelligently:

```python
{
    "msg_type": "respond.query",
    "payload": {
        "question": "Giá $325 thì có nên mua không?",
        "session_context": {
            "session_goal": "Phân tích giá iPhone 15 trên eBay",
            "conversation_history": [
                ConversationTurn(role="user", content="Phân tích giá iPhone 15..."),
                ConversationTurn(role="responder", content="Đã hoàn thành phân tích. Kết quả chính:..."),
                ConversationTurn(role="user", content="Giá $325 thì có nên mua không?"),
            ],
            "task_results": [
                TaskResultSummary(
                    task_id=UUID("..."),
                    skill_name="data_provider",
                    status=TaskStatus.DONE,
                    summary="Đã crawl 1000 listings iPhone 15 trên eBay. Giá từ $190 đến $1200.",
                    key_metrics={"total": 1000, "min": 190, "max": 1200},
                    artifact_paths=["output/ebay_iphone15_raw.csv"],
                ),
                TaskResultSummary(
                    task_id=UUID("..."),
                    skill_name="analyze_pricing",
                    status=TaskStatus.DONE,
                    summary="Median $325, normal range $190-$580. 8 listing nghi ngờ scam (<$190). Phân khúc $0-300 chiếm 45%.",
                    key_metrics={"median": 325, "normal_low": 190, "normal_high": 580, "outlier_cheap": 8},
                    artifact_paths=["output/iphone15_price_segments.csv", "output/analysis_report.json"],
                ),
            ],
            "active_task": None,
            "user_profile": UserProfile(
                preferred_platforms=["ebay"],
                budget_conscious=True,
                favorite_categories=["electronics"],
            ),
        },
    },
}
```

### 5.2 Context Compression

Task outputs can be huge (15k listings, 50MB CSV). Responder does NOT receive raw data. Instead:

1. **Orchestrator summarizes** each `task.result` into `TaskResultSummary` (1-3 sentences + key numbers)
2. **Responder can request details** via `respond.request_tool` if it needs to dig deeper:
   - "Cho tôi xem 8 listing nghi ngờ scam" → Orch loads artifact, returns specific data
3. **Full artifacts** are referenced by path — Responder quotes paths when relevant

**Summarization is an LLM call** by Orchestrator after each task completes. Cheap model, fast. Stored in session state for all downstream use.

---

## 6. Tool Delegation

### 6.1 The Flow

Responder can ask Orchestrator to run additional tasks when it lacks data:

```
User: "So với Amazon thì sao?"
  │
  ▼
Orchestrator: classify → conversational question
  │
  ▼
Message(respond.query, question="So với Amazon thì sao?", session_context=...)
  │
  ▼
Responder: "Tôi chưa có data Amazon. Cần crawl thêm thì mới so sánh được."
  │
  ▼
Message(respond.request_tool, {
    reason: "Need Amazon pricing data to compare with eBay",
    suggested_skill: "data_provider",
    tool_params: {"platform": "amazon", "query": "iPhone 15", "max_items": 100},
})
  │
  ▼
Orchestrator: approve? (see §6.2)
  │
  ▼  YES
Orchestrator → SkillPipeline.build_plan("data_provider", ...)
  → Executor crawls Amazon → task.result
  │
  ▼
Message(respond.tool_result, {
    request_id: UUID("..."),
    result: TaskResultSummary(
        skill_name="data_provider",
        summary="Amazon: 500 listings, median $310, free shipping phổ biến.",
        key_metrics={"median": 310, "count": 500},
    ),
})
  │
  ▼
Responder: "So sánh: eBay median $325, Amazon median $310 (rẻ hơn 5%). 
           Nhưng eBay có chính sách trả góp 12 tháng 0% APR. 
           Với hồ sơ của bạn (budget-conscious, thích trả góp) → eBay hợp lý hơn."
  │
  ▼
Message(respond.reply, answer=..., confidence=0.85)
```

### 6.2 Orchestrator Approval Gate

Responder CANNOT spawn Executors directly. Every `respond.request_tool` goes through Orchestrator's approval:

| Condition | Orchestrator Action |
|-----------|-------------------|
| Requested skill exists in registry | ✅ Approve |
| Requested skill is cheap (timeout < 10s) | ✅ Approve silently |
| Requested skill is expensive (crawl, deep analysis) | ⚠️ Raise `interrupt` to ask human: "Responder wants to crawl Amazon (~2 min). Allow?" |
| Requested skill not found in registry | ❌ Reject, tell Responder skill unavailable |
| Same skill already ran in this session for same params | ⚠️ Return cached result, don't re-run |
| 3+ tool requests in the same conversation turn | ⚠️ Warn: "Responder is requesting many tools. Continue?" |

**Default:** human must approve expensive operations. Cheap/fast operations auto-approved.

---

## 7. Persona System

### 7.1 Available Personas

| Persona | System Prompt Snippet | Best For |
|---------|----------------------|----------|
| `analyst` | "Bạn là chuyên gia phân tích dữ liệu. Trả lời chính xác, dẫn số liệu cụ thể từ kết quả đã thu thập. Không suy đoán." | Pricing analysis, market research |
| `advisor` | "Bạn là cố vấn mua sắm. Đưa lời khuyên thực tế, cân nhắc pros/cons, dựa trên ngân sách và nhu cầu người dùng." | Purchase decisions |
| `friend` | "Bạn là trợ lý thân thiện. Giải thích đơn giản, dễ hiểu, dùng ngôn ngữ đời thường." | Casual Q&A, beginners |
| `auto` | Tự chọn persona phù hợp nhất dựa trên loại câu hỏi (default) | General purpose |

### 7.2 How It Works

1. **Default:** `Persona.AUTO` — Responder detects question type and self-selects persona.
2. **User override:** "Trả lời kiểu chuyên gia đi" → Orchestrator sets persona for current turn.
3. **Profile persistence:** User's preferred persona is saved in `UserProfile.persona_preference` and loaded next session.

```python
# Orchestrator sends persona in respond.query
Message(respond.query, payload={
    "question": "...",
    "session_context": {...},
    "persona": Persona.ANALYST,    # override for this turn
})
```

### 7.3 Persona Switching Mid-Session

Responder can suggest persona switches:
```
User: "Nói đơn giản thôi, tôi không rành"
Responder (auto-detect): Chuyển sang persona "friend" cho phần còn lại của session
```

---

## 8. Streaming Responses

### 8.1 Why Stream

Conversational AI feels more natural when text appears token-by-token rather than all at once. Streaming reduces perceived latency and lets the user interrupt mid-response.

### 8.2 How It Works

Responder sends multiple `respond.reply` messages — one per chunk:

```python
# Non-streaming (v0 fallback)
Message(respond.reply, payload={
    "answer": "Giá $325 là mức median — nghĩa là một nửa số listing rẻ hơn, một nửa đắt hơn...",
    "stream": False,
    "confidence": 0.85,
})

# Streaming (v0 default)
Message(respond.reply, payload={    # chunk 1
    "chunk_index": 0,
    "chunk_content": "Giá $325 là mức median — ",
    "stream": True,
    "done": False,
})
Message(respond.reply, payload={    # chunk 2
    "chunk_index": 1,
    "chunk_content": "nghĩa là một nửa số listing rẻ hơn, một nửa đắt hơn...",
    "stream": True,
    "done": False,
})
Message(respond.reply, payload={    # final chunk
    "chunk_index": 2,
    "chunk_content": "",
    "stream": True,
    "done": True,
    "final_answer": "Giá $325 là mức median...",   # full text for logging
    "confidence": 0.85,
})
```

**Orchestrator behavior:**
- Assembles chunks in order by `chunk_index`
- Forwards chunks to UI/human as they arrive
- On `done: True`, stores the full `final_answer` in conversation history

**Fallback:** If streaming is not supported (e.g., notification via Telegram), send single non-streaming `respond.reply`.

### 8.3 Interrupt During Streaming

User can interrupt mid-stream: "Dừng lại, tôi hiểu rồi" → Orchestrator kills the in-progress Responder's current generation, but Responder instance stays alive. Next `respond.query` resumes normally.

---

## 9. Memory Across Sessions

### 9.1 User Profile

Stored at `~/.llend/user_profile.json`:

```json
{
  "preferred_platforms": ["ebay", "amazon"],
  "budget_conscious": true,
  "favorite_categories": ["electronics", "furniture"],
  "persona_preference": "advisor",
  "custom_notes": {
    "shipping_preference": "Miễn phí ship hoặc dưới $10",
    "avoid_sellers": "Người bán mới (<10 feedback)"
  },
  "last_updated": "2026-06-27T14:30:00Z"
}
```

### 9.2 How Profile Is Built

| Source | What's Learned |
|--------|---------------|
| User explicitly states | "Tôi thích mua trên eBay hơn" → `preferred_platforms: ["ebay"]` |
| Behavior patterns | User consistently picks cheapest option → `budget_conscious: true` |
| Session summaries | After each session, Orchestrator extracts preference signals |

```python
# After session completes, Orchestrator updates profile:
profile = load_profile()
profile.preferred_platforms = list(set(profile.preferred_platforms + ["ebay"]))
profile.budget_conscious = True  # user chose cheapest option 3/3 times
save_profile(profile)
```

### 9.3 Profile Injection

At session start, `UserProfile` is loaded and included in the first `respond.query`'s `SessionContext`. Responder uses it to personalize answers without being told:

```
User: "Tìm cho tôi cái bàn làm việc"
Responder (sees profile): "Tôi sẽ tìm trên eBay trước (nền tảng bạn hay dùng). 
                          Budget của bạn khoảng bao nhiêu?"
```

---

## 10. Integration with Spec 001 & Spec 002

### 10.1 New Message Routing

Extend Spec 001's routing diagram:

```
Executor ──→ Orchestrator ──→ Reviewer
                │
                ├──→ Responder          ← NEW
                │       │
                │       └──→ respond.request_tool → Orchestrator → Executor
                │
                ├──→ Interrupt (human)
                │
                └──→ Executor (re-spawn on review fail)
```

### 10.2 task.dispatch for Responder-Requested Tools

When Responder requests a tool, Orchestrator uses SkillPipeline (Spec 002 §7) to build the execution plan, then dispatches normally. The only difference: the resulting `TaskResultSummary` is routed back to Responder via `respond.tool_result`, not to the main session output.

### 10.3 Responder's Action Access

Responder does NOT have `ActionDispatcher` (Spec 002 §4.4). It cannot call actions directly. All tool access goes through:

```
Responder → respond.request_tool → Orchestrator → SkillPipeline → Executor
```

This preserves the hub-and-spoke model and ensures Orchestrator always knows what's happening.

---

## 11. Walkthrough

### 11.1 Full Session with Responder

```
1. Human: "Phân tích giá bàn làm việc trên eBay"

2. Runtime spawns Orchestrator (Spec 001)
   → Orchestrator spawns Responder (lives entire session)
   → SessionContext initialized with session_goal

3. Orchestrator classifies: task request → SkillPipeline.build_plan("analyze_pricing")
   → ExecutionPlan: [data_provider(eBay, "bàn làm việc"), analyze_pricing]

4. Executor #1: data_provider → crawl eBay → 500 listings → task.result
   → Orchestrator summarizes → TaskResultSummary

5. Executor #2: analyze_pricing → median $150, normal range $80-$250, 5 scam listings
   → Reviewer passes → Orchestrator summarizes → TaskResultSummary

6. Orchestrator reports to human: "Hoàn thành. Median $150. Có 5 listing nghi ngờ scam dưới $50."

7. Human: "Có nên mua cái $120 không? Thấy có trả góp 12 tháng."

8. Orchestrator classifies: conversational → respond.query
   → Payload: question + SessionContext (task results + empty conversation_history)

9. Responder (persona: advisor, sees user_profile.budget_conscious=True):
   "Giá $120 nằm trong normal range, thấp hơn median $150 → deal tốt.
    Trả góp 12 tháng: nếu 0% APR thì chỉ ~$10/tháng, rất hợp lý.
    Tuy nhiên, kiểm tra kỹ:
    - Seller có rating >95% không?
    - Phí ship có bị độn lên không?
    Bạn có muốn tôi kiểm tra chi tiết listing này không?"

10. Human: "Có, kiểm tra giúp tôi"

11. Responder: respond.request_tool(
        reason="Kiểm tra chi tiết listing #42 (giá $120)",
        suggested_skill="data_provider",
        tool_params={"platform": "ebay", "listing_id": 42},
    )

12. Orchestrator: tool is fast (<10s) → auto-approve → dispatch Executor
    → Executor returns listing detail → Orchestrator summarizes → respond.tool_result

13. Responder:
    "Listing #42:
     - Seller: 'giatot24h' (rating 98.7%, 2,340 feedback) → đáng tin
     - Ship: $15 (hơi cao so với trung bình $8-12)
     - Trả góp: 12 tháng, 0% APR qua PayPal Credit
     - Tổng: $120 + $15 ship = $135
     
     Kết luận: ĐÁNG MUA nếu bạn cần trả góp. Lưu ý deal $125 free ship từ seller khác 
     có thể rẻ hơn nếu bạn trả thẳng. Bạn muốn tôi so sánh không?"

14. Human: "Ok cảm ơn, tôi chốt cái này"

15. Orchestrator → session.complete
    → Update user_profile (preferred_platforms + ["ebay"], budget_conscious=True)
    → Session ends
```

---

## 12. File Layout (After Spec 003)

```
llend_harness/
├── runtime/              # Spec 001
│   └── ...
├── skills/               # Spec 002
│   └── ...
├── registry/             # Spec 002
│   └── ...
├── tool_bridge/          # Spec 002
│   └── ...
├── responder/            # Spec 003 — NEW
│   ├── __init__.py
│   ├── agent.py          # ResponderAgent class (extends BaseAgent)
│   ├── context.py        # SessionContext, ConversationTurn, TaskResultSummary
│   ├── persona.py        # Persona enum, persona prompts
│   ├── memory.py         # UserProfile load/save, preference extraction
│   └── stream.py         # Streaming chunk assembly
├── bootstrap/            # Future spec
├── telemetry/            # Future spec
├── testing/              # Future spec
├── plugins/              # Future spec
└── docs/
    └── specs/
        ├── 001-message-protocol-runtime-core.md
        ├── 002-skill-format-registry.md
        └── 003-responder-agent-conversation-module.md   # ← THIS SPEC
```

---

## 13. Decisions & Resolved Questions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Separate agent type or Orchestrator capability? | Separate agent (Responder) | Separation of concerns. Orchestrator stays lean. Responder has its own system prompt, context window, and lifecycle. |
| Spawn per question or reuse? | Reuse (lives entire session) | Conversation needs continuity. History and rapport accumulate. |
| Can Responder call tools directly? | No — always through Orchestrator | Preserves hub-and-spoke model. Orchestrator gates expensive operations. |
| Streaming: v0 or v1? | v0, default on | Feels natural. Non-streaming fallback for constrained channels. |
| Memory: this spec or separate? | This spec (lightweight) | User profile is a JSON file. Simple enough to include now. Heavyweight memory (vector DB) is future. |
| Persona system: this spec or separate? | This spec | Only 3 explicit personas + auto. Small enough. |

---

## 14. Open Questions

- **Q1:** Responder model selection — should Responder use a cheaper/faster LLM than Executor (since it doesn't need to crawl/analyze, just converse)? → **Defer to implementation.** Default: same model, configurable.
- **Q2:** Multi-language support? User speaks Vietnamese, data is in English. Should Responder auto-detect and match the user's language? → **Yes, default behavior.** Responder matches the language of the question.
- **Q3:** Should Responder be able to see *in-progress* Executor output? E.g., user asks "Đang crawl đến đâu rồi?" while Executor is still running. → **Defer to Spec 004.** Requires progress reporting from Executor.
- **Q4:** Conversation history persistence — save full history to disk between sessions? → **Defer.** v0 = in-memory only (dies with session). v1 = conversation log written to session output dir.
- **Q5:** Should Responder ever be mandatory? Can user disable it and just use Executor pipeline? → **Configurable.** `settings.toml`: `responder.enabled = true/false`. If disabled, Orchestrator answers simple questions itself or returns "I can only execute tasks."

---

*Next spec: 004 — Orchestrator Logic & Session Orchestration*
