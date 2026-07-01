# Spec 004: Orchestrator Logic & Session Orchestration

**Status:** Draft
**Date:** 2026-07-01
**Author:** Human + Claude
**Depends on:** [Spec 001 — Message Protocol & Runtime Core](./001-message-protocol-runtime-core.md), [Spec 002 — Skill Format & Registry](./002-skill-format-registry.md), [Spec 003 — Responder Agent & Conversation Module](./003-responder-agent-conversation-module.md)

---

## 1. Motivation

Spec 001 defines the runtime (spawn, send, kill, interrupt). Spec 002 defines skills and their resolution. Spec 003 defines the Responder for conversational Q&A. But **nothing ties them together yet.**

The Orchestrator is the "brain" — the only long-lived agent that:

- Receives the user's request
- Decides whether it's a task or a conversational question
- Builds an execution plan via SkillPipeline
- Spawns Executor → Reviewer loops per task
- Adjudicates verdicts (pass → next, fail → re-do)
- Summarizes results for the Responder
- Relays interrupts to the human
- Gates Responder's tool requests
- Manages the session from start to completion

Without the Orchestrator, all the components built so far are a car engine without a steering wheel — powerful, but going nowhere.

---

## 2. Orchestrator Agent

### 2.1 Role & Mindset

| | Orchestrator | Executor | Reviewer | Responder |
|---|---|---|---|---|
| **Role** | Plan, dispatch, adjudicate, synthesize | Execute 1 task | Adversarially verify | Converse, advise, explain |
| **Mindset** | Manager — balance quality, cost, and time | Constructor — build the output | Skeptic — find flaws | Advisor — help the human |
| **Lifespan** | Entire session | Per task | Per task | Entire session |
| **Input** | User request, agent outputs, verdicts | `task.dispatch` | `task.review` | `respond.query` |
| **Output** | Execution plan, dispatch messages, final synthesis | `task.result` | `task.verdict` | `respond.reply` |
| **State** | Holds session state, plan progress, conversation history | Stateless | Stateless | Maintains conversation history |

### 2.2 Lifecycle

```
Session Start
     │
     ▼
Orchestrator INIT ──→ RUNNING
     │                    │
     │                    ├── Spawn Responder (once)
     │                    │
     │                    ├── User message arrives
     │                    │   ├── classify → task
     │                    │   │   └── build plan → execute loop
     │                    │   └── classify → conversational
     │                    │       └── route to Responder
     │                    │
     │                    ├── Interrupt raised by Executor/Reviewer
     │                    │   └── relay to human → wait → resume agent
     │                    │
     │                    └── Session goal achieved (or human says "done")
     │                        └── COMPLETE → synthesize → DEAD
     │
     └── ERROR (crash) → attempt recovery → DEAD
```

Orchestrator uses the same lifecycle states as all agents (Spec 001 §3.3): `INIT → RUNNING → (INTERRUPT)* → COMPLETE/ERROR → DEAD`.

### 2.3 Orchestrator vs Other Agents

Orchestrator is the **only agent that spawns other agents**. It does NOT call tools/actions directly — it delegates all work to Executors. It does NOT verify output quality — it delegates that to Reviewers. Its job is **coordination**, not execution.

---

## 3. Message Classification

### 3.1 The Classification Problem

When a human message arrives, the Orchestrator must decide: is this a task to execute, or a question to answer?

| Human says | Classification | Action |
|------------|---------------|--------|
| "Phân tích giá iPhone 15 trên eBay" | **task** | Build plan → dispatch Executor pipeline |
| "Crawl thêm Amazon để so sánh" | **task** | Build plan → dispatch |
| "Giá $325 thì có nên mua không?" | **conversational** | Route to Responder |
| "Giải thích tại sao median khác mean?" | **conversational** | Route to Responder |
| "Cảm ơn, tạm biệt" | **session_end** | Complete session |
| "Dừng task đang chạy đi" | **control** | Cancel active task, notify |

### 3.2 Classification Logic

```
User message
     │
     ▼
┌─────────────────────────┐
│ LLM Classification Call │  ← cheap, fast model (e.g. Haiku)
│ (structured output)     │
│                         │
│ Classify as:            │
│ - "task"                │  → has actionable verb + clear deliverable
│ - "conversational"      │  → asks for opinion, explanation, advice
│ - "session_end"         │  → farewell, "done", "thanks bye"
│ - "control"             │  → cancel, pause, status check
└───────────┬─────────────┘
            │
            ▼
     Route accordingly
```

### 3.3 Classification Prompt

```
You are a message classifier for an AI agent harness.

Given a user message, classify it into exactly ONE category:
- "task": The user wants an ACTION performed (crawl, analyze, export, search, compare).
           These map to skills in the registry.
- "conversational": The user wants an OPINION, EXPLANATION, ADVICE, or FOLLOW-UP question.
                    These do NOT require running a skill.
- "session_end": The user is saying goodbye or indicating the session is done.
- "control": The user wants to control the session itself (cancel, pause, status).

User message: "{message}"

Respond with JSON: {"category": "...", "confidence": 0.0-1.0, "reasoning": "..."}
```

### 3.4 Routing Table

| Category | Route To | Message Type |
|----------|----------|-------------|
| `task` | SkillPipeline → Executor loop | `task.dispatch` |
| `conversational` | Responder | `respond.query` |
| `session_end` | Session completion flow | `session.complete` |
| `control` | Orchestrator internal handler | (cancel/pause/status) |

---

## 4. Task Execution Loop

### 4.1 The Core Pattern

This is the heart of the Orchestrator. For each task in the execution plan:

```
                    ┌─────────────────────────────┐
                    │     Orchestrator             │
                    │                              │
                    │  1. Build skill_context      │
                    │  2. Spawn Executor           │
                    │  3. Send task.dispatch       │
                    │  4. Wait for task.result      │
                    │  5. Validate output schema   │
                    │  6. Spawn Reviewer           │
                    │  7. Send task.review         │
                    │  8. Wait for task.verdict    │
                    │                              │
                    │  ┌── pass ──→ next task      │
                    │  │                           │
                    │  └── fail ──→ re-spawn        │
                    │       Executor (max N times) │
                    └─────────────────────────────┘
```

### 4.2 Step-by-Step

```
Step 1: Build skill_context
  → Call SkillRegistry.resolve(skill_name)
  → Extract: skill_md, allowed_actions, action_bindings, output_schema, enforcement

Step 2: Spawn Executor
  → Runtime.spawn("executor", context={skill_context, task_spec})
  → Executor INIT → RUNNING

Step 3: Send task.dispatch
  → Message(msg_type=TASK_DISPATCH, payload={task_id, skill_name, task_spec, skill_context})
  → Wait for response (with timeout)

Step 4: Wait for task.result
  → Executor sends Message(msg_type=TASK_RESULT, payload={output, concerns?, artifacts?})
  → On timeout → kill Executor → retry or fail task

Step 5: Validate output schema
  → If output_schema exists:
      try: SkillOutputModel.model_validate(executor_output)
      except ValidationError:
        → enforcement=mandatory: retry (max 3) with error feedback
        → enforcement=strict: retry (max 2) then send to Reviewer with issues
        → enforcement=suggested: send to Reviewer as-is, note schema violations
  → If no output_schema (primitive output):
      → Send to Reviewer for quality check

Step 6-8: Reviewer Cycle
  → Spawn Reviewer (fresh context)
  → Message(msg_type=TASK_REVIEW, payload={
        task_id, original_task_spec, executor_output,
        concerns_from_executor?, schema_validation_issues?
    })
  → Wait for task.verdict
  → ReviewIssue model (Spec 001 §2.2.1): severity, field, message
```

### 4.3 Adjudication Logic

```
Reviewer returns task.verdict
     │
     ├── Verdict.PASS
     │   → Task complete. Summarize → TaskResultSummary.
     │   → Move to next task in plan.
     │
     ├── Verdict.PASS_WITH_WARNINGS
     │   → Task complete with noted issues.
     │   → Attach warnings to TaskResultSummary.
     │   → Move to next task.
     │   → If accumulated warnings > threshold → notify human.
     │
     └── Verdict.FAIL
         → Check retry count for this task.
         → If retries < max_retries (default 3):
             → Build improved task_spec incorporating Reviewer's issues.
             → Spawn NEW Executor (fresh context, no bias from prior attempt).
             → Go back to Step 2.
         → If retries exhausted:
             → Mark task as FAILED.
             → If enforcement=mandatory → abort entire session, notify human.
             → If enforcement=strict → skip task, continue plan with warning.
             → If enforcement=suggested → skip task, continue plan.
```

### 4.4 Max Retries Per Enforcement Level

| Enforcement | Max Retries | On Exhaustion |
|-------------|-------------|---------------|
| `mandatory` | 5 | Abort session. Notify human: "Critical task X failed after 5 attempts." |
| `strict` | 3 | Skip task. Continue plan. Attach failure note to final output. |
| `suggested` | 1 | Skip task. Continue plan. Minimal logging. |

### 4.5 Reviewer Prompt Construction

The Orchestrator builds the Reviewer's system prompt adversarially:

```
You are a REVIEWER. Your job is to find flaws in the Executor's output.

Task: {task_spec}
Expected output schema: {output_schema}
Enforcement level: {enforcement}

Executor's output: {executor_output}
Executor's own concerns: {concerns}

Verify:
1. Does the output match the expected schema? (if schema exists)
2. Are the numbers/data consistent and reasonable?
3. Are there logical errors or unsupported claims?
4. Did the Executor address ALL requirements in the task spec?
5. [If concerns from Executor]: Verify each concern — confirm or dismiss.

For each issue found, provide:
- severity: "critical" | "important" | "minor"
- field: which part of the output
- message: what's wrong

Return your verdict:
- "pass": no issues found
- "pass_with_warnings": minor issues only, output is still usable
- "fail": critical or important issues make the output unreliable
```

---

## 5. Execution Plan Execution

### 5.1 Consuming SkillPipeline Output

```
User: "Phân tích giá iPhone 15 trên eBay"
     │
     ▼
Orchestrator classify → "task"
     │
     ▼
extract skill name + params via LLM:
  → skill_name: "analyze_pricing"
  → params: {target_item: "iPhone 15", platform: "ebay"}
     │
     ▼
SkillPipeline.build_plan("analyze_pricing", params)
  → ExecutionPlan(
      skills=[
        TaskSpec(step=1, skill_name="data_provider", ...),
        TaskSpec(step=2, skill_name="analyze_pricing", ...),
      ],
      terminal_skill="analyze_pricing",
    )
     │
     ▼
Iterate plan.skills in order:
  for each TaskSpec → execute loop (§4) → TaskResultSummary
     │
     ▼
All tasks complete → synthesize final output → respond to human
```

### 5.2 Skill Name & Param Extraction

The Orchestrator uses an LLM call to extract the target skill and parameters from the user's natural-language request:

```
You are a task parser. Given a user request and a list of available skills,
extract the target skill and its parameters.

Available skills:
{skill_listings}

User request: "{message}"

Respond with JSON:
{
  "skill_name": "...",        // must match an available skill
  "params": {...},            // skill-specific parameters
  "confidence": 0.0-1.0
}
```

If no skill matches → Orchestrator tells the user what skills are available and asks for clarification.

### 5.3 Sequential Execution (Default)

Tasks execute sequentially by default. Each task's output is available to the next task via `input_from` wiring.

### 5.4 Parallel Execution

When `TaskSpec.parallelizable = True` for multiple tasks at the same depth with no interdependency, the Orchestrator MAY run them concurrently:

```python
async def execute_plan(plan: ExecutionPlan) -> list[TaskResultSummary]:
    results: list[TaskResultSummary] = []
    i = 0
    while i < len(plan.skills):
        # Collect parallelizable batch
        batch = [plan.skills[i]]
        while i + 1 < len(plan.skills) and plan.skills[i + 1].parallelizable:
            batch.append(plan.skills[i + 1])
            i += 1
        i += 1
        # Execute batch concurrently
        batch_results = await asyncio.gather(*[execute_task(ts) for ts in batch])
        results.extend(batch_results)
    return results
```

**Guard:** Parallel execution is gated by a config flag (`execution.allow_parallel = true`). Default: `false` (sequential only) for v0.

---

## 6. Input/Output Wiring

### 6.1 The Problem

`analyze_pricing` declares `inputs: dataset:list[dict]`, but `data_provider` outputs `ScrapeResult` (a Pydantic model wrapping `list[ProductListing]`). How does the Orchestrator connect them?

### 6.2 Auto-Unwrap Convention

When wiring upstream output → downstream input:

1. **Exact match**: If upstream output type name == downstream input type name → pass directly.
2. **Wrapper unwrap**: If upstream output is a Pydantic model with a **single list field** → unwrap it.
   - `ScrapeResult.listings: list[ProductListing]` → unwrap to `list[dict]` for downstream `dataset: list[dict]`.
3. **Named field match**: If upstream output has a field whose name matches the downstream input name → extract that field.
   - `{dataset_ref: "data_provider"}` → Orchestrator knows `data_provider.output_as = "dataset"` → extracts `ScrapeResult.listings` as `dataset`.
4. **Pass-through**: If none of the above → pass the entire upstream output object. Downstream Executor is responsible for extracting what it needs.

### 6.3 Wiring in Practice

```python
# During plan execution, after task N completes:
upstream_skill = registry.get(upstream_name)  # e.g. data_provider
upstream_output = task_result.output           # ScrapeResult instance

for downstream_task in plan.skills:
    if upstream_name in (downstream_task.input_from or []):
        ref_name = f"{upstream_name}_ref"      # "data_provider_ref"
        # Auto-unwrap: ScrapeResult → list[ProductListing]
        if has_single_list_field(upstream_output):
            wired_data = get_list_field(upstream_output)
        else:
            wired_data = upstream_output
        downstream_task.task_spec[ref_name] = wired_data
```

### 6.4 Type Coercion

If downstream expects `list[dict]` but receives `list[ProductListing]` (Pydantic models), the Orchestrator calls `.model_dump()` on each item:

```python
def coerce_to_expected_type(data: Any, expected_type: str) -> Any:
    """Best-effort type coercion for downstream consumption."""
    if expected_type == "list[dict]" and isinstance(data, list):
        return [
            item.model_dump() if isinstance(item, BaseModel) else item
            for item in data
        ]
    return data  # pass-through
```

---

## 7. Context Summarization

### 7.1 Why Summarize

After each task completes, the Orchestrator generates a `TaskResultSummary` (Spec 003 §3.1). This serves two purposes:

1. **Responder context**: Responder gets summaries, not raw data (Spec 003 §5.2).
2. **Session memory**: Orchestrator accumulates summaries to track what's been done.

### 7.2 Summarization Logic

```python
async def summarize_task_result(
    task_result: Message,      # task.result payload
    skill: Skill,              # resolved skill
    task_spec: TaskSpec,       # from execution plan
) -> TaskResultSummary:
    """Generate a 1-3 sentence summary + key metrics from a task result."""

    # LLM call (cheap, fast model):
    prompt = f"""
    Summarize this task execution result in 1-3 sentences.
    Extract up to 5 key numeric metrics.

    Skill: {skill.name} — {skill.description}
    Task params: {task_spec.task_spec}
    Output: {json.dumps(task_result.payload.get("output", {}))[:2000]}

    Respond with JSON:
    {{
      "summary": "...",
      "key_metrics": {{"metric_name": value, ...}},
      "notable_findings": ["..."]  // optional, empty list if none
    }}
    """

    # ... LLM call ...
    return TaskResultSummary(
        task_id=task_result.payload["task_id"],
        skill_name=skill.name,
        status=task_result.payload.get("status", "done"),
        summary=llm_response["summary"],
        key_metrics=llm_response["key_metrics"],
        artifact_paths=task_result.payload.get("artifacts", []),
    )
```

### 7.3 Session Context Accumulation

```python
class SessionState:
    """Orchestrator's in-memory session state."""
    session_id: UUID
    session_goal: str
    plan: ExecutionPlan | None = None
    completed_tasks: list[TaskResultSummary] = []
    active_task: TaskResultSummary | None = None
    conversation_history: list[ConversationTurn] = []
    accumulated_warnings: list[str] = []
    artifacts: list[Artifact] = []
```

---

## 8. Interrupt Propagation

### 8.1 Orchestrator as Interrupt Relay

Any agent (Executor, Reviewer, Responder) can raise an interrupt (Spec 001 §3.4). The Orchestrator is the relay:

```
Executor/Reviewer/Responder
     │
     │  interrupt.raise
     ▼
Orchestrator
     │
     ├── Evaluate: can Orchestrator decide this?
     │   ├── Yes → auto-respond (don't bother human)
     │   └── No → relay to human
     │
     ├── Forward to human channel (Telegram, Discord, Web UI)
     ├── Wait for human response (with TTL)
     │
     └── Send interrupt.response back to agent
         → Agent resumes execution
```

### 8.2 Auto-Response Heuristics

Some interrupts don't need human input. The Orchestrator can auto-respond:

| Interrupt Pattern | Auto-Response |
|-------------------|---------------|
| "Continue with default?" | `yes` (use default) |
| "Retry after error?" (attempt < max) | `yes` (retry) |
| "Proceed with {only_one_option}?" | `yes` |
| Cost/time estimate within budget | `yes` (auto-approve) |
| Cost/time estimate exceeds budget | Relay to human |
| Ambiguous choice (3+ options) | Relay to human |

```python
async def handle_interrupt(self, msg: Message) -> None:
    """Evaluate and potentially auto-resolve an interrupt."""
    payload = msg.payload
    options = payload.get("options", [])

    if len(options) == 1:
        # Only one option → auto-select
        await self._resolve_interrupt(msg.sender_instance, options[0], "auto")
    elif self._is_trivial_decision(payload):
        await self._resolve_interrupt(msg.sender_instance, options[0], "auto")
    else:
        # Relay to human
        await self._relay_to_human(msg)
```

### 8.3 Interrupt During Streaming

If human interrupts Responder mid-stream (Spec 003 §8.3): Orchestrator kills the in-progress generation but keeps the Responder instance alive. The next `respond.query` resumes normally.

---

## 9. Responder Tool Approval Gate

### 9.1 The Flow (from Spec 003 §6.2)

When Responder sends `respond.request_tool`, the Orchestrator acts as the approval gate:

```
Responder → respond.request_tool
     │
     ▼
Orchestrator Approval Gate
     │
     ├── Skill exists in registry?
     │   ├── No → Reject. Tell Responder: "Skill X not available."
     │   └── Yes → continue
     │
     ├── Is it cheap? (timeout < 10s, no crawl/scrape)
     │   ├── Yes → Auto-approve. Dispatch Executor silently.
     │   └── No → continue
     │
     ├── Already ran with same params this session?
     │   ├── Yes → Return cached TaskResultSummary.
     │   └── No → continue
     │
     ├── 3+ tool requests in this conversation turn?
     │   ├── Yes → Raise interrupt: "Responder wants many tools. Allow?"
     │   └── No → continue
     │
     └── Expensive operation → Raise interrupt to human:
         "Responder wants to {skill.description}. Estimated cost: {cost}.
         [A] Allow  [B] Deny  [C] Allow with limits (specify)"
```

### 9.2 Cheap vs Expensive Heuristic

| Criteria | Classification |
|----------|---------------|
| `timeout_ms < 10000` (from ActionBinding) | Cheap |
| Skill has no `fetch_web_page` or `crawl*` actions | Cheap |
| Skill is `data_provider` with `platform=*` | Expensive |
| Skill has `max_items > 100` | Expensive |
| Default | Cheap (opt-in to expensive) |

---

## 10. Error Recovery

### 10.1 Error Types & Responses

| Error | Detection | Response |
|-------|-----------|----------|
| **Executor crash** | `agent.error(CRASH)` or timeout | Re-spawn Executor (max 2). On exhaustion → mark task FAILED. |
| **Reviewer crash** | `agent.error(CRASH)` or timeout | Re-spawn Reviewer (max 2). On exhaustion → auto-pass with warning. |
| **LLM API error** | `agent.error(LLM_ERROR)` | Exponential backoff (1s, 2s, 4s, 8s). Max 4 retries. On exhaustion → escalate to human. |
| **Tool error** | `agent.error(TOOL_ERROR)` | Executor handles internally (ActionDispatcher retry). If Executor reports failure → Reviewer evaluates partial output. |
| **Validation error** | `agent.error(VALIDATION_ERROR)` | Retry with schema feedback (see §4.2 step 5). |
| **Interrupt timeout** | `agent.error(INTERRUPT_TIMEOUT)` | Mark task as INCOMPLETE. Continue plan if possible. Notify human. |
| **Orchestrator crash** | Runtime detects | Attempt restart from last checkpoint (Spec 001 §3.4). Session state recovered from disk. |

### 10.2 Exponential Backoff for LLM Errors

```python
async def _with_llm_retry(self, callable, max_retries=4, base_delay=1.0):
    """Wrap an LLM call with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await callable()
        except LLMError:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning("LLM error, retrying in %.1fs (attempt %d/%d)", delay, attempt+1, max_retries)
            await asyncio.sleep(delay)
```

### 10.3 Graceful Degradation

If a non-mandatory task fails after all retries:

1. Skip the task.
2. Note the failure in `SessionState.accumulated_warnings`.
3. Continue the plan with remaining tasks.
4. In final synthesis, mention: "Note: {skill_name} could not be completed. {reason}."
5. If the failed task was a dependency of downstream tasks → those downstream tasks are also skipped.

---

## 11. Session Lifecycle

### 11.1 Start

```
Runtime starts
     │
     ▼
Orchestrator.spawn("orchestrator", {session_goal, user_profile})
     │
     ├── Load UserProfile from ~/.llend/user_profile.json
     ├── Spawn Responder (if responder.enabled)
     ├── Send session.start to Runtime
     └── Ready for human input
```

### 11.2 During Session

```
Loop:
  ├── Wait for human message
  ├── Classify (§3)
  ├── Route to Executor pipeline OR Responder
  ├── Handle interrupts if raised
  └── Accumulate results in SessionState
```

### 11.3 Complete

```
Human says "done" / session goal achieved
     │
     ▼
Orchestrator COMPLETE sequence:
  ├── Cancel any running tasks (with grace period)
  ├── Generate final synthesis (LLM call):
  │     "Here's what we found: {summary of all tasks}.
  │      Key takeaways: {...}.
  │      Artifacts: {list of files}."
  ├── Save artifacts to session output directory
  ├── Update UserProfile (preference extraction)
  ├── Kill Responder
  ├── Send session.complete to Runtime
  └── Transition to DEAD
```

### 11.4 Final Synthesis

The Orchestrator generates a human-readable summary of the entire session:

```
You are a session synthesizer. Given the completed tasks and conversation,
write a clear summary of what was accomplished.

Session goal: {session_goal}

Completed tasks:
{for each TaskResultSummary:
  - {skill_name}: {summary}
    Key metrics: {key_metrics}
}

Conversation highlights:
{last 5 conversation turns}

Write a summary that:
1. Answers the original session goal directly
2. Highlights the most important findings
3. Mentions any limitations or skipped tasks
4. Suggests next steps if applicable
5. Lists all generated artifact files
```

---

## 12. Progress Reporting

### 12.1 What the Human Sees

While tasks execute, the Orchestrator emits progress updates:

```
🔄 Building execution plan...
📋 Plan: data_provider → analyze_pricing (2 tasks)

[1/2] 🔄 Crawling eBay for "iPhone 15"... (data_provider)
[1/2] ✅ Crawled 500 listings. Median price: $325.

[2/2] 🔄 Analyzing pricing... (analyze_pricing)
[2/2] ✅ Analysis complete. 8 suspicious listings detected.

📊 Session complete. Report saved to output/iphone15_report.xlsx.
```

### 12.2 Progress Events

Not formal `msg_type` values — these are internal Orchestrator events forwarded to the UI channel:

```python
class ProgressEvent:
    level: Literal["info", "task_start", "task_complete", "warning", "error"]
    message: str
    task_id: UUID | None = None
    step: tuple[int, int] | None = None  # (current, total)
```

### 12.3 In-Progress Status

When human asks "Đang làm gì đấy?" (Spec 003 §14 Q3), the Orchestrator responds with the current plan progress without needing to ask the Executor:

```
📋 Progress:
  [✅] data_provider — completed (500 listings)
  [🔄] analyze_pricing — running (elapsed: 45s)
  [⏳] write_report — waiting
```

---

## 13. Configuration

### 13.1 Orchestrator Settings

```toml
# llend/settings.toml (or ~/.llend/settings.toml)

[orchestrator]
# Model selection
classification_model = "claude-haiku-4-5-20251001"  # cheap & fast for classify
summarization_model = "claude-haiku-4-5-20251001"    # cheap for TaskResultSummary
synthesis_model = "claude-sonnet-5"                   # quality for final synthesis

[execution]
max_retries_mandatory = 5
max_retries_strict = 3
max_retries_suggested = 1
task_timeout_default = 300       # seconds (5 min)
review_timeout_default = 120     # seconds (2 min)
allow_parallel = false           # v0: sequential only

[responder]
enabled = true                   # can disable Responder entirely
tool_auto_approve_timeout_ms = 10000  # cheap threshold for auto-approval
max_tool_requests_per_turn = 3   # warn if Responder exceeds this

[session]
output_dir = "output"            # relative to cwd, or absolute
checkpoint_dir = "~/.llend/checkpoints"
max_session_duration = 3600      # seconds (1 hour) — safety kill-switch
```

### 13.2 User Profile

```json
// ~/.llend/user_profile.json
{
  "preferred_platforms": ["ebay"],
  "budget_conscious": true,
  "favorite_categories": ["electronics"],
  "persona_preference": "advisor",
  "custom_notes": {},
  "last_updated": "2026-07-01T00:00:00Z"
}
```

Loaded at session start, saved on session complete.

---

## 14. Integration Summary

### 14.1 How Everything Connects

```
                         ┌──────────────────────────────────────┐
                         │           Orchestrator               │
                         │   (this spec — the central hub)      │
                         └──────────┬───────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│ SkillRegistry │          │ SkillPipeline │          │   Responder   │
│ (Spec 002)    │          │ (Spec 002)    │          │ (Spec 003)    │
│               │          │               │          │               │
│ • resolve()   │          │ • build_plan()│          │ • answer Q&A  │
│ • validate()  │          │ • validate()  │          │ • request tool│
└───────┬───────┘          └───────┬───────┘          └───────┬───────┘
        │                          │                          │
        ▼                          ▼                          │
┌───────────────┐          ┌───────────────┐                  │
│  ToolBridge   │          │  (plan sent   │                  │
│  (Spec 002)   │          │   to execute  │                  │
│               │          │   loop below) │                  │
└───────────────┘          └───────┬───────┘                  │
                                   │                          │
                    ┌──────────────┴──────────────┐           │
                    │                             │           │
                    ▼                             ▼           │
            ┌─────────────┐              ┌─────────────┐      │
            │  Executor   │──output──▶  │  Reviewer   │      │
            │  (Spec 001) │              │  (Spec 001) │      │
            │             │◀─fail/retry─│             │      │
            │ ActionDisp. │              │ adversarial │      │
            │ (Spec 002)  │              │   verify    │      │
            └─────────────┘              └─────────────┘      │
                                   │                          │
                                   ▼                          │
                            ┌─────────────┐                   │
                            │  Interrupt  │◀──────────────────┘
                            │  (Spec 001) │  (tool requests
                            │  HITL node  │   go through Orch
                            └─────────────┘   approval gate)
```

### 14.2 Message Flow (Complete Session)

```
Human: "Phân tích giá iPhone 15"
  │
  ▼
Orchestrator classify → "task"
  │
  ▼
SkillPipeline.build_plan("analyze_pricing", {target_item: "iPhone 15"})
  → ExecutionPlan: [data_provider, analyze_pricing]
  │
  ▼
Task 1: data_provider
  ├── Spawn Executor #1
  ├── task.dispatch → Executor
  ├── Executor runs (ActionDispatcher calls crawl4ai)
  ├── task.result → Orchestrator
  ├── Validate output (ScrapeResult ✓)
  ├── Spawn Reviewer #1
  ├── task.review → Reviewer
  ├── task.verdict: PASS ✓
  └── Summarize → TaskResultSummary
  │
  ▼
Task 2: analyze_pricing
  ├── Wire dataset_ref = Task 1 output
  ├── Spawn Executor #2
  ├── task.dispatch → Executor
  ├── Executor runs (ActionDispatcher: calc median, detect outliers, segment, export CSV)
  ├── task.result → Orchestrator
  ├── Validate output (AnalysisReport ✓)
  ├── Spawn Reviewer #2
  ├── task.review → Reviewer
  ├── task.verdict: PASS ✓
  └── Summarize → TaskResultSummary
  │
  ▼
Session complete:
  ├── Final synthesis: "Median $325. 8 suspicious listings. Report saved."
  ├── Save artifacts
  ├── Update user profile
  └── session.complete → Runtime
```

---

## 15. Walkthrough: Full Session with Responder

```
1. Human: "Phân tích giá bàn làm việc trên eBay"

2. Runtime spawns Orchestrator
   → Orchestrator spawns Responder

3. Orchestrator classify: "task"
   → LLM extraction: skill="analyze_pricing", params={target_item: "bàn làm việc", platform: "ebay"}
   → SkillPipeline.build_plan("analyze_pricing", params)
   → Plan: [data_provider, analyze_pricing]

4. Execute Task 1 (data_provider):
   → Executor crawls eBay → 500 listings → task.result
   → Reviewer verifies → pass
   → Orchestrator summarizes → TaskResultSummary

5. Execute Task 2 (analyze_pricing):
   → Executor analyzes → median $150, normal $80-$250, 5 scam
   → Reviewer verifies → pass
   → Orchestrator summarizes → TaskResultSummary

6. Orchestrator synthesis: "Hoàn thành. Median $150. 5 listing nghi ngờ scam."

7. Human: "Có nên mua cái $120 không? Có trả góp 12 tháng."

8. Orchestrator classify: "conversational" → route to Responder
   → respond.query(question="...", session_context={task_summaries, conversation_history})

9. Responder (persona: advisor):
   "Giá $120 thấp hơn median $150 → deal tốt.
    Trả góp 12 tháng 0% APR → chỉ ~$10/tháng.
    Kiểm tra seller rating và phí ship trước khi chốt nhé."

10. Human: "Kiểm tra giúp listing #42"

11. Responder → respond.request_tool(
        reason="Check listing #42 details",
        suggested_skill="data_provider",
        tool_params={listing_id: 42}
    )

12. Orchestrator gate: cheap (timeout < 10s) → auto-approve
    → Dispatch Executor → result → respond.tool_result

13. Responder:
    "Listing #42: Seller 'giatot24h' (98.7%, 2340 feedback) → đáng tin.
     Ship $15 (hơi cao). Tổng $135.
     Kết luận: ĐÁNG MUA nếu cần trả góp."

14. Human: "Ok cảm ơn"

15. Orchestrator classify: "session_end"
    → Final synthesis
    → Update user_profile
    → session.complete
```

---

## 16. File Layout (After Spec 004)

```
llend/
├── runtime/              # Spec 001
│   ├── __init__.py
│   ├── base.py           # AgentRuntime ABC
│   ├── asyncio_runtime.py
│   ├── langgraph_runtime.py
│   ├── message.py        # Message, MsgType, enums
│   ├── lifecycle.py      # AgentState, AgentType
│   ├── checkpoint.py     # Checkpoint persistence
│   └── notifications.py  # Notification channels
├── skills/               # Spec 002
│   ├── data_provider/
│   │   ├── skill.md
│   │   ├── handler.py
│   │   └── models.py
│   └── analyze_pricing/
│       ├── skill.md
│       ├── handler.py
│       └── models.py
├── registry/             # Spec 002
│   ├── __init__.py
│   ├── registry.py       # SkillRegistry
│   ├── models.py         # SkillMeta, Skill, ActionBinding
│   ├── pipeline.py       # SkillPipeline, TaskSpec, ExecutionPlan
│   ├── validator.py      # Skill validation
│   ├── parser.py         # Input parser
│   └── action_dispatcher.py  # ActionDispatcher
├── tool_bridge/          # Spec 002
│   ├── __init__.py
│   ├── bridge.py         # ToolBridge
│   └── mappings.toml
├── responder/            # Spec 003
│   ├── __init__.py
│   ├── agent.py          # ResponderAgent
│   ├── context.py        # SessionContext, ConversationTurn, TaskResultSummary
│   ├── persona.py        # Persona enum, prompts
│   ├── memory.py         # UserProfile load/save
│   └── stream.py         # Streaming chunk assembly
├── orchestrator/         # Spec 004 — NEW
│   ├── __init__.py
│   ├── agent.py          # OrchestratorAgent (extends BaseAgent, runs the main loop)
│   ├── classifier.py     # Message classification (task vs conversational)
│   ├── executor.py       # Task execution loop (Executor → Reviewer → adjudicate)
│   ├── adjudicator.py    # Verdict adjudication + retry logic
│   ├── summarizer.py     # TaskResultSummary + final synthesis
│   ├── wiring.py         # Input/output wiring + auto-unwrap
│   ├── gate.py           # Responder tool approval gate
│   ├── recovery.py       # Error recovery + exponential backoff
│   ├── progress.py       # Progress reporting
│   ├── session.py        # Session state, start/complete lifecycle
│   └── config.py         # Orchestrator settings (pydantic model)
├── __init__.py
└── settings.toml         # Global settings (orchestrator, execution, responder, session)
```

---

## 17. Decisions & Resolved Questions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Classification model | Cheap LLM (Haiku) | Classification is simple — 4 categories, no complex reasoning. |
| Summarization model | Cheap LLM (Haiku) | TaskResultSummary is 1-3 sentences + metrics. |
| Synthesis model | Capable LLM (Sonnet) | Final synthesis is user-facing, needs quality. |
| Sequential vs parallel (v0) | Sequential only | Simpler to debug. Parallel is gated behind config flag. |
| Auto-unwrap convention | Single list field → unwrap | Covers 90% of cases (ScrapeResult wraps list[ProductListing]). |
| Interrupt auto-response | Only single-option or trivial decisions | Don't take agency from human on real choices. |
| Responder tool auto-approve | timeout < 10s = auto | Fast lookups shouldn't need human approval. |
| Max retries per enforcement | mandatory=5, strict=3, suggested=1 | Higher stakes → more chances. |
| Progress reporting | Text events via UI channel | Not formal msg_types — just forwarded to human. |
| Session timeout | 1 hour default, configurable | Safety kill-switch. Prevents runaway sessions. |

---

## 18. Open Questions

- **Q1:** Should the Orchestrator be able to re-plan mid-session? E.g., human says "actually, skip the pricing analysis and just give me the raw data." → **Defer to v1.** v0 = plan is fixed once built.
- **Q2:** Multi-skill composition — can the user request multiple unrelated skills in one message? "Phân tích giá iPhone VÀ phân tích xu hướng bàn làm việc." → **Defer to v1.** v0 = one plan per user request.
- **Q3:** Reviewer model selection — same as Executor, or a different model? → **Configurable.** Default: same model, but can be set to a different one for cost optimization.
- **Q4:** Should the Orchestrator cache LLM responses (classification, summarization) for identical inputs? → **Defer to v1.** v0 = no caching.
- **Q5:** How does the Orchestrator handle concurrent human requests in the same session? → **Queue them.** v0 = one request at a time. If human sends a new message while a task is running, queue it until the current task completes.

---

*This concludes the core architecture specs. The harness is now fully specified and ready for implementation.*
