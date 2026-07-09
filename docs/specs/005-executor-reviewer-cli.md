# Spec 005: Executor Agent, LLM Providers & CLI Bootstrap

**Status:** Draft
**Date:** 2026-07-09
**Author:** Human + Claude
**Depends on:** [Spec 001 — Message Protocol & Runtime Core](./001-message-protocol-runtime-core.md), [Spec 002 — Skill Format & Registry](./002-skill-format-registry.md), [Spec 003 — Responder Agent & Conversation Module](./003-responder-agent-conversation-module.md), [Spec 004 — Orchestrator Logic & Session Orchestration](./004-orchestrator-logic.md)

---

## 1. Motivation

Spec 001 defines the agent topology (Orchestrator → Executor → Reviewer). Spec 002 defines skills and the ActionDispatcher. Spec 003 defines the Responder. Spec 004 defines the Orchestrator and the Reviewer's adversarial prompt (§4.5).

But the **Executor** exists only as an **agent type** — there is no class that actually:

- Receives `task.dispatch`
- Calls an LLM in a ReAct loop
- Invokes tools via ActionDispatcher
- Returns `task.result`

Additionally, only `AnthropicClient` exists as an LLM backend. There is no DeepSeek provider, no provider factory, and no CLI entry point to bootstrap the harness.

> **Note on ReviewerAgent:** The Reviewer agent type, message types (`task.review` / `task.verdict`), and adversarial prompt are already fully specified in Spec 001 (§2.2) and Spec 004 (§4.5). The `ReviewerAgent` class implementation is a thin wrapper (~50 lines): receive `task.review` → call LLM with the already-constructed `system_prompt` → parse JSON verdict → send `task.verdict`. It needs no new spec content.

**This spec closes the remaining gaps.** After Spec 005, the harness will be runnable end-to-end with `python -m llend`.

---

## 2. Executor Agent

### 2.1 Role & Mindset

| | Executor |
|---|---|
| **Role** | Execute 1 skill (1 task in the plan) |
| **Mindset** | Constructor — build the output, call tools as needed, report concerns honestly |
| **Lifespan** | Per task — spawned fresh, killed after `task.result` |
| **Input** | `task.dispatch` message with `{task_id, skill_name, task_spec, skill_context}` (Spec 001 §2.2) |
| **Output** | `task.result` message with `{task_id, status: TaskStatus, output, concerns?}` (Spec 001 §2.2) |
| **Internal state** | Stateless beyond the current task |

### 2.2 Lifecycle

```
Runtime spawns Executor
        │
        ▼
    INIT ──→ RUNNING
        │
        ├── Receive task.dispatch
        ├── ReAct loop:
        │   ├── LLM decides: call tool OR return answer
        │   ├── If tool → ActionDispatcher.dispatch() → result → feed back to LLM
        │   └── If answer → format output, exit loop
        ├── Send task.result (status=DONE or DONE_WITH_CONCERNS)
        └── COMPLETE → DEAD
```

### 2.3 System Prompt

```
You are an EXECUTOR agent. Your job is to complete ONE task using the tools provided.

## Task
- Skill: {skill_name} — {skill_description}
- Task spec: {task_spec}

## Available Tools
{tool_descriptions}

## Instructions
1. Read the task spec carefully. Understand what output is expected.
2. Use the available tools to gather data or perform actions.
3. If you need data from a tool, CALL it. Don't guess — use tools for real data.
4. When you have everything you need, produce the final output.
5. If something is wrong or uncertain, report it in "concerns".
6. Output must match the expected schema: {output_schema}

## Output Format
Respond with JSON:
{
  "status": "done" | "done_with_concerns" | "partial" | "error",
  "output": <task output — must match the expected schema>,
  "concerns": ["list of concerns, if any"]
}
```

### 2.4 ReAct Loop

```python
async def _react_loop(dispatch_msg, llm_client, action_dispatcher, skill_context):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Execute: {task_spec}"},
    ]
    tools = _build_tool_definitions(skill_context.action_bindings)

    while True:
        response = await llm_client.generate(messages, tools=tools)

        if response.has_tool_calls():
            for tool_call in response.tool_calls:
                result = await action_dispatcher.dispatch(
                    tool_call.name, tool_call.arguments
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })
            continue  # back to LLM with tool results

        # No tool calls — LLM produced final answer
        parsed = json.loads(response.text)
        return parsed  # {status, output, concerns}
```

### 2.5 Tool Definitions

Tools are described to the LLM using OpenAI function-calling format (universal — works with Anthropic, OpenAI, and DeepSeek):

```python
def _build_tool_definitions(action_bindings):
    tools = []
    for name, binding in action_bindings.items():
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"Action: {name}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        key: {"type": "string", "description": key}
                        for key in binding.config.get("params", [])
                    } if binding.config.get("params") else {},
                },
            },
        })
    return tools
```

### 2.6 Error Handling

| Scenario | Response |
|----------|----------|
| Tool call fails (ActionDispatchError) | Add tool error to messages, let LLM decide: retry OR return `status=error` |
| LLM API error | Send `agent.error(LLM_ERROR)` to Orchestrator, exit |
| Output doesn't parse as JSON | Send `agent.error(VALIDATION_ERROR)` with raw text |
| Timeout (no LLM response within task timeout) | Send `agent.error(TIMEOUT)` |
| Unhandled exception | Send `agent.error(CRASH)` with traceback |

### 2.7 File Layout

```
llend/executor/
├── __init__.py          # Re-exports
└── agent.py             # ExecutorAgent class
```

---

## 3. Reviewer Agent

> **Already specified in existing specs.** No new spec content needed.

| Aspect | Where defined |
|--------|--------------|
| Agent type, lifecycle states | Spec 001 §1, §3.3 |
| `task.review` / `task.verdict` message types | Spec 001 §2.2 |
| Adversarial system prompt | Spec 004 §4.5 (`REVIEWER_SYSTEM_PROMPT`) |
| ReviewIssue model, Verdict enum | Spec 001 §2.2.1 |
| Adjudication after verdict | Spec 004 §4.3–§4.4 |
| Error handling (crash, timeout, LLM error) | Spec 004 §10.1 |

The implementation is a thin class:

```python
class ReviewerAgent:
    """Receives task.review → calls LLM with system_prompt → sends task.verdict."""

    async def start(self):
        await self._runtime.register_handler(self._instance_id, self._handle_message)

    async def _handle_message(self, msg: Message):
        if msg.msg_type != MsgType.TASK_REVIEW:
            return
        system_prompt = msg.payload.get("system_prompt", "")
        response = await self._llm.generate(
            messages=[{"role": "user", "content": "Review carefully."}],
            system=system_prompt,
        )
        verdict_data = json.loads(response)
        await self._send_verdict(msg, verdict_data)
```

File: `llend/reviewer/agent.py` (~50-60 dòng).

---

## 4. DeepSeek Client (LLM Provider)

### 4.1 Background

DeepSeek V4 Pro exposes an **OpenAI-compatible API**. We implement `DeepSeekClient` by wrapping the `openai` SDK's `AsyncOpenAI` client, exactly as `AnthropicClient` wraps `anthropic.AsyncAnthropic`.

### 4.2 API Details

| Field | Value |
|-------|-------|
| Base URL | `https://api.deepseek.com` (or custom endpoint via env `DEEPSEEK_BASE_URL`) |
| API Key | `DEEPSEEK_API_KEY` environment variable |
| Model | `deepseek-chat` (V4 Pro) |
| SDK | `openai` (OpenAI-compatible) — `pip install openai` |

### 4.3 Implementation

```python
class DeepSeekClient(LLMClient):
    """LLMClient backed by DeepSeek (OpenAI-compatible API)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_tokens: int = 4096,
    ) -> None:
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self._model = model
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI
            if not self._api_key:
                raise RuntimeError(
                    "DEEPSEEK_API_KEY environment variable is not set."
                )
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    async def generate(self, messages, system=None):
        client = self._ensure_client()
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        response = await client.chat.completions.create(
            model=self._model,
            messages=msgs,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""

    async def stream_generate(self, messages, system=None, tools=None):
        client = self._ensure_client()
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        kwargs = {
            "model": self._model,
            "messages": msgs,
            "max_tokens": self._max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield LLMStreamEvent(type="text", text_delta=delta.content)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        yield LLMStreamEvent(
                            type="tool_call",
                            tool_name=tc.function.name if tc.function else "",
                            tool_call_id=tc.id or "",
                            tool_input=(
                                json.loads(tc.function.arguments)
                                if tc.function and tc.function.arguments else {}
                            ),
                        )
            yield LLMStreamEvent(type="done", finish_reason="end_turn")
        except Exception as exc:
            yield LLMStreamEvent(type="error", error_message=str(exc))
```

### 4.4 Provider Factory

```python
def create_llm_client(provider: str, **kwargs) -> LLMClient:
    """Factory for LLM providers.  Provider is one of: anthropic, deepseek, openai."""
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    elif provider == "deepseek":
        return DeepSeekClient(**kwargs)
    elif provider == "openai":
        return OpenAIClient(**kwargs)  # future
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
```

### 4.5 Model Override Per Role

Per Spec 004 §17, different Orchestrator functions use different models. When using DeepSeek, all roles default to `deepseek-chat`:

| Function | Config Key | Default (DeepSeek) | Default (Anthropic) |
|----------|------------|-------------------|---------------------|
| Classification | `orchestrator.classification_model` | `deepseek-chat` | `claude-haiku-4-5-20251001` |
| Summarization | `orchestrator.summarization_model` | `deepseek-chat` | `claude-haiku-4-5-20251001` |
| Synthesis | `orchestrator.synthesis_model` | `deepseek-chat` | `claude-sonnet-5` |
| Executor (task) | `executor.model` | `deepseek-chat` | `claude-sonnet-4-20250514` |
| Reviewer | `reviewer.model` | `deepseek-chat` | `claude-sonnet-4-20250514` |
| Responder | `responder.model` | `deepseek-chat` | `claude-sonnet-4-20250514` |

---

## 5. Runtime Agent Spawning (Fix)

### 5.1 The Problem

Currently `runtime.spawn(AgentType.EXECUTOR, ...)` creates an `_AgentHandle` with a queue but **no processing loop**. Messages sent to the Executor's queue are never read. The Orchestrator's `_await_response()` blocks forever.

### 5.2 Solution: Agent Registry

The runtime needs an **agent registry** — a mapping from `AgentType` to a factory function that creates and starts an agent's processing loop:

```python
# In AsyncioRuntime.__init__:
self._agent_factories: dict[str, Callable] = {}

def register_agent_type(self, agent_type: str, factory: Callable):
    """Register a factory that creates an agent's processing task."""
    self._agent_factories[agent_type] = factory

async def spawn(self, agent_type, context):
    instance_id = f"{agent_type}-{uuid4().hex[:8]}"
    handle = _AgentHandle(instance_id=instance_id, agent_type=agent_type, context=context)
    # ...
    self._agents[instance_id] = handle

    # If a factory is registered, launch the agent's processing loop
    if agent_type in self._agent_factories:
        handle.main_task = asyncio.create_task(
            self._agent_factories[agent_type](handle, context),
            name=f"agent-{instance_id}",
        )

    return instance_id
```

### 5.3 Bootstrap Wiring

At CLI bootstrap time, register the agent types:

```python
runtime = AsyncioRuntime()
registry = SkillRegistry(...)

# Register agent factories
runtime.register_agent_type(
    AgentType.EXECUTOR.value,
    lambda handle, ctx: ExecutorAgent(
        runtime=runtime,
        instance_id=handle.instance_id,
        session_id=runtime.session_id,
        llm_client=llm_client,
        registry=registry,
    ).start(),
)
runtime.register_agent_type(
    AgentType.REVIEWER.value,
    lambda handle, ctx: ReviewerAgent(
        runtime=runtime,
        instance_id=handle.instance_id,
        session_id=runtime.session_id,
        llm_client=llm_client,
    ).start(),
)
```

---

## 6. CLI Bootstrap

### 6.1 Entry Point

`python -m llend` launches the harness. The bootstrap sequence:

```
1. Load settings from llend/settings.toml (or use defaults)
2. Create LLMClient via provider factory (DEEPSEEK_API_KEY or ANTHROPIC_API_KEY)
3. Create AsyncioRuntime
4. Create ToolBridge from mappings.toml
5. Create SkillRegistry, discover + resolve skills
6. Create SkillPipeline
7. Register ExecutorAgent and ReviewerAgent factories with runtime
8. Create OrchestratorAgent (injects runtime, registry, llm_client, pipeline)
9. Orchestrator.start(session_goal=...)
10. CLI read-eval-print loop:
    - Read human input (stdin)
    - Send SESSION_START message to Orchestrator with user text
    - Wait for response (Responder reply or task completion)
    - Print output to stdout
11. On "exit" or Ctrl+C → Orchestrator.shutdown()
```

### 6.2 CLI REPL

```python
async def main():
    config = OrchestratorConfig.from_toml()
    provider = os.environ.get("LLEND_PROVIDER", "deepseek")
    llm_client = create_llm_client(provider, model=config.classification_model)

    runtime = AsyncioRuntime()
    tool_bridge = ToolBridge(mappings_path=Path("llend/tool_bridge/mappings.toml"))
    registry = SkillRegistry(Path("llend/skills"), tool_bridge)
    await registry.discover()
    registry.resolve_all()

    pipeline = SkillPipeline(registry)

    # Register agent factories
    runtime.register_agent_type(AgentType.EXECUTOR.value, _executor_factory(llm_client, registry))
    runtime.register_agent_type(AgentType.REVIEWER.value, _reviewer_factory(llm_client))

    orch = OrchestratorAgent(
        runtime=runtime,
        registry=registry,
        llm_client=llm_client,
        config=config,
        pipeline=pipeline,
        on_progress=lambda ev: print(f"  {ev.message}"),
    )
    await orch.start(session_goal="Interactive session")

    print("llend harness ready. Type your request or 'exit'.")
    try:
        while True:
            user_input = input("> ")
            if user_input.lower() in ("exit", "quit"):
                break
            msg = Message(
                session_id=runtime.session_id,
                sender="human",
                sender_instance="cli",
                recipient=AgentType.ORCHESTRATOR.value,
                msg_type=MsgType.SESSION_START,
                payload={"text": user_input},
            )
            await runtime.send(msg)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        await orch.shutdown()
        await runtime.shutdown()
```

### 6.3 File Layout (After Spec 005)

```
llend/
├── __main__.py           # NEW: entry point — python -m llend
├── __init__.py
├── settings.toml         # Updated: +[llm], [executor], [reviewer]
├── llm/
│   ├── __init__.py
│   └── client.py         # Updated: +DeepSeekClient, +create_llm_client()
├── executor/
│   ├── __init__.py       # NEW
│   └── agent.py          # NEW: ExecutorAgent
├── reviewer/
│   ├── __init__.py       # NEW
│   └── agent.py          # NEW: ReviewerAgent (thin wrapper)
├── orchestrator/         # (existing)
├── responder/            # (existing)
├── runtime/              # Patched: +register_agent_type()
├── registry/             # (existing)
├── skills/               # (existing)
└── tool_bridge/          # (existing)
```

---

## 7. Configuration Extensions

### 7.1 New Settings in settings.toml

```toml
[llm]
provider = "deepseek"           # "deepseek" | "anthropic" | "openai"
api_key_env = "DEEPSEEK_API_KEY" # env var name for the API key
base_url = "https://api.deepseek.com"  # override for custom endpoints

[executor]
model = "deepseek-chat"         # model for Executor LLM calls
max_tool_calls_per_task = 20    # safety limit on ReAct loop iterations

[reviewer]
model = "deepseek-chat"         # model for Reviewer LLM calls
```

### 7.2 OrchestratorConfig Extensions

```python
class OrchestratorConfig(BaseModel):
    # ... existing fields from Spec 004 §13.1 ...

    # LLM provider
    llm_provider: str = "deepseek"
    llm_api_key_env: str = "DEEPSEEK_API_KEY"
    llm_base_url: str = "https://api.deepseek.com"

    # Executor
    executor_model: str = "deepseek-chat"
    executor_max_tool_calls: int = 20

    # Reviewer
    reviewer_model: str = "deepseek-chat"
```

---

## 8. Integration Test

### 8.1 End-to-End Test (Mock LLM)

```python
@pytest.mark.asyncio
async def test_full_task_execution_cycle():
    """Orchestrator → Executor → Reviewer → complete."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate = AsyncMock(side_effect=[
        # Classification: "task"
        '{"category": "task", "confidence": 0.9, "reasoning": "clear task"}',
        # Skill extraction
        '{"skill_name": "analyze_pricing", "params": {}, "confidence": 0.9}',
        # Executor output (via ReAct loop)
        '{"status": "done", "output": {"median": 325, "count": 500}, "concerns": []}',
        # Reviewer verdict
        '{"verdict": "pass", "issues": [], "confidence": 0.95}',
        # Task summary
        '{"summary": "Analyzed pricing: 500 items, median $325", "key_metrics": {"median": 325, "count": 500}, "notable_findings": []}',
        # Session synthesis
        "Session complete. Analyzed pricing data. Median: $325.",
    ])

    runtime = AsyncioRuntime()
    # Register factories, create orchestrator, start session...

    msg = Message(
        session_id=runtime.session_id,
        sender="human", sender_instance="cli",
        recipient=AgentType.ORCHESTRATOR.value,
        msg_type=MsgType.SESSION_START,
        payload={"text": "Phân tích giá iPhone 15"},
    )
    await runtime.send(msg)
    await asyncio.sleep(2)

    assert orch.session_state.completed_tasks  # at least one task finished
```

---

## 9. Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Executor LLM loop pattern | ReAct (tool-use loop) | Standard agent pattern; works with all LLM providers |
| Tool definition format | OpenAI function-calling JSON | Universal — Anthropic, OpenAI, and DeepSeek all support it |
| Reviewer: single call or loop? | Single call | Already specified in Spec 004 §4.5 — no tools needed |
| Provider SDK for DeepSeek | `openai` package | DeepSeek API is OpenAI-compatible; no custom SDK needed |
| Agent registration | Factory pattern on runtime | Keeps runtime agnostic; agent types are pluggable |
| CLI framework | Plain `asyncio` REPL (no click/typer) | Minimal dependencies; matches the harness's asyncio-native philosophy |
| DeepSeek model name | `deepseek-chat` | DeepSeek V4 Pro's standard model ID |

---

## 10. Open Questions

- **Q1:** Should Executor support streaming (LLM streams tool calls and text)? → **Defer to v1.** v0 uses non-streaming `generate()` for simplicity.
- **Q2:** Should multiple Executors share an LLM connection pool? → **Defer to v1.** v0: one client per agent.
- **Q3:** Should Reviewer use a different provider than Executor? → **Configurable.** Default: same provider, but `reviewer_model` can be set independently.
- **Q4:** Web UI vs CLI? → **CLI first (v0).** Web UI / Telegram bot deferred to v1.

---

*This concludes Spec 005. After implementation, the harness will be runnable end-to-end with `python -m llend`.*
