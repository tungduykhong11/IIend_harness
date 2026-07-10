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

**IMPORTANT: The task spec above contains the EXACT parameters to use.**
Do NOT invent your own query, URL, or search terms.  Use the values from
the task spec exactly as given.  If the task spec says "iPhone 15", do NOT
search for "laptop" or any other product.  If a required parameter is
missing from the task spec, report it as a concern — do not substitute
your own defaults.

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

`generate()` returns a plain `str`. When tools are passed via the API, native
function-calling responses are serialized back to JSON text blocks (see §4.4).
The Executor parses the response text to detect tool calls vs final answers.

```python
class LLMError(Exception):
    """Raised when an LLM API call fails.  §2.6."""

async def _react_loop(system_prompt, tools, llm_client, action_dispatcher):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Execute: {task_spec}"},
    ]

    for _iteration in range(max_tool_calls):
        try:
            response_text = await llm_client.generate(messages, tools=tools)
        except Exception:
            raise LLMError("LLM API call failed")  # caught → agent.error(LLM_ERROR)

        # Parse response text for tool calls (JSON blocks with name/id/arguments)
        tool_calls = _extract_tool_calls(response_text, tools)

        if tool_calls:
            for tc in tool_calls:
                try:
                    result = await action_dispatcher.dispatch(
                        tc["name"], tc.get("arguments", {}),
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": json.dumps(result, default=str),
                    })
                except ActionDispatchError:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": json.dumps({"error": str(exc)}),
                    })
            continue  # back to LLM with tool results

        # No tool calls — LLM produced final answer
        parsed = _parse_output(response_text)
        if not parsed.get("parsed_ok", True):
            # §2.6: Output doesn't parse as JSON → VALIDATION_ERROR
            raise ValidationError(parsed.get("raw_text", response_text))
        return parsed

    # Exhausted max_tool_calls
    return {"status": "error", "output": None,
            "concerns": [f"Exceeded max tool calls ({max_tool_calls})"]}
```

### 2.5 Tool Definitions

Tools are described to the LLM using OpenAI function-calling format (universal — works with Anthropic, OpenAI, and DeepSeek).

Each `ActionBinding` (Spec 002 §6.2) carries an optional `input_schema` — a JSON Schema object describing the action's parameters. This is populated by the Registry during `resolve()`: for custom actions, the schema is extracted from the handler method's docstring; for global actions, it comes from the tool bridge mapping.

```python
# Mapping from Python type annotations to JSON Schema types
_TYPE_MAP = {
    "str": "string", "int": "integer", "float": "number",
    "bool": "boolean", "list": "array", "dict": "object",
}

def _build_tool_definitions(action_bindings: dict) -> list[dict]:
    tools = []
    for name, binding in action_bindings.items():
        params_schema = {"type": "object", "properties": {}, "required": []}

        if binding.input_schema:
            # Use the Registry-resolved JSON Schema directly
            params_schema = binding.input_schema
        else:
            # Fallback: no schema available → empty parameters object.
            # The LLM will call the tool with no arguments.
            pass

        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"Action: {name}",
                "parameters": params_schema,
            },
        })
    return tools
```

> **Note:** `ActionBinding.input_schema` is populated by the Registry during `Skill.resolve()` (Spec 002 §6.1). For custom actions, it is derived from the handler method's docstring `INPUT:` block. For global actions, it comes from the tool bridge mapping. If neither source provides a schema, the parameters object is empty and the LLM will call the tool with no arguments.

### 2.5.1 JSON Extraction from Mixed Text

LLMs (especially DeepSeek) often output natural-language explanation followed
by JSON blocks, or wrap JSON in markdown fences (`` ```json ``` ``).  A shared
helper ``_find_json_objects()`` scans the entire response for ``{…}`` pairs
(tracking brace depth) and returns all successfully parsed dicts:

```python
def _find_json_objects(text: str) -> list[dict[str, Any]]:
    """Find all top-level JSON objects in *text*, handling mixed NL + JSON."""
    results = []
    depth = 0; start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                segment = text[start:i+1]
                try:
                    obj = json.loads(segment)
                    if isinstance(obj, dict): results.append(obj)
                except json.JSONDecodeError: pass
                start = None
        if depth < 0: depth = 0; start = None  # unbalanced reset
    return results
```

Both ``_extract_tool_calls`` and ``_parse_output`` use this helper:
- **Tool calls:** Any JSON object with ``"name"`` matching a known tool is a tool call.
- **Final output:** Any JSON object with ``"output"`` key is the final answer.
- **Fallback:** If neither pattern matches, try parsing the whole text as JSON.

### 2.6 Error Handling

The Executor distinguishes error types at the ReAct loop level using a custom
`LLMError` exception to separate LLM API failures from general crashes:

```python
class LLMError(Exception):
    """Raised when an LLM API call fails — caught by the caller to send
    ``agent.error(LLM_ERROR)`` rather than ``agent.error(CRASH)``.  §2.6."""
```

| Scenario | Exception | Response |
|----------|-----------|----------|
| Tool call fails (ActionDispatchError) | `ActionDispatchError` | Add tool error to messages, let LLM decide: retry OR return `status=error` |
| LLM API error | `LLMError` | Send `agent.error(LLM_ERROR)` to Orchestrator, exit |
| Output doesn't parse as JSON | (caught in `_parse_output`) | Return `parsed_ok=False` → caller sends `agent.error(VALIDATION_ERROR)` with raw text |
| Timeout (no LLM response within task timeout) | `asyncio.TimeoutError` | Send `agent.error(TIMEOUT)` |
| Unhandled exception | `Exception` | Send `agent.error(CRASH)` with traceback |

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

### 4.3 LLMStreamEvent

```python
from pydantic import BaseModel

class LLMStreamEvent(BaseModel):
    """One event in an LLM streaming response — shared by all providers.

    Uses Pydantic BaseModel (not dataclass) for validation and
    serialization.  Fields use None defaults to distinguish "not set"
    from "empty string".
    """
    type: str  # "text" | "tool_call" | "done" | "error"
    # text
    text_delta: str | None = None
    # tool_call (only emitted when a tool call is fully assembled)
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_input: dict[str, Any] | None = None
    # done
    finish_reason: str | None = None  # "end_turn" | "max_tokens" | "tool_use"
    # error
    error_message: str | None = None
```

### 4.4 Implementation

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

    async def generate(self, messages, system=None, tools=None):
        client = self._ensure_client()
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        kwargs = {
            "model": self._model,
            "messages": msgs,
            "max_tokens": self._max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # If the LLM returned native tool calls, serialize them as JSON
        # so the Executor's text parser can extract them.
        if choice.message.tool_calls:
            parts = []
            if choice.message.content:
                parts.append(choice.message.content)
            for tc in choice.message.tool_calls:
                parts.append(json.dumps({
                    "name": tc.function.name,
                    "id": tc.id,
                    "arguments": (
                        json.loads(tc.function.arguments)
                        if tc.function.arguments else {}
                    ),
                }))
            return "\n".join(parts)

        return choice.message.content or ""

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

        # Accumulate streaming tool call deltas by index.
        # OpenAI-compatible APIs send tool call arguments in fragments —
        # a single tool call's name, id, and arguments arrive across
        # multiple chunks.  We accumulate until each tool call is complete.
        tool_call_acc: dict[int, dict] = {}

        def _try_finalize_tool_call(idx: int) -> LLMStreamEvent | None:
            """If the accumulated tool call at *idx* is complete, return an
            event and clear the accumulator slot.  Otherwise return None."""
            tc = tool_call_acc.get(idx)
            if tc is None:
                return None
            # A tool call is complete when we have an id, a name, and
            # the accumulated arguments parse as valid JSON.
            if not (tc.get("id") and tc.get("name")):
                return None
            try:
                parsed = json.loads(tc["arguments"])
            except json.JSONDecodeError:
                return None  # still receiving fragments
            # Complete — yield and clear
            del tool_call_acc[idx]
            return LLMStreamEvent(
                type="tool_call",
                tool_name=tc["name"],
                tool_call_id=tc["id"],
                tool_input=parsed,
            )

        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                delta = chunk.choices[0].delta

                # Text content
                if delta.content:
                    yield LLMStreamEvent(type="text", text_delta=delta.content)

                # Tool call deltas — accumulate by index
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_acc:
                            tool_call_acc[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        acc = tool_call_acc[idx]
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                acc["name"] = tc.function.name
                            if tc.function.arguments:
                                acc["arguments"] += tc.function.arguments
                        # Try to finalize — yields only when the tool call
                        # arguments JSON is complete.
                        event = _try_finalize_tool_call(idx)
                        if event is not None:
                            yield event

            # End of stream — emit any remaining incomplete tool calls
            # (arguments never completed; surface as error)
            for idx in sorted(tool_call_acc.keys()):
                tc = tool_call_acc[idx]
                yield LLMStreamEvent(
                    type="error",
                    error_message=(
                        f"Tool call '{tc.get('name', 'unknown')}' "
                        f"(id={tc.get('id', '?')}): arguments stream ended "
                        f"before JSON was complete"
                    ),
                )
            tool_call_acc.clear()

            yield LLMStreamEvent(type="done", finish_reason=chunk.choices[0].finish_reason or "end_turn")
        except Exception as exc:
            yield LLMStreamEvent(type="error", error_message=str(exc))
```

### 4.5 Provider Factory

```python
def create_llm_client(provider: str, **kwargs) -> LLMClient:
    """Factory for LLM providers.  Provider is one of: anthropic, deepseek, openai."""
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    elif provider == "deepseek":
        return DeepSeekClient(**kwargs)
    elif provider == "openai":
        raise NotImplementedError("OpenAI provider not yet implemented — deferred to v1.")
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
```

### 4.6 Model Override Per Role

Per Spec 004 §17, different Orchestrator functions use different models. When using DeepSeek, all roles default to `deepseek-chat`:

| Function | Config Key | Default (DeepSeek) | Default (Anthropic) |
|----------|------------|-------------------|---------------------|
| Classification | `orchestrator.classification_model` | `deepseek-chat` | `claude-haiku-4-5-20251001` |
| Summarization | `orchestrator.summarization_model` | `deepseek-chat` | `claude-haiku-4-5-20251001` |
| Synthesis | `orchestrator.synthesis_model` | `deepseek-chat` | `claude-sonnet-5` |
| Executor (task) | `executor.model` | `deepseek-chat` | `claude-sonnet-5` |
| Reviewer | `reviewer.model` | `deepseek-chat` | `claude-sonnet-5` |
| Responder | `responder.model` | `deepseek-chat` | `claude-sonnet-5` |

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
runtime.register_agent_type(
    AgentType.RESPONDER.value,
    lambda handle, ctx: ResponderAgent(
        runtime=runtime,
        instance_id=handle.instance_id,
        session_id=runtime.session_id,
        llm_client=llm_client,
        persona=Persona(ctx.get("persona", "auto")),
    ).start(),
)
```

> **Important:** All three agent types (Executor, Reviewer, **and Responder**) must be registered.  Without the Responder factory, the Orchestrator's `_spawn_responder()` creates a handle but the Responder's processing loop never starts — conversational messages are silently dropped.

---

## 6. CLI Bootstrap

### 6.1 Entry Point

`python -m llend` launches the harness. The bootstrap sequence:

```
0. Load .env file if present (project root or cwd) — API keys, provider config
1. Load settings from llend/settings.toml (or use defaults)
2. Create LLMClient via provider factory (DEEPSEEK_API_KEY or ANTHROPIC_API_KEY)
3. Create AsyncioRuntime
4. Create ToolBridge from mappings.toml (warns on missing tools, does NOT crash)
5. Create SkillRegistry, discover + resolve skills
6. Create SkillPipeline
7. Register ExecutorAgent, ReviewerAgent, AND ResponderAgent factories
8. Create OrchestratorAgent (injects runtime, registry, llm_client, pipeline)
9. Orchestrator.start(session_goal=...) — sends session.start internally
10. CLI read-eval-print loop:
    - Read human input (stdin)
    - Send USER_MESSAGE with user text (Spec 001 §2.2)
    - Wait for response via Orchestrator.wait_for_response()
    - (Progress callback prints task events & Responder answers)
    - Loop to next prompt
11. On "exit" or Ctrl+C → Orchestrator.shutdown()
```

> **Note on step 10:** The CLI does NOT send a second `SESSION_START` — the Orchestrator sends one internally during `start()`. Progress output and Responder replies are printed by the `on_progress` callback; the CLI's `wait_for_response()` only blocks until the answer is ready before showing the next prompt.

### 6.2 CLI REPL

```python
async def main():
    # 0. Load .env for API keys  (python-dotenv, if installed)
    _load_dotenv()

    # 1-2. Config + LLM client
    config = OrchestratorConfig.from_toml()
    provider = os.environ.get("LLEND_PROVIDER", "deepseek")
    llm_client = create_llm_client(provider, model=config.classification_model)

    # 3-6. Runtime, ToolBridge, Registry, Pipeline
    runtime = AsyncioRuntime()
    tool_bridge = ToolBridge(mappings_path=Path("llend/tool_bridge/mappings.toml"))
    registry = SkillRegistry(Path("llend/skills"), tool_bridge)
    await registry.discover()
    registry.resolve_all()
    pipeline = SkillPipeline(registry)

    # 7. Register agent factories — Executor, Reviewer, AND Responder
    runtime.register_agent_type(AgentType.EXECUTOR.value, _executor_factory(llm_client))
    runtime.register_agent_type(AgentType.REVIEWER.value, _reviewer_factory(llm_client))
    runtime.register_agent_type(AgentType.RESPONDER.value, _responder_factory(llm_client))

    # 8-9. Orchestrator (sends session.start internally — no duplicate)
    orch = OrchestratorAgent(
        runtime=runtime, registry=registry, llm_client=llm_client,
        config=config, pipeline=pipeline,
        on_progress=lambda ev: print(f"  {ev.message}"),
    )
    await orch.start(session_goal="Interactive session")

    print("llend harness ready. Type your request or 'exit'.")
    try:
        while True:
            user_input = input("> ")
            if user_input.lower() in ("exit", "quit"):
                break
            # Each input uses user.message (Spec 001 §2.2)
            msg = Message(
                session_id=runtime.session_id,
                sender="human", sender_instance="cli",
                recipient=AgentType.ORCHESTRATOR.value,
                msg_type=MsgType.USER_MESSAGE,
                payload={"text": user_input},
            )
            await runtime.send(msg)
            # Wait for response before showing next prompt.
            # (Progress callback prints the answer; we just block.)
            await orch.wait_for_response(timeout=120.0)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        await orch.shutdown()
        await runtime.shutdown()


if __name__ == "__main__":
    # WARNING level keeps CLI output clean — progress uses callback, not logger
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    asyncio.run(main())
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
