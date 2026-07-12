# Spec 006: Testing & Skill Test Harness

**Status:** Draft
**Date:** 2026-07-12
**Author:** Human + Claude
**Depends on:** [Spec 001 — Message Protocol & Runtime Core](./001-message-protocol-runtime-core.md), [Spec 002 — Skill Format & Registry](./002-skill-format-registry.md), [Spec 005 — Executor Agent, LLM Providers & CLI Bootstrap](./005-executor-reviewer-cli.md)

---

## 1. Scope

This spec defines:

- **Test Harness** — infrastructure for testing skills in isolation with mock agent contexts.
- **Test Patterns** — mock LLM, integration end-to-end, skill unit tests.
- **File Layout** — `tests/` directory structure and conventions.

Out of scope: CI/CD pipeline configuration, performance/load testing (future spec).

---

## 2. Testing Philosophy

### 2.1 Principles

1. **Skills are testable in isolation.** A skill author should be able to verify their skill's behavior without running the full harness (no Orchestrator, no real LLM).
2. **Mock boundary is at the LLM.** Everything below the LLM (ActionDispatcher, ToolBridge, handlers) is real code tested directly.
3. **Message protocol is tested independently.** Message creation, serialization, enums — pure data, no agents involved.
4. **Integration tests close the loop.** End-to-end: Orchestrator → Executor → Reviewer → verdict, with a mock LLM returning predetermined responses.

### 2.2 Test Categories

| Category | Scope | Mock LLM? | Real agents? | Example |
|----------|-------|-----------|-------------|---------|
| **Unit: Models** | Pydantic models, enums, data structures | N/A | No | `test_message.py` — Message creation, MsgType enum |
| **Unit: Registry** | Skill discovery, validation, resolution | No | No | `test_registry.py` — parse skill.md, resolve actions |
| **Unit: Pipeline** | Dependency resolution, execution plan | No | No | `test_pipeline.py` — build_plan(), cycle detection |
| **Unit: Tool Bridge** | Action-to-tool mappings, schema generation | No | No | `test_tool_bridge.py` — signature → JSON Schema |
| **Unit: Parsers** | HTML parsing, CSV export, web fetcher | No | No | Test with fixture HTML/CSV data |
| **Integration: Agent** | Executor/Reviewer/Orchestrator logic | Yes (mock) | Real (spawn/send in-memory) | `test_executor_agent.py` — full ReAct loop |
| **Integration: E2E** | Full session with mock LLM | Yes (mock) | All real | Orchestrator → plan → execute → synthesize |
| **E2E: Live** | Full session with real LLM API | No | All real | Manual: `python -m llend` with real API key |

---

## 3. Mock LLM Pattern

### 3.1 Why Mock LLM

Integration tests should not hit real LLM APIs — they're slow, expensive, and non-deterministic. A mock LLM returns predetermined responses for known inputs.

### 3.2 Mock Interface

```python
from unittest.mock import AsyncMock, MagicMock
from llend.llm.client import LLMClient

def create_mock_llm(responses: list[str]) -> AsyncMock:
    """Create a mock LLMClient that returns *responses* in order.
    
    Each response is a JSON string.  The mock raises RuntimeError
    if called more times than responses provided.
    """
    mock = MagicMock(spec=LLMClient)
    mock.generate = AsyncMock(side_effect=responses)
    return mock
```

### 3.3 Example: Full Task Execution Cycle

```python
@pytest.mark.asyncio
async def test_full_task_cycle():
    """Orchestrator → Executor → Reviewer → complete."""
    mock_llm = create_mock_llm([
        # Classification: "task"
        '{"category": "task", "confidence": 0.9, "reasoning": "clear task"}',
        # Skill extraction
        '{"skill_name": "analyze_pricing", "params": {"query": "iPhone"}, "confidence": 0.9}',
        # Executor output (via ReAct loop)
        '{"status": "done", "output": {"median": 325, "count": 500}, "concerns": []}',
        # Reviewer verdict
        '{"verdict": "pass", "issues": [], "confidence": 0.95}',
        # Task summary
        '{"summary": "Analyzed 500 items, median $325", "key_metrics": {"median": 325}, "notable_findings": []}',
    ])

    runtime = AsyncioRuntime()
    # ... setup registry, pipeline, orchestrator ...

    assert orch.session_state.completed_tasks
    assert orch.session_state.completed_tasks[0].summary == "Analyzed 500 items, median $325"
```

---

## 4. Skill Test Harness

### 4.1 Goal

Skill authors should be able to test their skill in isolation:

```python
from llend.testing import SkillHarness

async def test_my_skill():
    harness = SkillHarness(skill_name="analyze_pricing")
    result = await harness.run(
        task_spec={"target_item": "iPhone 15", "dataset": [...]},
        mock_llm_responses=[...],
    )
    assert result.status == "done"
    assert result.output["market"]["median"] > 0
```

### 4.2 SkillHarness API

```python
class SkillHarness:
    """Lightweight test harness for a single skill.
    
    Creates an in-memory runtime, spawns Executor → runs task → returns result.
    No Orchestrator, no Reviewer, no real LLM.
    """

    def __init__(self, skill_name: str, skills_dir: Path | None = None): ...

    async def run(
        self,
        task_spec: dict[str, Any],
        mock_llm_responses: list[str],
        *,
        handler: object | None = None,
    ) -> TaskResult:
        """Execute the skill once and return the result."""
        ...

    async def run_with_handler(
        self,
        task_spec: dict[str, Any],
    ) -> TaskResult:
        """Execute using the real handler (no LLM — handler is called directly)."""
        ...
```

**Deferred to v1** — the harness currently exists as a pattern, not a reusable class.

---

## 5. Current Test Suite

### 5.1 File Layout

```
tests/
├── __init__.py
├── test_checkpoint.py       # Checkpoint model + disk persistence (Spec 001 §3.4)
├── test_lifecycle.py        # Agent lifecycle states (Spec 001 §3.3)
├── test_message.py          # Message envelope + enums (Spec 001 §2.1-§2.2)
├── test_notifications.py    # Notification channels (Spec 001 §3.4.1)
├── test_parser.py           # Input parser for skill.md frontmatter (Spec 002 §2.4)
├── test_pipeline.py         # SkillPipeline — plan building, cycle detection (Spec 002 §7)
├── test_registry.py         # SkillRegistry — discovery, validation, resolution (Spec 002 §6)
├── test_responder_agent.py  # ResponderAgent — query handling (Spec 003)
├── test_responder_models.py # Responder Pydantic models (Spec 003)
├── test_runtime.py          # AsyncioRuntime — spawn, send, kill (Spec 001 §3)
└── test_tool_bridge.py      # ToolBridge — mapping resolution, schema generation (Spec 002 §5)
```

### 5.2 What's Covered

| Component | Test File | Status |
|-----------|-----------|--------|
| Message protocol | `test_message.py` | ✅ |
| Agent lifecycle | `test_lifecycle.py` | ✅ |
| Runtime spawn/send/kill | `test_runtime.py` | ✅ |
| Checkpoint persistence | `test_checkpoint.py` | ✅ |
| Notification channels | `test_notifications.py` | ✅ |
| Skill parsing | `test_parser.py` | ✅ |
| Skill registry | `test_registry.py` | ✅ |
| Skill pipeline | `test_pipeline.py` | ✅ |
| Tool bridge | `test_tool_bridge.py` | ✅ |
| Responder agent | `test_responder_agent.py` | ✅ |
| Responder models | `test_responder_models.py` | ✅ |

### 5.3 What's Missing (v1)

| Component | Gap | Priority |
|-----------|-----|----------|
| Executor ReAct loop | No mock-LLM integration test for `ExecutorAgent` | High |
| Reviewer verdict | No mock-LLM integration test for `ReviewerAgent` | High |
| Orchestrator full flow | No end-to-end test with mock LLM | Medium |
| Handler auto-wrap | No test for `_auto_wrap_from_tools()` | Medium |
| Classifier | No test for message classification accuracy | Medium |
| SkillHarness class | Not yet extracted as reusable test utility | Low |
| Live E2E | Requires real API key — manual only for v0 | Low |

---

## 6. Running Tests

```powershell
# All tests
pytest tests/ -v

# Specific component
pytest tests/test_registry.py -v

# With coverage
pytest tests/ --cov=llend --cov-report=term-missing
```

---

## 7. Decisions & Open Questions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Mock LLM interface | `AsyncMock(side_effect=list[str])` | Simplest — each call returns next predetermined JSON string |
| SkillHarness priority | Defer to v1 | Current tests use component-level mocks; extracting a reusable harness needs design iteration |
| Live E2E in CI? | No for v0 | Requires API key + costs money; manual only |
| Test framework | pytest + pytest-asyncio | Standard Python async testing |

---

*This spec documents the current testing infrastructure and patterns. Full SkillHarness implementation deferred to v1.*
