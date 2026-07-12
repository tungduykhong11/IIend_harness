# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: llend_harness

A Python-native **Hierarchical Multi-Agent Harness** — a runtime that orchestrates AI agents through composable skills. Domain-agnostic: not tied to coding workflows.

**Status: v0.1 — end-to-end pipeline working.** First application (Market Researcher) runs: `data_provider` → `analyze_pricing` → recommendation text. Tested with DeepSeek V4 Pro on real e-commerce sites (cellphones.com.vn, eBay).

### Agent Topology

The harness uses a 3-agent model inspired by Superpowers' Subagent-Driven Development (implementer + reviewer pattern), but generalized beyond coding:

```
                    ┌──────────────┐
                    │ Orchestrator │  ← "sếp": nhận yêu cầu, lập plan,
                    └──────┬───────┘    dispatch, tổng hợp kết quả
                           │
                           │  per task: spawn Executor → output → spawn Reviewer → verdict
                           │  if fail → loop back to Executor
                           ▼
              ┌────────────────────────┐
              │   Executor ──► Reviewer │  ← mỗi Executor có 1 Reviewer đi kèm
              │   (làm)       (kiểm)   │    cả 2 đều disposable, fresh context
              └────────────────────────┘
```

| Agent | Trách nhiệm | Mindset | Vòng đời |
|-------|-------------|---------|----------|
| **Orchestrator** | Tiếp nhận yêu cầu → phân rã thành task plan → dispatch Executor + Reviewer cho từng task → adjudicate verdict → tổng hợp kết quả cuối | Quản lý, cân bằng | Sống suốt session |
| **Executor** | Nhận 1 task + 1 skill + context → thực thi → trả output | "Làm" — constructive, hoàn thành nhiệm vụ | Spawn mới mỗi task, stateless |
| **Reviewer** | Nhận task spec + Executor output → adversarial verify → trả verdict (pass/fail + issues) | "Kiểm" — hoài nghi, refute, bắt lỗi | Spawn mới sau mỗi Executor, stateless |

**Per-task loop:**

```
Orchestrator dispatch
        │
        ▼
    Executor (làm) ──→ output
        │
        ▼
    Reviewer (kiểm) ──→ pass? ──→ next task
        │                  │
        │ fail             │
        └──────────────────┘
        loop: spawn Executor mới fix issues → Reviewer mới verify lại
```

**Nguyên tắc cốt lõi:**
- Số lượng agent **không hardcode** — topology do app định nghĩa, harness cung cấp runtime
- Executor và Reviewer đều **vô danh, stateless, disposable** — như worker trong pool
- Reviewer tách rời khỏi Orchestrator để: (1) tránh context overload cho Orchestrator, (2) Reviewer cần adversarial mindset khác hẳn Executor, (3) tương lai có thể có nhiều loại Reviewer (data quality, logic, compliance...)
- Hỗ trợ **N-level hierarchy** trong tương lai (Executor có thể tự spawn sub-Executor + sub-Reviewer)

### Human-in-the-Loop (Interrupt Node)

HITL is not a 4th agent type — it's a **control flow primitive** in the harness state graph, inspired by LangGraph's `interrupt()`. Any agent (Executor, Reviewer, or Orchestrator) can raise an interrupt when it encounters a decision that shouldn't be made autonomously.

```
     ┌──────────┐
     │  Agent   │  ← Executor / Reviewer / Orchestrator
     └────┬─────┘
          │
          │  "Cần hỏi sếp cái này"
          ▼
     ┌──────────┐
     │ Interrupt │  ← Node trong state graph
     │   Node    │    - Tạm dừng execution
     └────┬─────┘    - Gửi message + options cho human
          │          - Chờ human response
          │          - Resume với decision của human
          ▼
     ┌──────────┐
     │  Resume   │  ← Agent nhận human decision, tiếp tục
     └──────────┘
```

**Interrupt contract:**
- **Raise**: Agent gọi `interrupt(message, options?)` — message mô tả tình huống, options là các lựa chọn cho human
- **Pause**: Runtime tạm dừng execution graph, lưu checkpoint
- **Notify**: Gửi notification đến human (Telegram, Discord, Web UI...)
- **Wait**: Block cho đến khi human response
- **Resume**: Human decision được inject vào agent context, execution tiếp tục

**Ví dụ Market Researcher:**
```
Executor (data_provider):
  "Đã cào 15,000 listings cho 'Book'.
   Dữ liệu này quá lớn, phân tích hết sẽ tốn nhiều token API.
   Sếp muốn:
   [A] Phân tích cả 15,000 dòng
   [B] Chỉ lấy 1,000 dòng mới nhất trong tháng này
   [C] Sample ngẫu nhiên 2,000 dòng"

  → interrupt(message, options=[A, B, C])
  → Hệ thống tạm ngưng, gửi cho human
  → Human chọn [B]
  → Executor resume, xử lý 1,000 dòng mới nhất
```

**Khác với approval thông thường:** Interrupt không phải "xin phép" — nó là "phát hiện ambiguity, cần human judgment". Agent vẫn tự quyết các việc rõ ràng, chỉ dừng khi cần.

### First Application: Market Researcher

Ứng dụng đầu tiên chạy trên harness này — tự động hóa toàn bộ quy trình nghiên cứu thị trường.

Pipeline: `Tiếp nhận yêu cầu → data_provider (crawl + clean) → analyze_pricing (stats + recommendation) → In kết quả`

**v0.1 flow (financial-research pattern + skip Reviewer):**

```
        Orchestrator (sống suốt session)
              │
              ├─ Stage 1: data_provider ─────────────────────────────────┐
              │   ├─ spawn Executor                                       │
              │   │   └─ LLM calls search_listings(platform, query) ← 1 CALL
              │   │       Handler crawls 5 URLs, LLMExtractionStrategy,
              │   │       aggregates, returns ScrapeResult
              │   ├─ Auto-wrap: {status:"done", output, _handler_produced:true}
              │   └─ _handler_produced? YES → **skip Reviewer** → [OK]
              │                                                           │
              ├─ Stage 2: analyze_pricing ───────────────────────────────┤
              │   ├─ spawn Executor                                       │
              │   │   └─ LLM calls analyze_prices(dataset) ← 1 CALL
              │   │       Handler computes stats, outliers, segments,
              │   │       recommendation text
              │   ├─ Auto-wrap: {status:"done", output, _handler_produced:true}
              │   └─ _handler_produced? YES → **skip Reviewer** → [OK]
              │                                                           │
              └─ Hiển thị recommendation text cho human                   │
```

**Reviewer chỉ được gọi khi Executor output là do LLM sinh ra** (không có handler, hoặc handler không return complete result). Khi handler.py code tự produce output → trust & skip.

### Reference Implementations

Two reference codebases are studied for patterns — neither is a fork or dependency.

#### Reference 1: Superpowers (skill system + methodology)

`../ref/superpowers/` — [obra/superpowers](https://github.com/obra/superpowers) (v6.0.3, MIT). A composable skill framework for coding agents, supporting 10+ harnesses.

**Patterns to borrow:**
- **Skill format**: `SKILL.md` with YAML frontmatter (`name`, `description`) — simple, version-controllable
- **Action vocabulary**: Skills describe *actions* ("dispatch a subagent", "read a file") never concrete tool names
- **Tool mapping**: Per-harness translation layer (`references/<harness>-tools.md`)
- **Bootstrap injection**: At session start, inject the `using-superpowers` skill content so agents know skills exist
- **Implementer + Reviewer pattern**: Dual-agent per task with adversarial verification

See `superpowers/docs/porting-to-a-new-harness.md` — effectively an architecture spec for building a harness.

**Limitations for llend_harness:** No runtime (relies on host agents), coding-only domain, markdown-only skills, no HITL, no observability.

#### Reference 2: Financial Research Analyst (multi-agent Python app)

`../ref/financial-research-analyst-agent/` — a hierarchical multi-agent system for stock analysis. 14 agents (Orchestrator + 13 specialist analysts) built on LangChain/LangGraph.

**Patterns to borrow:**

| Pattern | Source file | What to extract |
|---------|-----------|-----------------|
| **BaseAgent ABC** | `src/agents/base.py` | Abstract agent with `execute()`, `execute_sync()`, tool binding, state management, result standardization |
| **AgentState/AgentResult** | `src/agents/base.py:27-66` | Pydantic models for agent lifecycle tracking (status, confidence, timing, errors) |
| **Multi-provider LLM factory** | `src/agents/base.py:125-189` | Provider-agnostic LLM creation (Ollama, LM Studio, vLLM, Groq, OpenAI, Anthropic) |
| **Provider protocol + fallback** | `src/data/provider.py` | Abstract protocol → concrete implementations → fallback chain. Textbook dependency inversion |
| **ReAct prompt injection** | `src/agents/base.py:312-356` | Structured 5-step reasoning protocol injected into every agent prompt |
| **Confidence scoring** | `src/agents/base.py:358-377` | Regex extraction of confidence scores from agent output |
| **RAG mixin** | `src/agents/rag_mixin.py` | Optional capability pattern: agent gets document awareness with graceful degradation |
| **Pydantic config** | `src/config.py` | Nested BaseSettings with env var overrides, sub-configs per module |
| **Tool decorator pattern** | `src/tools/*.py` | LangChain `@tool` pattern for defining agent-callable functions |

**Limitations for llend_harness:** No harness abstraction (all logic is app-specific), hardcoded agent instantiation, no plugin/skill system, direct method calls instead of message bus, no hierarchical spawning, no HITL interrupts.

#### Reference 3: Crawl4AI (scraping engine for data_provider skill)

`../ref/crawl4ai/` — [unclecode/crawl4ai](https://github.com/unclecode/crawl4ai) (v0.9.0, Apache 2.0 with attribution). An open-source, LLM-friendly web crawler and scraper. This is NOT a pattern reference — it's a **direct dependency** for the Market Researcher's `data_provider` skill.

**Core API (how data_provider will use it):**
```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai import JsonCssExtractionStrategy, LLMExtractionStrategy

async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
    config = CrawlerRunConfig(
        extraction_strategy=JsonCssExtractionStrategy({...}),
        cache_mode=CacheMode.BYPASS
    )
    result = await crawler.arun(url="https://ebay.com/...")
    print(result.markdown)         # LLM-ready clean markdown
    print(result.structured_data)  # Extracted JSON
```

**Key capabilities data_provider will leverage:**
- Async architecture (asyncio + Playwright) — matches llend_harness runtime
- Structured extraction: CSS selector, XPath, LLM-driven (`JsonCssExtractionStrategy`, `LLMExtractionStrategy`)
- Anti-bot: stealth mode, proxy support, session management, rate limiting
- Deep crawl: BFS/DFS/Best-First strategies for multi-page listing collection
- Batch operations: `arun_many()` with concurrency control
- Output: clean markdown + structured JSON — LLM-ready

**What crawl4AI does NOT provide (must build in data_provider skill):**

| Gap | What to build |
|-----|--------------|
| ❌ No eBay crawler | eBay listing parser (title, price, condition, seller, shipping) |
| ❌ No eBay pagination | Custom pagination handler (eBay's "Next" button, URL pattern) |
| ❌ No eBay anti-detection tuning | eBay-specific rate limits, session rotation, CAPTCHA handling |
| ❌ No data normalization | Clean + validate listing fields, dedup, outlier detection |
| ❌ No market-specific extraction | Schemas for product categories (electronics, books, fashion...) |

**Architecture fit:**
```
data_provider skill (code-based Python skill)
    │
    ├─ Crawl4AI engine        ← async crawl + extraction
    ├─ eBay extraction schema  ← custom JsonCssExtractionStrategy
    ├─ eBay pagination logic   ← custom BFS deep crawl config
    ├─ Anti-bot config         ← proxy + stealth + rate limits
    └─ Data normalizer         ← clean, validate, dedup → output clean dataset
```

**License note:** Apache 2.0 with attribution requirement — must include attribution in any public use/distribution.

### Where llend_harness Diverges

Neither reference alone is sufficient. llend_harness synthesizes both:

| Dimension | Superpowers | Financial Research Analyst | llend_harness |
|-----------|------------|--------------------------|---------------|
| Runtime | None (hosted) | LangChain/LangGraph | Python asyncio event loop + message bus |
| Skill/Plugin system | ✅ Markdown skills | ❌ None | ✅ Code + Markdown skills, registry |
| Agent topology | Implicit (host agent spawns) | Hardcoded 14 agents | Configurable N-agent, dynamic spawn |
| Agent communication | Context injection | Direct method calls | Async message bus with routing |
| Domain | Coding only | Finance only | Domain-agnostic |
| HITL | Manual (human in chat) | None | Interrupt node (LangGraph-style) |
| Adversarial verify | ✅ Reviewer agent | ❌ None | ✅ Reviewer agent (from superpowers) |
| Testing | Manual tmux | None | Automated skill test harness |
| Observability | None | Logging | Structured logging + OpenTelemetry |

### Architecture (v0.1 — implemented)

```
llend/
├── __main__.py           # CLI entry: python -m llend
├── __init__.py
├── settings.toml         # Global config (orchestrator, execution, llm, session)
├── runtime/              # Spec 001 — asyncio event loop, message bus, lifecycle
│   ├── base.py           #   AgentRuntime ABC
│   ├── asyncio_runtime.py #  v0 primary backend
│   ├── langgraph_runtime.py # Future LangGraph backend
│   ├── message.py        #   Message envelope, MsgType, enums
│   ├── lifecycle.py      #   AgentState, AgentType
│   ├── checkpoint.py     #   Interrupt checkpoint persistence
│   └── notifications.py  #   Human notification channels
├── registry/             # Spec 002 — skill discovery, validation, resolution
│   ├── registry.py       #   SkillRegistry
│   ├── models.py         #   SkillMeta, Skill, ActionBinding, ValidationIssue
│   ├── pipeline.py       #   SkillPipeline, TaskSpec, ExecutionPlan
│   ├── parser.py         #   Input parser (skill.md frontmatter)
│   ├── validator.py      #   Skill validation logic
│   └── action_dispatcher.py # ActionDispatcher (global + custom routing)
├── tool_bridge/          # Spec 002 — global action→tool mapping
│   ├── bridge.py         #   ToolBridge + auto input_schema from signatures
│   └── mappings.toml     #   action → module.function bindings
├── skills/               # Skill definitions (SKILL.md + handler.py + models.py)
│   ├── data_provider/    #   search_listings (financial-research pattern)
│   └── analyze_pricing/  #   analyze_prices (financial-research pattern)
├── orchestrator/         # Spec 004 — central hub: classify, plan, dispatch, adjudicate
│   ├── agent.py          #   OrchestratorAgent (main loop, session lifecycle)
│   ├── classifier.py     #   Message classification (task vs conversational)
│   ├── executor.py       #   Task execution loop (extracted module)
│   ├── adjudicator.py    #   Verdict adjudication + retry logic
│   ├── summarizer.py     #   TaskResultSummary + final synthesis
│   ├── wiring.py         #   Input/output wiring + auto-unwrap
│   ├── gate.py           #   Responder tool approval gate
│   ├── recovery.py       #   Error recovery + exponential backoff
│   ├── progress.py       #   Progress reporting
│   ├── session.py        #   Session state, start/complete lifecycle
│   └── config.py         #   OrchestratorConfig (Pydantic model)
├── executor/             # Spec 005 — ReAct loop, tool-use, auto-wrap
│   ├── agent.py          #   ExecutorAgent (ReAct loop, _auto_wrap_from_tools)
│   └── __init__.py
├── reviewer/             # Spec 001/004 — adversarial verification
│   ├── agent.py          #   ReviewerAgent (single LLM call, verdict)
│   └── __init__.py
├── responder/            # Spec 003 — conversational Q&A
│   ├── agent.py          #   ResponderAgent
│   ├── context.py        #   SessionContext, TaskResultSummary
│   ├── persona.py        #   Persona enum, prompts
│   ├── memory.py         #   UserProfile load/save
│   └── stream.py         #   Streaming chunk assembly
├── parsers/              # Web fetcher, HTML parser, CSV exporter
│   ├── web_fetcher.py    #   crawl4ai integration, URL cache, accumulator
│   ├── html_parser.py    #   CSS + LLM extraction strategies
│   └── csv_exporter.py   #   CSV export wrapper
├── llm/                  # LLM provider abstraction
│   └── client.py         #   AnthropicClient, DeepSeekClient, create_llm_client()
├── docs/specs/           # Spec documents (001-006)
└── tests/                # Test suite (pytest)
```

### Key Design Decisions (so far)

1. **Python-native**: Runtime, not a plugin for another agent. Python chosen for ecosystem (asyncio, Pydantic, pytest, OpenTelemetry).
2. **Action vocabulary**: Inherit the superpowers pattern — skills never name concrete tools.
3. **Code + Markdown skills**: Simple skills are markdown-only; complex skills can include Python handlers.
4. **Configurable enforcement**: Unlike superpowers' "mandatory only" approach, llend_harness supports levels.
5. **Batteries-included testing**: Skill authors get a test harness — mock agent context, assert skill behavior.

### v0.1 Patterns (Implemented)

**Financial-Research Pattern (1 tool = 1 complete unit of work):**
Skills expose high-level actions, not low-level primitives. `data_provider` has ONE action `search_listings(platform, query)` that internally crawls 3-5 URLs, extracts via LLMExtractionStrategy, aggregates, and returns a complete `ScrapeResult`. The Executor LLM calls ONE action — no URL construction, no aggregation, no dedup.

**Auto-Wrap & Skip Reviewer:**
When `handler.py` produces the output (not LLM), the Executor auto-detects complete results via `_auto_wrap_from_tools()` and wraps them as `{status: "done", output: <result>, concerns: [], _handler_produced: true}`. The Orchestrator sees `_handler_produced: true` → **skips Reviewer entirely**. Rationale: handler code is trusted — no hallucination risk. Reviewer adds no value for code-produced output.

**LLMExtractionStrategy (unknown sites):**
For unknown e-commerce sites, `web_fetcher.py` uses crawl4ai's `LLMExtractionStrategy` with schema + instruction + `force_json_response`. Known sites (eBay, Amazon) use fast `JsonCssExtractionStrategy` (CSS). Vietnamese price format handling: `12.990.000₫` → `12990000`.

**URL Cache + Accumulator:**
Module-level `_url_cache`, `_accumulated_listings`, `_accumulated_errors` in `web_fetcher.py`. Each `fetch_web_page` call adds to the accumulator. `clear_cache()` between searches.

**Custom Action `input_schema` Auto-Generate:**
`ToolBridge.validate_mapping()` and `SkillRegistry._resolve_custom_bindings()` auto-generate JSON Schema from Python function signatures (`inspect.signature` + type annotations). Supports `list[float]` → `array items: number`.

### Branch → Spec Mapping

| Branch | Spec | Component |
|--------|------|-----------|
| `feat/runtime-core` | [001](docs/specs/001-message-protocol-runtime-core.md) | Message protocol, AgentRuntime, lifecycle |
| `feat/skill-registry` | [002](docs/specs/002-skill-format-registry.md) | Skill format, registry, tool bridge, pipeline |
| `feat/responder-agent` | [003](docs/specs/003-responder-agent-conversation-module.md) | Responder, conversation, persona |
| `feat/orchestrator` | [004](docs/specs/004-orchestrator-logic.md) | Orchestrator, classify, dispatch, adjudicate |
| `feat/executor-reviewer` | [005](docs/specs/005-executor-reviewer-cli.md) | Executor, Reviewer, LLM providers, CLI bootstrap |
| (none yet) | [006](docs/specs/006-testing.md) | Testing & skill test harness |

### Git Workflow Rules

1. **KHÔNG code thẳng trên `master`.** Mỗi component có branch riêng (xem bảng trên). Khi cần sửa code, xác định code đó thuộc spec/component nào → checkout branch tương ứng → sửa ở đó.
2. **Sửa spec trước, code sau.** Mọi thay đổi behavior phải được document trong spec doc tương ứng TRƯỚC KHI viết code. Spec là source of truth.
3. **Merge về master sau khi xong.** Khi feature hoàn tất trên branch → merge vào master.
4. **Branch naming:** `feat/<component-name>` cho feature branches, khớp với spec number.
