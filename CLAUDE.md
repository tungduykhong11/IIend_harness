# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: llend_harness

A Python-native **Hierarchical Multi-Agent Harness** — a runtime that orchestrates AI agents through composable skills. Domain-agnostic: not tied to coding workflows. Currently in the design/research phase; no code yet.

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

Pipeline: `Tiếp nhận yêu cầu → Crawl + Làm sạch dữ liệu → Phân tích giá → Xuất báo cáo Excel`

**Mỗi stage** có pattern: Orchestrator spawn Executor → đợi output → spawn Reviewer verify → pass thì tiếp, fail thì loop. Bất kỳ agent nào cũng có thể raise interrupt nếu gặp ambiguity:

```
        Orchestrator (sống suốt session)
              │
              ├─ Stage 1: data_provider ──────────────────────────────────┐
              │   ├─ spawn Executor: crawl + clean                        │
              │   │   ├─ "Đã cào 15,000 listings, quá lớn"               │
              │   │   ├─ ⚡ INTERRUPT: "Phân tích hết hay lọc 1,000?"    │
              │   │   ├─ Human: "Lọc 1,000 dòng mới nhất"               │
              │   │   └─ Resume: lọc + clean 1,000 dòng                  │
              │   ├─ spawn Reviewer: verify dữ liệu                      │
              │   │   ├─ đủ listing? outlier? field hợp lệ?              │
              │   │   ├─ pass → tiếp                                     │
              │   │   └─ fail → loop Executor mới                        │
              │   └─ output: clean dataset                               │
              │                                                           │
              ├─ Stage 2: analyze_pricing ────────────────────────────────┤
              │   ├─ spawn Executor: phân tích giá                        │
              │   ├─ spawn Reviewer: verify phân tích                     │
              │   │   ├─ insight có evidence? số mâu thuẫn?              │
              │   │   ├─ pass → tiếp                                     │
              │   │   └─ fail → loop Executor mới                        │
              │   └─ output: pricing insights                             │
              │                                                           │
              ├─ Stage 3: write_report ───────────────────────────────────┤
              │   ├─ spawn Executor: tạo Excel                            │
              │   ├─ spawn Reviewer: verify báo cáo                       │
              │   │   ├─ đúng format? chart chính xác?                   │
              │   │   ├─ pass → done                                     │
              │   │   └─ fail → loop Executor mới                        │
              │   └─ output: report.xlsx                                  │
              │                                                           │
              └─ Tổng hợp → trả kết quả cho human                         │
```

**Tại mỗi stage:** Executor làm, Reviewer kiểm — 2 mindset khác nhau. Reviewer có quyền bắt làm lại nếu output không đạt spec.

**Tại sao Reviewer được gọi sau mỗi Executor thay vì để Orchestrator tự kiểm?**
- Reviewer cần **adversarial mindset** ("refute đi", "có chắc số này đúng không?") — khác hẳn mindset "quản lý" của Orchestrator
- Giảm context overload cho Orchestrator
- Reviewer cũng disposable, fresh context → không bias bởi output của stage trước

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

### Architecture Direction (draft)

```
llend_harness/
├── runtime/          # Core event loop, task scheduler, message bus
├── skills/           # Skill definitions (SKILL.md + optional code)
│   └── data_provider/  # Market Researcher scraping skill (uses crawl4AI)
├── registry/         # Skill discovery, versioning, dependencies
├── tool_bridge/      # Action vocabulary → concrete tool mappings (crawl4AI registered here)
├── bootstrap/        # Session initialization, context injection
├── telemetry/        # Logging, tracing, metrics
├── testing/          # Skill test harness
├── plugins/          # Harness adapter plugins
└── docs/             # Design docs, specs

Dependencies (for Market Researcher):
├── crawl4ai/         # Scraping engine (Apache 2.0) — used by data_provider skill
├── openpyxl/         # Excel report generation (write_report skill)
└── pandas/           # Data manipulation (shared across skills)
```

### Key Design Decisions (so far)

1. **Python-native**: Runtime, not a plugin for another agent. Python chosen for ecosystem (asyncio, Pydantic, pytest, OpenTelemetry).
2. **Action vocabulary**: Inherit the superpowers pattern — skills never name concrete tools.
3. **Code + Markdown skills**: Simple skills are markdown-only; complex skills can include Python handlers.
4. **Configurable enforcement**: Unlike superpowers' "mandatory only" approach, llend_harness supports levels.
5. **Batteries-included testing**: Skill authors get a test harness — mock agent context, assert skill behavior.
