# Spec 002: Skill Format & Registry

**Status:** Draft
**Date:** 2026-06-27
**Author:** Human + Claude
**Depends on:** [Spec 001 — Message Protocol & Runtime Core](./001-message-protocol-runtime-core.md)

---

## 1. Scope

This spec defines:

- **Skill Format** — how a skill is declared (skill.md + optional handler.py), its metadata, inputs, outputs, and actions.
- **Action Vocabulary** — the abstraction layer between skills and concrete tool implementations. Skills declare *what they need* (actions), never *how to do it* (tools).
- **Tool Bridge** — global mapping from actions → concrete implementations. Swap implementations without touching skill code.
- **Skill Registry** — discovery, validation, version resolution, and action binding resolution.
- **Skill Pipeline** — dependency resolution and execution plan generation (so Orchestrator stays lean).
- **Pydantic Models** — all skill I/O is typed with Pydantic so LLM outputs are validated & structured.

Out of scope: skill execution (belongs to Executor, Spec 001), skill authoring tooling (future spec), telemetry.

---

## 2. Skill Format

### 2.1 Directory Layout

```
skills/
└── analyze_pricing/
    ├── skill.md          # Required — skill definition (YAML frontmatter + markdown body)
    ├── handler.py        # Optional — custom Python actions
    └── models.py         # Optional — Pydantic models for typed I/O
```

**Rule:**
- `skill.md` only → simple skill. Executor/LLM follows markdown instructions.
- `skill.md` + `handler.py` → complex skill. Custom Python logic for domain-specific actions.
- `+ models.py` → typed skill. Input/output validated with Pydantic. **Recommended for all skills.**

### 2.2 skill.md Format

```markdown
---
name: analyze_pricing
version: 0.1.0
description: Analyze pricing from a product dataset — compute median, detect outliers, segment by price brackets.
inputs: dataset:list[dict], target_item:str, brackets:list[int]=[0,300,500,1000]
outputs: AnalysisReport
actions:
  - export_csv
  - calculate_market_median
  - detect_outliers_iqr
  - segment_by_price_range
dependencies:
  - data_provider
enforcement: strict
---

# Analyze Pricing Skill

## Flow
1. Receive dataset from `data_provider`
2. Compute market median price
3. Detect outlier listings (suspicious cheap & suspicious expensive)
4. Segment products by configurable price brackets
5. Export segments to CSV
6. Return `AnalysisReport` with recommendations

## Notes
- Outlier detection uses IQR method (Q1 - 1.5×IQR / Q3 + 1.5×IQR).
- If >20% of listings are outliers, raise `interrupt` to ask human whether to filter or keep.
- Default brackets [0, 300, 500, 1000] work for mid-range electronics. Adjust for luxury goods.
```

### 2.3 Frontmatter Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `name` | ✅ | string | Unique skill identifier. snake_case. |
| `version` | ✅ | string | Semver. Registry resolves latest by default. |
| `description` | ✅ | string | One-line summary. Shown in skill list. |
| `inputs` | ✅ | string | Comma-separated `name:type` or `name:type=default`. Types reference Pydantic models or primitives. |
| `outputs` | ✅ | string | Return type name. Must match a Pydantic model in `models.py` or be a primitive (`str`, `list[dict]`, `None`). |
| `actions` | ✅ | list[string] | Actions this skill needs. Can reference global tool bridge actions AND custom handler methods. |
| `dependencies` | ❌ | list[string] | Skills whose outputs feed into this skill's inputs. Resolved by SkillPipeline. |
| `enforcement` | ❌ | string | `suggested` (default), `strict`, or `mandatory`. See §8. |

### 2.4 Input Type Syntax

```
inputs: name:type=default, name:type, name:type=default
```

Examples:
```yaml
# Simple primitives
inputs: query:str, max_items:int=100, sort_by:str="price_asc"

# Pydantic model reference
inputs: dataset:list[ProductListing], config:AnalysisConfig

# Mixed
inputs: platform:str, raw_data:list[dict], report_config:ReportConfig=ReportConfig()
```

**Parsing rules** (YAML frontmatter string → `dict[str, str]`):

| Rule | Input | Output |
|------|-------|--------|
| Split on `,` (comma) | `a:str, b:int=5` | `["a:str", "b:int=5"]` |
| Split each on first `:` | `a:str` | key=`a`, type=`str` |
| Split type on `=` | `int=5` | type=`int`, default=`5` |
| Quoted defaults preserved | `str="hello, world"` | key=`msg`, type=`str`, default=`"hello, world"` (commas inside quotes NOT split) |
| Trailing whitespace trimmed | `a:str , b:int` | `{"a": "str", "b": "int"}` |

```python
# registry/parser.py
import re

def parse_inputs(raw: str) -> dict[str, str]:
    """Parse 'name:type=default, name:type' into {name: type_spec}."""
    if not raw.strip():
        return {}
    result = {}
    # Split on commas not inside quotes
    parts = re.split(r',\s*(?=(?:[^"]*"[^"]*")*[^"]*$)', raw)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        key, _, type_spec = part.partition(":")
        result[key.strip()] = type_spec.strip()
    return result
```

---

## 3. Pydantic Models (Typed I/O)

### 3.1 Why Pydantic

LLM text output is unstructured by nature. Pydantic models serve dual purpose:

1. **Contract** — LLM knows exactly what shape to produce (schema injected into system prompt).
2. **Validation** — Runtime catches malformed output immediately, triggers retry or review.
3. **Documentation** — `models.py` is self-documenting. Skill authors and LLMs both read it.

### 3.2 models.py Convention

```python
# skills/analyze_pricing/models.py
from pydantic import BaseModel, Field
from typing import Optional

class MarketSummary(BaseModel):
    """Tổng quan thị trường."""
    median_price: float = Field(..., description="Median price across all valid listings")
    mean_price: float = Field(..., description="Mean price")
    min_price: float
    max_price: float
    normal_range: tuple[float, float] = Field(..., description="[lower_bound, upper_bound] — IQR normal range")
    total_listings: int
    outlier_count: int = Field(..., description="Total suspicious listings detected")

class OutlierDetail(BaseModel):
    """Một listing bất thường."""
    index: int
    price: float
    title: str
    reason: str = Field(..., description="Why this listing is flagged (e.g. 'Below Q1-1.5*IQR')")

class OutlierReport(BaseModel):
    """Báo cáo outlier."""
    suspicious_cheap: list[OutlierDetail] = Field(default_factory=list)
    suspicious_expensive: list[OutlierDetail] = Field(default_factory=list)
    cheap_count: int
    expensive_count: int

class PriceSegment(BaseModel):
    """Một khoảng giá."""
    range_label: str = Field(..., description='e.g. "0-300", "1000+"')
    lower: float
    upper: Optional[float] = None   # None = unlimited upper
    count: int
    avg_price: float
    sample_products: list[str] = Field(default_factory=list, description="Top 5 product titles in this segment")

class AnalysisReport(BaseModel):
    """Output chính của analyze_pricing."""
    market_summary: MarketSummary
    outliers: OutlierReport
    price_segments: list[PriceSegment]
    recommendation: str = Field(..., description="Human-readable buying advice based on the data")
    export_csv_path: Optional[str] = None
```

### 3.3 How Models Reach the LLM

When Executor receives `task.dispatch`, the `skill_context` includes JSON Schema generated from Pydantic models.

**How Registry resolves the output model:**

1. skill.md declares `outputs: AnalysisReport`
2. Registry imports `skills/analyze_pricing/models.py`
3. Registry looks up the class named `AnalysisReport` in the module
4. Verifies it's a `BaseModel` subclass
5. Calls `AnalysisReport.model_json_schema()` → injects into `skill_context.output_schema`

If `outputs` references a model that doesn't exist in `models.py` (or `models.py` doesn't exist), and it's not a primitive type — `ValidationIssue(severity="error", ...)` is raised.

```python
# Registry generates this on resolve():
skill_context = {
    "skill_md": "...",            # raw skill.md content
    "allowed_actions": [...],
    "action_bindings": {...},
    "output_schema": AnalysisReport.model_json_schema(),   # ← from models.py, resolved by name
    "input_schemas": {                                      # ← parsed from skill.md inputs:
        "dataset": {"type": "list[dict]"},                  #   primitive type (no Pydantic model)
        "target_item": {"type": "str"},                     #   primitive type
        "brackets": {"type": "list[int]", "default": [0,300,500,1000]},  # primitive with default
    },
}
```

> **Note:** `input_schemas` contains both Pydantic-generated JSON Schemas (when a model like `AnalysisConfig` is referenced) and simple type descriptors (when primitives like `str`, `list[dict]` are used). The Executor uses these to understand what data it receives, not for validation (inputs come from upstream tasks, already validated by their output schemas).

Executor's system prompt gets:

```
## Expected Output Schema
Your final output MUST conform to this JSON Schema. Wrap it in ```json```:
{
  "market_summary": {
    "median_price": float,
    "mean_price": float,
    ...
  },
  ...
}
```

On `task.result`, Orchestrator validates: `AnalysisReport.model_validate(executor_output)` → if ValidationError → auto-retry or send to Reviewer.

### 3.4 Primitive Outputs

Skills without `models.py` declare primitive outputs:

```yaml
outputs: str          # free-text report
outputs: list[dict]   # raw listing array
outputs: None         # side-effect only skill (e.g., send_email)
```

No schema validation for primitives — Reviewer is the safety net.

---

## 4. Action Vocabulary

### 4.1 The Golden Rule

> **Skills declare actions, never tools.**

| ❌ Wrong | ✅ Right |
|----------|----------|
| `crawl4ai.AsyncWebCrawler.arun(url)` | `fetch_web_page(url)` |
| `BeautifulSoup(html).find_all(".listing")` | `parse_listing_html(html, schema)` |
| `pandas.DataFrame.to_csv(df, path)` | `export_csv(data, filename)` |

This keeps skills **harness-agnostic**. The same `analyze_pricing` skill works whether the tool bridge maps `export_csv` to pandas, polars, or a custom SQL exporter.

### 4.2 Action Sources

```
┌──────────────────────────────────────────────┐
│              Action Resolution                │
│                                               │
│  Global Tool Bridge (mappings.toml)           │
│  ├── fetch_web_page     → crawl4ai            │
│  ├── parse_html         → beautifulsoup4       │
│  ├── export_csv         → pandas               │
│  └── export_excel       → openpyxl             │
│                                               │
│  Custom (handler.py per skill)                │
│  ├── calculate_market_median  (analyze_pricing)│
│  ├── detect_outliers_iqr      (analyze_pricing)│
│  ├── segment_by_price_range   (analyze_pricing)│
│  └── detect_trend_pattern      (trend_analysis)│
└──────────────────────────────────────────────┘
```

When a skill declares:
```yaml
actions:
  - export_csv              # resolved from global tool bridge
  - calculate_market_median # resolved from handler.py methods
  - detect_outliers_iqr     # resolved from handler.py methods
```

Registry merges both sources into `action_bindings` in the dispatch payload (§9).

### 4.3 Custom Action Discovery

Registry scans `handler.py` for public async methods with docstrings:

```python
# handler.py
class AnalyzePricingSkill:
    """Handler for analyze_pricing."""

    # ✅ Discovered as action "calculate_market_median"
    async def calculate_market_median(self, prices: list[float]) -> dict:
        """
        Compute median and basic price statistics.
        INPUT: prices: list[float]  — list of prices from dataset
        OUTPUT: {median: float, mean: float, min: float, max: float, count: int}
        """
        ...

    # ✅ Discovered as action "detect_outliers_iqr"
    async def detect_outliers_iqr(self, prices: list[float]) -> dict:
        """
        Detect price outliers using IQR method.
        ...
        """

    # ❌ Not discovered — starts with _
    async def _internal_helper(self, x: float) -> float:
        ...

    # ❌ Not discovered — no docstring
    async def utility_func(self, data: list) -> list:
        ...
```

**Discovery rule:** Public method (no leading `_`) + has docstring → registered as custom action. Method name becomes action name.

**Handler class discovery convention:**

Registry identifies the handler class in `handler.py` as follows:

1. Look for a class named `<PascalCaseSkillName>Skill` (e.g., `analyze_pricing` → `AnalyzePricingSkill`)
2. If not found, look for the **first** class that has at least one public async method with a docstring
3. If no such class exists → `ValidationIssue(severity="warning", field="handler.py", message="No handler class found")`
4. If multiple classes match rule #2 → use the first one, emit warning about ambiguity

```python
# Example: skill "analyze_pricing" → looks for AnalyzePricingSkill first
class AnalyzePricingSkill:  # ✅ Exact match
    async def calculate_market_median(self, ...): ...

# Fallback example: skill "data_provider" → no DataProviderSkill class,
# uses first class with discoverable methods
class EbayScraper:          # ✅ First match (no exact name match)
    async def fetch_web_page(self, ...): ...
```

### 4.4 Action Execution Model

**How Executor invokes an action at runtime.**

Actions declared in `allowed_actions` are exposed to the Executor (LLM) as **tool definitions** using the standard LLM function-calling format. The Executor does NOT send messages through the bus for each action call — that would be far too chatty. Instead:

```
Executor Process
┌─────────────────────────────────────────┐
│  LLM (Claude / GPT)                     │
│    │                                     │
│    │  tool_use: "calculate_market_median"│
│    │  arguments: {prices: [299, 350...]} │
│    ▼                                     │
│  Action Dispatcher                       │
│    │                                     │
│    ├── source == "global"                │
│    │   → import tool (e.g. pandas)       │
│    │   → call function (e.g. to_csv)     │
│    │                                     │
│    └── source == "custom"                │
│        → lookup handler instance         │
│        → call method (e.g. calculate...) │
│    │                                     │
│    ▼                                     │
│  Return result → back to LLM context     │
└─────────────────────────────────────────┘
```

**Why not messages for action calls?** Actions are deterministic, fast, and local. Wrapping them in messages would add serialization overhead with zero benefit — there's no routing decision to make, no human to interrupt. The message bus is for *agent-to-agent* communication; action calls are *intra-agent* function calls.

**How it works step by step:**

1. Executor's system prompt includes tool definitions generated from `action_bindings`:
   ```json
   {
     "tools": [
       {
         "name": "calculate_market_median",
         "description": "Compute median and basic price statistics.",
         "input_schema": {
           "type": "object",
           "properties": {
             "prices": {"type": "array", "items": {"type": "number"}}
           },
           "required": ["prices"]
         }
       },
       {
         "name": "export_csv",
         "description": "Export data to CSV file.",
         "input_schema": {
           "type": "object",
           "properties": {
             "data": {"type": "array"},
             "filename": {"type": "string"}
           },
           "required": ["data", "filename"]
         }
       }
     ]
   }
   ```

2. LLM decides to call a tool → emits `tool_use` block with action name + arguments.

3. **ActionDispatcher** (inside Executor process):
   ```python
   class ActionDispatcher:
       """Resolves and executes action calls within an Executor."""

       def __init__(self, action_bindings: dict[str, ActionBinding], handler: object | None):
           self._bindings = action_bindings
           self._handler = handler  # skill's handler instance (if handler.py exists)

       async def dispatch(self, action_name: str, arguments: dict) -> Any:
           binding = self._bindings[action_name]
           if binding.source == "global":
               # Import tool module, call function
               mod = importlib.import_module(binding.tool)
               func = getattr(mod, binding.function)
               return await func(**arguments)
           elif binding.source == "custom":
               # Call handler method
               method = getattr(self._handler, binding.function)
               return await method(**arguments)
   ```

4. Result is injected back into LLM context. LLM continues reasoning → may call more actions → eventually produces final output.

5. If an action raises an exception → ActionDispatcher catches it, returns error to LLM context. LLM can retry or adapt. If retries exhausted → Executor sends `agent.error` to Orchestrator.

**Action timeout & retry** are enforced by ActionDispatcher using `asyncio.wait_for()` and the `timeout_ms`/`retry` fields from `ActionBinding`.

**Consequence for tool_bridge validation:** Since ActionDispatcher does `importlib.import_module(tool)`, `ToolBridge.validate_mapping()` MUST verify the module is importable at registry startup time (fail-fast, not mid-execution).

---

## 5. Tool Bridge

### 5.1 Configuration

`tool_bridge/mappings.toml` — single source of truth for global action→tool bindings:

```toml
[actions.fetch_web_page]
tool = "crawl4ai"
function = "AsyncWebCrawler.arun"
timeout_ms = 30000
retry = 3

[actions.fetch_web_page.config]
stealth_mode = true
user_agent = "llend-harness/0.1"

[actions.parse_listing_html]
tool = "llend_harness.parsers"
function = "parse_product_listing"
timeout_ms = 5000

[actions.export_csv]
tool = "pandas"
function = "DataFrame.to_csv"

[actions.export_csv.config]
index = false
encoding = "utf-8-sig"

[actions.export_excel]
tool = "openpyxl"
function = "Workbook.save"
```

### 5.2 ToolBridge Class

```python
# tool_bridge/bridge.py
from pathlib import Path
from registry.models import ActionBinding  # canonical definition in §6.2

class ToolBridge:
    """Resolves global actions → concrete tool implementations."""

    def __init__(self, mappings_path: Path): ...

    def resolve(self, action_name: str) -> ActionBinding | None:
        """Look up a global action. Returns None if not found."""
        ...

    def resolve_all(self) -> dict[str, ActionBinding]:
        """All registered global actions with their bindings."""
        ...

    def list_actions(self) -> list[str]:
        """All registered global action names."""
        ...

    def validate_mapping(self, action_name: str) -> bool:
        """Check the mapped tool is importable and the function exists.
        Tries: importlib.import_module(tool) → getattr(module, function).
        Returns True if importable, False otherwise."""
        ...
```

### 5.3 Why TOML + Python (not just TOML)

TOML for static mappings (human-editable). Python `ToolBridge` class wraps it with:
- Import validation at startup (fail fast if `crawl4ai` is not installed)
- Config merging (TOML base + env var overrides + per-skill overrides)
- Hot-reload: watcher on `mappings.toml` re-validates without restart

---

## 6. Skill Registry

### 6.1 Registry Class

```python
# registry/registry.py
class SkillRegistry:
    """Discover, validate, and resolve skills."""

    def __init__(self, skills_dir: Path, tool_bridge: ToolBridge): ...

    # ── Discovery ──────────────────────────────

    async def discover(self) -> list[SkillMeta]:
        """Scan skills_dir for all skill.md files, parse YAML frontmatter.
        Returns metadata for all discovered skills. Does NOT validate yet."""
        ...

    # ── Validation ─────────────────────────────

    def validate(self, meta: SkillMeta) -> list[ValidationIssue]:
        """Check a skill is usable:
        1. All declared actions resolve (global tool bridge OR custom handler)
        2. Input/output types are parseable
        3. If models.py exists, output model is importable
        4. handler.py (if exists) is importable, has expected class, methods have docstrings
        5. Dependencies reference existing skill names
        Returns list of Issues (empty = valid)."""
        ...

    # ── Validation Behavior ──────────────────

    # validate() returns a list. The registry does NOT automatically reject
    # skills with issues — it calls validate() and lets the caller decide:
    #
    #   issues = registry.validate(meta)
    #   if any(i.severity == "error" for i in issues):
    #       # Skill cannot be resolved — resolve() will raise ResolutionError
    #       log.error(f"Skill {meta.name} has {len(issues)} errors, cannot dispatch")
    #   elif any(i.severity == "warning" for i in issues):
    #       # Skill can be resolved but may behave unexpectedly
    #       log.warning(f"Skill {meta.name} has warnings: {issues}")
    #
    # Errors (block resolve):
    #   - Unknown action with no handler.py to provide it
    #   - handler.py has syntax error / ImportError
    #   - models.py output model not found or not a BaseModel
    #   - Dependency references a skill not in registry
    #
    # Warnings (resolve still succeeds):
    #   - handler.py exists but some declared actions not found as methods
    #   - Deprecated action name used
    #   - No models.py (output will be untyped)

    # ── Resolution ─────────────────────────────

    def resolve(self, name: str, version: str | None = None) -> Skill:
        """Fully resolve a skill ready for dispatch.
        - If version is None, resolve latest.
        - Merge global tool bridge actions + custom handler actions.
        - Load Pydantic models if models.py exists.
        - Return Skill with all action_bindings populated."""
        ...

    def resolve_all(self) -> dict[str, Skill]:
        """Resolve all discovered skills. Keyed by name."""
        ...

    # ── Query ─────────────────────────────────

    def list_skills(self) -> dict[str, list[SkillMeta]]:
        """{category_dir: [SkillMeta, ...]} grouped by skill subdirectory."""
        ...

    def get(self, name: str) -> Skill | None:
        """Get a resolved skill by name."""
        ...

    # ── Hot Reload ────────────────────────────

    async def watch(self) -> None:
        """Start filesystem watcher on skills_dir.
        On change → re-discover affected skill → re-validate → update cache.
        Runs as background task in the asyncio event loop."""
        ...
```

### 6.2 Pydantic Models for Registry

```python
# registry/models.py
from pydantic import BaseModel
from pathlib import Path
from typing import Any, Literal

class SkillMeta(BaseModel):
    """Parsed from skill.md frontmatter."""
    name: str
    version: str
    description: str
    inputs: dict[str, str]              # {param_name: "type_spec"}
    outputs: str                        # "AnalysisReport" | "list[dict]" | "None"
    actions: list[str]                  # ["export_csv", "calculate_market_median", ...]
    dependencies: list[str] = []
    enforcement: Literal["suggested", "strict", "mandatory"] = "suggested"

class ValidationIssue(BaseModel):
    """Skill validation issue — distinct from Spec 001's ReviewIssue (Reviewer verdict)."""
    severity: Literal["error", "warning"]
    field: str                          # which part of the skill has the issue
    message: str

class ActionBinding(BaseModel):
    """Resolved action → implementation."""
    action_name: str
    source: Literal["global", "custom"]
    tool: str | None = None             # global: "crawl4ai" (None for custom)
    function: str                       # "AsyncWebCrawler.arun" or "calculate_market_median"
    handler_class: str | None = None    # custom: "AnalyzePricingSkill" (None for global)
    timeout_ms: int = 30000
    retry: int = 0
    config: dict[str, Any] = {}

class Skill(SkillMeta):
    """Fully resolved skill, ready for dispatch."""
    path: Path                          # skill directory path
    skill_md: str                       # raw skill.md content
    output_schema: dict | None = None   # JSON Schema from Pydantic model (None if no models.py)
    input_schemas: dict[str, dict]      # {param_name: json_schema}
    action_bindings: dict[str, ActionBinding]  # merged global + custom
    handler: object | None = None       # Python handler instance (None if no handler.py)
```

---

## 7. Skill Pipeline (Dependency Resolver)

### 7.1 Rationale

Orchestrator should NOT know how skills depend on each other. That knowledge belongs to the skill itself (via `dependencies`). The pipeline resolves the dependency graph and returns an ordered execution plan.

```
Orchestrator                    SkillPipeline
────────────                    ─────────────
"I need analyze_pricing"   →    Read dependencies: [data_provider]
                                → data_provider has no deps
                                → analyze_pricing needs output of data_provider
                                → Build plan:
                                  Step 1: data_provider
                                  Step 2: analyze_pricing (gets step1.output as input)
                                → Return [TaskSpec(step=1, ...), TaskSpec(step=2, ...)]
```

### 7.2 Pipeline Class

```python
# registry/pipeline.py
from pydantic import BaseModel

class TaskSpec(BaseModel):
    """One task in the execution plan."""
    step: int
    skill_name: str
    task_spec: dict[str, Any]          # task-specific params (merged from upstream outputs)
    input_from: list[str] | None = None  # skill names whose outputs feed this task
    output_as: str                      # variable name for downstream tasks to reference
    parallelizable: bool = False        # True if this task can run concurrently with siblings at same depth

class ExecutionPlan(BaseModel):
    """Ordered list of tasks ready for dispatch."""
    skills: list[TaskSpec]
    terminal_skill: str                 # the last skill (requested by user)

class SkillPipeline:
    """Resolve dependencies and build execution plans."""

    def __init__(self, registry: SkillRegistry): ...

    def build_plan(
        self,
        skill_name: str,
        params: dict[str, Any] | None = None,
        version: str | None = None,
    ) -> ExecutionPlan:
        """
        1. Resolve the requested skill
        2. Walk its dependencies recursively (DFS)
        3. Topological sort → ordered task list
        4. Wire output→input connections between tasks
        5. Return ExecutionPlan
        """
        ...

    def validate_plan(self, plan: ExecutionPlan) -> list[ValidationIssue]:
        """Check all skills in the plan are resolvable and inputs match upstream outputs."""
        ...
```

### 7.3 Example: Build Plan for analyze_pricing

```python
pipeline = SkillPipeline(registry)
plan = pipeline.build_plan(
    skill_name="analyze_pricing",
    params={
        "target_item": "iPhone 15",
        "brackets": [0, 300, 500, 1000],
    },
)
# → ExecutionPlan(
#     skills=[
#         TaskSpec(step=1, skill_name="data_provider", task_spec={
#             "platform": "ebay",       # ← inherited from session context
#             "query": "iPhone 15",
#             "max_items": 500,
#         }, input_from=None, output_as="dataset"),
#
#         TaskSpec(step=2, skill_name="analyze_pricing", task_spec={
#             "dataset_ref": "dataset", # ← wired from step 1 output
#             "target_item": "iPhone 15",
#             "brackets": [0, 300, 500, 1000],
#         }, input_from=["data_provider"], output_as="analysis_report"),
#     ],
#     terminal_skill="analyze_pricing",
# )
```

Orchestrator then simply iterates `plan.skills` in order, dispatching each as `task.dispatch`.

### 7.4 Circular Dependency Detection

Pipeline must detect circular deps (A → B → A) and fail at plan-build time, not at runtime:

```python
def build_plan(self, skill_name: str, ...) -> ExecutionPlan:
    visited = set()
    resolving = set()  # stack for cycle detection

    def walk(name):
        if name in resolving:
            raise CircularDependencyError(f"Cycle detected: ...")
        if name in visited:
            return
        resolving.add(name)
        skill = self.registry.get(name)
        for dep in skill.dependencies:
            walk(dep)
        resolving.discard(name)
        visited.add(name)

    walk(skill_name)
    # ... topological sort ...
```

---

## 8. Enforcement Levels

| Level | Meaning | Executor Behavior | On Violation |
|-------|---------|-------------------|--------------|
| `suggested` | Skill is a recommendation | Executor may choose alternative approaches | No penalty. Reviewer checks quality only. |
| `strict` | Must use declared actions | Executor's allowed_actions list is locked. Cannot call unlisted actions. | Executor error → Orchestrator retries. |
| `mandatory` | Skill MUST be used, cannot skip | Same as strict + Orchestrator verifies skill was actually executed. | If Executor bypasses skill → task fails immediately, re-spawn with stronger prompt. |

Default: `suggested`.

Setting `enforcement: mandatory` is for critical skills like `data_validation` or `compliance_check` — skills the user insists must run.

---

## 9. Integration with Message Protocol (Spec 001)

### 9.1 task.dispatch Payload (Extended)

Spec 001 defines `task.dispatch` with payload `{task_id, skill_name, task_spec, skill_context}`. Spec 002 defines what goes inside `skill_context`:

```python
{
    "msg_type": "task.dispatch",
    "sender": "orchestrator",
    "recipient": "executor",
    "payload": {
        "task_id": "550e8400-e29b-41d4-a716-446655440000",
        "skill_name": "analyze_pricing",
        "task_spec": {
            "dataset_ref": "task-1.output",
            "target_item": "iPhone 15",
            "brackets": [0, 300, 500, 1000],
        },
        "skill_context": {
            "skill_md": "---\nname: analyze_pricing\n...\n",
            "allowed_actions": [
                "export_csv",
                "calculate_market_median",
                "detect_outliers_iqr",
                "segment_by_price_range",
            ],
            "action_bindings": {
                "export_csv": {
                    "action_name": "export_csv",
                    "source": "global",
                    "tool": "pandas",
                    "function": "DataFrame.to_csv",
                    "timeout_ms": 10000,
                    "config": {"index": False, "encoding": "utf-8-sig"},
                },
                "calculate_market_median": {
                    "action_name": "calculate_market_median",
                    "source": "custom",
                    "handler_class": "AnalyzePricingSkill",
                    "function": "calculate_market_median",
                    "timeout_ms": 5000,
                },
                # ... etc for all allowed actions
            },
            "output_schema": {  # JSON Schema from AnalysisReport.model_json_schema()
                "type": "object",
                "properties": {
                    "market_summary": {...},
                    "outliers": {...},
                    "price_segments": {...},
                    "recommendation": {"type": "string"},
                },
                "required": ["market_summary", "outliers", "price_segments", "recommendation"],
            },
            "enforcement": "strict",
        },
    },
}
```

### 9.2 task.result Validation

When Executor returns `task.result`, Orchestrator:

1. Extracts `output` from payload
2. If `output_schema` is present → `SkillOutputModel.model_validate(output)`
3. On `ValidationError`:
   - Log the error
   - If `enforcement: mandatory` → retry with error feedback (max 3 retries)
   - If `enforcement: strict` → retry (max 2 retries) then send to Reviewer with issues flagged
   - If `enforcement: suggested` → send to Reviewer as-is, Reviewer will note schema violations
4. **Concerns handling:** If Executor included `concerns: ["..."]` in `task.result` (Spec 001 §2.2):
   - `concerns` are attached to the `task.review` payload as additional context
   - Reviewer is instructed to verify each concern (confirm or dismiss)
   - If output passes schema validation but has concerns → still send to Reviewer (don't auto-pass)
   - If output fails schema validation AND has concerns → concerns guide the retry prompt

### 9.3 No New msg_types

Spec 002 does not add new `msg_type` values. The existing `task.dispatch` and `task.result` messages carry skill context in their payload. New message types come in Spec 003 (Responder).

---

## 10. Walkthrough: Skill from Discovery to Completion

```
1. Runtime starts
   → SkillRegistry.discover()
   → scans skills/ recursively
   → finds: data_provider/skill.md, analyze_pricing/skill.md
   → returns [SkillMeta(data_provider), SkillMeta(analyze_pricing)]

2. ToolBridge loads mappings.toml
   → registers: fetch_web_page, parse_listing_html, export_csv, export_excel

3. SkillRegistry.validate("analyze_pricing")
   → action "export_csv" → found in global tool bridge ✓
   → action "calculate_market_median" → found in handler.py ✓
   → action "detect_outliers_iqr" → found in handler.py ✓
   → action "segment_by_price_range" → found in handler.py ✓
   → dependency "data_provider" → exists in registry ✓
   → models.py → AnalysisReport imported ✓
   → VALID

4. SkillRegistry.resolve("analyze_pricing")
   → returns Skill with all action_bindings merged + output_schema populated

5. User: "Phân tích giá iPhone 15 trên eBay"
   → Orchestrator calls SkillPipeline.build_plan("analyze_pricing", ...)
   → Pipeline resolves: data_provider → analyze_pricing
   → Returns ExecutionPlan with 2 TaskSpecs

6. Orchestrator dispatches Task 1: data_provider
   → Message(task.dispatch) with skill_context for data_provider
   → Executor #1 runs → crawl → clean → task.result(dataset)

7. Orchestrator dispatches Task 2: analyze_pricing
   → Message(task.dispatch) with skill_context + dataset_ref = task-1.output
   → Executor #2 runs:
     a. Loads dataset from task-1.output (1000 iPhone 15 listings)
     b. Extracts prices: `prices = [item["price"] for item in dataset]`
     c. Calls calculate_market_median(prices) → {median: 325, ...}
     d. Calls detect_outliers_iqr(prices) → {suspicious_cheap: [...], ...}
     e. Calls segment_by_price_range(data, [0,300,500,1000]) → {...}
     f. Calls export_csv(segments, "iphone15_segments.csv")
     g. Constructs AnalysisReport (validated against output_schema)
     h. Returns task.result(analysis_report)

8. Reviewer checks → verdict: pass

9. Session complete → Orchestrator returns final result to human
```

---

## 11. File Layout (After Spec 002)

```
llend_harness/
├── runtime/              # Spec 001
│   ├── __init__.py
│   ├── base.py           # AgentRuntime ABC
│   ├── asyncio_runtime.py
│   ├── message.py        # Message model, msg_type enum
│   ├── lifecycle.py      # Agent states
│   └── checkpoint.py     # Interrupt save/load
├── skills/               # Spec 002 — skill definitions
│   ├── data_provider/
│   │   ├── skill.md
│   │   ├── handler.py
│   │   └── models.py     # ProductListing, ScrapeConfig, ScrapeResult
│   └── analyze_pricing/
│       ├── skill.md
│       ├── handler.py
│       └── models.py     # MarketSummary, OutlierReport, PriceSegment, AnalysisReport
├── registry/             # Spec 002 — discovery & resolution
│   ├── __init__.py
│   ├── registry.py       # SkillRegistry
│   ├── models.py         # SkillMeta, Skill, ActionBinding, Issue
│   ├── pipeline.py       # SkillPipeline, TaskSpec, ExecutionPlan
│   └── validator.py      # Skill validation logic
├── tool_bridge/          # Spec 002 — global action→tool mapping
│   ├── __init__.py
│   ├── bridge.py         # ToolBridge
│   └── mappings.toml     # Global action bindings
├── bootstrap/            # Future spec
├── telemetry/            # Future spec
├── testing/              # Future spec
├── plugins/              # Future spec
└── docs/
    └── specs/
        ├── 001-message-protocol-runtime-core.md
        └── 002-skill-format-registry.md          # ← THIS SPEC
```

---

## 12. Decisions & Resolved Questions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Version resolution | Latest by default | Simple. Pin later if needed. |
| Dependency resolution | SkillPipeline (separate from Orchestrator) | Keeps Orchestrator lean. Pipeline is a dedicated component. |
| Hot-reload | Auto-detect via filesystem watcher | `watchdog` library. Registry re-scans on change. |
| Custom actions | Yes — auto-discovered from handler.py | Public methods with docstrings become actions. Merged with global tool bridge actions at resolve time. |
| Output validation | Pydantic if models.py exists; primitive otherwise | Typed skills get schema validation. Simple skills rely on Reviewer. |
| Circular dependency | Detected at plan-build time | Pipeline DFS catches cycles before any task is dispatched. |

---

## 13. Open Questions

- **Q1:** Should `SkillPipeline` also handle *parallel* tasks? If two dependencies are independent (A and B both feed C, but A and B don't depend on each other), should the plan mark them as parallelizable? → **Defer to Spec 004 (Orchestrator Logic).**
- **Q2:** How are `inputs` resolved when a dependency's output doesn't match exactly? E.g., `analyze_pricing` expects `dataset: list[ProductListing]` but `data_provider` outputs `ScrapeResult` which wraps `list[ProductListing]`. → **Auto-unwrap convention:** if output is a Pydantic model with a single list field, unwrap it. Otherwise, Orchestrator passes the whole object and Executor extracts. Formalize in Spec 004.
- **Q3:** Skill version pinning syntax? `data_provider@0.1.0` vs `data_provider@^0.1`? → **Defer.** Latest-only for v0. Add pinning when multi-version support is needed.
- **Q4:** Skill namespaces / categories — what if two skills have the same name but serve different platforms (e.g., `ebay/data_provider` vs `amazon/data_provider`)? → **Defer to Spec 004.** Current layout uses flat `skills/<name>/`; subdirectories are treated as categories for `list_skills()` grouping but do not create namespaces. Collision detection in registry raises a warning.

---

*Next spec: 003 — Responder Agent & Conversation Module*
