"""Registry data models — canonical types for skill discovery, validation, and resolution.

Every model here is a Pydantic ``BaseModel``.  They are consumed by ``SkillRegistry``,
``ToolBridge``, ``SkillPipeline``, and eventually the ``ActionDispatcher`` inside Executor.

Spec references
===============
- **§6.2** → ``SkillMeta``, ``Skill``, ``ActionBinding``, ``ValidationIssue``
- **§2.3** → ``SkillMeta.inputs`` format (parsed by ``registry.parser``)
- **§3.2** → ``Skill.output_schema`` (resolved from ``models.py`` by name)
- **§4.3** → ``ActionBinding.source`` / ``ActionBinding.handler_class``
- **§8** → ``SkillMeta.enforcement`` — ``suggested`` (default), ``strict``, ``mandatory``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# SkillMeta  —  Spec 002 §6.2, §2.3
# ---------------------------------------------------------------------------


class SkillMeta(BaseModel):
    """Parsed from a SKILL.md YAML frontmatter.  Spec 002 §6.2.

    This is the raw metadata **before** resolution — no models loaded,
    no action bindings merged, no handler instantiated.
    """

    name: str
    version: str
    description: str

    inputs: dict[str, str]  # {param_name: "type_spec"} parsed by parser.parse_inputs()
    outputs: str  # "AnalysisReport" | "list[dict]" | "None"
    actions: list[str]  # ["export_csv", "calculate_market_median", ...]

    dependencies: list[str] = Field(default_factory=list)
    enforcement: Literal["suggested", "strict", "mandatory"] = "suggested"


# ---------------------------------------------------------------------------
# ValidationIssue  —  Spec 002 §6.2
# ---------------------------------------------------------------------------


class ValidationIssue(BaseModel):
    """An issue found during skill validation.  Spec 002 §6.2.

    **Distinct** from Spec 001 ``ReviewIssue`` (which is a Reviewer's verdict
    on Executor output quality).  This is a *compile-time* check on the skill
    definition itself.

    ``severity == "error"`` blocks ``SkillRegistry.resolve()``.
    ``severity == "warning"`` allows resolution but indicates risk.
    """

    severity: Literal["error", "warning"]
    field: str  # which part of the skill has the issue
    message: str  # human-readable description


# ---------------------------------------------------------------------------
# ActionBinding  —  Spec 002 §6.2, §4.3–§4.4
# ---------------------------------------------------------------------------


class ActionBinding(BaseModel):
    """Resolved action → implementation mapping.  Spec 002 §6.2.

    Used by the Executor's ``ActionDispatcher`` to invoke an action at runtime
    (§4.4): global bindings are dispatched via ``importlib.import_module``;
    custom bindings are dispatched via handler method calls.
    """

    action_name: str
    source: Literal["global", "custom"]
    tool: str | None = None  # global: "crawl4ai" (None for custom)
    function: str  # "AsyncWebCrawler.arun" or "calculate_market_median"
    handler_class: str | None = None  # custom: "AnalyzePricingSkill" (None for global)
    timeout_ms: int = 30000
    retry: int = 0
    config: dict[str, Any] = Field(default_factory=dict)
    input_schema: dict[str, Any] | None = None  # JSON Schema for LLM function-calling (Spec 005 §2.5)


# ---------------------------------------------------------------------------
# Skill (resolved)  —  Spec 002 §6.2
# ---------------------------------------------------------------------------


class Skill(SkillMeta):
    """Fully resolved skill, ready for dispatch.  Spec 002 §6.2.

    Produced by ``SkillRegistry.resolve()``.  Includes loaded Pydantic models,
    merged action bindings (global + custom), and the handler instance (if any).
    """

    path: Path  # absolute skill directory path
    skill_md: str  # raw SKILL.md content
    output_schema: dict | None = None  # JSON Schema from Pydantic model (None if no models.py)
    input_schemas: dict[str, dict] = Field(default_factory=dict)  # {param_name: json_schema}
    action_bindings: dict[str, ActionBinding] = Field(default_factory=dict)
    handler: object | None = None  # handler instance (None if no handler.py)


# ---------------------------------------------------------------------------
# ResolutionError  —  Spec 002 §6.1
# ---------------------------------------------------------------------------


class ResolutionError(Exception):
    """Raised by ``SkillRegistry.resolve()`` when a skill has blocking errors.

    Spec 002 §6.1 — errors (as opposed to warnings) prevent resolution.  The
    caller should call ``validate()`` first to inspect the issues and decide
    whether to proceed.
    """

    def __init__(self, skill_name: str, issues: list[ValidationIssue]) -> None:
        self.skill_name = skill_name
        self.issues = issues
        errors = [i for i in issues if i.severity == "error"]
        super().__init__(
            f"Cannot resolve {skill_name!r}: {len(errors)} error(s), "
            f"{len(issues) - len(errors)} warning(s)"
        )
