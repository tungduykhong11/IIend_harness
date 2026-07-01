"""Skill format & registry — discovery, validation, resolution.  Spec 002.

Exports the core types and classes that other components (Orchestrator, Executor)
need to work with skills: ``SkillRegistry`` for managing skills, ``SkillPipeline``
for building execution plans, and the Pydantic models defining the skill format.
"""

from llend.registry.models import (
    ActionBinding,
    ResolutionError,
    Skill,
    SkillMeta,
    ValidationIssue,
)
from llend.registry.parser import parse_inputs
from llend.registry.pipeline import (
    CircularDependencyError,
    ExecutionPlan,
    SkillPipeline,
    TaskSpec,
)
from llend.registry.registry import SkillRegistry
from llend.registry.validator import validate_skill

__all__ = [
    # §6 — main classes
    "SkillRegistry",
    "SkillPipeline",
    # §6.2 / §7.2 — data models
    "SkillMeta",
    "Skill",
    "ActionBinding",
    "ValidationIssue",
    "TaskSpec",
    "ExecutionPlan",
    # §2.4 — parser
    "parse_inputs",
    # §6.1 / §7.4 — exceptions
    "ResolutionError",
    "CircularDependencyError",
    # validator
    "validate_skill",
]
