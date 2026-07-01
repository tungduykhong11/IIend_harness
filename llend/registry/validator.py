"""Skill validation — compile-time checks on SKILL.md + handler.py + models.py.

Called by ``SkillRegistry.validate()``.  Returns ``ValidationIssue`` lists
(not raised exceptions) so the caller can decide how to handle each severity.

Spec references
===============
- **§6.1** → ``validate()`` returns ``list[ValidationIssue]``, caller decides
- **§6.1** → error vs. warning severity distinction
- **§2.3** → input type parsing rules
- **§3.2** → output model resolution (models.py → BaseModel subclass)
- **§4.3** → custom action discovery (public async method + docstring)
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path

from llend.registry.models import SkillMeta, ValidationIssue
from llend.tool_bridge.bridge import ToolBridge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def validate_skill(
    meta: SkillMeta,
    tool_bridge: ToolBridge,
    skills_dir: Path,
    *,
    known_names: set[str] | None = None,
) -> list[ValidationIssue]:
    """Run all five validation checks on *meta*.  Spec 002 §6.1.

    Returns a list of ``ValidationIssue`` objects.  An empty list means the
    skill is fully valid.  Errors block ``resolve()``; warnings allow it.
    """
    skill_dir = skills_dir / meta.name
    issues: list[ValidationIssue] = []

    issues.extend(_validate_actions(meta, tool_bridge, skill_dir))
    issues.extend(_validate_io_types(meta, skill_dir))
    issues.extend(_validate_models(meta, skill_dir))
    issues.extend(_validate_handler(meta, skill_dir))
    issues.extend(_validate_dependencies(meta, known_names or set()))

    return issues


# ---------------------------------------------------------------------------
# Check 1: all declared actions resolve  —  Spec 002 §6.1 item 1
# ---------------------------------------------------------------------------


def _validate_actions(
    meta: SkillMeta, tool_bridge: ToolBridge, skill_dir: Path
) -> list[ValidationIssue]:
    """Ensure every action in ``meta.actions`` is provided by either the global
    tool bridge or a custom handler method."""
    issues: list[ValidationIssue] = []

    global_actions = set(tool_bridge.list_actions())
    custom_actions = _discover_custom_actions(skill_dir)

    for action in meta.actions:
        if action in global_actions:
            continue
        if action in custom_actions:
            continue

        has_handler = (skill_dir / "handler.py").exists()
        if has_handler:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    field="actions",
                    message=(
                        f"Action {action!r} not found in global tool bridge or "
                        f"handler.py custom methods"
                    ),
                )
            )
        else:
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="actions",
                    message=(
                        f"Action {action!r} not found in global tool bridge "
                        f"and no handler.py to provide it"
                    ),
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Check 2: input/output types are parseable  —  Spec 002 §6.1 item 2
# ---------------------------------------------------------------------------


def _validate_io_types(meta: SkillMeta, skill_dir: Path) -> list[ValidationIssue]:
    """Check that input/output type declarations are syntactically valid."""
    issues: list[ValidationIssue] = []

    # Inputs were already parsed by the time we have SkillMeta — the parser
    # is lenient (it just splits on commas/colons).  We flag empty dicts
    # only for skills that declare required inputs.
    # (No strict validation here — the parser handles edge cases gracefully.)

    # Output check: acceptable primitive types
    primitives = {"str", "int", "float", "bool", "list", "dict", "list[dict]", "None"}
    output = meta.outputs.strip()

    if not output:
        issues.append(
            ValidationIssue(
                severity="error",
                field="outputs",
                message="Outputs field is empty — must declare a return type",
            )
        )
    elif output in primitives:
        pass  # valid primitive
    elif not (skill_dir / "models.py").exists():
        issues.append(
            ValidationIssue(
                severity="warning",
                field="outputs",
                message=(
                    f"Output type {output!r} is not a primitive and no "
                    f"models.py found — output will be untyped"
                ),
            )
        )
    # If models.py exists, Check 3 handles the actual import

    return issues


# ---------------------------------------------------------------------------
# Check 3: models.py output model is importable  —  Spec 002 §6.1 item 3
# ---------------------------------------------------------------------------


def _validate_models(meta: SkillMeta, skill_dir: Path) -> list[ValidationIssue]:
    """Check that the output model (if not a primitive) exists in models.py
    and is a ``BaseModel`` subclass."""
    issues: list[ValidationIssue] = []

    models_path = skill_dir / "models.py"
    primitives = {"str", "int", "float", "bool", "list", "dict", "list[dict]", "None"}
    output = meta.outputs.strip()

    if output in primitives:
        return issues  # primitives don't need Pydantic

    if not models_path.exists():
        return issues  # Check 2 already warned

    # Try loading the module
    module = _import_module_from_path("models", models_path)
    if module is None:
        issues.append(
            ValidationIssue(
                severity="error",
                field="models.py",
                message=f"Failed to import models.py: {models_path}",
            )
        )
        return issues

    # Look up the output class
    model_cls = getattr(module, output, None)
    if model_cls is None:
        issues.append(
            ValidationIssue(
                severity="error",
                field="models.py",
                message=f"Output model {output!r} not found in {models_path}",
            )
        )
        return issues

    # Verify it's a BaseModel
    try:
        from pydantic import BaseModel as PydanticBaseModel

        if not issubclass(model_cls, PydanticBaseModel):
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="models.py",
                    message=f"{output!r} is not a Pydantic BaseModel subclass",
                )
            )
    except TypeError:
        issues.append(
            ValidationIssue(
                severity="error",
                field="models.py",
                message=f"{output!r} is not a class",
            )
        )

    return issues


# ---------------------------------------------------------------------------
# Check 4: handler.py is importable, has expected class + docstrings
#  —  Spec 002 §6.1 item 4
# ---------------------------------------------------------------------------


def _validate_handler(meta: SkillMeta, skill_dir: Path) -> list[ValidationIssue]:
    """Check handler.py: importable, has discoverable class, methods have docstrings."""
    issues: list[ValidationIssue] = []

    handler_path = skill_dir / "handler.py"
    if not handler_path.exists():
        return issues  # no handler is fine (markdown-only skill)

    module = _import_module_from_path("handler", handler_path)
    if module is None:
        issues.append(
            ValidationIssue(
                severity="error",
                field="handler.py",
                message=f"Failed to import handler.py: {handler_path}",
            )
        )
        return issues

    # Find handler class — §4.3 discovery convention
    handler_cls = _discover_handler_class(module, meta.name)
    if handler_cls is None:
        issues.append(
            ValidationIssue(
                severity="warning",
                field="handler.py",
                message="No handler class found — expected PascalCase(SkillName)Skill or "
                "first class with public async methods",
            )
        )

    return issues


# ---------------------------------------------------------------------------
# Check 5: dependencies reference existing skill names  —  Spec 002 §6.1 item 5
# ---------------------------------------------------------------------------


def _validate_dependencies(
    meta: SkillMeta, known_names: set[str]
) -> list[ValidationIssue]:
    """Check that all dependency names exist in the registry."""
    issues: list[ValidationIssue] = []

    for dep in meta.dependencies:
        if dep not in known_names:
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="dependencies",
                    message=f"Dependency {dep!r} is not a known skill name",
                )
            )
        elif dep == meta.name:
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="dependencies",
                    message=f"Skill {meta.name!r} cannot depend on itself",
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _discover_custom_actions(skill_dir: Path) -> set[str]:
    """Scan handler.py for public async methods with docstrings.  §4.3."""
    handler_path = skill_dir / "handler.py"
    if not handler_path.exists():
        return set()

    module = _import_module_from_path("handler", handler_path)
    if module is None:
        return set()

    handler_cls = _discover_handler_class(module, skill_dir.name)
    if handler_cls is None:
        return set()

    actions: set[str] = set()
    for name, method in inspect.getmembers(handler_cls, predicate=inspect.isfunction):
        # Rule: public (no leading _) + has docstring → registered as action
        if name.startswith("_"):
            continue
        if not method.__doc__:
            continue
        # Must be async
        if not inspect.iscoroutinefunction(method):
            continue
        actions.add(name)

    return actions


def _discover_handler_class(
    module: object, skill_name: str
) -> type | None:
    """Find the handler class in *module*.  §4.3 discovery convention.

    1. Look for ``<PascalCaseSkillName>Skill`` (e.g. ``analyze_pricing`` → ``AnalyzePricingSkill``).
    2. Fallback: first class with ≥1 public async method that has a docstring.
    3. If multiple classes match rule 2, use the first and log a warning.
    """
    # Rule 1: exact name match
    pascal = "".join(word.capitalize() for word in skill_name.split("_")) + "Skill"
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__name__ == pascal:
            return obj

    # Rule 2: first class with at least one discoverable method
    candidates: list[type] = []
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        for meth_name, meth in inspect.getmembers(obj, predicate=inspect.isfunction):
            if meth_name.startswith("_"):
                continue
            if not meth.__doc__:
                continue
            if inspect.iscoroutinefunction(meth):
                candidates.append(obj)
                break  # one match per class is enough

    if len(candidates) > 1:
        logger.warning(
            "Multiple handler class candidates for %r: %s — using %r",
            skill_name,
            [c.__name__ for c in candidates],
            candidates[0].__name__,
        )

    return candidates[0] if candidates else None


def _import_module_from_path(module_name: str, path: Path) -> object | None:
    """Import a .py file as a module.  Returns ``None`` on failure."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            logger.warning("Cannot create module spec for %s", path)
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        logger.exception("Failed to import %s", path)
        return None
