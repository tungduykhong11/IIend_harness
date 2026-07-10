"""Skill Pipeline — dependency resolution & execution plan generation.

Resolves the dependency graph declared by skills (via ``dependencies``)
and produces an ordered ``ExecutionPlan`` that the Orchestrator can follow
step-by-step.

Spec references
===============
- **§7** → Skill Pipeline concept
- **§7.2** → ``SkillPipeline`` class, ``TaskSpec``, ``ExecutionPlan``
- **§7.3** → example plan for analyze_pricing
- **§7.4** → circular dependency detection
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from llend.registry.models import ResolutionError, ValidationIssue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TaskSpec  —  Spec 002 §7.2
# ---------------------------------------------------------------------------


class TaskSpec(BaseModel):
    """One task in the execution plan.  Spec 002 §7.2."""

    step: int
    skill_name: str
    task_spec: dict[str, Any]  # task-specific params (merged from upstream outputs)
    input_from: list[str] | None = None  # skill names whose outputs feed this task
    output_as: str  # variable name for downstream tasks to reference
    parallelizable: bool = False  # True if can run concurrently with siblings at same depth


# ---------------------------------------------------------------------------
# ExecutionPlan  —  Spec 002 §7.2
# ---------------------------------------------------------------------------


class ExecutionPlan(BaseModel):
    """Ordered list of tasks ready for dispatch by Orchestrator.  Spec 002 §7.2."""

    skills: list[TaskSpec]
    terminal_skill: str  # the last skill (the one requested by user)


# ---------------------------------------------------------------------------
# CircularDependencyError  —  Spec 002 §7.4
# ---------------------------------------------------------------------------


class CircularDependencyError(Exception):
    """Raised by ``SkillPipeline.build_plan()`` when a circular dependency is detected.

    Spec 002 §7.4 — cycles are caught at plan-build time, never at runtime.
    """

    def __init__(self, cycle_path: list[str]) -> None:
        self.cycle_path = cycle_path
        cycle_str = " → ".join(cycle_path)
        super().__init__(f"Circular dependency detected: {cycle_str}")


# ---------------------------------------------------------------------------
# SkillPipeline  —  Spec 002 §7.2
# ---------------------------------------------------------------------------


class SkillPipeline:
    """Resolve dependencies and build execution plans.  Spec 002 §7.2.

    Takes a ``SkillRegistry`` for skill lookup.  The ``build_plan()`` method
    walks dependencies via DFS, detects cycles, topologically sorts, and wires
    output→input connections between tasks.
    """

    def __init__(self, registry: object) -> None:
        """*registry* must have a ``.get(name: str)`` method returning a skill-like
        object with ``dependencies``, ``name``, ``inputs``, ``outputs`` attributes.

        We take ``object`` to avoid a circular import with ``SkillRegistry``.
        """
        self._registry = registry

    # ------------------------------------------------------------------
    # §7.2 — build_plan
    # ------------------------------------------------------------------

    def build_plan(
        self,
        skill_name: str,
        params: dict[str, Any] | None = None,
        version: str | None = None,
    ) -> ExecutionPlan:
        """Build an ordered execution plan for *skill_name*.  §7.2.

        1. Resolve the requested skill
        2. Walk dependencies recursively (DFS with cycle detection)
        3. Topological sort → ordered task list
        4. Wire output→input connections between tasks
        5. Return ``ExecutionPlan``

        Raises ``CircularDependencyError`` if a cycle is found.
        Raises ``ResolutionError`` if any skill in the chain cannot be resolved.
        """
        params = params or {}

        # Step 1-2: DFS walk with cycle detection — §7.4
        visited: set[str] = set()
        resolving: set[str] = set()  # stack for cycle detection
        order: list[str] = []

        def walk(name: str) -> None:
            if name in resolving:
                # Find the cycle path for the error message
                cycle_start = name
                cycle = list(resolving)
                # Append the closing link
                cycle_idx = cycle.index(cycle_start) if cycle_start in cycle else 0
                cycle_path = cycle[cycle_idx:] + [cycle_start]
                raise CircularDependencyError(cycle_path)

            if name in visited:
                return

            resolving.add(name)

            # Try resolved skill first, fall back to metadata (for dep walking)
            skill = self._registry.get(name)  # type: ignore[union-attr]
            deps: list[str] = []
            if skill is not None:
                deps = getattr(skill, "dependencies", [])
            elif hasattr(self._registry, "_metas"):
                meta = self._registry._metas.get(name)  # type: ignore[union-attr]
                if meta is not None:
                    deps = getattr(meta, "dependencies", [])
                else:
                    raise ResolutionError(name, [
                        ValidationIssue(
                            severity="error",
                            field="name",
                            message=f"Skill {name!r} not found in registry",
                        )
                    ])
            else:
                raise ResolutionError(name, [
                    ValidationIssue(
                        severity="error",
                        field="name",
                        message=f"Skill {name!r} not found in registry",
                    )
                ])

            for dep in deps:
                walk(dep)

            resolving.discard(name)
            visited.add(name)
            order.append(name)

        walk(skill_name)

        # Step 3: topological sort — order is already DFS post-order
        # Build TaskSpecs
        task_specs: list[TaskSpec] = []
        skill_outputs: dict[str, str] = {}  # skill_name → output_as variable name

        for idx, name in enumerate(order):
            skill = self._registry.get(name)  # type: ignore[union-attr]
            deps = getattr(skill, "dependencies", [])
            output_as = name

            # Merge params: forward all user params to every skill in the chain.
            # Each Executor extracts what it needs from task_spec.  Upstream
            # skills often need params too (e.g. data_provider needs "query"
            # from the user request, even though analyze_pricing is the terminal).
            merged_params: dict[str, Any] = dict(params)

            if deps:
                for dep in deps:
                    if dep in skill_outputs:
                        merged_params[f"{dep}_ref"] = skill_outputs[dep]

            task_specs.append(
                TaskSpec(
                    step=idx + 1,
                    skill_name=name,
                    task_spec=merged_params,
                    input_from=list(deps) if deps else None,
                    output_as=output_as,
                    parallelizable=False,  # Defer to Spec 004
                )
            )

            skill_outputs[name] = output_as

        # Mark parallelizable tasks: tasks at the same "depth" with no
        # interdependency can run concurrently.  This is a simple heuristic;
        # Spec 004 will refine this.
        for ts in task_specs:
            if not ts.input_from:
                # No deps — check if any sibling at same "level" has the same
                siblings = [
                    s for s in task_specs
                    if s.step != ts.step and not s.input_from
                ]
                if siblings:
                    ts.parallelizable = True

        plan = ExecutionPlan(skills=task_specs, terminal_skill=skill_name)

        logger.info(
            "Built plan for %r: %d task(s), terminal=%r",
            skill_name,
            len(task_specs),
            skill_name,
        )

        return plan

    # ------------------------------------------------------------------
    # §7.2 — validate_plan
    # ------------------------------------------------------------------

    def validate_plan(self, plan: ExecutionPlan) -> list[ValidationIssue]:
        """Check all skills in the plan are resolvable and inputs match upstream outputs.

        §7.2 — returns issues; empty list = valid.
        """
        issues: list[ValidationIssue] = []

        skill_names = {ts.skill_name for ts in plan.skills}

        for ts in plan.skills:
            skill = self._registry.get(ts.skill_name)  # type: ignore[union-attr]
            if skill is None:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        field="skill_name",
                        message=f"Skill {ts.skill_name!r} not found in registry",
                    )
                )
                continue

            # Check that input_from references exist
            if ts.input_from:
                for upstream in ts.input_from:
                    if upstream not in skill_names:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                field="input_from",
                                message=(
                                    f"Task {ts.step!r} ({ts.skill_name!r}) depends on "
                                    f"{upstream!r} which is not in the plan"
                                ),
                            )
                        )

        return issues
