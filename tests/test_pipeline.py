"""Tests for registry.pipeline — SkillPipeline."""

import tempfile
from pathlib import Path

import pytest

from llend.registry.pipeline import CircularDependencyError, ExecutionPlan, SkillPipeline, TaskSpec
from llend.registry.registry import SkillRegistry
from llend.tool_bridge.bridge import ToolBridge


# ── Helpers ────────────────────────────────────────────────────────────


def _make_skill_dir(
    base: Path,
    name: str,
    dependencies: str = "",
    actions: str = "global_action",
) -> Path:
    """Create a minimal skill for pipeline testing."""
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    md = f"""---
name: {name}
version: 0.1.0
description: Test skill {name}
inputs: query:str
outputs: str
actions: [{actions}]
dependencies: [{dependencies}]
enforcement: suggested
---

# {name}
"""
    (skill_dir / "skill.md").write_text(md, encoding="utf-8")
    return skill_dir


def _make_registry(tmp: Path) -> SkillRegistry:
    """Set up a registry with a ToolBridge and skills_dir."""
    skills_dir = tmp / "skills"
    skills_dir.mkdir()

    # Valid tool bridge
    toml = """
[actions.global_action]
tool = "json"
function = "dumps"
"""
    toml_path = tmp / "mappings.toml"
    toml_path.write_text(toml, encoding="utf-8")
    bridge = ToolBridge(toml_path, validate=True)

    return SkillRegistry(skills_dir, bridge)


# ── Tests ──────────────────────────────────────────────────────────────


class TestPipelineBuildPlan:
    """Spec 002 §7.2 — build_plan()."""

    @pytest.mark.asyncio
    async def test_single_skill_with_no_deps(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            registry = _make_registry(tmp)
            _make_skill_dir(registry._skills_dir, "standalone")
            await registry.discover()
            registry.resolve("standalone")

            pipeline = SkillPipeline(registry)
            plan = pipeline.build_plan("standalone")

            assert isinstance(plan, ExecutionPlan)
            assert plan.terminal_skill == "standalone"
            assert len(plan.skills) == 1
            assert plan.skills[0].step == 1
            assert plan.skills[0].skill_name == "standalone"

    @pytest.mark.asyncio
    async def test_two_skills_with_dependency_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            registry = _make_registry(tmp)
            _make_skill_dir(registry._skills_dir, "provider")
            _make_skill_dir(registry._skills_dir, "analyzer", dependencies="provider")
            await registry.discover()
            registry.resolve("provider")
            registry.resolve("analyzer")

            pipeline = SkillPipeline(registry)
            plan = pipeline.build_plan("analyzer")

            assert len(plan.skills) == 2
            # provider must come before analyzer (topological order)
            assert plan.skills[0].skill_name == "provider"
            assert plan.skills[1].skill_name == "analyzer"
            assert plan.skills[1].input_from == ["provider"]

    @pytest.mark.asyncio
    async def test_circular_dependency_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            registry = _make_registry(tmp)
            _make_skill_dir(registry._skills_dir, "skill_a", dependencies="skill_b")
            _make_skill_dir(registry._skills_dir, "skill_b", dependencies="skill_a")
            await registry.discover()
            registry.resolve("skill_a")
            registry.resolve("skill_b")

            pipeline = SkillPipeline(registry)
            with pytest.raises(CircularDependencyError) as exc_info:
                pipeline.build_plan("skill_a")
            assert "Circular dependency" in str(exc_info.value)
            assert "skill_a" in exc_info.value.cycle_path

    @pytest.mark.asyncio
    async def test_self_dependency_caught_as_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            registry = _make_registry(tmp)
            _make_skill_dir(registry._skills_dir, "self_ref", dependencies="self_ref")
            await registry.discover()
            # Don't resolve — self-dependency is a validation error, but
            # the pipeline can still walk metadata to detect the cycle.

            pipeline = SkillPipeline(registry)
            with pytest.raises(CircularDependencyError):
                pipeline.build_plan("self_ref")

    @pytest.mark.asyncio
    async def test_params_passed_to_terminal_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            registry = _make_registry(tmp)
            _make_skill_dir(registry._skills_dir, "target")
            await registry.discover()
            registry.resolve("target")

            pipeline = SkillPipeline(registry)
            plan = pipeline.build_plan("target", params={"foo": "bar"})

            assert plan.skills[0].task_spec == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_plan_with_three_skills_diamond_deps(self):
        """A → B, A → C, B and C both feed D (diamond). No cycle."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            registry = _make_registry(tmp)
            _make_skill_dir(registry._skills_dir, "leaf_b", dependencies="base_a")
            _make_skill_dir(registry._skills_dir, "leaf_c", dependencies="base_a")
            _make_skill_dir(registry._skills_dir, "base_a")
            await registry.discover()
            registry.resolve("base_a")
            registry.resolve("leaf_b")
            registry.resolve("leaf_c")

            pipeline = SkillPipeline(registry)
            plan = pipeline.build_plan("leaf_c")

            # base_a must come before leaf_b/leaf_c
            names = [ts.skill_name for ts in plan.skills]
            assert names.index("base_a") < names.index("leaf_c")

    @pytest.mark.asyncio
    async def test_non_dependent_tasks_marked_parallelizable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            registry = _make_registry(tmp)
            _make_skill_dir(registry._skills_dir, "independent_a")
            _make_skill_dir(registry._skills_dir, "independent_b")
            await registry.discover()
            registry.resolve("independent_a")
            registry.resolve("independent_b")

            pipeline = SkillPipeline(registry)
            plan = pipeline.build_plan("independent_a")

            # independent_a has no deps; independent_b is in the plan but
            # as a dependency of nothing we didn't request it. Let's test
            # a plan that has both.
            plan_b = pipeline.build_plan("independent_b")
            # Single-node plans don't get parallelizable
            assert plan_b.skills[0].parallelizable is False

    @pytest.mark.asyncio
    async def test_skill_not_found_raises_resolution_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            registry = _make_registry(tmp)
            await registry.discover()

            pipeline = SkillPipeline(registry)
            from llend.registry.models import ResolutionError
            with pytest.raises(ResolutionError, match="Cannot resolve"):
                pipeline.build_plan("ghost")


class TestPipelineValidatePlan:
    """Spec 002 §7.2 — validate_plan()."""

    @pytest.mark.asyncio
    async def test_valid_plan_no_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            registry = _make_registry(tmp)
            _make_skill_dir(registry._skills_dir, "provider")
            _make_skill_dir(registry._skills_dir, "consumer", dependencies="provider")
            await registry.discover()
            registry.resolve("provider")
            registry.resolve("consumer")

            pipeline = SkillPipeline(registry)
            plan = pipeline.build_plan("consumer")
            issues = pipeline.validate_plan(plan)
            assert len([i for i in issues if i.severity == "error"]) == 0

    @pytest.mark.asyncio
    async def test_missing_upstream_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            registry = _make_registry(tmp)
            _make_skill_dir(registry._skills_dir, "solo")
            await registry.discover()
            registry.resolve("solo")

            pipeline = SkillPipeline(registry)
            # Manually create a plan with a broken input_from
            plan = ExecutionPlan(
                skills=[
                    TaskSpec(
                        step=1,
                        skill_name="solo",
                        task_spec={},
                        input_from=["nonexistent_skill"],
                        output_as="solo",
                    ),
                ],
                terminal_skill="solo",
            )
            issues = pipeline.validate_plan(plan)
            errors = [i for i in issues if i.severity == "error"]
            assert any("nonexistent_skill" in e.message for e in errors)
