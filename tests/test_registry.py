"""Tests for registry.registry — SkillRegistry class."""

import tempfile
from pathlib import Path

import pytest

from llend.registry.models import ResolutionError, Skill, SkillMeta, ValidationIssue
from llend.registry.registry import SkillRegistry
from llend.tool_bridge.bridge import ToolBridge


# ── Helpers ────────────────────────────────────────────────────────────


def _make_skill_dir(
    base: Path,
    name: str,
    version: str = "0.1.0",
    description: str = "A test skill",
    inputs: str = "query:str",
    outputs: str = "str",
    actions: str = "",
    dependencies: str = "",
    enforcement: str = "suggested",
    *,
    with_handler: bool = False,
    with_models: bool = False,
) -> Path:
    """Create a minimal skill directory for testing."""
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    # skill.md
    md = f"""---
name: {name}
version: {version}
description: {description}
inputs: {inputs}
outputs: {outputs}
actions: [{actions}]
dependencies: [{dependencies}]
enforcement: {enforcement}
---

# {name}

Test skill.
"""
    (skill_dir / "skill.md").write_text(md, encoding="utf-8")

    # handler.py
    if with_handler:
        handler_code = f'''"""Handler for {name}."""
class {_pascal(name)}Skill:
    async def do_custom_action(self, data: list) -> dict:
        """
        A custom action.
        INPUT: data: list
        OUTPUT: dict
        """
        return {{"result": "ok"}}
'''
        (skill_dir / "handler.py").write_text(handler_code, encoding="utf-8")

    # models.py
    if with_models:
        models_code = f'''"""Models for {name}."""
from pydantic import BaseModel

class MyOutput(BaseModel):
    value: str = "hello"
'''
        (skill_dir / "models.py").write_text(models_code, encoding="utf-8")

    return skill_dir


def _pascal(snake: str) -> str:
    return "".join(word.capitalize() for word in snake.split("_"))


def _make_stub_tool_bridge(tmp: Path) -> ToolBridge:
    """Create a ToolBridge with a minimal valid TOML referencing built-in modules."""
    toml = """
[actions.global_action]
tool = "json"
function = "dumps"
"""
    toml_path = tmp / "mappings.toml"
    toml_path.write_text(toml, encoding="utf-8")
    return ToolBridge(toml_path, validate=True)


# ── Tests ──────────────────────────────────────────────────────────────


class TestSkillRegistryDiscover:
    """Spec 002 §6.1 — discover()."""

    @pytest.mark.asyncio
    async def test_discovers_single_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(skills_dir, "my_skill", actions="global_action")

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            discovered = await registry.discover()

            assert len(discovered) == 1
            assert discovered[0].name == "my_skill"
            assert discovered[0].version == "0.1.0"

    @pytest.mark.asyncio
    async def test_discovers_multiple_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(skills_dir, "skill_a", actions="global_action")
            _make_skill_dir(skills_dir, "skill_b", actions="global_action")

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            discovered = await registry.discover()

            assert len(discovered) == 2
            names = {m.name for m in discovered}
            assert names == {"skill_a", "skill_b"}

    @pytest.mark.asyncio
    async def test_discover_handles_parse_errors_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            skill_dir = skills_dir / "bad_skill"
            skill_dir.mkdir()
            (skill_dir / "skill.md").write_text("no frontmatter here", encoding="utf-8")

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            discovered = await registry.discover()

            # Bad skill should be skipped, no crash
            assert len(discovered) == 0


class TestSkillRegistryValidate:
    """Spec 002 §6.1 — validate()."""

    @pytest.mark.asyncio
    async def test_valid_skill_no_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(skills_dir, "ok_skill", actions="global_action")

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            meta = registry._metas["ok_skill"]
            issues = registry.validate(meta)
            assert len(issues) == 0

    @pytest.mark.asyncio
    async def test_error_when_action_not_resolvable_and_no_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(skills_dir, "bad_skill", actions="unknown_action")

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            meta = registry._metas["bad_skill"]
            issues = registry.validate(meta)

            errors = [i for i in issues if i.severity == "error"]
            assert len(errors) >= 1
            assert any("unknown_action" in e.message for e in errors)

    @pytest.mark.asyncio
    async def test_warning_when_action_in_handler_but_not_global(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(
                skills_dir, "custom_skill",
                actions="do_custom_action", with_handler=True,
            )

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            meta = registry._metas["custom_skill"]
            issues = registry.validate(meta)
            errors = [i for i in issues if i.severity == "error"]
            assert len(errors) == 0  # custom handler satisfies the action

    @pytest.mark.asyncio
    async def test_error_when_dependency_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(skills_dir, "skill_a", actions="global_action", dependencies="nonexistent")

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            meta = registry._metas["skill_a"]
            issues = registry.validate(meta)

            errors = [i for i in issues if i.severity == "error"]
            assert any("nonexistent" in e.message for e in errors)

    @pytest.mark.asyncio
    async def test_error_when_self_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(skills_dir, "recursive", actions="global_action", dependencies="recursive")

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            meta = registry._metas["recursive"]
            issues = registry.validate(meta)
            errors = [i for i in issues if i.severity == "error"]
            assert any("depend on itself" in e.message for e in errors)

    @pytest.mark.asyncio
    async def test_warning_when_no_models_py_for_non_primitive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(skills_dir, "untyped", outputs="CustomModel", actions="global_action")

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            meta = registry._metas["untyped"]
            issues = registry.validate(meta)

            warnings = [i for i in issues if i.severity == "warning"]
            assert any("untyped" in w.message for w in warnings)

    @pytest.mark.asyncio
    async def test_models_py_with_valid_output_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(
                skills_dir, "typed_skill",
                outputs="MyOutput", actions="global_action",
                with_models=True,
            )

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            meta = registry._metas["typed_skill"]
            issues = registry.validate(meta)
            errors = [i for i in issues if i.severity == "error"]
            assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_handler_validation_warning_when_no_class_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            skill_dir = _make_skill_dir(
                skills_dir, "empty_handler",
                actions="global_action",
            )
            # Write a handler.py with no discoverable class
            (skill_dir / "handler.py").write_text("# empty handler", encoding="utf-8")

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            meta = registry._metas["empty_handler"]
            issues = registry.validate(meta)
            warnings = [i for i in issues if i.severity == "warning"]
            assert any("No handler class" in w.message for w in warnings)


class TestSkillRegistryResolve:
    """Spec 002 §6.1 — resolve()."""

    @pytest.mark.asyncio
    async def test_resolve_valid_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(skills_dir, "good", actions="global_action")

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            skill = registry.resolve("good")
            assert isinstance(skill, Skill)
            assert skill.name == "good"
            assert skill.skill_md  # raw markdown
            assert "global_action" in skill.action_bindings

    @pytest.mark.asyncio
    async def test_resolve_with_models_gets_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(
                skills_dir, "typed",
                outputs="MyOutput", actions="global_action",
                with_models=True,
            )

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            skill = registry.resolve("typed")
            assert skill.output_schema is not None
            assert skill.output_schema["type"] == "object"
            assert "value" in skill.output_schema["properties"]

    @pytest.mark.asyncio
    async def test_resolve_with_handler_instantiates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(
                skills_dir, "with_handler",
                actions="do_custom_action", with_handler=True,
            )

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            skill = registry.resolve("with_handler")
            assert skill.handler is not None
            assert "do_custom_action" in skill.action_bindings
            assert skill.action_bindings["do_custom_action"].source == "custom"

    @pytest.mark.asyncio
    async def test_resolve_missing_skill_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            with pytest.raises(ResolutionError, match="Cannot resolve"):
                registry.resolve("nonexistent")

    @pytest.mark.asyncio
    async def test_resolve_with_errors_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(skills_dir, "bad", actions="unknown_action")

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            with pytest.raises(ResolutionError, match="Cannot resolve"):
                registry.resolve("bad")

    @pytest.mark.asyncio
    async def test_resolve_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(skills_dir, "skill_a", actions="global_action")
            _make_skill_dir(skills_dir, "skill_b", actions="global_action")

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            resolved = registry.resolve_all()
            assert len(resolved) == 2

    @pytest.mark.asyncio
    async def test_get_returns_resolved_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(skills_dir, "my_skill", actions="global_action")

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            assert registry.get("my_skill") is None  # not resolved yet
            registry.resolve("my_skill")
            assert registry.get("my_skill") is not None

    @pytest.mark.asyncio
    async def test_list_skills_groups_by_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            skills_dir = tmp / "skills"
            skills_dir.mkdir()
            _make_skill_dir(skills_dir, "skill_a", actions="global_action")
            _make_skill_dir(skills_dir, "skill_b", actions="global_action")

            bridge = _make_stub_tool_bridge(tmp)
            registry = SkillRegistry(skills_dir, bridge)
            await registry.discover()

            grouped = registry.list_skills()
            assert "__root__" in grouped
            assert len(grouped["__root__"]) == 2
