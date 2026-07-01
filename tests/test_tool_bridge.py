"""Tests for tool_bridge.bridge — ToolBridge class."""

import tempfile
from pathlib import Path

import pytest

from llend.tool_bridge.bridge import ToolBridge


def _write_toml(content: str, dir_path: Path) -> Path:
    """Write a temp mappings.toml and return its path."""
    p = dir_path / "mappings.toml"
    p.write_text(content, encoding="utf-8")
    return p


class TestToolBridge:
    """Spec 002 §5.2 — ToolBridge."""

    def test_loads_valid_toml_without_validation(self):
        toml = """
[actions.export_csv]
tool = "csv"
function = "writer"
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_toml(toml, Path(tmp))
            bridge = ToolBridge(path, validate=False)
            assert "export_csv" in bridge.list_actions()

    def test_resolve_returns_binding(self):
        toml = """
[actions.do_thing]
tool = "json"
function = "dumps"
timeout_ms = 5000
retry = 2

[actions.do_thing.config]
indent = 2
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_toml(toml, Path(tmp))
            bridge = ToolBridge(path, validate=False)

            binding = bridge.resolve("do_thing")
            assert binding is not None
            assert binding.action_name == "do_thing"
            assert binding.source == "global"
            assert binding.tool == "json"
            assert binding.function == "dumps"
            assert binding.timeout_ms == 5000
            assert binding.retry == 2
            assert binding.config == {"indent": 2}

    def test_resolve_missing_returns_none(self):
        toml = """
[actions.only_one]
tool = "sys"
function = "exit"
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_toml(toml, Path(tmp))
            bridge = ToolBridge(path, validate=False)
            assert bridge.resolve("does_not_exist") is None

    def test_resolve_all_returns_all(self):
        toml = """
[actions.a]
tool = "os"
function = "getcwd"

[actions.b]
tool = "os"
function = "listdir"
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_toml(toml, Path(tmp))
            bridge = ToolBridge(path, validate=False)
            all_bindings = bridge.resolve_all()
            assert set(all_bindings.keys()) == {"a", "b"}

    def test_list_actions(self):
        toml = """
[actions.first]
tool = "sys"
function = "version"

[actions.second]
tool = "sys"
function = "platform"
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_toml(toml, Path(tmp))
            bridge = ToolBridge(path, validate=False)
            actions = bridge.list_actions()
            assert sorted(actions) == ["first", "second"]

    def test_validate_mapping_importable(self):
        """Built-in modules should validate successfully."""
        toml = """
[actions.my_action]
tool = "json"
function = "dumps"
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_toml(toml, Path(tmp))
            bridge = ToolBridge(path, validate=False)
            assert bridge.validate_mapping("my_action") is True

    def test_validate_mapping_not_importable(self):
        toml = """
[actions.bad_action]
tool = "nonexistent_module_xyz"
function = "foo"
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_toml(toml, Path(tmp))
            bridge = ToolBridge(path, validate=False)
            assert bridge.validate_mapping("bad_action") is False

    def test_validate_mapping_function_not_found(self):
        toml = """
[actions.wrong_func]
tool = "json"
function = "nonexistent_function"
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_toml(toml, Path(tmp))
            bridge = ToolBridge(path, validate=False)
            assert bridge.validate_mapping("wrong_func") is False

    def test_validation_raises_on_unimportable_tool(self):
        toml = """
[actions.bad]
tool = "nonexistent_xyz"
function = "foo"
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_toml(toml, Path(tmp))
            with pytest.raises(ValueError, match="unimportable tools"):
                ToolBridge(path, validate=True)

    def test_empty_actions(self):
        toml = ""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_toml(toml, Path(tmp))
            bridge = ToolBridge(path, validate=False)
            assert bridge.list_actions() == []

    def test_skips_missing_tool_field(self):
        toml = """
[actions.incomplete]
function = "foo"
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_toml(toml, Path(tmp))
            bridge = ToolBridge(path, validate=False)
            assert bridge.resolve("incomplete") is None
