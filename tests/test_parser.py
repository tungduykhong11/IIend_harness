"""Tests for registry.parser — SKILL.md inputs: parsing."""

import pytest

from llend.registry.parser import parse_inputs


class TestParseInputs:
    """§2.4 — input type parsing rules."""

    def test_empty_string_returns_empty_dict(self):
        assert parse_inputs("") == {}
        assert parse_inputs("   ") == {}

    def test_simple_key_type(self):
        result = parse_inputs("query:str")
        assert result == {"query": "str"}

    def test_key_type_with_default(self):
        result = parse_inputs("max_items:int=100")
        assert result == {"max_items": "int=100"}

    def test_multiple_params(self):
        result = parse_inputs("a:str, b:int=5")
        assert result == {"a": "str", "b": "int=5"}

    def test_comma_inside_quotes_preserved(self):
        result = parse_inputs('msg:str="hello, world"')
        assert result == {"msg": 'str="hello, world"'}

    def test_trailing_whitespace_trimmed(self):
        result = parse_inputs("a:str , b:int")
        assert result == {"a": "str", "b": "int"}

    def test_pydantic_model_reference(self):
        result = parse_inputs("dataset:list[ProductListing], config:AnalysisConfig")
        assert result == {
            "dataset": "list[ProductListing]",
            "config": "AnalysisConfig",
        }

    def test_mixed_primitives_and_models(self):
        result = parse_inputs("platform:str, raw_data:list[dict], report_config:ReportConfig=ReportConfig()")
        assert result == {
            "platform": "str",
            "raw_data": "list[dict]",
            "report_config": "ReportConfig=ReportConfig()",
        }

    def test_single_param_no_colon_skipped(self):
        result = parse_inputs("foobar")
        assert result == {}

    def test_empty_parts_skipped(self):
        result = parse_inputs("a:str, , b:int")
        assert result == {"a": "str", "b": "int"}

    def test_spec_example_brackets_default(self):
        result = parse_inputs("dataset:list[dict], target_item:str, brackets:list[int]=[0,300,500,1000]")
        assert result == {
            "dataset": "list[dict]",
            "target_item": "str",
            "brackets": "list[int]=[0,300,500,1000]",
        }

    def test_quoted_default_with_comma(self):
        """Commas inside quoted defaults are preserved as part of the type spec."""
        result = parse_inputs('name:str="Doe, John", age:int=30')
        assert result == {"name": 'str="Doe, John"', "age": "int=30"}
