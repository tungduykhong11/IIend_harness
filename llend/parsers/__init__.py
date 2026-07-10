"""Parsers — HTML/data extraction utilities for the tool bridge.

These modules implement the ``parse_listing_html`` action declared in
``tool_bridge/mappings.toml`` (Spec 002 §5.1).
"""

from llend.parsers.html_parser import parse_product_listing

__all__ = ["parse_product_listing"]
