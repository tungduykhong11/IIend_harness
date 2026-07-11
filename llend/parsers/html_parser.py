"""HTML listing parser — extracts product fields from e-commerce pages.

Used by the ``parse_listing_html`` action in ``tool_bridge/mappings.toml``
(Spec 002 §5.1).  Called by ``ActionDispatcher`` inside the Executor during
``data_provider`` skill execution.
"""

from __future__ import annotations

import json as _json
import logging
import warnings
from typing import Any

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

# Suppress BeautifulSoup warning when HTML content looks like a URL
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field extraction patterns
# ---------------------------------------------------------------------------

# Common CSS selectors / patterns for e-commerce listing fields.
# Each entry is a list of candidate selectors tried in order; first match wins.
_FIELD_SELECTORS: dict[str, list[str]] = {
    "title": [
        ".s-item__title",           # eBay
        "h2 a span",                # eBay fallback
        ".a-size-medium",           # Amazon
        "h2 .a-link-normal span",   # Amazon fallback
        "[data-component-type='s-product-image'] img",  # generic alt text
        "h3",                       # generic
    ],
    "price": [
        ".s-item__price",           # eBay
        ".a-price .a-offscreen",    # Amazon
        ".a-price-whole",           # Amazon whole part
        "[data-price]",             # generic data attribute
    ],
    "condition": [
        ".s-item__subtitle .SECONDARY_INFO",  # eBay
        ".a-size-base",             # Amazon (condition line)
    ],
    "seller": [
        ".s-item__seller-info-text",  # eBay
        ".a-size-small .a-link-normal",  # Amazon
    ],
    "shipping": [
        ".s-item__shipping",        # eBay
        ".s-item__delivery",        # eBay delivery
    ],
    "url": [
        ".s-item__link[href]",      # eBay
        "h2 a[href]",               # Amazon
        "a.a-link-normal[href]",    # Amazon fallback
    ],
}


def parse_product_listing(
    html: str = "",
    schema: dict[str, Any] | None = None,
    platform: str = "auto",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Parse raw HTML into a list of product listing dicts.

    Parameters
    ----------
    html:
        Raw HTML content from a search results page (eBay, Amazon, …).
    schema:
        Optional field schema.  If provided, only extract fields named in
        the schema's ``properties`` keys.  Otherwise extract all known fields.
    platform:
        Hint for the parser: ``"ebay"``, ``"amazon"``, or ``"auto"``.
        When ``"auto"`` and no listing containers are found, returns an empty
        list immediately (rather than trying the poor full-page fallback).
        The Executor's LLM should handle extraction from markdown instead.
    **kwargs:
        Accepts ``_raw`` fallback from malformed LLM arguments — attempts
        to extract HTML content from it.

    Returns
    -------
    list[dict]
        One dict per product listing found on the page.
    """
    # Normalize: if html is empty but we got _raw, try to extract from it
    if not html and "_raw" in kwargs:
        raw = kwargs["_raw"]
        if isinstance(raw, str):
            try:
                raw_obj = _json.loads(raw)
                if isinstance(raw_obj, dict):
                    html = raw_obj.get("html", raw_obj.get("content", raw))
            except (_json.JSONDecodeError, TypeError):
                html = raw

    # Handle dict input (LLM may pass structured data)
    if isinstance(html, dict):
        html = html.get("html", html.get("content", str(html)))

    if not html or not isinstance(html, str):
        logger.warning("No valid HTML content to parse")
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Determine which fields to extract
    if schema and "properties" in schema:
        fields = list(schema["properties"].keys())
    else:
        fields = list(_FIELD_SELECTORS.keys())

    # Find listing containers (generic approach)
    listings = _find_listing_containers(soup)

    if not listings:
        if platform == "auto":
            # Unknown platform — no CSS selectors match.  Don't waste time with
            # the full-page fallback; return empty and let the Executor's LLM
            # extract structured data from markdown instead.
            logger.info(
                "No listing containers found for platform=%r — "
                "returning empty; Executor LLM should parse markdown",
                platform,
            )
            return []
        # Known platform but no containers — try full-page fallback
        logger.warning(
            "No listing containers found for platform=%r — "
            "extracting from full page",
            platform,
        )
        listings = [soup]

    results: list[dict[str, Any]] = []
    for container in listings:
        item: dict[str, Any] = {}
        for field in fields:
            value = _extract_field(container, field)
            if value is not None:
                item[field] = value
        # Only include items that have at least a title or price
        if item.get("title") or item.get("price"):
            results.append(item)

    logger.info("Parsed %d listings from HTML", len(results))
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_listing_containers(soup: BeautifulSoup) -> list[Any]:
    """Find individual listing elements on the page."""
    # Try common listing container selectors (most specific first)
    container_selectors = [
        "li.s-item",                              # eBay search results
        ".srp-results li.s-item",                 # eBay (more specific)
        "div[data-viewport]" "[data-view] li",    # eBay modern layout
        "[data-component-type='s-search-result']", # Amazon
        "div.s-result-item",                      # Amazon
        "article.product",                        # generic
        ".listing-item",                          # generic
        "li[data-listing-id]",                    # generic
        "div[class*='listing']",                  # generic class pattern
        "li[class*='item']",                      # generic class pattern
    ]
    for selector in container_selectors:
        try:
            items = soup.select(selector)
            if len(items) >= 1:
                return items
        except Exception:
            continue

    # Fallback: find repeating <li> patterns in main content
    main = soup.select_one(
        "main, #main, .main, #content, .content, "
        "div[role='main'], .srp-results, #srp-river-results"
    )
    if main is None:
        main = soup

    # Look for ul > li patterns (most common listing format)
    for ul_selector in [
        "ul.srp-results", "ul.listings", "ul[class*='result']",
        "ul[class*='list']", "div[class*='result'] ul",
    ]:
        ul = main.select_one(ul_selector)
        if ul is not None:
            items = ul.select("li")
            if len(items) >= 2:
                return items

    # Last resort: any <li> elements with links inside (likely listings)
    li_items = main.select("li a[href]")
    if len(li_items) >= 3:
        # Return parent <li> elements
        parents = set()
        for a in li_items:
            parent_li = a.find_parent("li")
            if parent_li is not None:
                parents.add(parent_li)
        if len(parents) >= 2:
            return list(parents)

    return []


def _extract_field(container: Any, field: str) -> str | None:
    """Extract a single field from a listing container.

    Tries CSS selectors then falls back to text search.
    Cleans and normalizes the result.
    """
    selectors = _FIELD_SELECTORS.get(field, [])
    for selector in selectors:
        el = container.select_one(selector)
        if el is not None:
            text = el.get_text(strip=True)
            if field == "url" and el.get("href"):
                text = el["href"]
            if text:
                return _clean_text(text, field)
    return None


def _clean_text(text: str, field: str) -> str:
    """Normalize extracted text."""
    text = " ".join(text.split())  # collapse whitespace

    if field == "price":
        # Extract numeric price from text like "$1,234.56" or "1 234,56 VND"
        import re
        # Try to find a price pattern
        match = re.search(r'[\d,.]+', text.replace(" ", ""))
        if match:
            return match.group()

    return text
