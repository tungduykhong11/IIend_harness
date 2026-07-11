"""Web fetcher — wraps crawl4ai's AsyncWebCrawler for the ``fetch_web_page`` action.

Extraction strategy (crawl4ai reference pattern):
- **Known sites (eBay, Amazon):** CSS extraction via JsonCssExtractionStrategy
  (fast, deterministic, zero LLM cost)
- **Unknown sites (CellphoneS, any):** Return clean markdown — the Executor's
  ReAct-loop LLM extracts structured data itself (no extra API call needed;
  crawl4ai's markdown output is LLM-ready)

Also provides a module-level URL cache to prevent duplicate crawls across
retries.
"""

from __future__ import annotations

import logging
from typing import Any

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    JsonCssExtractionStrategy,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL cache — prevents duplicate crawling across retries
# ---------------------------------------------------------------------------

_url_cache: dict[str, dict[str, Any]] = {}


def clear_cache() -> None:
    """Clear the URL fetch cache (useful between sessions)."""
    _url_cache.clear()


# ---------------------------------------------------------------------------
# Known-site CSS extraction schemas
# ---------------------------------------------------------------------------

_EBAY_SCHEMA = {
    "name": "eBay Listings",
    "baseSelector": "li.s-item",
    "fields": [
        {"name": "title", "selector": ".s-item__title", "type": "text"},
        {"name": "price", "selector": ".s-item__price", "type": "text"},
        {"name": "subtitle", "selector": ".s-item__subtitle", "type": "text"},
        {"name": "url", "selector": ".s-item__link", "type": "attribute", "attribute": "href"},
        {"name": "shipping", "selector": ".s-item__shipping", "type": "text"},
        {"name": "seller", "selector": ".s-item__seller-info-text", "type": "text"},
        {"name": "condition", "selector": ".SECONDARY_INFO", "type": "text"},
    ],
}

_KNOWN_SCHEMAS: dict[str, dict[str, Any]] = {
    "ebay": _EBAY_SCHEMA,
    # "amazon": _AMAZON_SCHEMA,  # add when needed
}


# ---------------------------------------------------------------------------
# fetch_web_page  —  called by ActionDispatcher via tool_bridge/mappings.toml
# ---------------------------------------------------------------------------


async def fetch_web_page(
    url: str,
    platform: str = "auto",
    stealth_mode: bool = True,
    user_agent: str = "llend-harness/0.1",
    extract_listings: bool = False,
    use_cache: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fetch a web page and return clean data.

    Parameters
    ----------
    url:
        The page URL to fetch.
    platform:
        Hint for extraction: ``"ebay"``, ``"amazon"``, or ``"auto"`` for
        unknown sites.  Known platforms use CSS extraction; unknown sites
        return clean markdown for the Executor's LLM to parse.
    stealth_mode:
        Enable crawl4ai anti-bot measures (stealth mode).
    user_agent:
        Custom User-Agent header.
    extract_listings:
        If True and *platform* is known, use CSS extraction.  If *platform*
        is ``"auto"``, markdown is always returned — the Executor's LLM
        handles extraction itself.
    use_cache:
        If True (default), return cached result for duplicate URLs.
    """
    # --- URL cache ---
    if use_cache and url in _url_cache:
        logger.info("Cache hit: %s", url)
        return _url_cache[url]

    browser_config = BrowserConfig(headless=True)
    if user_agent:
        browser_config.user_agent = user_agent

    # Decide extraction strategy based on platform
    schema = _KNOWN_SCHEMAS.get(platform) if extract_listings else None
    run_config: CrawlerRunConfig
    if schema is not None:
        strategy = JsonCssExtractionStrategy(schema)
        run_config = CrawlerRunConfig(extraction_strategy=strategy)
    else:
        # Unknown platform or no extraction requested — just get clean markdown.
        # The Executor's ReAct-loop LLM will extract structured data from it.
        run_config = CrawlerRunConfig()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)

    # Parse structured listings if CSS extraction ran
    listings: list[dict[str, Any]] = []
    if result.extracted_content:
        try:
            import json as _json

            extracted = result.extracted_content
            if isinstance(extracted, str):
                extracted = _json.loads(extracted)
            if isinstance(extracted, list):
                listings = extracted
        except Exception:
            logger.warning("Failed to parse extracted content", exc_info=True)

    response = {
        "url": url,
        "markdown": result.markdown[:50000] if result.markdown else "",
        "cleaned_html": result.cleaned_html[:50000] if result.cleaned_html else "",
        "title": getattr(result, "title", ""),
        "success": result.success,
        "status_code": getattr(result, "status_code", 0),
        "listings": listings,
        "listing_count": len(listings),
    }

    # Cache the result
    if use_cache:
        _url_cache[url] = response

    return response
