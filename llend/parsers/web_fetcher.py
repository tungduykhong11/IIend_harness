"""Web fetcher — wraps crawl4ai's AsyncWebCrawler for the ``fetch_web_page`` action.

Extraction strategy (crawl4ai reference pattern):
- **Known sites (eBay, Amazon):** CSS extraction via JsonCssExtractionStrategy
  (fast, deterministic, zero LLM cost)
- **Unknown sites (CellphoneS, any):** LLM extraction via LLMExtractionStrategy
  (crawl4ai's built-in — pass markdown + instruction, LLM returns structured JSON)

Also provides a module-level URL cache to prevent duplicate crawls across
retries.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    JsonCssExtractionStrategy,
    LLMConfig,
    LLMExtractionStrategy,
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
# LLM extraction instruction — used for unknown sites
# ---------------------------------------------------------------------------

_EXTRACTION_INSTRUCTION = (
    "Extract all product listings from this page. "
    "For each listing, return: title, price (as a number), currency, "
    "condition (new/used/unknown), seller name, shipping info, "
    "and the product URL. "
    "Return as a JSON array of objects."
)


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
        use crawl4ai's ``LLMExtractionStrategy`` with an LLM instruction.
    stealth_mode:
        Enable crawl4ai anti-bot measures (stealth mode).
    user_agent:
        Custom User-Agent header.
    extract_listings:
        If True, use the extraction strategy appropriate for *platform*.
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
        # Known platform → fast CSS extraction
        strategy = JsonCssExtractionStrategy(schema)
        run_config = CrawlerRunConfig(extraction_strategy=strategy)
    elif extract_listings:
        # Unknown platform → LLM extraction (crawl4ai reference pattern)
        strategy = _build_llm_strategy()
        run_config = CrawlerRunConfig(extraction_strategy=strategy)
    else:
        run_config = CrawlerRunConfig()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)

    # Parse structured listings
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_llm_strategy() -> LLMExtractionStrategy:
    """Build an LLMExtractionStrategy using the configured LLM provider.

    Uses environment variables (DEEPSEEK_API_KEY or OPENAI_API_KEY) to
    configure crawl4ai's built-in LLM extraction.  Falls back to
    crawl4ai's default provider if no env vars are set.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if api_key:
        # Use the project's configured LLM provider
        provider = os.environ.get("LLEND_PROVIDER", "deepseek")
        if provider == "deepseek":
            llm_config = LLMConfig(
                provider="openai/deepseek-chat",
                api_token=api_key,
                base_url="https://api.deepseek.com/v1",
            )
        else:
            # Anthropic or other — use as-is
            llm_config = LLMConfig(
                provider=f"{provider}/claude-sonnet-4-20250514",
                api_token=api_key,
            )
    else:
        # No API key configured — let crawl4ai use its defaults
        logger.warning("No LLM API key found for extraction — using crawl4ai defaults")
        llm_config = None

    return LLMExtractionStrategy(
        llm_config=llm_config,
        instruction=_EXTRACTION_INSTRUCTION,
        extraction_type="schema",
        input_format="markdown",  # crawl4ai's clean markdown — LLM-friendly
    )
