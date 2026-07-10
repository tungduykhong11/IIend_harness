"""Web fetcher — wraps crawl4ai's AsyncWebCrawler for the ``fetch_web_page`` action.

Uses crawl4ai's built-in ``JsonCssExtractionStrategy`` to extract structured
product listings directly from e-commerce pages (eBay, Amazon).  Falls back
to raw markdown/HTML on extraction failure.
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
# eBay listing extraction schema
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


async def fetch_web_page(
    url: str,
    stealth_mode: bool = True,
    user_agent: str = "llend-harness/0.1",
    extract_listings: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fetch a web page and return clean data.

    When *extract_listings* is True, uses crawl4ai's CSS extraction to
    pull structured product listings from the page.  Falls back to
    markdown on extraction failure.
    """
    browser_config = BrowserConfig(headless=True)
    if user_agent:
        browser_config.user_agent = user_agent

    run_config: CrawlerRunConfig
    if extract_listings:
        strategy = JsonCssExtractionStrategy(_EBAY_SCHEMA)
        run_config = CrawlerRunConfig(extraction_strategy=strategy)
    else:
        run_config = CrawlerRunConfig()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)

    # If extraction produced structured data, return it directly
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

    return {
        "url": url,
        "markdown": result.markdown[:50000] if result.markdown else "",
        "cleaned_html": result.cleaned_html[:50000] if result.cleaned_html else "",
        "title": getattr(result, "title", ""),
        "success": result.success,
        "status_code": getattr(result, "status_code", 0),
        "listings": listings,
        "listing_count": len(listings),
    }
