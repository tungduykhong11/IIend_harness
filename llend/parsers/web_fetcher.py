"""Web fetcher — wraps crawl4ai's AsyncWebCrawler for the ``fetch_web_page`` action.

The ``ActionDispatcher`` calls functions directly, but ``AsyncWebCrawler``
requires async-context-manager lifecycle management.  This module provides
a plain async function that handles the lifecycle internally.
"""

from __future__ import annotations

import logging
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

logger = logging.getLogger(__name__)


async def fetch_web_page(
    url: str,
    stealth_mode: bool = True,
    user_agent: str = "llend-harness/0.1",
    **kwargs: Any,
) -> dict[str, Any]:
    """Fetch a web page and return clean markdown + metadata.

    Called by ``ActionDispatcher`` as the ``fetch_web_page`` action.
    Handles the ``AsyncWebCrawler`` lifecycle internally.
    """
    browser_config = BrowserConfig(headless=True)
    if user_agent:
        browser_config.user_agent = user_agent

    run_config = CrawlerRunConfig()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)

    return {
        "url": url,
        "markdown": result.markdown[:50000] if result.markdown else "",
        "cleaned_html": result.cleaned_html[:50000] if result.cleaned_html else "",
        "title": getattr(result, "title", ""),
        "success": result.success,
        "status_code": getattr(result, "status_code", 0),
    }
