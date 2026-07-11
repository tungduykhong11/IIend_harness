"""Handler for data_provider skill — scraping orchestration.  Spec 002 §4.3.

Provides the ``get_cached_listings`` custom action that aggregates
pre-extracted listings from ``fetch_web_page`` (which stores results
in ``web_fetcher._url_cache``).  The Executor LLM calls ``fetch_web_page``
for multiple URLs, then calls ``get_cached_listings`` to get the
pre-aggregated result in ``ScrapeResult`` format.
"""

import logging

logger = logging.getLogger(__name__)


class DataProviderSkill:
    """Handler for data_provider skill."""

    async def get_cached_listings(
        self, platform: str = "", query: str = ""
    ) -> dict:
        """
        Return all listings accumulated by fetch_web_page calls.
        INPUT: platform: str — platform name (e.g. 'cellphones.com.vn')
        INPUT: query: str — search query used
        OUTPUT: {listings: list[dict], total_scraped: int, total_valid: int, platform: str, query: str, errors: list}
        """
        from llend.parsers.web_fetcher import (
            _accumulated_errors,
            _accumulated_listings,
            _url_cache,
        )

        total = len(_accumulated_listings)
        logger.info(
            "get_cached_listings: %d accumulated from %d cached URLs",
            total, len(_url_cache),
        )

        return {
            "listings": _accumulated_listings,
            "total_scraped": total,
            "total_valid": total,
            "platform": platform,
            "query": query,
            "errors": list(_accumulated_errors),
        }
