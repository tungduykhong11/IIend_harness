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
        from llend.parsers.web_fetcher import _url_cache

        all_listings: list[dict] = []
        errors: list[str] = []

        for url, cached in _url_cache.items():
            if cached.get("success") and cached.get("listings"):
                all_listings.extend(cached["listings"])
            elif not cached.get("success"):
                errors.append(f"Failed to fetch {url}: status {cached.get('status_code')}")

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique: list[dict] = []
        for item in all_listings:
            item_url = item.get("url", "")
            if item_url and item_url not in seen_urls:
                seen_urls.add(item_url)
                unique.append(item)
            elif not item_url:
                unique.append(item)

        total = len(all_listings)
        valid = len(unique)

        logger.info(
            "get_cached_listings: %d total, %d unique from %d cached URLs",
            total, valid, len(_url_cache),
        )

        return {
            "listings": unique,
            "total_scraped": total,
            "total_valid": valid,
            "platform": platform,
            "query": query,
            "errors": errors,
        }
