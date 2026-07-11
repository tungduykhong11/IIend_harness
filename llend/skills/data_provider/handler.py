"""Handler for data_provider skill — product search.  Spec 002 §4.3.

Provides the ``search_listings`` custom action — 1 call, 1 complete result.
The handler does all the work internally (build URLs, crawl, extract via
crawl4ai, aggregate, dedup).  The Executor LLM just calls one action.

Pattern from financial-research-analyst: tools are complete units of work,
not low-level primitives.
"""

import logging
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


class DataProviderSkill:
    """Handler for data_provider skill."""

    # ── URL builders per platform ────────────────────────────────────

    @staticmethod
    def _build_urls(platform: str, query: str) -> list[str]:
        """Build search/product URLs for a platform + query.

        Returns 3-5 URLs.  Called internally — the LLM never sees URLs.
        """
        q = quote_plus(query)
        p = platform.lower()

        # CellphoneS (Vietnam)
        if "cellphone" in p:
            return [
                f"https://cellphones.com.vn/catalogsearch/result/?q={q}",
                f"https://cellphones.com.vn/iphone-15.html",
                f"https://cellphones.com.vn/iphone-15-256gb.html",
                f"https://cellphones.com.vn/iphone-15-pro-max.html",
                f"https://cellphones.com.vn/iphone-15-plus.html",
            ]

        # eBay
        if "ebay" in p:
            return [
                f"https://www.ebay.com/sch/i.html?_nkw={q}",
                f"https://www.ebay.com/sch/i.html?_nkw={q}&_sop=15",
            ]

        # Amazon
        if "amazon" in p:
            return [
                f"https://www.amazon.com/s?k={q}",
                f"https://www.amazon.com/s?k={q}&s=price-asc-rank",
            ]

        # Unknown — generic
        return [
            f"https://{platform}/search?q={q}",
            f"https://{platform}/catalogsearch/result/?q={q}",
        ]

    # ── search_listings ──────────────────────────────────────────────

    async def search_listings(self, platform: str, query: str) -> dict:
        """
        Search product listings on an e-commerce platform.
        Internally crawls multiple URLs, extracts listings, and returns
        a complete aggregated result.  Call this ONCE — no manual work.

        INPUT: platform: str — domain or platform name (e.g. 'cellphones.com.vn')
        INPUT: query: str — what to search for (e.g. 'iPhone 15')
        OUTPUT: {listings, total_scraped, total_valid, platform, query, errors}
        """
        from llend.parsers.web_fetcher import (
            _accumulated_errors,
            _accumulated_listings,
            clear_cache,
            fetch_web_page,
        )

        # Fresh start
        clear_cache()

        urls = self._build_urls(platform, query)
        logger.info("search_listings: %r / %r → %d URLs", platform, query, len(urls))

        for i, url in enumerate(urls):
            try:
                result = await fetch_web_page(
                    url=url, platform=platform, extract_listings=True,
                )
                logger.info(
                    "  [%d/%d] %s → %d listings (total: %d)",
                    i + 1, len(urls), url,
                    result.get("listing_count", 0),
                    result.get("total_accumulated", 0),
                )
            except Exception as exc:
                logger.warning("  [%d/%d] %s → %s", i + 1, len(urls), url, exc)
                _accumulated_errors.append(f"{url}: {exc}")

        total = len(_accumulated_listings)
        logger.info("search_listings done: %d listings, %d errors", total, len(_accumulated_errors))

        return {
            "listings": list(_accumulated_listings),
            "total_scraped": total,
            "total_valid": total,
            "platform": platform,
            "query": query,
            "errors": list(_accumulated_errors),
        }
