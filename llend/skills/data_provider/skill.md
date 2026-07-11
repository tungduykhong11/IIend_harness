---
name: data_provider
version: 0.1.0
description: Scrape product listings from e-commerce platforms — crawl, extract, clean, dedup.
inputs: platform:str, query:str, max_items:int=500
outputs: ScrapeResult
actions:
  - fetch_web_page
  - get_cached_listings
dependencies: []
enforcement: suggested
---

# Data Provider Skill

## Flow
1. Receive platform + query from Orchestrator
2. Call `fetch_web_page` for 3-5 key URLs (search page + top product pages).
   Each call already returns structured `listings` — no parsing needed.
3. Call `get_cached_listings(platform, query)` to get ALL accumulated listings
   in `ScrapeResult` format (pre-aggregated, deduplicated).
4. Use the returned result directly as your output.

## Notes
- **Always call `fetch_web_page` with `extract_listings=true`**.
- **Limit to 3-5 URLs max** — the `get_cached_listings` action handles
  aggregation.  Do NOT fetch more than 5 URLs.
- **Call `get_cached_listings` as your LAST action** — it returns the final
  aggregated result.  Use its output directly (no further processing needed).
