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
   Each call returns both per-URL listings AND `total_accumulated` — listings
   are auto-accumulated across calls (no manual aggregation needed).
3. Call `get_cached_listings(platform, query)` to get the final aggregated
   result in `ScrapeResult` format.
4. Use the returned result directly as your output.

## Notes
- **Always call `fetch_web_page` with `extract_listings=true`**.
- **Limit to 3-5 URLs max** — the accumulator handles combination.
- **Call `get_cached_listings` as your LAST action** — it returns the
  complete aggregated result. Use its output directly.
