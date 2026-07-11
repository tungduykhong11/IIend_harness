---
name: data_provider
version: 0.1.0
description: Scrape product listings from e-commerce platforms — crawl, extract, clean, dedup.
inputs: platform:str, query:str, max_items:int=500
outputs: ScrapeResult
actions:
  - fetch_web_page
dependencies: []
enforcement: suggested
---

# Data Provider Skill

## Flow
1. Receive platform + query from Orchestrator
2. Call `fetch_web_page` for 3-5 key product URLs.
   Each response includes `accumulated` — the running total of ALL listings
   fetched so far (auto-accumulated, deduplicated). No manual aggregation.
3. After your last fetch, use `response["accumulated"]` directly as your
   output — it is already in `ScrapeResult` format.

## Notes
- **Always call `fetch_web_page` with `extract_listings=true`**.
- **Limit to 3-5 URLs.** The accumulator handles combination.
- **Use `response["accumulated"]` as your final output** — it's pre-formatted.
- **Build URLs from the platform domain in task_spec** — do NOT guess or
  invent domains.  If task_spec says "cellphones.com.vn", use exactly that.
