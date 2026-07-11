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
2. Crawl + extract listings via `fetch_web_page` (uses CSS for known sites, LLM for unknown)
3. Clean and normalize fields (price, title, condition, seller, shipping)
4. Deduplicate listings
5. Return `ScrapeResult` with validated product listings

## Notes
- **Always call `fetch_web_page` with `extract_listings=true`** — the tool
  already returns structured listings (CSS extraction for eBay/Amazon, LLM
  extraction for unknown sites).  No separate parsing step needed.
- The `listings` field in the response contains the structured product data.
  Aggregate listings from all fetched URLs into a single `ScrapeResult`.
- Rate limiting: respect platform's robots.txt, add 1-3s delay between pages.
- If results exceed `max_items`, use pagination to collect more.
- If platform blocks the scraper (CAPTCHA, 403), raise `interrupt` to ask human for alternative approach.
- Default max_items=500 works for quick analysis. Increase for comprehensive reports.
