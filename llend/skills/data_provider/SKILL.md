---
name: data_provider
version: 0.1.0
description: Scrape product listings from e-commerce platforms — crawl, extract, clean, dedup.
inputs: platform:str, query:str, max_items:int=500
outputs: ScrapeResult
actions:
  - fetch_web_page
  - parse_listing_html
dependencies: []
enforcement: suggested
---

# Data Provider Skill

## Flow
1. Receive platform + query from Orchestrator
2. Crawl the platform's search results via `fetch_web_page`
3. Parse listing HTML via `parse_listing_html`
4. Clean and normalize fields (price, title, condition, seller, shipping)
5. Deduplicate listings
6. Return `ScrapeResult` with validated product listings

## Notes
- Rate limiting: respect platform's robots.txt, add 1-3s delay between pages.
- If results exceed `max_items`, use pagination to collect more.
- If platform blocks the scraper (CAPTCHA, 403), raise `interrupt` to ask human for alternative approach.
- Default max_items=500 works for quick analysis. Increase for comprehensive reports.
