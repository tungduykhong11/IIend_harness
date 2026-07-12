---
name: data_provider
version: 0.2.0
description: Search product listings from e-commerce platforms — one call, complete result.
inputs: platform:str, query:str
outputs: ScrapeResult
actions:
  - search_listings
dependencies: []
enforcement: strict
---

# Data Provider Skill

## Flow
1. Receive `platform` + `query` from Orchestrator
2. Call `search_listings(platform, query)` — returns complete `ScrapeResult`
3. Use the returned result directly as your output

## Notes
- `search_listings` handles everything internally (crawl, extract, aggregate, dedup).
  Call it ONCE — the result is ready to use.
- Pass `platform` and `query` exactly as received from the task spec.
