---
name: analyze_pricing
version: 0.2.0
description: Analyze pricing from a product dataset — one call, complete report with recommendation.
inputs: dataset:list[dict]
outputs: AnalysisReport
actions:
  - analyze_prices
dependencies:
  - data_provider
enforcement: strict
---

# Analyze Pricing Skill

## Flow
1. Receive `dataset` from `data_provider`
2. Call `analyze_prices(dataset)` — returns complete analysis with:
   - `market`: median, mean, min, max, count
   - `outliers`: cheap/expensive lists with reasons
   - `segments`: price bracket breakdowns with samples
   - `recommendation`: human-readable Vietnamese text
3. Use the returned result directly as your output

## Notes
- `analyze_prices` handles everything internally (stats, IQR outliers, segmentation, text).
  Call it ONCE — the result is ready to use.
- Pass `dataset` exactly as received from data_provider's output.
