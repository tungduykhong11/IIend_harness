---
name: analyze_pricing
version: 0.1.0
description: Analyze pricing from a product dataset — compute median, detect outliers, segment by price brackets.
inputs: dataset:list[dict], target_item:str, brackets:list[int]=[0,300,500,1000]
outputs: AnalysisReport
actions:
  - export_csv
  - calculate_market_median
  - detect_outliers_iqr
  - segment_by_price_range
dependencies:
  - data_provider
enforcement: strict
---

# Analyze Pricing Skill

## Flow
1. Receive dataset from `data_provider`
2. Compute market median price
3. Detect outlier listings (suspicious cheap & suspicious expensive)
4. Segment products by configurable price brackets
5. Export segments to CSV
6. Return `AnalysisReport` with recommendations

## Notes
- Outlier detection uses IQR method (Q1 - 1.5×IQR / Q3 + 1.5×IQR).
- If >20% of listings are outliers, raise `interrupt` to ask human whether to filter or keep.
- Default brackets [0, 300, 500, 1000] work for mid-range electronics. Adjust for luxury goods.
