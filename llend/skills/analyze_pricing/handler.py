"""Handler for analyze_pricing — custom pricing analysis actions.  Spec 002 §4.3."""

import logging
import statistics

logger = logging.getLogger(__name__)


class AnalyzePricingSkill:
    """Handler for analyze_pricing."""

    # ── calculate_market_median ───────────────────────────────────────

    async def calculate_market_median(self, prices: list[float]) -> dict:
        """
        Compute median and basic price statistics.
        INPUT: prices: list[float] — list of prices from dataset
        OUTPUT: {median: float, mean: float, min: float, max: float, count: int}
        """
        if not prices:
            return {"median": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        return {
            "median": statistics.median(prices),
            "mean": statistics.mean(prices),
            "min": min(prices),
            "max": max(prices),
            "count": len(prices),
        }

    # ── detect_outliers_iqr ───────────────────────────────────────────

    async def detect_outliers_iqr(self, prices: list[float]) -> dict:
        """
        Detect price outliers using IQR method.
        INPUT: prices: list[float] — list of prices from dataset
        OUTPUT: {suspicious_cheap: list[{index, price, reason}], suspicious_expensive: list[...]}
        """
        if len(prices) < 4:
            return {"suspicious_cheap": [], "suspicious_expensive": []}

        sorted_prices = sorted(prices)
        n = len(sorted_prices)

        q1 = statistics.median(sorted_prices[: n // 2])
        q3 = statistics.median(sorted_prices[(n + 1) // 2:])
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        suspicious_cheap = []
        suspicious_expensive = []

        for i, price in enumerate(prices):
            if price < lower_bound:
                suspicious_cheap.append({
                    "index": i,
                    "price": price,
                    "reason": f"Below Q1-1.5×IQR (lower_bound={lower_bound:.2f})",
                })
            elif price > upper_bound:
                suspicious_expensive.append({
                    "index": i,
                    "price": price,
                    "reason": f"Above Q3+1.5×IQR (upper_bound={upper_bound:.2f})",
                })

        return {
            "suspicious_cheap": suspicious_cheap,
            "suspicious_expensive": suspicious_expensive,
        }

    # ── segment_by_price_range ────────────────────────────────────────

    async def segment_by_price_range(
        self, data: list[dict], brackets: list[int]
    ) -> dict:
        """
        Segment data by configurable price brackets.
        INPUT: data: list[dict] — listings with at least {'price': float, 'title': str}
        INPUT: brackets: list[int] — price boundaries, e.g. [0, 300, 500, 1000]
        OUTPUT: {segments: [{range_label, lower, upper, count, avg_price, sample_products}]}
        """
        segments = []

        for i, lower in enumerate(brackets):
            upper = brackets[i + 1] if i + 1 < len(brackets) else None
            label = f"{lower}-{upper}" if upper is not None else f"{lower}+"

            matching = [
                item for item in data
                if item.get("price", 0) >= lower
                and (upper is None or item.get("price", 0) < upper)
            ]

            prices_in_range = [item.get("price", 0) for item in matching]
            avg = statistics.mean(prices_in_range) if prices_in_range else 0.0

            segments.append({
                "range_label": label,
                "lower": lower,
                "upper": upper,
                "count": len(matching),
                "avg_price": round(avg, 2),
                "sample_products": [
                    item.get("title", "") for item in matching[:5]
                ],
            })

        return {"segments": segments}
