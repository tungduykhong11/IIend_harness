"""Handler for analyze_pricing skill — complete market analysis.  Spec 002 §4.3.

Provides the ``analyze_prices`` custom action — 1 call, 1 complete result.
The handler does all calculations internally (median, mean, IQR outliers,
price segments, recommendation).  The Executor LLM just calls one action.

Pattern from financial-research-analyst: tools are complete units of work.
"""

import logging
import statistics

logger = logging.getLogger(__name__)


class AnalyzePricingSkill:
    """Handler for analyze_pricing skill."""

    async def analyze_prices(self, dataset: list[dict]) -> dict:
        """
        Analyze a product dataset: compute price stats, detect outliers,
        segment by price range, and generate a natural-language recommendation.
        Call this ONCE — returns a complete analysis report.

        INPUT: dataset: list[dict] — listings with at least {'price': float, 'title': str}
        OUTPUT: {market: {median, mean, min, max, count},
                 outliers: {cheap: [...], expensive: [...], total},
                 segments: [{label, lower, upper, count, avg_price, samples}],
                 recommendation: str}
        """
        if not dataset:
            return {
                "market": {"median": 0, "mean": 0, "min": 0, "max": 0, "count": 0},
                "outliers": {"cheap": [], "expensive": [], "total": 0},
                "segments": [],
                "recommendation": "No data available for analysis.",
            }

        # Extract prices
        prices: list[float] = []
        for item in dataset:
            p = item.get("price", 0)
            if isinstance(p, (int, float)) and p > 0:
                prices.append(float(p))

        if not prices:
            return {
                "market": {"median": 0, "mean": 0, "min": 0, "max": 0, "count": 0},
                "outliers": {"cheap": [], "expensive": [], "total": 0},
                "segments": [],
                "recommendation": "No valid prices found in dataset.",
            }

        sorted_prices = sorted(prices)
        n = len(sorted_prices)

        # ── Market stats ─────────────────────────────────────────────
        median = statistics.median(sorted_prices)
        mean = statistics.mean(sorted_prices)
        min_p = min(prices)
        max_p = max(prices)

        # ── Outlier detection (IQR) ───────────────────────────────────
        q1 = statistics.median(sorted_prices[: n // 2])
        q3 = statistics.median(sorted_prices[(n + 1) // 2:])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        cheap = []
        expensive = []
        for item in dataset:
            p = item.get("price", 0)
            if isinstance(p, (int, float)) and p > 0:
                if p < lower:
                    cheap.append({
                        "title": str(item.get("title", ""))[:80],
                        "price": p,
                        "reason": f"Below Q1-1.5*IQR (threshold: {lower:,.0f})",
                    })
                elif p > upper:
                    expensive.append({
                        "title": str(item.get("title", ""))[:80],
                        "price": p,
                        "reason": f"Above Q3+1.5*IQR (threshold: {upper:,.0f})",
                    })

        # ── Price segments ────────────────────────────────────────────
        brackets = [0, 10_000_000, 20_000_000, 30_000_000]
        segments = []
        for i, lo in enumerate(brackets):
            hi = brackets[i + 1] if i + 1 < len(brackets) else None
            label = f"{lo/1e6:.0f}-{hi/1e6:.0f}M" if hi else f"{lo/1e6:.0f}M+"
            in_range = [p for p in prices if p >= lo and (hi is None or p < hi)]
            samples = []
            for item in dataset:
                p = item.get("price", 0)
                if isinstance(p, (int, float)) and p >= lo and (hi is None or p < hi):
                    samples.append(str(item.get("title", ""))[:60])
                    if len(samples) >= 3:
                        break
            segments.append({
                "label": label,
                "lower": lo,
                "upper": hi,
                "count": len(in_range),
                "avg_price": round(statistics.mean(in_range), 0) if in_range else 0,
                "samples": samples,
            })

        # ── Recommendation ────────────────────────────────────────────
        rec = self._build_recommendation(median, mean, min_p, max_p, n, len(cheap) + len(expensive), segments)

        logger.info(
            "analyze_prices: %d items, median=%.0f, mean=%.0f, outliers=%d",
            n, median, mean, len(cheap) + len(expensive),
        )

        return {
            "market": {"median": median, "mean": round(mean, 0), "min": min_p, "max": max_p, "count": n},
            "outliers": {"cheap": cheap, "expensive": expensive, "total": len(cheap) + len(expensive)},
            "segments": segments,
            "recommendation": rec,
        }

    @staticmethod
    def _build_recommendation(
        median: float, mean: float, min_p: float, max_p: float,
        count: int, outlier_count: int, segments: list[dict],
    ) -> str:
        """Build a human-readable Vietnamese recommendation."""
        def fmt(v: float) -> str:
            if v >= 1_000_000:
                return f"{v/1_000_000:,.1f} triệu"
            return f"{v:,.0f}"

        parts = [
            f"Giá trung bình (median) là {fmt(median)} VNĐ, "
            f"dao động từ {fmt(min_p)} đến {fmt(max_p)} VNĐ "
            f"dựa trên {count} sản phẩm.",
        ]

        if outlier_count > 0:
            parts.append(
                f"Có {outlier_count} sản phẩm có giá bất thường "
                f"(quá rẻ hoặc quá đắt so với mặt bằng chung)."
            )

        # Most popular segment
        best_seg = max(segments, key=lambda s: s["count"]) if segments else None
        if best_seg and best_seg["count"] > 0:
            parts.append(
                f"Nhiều sản phẩm nhất nằm trong khoảng {best_seg['label']} VNĐ "
                f"({best_seg['count']} sản phẩm, giá trung bình {fmt(best_seg['avg_price'])})."
            )

        if mean > median * 1.1:
            parts.append("Giá trung bình cao hơn median — có thể có vài sản phẩm cao cấp kéo giá lên.")
        elif median > mean * 1.1:
            parts.append("Median cao hơn trung bình — phần lớn sản phẩm tập trung ở phân khúc cao.")

        return " ".join(parts)
