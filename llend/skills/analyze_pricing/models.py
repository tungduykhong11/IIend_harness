"""Pydantic models for analyze_pricing skill — Spec 002 §3.2."""


from pydantic import BaseModel, Field


class MarketSummary(BaseModel):
    """Tổng quan thị trường."""

    median_price: float = Field(..., description="Median price across all valid listings")
    mean_price: float = Field(..., description="Mean price")
    min_price: float
    max_price: float
    normal_range: tuple[float, float] = Field(
        ..., description="[lower_bound, upper_bound] — IQR normal range"
    )
    total_listings: int
    outlier_count: int = Field(..., description="Total suspicious listings detected")


class OutlierDetail(BaseModel):
    """Một listing bất thường."""

    index: int
    price: float
    title: str
    reason: str = Field(
        ..., description="Why this listing is flagged (e.g. 'Below Q1-1.5*IQR')"
    )


class OutlierReport(BaseModel):
    """Báo cáo outlier."""

    suspicious_cheap: list[OutlierDetail] = Field(default_factory=list)
    suspicious_expensive: list[OutlierDetail] = Field(default_factory=list)
    cheap_count: int
    expensive_count: int


class PriceSegment(BaseModel):
    """Một khoảng giá."""

    range_label: str = Field(..., description='e.g. "0-300", "1000+"')
    lower: float
    upper: float | None = None  # None = unlimited upper
    count: int
    avg_price: float
    sample_products: list[str] = Field(
        default_factory=list, description="Top 5 product titles in this segment"
    )


class AnalysisReport(BaseModel):
    """Output chính của analyze_pricing."""

    market_summary: MarketSummary
    outliers: OutlierReport
    price_segments: list[PriceSegment]
    recommendation: str = Field(
        ..., description="Human-readable buying advice based on the data"
    )
    export_csv_path: str | None = None
