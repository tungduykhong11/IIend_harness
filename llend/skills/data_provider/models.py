"""Pydantic models for data_provider skill — scraped product listings."""

from pydantic import BaseModel, Field


class ProductListing(BaseModel):
    """A single product listing scraped from an e-commerce platform."""

    title: str = Field(..., description="Product title as displayed on the platform")
    price: float = Field(..., description="Price in the listing's currency")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    condition: str = Field(default="unknown", description="e.g. 'new', 'used', 'refurbished'")
    seller: str = Field(default="unknown", description="Seller name or ID")
    shipping: str | None = Field(default=None, description="Shipping cost or 'free'")
    url: str = Field(..., description="Permalink to the listing")
    platform: str = Field(..., description="Platform source: 'ebay', 'amazon', etc.")
    scraped_at: str | None = Field(default=None, description="ISO 8601 timestamp when scraped")


class ScrapeConfig(BaseModel):
    """Configuration for a scraping session."""

    platform: str = Field(..., description="Target platform: 'ebay', 'amazon', etc.")
    query: str = Field(..., description="Search query string")
    max_items: int = Field(default=500, description="Maximum listings to collect")
    headless: bool = Field(default=True, description="Run browser in headless mode")
    stealth_mode: bool = Field(default=True, description="Enable anti-bot measures")
    request_delay: float = Field(default=2.0, description="Delay between requests in seconds")


class ScrapeResult(BaseModel):
    """Output of the data_provider skill — validated & deduplicated listings."""

    listings: list[ProductListing] = Field(default_factory=list)
    total_scraped: int = Field(default=0, description="Total listings scraped (before dedup)")
    total_valid: int = Field(default=0, description="Valid listings after cleaning & dedup")
    platform: str = ""
    query: str = ""
    errors: list[str] = Field(default_factory=list, description="Non-fatal errors encountered")
