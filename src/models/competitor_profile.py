"""Competitor profile model for structured competitor data.

This module defines the CompetitorProfile Pydantic model that represents
structured competitor information collected from various sources.

Example:
    ```python
    from src.models.competitor_profile import CompetitorProfile
    
    profile = CompetitorProfile(
        name="Competitor Inc",
        website="https://competitor.com",
        products=["Product A", "Product B"],
        source_url="https://source.com/article"
    )
    ```
"""

from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator


class CompetitorProfile(BaseModel):
    """Structured competitor data.
    
    This model represents structured information about a competitor collected
    from various sources. It includes basic information, products, pricing,
    and market presence data.
    
    Attributes:
        name: Competitor company name (required)
        website: Optional competitor website URL
        products: List of competitor products or services
        pricing: Optional pricing information as a dictionary
        market_presence: Optional description of market presence
        source_url: URL of the source where this information was found (required)
    """
    
    model_config = {"extra": "forbid"}
    
    name: str = Field(
        ...,
        description="Competitor company name",
        min_length=1,
    )
    
    website: HttpUrl | None = Field(
        default=None,
        description="Competitor website URL",
    )
    
    products: list[str] = Field(
        default_factory=list,
        description="List of competitor products or services",
    )
    
    pricing: dict[str, Any] | None = Field(
        default=None,
        description="Pricing information as a dictionary",
    )
    
    market_presence: str | None = Field(
        default=None,
        description="Description of market presence",
    )
    
    source_url: HttpUrl = Field(
        ...,
        description="URL of the source where this information was found",
    )
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Validate that name is not empty.
        
        Args:
            value: Company name string to validate
            
        Returns:
            Validated company name string
            
        Raises:
            ValueError: If name is empty or only whitespace
        """
        if not value or not value.strip():
            raise ValueError("Competitor name cannot be empty")
        return value.strip()
    
    @field_validator("products")
    @classmethod
    def validate_products(cls, value: list[str]) -> list[str]:
        """Validate and clean product list.
        
        Args:
            value: List of product strings to validate
            
        Returns:
            Validated list of product strings (empty strings removed)
        """
        return [product.strip() for product in value if product.strip()]
