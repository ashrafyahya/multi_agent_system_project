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
    market presence data, and quantitative metrics.
    
    Attributes:
        name: Competitor company name (required)
        website: Optional competitor website URL
        products: List of competitor products or services
        pricing: Optional pricing information as a dictionary
        market_presence: Optional description of market presence
        source_url: URL of the source where this information was found (required)
        market_share: Optional market share percentage (0-100)
        revenue: Optional revenue figure (float or string for ranges like "$1B-$2B")
        user_count: Optional user count (int or string for ranges)
        founded_year: Optional year the company was founded
        headquarters: Optional headquarters location
        key_features: Optional list of key product features
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
    
    market_share: float | None = Field(
        default=None,
        description="Market share percentage (0-100)",
        ge=0.0,
        le=100.0,
    )
    
    revenue: float | str | None = Field(
        default=None,
        description="Revenue figure (float for exact amount or string for ranges like '$1B-$2B')",
    )
    
    user_count: int | str | None = Field(
        default=None,
        description="User count (int for exact count or string for ranges)",
    )
    
    founded_year: int | None = Field(
        default=None,
        description="Year the company was founded",
        ge=1800,
        le=2100,
    )
    
    headquarters: str | None = Field(
        default=None,
        description="Headquarters location",
    )
    
    key_features: list[str] = Field(
        default_factory=list,
        description="List of key product features",
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
    
    @field_validator("key_features")
    @classmethod
    def validate_key_features(cls, value: list[str]) -> list[str]:
        """Validate and clean key features list.
        
        Args:
            value: List of feature strings to validate
            
        Returns:
            Validated list of feature strings (empty strings removed)
        """
        return [feature.strip() for feature in value if feature.strip()]
    
    @field_validator("headquarters")
    @classmethod
    def validate_headquarters(cls, value: str | None) -> str | None:
        """Validate headquarters location.
        
        Args:
            value: Headquarters string to validate or None
            
        Returns:
            Validated headquarters string or None
            
        Raises:
            ValueError: If headquarters is provided but empty
        """
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("Headquarters cannot be empty if provided")
        return value
