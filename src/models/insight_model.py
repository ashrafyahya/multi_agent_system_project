"""Insight model for business insights and SWOT analysis.

This module defines the Insight and SWOT Pydantic models that represent
business insights extracted from competitor data.

Example:
    ```python
    from src.models.insight_model import Insight, SWOT
    
    swot = SWOT(
        strengths=["Strong brand", "Market leader"],
        weaknesses=["High prices"],
        opportunities=["Emerging markets"],
        threats=["New competitors"]
    )
    
    insight = Insight(
        swot=swot,
        positioning="Premium market leader",
        trends=["Digital transformation", "AI integration"],
        opportunities=["Expansion into Asia", "B2B market"]
    )
    ```
"""

from pydantic import BaseModel, Field, field_validator


class SWOT(BaseModel):
    """SWOT analysis components.
    
    This model represents the four components of a SWOT analysis:
    Strengths, Weaknesses, Opportunities, and Threats.
    
    Attributes:
        strengths: List of competitor strengths
        weaknesses: List of competitor weaknesses
        opportunities: List of market opportunities
        threats: List of market threats
    """
    
    model_config = {"extra": "forbid"}
    
    strengths: list[str] = Field(
        default_factory=list,
        description="List of competitor strengths",
    )
    
    weaknesses: list[str] = Field(
        default_factory=list,
        description="List of competitor weaknesses",
    )
    
    opportunities: list[str] = Field(
        default_factory=list,
        description="List of market opportunities",
    )
    
    threats: list[str] = Field(
        default_factory=list,
        description="List of market threats",
    )
    
    @field_validator("strengths", "weaknesses", "opportunities", "threats")
    @classmethod
    def validate_swot_items(cls, value: list[str]) -> list[str]:
        """Validate and clean SWOT list items.
        
        Args:
            value: List of SWOT item strings to validate
            
        Returns:
            Validated list of SWOT items (empty strings removed)
        """
        return [item.strip() for item in value if item.strip()]


class Insight(BaseModel):
    """Business insights extracted from data.
    
    This model represents business insights extracted from competitor data,
    including SWOT analysis, positioning, trends, and opportunities.
    
    Attributes:
        swot: SWOT analysis object containing strengths, weaknesses,
            opportunities, and threats
        positioning: Description of competitor positioning in the market
        trends: List of market trends identified
        opportunities: List of business opportunities identified
    """
    
    model_config = {"extra": "forbid"}
    
    swot: SWOT = Field(
        ...,
        description="SWOT analysis object",
    )
    
    positioning: str = Field(
        ...,
        description="Description of competitor positioning",
        min_length=1,
    )
    
    trends: list[str] = Field(
        default_factory=list,
        description="List of market trends identified",
    )
    
    opportunities: list[str] = Field(
        default_factory=list,
        description="List of business opportunities identified",
    )
    
    @field_validator("positioning")
    @classmethod
    def validate_positioning(cls, value: str) -> str:
        """Validate that positioning is not empty.
        
        Args:
            value: Positioning string to validate
            
        Returns:
            Validated positioning string
            
        Raises:
            ValueError: If positioning is empty or only whitespace
        """
        if not value or not value.strip():
            raise ValueError("Positioning cannot be empty")
        return value.strip()
    
    @field_validator("trends", "opportunities")
    @classmethod
    def validate_list_items(cls, value: list[str]) -> list[str]:
        """Validate and clean list items.
        
        Args:
            value: List of item strings to validate
            
        Returns:
            Validated list of items (empty strings removed)
        """
        return [item.strip() for item in value if item.strip()]
