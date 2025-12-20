"""Report model for final competitor analysis reports.

This module defines the Report Pydantic model that represents the final
formatted competitor analysis report.

Example:
    ```python
    from src.models.report_model import Report
    
    report = Report(
        executive_summary="Competitor analysis summary...",
        swot_breakdown="SWOT analysis details...",
        competitor_overview="Overview of competitors...",
        recommendations="Strategic recommendations...",
        methodology="Data collection approach...",
        sources=["https://example.com/source1"],
        min_length=500
    )
    ```
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Report(BaseModel):
    """Final competitor analysis report.
    
    This model represents the final formatted competitor analysis report
    with all required sections. It ensures minimum length requirements
    are met for quality assurance.
    
    Attributes:
        executive_summary: Executive summary of the analysis (minimum 200 characters)
        swot_breakdown: Detailed SWOT analysis breakdown (minimum 300 characters)
        competitor_overview: Overview of competitors analyzed (minimum 300 characters)
        recommendations: Strategic recommendations based on analysis (minimum 300 characters)
        methodology: Optional methodology section describing data collection approach (minimum 200 characters)
        sources: Optional list of source URLs with metadata
        min_length: Minimum total length of the report in characters (default: 1200)
    """
    
    model_config = {"extra": "forbid"}
    
    executive_summary: str = Field(
        ...,
        description="Executive summary of the analysis",
        min_length=200,
    )
    
    swot_breakdown: str = Field(
        ...,
        description="Detailed SWOT analysis breakdown",
        min_length=300,
    )
    
    competitor_overview: str = Field(
        ...,
        description="Overview of competitors analyzed",
        min_length=300,
    )
    
    recommendations: str = Field(
        ...,
        description="Strategic recommendations based on analysis",
        min_length=300,
    )
    
    methodology: str | None = Field(
        default=None,
        description="Methodology section describing data collection approach",
        min_length=200,
    )
    
    sources: list[str] | None = Field(
        default=None,
        description="List of source URLs with metadata",
    )
    
    min_length: int = Field(
        default=1200,
        description="Minimum total length of the report in characters",
        ge=1000,
        le=10000,
    )
    
    @field_validator("executive_summary", "swot_breakdown", "competitor_overview", "recommendations")
    @classmethod
    def validate_section(cls, value: str) -> str:
        """Validate that report section is not empty.
        
        Args:
            value: Report section string to validate
            
        Returns:
            Validated report section string
            
        Raises:
            ValueError: If section is empty or only whitespace
        """
        if not value or not value.strip():
            raise ValueError("Report section cannot be empty")
        return value.strip()
    
    @field_validator("methodology")
    @classmethod
    def validate_methodology(cls, value: str | None) -> str | None:
        """Validate methodology section if provided.
        
        Args:
            value: Methodology string to validate or None
            
        Returns:
            Validated methodology string or None
            
        Raises:
            ValueError: If methodology is provided but empty or too short
        """
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("Methodology cannot be empty if provided")
            if len(value) < 200:
                raise ValueError("Methodology must be at least 200 characters if provided")
        return value
    
    @model_validator(mode="after")
    def validate_sources_require_methodology(self) -> "Report":
        """Validate that methodology is included when sources are present.
        
        Returns:
            Self after validation
            
        Raises:
            ValueError: If sources are provided but methodology is missing
        """
        if self.sources and not self.methodology:
            raise ValueError(
                "Methodology section is required when sources are provided. "
                "Please include a methodology section describing how the sources were used."
            )
        return self
    
    def model_post_init(self, __context: object) -> None:
        """Validate total report length after initialization.
        
        Args:
            __context: Pydantic context (unused)
            
        Raises:
            ValueError: If total report length is less than min_length
        """
        total_length = (
            len(self.executive_summary)
            + len(self.swot_breakdown)
            + len(self.competitor_overview)
            + len(self.recommendations)
        )
        
        # Include methodology in length calculation if present
        if self.methodology:
            total_length += len(self.methodology)
        
        if total_length < self.min_length:
            raise ValueError(
                f"Total report length ({total_length}) is less than "
                f"minimum required length ({self.min_length})"
            )
