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
        min_length=500
    )
    ```
"""

from pydantic import BaseModel, Field, field_validator


class Report(BaseModel):
    """Final competitor analysis report.
    
    This model represents the final formatted competitor analysis report
    with all required sections. It ensures minimum length requirements
    are met for quality assurance.
    
    Attributes:
        executive_summary: Executive summary of the analysis
        swot_breakdown: Detailed SWOT analysis breakdown
        competitor_overview: Overview of competitors analyzed
        recommendations: Strategic recommendations based on analysis
        min_length: Minimum total length of the report in characters (default: 500)
    """
    
    model_config = {"extra": "forbid"}
    
    executive_summary: str = Field(
        ...,
        description="Executive summary of the analysis",
        min_length=50,
    )
    
    swot_breakdown: str = Field(
        ...,
        description="Detailed SWOT analysis breakdown",
        min_length=50,
    )
    
    competitor_overview: str = Field(
        ...,
        description="Overview of competitors analyzed",
        min_length=50,
    )
    
    recommendations: str = Field(
        ...,
        description="Strategic recommendations based on analysis",
        min_length=50,
    )
    
    min_length: int = Field(
        default=500,
        description="Minimum total length of the report in characters",
        ge=100,
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
        
        if total_length < self.min_length:
            raise ValueError(
                f"Total report length ({total_length}) is less than "
                f"minimum required length ({self.min_length})"
            )
