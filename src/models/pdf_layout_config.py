"""PDF layout configuration model.

This module defines the PDFLayoutConfig Pydantic model that represents
layout configuration for PDF document generation, including page size,
orientation, margins, and column layout.

Example:
    ```python
    from src.models.pdf_layout_config import PDFLayoutConfig
    
    layout = PDFLayoutConfig(
        page_size="A4",
        orientation="portrait",
        margins={"top": 72, "bottom": 72, "left": 72, "right": 72},
        columns=1,
        header_height=0.5,
        footer_height=0.3
    )
    ```
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PDFLayoutConfig(BaseModel):
    """PDF layout configuration for document formatting.
    
    This model represents layout configuration for PDF documents, including
    page dimensions, orientation, margins, and column layout. All values are
    in points (1/72 inch) for margins and inches for header/footer heights.
    
    Attributes:
        page_size: Page size format (default: "A4", options: "A4", "Letter", "Legal")
        orientation: Page orientation (default: "portrait", options: "portrait", "landscape")
        margins: Dictionary with margin values in points (default: all 72pt)
        columns: Number of columns for text layout (default: 1, options: 1, 2, 3)
        header_height: Height of header in inches (default: 0.5)
        footer_height: Height of footer in inches (default: 0.3)
    """
    
    model_config = {"extra": "forbid"}
    
    page_size: Literal["A4", "Letter", "Legal"] = Field(
        default="A4",
        description="Page size format",
    )
    
    orientation: Literal["portrait", "landscape"] = Field(
        default="portrait",
        description="Page orientation",
    )
    
    margins: dict[str, float] = Field(
        default_factory=lambda: {"top": 72.0, "bottom": 72.0, "left": 72.0, "right": 72.0},
        description="Margin values in points (1/72 inch)",
    )
    
    columns: Literal[1, 2, 3] = Field(
        default=1,
        description="Number of columns for text layout",
    )
    
    header_height: float = Field(
        default=0.5,
        description="Height of header in inches",
        gt=0.0,
        le=2.0,
    )
    
    footer_height: float = Field(
        default=0.3,
        description="Height of footer in inches",
        gt=0.0,
        le=2.0,
    )
    
    @field_validator("margins")
    @classmethod
    def validate_margins(cls, value: dict[str, float]) -> dict[str, float]:
        """Validate margins dictionary structure and values.
        
        Args:
            value: Margins dictionary to validate
            
        Returns:
            Validated margins dictionary
            
        Raises:
            ValueError: If margins structure is invalid or values are negative
        """
        required_keys = {"top", "bottom", "left", "right"}
        
        if not isinstance(value, dict):
            raise ValueError("Margins must be a dictionary")
        
        missing_keys = required_keys - set(value.keys())
        if missing_keys:
            raise ValueError(
                f"Margins dictionary missing required keys: {missing_keys}"
            )
        
        extra_keys = set(value.keys()) - required_keys
        if extra_keys:
            raise ValueError(
                f"Margins dictionary contains invalid keys: {extra_keys}"
            )
        
        validated_margins: dict[str, float] = {}
        for key in required_keys:
            margin_value = value[key]
            if not isinstance(margin_value, (int, float)):
                raise ValueError(f"Margin '{key}' must be a number, got {type(margin_value).__name__}")
            
            if margin_value < 0:
                raise ValueError(f"Margin '{key}' must be non-negative, got {margin_value}")
            
            validated_margins[key] = float(margin_value)
        
        return validated_margins
    
    @field_validator("header_height", "footer_height")
    @classmethod
    def validate_height(cls, value: float) -> float:
        """Validate header/footer height values.
        
        Args:
            value: Height value to validate
            
        Returns:
            Validated height value
            
        Raises:
            ValueError: If height is not positive or exceeds maximum
        """
        if value <= 0:
            raise ValueError(f"Height must be positive, got {value}")
        
        if value > 2.0:
            raise ValueError(f"Height must not exceed 2.0 inches, got {value}")
        
        return float(value)

