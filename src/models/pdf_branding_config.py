"""PDF branding configuration model.

This module defines the PDFBrandingConfig Pydantic model that represents
branding configuration for PDF document generation, including company
information, colors, fonts, and template selection.

Example:
    ```python
    from src.models.pdf_branding_config import PDFBrandingConfig
    from pathlib import Path
    
    branding = PDFBrandingConfig(
        company_name="Acme Corp",
        company_logo_path=Path("./logos/acme.png"),
        primary_color="#1a1a1a",
        secondary_color="#0066cc",
        accent_color="#ff6600",
        font_family="Helvetica",
        header_font="Helvetica-Bold",
        footer_text="Confidential",
        cover_page_template="executive"
    )
    ```
"""

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PDFBrandingConfig(BaseModel):
    """PDF branding configuration for company customization.
    
    This model represents branding configuration for PDF documents, including
    company information, visual styling (colors, fonts), and template selection.
    All fields are optional except company_name to allow partial branding.
    
    Attributes:
        company_name: Company name for branding (required)
        company_logo_path: Path to company logo image file (optional)
        primary_color: Primary brand color in hex format (default: "#1a1a1a")
        secondary_color: Secondary brand color in hex format (default: "#0066cc")
        accent_color: Accent color for highlights in hex format (default: "#ff6600")
        font_family: Font family for body text (default: "Helvetica")
        header_font: Font family for headers (default: "Helvetica-Bold")
        footer_text: Optional footer text to display on each page
        watermark: Optional watermark text to display on pages
        cover_page_template: Template style for cover page
            (default: "default", options: "default", "executive", "minimal")
    """
    
    model_config = {"extra": "forbid"}
    
    company_name: str = Field(
        ...,
        description="Company name for branding",
        min_length=1,
    )
    
    company_logo_path: Path | None = Field(
        default=None,
        description="Path to company logo image file",
    )
    
    primary_color: str = Field(
        default="#1a1a1a",
        description="Primary brand color in hex format",
    )
    
    secondary_color: str = Field(
        default="#0066cc",
        description="Secondary brand color in hex format",
    )
    
    accent_color: str = Field(
        default="#ff6600",
        description="Accent color for highlights in hex format",
    )
    
    font_family: str = Field(
        default="Helvetica",
        description="Font family for body text",
    )
    
    header_font: str = Field(
        default="Helvetica-Bold",
        description="Font family for headers",
    )
    
    footer_text: str | None = Field(
        default=None,
        description="Optional footer text to display on each page",
    )
    
    watermark: str | None = Field(
        default=None,
        description="Optional watermark text to display on pages",
    )
    
    cover_page_template: Literal["default", "executive", "minimal"] = Field(
        default="default",
        description="Template style for cover page",
    )
    
    document_classification: str | None = Field(
        default=None,
        description="Optional document classification (e.g., 'Confidential', 'Draft', 'Final', 'Internal Use Only')",
    )
    
    @field_validator("company_name")
    @classmethod
    def validate_company_name(cls, value: str) -> str:
        """Validate that company name is not empty.
        
        Args:
            value: Company name string to validate
            
        Returns:
            Validated company name string
            
        Raises:
            ValueError: If company name is empty or only whitespace
        """
        if not value or not value.strip():
            raise ValueError("Company name cannot be empty")
        return value.strip()
    
    @field_validator("company_logo_path", mode="before")
    @classmethod
    def validate_logo_path(cls, value: str | Path | None) -> Path | None:
        """Validate and convert logo path.
        
        Args:
            value: Logo path as string, Path, or None
            
        Returns:
            Path object if provided, None otherwise
            
        Raises:
            ValueError: If path is provided but file does not exist
        """
        if value is None:
            return None
        
        path = Path(value) if isinstance(value, str) else value
        
        if not path.exists():
            raise ValueError(f"Logo file does not exist: {path}")
        
        if not path.is_file():
            raise ValueError(f"Logo path is not a file: {path}")
        
        return path
    
    @field_validator("primary_color", "secondary_color", "accent_color")
    @classmethod
    def validate_hex_color(cls, value: str) -> str:
        """Validate hex color format.
        
        Args:
            value: Hex color string to validate (e.g., "#1a1a1a" or "#1A1A1A")
            
        Returns:
            Validated hex color string in lowercase
            
        Raises:
            ValueError: If color is not a valid hex format
        """
        # Pattern for hex color: # followed by 3 or 6 hex digits
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{3}([0-9A-Fa-f]{3})?$")
        
        if not hex_pattern.match(value):
            raise ValueError(
                f"Color must be a valid hex format (e.g., '#1a1a1a' or '#1A1A1A'), "
                f"got: {value}"
            )
        
        return value.lower()
    
    @field_validator("font_family", "header_font")
    @classmethod
    def validate_font_family(cls, value: str) -> str:
        """Validate that font family is not empty.
        
        Args:
            value: Font family string to validate
            
        Returns:
            Validated font family string
            
        Raises:
            ValueError: If font family is empty or only whitespace
        """
        if not value or not value.strip():
            raise ValueError("Font family cannot be empty")
        return value.strip()
    
    @field_validator("footer_text", "watermark")
    @classmethod
    def validate_text_field(cls, value: str | None) -> str | None:
        """Validate optional text fields.
        
        Args:
            value: Text string or None to validate
            
        Returns:
            Validated text string (stripped) or None
            
        Raises:
            ValueError: If value is empty string (use None instead)
        """
        if value is not None and not value.strip():
            raise ValueError("Text field cannot be empty string, use None instead")
        return value.strip() if value else None

