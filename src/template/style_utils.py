"""Styling utilities for PDF template generation.

This module provides utility functions for color conversion, branding colors,
and responsive spacing calculations used across PDF template components.
"""

import logging
from typing import Any

from src.models.pdf_branding_config import PDFBrandingConfig
from src.models.pdf_layout_config import PDFLayoutConfig

logger = logging.getLogger(__name__)


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color to RGB tuple for ReportLab.
    
    Args:
        hex_color: Hex color string (e.g., "#1a1a1a" or "#1a1")
        
    Returns:
        RGB tuple with values 0.0-1.0
    """
    hex_color = hex_color.lstrip("#")
    
    # Handle 3-digit hex colors
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    
    # Convert to RGB (0-255) then normalize to 0.0-1.0
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    
    return (r, g, b)


def get_branding_color(
    branding_config: PDFBrandingConfig | None, color_type: str
) -> tuple[float, float, float]:
    """Get branding color as RGB tuple.
    
    Args:
        branding_config: PDF branding configuration (optional)
        color_type: Type of color ("primary", "secondary", "accent")
        
    Returns:
        RGB tuple with values 0.0-1.0, defaults to black if no branding
    """
    if not branding_config:
        return (0.0, 0.0, 0.0)  # Black default
    
    color_map = {
        "primary": branding_config.primary_color,
        "secondary": branding_config.secondary_color,
        "accent": branding_config.accent_color,
    }
    
    hex_color = color_map.get(color_type, "#000000")
    return hex_to_rgb(hex_color)


def calculate_responsive_spacing(
    template_style: str,
    has_logo: bool,
    title_length: int,
    metadata_count: int,
    page_height: float | None = None,
) -> dict[str, float]:
    """Calculate responsive spacing based on content and template.
    
    This function calculates dynamic spacing that adapts to:
    - Template style (executive, default, minimal)
    - Presence of logo
    - Title length
    - Number of metadata fields
    - Page height (if provided)
    
    Args:
        template_style: Template style ("executive", "default", "minimal")
        has_logo: Whether logo is present
        title_length: Length of title in characters
        metadata_count: Number of metadata fields
        page_height: Page height in points (optional, defaults to A4)
        
    Returns:
        Dictionary with spacing values in inches:
        - top_margin: Top spacing
        - logo_spacing: Spacing after logo
        - title_spacing_before: Spacing before title
        - title_spacing_after: Spacing after title
        - metadata_spacing: Spacing between metadata items
        - bottom_margin: Bottom spacing
    """
    from reportlab.lib.pagesizes import A4

    # Default page height (A4 portrait in points)
    if page_height is None:
        page_height = A4[1]  # Height in points
    
    # Convert to inches (1 point = 1/72 inch)
    page_height_inches = page_height / 72.0
    
    # Base spacing multipliers by template
    base_multipliers = {
        "executive": {"top": 0.15, "bottom": 0.20},
        "default": {"top": 0.12, "bottom": 0.15},
        "minimal": {"top": 0.18, "bottom": 0.25},
    }
    
    multiplier = base_multipliers.get(template_style, base_multipliers["default"])
    
    # Calculate base top margin (proportional to page height)
    base_top = page_height_inches * multiplier["top"]
    base_bottom = page_height_inches * multiplier["bottom"]
    
    # Adjust for logo presence
    if has_logo:
        # Less top spacing needed when logo is present
        top_adjustment = -0.3
        logo_spacing = 0.5
    else:
        top_adjustment = 0.0
        logo_spacing = 0.0
    
    # Adjust for title length (longer titles need more space)
    title_length_factor = min(title_length / 50.0, 1.5)  # Cap at 1.5x
    title_spacing_before = 0.2 + (title_length_factor * 0.1)
    title_spacing_after = 0.6 + (title_length_factor * 0.2)
    
    # Adjust for metadata count
    metadata_spacing = 0.3 + (metadata_count * 0.1)
    
    # Calculate final spacing with min/max constraints
    top_margin = max(0.8, min(3.0, base_top + top_adjustment))
    bottom_margin = max(1.0, min(3.5, base_bottom))
    
    return {
        "top_margin": top_margin,
        "logo_spacing": logo_spacing,
        "title_spacing_before": title_spacing_before,
        "title_spacing_after": title_spacing_after,
        "metadata_spacing": metadata_spacing,
        "bottom_margin": bottom_margin,
    }


def get_page_height(layout_config: PDFLayoutConfig | None) -> float | None:
    """Get page height from layout configuration.
    
    Args:
        layout_config: PDF layout configuration (optional)
        
    Returns:
        Page height in points, or None if not configured
    """
    if not layout_config:
        return None
    
    from reportlab.lib.pagesizes import A4, legal, letter
    
    page_size_map = {
        "A4": A4,
        "Letter": letter,
        "Legal": legal,
    }
    
    pagesize = page_size_map.get(layout_config.page_size, A4)
    return pagesize[1]  # Height in points

