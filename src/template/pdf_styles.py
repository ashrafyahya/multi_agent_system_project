"""Custom PDF styles for professional document formatting.

This module provides custom style definitions that enhance the default
ReportLab stylesheet with professional typography, spacing, and colors.
"""

import logging
from typing import Any

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from src.models.pdf_branding_config import PDFBrandingConfig

logger = logging.getLogger(__name__)


def hex_to_color(hex_color: str) -> colors.Color:
    """Convert hex color string to ReportLab Color object.
    
    Args:
        hex_color: Hex color string (e.g., "#1a1a1a" or "#0066cc")
        
    Returns:
        ReportLab Color object
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return colors.Color(r, g, b)
    return colors.black


def get_professional_styles(
    branding_config: PDFBrandingConfig | None = None
) -> Any:
    """Get professional PDF styles with custom formatting.
    
    Creates a custom stylesheet with improved typography, spacing, and colors
    for professional document appearance. Uses branding colors if provided.
    
    Args:
        branding_config: Optional PDF branding configuration
        
    Returns:
        Customized ReportLab stylesheet
    """
    styles = getSampleStyleSheet()
    
    # Get branding colors or use defaults
    primary_color = colors.Color(0.1, 0.1, 0.1)  # Dark gray default
    secondary_color = colors.Color(0.0, 0.4, 0.8)  # Blue default
    accent_color = colors.Color(1.0, 0.4, 0.0)  # Orange default
    
    if branding_config:
        try:
            primary_color = hex_to_color(branding_config.primary_color)
            secondary_color = hex_to_color(branding_config.secondary_color)
            accent_color = hex_to_color(branding_config.accent_color)
        except Exception as e:
            logger.warning(f"Failed to parse branding colors: {e}. Using defaults.")
    
    # Customize Title style (H1)
    styles["Title"].fontSize = 24
    styles["Title"].fontName = "Helvetica-Bold"
    styles["Title"].textColor = primary_color
    styles["Title"].leading = 30
    styles["Title"].spaceAfter = 0.5 * inch
    styles["Title"].alignment = 0  # Left align
    
    # Customize Heading1 style (H2)
    styles["Heading1"].fontSize = 18
    styles["Heading1"].fontName = "Helvetica-Bold"
    styles["Heading1"].textColor = primary_color
    styles["Heading1"].leading = 22
    styles["Heading1"].spaceBefore = 0.35 * inch
    styles["Heading1"].spaceAfter = 0.25 * inch
    styles["Heading1"].alignment = 0  # Left align
    # Add subtle underline for H2 headings
    styles["Heading1"].borderWidth = 0
    styles["Heading1"].borderPadding = 0
    
    # Customize Heading2 style (H3)
    styles["Heading2"].fontSize = 14
    styles["Heading2"].fontName = "Helvetica-Bold"
    styles["Heading2"].textColor = secondary_color
    styles["Heading2"].leading = 18
    styles["Heading2"].spaceBefore = 0.25 * inch
    styles["Heading2"].spaceAfter = 0.18 * inch
    styles["Heading2"].alignment = 0  # Left align
    
    # Customize Normal (paragraph) style
    styles["Normal"].fontSize = 11
    styles["Normal"].fontName = "Helvetica"
    styles["Normal"].textColor = colors.black
    styles["Normal"].leading = 15  # Better line spacing (1.36x font size)
    styles["Normal"].spaceAfter = 0.12 * inch
    styles["Normal"].alignment = 4  # Justify text
    styles["Normal"].firstLineIndent = 0  # No indent
    
    # Create custom styles for better formatting
    # Bullet list style
    if "BulletList" not in styles.byName:
        styles.add(ParagraphStyle(
            name="BulletList",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
            spaceAfter=0.08 * inch,
            leftIndent=0.25 * inch,
            bulletIndent=0.15 * inch,
        ))
    
    # Numbered list style
    if "NumberedList" not in styles.byName:
        styles.add(ParagraphStyle(
            name="NumberedList",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
            spaceAfter=0.08 * inch,
            leftIndent=0.25 * inch,
        ))
    
    # Table header style
    if "TableHeader" not in styles.byName:
        styles.add(ParagraphStyle(
            name="TableHeader",
            parent=styles["Normal"],
            fontSize=11,
            fontName="Helvetica-Bold",
            textColor=colors.white,
            leading=13,
            alignment=1,  # Center
        ))
    
    # Table cell style
    # Note: Text wrapping is automatic when Paragraph objects are placed in Tables
    # with defined column widths (which we now provide via colWidths)
    if "TableCell" not in styles.byName:
        styles.add(ParagraphStyle(
            name="TableCell",
            parent=styles["Normal"],
            fontSize=10,
            leading=12,
            alignment=0,  # Left
        ))
    
    return styles


def get_professional_table_style(
    branding_config: PDFBrandingConfig | None = None
) -> list[tuple]:
    """Get professional table style with better colors.
    
    Args:
        branding_config: Optional PDF branding configuration
        
    Returns:
        List of TableStyle tuples for ReportLab
    """
    # Get branding colors or use professional defaults
    header_bg = colors.Color(0.2, 0.3, 0.5)  # Professional dark blue
    header_text = colors.white
    row_bg_even = colors.Color(0.95, 0.95, 0.97)  # Light gray
    row_bg_odd = colors.white
    border_color = colors.Color(0.7, 0.7, 0.7)  # Medium gray
    
    if branding_config:
        try:
            secondary_color = hex_to_color(branding_config.secondary_color)
            # Use secondary color for header, but ensure it's dark enough
            header_bg = secondary_color
            # Make header darker if it's too light
            if sum(header_bg.rgb) > 1.5:
                header_bg = colors.Color(
                    max(0, secondary_color.rgb[0] * 0.6),
                    max(0, secondary_color.rgb[1] * 0.6),
                    max(0, secondary_color.rgb[2] * 0.6)
                )
        except Exception as e:
            logger.warning(f"Failed to use branding color for table: {e}")
    
    return [
        # Header row styling
        ('BACKGROUND', (0, 0), (-1, 0), header_bg),
        ('TEXTCOLOR', (0, 0), (-1, 0), header_text),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        
        # Alternating row colors for better readability
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [row_bg_odd, row_bg_even]),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        
        # Alignment
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        
        # Grid and borders
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('LINEBELOW', (0, 0), (-1, 0), 2, header_bg),
        
        # Padding
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]

