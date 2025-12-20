"""Header and footer generation for PDF documents.

This module provides functionality for generating headers and footers
that appear on each page of the PDF document.
"""

import logging
from datetime import datetime
from typing import Any

from src.models.pdf_branding_config import PDFBrandingConfig

from src.template.style_utils import get_branding_color

logger = logging.getLogger(__name__)


def create_header(
    branding_config: PDFBrandingConfig | None,
    report_title: str,
    page_num: int,
    total_pages: int,
) -> Any:
    """Create header callback function for PDF pages.
    
    This function returns a callable that will be used by ReportLab's
    onFirstPage/onLaterPages callbacks to draw headers on each page.
    The callback receives a canvas object and document object.
    
    Args:
        branding_config: PDF branding configuration (optional)
        report_title: Title of the report
        page_num: Current page number (1-indexed)
        total_pages: Total number of pages in document
        
    Returns:
        Callable function(canvas, doc) for drawing header, or None if no header
    """
    if not branding_config:
        return None
    
    def draw_header(canvas: Any, doc: Any) -> None:
        """Draw header on PDF page using canvas.
        
        Args:
            canvas: ReportLab canvas object
            doc: ReportLab document object
        """
        try:
            from reportlab.lib.colors import HexColor
            from reportlab.lib.units import inch
            
            # Get page dimensions
            page_width = doc.pagesize[0]
            page_height = doc.pagesize[1]
            
            # Get branding colors
            primary_color = get_branding_color(branding_config, "primary")
            primary_hex = (
                branding_config.primary_color
                if branding_config
                else "#1a1a1a"
            )
            
            # Set font and color
            canvas.setFont("Helvetica-Bold", 10)
            canvas.setFillColor(HexColor(primary_hex))
            
            # Draw company name/logo area
            y_position = page_height - 0.5 * inch
            
            # Company name
            if branding_config:
                company_name = branding_config.company_name
                canvas.drawString(0.75 * inch, y_position, company_name)
            
            # Report title (right-aligned)
            title_width = canvas.stringWidth(report_title, "Helvetica-Bold", 10)
            canvas.drawString(
                page_width - title_width - 0.75 * inch,
                y_position,
                report_title,
            )
            
            # Draw line separator
            canvas.setStrokeColor(HexColor(primary_hex))
            canvas.setLineWidth(0.5)
            canvas.line(
                0.75 * inch,
                y_position - 0.1 * inch,
                page_width - 0.75 * inch,
                y_position - 0.1 * inch
            )
            
        except Exception as e:
            logger.warning(f"Error drawing header: {e}")
    
    return draw_header


def create_footer(
    branding_config: PDFBrandingConfig | None,
    report_date: datetime,
    page_num: int,
    total_pages: int,
) -> Any:
    """Create footer callback function for PDF pages.
    
    This function returns a callable that will be used by ReportLab's
    onFirstPage/onLaterPages callbacks to draw footers on each page.
    The callback receives a canvas object and document object.
    
    Args:
        branding_config: PDF branding configuration (optional)
        report_date: Date of the report
        page_num: Current page number (1-indexed)
        total_pages: Total number of pages in document
        
    Returns:
        Callable function(canvas, doc) for drawing footer, or None if no footer
    """
    def draw_footer(canvas: Any, doc: Any) -> None:
        """Draw footer on PDF page using canvas.
        
        Args:
            canvas: ReportLab canvas object
            doc: ReportLab document object
        """
        try:
            from reportlab.lib.colors import HexColor
            from reportlab.lib.units import inch
            
            # Get page dimensions
            page_width = doc.pagesize[0]
            
            # Get branding colors
            primary_color = get_branding_color(branding_config, "primary")
            primary_hex = (
                branding_config.primary_color
                if branding_config
                else "#1a1a1a"
            )
            
            # Set font and color
            canvas.setFont("Helvetica", 9)
            canvas.setFillColor(HexColor(primary_hex))
            
            # Footer Y position
            y_position = 0.5 * inch
            
            # Get actual page number from canvas
            actual_page_num = canvas.getPageNumber()
            
            # Date (left side)
            date_str = report_date.strftime("%B %d, %Y")
            canvas.drawString(0.75 * inch, y_position, date_str)
            
            # Page number (center) - use actual page number
            page_text = f"Page {actual_page_num}"
            page_width_text = canvas.stringWidth(page_text, "Helvetica", 9)
            canvas.drawString(
                (page_width - page_width_text) / 2,
                y_position,
                page_text,
            )
            
            # Footer text (right side, if provided)
            if branding_config and branding_config.footer_text:
                footer_text = branding_config.footer_text
                footer_width = canvas.stringWidth(footer_text, "Helvetica", 9)
                canvas.drawString(
                    page_width - footer_width - 0.75 * inch,
                    y_position,
                    footer_text,
                )
            
            # Draw line separator
            canvas.setStrokeColor(HexColor(primary_hex))
            canvas.setLineWidth(0.5)
            canvas.line(
                0.75 * inch,
                y_position + 0.15 * inch,
                page_width - 0.75 * inch,
                y_position + 0.15 * inch
            )
            
        except Exception as e:
            logger.warning(f"Error drawing footer: {e}")
    
    return draw_footer

