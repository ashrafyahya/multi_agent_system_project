"""PDF formatting utilities for lists, tables, and images.

This module provides formatting functions for converting markdown content
to ReportLab flowables for PDF generation.
"""

import logging
from pathlib import Path
from typing import Any

from src.models.pdf_branding_config import PDFBrandingConfig
from src.models.pdf_layout_config import PDFLayoutConfig
from src.template.markdown_converter import convert_markdown_to_html
from src.template.pdf_styles import get_professional_table_style

logger = logging.getLogger(__name__)


def format_list_items(items: list[str], styles: Any) -> list[Any]:
    """Format list items for PDF using reportlab.

    Args:
        items: List of item strings
        styles: ReportLab stylesheet

    Returns:
        List of reportlab flowables (Paragraphs and Spacers)
    """
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer

    flowables = []
    for item in items:
        # Remove trailing "-" if present (common issue from LLM formatting)
        # This is a safety net in case trailing "-" weren't removed earlier
        while item.rstrip().endswith(' -'):
            item = item.rstrip()[:-2].rstrip()
        if item.rstrip().endswith('-') and len(item.rstrip()) > 1:
            # Only remove if preceded by space (not part of citation like "[1]-")
            if item.rstrip()[-2] == ' ':
                item = item.rstrip()[:-1].rstrip()
        # Use bullet style for list items
        flowables.append(Paragraph(f"• {item}", styles["Normal"]))
        flowables.append(Spacer(1, 0.08 * inch))
    # Add spacing after list
    flowables.append(Spacer(1, 0.15 * inch))
    return flowables


def format_numbered_list_items(items: list[str], styles: Any) -> list[Any]:
    """Format numbered list items for PDF using reportlab.

    Args:
        items: List of item strings
        styles: ReportLab stylesheet

    Returns:
        List of reportlab flowables (Paragraphs and Spacers)
    """
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer

    flowables = []
    for idx, item in enumerate(items, start=1):
        # Use numbered format: "1. item", "2. item", etc.
        formatted_text = f"{idx}. {item}"
        flowables.append(Paragraph(formatted_text, styles["Normal"]))
        flowables.append(Spacer(1, 0.08 * inch))
    # Add spacing after list
    flowables.append(Spacer(1, 0.15 * inch))
    return flowables


def format_table(
    rows: list[list[str]], 
    styles: Any, 
    branding_config: PDFBrandingConfig | None = None,
    layout_config: PDFLayoutConfig | None = None
) -> list[Any]:
    """Format markdown table as PDF table with professional styling.

    Args:
        rows: List of table rows, each row is a list of cells
        styles: ReportLab stylesheet
        branding_config: Optional PDF branding configuration
        layout_config: Optional PDF layout configuration for calculating column widths

    Returns:
        List of reportlab flowables
    """
    from reportlab.lib.pagesizes import A4, legal, letter
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    # Validate table before rendering
    if not rows or len(rows) == 0:
        logger.warning("format_table called with empty rows")
        return []
    
    if len(rows) < 2:
        logger.warning(f"format_table: Table has {len(rows)} rows, minimum 2 required (header + 1 data row)")
        return []
    
    # Check column counts before normalization
    col_counts = [len(row) for row in rows]
    if min(col_counts) < 2:
        logger.warning(f"format_table: Table has row with <2 columns, minimum 2 required")
        return []
    if max(col_counts) != min(col_counts):
        logger.warning(
            f"format_table: Inconsistent column counts: min={min(col_counts)}, max={max(col_counts)}. "
            "Table should have been validated before calling format_table."
        )
        return []

    flowables = []

    # All rows have the same column count (validated above)
    max_cols = col_counts[0]
    logger.debug(f"Formatting table with {len(rows)} rows and {max_cols} columns")
    
    # Use rows as-is since column counts are consistent
    normalized_rows = rows

    # Convert cells to Paragraphs with appropriate styles
    table_data = []
    for row_idx, row in enumerate(normalized_rows):
        formatted_row = []
        for cell in row:
            cell_html = convert_markdown_to_html(cell)
            # Use TableHeader style for first row, TableCell for others
            if row_idx == 0 and "TableHeader" in styles.byName:
                cell_style = styles["TableHeader"]
            elif "TableCell" in styles.byName:
                cell_style = styles["TableCell"]
            else:
                cell_style = styles["Normal"]
            formatted_row.append(Paragraph(cell_html, cell_style))
        table_data.append(formatted_row)

    # Calculate column widths based on page size and margins
    if layout_config:
        page_size_map = {
            "A4": A4,
            "Letter": letter,
            "Legal": legal,
        }
        pagesize = page_size_map.get(layout_config.page_size, letter)
        margins = layout_config.margins
        left_margin = margins.get("left", 72.0)
        right_margin = margins.get("right", 72.0)
    else:
        # Default to Letter page with 72pt margins
        pagesize = letter
        left_margin = 72.0
        right_margin = 72.0
    
    # Calculate available width (page width minus margins)
    page_width = pagesize[0]
    available_width = page_width - left_margin - right_margin
    
    # Enforce fixed column widths with deterministic calculation
    # Use fixed widths based on column count to ensure consistent rendering
    min_col_width = 72.0  # 1 inch minimum per column
    if max_cols > 0:
        equal_width = available_width / max_cols
        
        # Use equal width if it's above minimum
        if equal_width >= min_col_width:
            col_widths = [equal_width] * max_cols
        else:
            # If too many columns, use minimum width (table may overflow, but better than collapse)
            col_widths = [min_col_width] * max_cols
            logger.warning(
                f"Table has {max_cols} columns, using minimum width {min_col_width}pt per column. "
                f"Table may exceed page width."
            )
    else:
        col_widths = None

    # Create table with calculated column widths
    if col_widths:
        table = Table(table_data, colWidths=col_widths)
    else:
        table = Table(table_data)

    # Apply professional table style
    table_style_list = get_professional_table_style(branding_config)
    table_style = TableStyle(table_style_list)
    table.setStyle(table_style)

    flowables.append(table)
    flowables.append(Spacer(1, 0.2 * inch))

    return flowables


def embed_image(story: list[Any], image_path: str, caption: str) -> None:
    """Embed an image in the PDF story.

    Args:
        story: List of reportlab flowables
        image_path: Path to image file
        caption: Caption text for the image
    """
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, Paragraph, Spacer

    try:
        styles = getSampleStyleSheet()
        img_path = Path(image_path)

        if img_path.exists():
            # Add caption
            story.append(Paragraph(caption, styles["Heading2"]))
            story.append(Spacer(1, 0.1 * inch))

            # Add image (scale to fit page width)
            img = Image(str(img_path), width=6 * inch, height=4.5 * inch)
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))
        else:
            logger.warning(f"Image not found: {image_path}")
    except Exception as e:
        logger.warning(f"Failed to embed image {image_path}: {e}")

