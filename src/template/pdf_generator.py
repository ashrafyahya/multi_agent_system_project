"""PDF generator for creating professional PDF documents from markdown reports.

This module provides the main PDF generation functionality, orchestrating
the template engine, markdown parsing, and PDF document creation.
"""

import logging
import re
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4, legal, letter
from reportlab.platypus import SimpleDocTemplate

from src.exceptions.workflow_error import WorkflowError
from src.models.pdf_branding_config import PDFBrandingConfig
from src.models.pdf_layout_config import PDFLayoutConfig
from src.template.markdown_parser import parse_markdown_to_story
from src.template.pdf_styles import get_professional_styles
from src.template.pdf_utils import set_pdf_metadata
from src.template.template_engine import DefaultPDFTemplateEngine

logger = logging.getLogger(__name__)


def _get_page_size_and_margins(
    layout_config: PDFLayoutConfig | None,
) -> tuple[Any, float, float, float, float]:
    """Get page size and margins from layout config or defaults.

    Args:
        layout_config: Optional PDF layout configuration

    Returns:
        Tuple of (pagesize, right_margin, left_margin, top_margin, bottom_margin)
    """
    if layout_config:
        page_size_map = {
            "A4": A4,
            "Letter": letter,
            "Legal": legal,
        }
        pagesize = page_size_map.get(layout_config.page_size, letter)
        margins = layout_config.margins
        right_margin = margins.get("right", 72.0)
        left_margin = margins.get("left", 72.0)
        top_margin = margins.get("top", 72.0)
        bottom_margin = margins.get("bottom", 72.0)
    else:
        pagesize = letter
        right_margin = 72.0
        left_margin = 72.0
        top_margin = 72.0
        bottom_margin = 18.0

    return pagesize, right_margin, left_margin, top_margin, bottom_margin


def _extract_report_title(report: str) -> str:
    """Extract report title from first H1 heading or use default.

    Args:
        report: Markdown-formatted report string

    Returns:
        Report title string
    """
    report_title = "Competitor Analysis Report"
    title_match = re.search(r'^#\s+(.+)$', report, re.MULTILINE)
    if title_match:
        report_title = title_match.group(1).strip()
    return report_title


def _setup_template_engine(
    report: str,
    branding_config: PDFBrandingConfig | None,
    layout_config: PDFLayoutConfig | None,
    story: list[Any],
) -> DefaultPDFTemplateEngine | None:
    """Setup template engine and add cover page.

    Args:
        report: Markdown-formatted report string
        branding_config: Optional PDF branding configuration
        layout_config: Optional PDF layout configuration
        story: List of flowables to append to

    Returns:
        Template engine instance or None if setup failed
    """
    report_title = _extract_report_title(report)

    # Ensure we have configs (should always be set by get_pdf_configs defaults)
    if branding_config is None:
        branding_config = PDFBrandingConfig(company_name="Competitor Analysis Report")
    if layout_config is None:
        layout_config = PDFLayoutConfig()

    try:
        template_engine = DefaultPDFTemplateEngine(
            branding_config=branding_config,
            layout_config=layout_config,
            report_title=report_title,
        )

        # Add cover page
        cover_page_flowables = template_engine.create_cover_page()
        if cover_page_flowables:
            story.extend(cover_page_flowables)
            logger.debug("Cover page added to PDF")

        return template_engine

    except Exception as e:
        logger.warning(
            f"Template engine failed, using basic PDF: {e}. "
            "Continuing with standard PDF generation."
        )
        return None


def _create_pdf_callbacks(
    template_engine: DefaultPDFTemplateEngine | None,
    report_title: str,
    report: str,
    branding_config: PDFBrandingConfig | None,
) -> tuple[Any, Any]:
    """Create PDF callbacks for headers, footers, and metadata.

    Args:
        template_engine: Optional template engine instance
        report_title: Report title
        report: Full report content
        branding_config: Optional branding configuration

    Returns:
        Tuple of (on_first_page, on_later_pages) callback functions
    """
    def on_first_page(canvas: Any, doc: Any) -> None:
        """Draw header and footer on first page."""
        # Set metadata on first page
        set_pdf_metadata(canvas, report_title, report, branding_config)

        if template_engine:
            header_callback = template_engine.create_header(1, 0)  # 0 = unknown total
            footer_callback = template_engine.create_footer(1, 0)
            if header_callback:
                header_callback(canvas, doc)
            if footer_callback:
                footer_callback(canvas, doc)

    def on_later_pages(canvas: Any, doc: Any) -> None:
        """Draw header and footer on subsequent pages."""
        if template_engine:
            page_num = canvas.getPageNumber()
            header_callback = template_engine.create_header(page_num, 0)
            footer_callback = template_engine.create_footer(page_num, 0)
            if header_callback:
                header_callback(canvas, doc)
            if footer_callback:
                footer_callback(canvas, doc)

    return on_first_page, on_later_pages


def export_to_pdf(
    report: str,
    output_dir: Path,
    base_filename: str,
    branding_config: PDFBrandingConfig | None = None,
    layout_config: PDFLayoutConfig | None = None,
) -> Path:
    """Export report to PDF format with professional formatting.

    Converts markdown report to a professional PDF document with support for:
    - Cover page, headers, footers
    - PDF metadata (title, author, keywords)
    - Markdown formatting (headings, lists, tables, etc.)

    Args:
        report: Markdown-formatted report string
        output_dir: Directory to save PDF file
        base_filename: Base filename (without extension)
        branding_config: Optional PDF branding configuration
        layout_config: Optional PDF layout configuration

    Returns:
        Path to generated PDF file

    Raises:
        WorkflowError: If PDF generation fails (e.g., reportlab not installed)
    """
    try:
        # Check for reportlab
        try:
            from reportlab.platypus import SimpleDocTemplate
        except ImportError as e:
            raise WorkflowError(
                "reportlab library is required for PDF export. "
                "Please install it: pip install reportlab",
                context={"error": str(e)}
            ) from e

        pdf_path = output_dir / f"{base_filename}.pdf"

        # Get page size and margins
        pagesize, right_margin, left_margin, top_margin, bottom_margin = (
            _get_page_size_and_margins(layout_config)
        )

        # Create PDF document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=pagesize,
            rightMargin=right_margin,
            leftMargin=left_margin,
            topMargin=top_margin,
            bottomMargin=bottom_margin,
        )

        # Build PDF content with professional styles
        styles = get_professional_styles(branding_config)
        story: list[Any] = []

        # Setup template engine and add cover page
        template_engine = _setup_template_engine(
            report, branding_config, layout_config, story
        )

        # Parse markdown to story
        markdown_story = parse_markdown_to_story(report, styles, branding_config, layout_config)
        story.extend(markdown_story)

        # Create callbacks
        report_title = _extract_report_title(report)
        on_first_page, on_later_pages = _create_pdf_callbacks(
            template_engine, report_title, report, branding_config
        )

        # Build PDF
        if template_engine:
            doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
            logger.debug("PDF built with template engine (cover, headers, footers)")
        else:
            doc.build(story, onFirstPage=on_first_page)
            logger.debug("PDF built with standard generation and metadata")

        return pdf_path

    except WorkflowError:
        raise
    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}", exc_info=True)
        raise WorkflowError(
            "PDF generation failed",
            context={"error": str(e), "output_dir": str(output_dir)}
        ) from e

