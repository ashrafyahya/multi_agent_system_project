"""PDF utility functions for metadata, bookmarks, and keywords.

This module provides utility functions for PDF document generation,
including metadata setting, bookmark extraction and management, keyword
extraction, and PDF configuration handling.

These utilities are separated from the main PDF generator to follow
the Single Responsibility Principle and improve code organization.
"""

import logging
import re
from typing import Any

from src.models.pdf_branding_config import PDFBrandingConfig
from src.models.pdf_layout_config import PDFLayoutConfig

logger = logging.getLogger(__name__)


def set_pdf_metadata(
    canvas: Any,
    report_title: str,
    report_content: str,
    branding_config: PDFBrandingConfig | None = None,
) -> None:
    """Set PDF document metadata properties via canvas.

    Sets title, author, subject, keywords, creator, and creation date
    for the PDF document. These properties are visible in PDF viewers.
    Note: Metadata must be set via canvas, not SimpleDocTemplate.

    Args:
        canvas: ReportLab canvas object
        report_title: Title of the report
        report_content: Full report content (for keyword extraction)
        branding_config: Optional branding configuration for author name
    """
    try:
        # Set metadata via canvas document info
        canvas.setTitle(report_title)

        # Set author (from branding config or default)
        if branding_config:
            author = branding_config.company_name
        else:
            author = "Ashraf Yahya"
        canvas.setAuthor(author)

        # Set subject
        canvas.setSubject("Competitor Analysis Report")

        # Extract keywords from report content
        keywords = extract_keywords(report_content)
        if keywords:
            canvas.setKeywords(keywords)

        # Set creator with GitHub link
        canvas.setCreator("Multi-Agent Analysis System v1.1 | https://github.com/ashrafyahya")

        logger.debug(
            f"PDF metadata set: title='{report_title}', author='{author}'"
        )

    except Exception as e:
        logger.warning(f"Failed to set PDF metadata: {e}")
        # Don't fail PDF generation if metadata setting fails


def extract_bookmarks(report_content: str) -> list[dict[str, Any]]:
    """Extract bookmarks from report headings.

    Parses markdown headings and creates bookmark structure for PDF navigation.
    Creates nested bookmarks for H1, H2, H3 headings.

    Args:
        report_content: Markdown-formatted report content

    Returns:
        List of bookmark dictionaries with 'title', 'level', and 'page' keys
    """
    bookmarks: list[dict[str, Any]] = []

    try:
        # Match headings: # Title, ## Title, ### Title
        headings = re.findall(r'^(#{1,3})\s+(.+)$', report_content, re.MULTILINE)

        for level_markers, heading_text in headings:
            level = len(level_markers)

            # Limit to H1, H2, H3 only
            if level > 3:
                continue

            # Clean heading text (remove markdown formatting)
            heading_clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', heading_text)  # Remove bold
            heading_clean = re.sub(r'\*([^*]+)\*', r'\1', heading_clean)  # Remove italic
            heading_clean = heading_clean.strip()

            if not heading_clean:
                continue

            # Create bookmark entry
            # Note: Page numbers will be set during PDF building
            bookmarks.append({
                "title": heading_clean,
                "level": level,
                "page": 0,  # Will be updated during build
            })

        logger.debug(f"Extracted {len(bookmarks)} bookmarks from report")

    except Exception as e:
        logger.warning(f"Error extracting bookmarks: {e}")

    return bookmarks


def extract_keywords(report_content: str) -> str:
    """Extract keywords from report content.

    Extracts key terms and topics from the report for PDF metadata.
    Looks for common business terms and section headings.

    Args:
        report_content: Report content string

    Returns:
        Comma-separated keywords string
    """
    keywords: list[str] = []

    # Add standard keywords
    keywords.extend(["competitor", "analysis", "market", "research"])

    # Extract section headings as keywords
    headings = re.findall(r'^#{1,3}\s+(.+)$', report_content, re.MULTILINE)
    for heading in headings[:5]:  # Limit to first 5 headings
        # Clean heading text
        heading_clean = re.sub(r'[^\w\s]', '', heading).strip()
        if heading_clean and len(heading_clean) > 3:
            keywords.append(heading_clean.lower())

    # Remove duplicates and limit length
    unique_keywords = list(dict.fromkeys(keywords))[:10]  # Max 10 keywords

    return ", ".join(unique_keywords)


def get_pdf_configs(
    export_config: dict[str, Any]
) -> tuple[PDFBrandingConfig | None, PDFLayoutConfig | None]:
    """Extract PDF branding and layout configurations from export config.

    This function extracts PDFBrandingConfig and PDFLayoutConfig from the
    export configuration dictionary. If not provided, creates default
    configs to enable professional PDF features (cover page, headers, footers).

    **Configuration Sources:**

    The function looks for configuration in `export_config`:
    - `pdf_branding`: PDFBrandingConfig instance or dict
    - `pdf_layout`: PDFLayoutConfig instance or dict

    **Default Configuration:**

    If no configuration is provided, defaults are created:
    - Branding: Company name "Ashraf Yahya", default colors, footer with GitHub link
    - Layout: A4 page size, portrait orientation, 72pt margins

    **Configuration Validation:**

    Invalid configurations are logged as warnings and defaults are used instead.
    This ensures PDF generation always succeeds even with invalid configs.

    Args:
        export_config: Export configuration dictionary containing optional
            `pdf_branding` and `pdf_layout` keys

    Returns:
        Tuple of (branding_config, layout_config). Both are guaranteed to be
        non-None (defaults created if not provided).
    """
    branding_config = None
    layout_config = None

    # Extract branding config if provided
    branding_data = export_config.get("pdf_branding")
    if branding_data is not None:
        if isinstance(branding_data, PDFBrandingConfig):
            branding_config = branding_data
        elif isinstance(branding_data, dict):
            try:
                branding_config = PDFBrandingConfig(**branding_data)
            except Exception as e:
                logger.warning(
                    f"Invalid PDF branding configuration: {e}. "
                    "Using default branding."
                )

    # If no branding config provided, create default to enable features
    if branding_config is None:
        try:
            branding_config = PDFBrandingConfig(
                company_name="Ashraf Yahya",
                primary_color="#1a1a1a",
                secondary_color="#0066cc",
                accent_color="#ff6600",
                footer_text="https://github.com/ashrafyahya",
            )
            logger.debug("Using default PDF branding configuration")
        except Exception as e:
            logger.warning(f"Failed to create default branding config: {e}")

    # Extract layout config if provided
    layout_data = export_config.get("pdf_layout")
    if layout_data is not None:
        if isinstance(layout_data, PDFLayoutConfig):
            layout_config = layout_data
        elif isinstance(layout_data, dict):
            try:
                layout_config = PDFLayoutConfig(**layout_data)
            except Exception as e:
                logger.warning(
                    f"Invalid PDF layout configuration: {e}. "
                    "Using default layout."
                )

    # If no layout config provided, create default to enable features
    if layout_config is None:
        try:
            layout_config = PDFLayoutConfig(
                page_size="A4",
                orientation="portrait",
                margins={"top": 72.0, "bottom": 72.0, "left": 72.0, "right": 72.0},
            )
            logger.debug("Using default PDF layout configuration")
        except Exception as e:
            logger.warning(f"Failed to create default layout config: {e}")

    return branding_config, layout_config

