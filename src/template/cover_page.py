"""Cover page generation for PDF documents.

This module provides functionality for generating professional cover pages
with logos, titles, metadata, and responsive spacing.
"""

import logging
from datetime import datetime
from typing import Any

from src.models.pdf_branding_config import PDFBrandingConfig
from src.models.pdf_layout_config import PDFLayoutConfig

from src.template.style_utils import (
    calculate_responsive_spacing,
    get_branding_color,
    get_page_height,
)

logger = logging.getLogger(__name__)


def create_cover_page(
    branding_config: PDFBrandingConfig | None,
    layout_config: PDFLayoutConfig | None,
    report_title: str,
    report_date: datetime,
) -> list[Any]:
    """Create cover page flowables for PDF.
    
    Generates a professional cover page with:
    - Company logo (if provided) with proper scaling
    - Report title with branding colors
    - Date and metadata
    - Support for different template styles
    
    Args:
        branding_config: PDF branding configuration (optional)
        layout_config: PDF layout configuration (optional)
        report_title: Title of the report
        report_date: Date of the report
        
    Returns:
        List of ReportLab Flowable objects for the cover page
    """
    try:
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import HRFlowable, Image, PageBreak, Paragraph, Spacer
        
        styles = getSampleStyleSheet()
        flowables: list[Any] = []
        
        # Determine template style
        template_style = "default"
        if branding_config:
            template_style = branding_config.cover_page_template
        
        # Get branding colors
        primary_color = get_branding_color(branding_config, "primary")
        secondary_color = get_branding_color(branding_config, "secondary")
        
        # Create custom styles with branding colors and improved typography
        # Title style - larger, bolder, with better spacing
        title_style = styles["Title"].clone("CustomTitle")
        title_style.textColor = primary_color
        title_style.fontSize = (
            36 if template_style == "executive" else (28 if template_style == "default" else 24)
        )
        title_style.fontName = "Helvetica-Bold"
        title_style.leading = title_style.fontSize * 1.2  # 20% line height
        title_style.spaceAfter = 0.5 * inch
        title_style.alignment = 1  # Center alignment (TA_CENTER)
        
        # Subtitle style - secondary color, medium size
        subtitle_style = styles["Heading1"].clone("CustomSubtitle")
        subtitle_style.textColor = secondary_color
        subtitle_style.fontSize = 16 if template_style == "executive" else 14
        subtitle_style.fontName = "Helvetica-Bold"
        subtitle_style.leading = subtitle_style.fontSize * 1.3
        subtitle_style.spaceAfter = 0.3 * inch
        subtitle_style.alignment = 1  # Center alignment
        
        # Metadata style - smaller, regular weight
        metadata_style = styles["Normal"].clone("CustomMetadata")
        metadata_style.textColor = primary_color
        metadata_style.fontSize = 11
        metadata_style.fontName = "Helvetica"
        metadata_style.leading = metadata_style.fontSize * 1.4
        metadata_style.alignment = 1  # Center alignment
        
        # Get page height from layout config if available
        page_height = get_page_height(layout_config)
        
        # Count metadata fields for responsive spacing
        metadata_count = 1  # Date is always present
        has_logo = False
        
        # Add logo if provided
        if branding_config and branding_config.company_logo_path:
            logo_path = branding_config.company_logo_path
            if logo_path.exists():
                has_logo = True
                try:
                    # Load image to get dimensions
                    from PIL import Image as PILImage
                    
                    pil_img = PILImage.open(str(logo_path))
                    img_width, img_height = pil_img.size
                    aspect_ratio = img_width / img_height
                    
                    # Calculate dimensions with constraints
                    max_width = 4 * inch
                    max_height = 2 * inch
                    
                    # Scale maintaining aspect ratio
                    if aspect_ratio > (max_width / max_height):
                        # Width is the limiting factor
                        scaled_width = max_width
                        scaled_height = max_width / aspect_ratio
                    else:
                        # Height is the limiting factor
                        scaled_height = max_height
                        scaled_width = max_height * aspect_ratio
                    
                    # Create ReportLab Image with proper centering
                    img = Image(
                        str(logo_path),
                        width=scaled_width,
                        height=scaled_height,
                    )
                    # Center the logo horizontally
                    img.hAlign = 'CENTER'
                    flowables.append(img)
                except ImportError:
                    # PIL not available, use basic ReportLab Image
                    try:
                        img = Image(str(logo_path), width=4 * inch)
                        img.hAlign = 'CENTER'
                        flowables.append(img)
                        has_logo = True
                    except Exception as e:
                        logger.warning(f"Failed to load logo: {e}")
                except Exception as e:
                    logger.warning(f"Failed to load logo: {e}")
                    # Fall through to text-only layout
            else:
                logger.warning(f"Logo file not found: {logo_path}")
        
        # Calculate responsive spacing
        title_length = len(report_title)
        if branding_config:
            if branding_config.company_name:
                metadata_count += 1
            if (
                branding_config.footer_text
                and "github.com" in branding_config.footer_text
            ):
                metadata_count += 1
            if branding_config.document_classification:
                metadata_count += 1
        
        spacing = calculate_responsive_spacing(
            template_style=template_style,
            has_logo=has_logo,
            title_length=title_length,
            metadata_count=metadata_count,
            page_height=page_height,
        )
        
        # Apply responsive top spacing
        flowables.append(Spacer(1, spacing["top_margin"] * inch))
        
        # Add logo spacing if logo was added
        if has_logo:
            flowables.append(Spacer(1, spacing["logo_spacing"] * inch))
        
        # Add project type label before title
        if template_style != "minimal":
            project_label = "Personal project"
            project_para = Paragraph(project_label, subtitle_style)
            flowables.append(project_para)
            flowables.append(Spacer(1, 0.3 * inch))
        
        # Add visual separator before title (responsive spacing)
        if template_style == "executive":
            separator = HRFlowable(
                width="60%",
                thickness=2,
                spaceBefore=spacing["title_spacing_before"] * inch,
                spaceAfter=0.3 * inch,
                color=primary_color,
                hAlign='CENTER'
            )
            flowables.append(separator)
        elif template_style == "default":
            # Lighter separator for default template
            separator = HRFlowable(
                width="50%",
                thickness=1,
                spaceBefore=spacing["title_spacing_before"] * inch,
                spaceAfter=0.3 * inch,
                color=secondary_color,
                hAlign='CENTER'
            )
            flowables.append(separator)
        else:
            # Minimal template - just spacing, no separator
            flowables.append(
                Spacer(1, spacing["title_spacing_before"] * inch)
            )
        
        # Add report title
        title_para = Paragraph(report_title, title_style)
        flowables.append(title_para)
        # Use same spacing as before title
        flowables.append(Spacer(1, spacing["title_spacing_before"] * inch))
        
        # Add visual separator after title (same spacing as before)
        if template_style == "executive":
            separator = HRFlowable(
                width="60%",
                thickness=2,
                spaceBefore=0.3 * inch,
                spaceAfter=spacing["title_spacing_before"] * inch,
                color=primary_color,
                hAlign='CENTER'
            )
            flowables.append(separator)
        elif template_style == "default":
            # Lighter separator for default template
            separator = HRFlowable(
                width="50%",
                thickness=1,
                spaceBefore=0.3 * inch,
                spaceAfter=spacing["title_spacing_before"] * inch,
                color=secondary_color,
                hAlign='CENTER'
            )
            flowables.append(separator)
        else:
            # Minimal template - just spacing
            flowables.append(Spacer(1, spacing["title_spacing_before"] * inch))
        
        # Add author name first (no box, simple text)
        if branding_config and branding_config.company_name:
            author_name = branding_config.company_name
            author_para = Paragraph(f"<b>{author_name}</b>", metadata_style)
            flowables.append(author_para)
            flowables.append(Spacer(1, 0.2 * inch))
        
        # Add GitHub link second (no box, simple text)
        if (
            branding_config
            and branding_config.footer_text
            and "github.com" in branding_config.footer_text
        ):
            github_url = branding_config.footer_text
            github_para = Paragraph(
                f'<font color="{branding_config.accent_color}">'
                f"{github_url}</font>",
                metadata_style
            )
            flowables.append(github_para)
            flowables.append(Spacer(1, 0.2 * inch))
        
        # Add date last (no box, simple text)
        date_str = report_date.strftime("%B %d, %Y")
        date_para = Paragraph(date_str, metadata_style)
        flowables.append(date_para)
        
        # Add responsive bottom spacing before page break
        flowables.append(Spacer(1, spacing["bottom_margin"] * inch))
        
        # Add page break after cover page
        flowables.append(PageBreak())
        
        return flowables
        
    except ImportError as e:
        logger.error(f"ReportLab not available: {e}")
        return []
    except Exception as e:
        logger.error(f"Error creating cover page: {e}", exc_info=True)
        return []

