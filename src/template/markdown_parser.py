"""Markdown parser for converting markdown to PDF story elements.

This module provides functions for parsing markdown content and converting
it to ReportLab flowables for PDF generation. The parsing logic is split
into smaller functions to follow the Single Responsibility Principle.
"""

import logging
import re
from typing import Any

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, Spacer

from src.models.pdf_branding_config import PDFBrandingConfig
from src.models.pdf_layout_config import PDFLayoutConfig
from src.template.markdown_converter import (convert_markdown_to_html,
                                             has_inline_list,
                                             has_inline_numbered_list,
                                             is_numbered_list_item,
                                             split_inline_lists,
                                             split_inline_numbered_lists)
from src.template.pdf_formatter import (embed_image, format_list_items,
                                        format_numbered_list_items,
                                        format_table)

logger = logging.getLogger(__name__)

# Sections that should start on a new page
PAGE_BREAK_SECTIONS = [
    "Methodology",
    "SWOT Analysis Breakdown",
    "Competitor Overview",
    "Strategic Recommendations",
    "Market Trends",
    "Business Opportunities",
    "Sources",
]


def is_table_header(line: str) -> bool:
    """Check if line is a valid markdown table header.
    
    Requirements:
    - Contains | character
    - Has ≥2 pipes (≥2 columns)
    - Not a separator line
    
    Args:
        line: Line to check
        
    Returns:
        True if line is a valid table header
    """
    if "|" not in line or line.count("|") < 2:
        return False
    # Not a separator - separator lines contain mostly dashes, colons, spaces, and pipes
    stripped = line.strip()
    if re.match(r'^[\|\s\-:]+$', stripped):
        return False
    return True


def is_table_separator(line: str) -> bool:
    """Check if line is a valid markdown table separator.
    
    Examples: | --- | --- |, --- | ---, |:---|:---|
    
    Args:
        line: Line to check
        
    Returns:
        True if line is a valid table separator
    """
    stripped = line.strip()
    if "|" not in stripped:
        return False
    # Check if line is mostly dashes, colons, spaces, and pipes
    test_line = re.sub(r'[\|\-\s:]+', '', stripped)
    return len(test_line) < max(3, len(stripped) * 0.2)


def parse_table_cells(line: str) -> list[str]:
    """Parse table row into cells, preserving empty cells.
    
    Args:
        line: Table row line to parse
        
    Returns:
        List of cell strings (empty cells preserved)
    """
    cells = [cell.strip() for cell in line.split("|")]
    # Remove empty cells at start/end (from leading/trailing |)
    while cells and not cells[0]:
        cells.pop(0)
    while cells and not cells[-1]:
        cells.pop()
    return cells


def get_column_count(cells: list[str]) -> int:
    """Get column count from parsed cells.
    
    Args:
        cells: List of parsed cell strings
        
    Returns:
        Number of columns
    """
    return len(cells)


def _render_table_or_drop(
    table_rows: list[list[str]],
    story: list[Any],
    styles: Any,
    branding_config: PDFBrandingConfig | None,
    layout_config: PDFLayoutConfig | None,
) -> None:
    """Render table if valid, otherwise drop it and render as text.
    
    Args:
        table_rows: Table rows to render
        story: List of flowables to append to
        styles: ReportLab stylesheet
        branding_config: Optional PDF branding configuration
        layout_config: Optional PDF layout configuration
    """
    if not table_rows or len(table_rows) < 2:
        logger.warning(f"Dropping invalid table: {len(table_rows)} rows (minimum 2 required)")
        # Render as text instead
        for row in table_rows:
            text = " | ".join(row)
            story.append(Paragraph(convert_markdown_to_html(text), styles["Normal"]))
        return
    
    # Check column count consistency
    if len(table_rows) > 0:
        expected_cols = get_column_count(table_rows[0])
        for i, row in enumerate(table_rows[1:], start=1):
            if get_column_count(row) != expected_cols:
                logger.warning(
                    f"Dropping table: row {i+1} has {get_column_count(row)} columns, "
                    f"expected {expected_cols}"
                )
                # Render as text instead
                for row_text in table_rows:
                    text = " | ".join(row_text)
                    story.append(Paragraph(convert_markdown_to_html(text), styles["Normal"]))
                return
    
    # Table is valid, render it
    story.extend(format_table(table_rows, styles, branding_config, layout_config))


def _handle_empty_line(
    story: list[Any],
    in_list: str | bool,
    list_items: list[str],
    in_table: bool,
    table_rows: list[list[str]],
    styles: Any,
    branding_config: PDFBrandingConfig | None = None,
    layout_config: PDFLayoutConfig | None = None,
) -> tuple[str | bool, list[str], bool, list[list[str]]]:
    """Handle empty line in markdown parsing.

    Args:
        story: List of flowables to append to
        in_list: Whether currently in a list
        list_items: Current list items
        in_table: Whether currently in a table
        table_rows: Current table rows
        styles: ReportLab stylesheet
        branding_config: Optional PDF branding configuration
        layout_config: Optional PDF layout configuration

    Returns:
        Tuple of (in_list, list_items, in_table, table_rows) updated state
    """
    if in_list and list_items:
        # Check list type and use appropriate formatter
        if in_list == "numbered":
            story.extend(format_numbered_list_items(list_items, styles))
        else:
            story.extend(format_list_items(list_items, styles))
        list_items = []
        in_list = False
    if in_table and table_rows:
        _render_table_or_drop(table_rows, story, styles, branding_config, layout_config)
        table_rows = []
        in_table = False
    # Add spacing after empty line to separate paragraphs
    story.append(Spacer(1, 0.15 * inch))
    return in_list, list_items, in_table, table_rows


def _handle_horizontal_rule(
    story: list[Any],
    in_list: str | bool,
    list_items: list[str],
    in_table: bool,
    table_rows: list[list[str]],
    styles: Any,
    branding_config: PDFBrandingConfig | None = None,
    layout_config: PDFLayoutConfig | None = None,
) -> tuple[str | bool, list[str], bool, list[list[str]]]:
    """Handle horizontal rule (---) in markdown.

    Args:
        story: List of flowables to append to
        in_list: Whether currently in a list
        list_items: Current list items
        in_table: Whether currently in a table
        table_rows: Current table rows
        styles: ReportLab stylesheet
        branding_config: Optional PDF branding configuration
        layout_config: Optional PDF layout configuration

    Returns:
        Tuple of (in_list, list_items, in_table, table_rows) updated state
    """
    if in_list and list_items:
        # Check list type and use appropriate formatter
        if in_list == "numbered":
            story.extend(format_numbered_list_items(list_items, styles))
        else:
            story.extend(format_list_items(list_items, styles))
        list_items = []
        in_list = False
    if in_table and table_rows:
        _render_table_or_drop(table_rows, story, styles, branding_config, layout_config)
        table_rows = []
        in_table = False
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(
        width="100%",
        thickness=1,
        spaceBefore=0.1 * inch,
        spaceAfter=0.1 * inch,
        color=(0.5, 0.5, 0.5)
    ))
    story.append(Spacer(1, 0.3 * inch))
    return in_list, list_items, in_table, table_rows


def _handle_heading(
    line_stripped: str,
    story: list[Any],
    in_list: str | bool,
    list_items: list[str],
    styles: Any,
) -> tuple[str | bool, list[str], bool]:
    """Handle markdown heading (H1, H2, H3).

    Args:
        line_stripped: Stripped line to process
        story: List of flowables to append to
        in_list: Whether currently in a list
        list_items: Current list items
        styles: ReportLab stylesheet

    Returns:
        Tuple of (in_list, list_items, should_continue)
    """
    # Check if this is actually a heading FIRST before flushing lists
    # Only flush lists if we're processing a real heading
    is_heading = (
        line_stripped.startswith("# ") or
        line_stripped.startswith("## ") or
        line_stripped.startswith("### ")
    )
    
    if is_heading and in_list and list_items:
        # Check list type and use appropriate formatter
        if in_list == "numbered":
            story.extend(format_numbered_list_items(list_items, styles))
        else:
            story.extend(format_list_items(list_items, styles))
        list_items = []
        in_list = False

    if line_stripped.startswith("# "):
        heading_text = line_stripped[2:].strip()
        # Only take the first line in case content leaked in
        heading_text = heading_text.split('\n')[0].strip()
        heading_clean = _clean_heading_text(heading_text)
        
        # Reduce spacing before "Executive Summary" heading
        if heading_clean.strip().lower() == "executive summary" and story and isinstance(story[-1], Spacer):
            story.pop()
        
        if heading_clean in PAGE_BREAK_SECTIONS:
            story.append(PageBreak())
        story.append(Paragraph(convert_markdown_to_html(heading_text), styles["Title"]))
        story.append(Spacer(1, 0.4 * inch))
        return in_list, list_items, True
    elif line_stripped.startswith("## "):
        heading_text = line_stripped[3:].strip()
        # Only take the first line in case content leaked in
        heading_text = heading_text.split('\n')[0].strip()
        heading_clean = _clean_heading_text(heading_text)
        
        # Check if this is a SWOT category heading - format as normal text instead
        # Use cleaned heading text for comparison to handle markdown formatting
        swot_keywords = ['Strengths', 'Weaknesses', 'Opportunities', 'Threats']
        heading_clean_lower = heading_clean.strip().lower()
        # Check for exact match or if heading starts with SWOT keyword (handles cases like "Strengths:" or "Strengths -")
        is_swot_category = any(
            keyword.lower() == heading_clean_lower or 
            heading_clean_lower.startswith(keyword.lower() + ' ') or
            heading_clean_lower.startswith(keyword.lower() + ':') or
            heading_clean_lower.startswith(keyword.lower() + '-')
            for keyword in swot_keywords
        )
        
        if is_swot_category:
            # Format as normal text with bold styling (same size and color as normal paragraphs)
            # Use cleaned heading text to avoid double-bold formatting
            heading_formatted = convert_markdown_to_html(heading_clean)
            story.append(Paragraph(f"<b>{heading_formatted}</b>", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))
            return in_list, list_items, True
        
        # Reduce spacing before "Executive Summary" heading
        if heading_clean.strip().lower() == "executive summary" and story and isinstance(story[-1], Spacer):
            story.pop()
        
        if heading_clean in PAGE_BREAK_SECTIONS:
            story.append(PageBreak())
        story.append(Paragraph(convert_markdown_to_html(heading_text), styles["Heading1"]))
        story.append(Spacer(1, 0.25 * inch))
        return in_list, list_items, True
    elif line_stripped.startswith("### "):
        heading_text = line_stripped[4:].strip()
        # Only take the first line in case content leaked in
        heading_text = heading_text.split('\n')[0].strip()
        heading_clean = _clean_heading_text(heading_text)
        
        # Check if this is a SWOT category heading - format as normal text instead
        # Use cleaned heading text for comparison to handle markdown formatting
        swot_keywords = ['Strengths', 'Weaknesses', 'Opportunities', 'Threats']
        heading_clean_lower = heading_clean.strip().lower()
        # Check for exact match or if heading starts with SWOT keyword (handles cases like "Strengths:" or "Strengths -")
        is_swot_category = any(
            keyword.lower() == heading_clean_lower or 
            heading_clean_lower.startswith(keyword.lower() + ' ') or
            heading_clean_lower.startswith(keyword.lower() + ':') or
            heading_clean_lower.startswith(keyword.lower() + '-')
            for keyword in swot_keywords
        )
        
        if is_swot_category:
            # Check if there's additional content on this line after the heading
            # If the original line is longer than just the heading, treat as paragraph
            # This handles cases like "### Strengths * item1 * item2 ### Weaknesses..."
            original_after_heading = line_stripped[4:].strip()
            heading_word_length = len(heading_clean.split()[0]) if heading_clean.split() else 0
            # Check if there's content after the heading word (more than just whitespace)
            has_content_after_heading = (
                len(original_after_heading) > heading_word_length + 5 or  # Allow some buffer
                '*' in original_after_heading[heading_word_length:] or  # Has bullet points
                '###' in original_after_heading[heading_word_length:]    # Has more SWOT categories
            )
            
            if has_content_after_heading:
                # This line contains SWOT content, let it be processed as a paragraph
                # so _detect_and_split_swot_in_paragraph() can handle it properly
                return in_list, list_items, False
            
            # Format as normal text with bold styling (same size and color as normal paragraphs)
            # Use cleaned heading text to avoid double-bold formatting
            heading_formatted = convert_markdown_to_html(heading_clean)
            story.append(Paragraph(f"<b>{heading_formatted}</b>", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))
            return in_list, list_items, True
        
        # Check if this is a Priority heading
        priority_keywords = ['Priority', 'priority']
        is_priority = any(
            heading_clean_lower.startswith(keyword.lower()) 
            for keyword in priority_keywords
        )
        
        if is_priority:
            # Check if there's additional content on this line after the heading
            # This handles cases like "### Priority 1: Title * item1 * item2 ### Priority 2..."
            original_after_heading = line_stripped[4:].strip()
            heading_word_length = len(heading_clean.split()[0]) if heading_clean.split() else 0
            # Check if there's content after the heading word (more than just whitespace)
            has_content_after_heading = (
                len(original_after_heading) > heading_word_length + 5 or  # Allow some buffer
                '*' in original_after_heading[heading_word_length:] or  # Has bullet points
                '###' in original_after_heading[heading_word_length:] or  # Has more Priority entries
                ':' in original_after_heading[heading_word_length:]     # Has colon separator
            )
            
            if has_content_after_heading:
                # This line contains Priority content, let it be processed as a paragraph
                # so _detect_and_split_priority_in_paragraph() can handle it properly
                return in_list, list_items, False
            
            # Format as normal text with bold styling (same size and color as normal paragraphs)
            # Use cleaned heading text to avoid double-bold formatting
            heading_formatted = convert_markdown_to_html(heading_clean)
            story.append(Paragraph(f"<b>{heading_formatted}</b>", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))
            return in_list, list_items, True
        
        # Check if this might be a competitor heading (not SWOT, not Priority)
        # Format competitor headings the same way as SWOT categories (bold normal text)
        # If it has content on the same line, treat as paragraph for competitor formatting
        if not is_priority:
            # Check if there's additional content on this line after the heading
            # This handles cases like "### Asana * Market leader... * Strong integrations..."
            original_after_heading = line_stripped[4:].strip()
            heading_word_length = len(heading_clean.split()[0]) if heading_clean.split() else 0
            # Check if there's content after the heading word (more than just whitespace)
            has_content_after_heading = (
                len(original_after_heading) > heading_word_length + 5 or  # Allow some buffer
                '*' in original_after_heading[heading_word_length:] or  # Has bullet points
                '###' in original_after_heading[heading_word_length:]    # Has more competitor entries
            )
            
            if has_content_after_heading:
                # This line contains competitor content, let it be processed as a paragraph
                # so _detect_and_split_competitor_in_paragraph() can handle it properly
                return in_list, list_items, False
            
            # Format standalone competitor headings as bold normal text (like SWOT categories)
            # This ensures consistent formatting with SWOT Analysis Breakdown section
            # Extract competitor name if format is "Competitor X: Name"
            competitor_heading = heading_clean
            competitor_pattern = r'^Competitor\s+\d+\s*:\s*(.+)$'
            competitor_match = re.match(competitor_pattern, heading_clean, re.IGNORECASE)
            if competitor_match:
                competitor_heading = competitor_match.group(1).strip()
            
            heading_formatted = convert_markdown_to_html(competitor_heading)
            story.append(Paragraph(f"<b>{heading_formatted}</b>", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))
            return in_list, list_items, True
        
        if heading_clean in PAGE_BREAK_SECTIONS:
            story.append(PageBreak())
        story.append(Paragraph(convert_markdown_to_html(heading_text), styles["Heading2"]))
        story.append(Spacer(1, 0.18 * inch))
        return in_list, list_items, True

    return in_list, list_items, False


def _clean_heading_text(heading_text: str) -> str:
    """Clean heading text by removing markdown formatting.

    Args:
        heading_text: Heading text to clean

    Returns:
        Cleaned heading text
    """
    # Remove markdown headers (#, ##, ###, etc.)
    heading_clean = re.sub(r'^#{1,6}\s+', '', heading_text)
    # Remove bold/italic
    heading_clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', heading_clean)
    heading_clean = re.sub(r'\*([^*]+)\*', r'\1', heading_clean)
    return heading_clean.strip()


def _handle_list_item(
    line_stripped: str,
    in_list: str | bool,
    list_items: list[str],
) -> tuple[str | bool, list[str], bool]:
    """Handle markdown list item.

    Args:
        line_stripped: Stripped line to process
        in_list: List type - False/None for no list, "bullet" for bullet list, "numbered" for numbered list
        list_items: Current list items

    Returns:
        Tuple of (in_list, list_items, should_continue)
        in_list is now a string: "bullet", "numbered", or False
    """
    
    # Handle unordered lists (-, *, •)
    if line_stripped.startswith(("- ", "* ", "• ")):
        in_list = "bullet"
        list_item = line_stripped[2:].strip()
        # Remove trailing "-" if present (common issue from LLM formatting)
        # Pattern: "Item text [1] -" or "Item text -"
        while list_item.rstrip().endswith(' -'):
            list_item = list_item.rstrip()[:-2].rstrip()
        if list_item.rstrip().endswith('-') and len(list_item.rstrip()) > 1:
            # Only remove if preceded by space (not part of citation like "[1]-")
            if list_item.rstrip()[-2] == ' ':
                list_item = list_item.rstrip()[:-1].rstrip()
        list_item = convert_markdown_to_html(list_item)
        if list_item:
            list_items.append(list_item)
        return in_list, list_items, True

    # Handle numbered lists
    if is_numbered_list_item(line_stripped):
        in_list = "numbered"
        parts = line_stripped.split(".", 1)
        if len(parts) > 1:
            list_item = parts[1].strip()
            list_item = convert_markdown_to_html(list_item)
            if list_item:
                list_items.append(list_item)
        return in_list, list_items, True

    return in_list, list_items, False


def _detect_and_split_swot_in_paragraph(text: str) -> list[tuple[str, str]] | None:
    """Detect and split SWOT content within paragraph text.
    
    Detects patterns like "Strengths ... ### Weaknesses ..." or 
    "### Strengths ... ### Weaknesses ..." and splits them.
    
    Args:
        text: Paragraph text to analyze
        
    Returns:
        List of (category, content) tuples if SWOT content detected, None otherwise
    """
    swot_keywords = ['Strengths', 'Weaknesses', 'Opportunities', 'Threats']
    
    # Pattern to match SWOT category markers (with or without ### prefix)
    # Matches: "Strengths ", "### Strengths ", "###Weaknesses ", etc.
    # Use word boundary or start of string to catch keywords at start
    pattern = r'(?:^|###\s*)(Strengths|Weaknesses|Opportunities|Threats)(?:\s+|$)'
    
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    
    if len(matches) < 2:  # Need at least 2 categories to be considered SWOT content
        return None
    
    # Split text at SWOT category markers
    sections = []
    for i, match in enumerate(matches):
        category = match.group(1)
        # Capitalize first letter for consistency
        category = category[0].upper() + category[1:].lower()
        
        # Get content for this category (from this marker to next marker or end)
        # match.end() points to the end of the match (after "Strengths " or "### Strengths ")
        start_pos = match.end()
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
            content = text[start_pos:end_pos].strip()
        else:
            content = text[start_pos:].strip()
        
        # Clean any remaining markdown headers from content
        # Remove ###, ##, # markers that might appear in content
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        # Also remove any stray category names that might have leaked into content
        content = re.sub(r'^\s*(Strengths|Weaknesses|Opportunities|Threats)\s+', '', content, flags=re.IGNORECASE)
        content = content.strip()
        
        if content:
            sections.append((category, content))
    
    return sections if len(sections) >= 2 else None


def _detect_and_split_competitor_in_paragraph(text: str) -> list[tuple[str, str]] | None:
    """Detect and split competitor content within paragraph text.
    
    Detects patterns like "### Asana * details... ### Trello * details..." 
    and splits them by competitor names.
    
    Args:
        text: Paragraph text to analyze
        
    Returns:
        List of (competitor_name, content) tuples if competitor content detected, None otherwise
    """
    # Pattern to match competitor markers: ### CompetitorName or ### Competitor X: Name
    # Competitor names are typically capitalized words (not SWOT keywords, not Priority)
    swot_keywords = ['Strengths', 'Weaknesses', 'Opportunities', 'Threats']
    priority_keywords = ['Priority', 'priority']
    
    # Pattern: ### followed by a capitalized word/phrase (competitor name)
    # Handles formats like:
    # - ### Competitor 1: Engagebay
    # - ### Engagebay
    # - ### Competitor 5: PCMag
    # Match until we hit *, ###, or end of string
    pattern = r'###\s+([A-Z][a-zA-Z0-9\s&.:-]+?)(?:\s+\*|\s+###|$)'
    
    matches = list(re.finditer(pattern, text))
    
    if len(matches) < 1:  # Need at least 1 competitor to be considered competitor content
        return None
    
    # Filter out SWOT and Priority keywords and extract competitor name
    valid_matches = []
    for match in matches:
        competitor_name = match.group(1).strip()
        competitor_lower = competitor_name.lower()
        
        # Extract actual competitor name if format is "Competitor X: Name"
        # Pattern: "Competitor" followed by number and colon, then the actual name
        competitor_pattern = r'^Competitor\s+\d+\s*:\s*(.+)$'
        competitor_match = re.match(competitor_pattern, competitor_name, re.IGNORECASE)
        if competitor_match:
            competitor_name = competitor_match.group(1).strip()
            competitor_lower = competitor_name.lower()
        
        # Skip if it's a SWOT keyword
        is_swot = any(
            keyword.lower() == competitor_lower or
            competitor_lower.startswith(keyword.lower() + ' ')
            for keyword in swot_keywords
        )
        
        # Skip if it's a Priority keyword
        is_priority = any(
            competitor_lower.startswith(keyword.lower())
            for keyword in priority_keywords
        )
        
        if not is_swot and not is_priority:
            # Create a new match with the cleaned competitor name
            # We'll use the original match but store the cleaned name separately
            valid_matches.append((match, competitor_name))
    
    if len(valid_matches) < 1:
        return None
    
    # Split text at competitor markers
    sections = []
    for i, (match, competitor_name) in enumerate(valid_matches):
        # Get content for this competitor (from this marker to next marker or end)
        # match.end() points to the end of the match (after "### CompetitorName " or "### Competitor X: Name ")
        start_pos = match.end()
        if i + 1 < len(valid_matches):
            end_pos = valid_matches[i + 1][0].start()  # valid_matches[i+1] is a tuple (match, name)
            content = text[start_pos:end_pos].strip()
        else:
            content = text[start_pos:].strip()
        
        # Clean any remaining markdown headers from content
        # Remove ###, ##, # markers that might appear in content
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        # Also remove any stray competitor names that might have leaked into content
        # Remove both "Competitor X: Name" format and just "Name" format
        content = re.sub(r'^\s*Competitor\s+\d+\s*:\s*' + re.escape(competitor_name) + r'\s+', '', content, flags=re.IGNORECASE)
        content = re.sub(r'^\s*' + re.escape(competitor_name) + r'\s+', '', content, flags=re.IGNORECASE)
        content = content.strip()
        
        if content:
            sections.append((competitor_name, content))
    
    return sections if len(sections) >= 1 else None


def _detect_and_split_priority_in_paragraph(text: str) -> list[tuple[str, str]] | None:
    """Detect and split Priority recommendations within paragraph text.
    
    Detects patterns like "### Priority 1: Title * details... ### Priority 2: Title * details..." 
    and splits them by Priority headings.
    
    Args:
        text: Paragraph text to analyze
        
    Returns:
        List of (priority_title, content) tuples if Priority content detected, None otherwise
    """
    # Pattern to match Priority markers: ### Priority 1: Title or ### Priority 1 Title
    # Matches: "### Priority 1:", "### Priority 1: Title", "### Priority 2 Title", etc.
    # The pattern captures everything from "Priority" to the next marker (* or ###) or end
    pattern = r'###\s+(Priority\s+\d+[:\s]*[^\*#]*?)(?:\s+\*|\s+###|$)'
    
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    
    if len(matches) < 1:  # Need at least 1 Priority to be considered Priority content
        return None
    
    # Split text at Priority markers
    sections = []
    for i, match in enumerate(matches):
        priority_title = match.group(1).strip()
        # Clean the title (remove extra spaces, normalize)
        priority_title = re.sub(r'\s+', ' ', priority_title)
        # Remove trailing colon if present
        priority_title = priority_title.rstrip(':').strip()
        
        # Get content for this Priority (from this marker to next marker or end)
        # match.end() points to the end of the match (after "### Priority 1: Title ")
        start_pos = match.end()
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
            content = text[start_pos:end_pos].strip()
        else:
            content = text[start_pos:].strip()
        
        # Clean any remaining markdown headers from content
        # Remove ###, ##, # markers that might appear in content
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        # Also remove any stray Priority titles that might have leaked into content
        # Match "Priority 1:" or "Priority 1" at the start of content
        content = re.sub(r'^\s*Priority\s+\d+[:\s]+', '', content, flags=re.IGNORECASE)
        content = content.strip()
        
        if content:
            sections.append((priority_title, content))
    
    return sections if len(sections) >= 1 else None


def _handle_paragraph(
    line_stripped: str,
    story: list[Any],
    in_list: str | bool,
    list_items: list[str],
    styles: Any,
) -> tuple[str | bool, list[str]]:
    """Handle regular paragraph or inline lists.

    Args:
        line_stripped: Stripped line to process
        story: List of flowables to append to
        in_list: Whether currently in a list
        list_items: Current list items
        styles: ReportLab stylesheet

    Returns:
        Tuple of (in_list, list_items) updated state
    """
    # Regular paragraph - check for inline lists
    if in_list and list_items:
        # Check list type and use appropriate formatter
        if in_list == "numbered":
            story.extend(format_numbered_list_items(list_items, styles))
        else:
            story.extend(format_list_items(list_items, styles))
        list_items = []
        in_list = False

    # Check if paragraph contains inline lists
    para_text = line_stripped
    
    # Pre-process: strip any markdown headers that might appear in paragraph text
    # This prevents literal ### from appearing in PDF output
    # Note: We do this before SWOT detection, but SWOT detection handles ### markers correctly
    # IMPORTANT: Check for Priority content BEFORE removing ### markers, as Priority detection
    # relies on ### markers to identify Priority sections
    priority_sections_raw = _detect_and_split_priority_in_paragraph(para_text)
    
    para_text = re.sub(r'^#{1,6}\s+', '', para_text, flags=re.MULTILINE)
    para_text = para_text.strip()
    
    # First, check if paragraph contains embedded SWOT content
    swot_sections = _detect_and_split_swot_in_paragraph(para_text)
    if swot_sections:
        # Format SWOT content with category headers and bullet points
        for category, content in swot_sections:
            # Add category header (bold, normal size)
            story.append(Paragraph(f"<b>{category}</b>", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))
            
            # Split content into items
            # First, try splitting by bullet point markers (*)
            # Pattern: "* item" or " * item" (with optional leading space)
            if '*' in content:
                # Split on * markers (but keep the * with the content for now)
                parts = re.split(r'\s*\*\s+', content)
                # Filter out empty parts and clean them
                items = []
                for part in parts:
                    part = part.strip()
                    # Skip empty parts and parts that are just whitespace
                    if part and len(part) > 5:
                        items.append(part)
                # If we got multiple items from * splitting, use those
                if len(items) > 1:
                    pass  # Use items as-is
                else:
                    # Fall back to other splitting methods
                    items = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)
            else:
                # No * markers, try splitting by sentence boundaries
                items = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)
            
            # If that didn't work well (single long item), split by capital letters
            # Pattern: Items start with capital letters (after space or at start)
            # Example: "Market leader... Strong integrations... Unlimited users..."
            if len(items) == 1 and len(content) > 100:
                # Split on: space + Capital letter (where it's likely start of new item)
                # But keep the capital letter with the item
                parts = re.split(r'\s+(?=[A-Z][a-z])', content)
                
                # Filter: keep items that are substantial (at least 15 chars)
                # and start with capital letter
                items = []
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 15 and part[0].isupper():
                        items.append(part)
                
                # If we didn't get good splits, try a different approach
                if len(items) <= 1:
                    # Try splitting on capital letters that are followed by lowercase
                    # and preceded by space or start of string
                    items = re.findall(r'(?:^|\s)([A-Z][a-z]+(?:\s+[a-z]+){2,}[^A-Z]*)', content)
                    items = [item.strip() for item in items if len(item.strip()) > 15]
            
            # Format each item as a bullet point
            for item in items:
                item = item.strip()
                if item and len(item) > 5:  # Minimum length check
                    # Remove trailing "-" if present (common issue from LLM formatting)
                    original_item = item
                    while item.rstrip().endswith(' -'):
                        item = item.rstrip()[:-2].rstrip()
                    if item.rstrip().endswith('-') and len(item.rstrip()) > 1:
                        # Only remove if preceded by space (not part of citation like "[1]-")
                        if item.rstrip()[-2] == ' ':
                            item = item.rstrip()[:-1].rstrip()
                    item_html = convert_markdown_to_html(item)
                    story.append(Paragraph(f"• {item_html}", styles["Normal"]))
                    story.append(Spacer(1, 0.08 * inch))
            
            # Add spacing after category
            story.append(Spacer(1, 0.15 * inch))
        
        return in_list, list_items
    
    # Check if paragraph contains embedded Priority recommendations
    # Use the result from before preprocessing (to preserve ### markers)
    # If not found before preprocessing, try again after (for edge cases)
    priority_sections = priority_sections_raw if priority_sections_raw else _detect_and_split_priority_in_paragraph(para_text)
    if priority_sections:
        # Format Priority content with Priority titles as headers and bullet points
        for priority_title, content in priority_sections:
            # Add Priority title header (bold, normal size)
            priority_formatted = convert_markdown_to_html(priority_title)
            story.append(Paragraph(f"<b>{priority_formatted}</b>", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))
            
            # Split content into items
            # First, try splitting by bullet point markers (*)
            if '*' in content:
                # Split on * markers
                parts = re.split(r'\s*\*\s+', content)
                # Filter out empty parts and clean them
                items = []
                for part in parts:
                    part = part.strip()
                    # Skip empty parts and parts that are just whitespace
                    if part and len(part) > 5:
                        items.append(part)
                # If we got multiple items from * splitting, use those
                if len(items) > 1:
                    pass  # Use items as-is
                else:
                    # Fall back to other splitting methods
                    items = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)
            else:
                # No * markers, try splitting by sentence boundaries
                items = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)
            
            # If that didn't work well (single long item), split by capital letters
            if len(items) == 1 and len(content) > 100:
                # Split on: space + Capital letter (where it's likely start of new item)
                parts = re.split(r'\s+(?=[A-Z][a-z])', content)
                
                # Filter: keep items that are substantial (at least 15 chars)
                # and start with capital letter
                items = []
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 15 and part[0].isupper():
                        items.append(part)
                
                # If we didn't get good splits, try a different approach
                if len(items) <= 1:
                    # Try splitting on capital letters that are followed by lowercase
                    items = re.findall(r'(?:^|\s)([A-Z][a-z]+(?:\s+[a-z]+){2,}[^A-Z]*)', content)
                    items = [item.strip() for item in items if len(item.strip()) > 15]
            
            # Format each item as a bullet point
            for item in items:
                item = item.strip()
                if item and len(item) > 5:  # Minimum length check
                    item_html = convert_markdown_to_html(item)
                    story.append(Paragraph(f"• {item_html}", styles["Normal"]))
                    story.append(Spacer(1, 0.08 * inch))
            
            # Add spacing after Priority
            story.append(Spacer(1, 0.15 * inch))
        
        return in_list, list_items
    
    # Check if paragraph contains embedded competitor content
    competitor_sections = _detect_and_split_competitor_in_paragraph(para_text)
    if competitor_sections:
        # Format competitor content with competitor names as headers and bullet points
        for competitor_name, content in competitor_sections:
            # Add competitor name header (bold, normal size)
            competitor_formatted = convert_markdown_to_html(competitor_name)
            story.append(Paragraph(f"<b>{competitor_formatted}</b>", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))
            
            # Split content into items
            # First, try splitting by bullet point markers (*)
            if '*' in content:
                # Split on * markers
                parts = re.split(r'\s*\*\s+', content)
                # Filter out empty parts and clean them
                items = []
                for part in parts:
                    part = part.strip()
                    # Skip empty parts and parts that are just whitespace
                    if part and len(part) > 5:
                        items.append(part)
                # If we got multiple items from * splitting, use those
                if len(items) > 1:
                    pass  # Use items as-is
                else:
                    # Fall back to other splitting methods
                    items = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)
            else:
                # No * markers, try splitting by sentence boundaries
                items = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)
            
            # If that didn't work well (single long item), split by capital letters
            if len(items) == 1 and len(content) > 100:
                # Split on: space + Capital letter (where it's likely start of new item)
                parts = re.split(r'\s+(?=[A-Z][a-z])', content)
                
                # Filter: keep items that are substantial (at least 15 chars)
                # and start with capital letter
                items = []
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 15 and part[0].isupper():
                        items.append(part)
                
                # If we didn't get good splits, try a different approach
                if len(items) <= 1:
                    # Try splitting on capital letters that are followed by lowercase
                    items = re.findall(r'(?:^|\s)([A-Z][a-z]+(?:\s+[a-z]+){2,}[^A-Z]*)', content)
                    items = [item.strip() for item in items if len(item.strip()) > 15]
            
            # Format each item as a bullet point
            for item in items:
                item = item.strip()
                if item and len(item) > 5:  # Minimum length check
                    item_html = convert_markdown_to_html(item)
                    story.append(Paragraph(f"• {item_html}", styles["Normal"]))
                    story.append(Spacer(1, 0.08 * inch))
            
            # Add spacing after competitor
            story.append(Spacer(1, 0.15 * inch))
        
        return in_list, list_items
    
    # Split long paragraphs that might contain multiple sentences on same line
    # Look for patterns like "sentence. sentence" or "item. item" and split them
    # But be careful not to split on abbreviations or decimals
    if len(para_text) > 150 and '. ' in para_text:
        # Check if it looks like multiple sentences (has period followed by space and capital)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para_text)
        if len(sentences) > 1:
            # Multiple sentences detected - process each separately
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if has_inline_list(sentence):
                    parts = split_inline_lists(sentence)
                    for part in parts:
                        if isinstance(part, list):
                            story.extend(format_list_items(part, styles))
                        else:
                            if part.strip():
                                story.append(Paragraph(convert_markdown_to_html(part), styles["Normal"]))
                                story.append(Spacer(1, 0.12 * inch))
                else:
                    para_html = convert_markdown_to_html(sentence)
                    if para_html.strip():
                        story.append(Paragraph(para_html, styles["Normal"]))
                        story.append(Spacer(1, 0.12 * inch))
            return in_list, list_items
    
    if has_inline_list(para_text):
        parts = split_inline_lists(para_text)
        for part in parts:
            if isinstance(part, list):
                story.extend(format_list_items(part, styles))
            else:
                if part.strip():
                    story.append(Paragraph(convert_markdown_to_html(part), styles["Normal"]))
                    story.append(Spacer(1, 0.12 * inch))
    elif has_inline_numbered_list(para_text):
        parts = split_inline_numbered_lists(para_text)
        for part in parts:
            if isinstance(part, list):
                story.extend(format_numbered_list_items(part, styles))
            else:
                if part.strip():
                    story.append(Paragraph(convert_markdown_to_html(part), styles["Normal"]))
                    story.append(Spacer(1, 0.12 * inch))
    else:
        # Regular paragraph
        para_text = convert_markdown_to_html(para_text)
        if para_text.strip():
            story.append(Paragraph(para_text, styles["Normal"]))
            story.append(Spacer(1, 0.12 * inch))

    return in_list, list_items


def parse_markdown_to_story(
    report: str, 
    styles: Any, 
    branding_config: PDFBrandingConfig | None = None,
    layout_config: PDFLayoutConfig | None = None
) -> list[Any]:
    """Parse markdown report and convert to PDF story flowables.

    This function parses markdown content and converts it to ReportLab
    flowables for PDF generation. It handles headings, lists, tables,
    paragraphs, and inline formatting.

    Args:
        report: Markdown-formatted report string
        styles: ReportLab stylesheet
        branding_config: Optional PDF branding configuration
        layout_config: Optional PDF layout configuration

    Returns:
        List of ReportLab flowables ready for PDF generation
    """
    lines = report.split("\n")
    story: list[Any] = []
    in_list = False
    list_items: list[str] = []
    in_table = False
    table_rows: list[list[str]] = []
    expected_column_count: int | None = None
    table_separator_seen = False

    i = 0
    while i < len(lines):
        line_stripped = lines[i].strip()

        # Handle empty lines - end table if in one
        if not line_stripped:
            if in_table and table_rows:
                _render_table_or_drop(table_rows, story, styles, branding_config, layout_config)
                table_rows = []
                expected_column_count = None
                table_separator_seen = False
            in_list, list_items, in_table, table_rows = _handle_empty_line(
                story, in_list, list_items, False, [], styles, branding_config, layout_config
            )
            in_table = False
            i += 1
            continue

        # STRICT TABLE DETECTION: Only detect tables with header + separator + consistent columns
        # Check if current line could be a table header
        # Handle case where table header appears at end of line with prose
        if not in_table and not line_stripped.startswith("#") and "|" in line_stripped:
            # Check if line ends with a table header pattern (multiple pipes at the end)
            # Pattern: prose text | col1 | col2 | col3 |
            # Strategy: Find the last occurrence of a pipe pattern that looks like a table header
            pipe_positions = [pos for pos, char in enumerate(line_stripped) if char == "|"]
            if len(pipe_positions) >= 2:
                # Try to find where table starts by looking for pattern: | col | col |
                # Start from the end and work backwards to find the longest valid table header
                best_match = None
                best_match_start = None
                
                for start_idx in range(len(pipe_positions) - 2, -1, -1):
                    # Need at least 2 pipes for a table (2 columns)
                    if len(pipe_positions) - start_idx >= 2:
                        table_start_pos = pipe_positions[start_idx]
                        # Extract potential table part
                        table_part = line_stripped[table_start_pos:].strip()
                        
                        # Check if this looks like a table header
                        if is_table_header(table_part):
                            # Check if there's prose before this
                            prose_part = line_stripped[:table_start_pos].rstrip()
                            if prose_part and not prose_part.endswith("|"):
                                # Found prose before table - this is a candidate
                                # Prefer longer table headers (more columns)
                                if best_match is None or len(table_part) > len(best_match):
                                    best_match = table_part
                                    best_match_start = table_start_pos
                
                # If we found a match, process it
                if best_match is not None and best_match_start is not None:
                    prose_part = line_stripped[:best_match_start].rstrip()
                    table_part = best_match
                    
                    # Process prose part first
                    if prose_part.strip():
                        in_list, list_items = _handle_paragraph(
                            prose_part, story, in_list, list_items, styles
                        )
                    
                    # Check if next line is separator
                    next_line_idx = i + 1
                    while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                        next_line_idx += 1
                    
                    if next_line_idx < len(lines):
                        next_line = lines[next_line_idx].strip()
                        if is_table_separator(next_line):
                            # Valid table start
                            cells = parse_table_cells(table_part)
                            expected_column_count = get_column_count(cells)
                            logger.debug(
                                f"Table detected (with prose): header has {expected_column_count} columns, "
                                f"table_part='{table_part[:80]}...'"
                            )
                            table_rows.append(cells)
                            in_table = True
                            table_separator_seen = False
                            i += 1
                            continue
                    # If no separator found, fall through to process as normal text
            
            # Standard check: whole line is a table header
            if is_table_header(line_stripped):
                # Look ahead to next non-empty line to check for separator
                next_line_idx = i + 1
                while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                    next_line_idx += 1
                
                if next_line_idx < len(lines):
                    next_line = lines[next_line_idx].strip()
                    if is_table_separator(next_line):
                        # Valid table start: header + separator found
                        cells = parse_table_cells(line_stripped)
                        expected_column_count = get_column_count(cells)
                        logger.debug(
                            f"Table detected: header has {expected_column_count} columns, "
                            f"header='{line_stripped[:50]}...'"
                        )
                        table_rows.append(cells)
                        in_table = True
                        table_separator_seen = False  # Will be set to True when we process separator
                        i += 1
                        continue
        
        # If we're in a table, process table rows
        if in_table:
            # Check if it's the separator line
            if is_table_separator(line_stripped):
                if not table_separator_seen:
                    table_separator_seen = True
                    i += 1
                    continue
                else:
                    # Second separator? End table
                    _render_table_or_drop(table_rows, story, styles, branding_config, layout_config)
                    table_rows = []
                    expected_column_count = None
                    table_separator_seen = False
                    in_table = False
                    # Fall through to process this line as non-table
            
            # Check if line is a valid table row with correct column count
            elif "|" in line_stripped:
                cells = parse_table_cells(line_stripped)
                col_count = get_column_count(cells)
                
                if expected_column_count is not None and col_count == expected_column_count:
                    # Valid table row
                    table_rows.append(cells)
                    i += 1
                    continue
                else:
                    # Column count mismatch - end table and render as text
                    logger.warning(
                        f"Table column mismatch: expected {expected_column_count}, got {col_count}. "
                        f"Row: '{line_stripped[:100]}...'"
                    )
                    if table_rows:
                        _render_table_or_drop(table_rows, story, styles, branding_config, layout_config)
                    table_rows = []
                    expected_column_count = None
                    table_separator_seen = False
                    in_table = False
                    # Fall through to render this line as text
            else:
                # Non-pipe line - end table
                if table_rows:
                    _render_table_or_drop(table_rows, story, styles, branding_config, layout_config)
                table_rows = []
                expected_column_count = None
                table_separator_seen = False
                in_table = False
                # Fall through to process this line normally

        # Handle horizontal rules (only if not part of a table)
        if line_stripped.startswith("---") or line_stripped == "---":
            in_list, list_items, in_table, table_rows = _handle_horizontal_rule(
                story, in_list, list_items, in_table, table_rows, styles, branding_config, layout_config
            )
            i += 1
            continue

        # Handle headings
        in_list, list_items, should_continue = _handle_heading(
            line_stripped, story, in_list, list_items, styles
        )
        if should_continue:
            i += 1
            continue

        # Handle list items
        in_list, list_items, should_continue = _handle_list_item(
            line_stripped, in_list, list_items
        )
        if should_continue:
            i += 1
            continue

        # Handle paragraphs
        in_list, list_items = _handle_paragraph(
            line_stripped, story, in_list, list_items, styles
        )

        i += 1

    # Handle trailing list or table
    if in_list and list_items:
        # Check list type and use appropriate formatter
        if in_list == "numbered":
            story.extend(format_numbered_list_items(list_items, styles))
        else:
            story.extend(format_list_items(list_items, styles))
    if in_table and table_rows:
        _render_table_or_drop(table_rows, story, styles, branding_config, layout_config)

    # Safety check: ensure story has content
    if not story:
        logger.warning("PDF story is empty, adding default content")
        story.append(Paragraph("Report Content", styles["Title"]))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(report[:500] if len(report) > 500 else report, styles["Normal"]))

    return story

