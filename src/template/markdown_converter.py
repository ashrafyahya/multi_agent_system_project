"""Markdown to PDF conversion utilities.

This module provides functions for converting markdown content to
ReportLab-compatible formats for PDF generation.
"""

import re
from typing import Any


def convert_markdown_to_html(text: str) -> str:
    """Convert markdown formatting to HTML for reportlab Paragraph.

    Converts **bold**, *italic*, and other markdown to HTML tags.
    Strips markdown headers that might appear in paragraph text.

    Args:
        text: Markdown text to convert

    Returns:
        HTML-formatted text
    """
    if not text:
        return ""

    # Strip markdown headers that might appear in text (###, ##, #)
    # This prevents literal ### from appearing in PDF output
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Escape HTML special characters first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Convert bold **text** to <b>text</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

    # Convert italic *text* to <i>text</i> (but not if it's part of **text**)
    text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', text)

    return text


def is_numbered_list_item(line: str) -> bool:
    """Check if line is a numbered list item (e.g., '1. Item', '2. Item').

    Args:
        line: Line to check

    Returns:
        True if line is a numbered list item
    """
    pattern = r'^\d+\.\s+'
    return bool(re.match(pattern, line))


def has_inline_list(text: str) -> bool:
    """Check if text contains inline list patterns.

    Args:
        text: Text to check

    Returns:
        True if text contains inline list patterns
    """
    # Pattern: "Label: - item1 - item2" or "Label: * item1 * item2"
    patterns = [
        r':\s*-\s+[^-]',  # Colon followed by dash list
        r':\s*\*\s+[^*]',  # Colon followed by asterisk list
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def split_inline_lists(text: str) -> list[str | list[str]]:
    """Split text containing inline lists into text and list parts.

    Args:
        text: Text that may contain inline lists

    Returns:
        List of strings (text) and lists (list items)
    """
    result: list[str | list[str]] = []

    # Pattern to match "Label: - item1 - item2 - item3"
    pattern = r'([^:]+):\s*((?:-\s+[^-]+(?:$|\s+(?=-)))+)'
    match = re.search(pattern, text)

    if match:
        # Text before the list
        before = text[:match.start()].strip()
        if before:
            result.append(before)

        # Label
        label = match.group(1).strip()
        if label:
            result.append(f"{label}:")

        # Extract list items
        list_text = match.group(2)
        list_items = re.findall(r'-\s+([^-]+)', list_text)
        list_items = [item.strip() for item in list_items if item.strip()]
        if list_items:
            # Convert markdown to HTML for each item
            list_items = [convert_markdown_to_html(item) for item in list_items]
            result.append(list_items)

        # Text after the list
        after = text[match.end():].strip()
        if after:
            result.append(after)
    else:
        # No inline list found, return text as-is
        result.append(text)

    return result


def has_inline_numbered_list(text: str) -> bool:
    """Check if text contains inline numbered list patterns.

    Args:
        text: Text to check

    Returns:
        True if text contains inline numbered list patterns
    """
    # Pattern: "Text: 1. item1 2. item2 3. item3" or similar
    pattern = r'\d+\.\s+[^\d]+(?:\s+\d+\.\s+[^\d]+)+'
    return bool(re.search(pattern, text))


def split_inline_numbered_lists(text: str) -> list[str | list[str]]:
    """Split text containing inline numbered lists into text and list parts.

    Args:
        text: Text that may contain inline numbered lists

    Returns:
        List of strings (text) and lists (list items)
    """
    result: list[str | list[str]] = []

    # Pattern to match numbered lists: "1. item1 2. item2 3. item3"
    # Find the start of the first numbered item
    pattern = r'(.+?)(\d+\.\s+[^\d]+(?:\s+\d+\.\s+[^\d]+)+)'
    match = re.search(pattern, text)

    if match:
        # Text before the list
        before = match.group(1).strip()
        if before:
            result.append(before)

        # Extract numbered list items
        list_text = match.group(2)
        # Match pattern: "1. text 2. text 3. text"
        list_items = re.findall(r'(\d+)\.\s+([^\d]+?)(?=\s+\d+\.|$)', list_text)

        if list_items:
            # Extract just the text part (without numbers)
            formatted_items = []
            for num, item in list_items:
                item_clean = item.strip()
                if item_clean:
                    formatted_items.append(convert_markdown_to_html(item_clean))

            if formatted_items:
                result.append(formatted_items)

        # Text after the list
        after = text[match.end():].strip()
        if after:
            result.append(after)
    else:
        # No inline numbered list found, return text as-is
        result.append(text)

    return result

