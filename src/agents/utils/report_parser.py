"""Report parsing utilities.

This module contains utilities for parsing LLM responses into structured
report dictionaries, including JSON extraction and validation.
"""

import json
import logging
import re
from typing import Any

from src.exceptions.workflow_error import WorkflowError

logger = logging.getLogger(__name__)


def parse_report_response(content: str) -> dict[str, Any]:
    """Parse LLM response into report dictionary.
    
    Extracts JSON from markdown code blocks or raw JSON, cleans it,
    and validates the structure.
    
    Args:
        content: LLM response content (may contain JSON or markdown)
    
    Returns:
        Dictionary containing report sections
    
    Raises:
        WorkflowError: If JSON cannot be parsed or structure is invalid
    """
    logger.debug(f"Parsing report response (length: {len(content)})")
    
    try:
        json_str = _extract_json_from_content(content)
        logger.debug(f"Extracted JSON string (length: {len(json_str)})")
    except WorkflowError as e:
        logger.error(f"Failed to extract JSON from content: {e}")
        logger.debug(f"Full content: {content[:1000]}")
        raise
    
    try:
        report_data = _parse_json_with_retry(json_str, content)
        _validate_report_structure(report_data)
        _normalize_report_data(report_data)
        logger.debug("Successfully parsed and validated report data")
        return report_data
    except WorkflowError as e:
        logger.error(f"Failed to parse or validate report: {e}")
        logger.debug(f"JSON string that failed: {json_str[:500]}")
        raise


def _extract_json_from_content(content: str) -> str:
    """Extract JSON string from LLM response content.
    
    Tries to find JSON in markdown code blocks first, then raw JSON.
    Uses multiple strategies to find JSON even if formatting is unusual.
    
    Args:
        content: LLM response content
    
    Returns:
        Extracted JSON string
    
    Raises:
        WorkflowError: If no JSON can be extracted from content
    """
    if not content or not content.strip():
        raise WorkflowError(
            "LLM response is empty or contains only whitespace",
            context={"content_length": len(content) if content else 0}
        )
    
    # Strategy 1: Try to find JSON in markdown code blocks (with or without json tag)
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if json_match:
        extracted = json_match.group(1).strip()
        # Fix double braces at start (common LLM mistake)
        if extracted.startswith('{{'):
            extracted = '{' + extracted[2:]
        if extracted:
            return extracted
    
    # Strategy 2: Try to find JSON object directly (non-greedy to get first complete object)
    # Look for opening brace and try to find matching closing brace
    brace_start = content.find('{')
    if brace_start != -1:
        # Fix double braces at start
        if brace_start + 1 < len(content) and content[brace_start + 1] == '{':
            brace_start += 1
        
        # Find the matching closing brace, respecting strings
        brace_count = 0
        in_string = False
        escape_next = False
        for i in range(brace_start, len(content)):
            char = content[i]
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        extracted = content[brace_start:i+1].strip()
                        if extracted:
                            return extracted
                        break
    
    # Strategy 3: Try regex to find any JSON-like structure
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
    if json_match:
        extracted = json_match.group(0).strip()
        if extracted:
            return extracted
    
    # Strategy 4: If content looks like it might be JSON, try it as-is
    content_stripped = content.strip()
    if content_stripped.startswith('{') and content_stripped.endswith('}'):
        return content_stripped
    
    # Strategy 5: Try to extract JSON from text that might have prefix/suffix
    # Look for JSON object even if there's text before/after
    json_start = content.find('{')
    if json_start != -1:
        # Try to find the end of a complete JSON object
        brace_count = 0
        in_string = False
        escape_next = False
        for i in range(json_start, len(content)):
            char = content[i]
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        potential_json = content[json_start:i+1]
                        # Quick validation: check if it looks like valid JSON
                        if potential_json.count('{') == potential_json.count('}'):
                            return potential_json.strip()
                        break
    
    # If all strategies fail, raise an error with context
    raise WorkflowError(
        "Could not extract JSON from LLM response",
        context={
            "content_preview": content[:500],
            "content_length": len(content),
            "has_braces": '{' in content and '}' in content,
        }
    )


def _parse_json_with_retry(json_str: str, original_content: str) -> dict[str, Any]:
    """Parse JSON string with retry logic for common issues.
    
    Attempts to clean and fix common JSON parsing issues like control
    characters and trailing commas.
    
    Args:
        json_str: JSON string to parse
        original_content: Original content for error messages
    
    Returns:
        Parsed JSON dictionary
    
    Raises:
        WorkflowError: If JSON cannot be parsed after retry attempts
    """
    if not json_str or not json_str.strip():
        raise WorkflowError(
            "Extracted JSON string is empty",
            context={"original_content_preview": original_content[:500]}
        )
    
    # Fix common issues before parsing
    # Fix double braces at start/end
    json_str_fixed = json_str.strip()
    if json_str_fixed.startswith('{{'):
        json_str_fixed = '{' + json_str_fixed[2:]
    if json_str_fixed.endswith('}}'):
        json_str_fixed = json_str_fixed[:-2] + '}'
    
    # Try 1: Direct parse
    try:
        return json.loads(json_str_fixed)
    except json.JSONDecodeError as e:
        logger.debug(f"Direct parse failed: {e}")
        last_error = e
    
    # Try 2: Clean control characters and trailing commas
    try:
        json_str_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str_fixed)
        json_str_clean = re.sub(r',\s*([}\]])', r'\1', json_str_clean)
        return json.loads(json_str_clean)
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse after cleaning: {e}")
        last_error = e
    
    # Try 3: Fix unclosed strings (common issue with long text)
    try:
        # Count quotes - if odd, might be unclosed string
        quote_count = json_str_fixed.count('"') - json_str_fixed.count('\\"')
        if quote_count % 2 != 0:
            # Try to close the last unclosed string
            last_quote = json_str_fixed.rfind('"')
            if last_quote != -1:
                # Check if it's inside a string value
                before_quote = json_str_fixed[:last_quote]
                if before_quote.count('"') % 2 == 0:
                    # This might be an unclosed string, try adding closing quote
                    json_str_fixed = json_str_fixed + '"'
        return json.loads(json_str_fixed)
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse after fixing strings: {e}")
        last_error = e
    
    # Try 4: Remove any text before first { and after last }
    try:
        first_brace = json_str_fixed.find('{')
        last_brace = json_str_fixed.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str_trimmed = json_str_fixed[first_brace:last_brace+1]
            return json.loads(json_str_trimmed)
    except (json.JSONDecodeError, ValueError) as e:
        logger.debug(f"Failed to parse after trimming: {e}")
        last_error = e
    
    # All attempts failed - raise error with detailed context including actual error
    logger.error(f"Failed to parse JSON from response after all retry attempts")
    logger.error(f"JSON parsing error: {last_error}")
    if isinstance(last_error, json.JSONDecodeError):
        logger.error(f"Error at line {last_error.lineno}, column {last_error.colno}: {last_error.msg}")
    logger.debug(f"JSON string preview (first 500 chars): {json_str_fixed[:500]}")
    logger.debug(f"JSON string preview (last 200 chars): {json_str_fixed[-200:]}")
    logger.debug(f"Original content preview: {original_content[:500]}")
    
    error_context = {
        "error": str(last_error),
        "json_preview_start": json_str_fixed[:200],
        "json_preview_end": json_str_fixed[-200:],
        "content_preview": original_content[:500],
        "json_length": len(json_str_fixed),
    }
    
    if isinstance(last_error, json.JSONDecodeError):
        error_context.update({
            "error_line": last_error.lineno,
            "error_column": last_error.colno,
            "error_message": last_error.msg,
        })
    
    raise WorkflowError(
        "Failed to parse report from LLM response",
        context=error_context
    )


def _validate_report_structure(report_data: dict[str, Any]) -> None:
    """Validate that report data has required structure.
    
    Args:
        report_data: Parsed report dictionary
    
    Raises:
        WorkflowError: If required fields are missing or have wrong types
    """
    required_fields = [
        "executive_summary",
        "swot_breakdown",
        "competitor_overview",
        "recommendations",
    ]
    
    missing_fields = [field for field in required_fields if field not in report_data]
    if missing_fields:
        raise WorkflowError(
            f"Report missing required sections: {missing_fields}",
            context={"report_data": report_data}
        )
    
    for field in required_fields:
        if not isinstance(report_data.get(field), str):
            raise WorkflowError(
                f"Report section '{field}' must be a string",
                context={"report_data": report_data}
            )


def _normalize_report_data(report_data: dict[str, Any]) -> None:
    """Normalize report data by setting defaults and validating optional fields.
    
    Args:
        report_data: Report dictionary to normalize (modified in place)
    
    Raises:
        WorkflowError: If optional fields have wrong types
    """
    if "methodology" in report_data:
        if not isinstance(report_data.get("methodology"), str):
            raise WorkflowError(
                "Report section 'methodology' must be a string if provided",
                context={"report_data": report_data}
            )
    else:
        report_data["methodology"] = None
    
    if "sources" in report_data:
        if not isinstance(report_data.get("sources"), list):
            raise WorkflowError(
                "Report field 'sources' must be a list if provided",
                context={"report_data": report_data}
            )
        report_data["sources"] = [str(s) for s in report_data["sources"] if s]
    else:
        report_data["sources"] = None
    
    report_data.setdefault("min_length", 1200)

