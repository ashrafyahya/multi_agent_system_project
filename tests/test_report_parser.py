"""Tests for report parser utilities."""

import json
import pytest

from src.agents.utils.report_parser import parse_report_response
from src.exceptions.workflow_error import WorkflowError


class TestParseReportResponse:
    """Tests for parse_report_response function."""
    
    def test_parse_valid_json(self) -> None:
        """Test parsing valid JSON response."""
        content = json.dumps({
            "executive_summary": "Test summary",
            "swot_breakdown": "SWOT analysis",
            "competitor_overview": "Overview",
            "recommendations": "Recommendations",
            "methodology": "Methodology",
            "sources": ["https://example.com"],
            "min_length": 1200
        })
        
        result = parse_report_response(content)
        
        assert result["executive_summary"] == "Test summary"
        assert result["swot_breakdown"] == "SWOT analysis"
        assert result["sources"] == ["https://example.com"]
    
    def test_parse_json_in_markdown_code_block(self) -> None:
        """Test parsing JSON from markdown code block."""
        content = """Here's the report:
```json
{
    "executive_summary": "Summary",
    "swot_breakdown": "SWOT",
    "competitor_overview": "Overview",
    "recommendations": "Recommendations"
}
```"""
        
        result = parse_report_response(content)
        
        assert result["executive_summary"] == "Summary"
        assert result["methodology"] is None
        assert result["sources"] is None
    
    def test_parse_missing_required_fields(self) -> None:
        """Test parsing with missing required fields."""
        content = json.dumps({
            "executive_summary": "Summary"
        })
        
        with pytest.raises(WorkflowError, match="missing required sections"):
            parse_report_response(content)
    
    def test_parse_invalid_field_types(self) -> None:
        """Test parsing with invalid field types."""
        content = json.dumps({
            "executive_summary": 123,  # Should be string
            "swot_breakdown": "SWOT",
            "competitor_overview": "Overview",
            "recommendations": "Recommendations"
        })
        
        with pytest.raises(WorkflowError, match="must be a string"):
            parse_report_response(content)
    
    def test_parse_with_methodology(self) -> None:
        """Test parsing with methodology section."""
        content = json.dumps({
            "executive_summary": "Summary",
            "swot_breakdown": "SWOT",
            "competitor_overview": "Overview",
            "recommendations": "Recommendations",
            "methodology": "Methodology section"
        })
        
        result = parse_report_response(content)
        
        assert result["methodology"] == "Methodology section"
    
    def test_parse_with_sources(self) -> None:
        """Test parsing with sources list."""
        content = json.dumps({
            "executive_summary": "Summary",
            "swot_breakdown": "SWOT",
            "competitor_overview": "Overview",
            "recommendations": "Recommendations",
            "sources": ["https://example.com", "https://test.com"]
        })
        
        result = parse_report_response(content)
        
        assert result["sources"] == ["https://example.com", "https://test.com"]
    
    def test_parse_sets_default_min_length(self) -> None:
        """Test that min_length defaults to 1200."""
        content = json.dumps({
            "executive_summary": "Summary",
            "swot_breakdown": "SWOT",
            "competitor_overview": "Overview",
            "recommendations": "Recommendations"
        })
        
        result = parse_report_response(content)
        
        assert result["min_length"] == 1200

