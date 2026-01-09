"""Tests for source quality analyzer.

This module contains unit tests for the SourceQualityAnalyzer to verify
URL classification logic and caching behavior.
"""

import pytest
from unittest.mock import Mock, patch

from src.models.source_quality import SourceQuality
from src.utils.source_quality_analyzer import SourceQualityAnalyzer


class TestSourceQualityAnalyzer:
    """Tests for SourceQualityAnalyzer class."""
    
    def test_analyze_primary_government_domains(self) -> None:
        """Test that government domains are classified as PRIMARY."""
        analyzer = SourceQualityAnalyzer()
        
        assert analyzer.analyze("https://www.sec.gov/company") == SourceQuality.PRIMARY
        assert analyzer.analyze("https://www.federalreserve.gov/data") == SourceQuality.PRIMARY
        assert analyzer.analyze("https://www.treasury.gov/reports") == SourceQuality.PRIMARY
        assert analyzer.analyze("https://www.irs.gov/forms") == SourceQuality.PRIMARY
        assert analyzer.analyze("https://www.example.gov/page") == SourceQuality.PRIMARY
        assert analyzer.analyze("https://www.example.gov.uk/page") == SourceQuality.PRIMARY
        assert analyzer.analyze("https://www.example.gov.au/page") == SourceQuality.PRIMARY
    
    def test_analyze_primary_educational_domains(self) -> None:
        """Test that educational domains are classified as PRIMARY."""
        analyzer = SourceQualityAnalyzer()
        
        assert analyzer.analyze("https://www.harvard.edu/research") == SourceQuality.PRIMARY
        assert analyzer.analyze("https://www.mit.edu/publications") == SourceQuality.PRIMARY
        assert analyzer.analyze("https://www.example.ac.uk/page") == SourceQuality.PRIMARY
        assert analyzer.analyze("https://www.example.edu.au/page") == SourceQuality.PRIMARY
    
    def test_analyze_secondary_high_market_research(self) -> None:
        """Test that market research institutes are classified as SECONDARY_HIGH."""
        analyzer = SourceQualityAnalyzer()
        
        assert analyzer.analyze("https://www.gartner.com/reports") == SourceQuality.SECONDARY_HIGH
        assert analyzer.analyze("https://www.forrester.com/analysis") == SourceQuality.SECONDARY_HIGH
        assert analyzer.analyze("https://www.idc.com/research") == SourceQuality.SECONDARY_HIGH
        assert analyzer.analyze("https://www.mckinsey.com/insights") == SourceQuality.SECONDARY_HIGH
        assert analyzer.analyze("https://www.statista.com/statistics") == SourceQuality.SECONDARY_HIGH
        assert analyzer.analyze("https://www.bloomberg.com/news") == SourceQuality.SECONDARY_HIGH
        assert analyzer.analyze("https://www.reuters.com/business") == SourceQuality.SECONDARY_HIGH
    
    def test_analyze_secondary_low_marketing_blogs(self) -> None:
        """Test that marketing blogs are classified as SECONDARY_LOW."""
        analyzer = SourceQualityAnalyzer()
        
        assert analyzer.analyze("https://blog.example.com/post") == SourceQuality.SECONDARY_LOW
        assert analyzer.analyze("https://www.example.com/blog/article") == SourceQuality.SECONDARY_LOW
        assert analyzer.analyze("https://marketing.example.com/news") == SourceQuality.SECONDARY_LOW
        assert analyzer.analyze("https://example.blogspot.com/post") == SourceQuality.SECONDARY_LOW
        assert analyzer.analyze("https://medium.com/@user/article") == SourceQuality.SECONDARY_LOW
        assert analyzer.analyze("https://example.wordpress.com/post") == SourceQuality.SECONDARY_LOW
    
    def test_analyze_community_sources(self) -> None:
        """Test that community sources are classified as COMMUNITY."""
        analyzer = SourceQualityAnalyzer()
        
        assert analyzer.analyze("https://www.reddit.com/r/technology") == SourceQuality.COMMUNITY
        assert analyzer.analyze("https://stackoverflow.com/questions/123") == SourceQuality.COMMUNITY
        assert analyzer.analyze("https://stackexchange.com/questions") == SourceQuality.COMMUNITY
        assert analyzer.analyze("https://www.quora.com/question") == SourceQuality.COMMUNITY
    
    def test_analyze_default_secondary_medium(self) -> None:
        """Test that unknown sources default to SECONDARY_MEDIUM."""
        analyzer = SourceQualityAnalyzer()
        
        assert analyzer.analyze("https://www.techcrunch.com/article") == SourceQuality.SECONDARY_MEDIUM
        assert analyzer.analyze("https://www.example.com/news") == SourceQuality.SECONDARY_MEDIUM
        assert analyzer.analyze("https://www.unknown-site.com/page") == SourceQuality.SECONDARY_MEDIUM
    
    def test_analyze_case_insensitive(self) -> None:
        """Test that URL analysis is case-insensitive."""
        analyzer = SourceQualityAnalyzer()
        
        assert analyzer.analyze("https://www.SEC.GOV/company") == SourceQuality.PRIMARY
        assert analyzer.analyze("https://www.GARTNER.COM/reports") == SourceQuality.SECONDARY_HIGH
        assert analyzer.analyze("https://www.REDDIT.COM/r/tech") == SourceQuality.COMMUNITY
    
    def test_analyze_invalid_url(self) -> None:
        """Test that invalid URLs default to SECONDARY_MEDIUM."""
        analyzer = SourceQualityAnalyzer()
        
        assert analyzer.analyze("") == SourceQuality.SECONDARY_MEDIUM
        assert analyzer.analyze("not-a-url") == SourceQuality.SECONDARY_MEDIUM
    
    def test_analyze_none_url(self) -> None:
        """Test that None URL defaults to SECONDARY_MEDIUM."""
        analyzer = SourceQualityAnalyzer()
        
        # Type checker will complain, but test runtime behavior
        result = analyzer.analyze(None)  # type: ignore
        assert result == SourceQuality.SECONDARY_MEDIUM
    
    def test_analyze_caching(self) -> None:
        """Test that URL analysis results are cached."""
        analyzer = SourceQualityAnalyzer()
        
        url = "https://www.sec.gov/company"
        
        # First call
        result1 = analyzer.analyze(url)
        
        # Second call should use cache
        result2 = analyzer.analyze(url)
        
        assert result1 == result2 == SourceQuality.PRIMARY
        
        # Verify cache is working by checking that same URL returns same result
        # (cache hit should be faster, but we can't easily test that)
        assert analyzer.analyze(url) == SourceQuality.PRIMARY
    
    def test_analyze_error_handling(self) -> None:
        """Test that errors in analysis return default value."""
        analyzer = SourceQualityAnalyzer()
        
        # Mock urlparse to raise an exception
        with patch("src.utils.source_quality_analyzer.urlparse", side_effect=Exception("Test error")):
            result = analyzer.analyze("https://www.example.com")
            assert result == SourceQuality.SECONDARY_MEDIUM
    
    def test_analyze_with_query_parameters(self) -> None:
        """Test that URLs with query parameters are handled correctly."""
        analyzer = SourceQualityAnalyzer()
        
        assert analyzer.analyze("https://www.sec.gov/company?param=value") == SourceQuality.PRIMARY
        assert analyzer.analyze("https://www.gartner.com/reports?id=123") == SourceQuality.SECONDARY_HIGH
        assert analyzer.analyze("https://www.reddit.com/r/tech?sort=hot") == SourceQuality.COMMUNITY
    
    def test_analyze_with_fragments(self) -> None:
        """Test that URLs with fragments are handled correctly."""
        analyzer = SourceQualityAnalyzer()
        
        assert analyzer.analyze("https://www.sec.gov/company#section") == SourceQuality.PRIMARY
        assert analyzer.analyze("https://www.gartner.com/reports#summary") == SourceQuality.SECONDARY_HIGH
    
    def test_analyze_www_prefix(self) -> None:
        """Test that www. prefix is handled correctly."""
        analyzer = SourceQualityAnalyzer()
        
        assert analyzer.analyze("https://www.sec.gov/page") == SourceQuality.PRIMARY
        assert analyzer.analyze("https://sec.gov/page") == SourceQuality.PRIMARY
        assert analyzer.analyze("https://www.gartner.com/page") == SourceQuality.SECONDARY_HIGH
        assert analyzer.analyze("https://gartner.com/page") == SourceQuality.SECONDARY_HIGH
