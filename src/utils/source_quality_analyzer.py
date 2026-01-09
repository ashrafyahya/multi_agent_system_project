"""Source quality analyzer for classifying data source reliability.

This module provides utilities to automatically classify the quality and
reliability of data sources based on URL patterns.
"""

import logging
import re
from functools import lru_cache
from urllib.parse import urlparse

from src.config import get_config
from src.models.source_quality import SourceQuality

logger = logging.getLogger(__name__)


class SourceQualityAnalyzer:
    """Analyzes source URLs to determine quality classification.
    
    Follows KISS principle: Simple pattern matching, no LLM calls.
    Stateless utility class following Tool Pattern.
    Uses caching to improve performance.
    
    The analyzer classifies sources into quality categories:
    - PRIMARY: Official sources (government, financial reports, scientific)
    - SECONDARY_HIGH: Renowned market research institutes
    - SECONDARY_MEDIUM: Tech news sites, trade journals
    - SECONDARY_LOW: Marketing blogs, company blogs
    - COMMUNITY: Community discussions (Reddit, forums)
    
    Example:
        ```python
        from src.utils.source_quality_analyzer import SourceQualityAnalyzer
        
        analyzer = SourceQualityAnalyzer()
        quality = analyzer.analyze("https://www.sec.gov/company")
        assert quality == SourceQuality.PRIMARY
        ```
    """
    
    def __init__(self) -> None:
        """Initialize analyzer with config values."""
        config = get_config()
        self.cache_size = getattr(config, "source_quality_cache_size", 1000)
    
    @lru_cache(maxsize=1000)
    def analyze(self, url: str) -> SourceQuality:
        """Analyze source URL and return quality classification.
        
        Uses pattern matching to classify URLs into quality categories.
        Returns SECONDARY_MEDIUM as default if analysis fails or URL
        doesn't match any known patterns.
        
        Args:
            url: Source URL to analyze
            
        Returns:
            SourceQuality enum value
            
        Note:
            Results are cached to improve performance. Cache size is
            configurable via source_quality_cache_size config option.
        """
        if not url or not isinstance(url, str):
            logger.warning(f"Invalid URL provided for analysis: {url}")
            return SourceQuality.SECONDARY_MEDIUM
        
        try:
            parsed = urlparse(url.lower().strip())
            domain = parsed.netloc or parsed.path.split("/")[0] if parsed.path else ""
            
            if not domain:
                logger.debug(f"Could not extract domain from URL: {url}")
                return SourceQuality.SECONDARY_MEDIUM
            
            # Remove www. prefix for matching
            domain = domain.replace("www.", "")
            
            # PRIMARY sources: Official government, financial, educational
            if self._is_primary_source(domain, url):
                return SourceQuality.PRIMARY
            
            # SECONDARY_HIGH: Renowned market research institutes
            if self._is_secondary_high_source(domain, url):
                return SourceQuality.SECONDARY_HIGH
            
            # SECONDARY_LOW: Marketing blogs, company blogs
            if self._is_secondary_low_source(domain, url):
                return SourceQuality.SECONDARY_LOW
            
            # COMMUNITY: Reddit, forums, community discussions
            if self._is_community_source(domain, url):
                return SourceQuality.COMMUNITY
            
            # Default: SECONDARY_MEDIUM (tech news, trade journals, etc.)
            return SourceQuality.SECONDARY_MEDIUM
            
        except Exception as e:
            logger.warning(f"Failed to analyze source quality for {url}: {e}")
            return SourceQuality.SECONDARY_MEDIUM
    
    def _is_primary_source(self, domain: str, url: str) -> bool:
        """Check if source is a primary source.
        
        Args:
            domain: Normalized domain name
            url: Full URL for additional pattern matching
            
        Returns:
            True if source is classified as PRIMARY
        """
        # Government domains
        if domain.endswith(".gov") or domain.endswith(".gov.uk") or domain.endswith(".gov.au"):
            return True
        
        # SEC (Securities and Exchange Commission)
        if "sec.gov" in domain:
            return True
        
        # Educational institutions
        if domain.endswith(".edu") or domain.endswith(".ac.uk") or domain.endswith(".edu.au"):
            return True
        
        # Official financial reporting sites
        primary_patterns = [
            r"sec\.gov",
            r"\.gov/",
            r"\.edu/",
            r"federalreserve\.gov",
            r"treasury\.gov",
            r"irs\.gov",
        ]
        
        for pattern in primary_patterns:
            if re.search(pattern, url.lower()):
                return True
        
        return False
    
    def _is_secondary_high_source(self, domain: str, url: str) -> bool:
        """Check if source is a secondary high-quality source.
        
        Args:
            domain: Normalized domain name
            url: Full URL for additional pattern matching
            
        Returns:
            True if source is classified as SECONDARY_HIGH
        """
        # Renowned market research institutes
        high_quality_domains = [
            "gartner.com",
            "forrester.com",
            "idc.com",
            "mckinsey.com",
            "bain.com",
            "bcg.com",
            "deloitte.com",
            "pwc.com",
            "kpmg.com",
            "ey.com",
            "statista.com",
            "bloomberg.com",
            "reuters.com",
            "wsj.com",
            "ft.com",
        ]
        
        for high_domain in high_quality_domains:
            if high_domain in domain:
                return True
        
        return False
    
    def _is_secondary_low_source(self, domain: str, url: str) -> bool:
        """Check if source is a secondary low-quality source.
        
        Args:
            domain: Normalized domain name
            url: Full URL for additional pattern matching
            
        Returns:
            True if source is classified as SECONDARY_LOW
        """
        # Marketing blogs, company blogs
        low_quality_patterns = [
            r"/blog/",
            r"blog\.",
            r"marketing\.",
            r"\.blogspot\.",
            r"medium\.com/@",
            r"wordpress\.com",
            r"tumblr\.com",
        ]
        
        url_lower = url.lower()
        for pattern in low_quality_patterns:
            if re.search(pattern, url_lower):
                return True
        
        # Company blogs (blog.company.com or company.com/blog)
        if "blog" in domain.split(".") or "/blog" in url_lower:
            return True
        
        return False
    
    def _is_community_source(self, domain: str, url: str) -> bool:
        """Check if source is a community source.
        
        Args:
            domain: Normalized domain name
            url: Full URL for additional pattern matching
            
        Returns:
            True if source is classified as COMMUNITY
        """
        # Reddit, forums, community discussions
        community_domains = [
            "reddit.com",
            "stackoverflow.com",
            "stackexchange.com",
            "quora.com",
            "discourse.org",
            "phpbb.com",
            "vbulletin.com",
        ]
        
        for comm_domain in community_domains:
            if comm_domain in domain:
                return True
        
        return False
