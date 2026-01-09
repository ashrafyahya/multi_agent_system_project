"""Validator for source quality distribution.

Validates that collected competitor data has sufficient primary sources
and doesn't rely too heavily on low-quality sources (marketing blogs,
community discussions).

This validator follows the Validator Pattern, returning ValidationResult
objects with errors for blocking issues and warnings for non-blocking.
Primary source requirements are now blocking to ensure data quality.
"""

import logging
from typing import Any

from src.config import get_config
from src.graph.validators.base_validator import BaseValidator, ValidationResult
from src.models.source_quality import SourceQuality

logger = logging.getLogger(__name__)


class SourceQualityValidator(BaseValidator):
    """Validator that checks source quality distribution.
    
    This validator analyzes the quality distribution of sources used in
    competitor data collection. It checks:
    - Minimum number of primary sources (blocking if not met)
    - Maximum ratio of low-quality sources (warning if exceeded)
    
    Primary source validation is blocking: if minimum primary sources are
    not found, validation fails. Low-quality source warnings are non-blocking.
    
    Attributes:
        min_primary_sources: Minimum number of primary sources required
        max_low_quality_ratio: Maximum ratio of low-quality sources
    """
    
    def __init__(self) -> None:
        """Initialize validator with config values."""
        super().__init__()
        config = get_config()
        self.min_primary_sources = config.min_primary_sources
        self.max_low_quality_ratio = config.max_low_quality_sources_ratio
    
    @property
    def name(self) -> str:
        """Return the validator name."""
        return "SourceQualityValidator"
    
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate source quality distribution.
        
        Analyzes the quality distribution of sources in collected competitor
        data and generates errors for blocking issues and warnings for
        non-blocking issues.
        
        Args:
            data: WorkflowState dictionary containing collected competitor data.
                Expected structure:
                - "collected_data": {"competitors": List of competitor dictionaries, each with:
                  - "source_quality": SourceQuality enum value (optional)
                  - Other competitor fields}
        
        Returns:
            ValidationResult with:
            - is_valid: False if minimum primary sources not met
            - errors: List of blocking error messages
            - warnings: List of non-blocking warning messages
        """
        result = ValidationResult.success()
        
        try:
            # Check if data structure is valid
            if not isinstance(data, dict):
                logger.warning("Source quality validation: data is not a dictionary")
                return result
            
            # Support both direct dict shape {"competitors": [...]}
            # and workflow state shape {"collected_data": {"competitors": [...]}}
            competitors = data.get("competitors")
            if competitors is None:
                collected_data = data.get("collected_data", {})
                competitors = collected_data.get("competitors", [])
            
            if not isinstance(competitors, list):
                logger.warning("Source quality validation: competitors is not a list")
                return result
            
            if len(competitors) == 0:
                # No competitors to validate
                return result
            
            # Count sources by quality
            quality_counts: dict[SourceQuality, int] = {
                SourceQuality.PRIMARY: 0,
                SourceQuality.SECONDARY_HIGH: 0,
                SourceQuality.SECONDARY_MEDIUM: 0,
                SourceQuality.SECONDARY_LOW: 0,
                SourceQuality.COMMUNITY: 0,
            }
            
            sources_with_quality = 0
            
            for comp_data in competitors:
                if not isinstance(comp_data, dict):
                    continue
                
                source_quality = comp_data.get("source_quality")
                
                if source_quality is None:
                    # Source quality not analyzed (backward compatibility)
                    continue
                
                # Handle both enum and string values
                if isinstance(source_quality, str):
                    try:
                        source_quality = SourceQuality(source_quality)
                    except ValueError:
                        logger.debug(f"Invalid source quality value: {source_quality}")
                        continue
                
                if isinstance(source_quality, SourceQuality):
                    quality_counts[source_quality] += 1
                    sources_with_quality += 1
            
            # If no sources have quality information, skip validation
            if sources_with_quality == 0:
                logger.debug("No sources with quality information found")
                return result
            
            # Check minimum primary sources
            primary_count = quality_counts[SourceQuality.PRIMARY]
            if primary_count < self.min_primary_sources:
                result.add_error(
                    f"Insufficient primary sources: found {primary_count}, "
                    f"required minimum is {self.min_primary_sources}. "
                    "Primary sources are required for reliable competitor analysis. "
                    "Official websites, government sites, financial reports, and "
                    "educational institutions are considered primary sources."
                )
            
            # Check low-quality sources ratio
            low_quality_count = (
                quality_counts[SourceQuality.SECONDARY_LOW] +
                quality_counts[SourceQuality.COMMUNITY]
            )
            total_sources = sources_with_quality
            low_quality_ratio = low_quality_count / total_sources if total_sources > 0 else 0.0
            
            if low_quality_ratio > self.max_low_quality_ratio:
                result.add_warning(
                    f"High ratio of low-quality sources ({low_quality_ratio:.1%}, "
                    f"max recommended: {self.max_low_quality_ratio:.1%}). "
                    f"Found {low_quality_count} low-quality sources "
                    f"(marketing blogs, community discussions) out of "
                    f"{total_sources} total sources. Consider prioritizing "
                    "primary sources and high-quality secondary sources."
                )
            
            # Log quality distribution for debugging
            logger.debug(
                f"Source quality distribution: "
                f"PRIMARY={quality_counts[SourceQuality.PRIMARY]}, "
                f"SECONDARY_HIGH={quality_counts[SourceQuality.SECONDARY_HIGH]}, "
                f"SECONDARY_MEDIUM={quality_counts[SourceQuality.SECONDARY_MEDIUM]}, "
                f"SECONDARY_LOW={quality_counts[SourceQuality.SECONDARY_LOW]}, "
                f"COMMUNITY={quality_counts[SourceQuality.COMMUNITY]}"
            )
            
        except Exception as e:
            logger.error(
                f"Error during source quality validation: {e}. "
                "Returning empty warnings list."
            )
            # Return empty warnings (non-blocking)
            return result
        
        return result
    
    def auto_classify_source_quality(
        self,
        url: str,
        title: str = "",
    ) -> SourceQuality:
        """Automatically classify source quality based on URL and title.
        
        Uses pattern matching to classify sources as primary, secondary_high,
        secondary_medium, secondary_low, or community based on URL domains
        and title keywords.
        
        Args:
            url: Source URL to classify
            title: Optional source title for additional context
            
        Returns:
            SourceQuality enum value
        """
        import re
        
        url_lower = url.lower()
        title_lower = title.lower()
        
        # Primary sources: government, education, financial reports
        primary_patterns = [
            r"\.gov$", r"\.gov\.",  # Government sites
            r"\.edu$", r"\.edu\.",  # Educational institutions
            r"sec\.gov", r"edgar",  # SEC filings
            r"investor", r"investors",  # Investor relations
            r"annual.*report", r"financial.*report", r"10-k", r"10-q",  # Financial reports
        ]
        
        for pattern in primary_patterns:
            if re.search(pattern, url_lower) or re.search(pattern, title_lower):
                return SourceQuality.PRIMARY
        
        # Secondary High: reputable business sources
        secondary_high_patterns = [
            r"bloomberg\.com", r"reuters\.com", r"wsj\.com", r"ft\.com",  # Major news
            r"gartner\.com", r"forrester\.com", r"idc\.com",  # Research firms
            r"crunchbase\.com", r"owler\.com",  # Business intelligence
        ]
        
        for pattern in secondary_high_patterns:
            if re.search(pattern, url_lower):
                return SourceQuality.SECONDARY_HIGH
        
        # Secondary Medium: tech news and industry sites
        secondary_medium_patterns = [
            r"techcrunch\.com", r"techrepublic\.com", r"zdnet\.com",  # Tech news
            r"venturebeat\.com", r"siliconangle\.com",  # Tech publications
            r"github\.com", r"stackoverflow\.com",  # Developer communities
        ]
        
        for pattern in secondary_medium_patterns:
            if re.search(pattern, url_lower):
                return SourceQuality.SECONDARY_MEDIUM
        
        # Secondary Low: marketing blogs and company blogs
        secondary_low_patterns = [
            r"blog", r"medium\.com", r"wordpress\.com",  # Blogs
            r"hubspot\.com", r"salesforce\.com.*blog",  # Company blogs
        ]
        
        for pattern in secondary_low_patterns:
            if re.search(pattern, url_lower) or re.search(pattern, title_lower):
                return SourceQuality.SECONDARY_LOW
        
        # Community: forums, reviews, social media
        community_patterns = [
            r"reddit\.com", r"quora\.com", r"twitter\.com", r"linkedin\.com",  # Social
            r"g2\.com", r"capterra\.com", r"trustpilot\.com",  # Review sites
            r"forum", r"community",  # Forums
        ]
        
        for pattern in community_patterns:
            if re.search(pattern, url_lower) or re.search(pattern, title_lower):
                return SourceQuality.COMMUNITY
        
        # Default to secondary_medium if no patterns match
        return SourceQuality.SECONDARY_MEDIUM
