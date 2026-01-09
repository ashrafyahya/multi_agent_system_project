"""Data verification service for validating quantitative data in sources.

This module provides utilities to verify whether quantitative data
(market share, revenue, user count, etc.) actually appears in the source
content, distinguishing between verified data and estimates.
"""

import logging
import re
from typing import Any

from src.config import get_config

logger = logging.getLogger(__name__)


class DataVerificationService:
    """Service for verifying quantitative data in source content.
    
    Follows KISS principle: Pattern matching first, LLM only if needed.
    Stateless utility class following Tool Pattern.
    
    The service checks if quantitative metrics (market_share, revenue, etc.)
    actually appear in the source content, classifying them as:
    - "verified": Data found in source content
    - "estimated": Data appears to be extrapolated or estimated
    - "not_found": Data not found in source content
    
    Example:
        ```python
        from src.utils.data_verification_service import DataVerificationService
        
        service = DataVerificationService()
        competitor_data = {
            "market_share": 35.0,
            "revenue": "$1.5B",
        }
        source_content = "The company has a 35% market share and reported revenue of $1.5B."
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["market_share"] == "verified"
        assert status["revenue"] == "verified"
        ```
    """
    
    def __init__(self) -> None:
        """Initialize service with config values."""
        config = get_config()
        self.enabled = getattr(config, "enable_data_verification", True)
        self.timeout = getattr(config, "data_verification_timeout", 10.0)
    
    def verify_quantitative_data(
        self,
        competitor_data: dict[str, Any],
        source_content: str,
    ) -> dict[str, str]:
        """Verify quantitative data against source content.
        
        Checks if each quantitative metric in competitor_data actually
        appears in the source_content. Uses pattern matching first (KISS),
        only uses LLM if pattern matching is uncertain.
        
        Args:
            competitor_data: Dictionary containing competitor metrics.
                Keys are metric names (e.g., "market_share", "revenue"),
                values are the metric values (float, int, or string).
            source_content: Source content (snippet, article text, etc.)
                to search for the metric values.
            
        Returns:
            Dictionary mapping metric names to verification status.
            Keys are metric names, values are "verified", "estimated", or "not_found".
            
        Note:
            If verification is disabled in config, returns "not_found" for all metrics.
            If source_content is empty, returns "not_found" for all metrics.
        """
        if not self.enabled:
            logger.debug("Data verification is disabled")
            return {
                metric: "not_found"
                for metric in self._extract_quantitative_metrics(competitor_data)
            }
        
        if not source_content or not isinstance(source_content, str):
            logger.debug("Empty or invalid source content provided")
            return {
                metric: "not_found"
                for metric in self._extract_quantitative_metrics(competitor_data)
            }
        
        try:
            metrics = self._extract_quantitative_metrics(competitor_data)
            verification_status: dict[str, str] = {}
            
            source_lower = source_content.lower()
            
            for metric_name, metric_value in metrics.items():
                status = self._verify_single_metric(
                    metric_name,
                    metric_value,
                    source_content,
                    source_lower,
                )
                verification_status[metric_name] = status
            
            return verification_status
            
        except Exception as e:
            logger.warning(
                f"Error during data verification: {e}. "
                "Returning 'not_found' for all metrics."
            )
            return {
                metric: "not_found"
                for metric in self._extract_quantitative_metrics(competitor_data)
            }
    
    def _extract_quantitative_metrics(
        self,
        competitor_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract quantitative metrics from competitor data.
        
        Args:
            competitor_data: Dictionary containing competitor data
            
        Returns:
            Dictionary of metric names to metric values
        """
        quantitative_fields = [
            "market_share",
            "revenue",
            "user_count",
            "founded_year",
        ]
        
        metrics: dict[str, Any] = {}
        for field in quantitative_fields:
            if field in competitor_data and competitor_data[field] is not None:
                metrics[field] = competitor_data[field]
        
        return metrics
    
    def _verify_single_metric(
        self,
        metric_name: str,
        metric_value: Any,
        source_content: str,
        source_lower: str,
    ) -> str:
        """Verify a single metric against source content.
        
        Args:
            metric_name: Name of the metric (e.g., "market_share")
            metric_value: Value of the metric (float, int, or string)
            source_content: Original source content (for exact matching)
            source_lower: Lowercased source content (for pattern matching)
            
        Returns:
            Verification status: "verified", "estimated", or "not_found"
        """
        # Convert metric value to searchable patterns
        patterns = self._generate_search_patterns(metric_name, metric_value)
        
        # Try to find metric value in source
        found = False
        for pattern in patterns:
            if re.search(pattern, source_lower):
                found = True
                break
        
        if found:
            return "verified"
        
        # Check if similar values exist (might be estimated)
        if self._has_similar_values(metric_name, metric_value, source_lower):
            return "estimated"
        
        return "not_found"
    
    def _generate_search_patterns(
        self,
        metric_name: str,
        metric_value: Any,
    ) -> list[str]:
        """Generate regex patterns to search for metric value in source.
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            
        Returns:
            List of regex patterns to search for
        """
        patterns: list[str] = []
        
        if metric_name == "market_share":
            # Market share: percentage (e.g., 35, 35%, 35 percent)
            if isinstance(metric_value, (int, float)):
                value = float(metric_value)
                # Exact match: "35%", "35 percent", "35% market share"
                patterns.append(rf"\b{re.escape(str(int(value)))}\s*%")
                patterns.append(rf"\b{re.escape(str(int(value)))}\s+percent")
                # Allow small variations (e.g., 34-36%)
                if value >= 1:
                    patterns.append(
                        rf"\b{re.escape(str(int(value) - 1))}\s*-\s*{re.escape(str(int(value) + 1))}\s*%"
                    )
        
        elif metric_name == "revenue":
            # Revenue: dollar amounts (e.g., $1.5B, $1.5 billion, 1.5B)
            if isinstance(metric_value, str):
                # Extract number and unit from string like "$1.5B" or "$1.5 billion"
                revenue_str = metric_value.lower().strip()
                # Pattern: $1.5B, $1.5B revenue, $1.5 billion
                patterns.append(re.escape(revenue_str))
                # Also try without $ sign
                if revenue_str.startswith("$"):
                    patterns.append(re.escape(revenue_str[1:]))
            elif isinstance(metric_value, (int, float)):
                # Convert to billions/millions format
                value = float(metric_value)
                if value >= 1_000_000_000:
                    billions = value / 1_000_000_000
                    patterns.append(rf"\${re.escape(str(billions))}\s*[Bb]")
                    patterns.append(rf"{re.escape(str(billions))}\s*[Bb]illion")
                elif value >= 1_000_000:
                    millions = value / 1_000_000
                    patterns.append(rf"\${re.escape(str(millions))}\s*[Mm]")
                    patterns.append(rf"{re.escape(str(millions))}\s*[Mm]illion")
        
        elif metric_name == "user_count":
            # User count: numbers (e.g., 1M, 1 million, 1,000,000)
            if isinstance(metric_value, (int, float)):
                value = int(metric_value)
                # Exact number
                patterns.append(rf"\b{re.escape(str(value))}\b")
                # With commas: 1,000,000
                if value >= 1000:
                    patterns.append(rf"\b{re.escape(f'{value:,}')}\b")
                # In millions: 1M, 1 million
                if value >= 1_000_000:
                    millions = value / 1_000_000
                    patterns.append(rf"\b{re.escape(str(int(millions)))}\s*[Mm]")
                    patterns.append(rf"\b{re.escape(str(int(millions)))}\s+[Mm]illion")
        
        elif metric_name == "founded_year":
            # Founded year: year (e.g., 2010, founded in 2010)
            if isinstance(metric_value, (int, float)):
                year = int(metric_value)
                patterns.append(rf"\b{re.escape(str(year))}\b")
                patterns.append(rf"founded\s+(?:in\s+)?{re.escape(str(year))}")
                patterns.append(rf"established\s+(?:in\s+)?{re.escape(str(year))}")
        
        return patterns
    
    def _has_similar_values(
        self,
        metric_name: str,
        metric_value: Any,
        source_lower: str,
    ) -> bool:
        """Check if source contains similar values (might indicate estimation).
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            source_lower: Lowercased source content
            
        Returns:
            True if similar values are found (suggests estimation)
        """
        if metric_name == "market_share":
            # Check for percentage mentions without exact value
            if isinstance(metric_value, (int, float)):
                value = float(metric_value)
                # Look for percentage ranges or "around X%" patterns
                if re.search(rf"\b(?:around|approximately|about|roughly)\s+\d+\s*%", source_lower):
                    return True
                # Look for percentage ranges
                if re.search(rf"\d+\s*-\s*\d+\s*%", source_lower):
                    return True
        
        elif metric_name == "revenue":
            # Check for revenue mentions without exact value
            if re.search(r"revenue\s+(?:of\s+)?(?:around|approximately|about|roughly|estimated)", source_lower):
                return True
            if re.search(r"estimated\s+revenue", source_lower):
                return True
        
        return False
    
    def classify_data_origin(
        self,
        metric_name: str,
        metric_value: Any,
        source_content: str,
    ) -> str:
        """Classify the origin of data (verified, estimated, or not_found).
        
        This is a convenience method that wraps verify_quantitative_data
        for a single metric.
        
        Args:
            metric_name: Name of the metric
            metric_value: Value of the metric
            source_content: Source content to search
            
        Returns:
            "verified", "estimated", or "not_found"
        """
        competitor_data = {metric_name: metric_value}
        status = self.verify_quantitative_data(competitor_data, source_content)
        return status.get(metric_name, "not_found")
    
    def verify_primary_sources(
        self,
        sources: list[dict[str, Any]],
    ) -> dict[str, bool]:
        """Verify if sources are primary sources.
        
        Classifies sources as primary or secondary based on URL patterns
        and content indicators. Primary sources include official websites,
        financial reports, government sites, and educational institutions.
        
        Args:
            sources: List of source dictionaries with 'url' and 'title' keys.
            
        Returns:
            Dictionary mapping source URLs to boolean indicating if primary.
        """
        primary_indicators = {
            "urls": [
                r"\.gov$",  # Government sites
                r"\.edu$",  # Educational institutions
                r"investor\.|\.ir$",  # Investor relations
                r"sec\.gov",  # SEC filings
                r"annual.*report",  # Annual reports
                r"financial.*report",  # Financial reports
            ],
            "titles": [
                r"annual report",
                r"financial report",
                r"investor presentation",
                r"sec filing",
                r"10-k",
                r"10-q",
            ],
        }
        
        verification_results: dict[str, bool] = {}
        
        for source in sources:
            url = source.get("url", "").lower()
            title = source.get("title", "").lower()
            
            is_primary = False
            
            # Check URL patterns
            for pattern in primary_indicators["urls"]:
                if re.search(pattern, url):
                    is_primary = True
                    break
            
            # Check title patterns if URL didn't match
            if not is_primary:
                for pattern in primary_indicators["titles"]:
                    if re.search(pattern, title):
                        is_primary = True
                        break
            
            verification_results[url] = is_primary
        
        return verification_results
