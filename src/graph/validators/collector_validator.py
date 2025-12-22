"""Validator for collector node output.

Validates collected competitor data against quality standards:
- Minimum 4 unique sources
- No empty required fields
- Valid URL formats
- Deduplicated competitor entries

This validator follows the Validator Pattern, returning ValidationResult
objects instead of raising exceptions for validation failures.
"""

import logging
from typing import Any

from src.config import get_config
from src.graph.validators.base_validator import BaseValidator, ValidationResult
from src.tools.text_utils import validate_url

logger = logging.getLogger(__name__)


class CollectorValidator(BaseValidator):
    """Validates collector node output.
    
    This validator checks collected competitor data against quality standards
    to ensure it meets minimum requirements before proceeding to the insight
    generation stage. It validates data structure, completeness, and quality.
    
    Validation checks:
    - Minimum 4 unique competitor sources
    - No empty required fields (name, source_url)
    - Valid URL formats for source_url and website
    - No duplicate competitor entries (by name)
    """
    
    def __init__(self) -> None:
        """Initialize validator with configuration values."""
        super().__init__()
        config = get_config()
        self.min_sources = config.min_collector_sources
    
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate collected competitor data.
        
        Performs comprehensive validation on collector node output to ensure
        data quality and completeness. Returns ValidationResult with errors
        and warnings, never raises exceptions.
        
        Validation checks performed:
        1. Minimum 4 unique competitor sources
        2. Each competitor has required fields (name, source_url)
        3. URL formats are valid
        4. No duplicate competitors (case-insensitive name matching)
        
        Args:
            data: Collector output dictionary expected to contain:
                - "competitors": List of competitor dictionaries, each with:
                  - "name": str (required)
                  - "source_url": str (required)
                  - "website": str (optional)
                  - Other optional fields
        
        Returns:
            ValidationResult with:
            - is_valid: True if all validations pass
            - errors: List of error messages for validation failures
            - warnings: List of warning messages for non-critical issues
        """
        result = ValidationResult.success()
        
        # Check if data structure is valid
        if not isinstance(data, dict):
            result.add_error("Collector output must be a dictionary")
            return result
        
        competitors = data.get("competitors", [])
        
        if not isinstance(competitors, list):
            result.add_error("'competitors' field must be a list")
            return result
        
        # Check minimum sources
        if len(competitors) < self.min_sources:
            result.add_error(
                f"Minimum {self.min_sources} sources required, "
                f"found {len(competitors)}"
            )
        
        # Validate each competitor
        seen_names: set[str] = set()
        seen_source_urls: set[str] = set()
        
        for idx, comp_data in enumerate(competitors):
            if not isinstance(comp_data, dict):
                result.add_error(f"Competitor at index {idx} must be a dictionary")
                continue
            
            # Check for duplicate names (case-insensitive)
            name = comp_data.get("name", "")
            if name:
                name_lower = name.lower().strip()
                if name_lower in seen_names:
                    result.add_error(
                        f"Duplicate competitor at index {idx}: '{name}' "
                        f"(case-insensitive match)"
                    )
                seen_names.add(name_lower)
            else:
                result.add_error(f"Competitor at index {idx} missing required field: 'name'")
            
            # Validate required fields
            if not comp_data.get("name") or not comp_data.get("name", "").strip():
                result.add_error(f"Competitor at index {idx} has empty or missing 'name' field")
            
            # Validate source_url (required)
            source_url = comp_data.get("source_url", "")
            if not source_url:
                result.add_error(
                    f"Competitor at index {idx} missing required field: 'source_url'"
                )
            else:
                # Check for duplicate source URLs
                source_url_str = str(source_url).strip()
                if source_url_str in seen_source_urls:
                    result.add_warning(
                        f"Competitor at index {idx} has duplicate source_url: '{source_url_str}'"
                    )
                seen_source_urls.add(source_url_str)
                
                # Validate URL format
                if not validate_url(source_url_str):
                    result.add_error(
                        f"Competitor at index {idx} has invalid source_url format: '{source_url_str}'"
                    )
            
            # Validate website URL if present (optional field)
            website = comp_data.get("website")
            if website:
                website_str = str(website).strip()
                if not validate_url(website_str):
                    result.add_warning(
                        f"Competitor at index {idx} has invalid website URL format: '{website_str}'"
                    )
        
        # Log validation result
        if result.is_valid:
            logger.info(
                f"Collector validation passed: {len(competitors)} competitors validated"
            )
        else:
            logger.warning(
                f"Collector validation failed: {len(result.errors)} errors, "
                f"{len(result.warnings)} warnings"
            )
        
        return result
    
    @property
    def name(self) -> str:
        """Return validator name.
        
        Returns:
            String identifier for this validator
        """
        return "collector_validator"
