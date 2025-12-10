"""Validator for report node output.

Validates final report against quality standards:
- Minimum length requirements
- All sections present
- Summary exists

This validator follows the Validator Pattern, returning ValidationResult
objects instead of raising exceptions for validation failures.
"""

import logging
from typing import Any

from src.graph.validators.base_validator import BaseValidator, ValidationResult

logger = logging.getLogger(__name__)


class ReportValidator(BaseValidator):
    """Validates report node output.
    
    This validator checks the final competitor analysis report to ensure
    it meets quality standards before being returned to the user. It validates
    report structure, completeness, and minimum length requirements.
    
    Validation checks:
    - All required sections are present (executive_summary, swot_breakdown,
      competitor_overview, recommendations)
    - Each section meets minimum length (50 characters)
    - Total report length meets minimum requirement (default 500 characters)
    - Executive summary exists and is meaningful
    
    Example:
        ```python
        from src.graph.validators.report_validator import ReportValidator
        
        validator = ReportValidator()
        result = validator.validate({
            "executive_summary": "Summary of findings...",
            "swot_breakdown": "SWOT analysis details...",
            "competitor_overview": "Overview of competitors...",
            "recommendations": "Strategic recommendations...",
            "min_length": 500
        })
        
        if result.is_valid:
            print("Validation passed")
        else:
            for error in result.errors:
                print(f"Error: {error}")
        ```
    """
    
    MIN_SECTION_LENGTH = 50
    DEFAULT_MIN_TOTAL_LENGTH = 500
    
    REQUIRED_SECTIONS = [
        "executive_summary",
        "swot_breakdown",
        "competitor_overview",
        "recommendations",
    ]
    
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate final report.
        
        Performs comprehensive validation on report node output to ensure
        report quality and completeness. Returns ValidationResult with errors
        and warnings, never raises exceptions.
        
        Validation checks performed:
        1. All required sections are present
        2. Each section meets minimum length (50 characters)
        3. Total report length meets minimum requirement
        4. Executive summary exists and is meaningful
        
        Args:
            data: Report output dictionary expected to contain:
                - "executive_summary": str (required, min 50 chars)
                - "swot_breakdown": str (required, min 50 chars)
                - "competitor_overview": str (required, min 50 chars)
                - "recommendations": str (required, min 50 chars)
                - "min_length": int (optional, default 500)
        
        Returns:
            ValidationResult with:
            - is_valid: True if all validations pass
            - errors: List of error messages for validation failures
            - warnings: List of warning messages for non-critical issues
            
        Example:
            ```python
            validator = ReportValidator()
            result = validator.validate(report_output)
            
            if not result.is_valid:
                logger.error(f"Validation failed: {result.get_summary()}")
                # Handle errors, possibly trigger retry
            ```
        """
        result = ValidationResult.success()
        
        # Check if data structure is valid
        if not isinstance(data, dict):
            result.add_error("Report output must be a dictionary")
            return result
        
        # Validate all required sections are present
        missing_sections = self._check_missing_sections(data)
        for section in missing_sections:
            result.add_error(f"Required section '{section}' is missing")
        
        # If sections are missing, return early
        if missing_sections:
            return result
        
        # Validate each section length
        section_errors = self._validate_section_lengths(data)
        for error in section_errors:
            result.add_error(error)
        
        # Validate total length
        min_length = data.get("min_length", self.DEFAULT_MIN_TOTAL_LENGTH)
        total_length = self._calculate_total_length(data)
        
        if total_length < min_length:
            result.add_error(
                f"Total report length ({total_length} characters) is less than "
                f"minimum required length ({min_length} characters)"
            )
        
        # Validate executive summary quality
        summary_warnings = self._validate_summary_quality(data)
        for warning in summary_warnings:
            result.add_warning(warning)
        
        # Log validation result
        if result.is_valid:
            logger.info(
                f"Report validation passed: {total_length} characters, "
                f"{len(self.REQUIRED_SECTIONS)} sections validated"
            )
        else:
            logger.warning(
                f"Report validation failed: {len(result.errors)} errors, "
                f"{len(result.warnings)} warnings"
            )
        
        return result
    
    def _check_missing_sections(self, data: dict[str, Any]) -> list[str]:
        """Check for missing required sections.
        
        Args:
            data: Report data dictionary
            
        Returns:
            List of missing section names (empty if all present)
        """
        missing: list[str] = []
        
        for section in self.REQUIRED_SECTIONS:
            if section not in data:
                missing.append(section)
        
        return missing
    
    def _validate_section_lengths(self, data: dict[str, Any]) -> list[str]:
        """Validate that each section meets minimum length.
        
        Args:
            data: Report data dictionary
            
        Returns:
            List of error messages (empty if validation passes)
        """
        errors: list[str] = []
        
        for section in self.REQUIRED_SECTIONS:
            section_content = data.get(section, "")
            
            if not isinstance(section_content, str):
                errors.append(
                    f"Section '{section}' must be a string, "
                    f"got {type(section_content).__name__}"
                )
                continue
            
            section_length = len(section_content.strip())
            
            if section_length < self.MIN_SECTION_LENGTH:
                errors.append(
                    f"Section '{section}' is too short ({section_length} characters). "
                    f"Minimum {self.MIN_SECTION_LENGTH} characters required."
                )
        
        return errors
    
    def _calculate_total_length(self, data: dict[str, Any]) -> int:
        """Calculate total length of all report sections.
        
        Args:
            data: Report data dictionary
            
        Returns:
            Total character count across all sections
        """
        total = 0
        
        for section in self.REQUIRED_SECTIONS:
            section_content = data.get(section, "")
            if isinstance(section_content, str):
                total += len(section_content.strip())
        
        return total
    
    def _validate_summary_quality(self, data: dict[str, Any]) -> list[str]:
        """Validate executive summary quality.
        
        Checks that the executive summary is meaningful and provides
        useful information. Adds warnings for potential quality issues.
        
        Args:
            data: Report data dictionary
            
        Returns:
            List of warning messages (empty if validation passes)
        """
        warnings: list[str] = []
        
        summary = data.get("executive_summary", "")
        if not isinstance(summary, str):
            return warnings
        
        summary = summary.strip()
        
        # Check if summary is too short (even if meets minimum)
        if 50 <= len(summary) < 100:
            warnings.append(
                "Executive summary is quite short. Consider providing more "
                "detail for better clarity."
            )
        
        # Check for very repetitive content (simple heuristic)
        words = summary.lower().split()
        if len(words) > 0:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.3:  # Less than 30% unique words
                warnings.append(
                    "Executive summary may be too repetitive. Consider "
                    "diversifying the content."
                )
        
        return warnings
    
    @property
    def name(self) -> str:
        """Return validator name.
        
        Returns:
            String identifier for this validator
        """
        return "report_validator"
