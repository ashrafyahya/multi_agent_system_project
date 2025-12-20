"""Tests for validator implementations.

This module contains unit tests for all validator classes to verify
validation logic, error handling, and ValidationResult behavior.
"""

import pytest

from src.graph.validators.base_validator import BaseValidator, ValidationResult


class TestValidationResult:
    """Tests for ValidationResult model."""
    
    def test_validation_result_creation_success(self) -> None:
        """Test creating a successful ValidationResult."""
        result = ValidationResult(is_valid=True)
        
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
    
    def test_validation_result_creation_failure(self) -> None:
        """Test creating a failed ValidationResult."""
        result = ValidationResult(is_valid=False)
        
        assert result.is_valid is False
        assert result.errors == []
        assert result.warnings == []
    
    def test_validation_result_with_errors(self) -> None:
        """Test ValidationResult with error messages."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
        )
        
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert "Error 1" in result.errors
        assert "Error 2" in result.errors
    
    def test_validation_result_with_warnings(self) -> None:
        """Test ValidationResult with warning messages."""
        result = ValidationResult(
            is_valid=True,
            warnings=["Warning 1", "Warning 2"],
        )
        
        assert result.is_valid is True
        assert len(result.warnings) == 2
        assert "Warning 1" in result.warnings
        assert "Warning 2" in result.warnings
    
    def test_add_error(self) -> None:
        """Test adding error messages."""
        result = ValidationResult(is_valid=True)
        
        result.add_error("First error")
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "First error" in result.errors
        
        result.add_error("Second error")
        assert len(result.errors) == 2
        assert "Second error" in result.errors
    
    def test_add_error_sets_is_valid_false(self) -> None:
        """Test that adding an error sets is_valid to False."""
        result = ValidationResult(is_valid=True)
        
        assert result.is_valid is True
        result.add_error("Test error")
        assert result.is_valid is False
    
    def test_add_error_strips_whitespace(self) -> None:
        """Test that add_error strips whitespace from messages."""
        result = ValidationResult(is_valid=True)
        
        result.add_error("  Error message  ")
        assert result.errors[0] == "Error message"
    
    def test_add_error_ignores_empty_strings(self) -> None:
        """Test that add_error ignores empty or whitespace-only strings."""
        result = ValidationResult(is_valid=True)
        
        result.add_error("")
        result.add_error("   ")
        result.add_error("\t\n")
        
        assert len(result.errors) == 0
        assert result.is_valid is True
    
    def test_add_warning(self) -> None:
        """Test adding warning messages."""
        result = ValidationResult(is_valid=True)
        
        result.add_warning("First warning")
        assert result.is_valid is True  # Warnings don't affect validity
        assert len(result.warnings) == 1
        assert "First warning" in result.warnings
        
        result.add_warning("Second warning")
        assert len(result.warnings) == 2
        assert "Second warning" in result.warnings
    
    def test_add_warning_does_not_affect_validity(self) -> None:
        """Test that warnings don't affect is_valid status."""
        result = ValidationResult(is_valid=True)
        
        result.add_warning("Warning message")
        assert result.is_valid is True
        
        result = ValidationResult(is_valid=False)
        result.add_warning("Warning message")
        assert result.is_valid is False  # Still False if already False
    
    def test_add_warning_strips_whitespace(self) -> None:
        """Test that add_warning strips whitespace from messages."""
        result = ValidationResult(is_valid=True)
        
        result.add_warning("  Warning message  ")
        assert result.warnings[0] == "Warning message"
    
    def test_add_warning_ignores_empty_strings(self) -> None:
        """Test that add_warning ignores empty or whitespace-only strings."""
        result = ValidationResult(is_valid=True)
        
        result.add_warning("")
        result.add_warning("   ")
        result.add_warning("\t\n")
        
        assert len(result.warnings) == 0
    
    def test_has_errors(self) -> None:
        """Test has_errors method."""
        result = ValidationResult(is_valid=True)
        assert result.has_errors() is False
        
        result.add_error("Test error")
        assert result.has_errors() is True
    
    def test_has_warnings(self) -> None:
        """Test has_warnings method."""
        result = ValidationResult(is_valid=True)
        assert result.has_warnings() is False
        
        result.add_warning("Test warning")
        assert result.has_warnings() is True
    
    def test_get_summary_success_no_warnings(self) -> None:
        """Test get_summary for successful validation with no warnings."""
        result = ValidationResult.success()
        summary = result.get_summary()
        
        assert "Validation passed" in summary
        assert "error" not in summary.lower()
        assert "warning" not in summary.lower()
    
    def test_get_summary_success_with_warnings(self) -> None:
        """Test get_summary for successful validation with warnings."""
        result = ValidationResult.success()
        result.add_warning("Warning 1")
        result.add_warning("Warning 2")
        summary = result.get_summary()
        
        assert "Validation passed" in summary
        assert "2 warning(s)" in summary
    
    def test_get_summary_failure_no_warnings(self) -> None:
        """Test get_summary for failed validation with no warnings."""
        result = ValidationResult.failure("Error 1", "Error 2")
        summary = result.get_summary()
        
        assert "Validation failed" in summary
        assert "2 error(s)" in summary
        assert "warning" not in summary.lower()
    
    def test_get_summary_failure_with_warnings(self) -> None:
        """Test get_summary for failed validation with warnings."""
        result = ValidationResult.failure("Error 1")
        result.add_warning("Warning 1")
        summary = result.get_summary()
        
        assert "Validation failed" in summary
        assert "1 error(s)" in summary
        assert "1 warning(s)" in summary
    
    def test_success_factory_method(self) -> None:
        """Test success factory method."""
        result = ValidationResult.success()
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_failure_factory_method_single_error(self) -> None:
        """Test failure factory method with single error."""
        result = ValidationResult.failure("Single error")
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Single error" in result.errors
    
    def test_failure_factory_method_multiple_errors(self) -> None:
        """Test failure factory method with multiple errors."""
        result = ValidationResult.failure("Error 1", "Error 2", "Error 3")
        
        assert result.is_valid is False
        assert len(result.errors) == 3
        assert "Error 1" in result.errors
        assert "Error 2" in result.errors
        assert "Error 3" in result.errors
    
    def test_failure_factory_method_no_errors(self) -> None:
        """Test failure factory method with no error messages."""
        result = ValidationResult.failure()
        
        assert result.is_valid is False
        assert len(result.errors) == 0
    
    def test_serialization(self) -> None:
        """Test that ValidationResult can be serialized."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1"],
            warnings=["Warning 1"],
        )
        
        result_dict = result.model_dump()
        assert isinstance(result_dict, dict)
        assert result_dict["is_valid"] is False
        assert result_dict["errors"] == ["Error 1"]
        assert result_dict["warnings"] == ["Warning 1"]
    
    def test_deserialization(self) -> None:
        """Test that ValidationResult can be deserialized."""
        result_data = {
            "is_valid": False,
            "errors": ["Error 1"],
            "warnings": ["Warning 1"],
        }
        
        result = ValidationResult.model_validate(result_data)
        assert result.is_valid is False
        assert result.errors == ["Error 1"]
        assert result.warnings == ["Warning 1"]
    
    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are not allowed."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            ValidationResult(
                is_valid=True,
                extra_field="not allowed",
            )


class TestBaseValidator:
    """Tests for BaseValidator abstract class."""
    
    def test_base_validator_cannot_be_instantiated(self) -> None:
        """Test that BaseValidator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseValidator()  # type: ignore
    
    def test_base_validator_requires_validate_method(self) -> None:
        """Test that subclasses must implement validate method."""
        
        class IncompleteValidator(BaseValidator):
            @property
            def name(self) -> str:
                return "incomplete"
        
        with pytest.raises(TypeError):
            IncompleteValidator()  # type: ignore
    
    def test_base_validator_requires_name_property(self) -> None:
        """Test that subclasses must implement name property."""
        
        class IncompleteValidator(BaseValidator):
            def validate(self, data: dict) -> ValidationResult:
                return ValidationResult.success()
        
        with pytest.raises(TypeError):
            IncompleteValidator()  # type: ignore
    
    def test_complete_validator_implementation(self) -> None:
        """Test a complete validator implementation."""
        
        class TestValidator(BaseValidator):
            @property
            def name(self) -> str:
                return "test_validator"
            
            def validate(self, data: dict) -> ValidationResult:
                result = ValidationResult.success()
                if not data.get("required_field"):
                    result.add_error("required_field is missing")
                return result
        
        validator = TestValidator()
        assert validator.name == "test_validator"
        
        # Test successful validation
        result = validator.validate({"required_field": "value"})
        assert result.is_valid is True
        
        # Test failed validation
        result = validator.validate({})
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "required_field is missing" in result.errors
    
    def test_validator_returns_validation_result(self) -> None:
        """Test that validators return ValidationResult objects."""
        
        class TestValidator(BaseValidator):
            @property
            def name(self) -> str:
                return "test_validator"
            
            def validate(self, data: dict) -> ValidationResult:
                return ValidationResult.success()
        
        validator = TestValidator()
        result = validator.validate({})
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
    
    def test_validator_never_raises_exceptions(self) -> None:
        """Test that validators return ValidationResult instead of raising exceptions."""
        
        class TestValidator(BaseValidator):
            @property
            def name(self) -> str:
                return "test_validator"
            
            def validate(self, data: dict) -> ValidationResult:
                result = ValidationResult.success()
                # Even for invalid data, return ValidationResult with errors
                if not data:
                    result.add_error("Data is empty")
                return result
        
        validator = TestValidator()
        
        # Should not raise exception, even for invalid data
        result = validator.validate({})
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_validator_composability(self) -> None:
        """Test that validators can be composed/chained."""
        
        class Validator1(BaseValidator):
            @property
            def name(self) -> str:
                return "validator1"
            
            def validate(self, data: dict) -> ValidationResult:
                result = ValidationResult.success()
                if not data.get("field1"):
                    result.add_error("field1 missing")
                return result
        
        class Validator2(BaseValidator):
            @property
            def name(self) -> str:
                return "validator2"
            
            def validate(self, data: dict) -> ValidationResult:
                result = ValidationResult.success()
                if not data.get("field2"):
                    result.add_error("field2 missing")
                return result
        
        validator1 = Validator1()
        validator2 = Validator2()
        
        # Chain validators
        data = {"field1": "value1", "field2": "value2"}
        result1 = validator1.validate(data)
        result2 = validator2.validate(data)
        
        assert result1.is_valid is True
        assert result2.is_valid is True
        
        # Test with missing fields
        data = {"field1": "value1"}
        result1 = validator1.validate(data)
        result2 = validator2.validate(data)
        
        assert result1.is_valid is True
        assert result2.is_valid is False
        assert "field2 missing" in result2.errors


class TestCollectorValidator:
    """Tests for CollectorValidator."""
    
    def test_collector_validator_success(self) -> None:
        """Test collector validator with valid data."""
        from src.graph.validators.collector_validator import CollectorValidator
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
                {"name": "Comp2", "source_url": "https://example2.com"},
                {"name": "Comp3", "source_url": "https://example3.com"},
                {"name": "Comp4", "source_url": "https://example4.com"},
            ]
        }
        validator = CollectorValidator()
        result = validator.validate(data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_collector_validator_insufficient_sources(self) -> None:
        """Test collector validator with insufficient sources."""
        from src.graph.validators.collector_validator import CollectorValidator
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
            ]
        }
        validator = CollectorValidator()
        result = validator.validate(data)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("Minimum" in err and "sources required" in err for err in result.errors)
    
    def test_collector_validator_duplicates(self) -> None:
        """Test collector validator detects duplicate competitors."""
        from src.graph.validators.collector_validator import CollectorValidator
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
                {"name": "Comp1", "source_url": "https://example2.com"},  # Duplicate name
            ]
        }
        validator = CollectorValidator()
        result = validator.validate(data)
        
        assert result.is_valid is False
        assert any("Duplicate" in err for err in result.errors)
    
    def test_collector_validator_case_insensitive_duplicates(self) -> None:
        """Test that duplicate detection is case-insensitive."""
        from src.graph.validators.collector_validator import CollectorValidator
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
                {"name": "comp1", "source_url": "https://example2.com"},  # Case variation
            ]
        }
        validator = CollectorValidator()
        result = validator.validate(data)
        
        assert result.is_valid is False
        assert any("Duplicate" in err for err in result.errors)
    
    def test_collector_validator_missing_name(self) -> None:
        """Test collector validator detects missing name field."""
        from src.graph.validators.collector_validator import CollectorValidator
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
                {"source_url": "https://example2.com"},  # Missing name
            ]
        }
        validator = CollectorValidator()
        result = validator.validate(data)
        
        assert result.is_valid is False
        assert any("missing" in err.lower() and "name" in err.lower() for err in result.errors)
    
    def test_collector_validator_missing_source_url(self) -> None:
        """Test collector validator detects missing source_url field."""
        from src.graph.validators.collector_validator import CollectorValidator
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
                {"name": "Comp2"},  # Missing source_url
            ]
        }
        validator = CollectorValidator()
        result = validator.validate(data)
        
        assert result.is_valid is False
        assert any("missing" in err.lower() and "source_url" in err.lower() for err in result.errors)
    
    def test_collector_validator_invalid_source_url(self) -> None:
        """Test collector validator detects invalid source_url format."""
        from src.graph.validators.collector_validator import CollectorValidator
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_url": "not-a-valid-url"},
                {"name": "Comp2", "source_url": "https://example.com"},
                {"name": "Comp3", "source_url": "https://example2.com"},
                {"name": "Comp4", "source_url": "https://example3.com"},
            ]
        }
        validator = CollectorValidator()
        result = validator.validate(data)
        
        assert result.is_valid is False
        assert any("invalid" in err.lower() and "source_url" in err.lower() for err in result.errors)
    
    def test_collector_validator_invalid_website_url(self) -> None:
        """Test collector validator warns about invalid website URL."""
        from src.graph.validators.collector_validator import CollectorValidator
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com", "website": "invalid-url"},
                {"name": "Comp2", "source_url": "https://example2.com"},
                {"name": "Comp3", "source_url": "https://example3.com"},
                {"name": "Comp4", "source_url": "https://example4.com"},
            ]
        }
        validator = CollectorValidator()
        result = validator.validate(data)
        
        # Should still be valid (website is optional), but have warning
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("invalid" in warn.lower() and "website" in warn.lower() for warn in result.warnings)
    
    def test_collector_validator_empty_name(self) -> None:
        """Test collector validator detects empty name."""
        from src.graph.validators.collector_validator import CollectorValidator
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
                {"name": "", "source_url": "https://example2.com"},  # Empty name
            ]
        }
        validator = CollectorValidator()
        result = validator.validate(data)
        
        assert result.is_valid is False
        assert any("empty" in err.lower() or "missing" in err.lower() for err in result.errors)
    
    def test_collector_validator_invalid_data_structure(self) -> None:
        """Test collector validator handles invalid data structure."""
        from src.graph.validators.collector_validator import CollectorValidator
        
        # Not a dictionary
        validator = CollectorValidator()
        result = validator.validate("not a dict")
        assert result.is_valid is False
        assert any("dictionary" in err.lower() for err in result.errors)
        
        # Competitors not a list
        result = validator.validate({"competitors": "not a list"})
        assert result.is_valid is False
        assert any("list" in err.lower() for err in result.errors)
        
        # Competitor not a dictionary
        result = validator.validate({"competitors": ["not a dict"]})
        assert result.is_valid is False
        assert any("dictionary" in err.lower() for err in result.errors)
    
    def test_collector_validator_never_raises_exceptions(self) -> None:
        """Test that collector validator never raises exceptions."""
        from src.graph.validators.collector_validator import CollectorValidator
        
        validator = CollectorValidator()
        
        # Test with various invalid inputs - should never raise
        invalid_inputs = [
            None,
            [],
            "string",
            123,
            {"competitors": None},
            {"competitors": [None]},
            {"competitors": [{"invalid": "data"}]},
        ]
        
        for invalid_input in invalid_inputs:
            result = validator.validate(invalid_input)  # type: ignore
            assert isinstance(result, ValidationResult)
            # Should return ValidationResult, not raise exception
    
    def test_collector_validator_duplicate_source_urls(self) -> None:
        """Test collector validator warns about duplicate source URLs."""
        from src.graph.validators.collector_validator import CollectorValidator
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
                {"name": "Comp2", "source_url": "https://example.com"},  # Same source_url
                {"name": "Comp3", "source_url": "https://example2.com"},
                {"name": "Comp4", "source_url": "https://example3.com"},
            ]
        }
        validator = CollectorValidator()
        result = validator.validate(data)
        
        # Should be valid but have warning
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("duplicate" in warn.lower() and "source_url" in warn.lower() for warn in result.warnings)
    
    def test_collector_validator_exactly_minimum_sources(self) -> None:
        """Test collector validator with exactly minimum sources."""
        from src.graph.validators.collector_validator import CollectorValidator
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
                {"name": "Comp2", "source_url": "https://example2.com"},
                {"name": "Comp3", "source_url": "https://example3.com"},
                {"name": "Comp4", "source_url": "https://example4.com"},
            ]
        }
        validator = CollectorValidator()
        result = validator.validate(data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0


class TestInsightValidator:
    """Tests for InsightValidator."""
    
    def test_insight_validator_success(self) -> None:
        """Test insight validator with valid data."""
        from src.graph.validators.insight_validator import InsightValidator
        
        data = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        validator = InsightValidator()
        result = validator.validate(data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_insight_validator_swot_completeness(self) -> None:
        """Test insight validator checks SWOT completeness."""
        from src.graph.validators.insight_validator import InsightValidator
        
        # Missing weaknesses
        data = {
            "swot": {
                "strengths": ["Strong brand"],
                "weaknesses": [],  # Empty
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": ["Digital transformation"],
        }
        validator = InsightValidator()
        result = validator.validate(data)
        
        assert result.is_valid is False
        assert any("weaknesses" in err.lower() for err in result.errors)
    
    def test_insight_validator_minimum_insights(self) -> None:
        """Test insight validator checks minimum insights."""
        from src.graph.validators.insight_validator import InsightValidator
        
        # Only 2 insights
        data = {
            "swot": {
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": [],
                "threats": [],
            },
            "positioning": "",
            "trends": [],
            "opportunities": [],
        }
        validator = InsightValidator()
        result = validator.validate(data)
        
        assert result.is_valid is False
        assert any("minimum" in err.lower() and "insights" in err.lower() for err in result.errors)
    
    def test_insight_validator_missing_positioning(self) -> None:
        """Test insight validator detects missing positioning."""
        from src.graph.validators.insight_validator import InsightValidator
        
        data = {
            "swot": {
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "",  # Empty
            "trends": ["Digital transformation"],
        }
        validator = InsightValidator()
        result = validator.validate(data)
        
        assert result.is_valid is False
        assert any("positioning" in err.lower() and "required" in err.lower() for err in result.errors)
    
    def test_insight_validator_short_positioning_warning(self) -> None:
        """Test insight validator warns about short positioning."""
        from src.graph.validators.insight_validator import InsightValidator
        
        data = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Short",  # Too short (< 50)
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        validator = InsightValidator()
        result = validator.validate(data)
        
        # Should fail validation (positioning too short)
        assert result.is_valid is False
        assert any("positioning" in err.lower() and "short" in err.lower() for err in result.errors)
    
    def test_insight_validator_trends_coherence(self) -> None:
        """Test insight validator validates trends coherence."""
        from src.graph.validators.insight_validator import InsightValidator
        
        # No trends
        data = {
            "swot": {
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": [],  # No trends
        }
        validator = InsightValidator()
        result = validator.validate(data)
        
        # Should have warning about no trends
        assert any("trend" in warn.lower() for warn in result.warnings)
    
    def test_insight_validator_short_trends_warning(self) -> None:
        """Test insight validator warns about short trends."""
        from src.graph.validators.insight_validator import InsightValidator
        
        data = {
            "swot": {
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": ["AI", "Cloud"],  # Very short
        }
        validator = InsightValidator()
        result = validator.validate(data)
        
        assert any("trend" in warn.lower() and "short" in warn.lower() for warn in result.warnings)
    
    def test_insight_validator_duplicate_trends_warning(self) -> None:
        """Test insight validator warns about duplicate trends."""
        from src.graph.validators.insight_validator import InsightValidator
        
        data = {
            "swot": {
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": ["Digital transformation", "Digital transformation"],  # Duplicate
        }
        validator = InsightValidator()
        result = validator.validate(data)
        
        assert any("duplicate" in warn.lower() and "trend" in warn.lower() for warn in result.warnings)
    
    def test_insight_validator_invalid_data_structure(self) -> None:
        """Test insight validator handles invalid data structure."""
        from src.graph.validators.insight_validator import InsightValidator
        
        validator = InsightValidator()
        
        # Not a dictionary
        result = validator.validate("not a dict")
        assert result.is_valid is False
        assert any("dictionary" in err.lower() for err in result.errors)
        
        # SWOT not a dictionary
        result = validator.validate({"swot": "not a dict"})
        assert result.is_valid is False
        assert any("swot" in err.lower() and "dictionary" in err.lower() for err in result.errors)
        
        # Trends not a list
        result = validator.validate({
            "swot": {
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": "not a list",
        })
        assert result.is_valid is False
        assert any("trends" in err.lower() and "list" in err.lower() for err in result.errors)
    
    def test_insight_validator_never_raises_exceptions(self) -> None:
        """Test that insight validator never raises exceptions."""
        from src.graph.validators.insight_validator import InsightValidator
        
        validator = InsightValidator()
        
        # Test with various invalid inputs - should never raise
        invalid_inputs = [
            None,
            [],
            "string",
            123,
            {"swot": None},
            {"swot": {"strengths": None}},
        ]
        
        for invalid_input in invalid_inputs:
            result = validator.validate(invalid_input)  # type: ignore
            assert isinstance(result, ValidationResult)
            # Should return ValidationResult, not raise exception
    
    def test_insight_validator_counts_insights_correctly(self) -> None:
        """Test that insight validator counts insights correctly."""
        from src.graph.validators.insight_validator import InsightValidator
        
        # Data with enough insights to meet minimum requirements
        data = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",  # +1
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        # Total: 8 SWOT items + 1 positioning + 2 trends + 2 opportunities = 13 insights (more than minimum of 8)
        validator = InsightValidator()
        result = validator.validate(data)
        
        assert result.is_valid is True
    
    def test_insight_validator_empty_swot_items(self) -> None:
        """Test insight validator handles empty SWOT items."""
        from src.graph.validators.insight_validator import InsightValidator
        
        data = {
            "swot": {
                "strengths": ["", "   ", "Valid strength", "Another strength"],  # Empty items should be ignored, need 2+
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        validator = InsightValidator()
        result = validator.validate(data)
        
        # Should count only non-empty items
        assert result.is_valid is True


class TestReportValidator:
    """Tests for ReportValidator."""
    
    def test_report_validator_success(self) -> None:
        """Test report validator with valid data."""
        from src.graph.validators.report_validator import ReportValidator
        
        # Create data with enough characters to meet min_length requirement
        # Executive summary needs at least 200 chars, others need 300 chars, total needs at least 1200
        data = {
            "executive_summary": "This is a comprehensive executive summary of the competitor analysis findings with detailed insights and key observations. The analysis covers multiple market segments and provides strategic recommendations for business growth and competitive positioning in the marketplace. " * 2,
            "swot_breakdown": "The SWOT analysis reveals key strengths, weaknesses, opportunities, and threats in the competitive landscape. This detailed breakdown helps identify strategic priorities and areas for improvement. The analysis covers multiple dimensions including market position, product capabilities, customer relationships, and operational efficiency across different market segments. " * 2,
            "competitor_overview": "The competitor overview provides detailed information about market players and their positioning strategies. This section analyzes competitive dynamics and market structure. It examines how different competitors approach the market, their unique value propositions, target customer segments, and strategic initiatives that drive their competitive advantage. " * 2,
            "recommendations": "Based on the analysis, we recommend strategic actions for market positioning and competitive advantage. These recommendations are designed to enhance market performance and drive sustainable growth. The strategic roadmap includes product development initiatives, market expansion strategies, customer acquisition approaches, and operational improvements. " * 2,
            "min_length": 1200,
        }
        validator = ReportValidator()
        result = validator.validate(data)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_report_validator_missing_section(self) -> None:
        """Test report validator detects missing sections."""
        from src.graph.validators.report_validator import ReportValidator
        
        # Missing recommendations
        data = {
            "executive_summary": "This is a comprehensive executive summary.",
            "swot_breakdown": "The SWOT analysis reveals key insights.",
            "competitor_overview": "The competitor overview provides detailed information.",
            # Missing recommendations
        }
        validator = ReportValidator()
        result = validator.validate(data)
        
        assert result.is_valid is False
        assert any("recommendations" in err.lower() and "missing" in err.lower() for err in result.errors)
    
    def test_report_validator_short_section(self) -> None:
        """Test report validator detects short sections."""
        from src.graph.validators.report_validator import ReportValidator
        
        # Short executive_summary
        data = {
            "executive_summary": "Short",  # Too short
            "swot_breakdown": "The SWOT analysis reveals key strengths, weaknesses, opportunities, and threats.",
            "competitor_overview": "The competitor overview provides detailed information about market players.",
            "recommendations": "Based on the analysis, we recommend strategic actions for market positioning.",
        }
        validator = ReportValidator()
        result = validator.validate(data)
        
        assert result.is_valid is False
        assert any("executive_summary" in err.lower() and "short" in err.lower() for err in result.errors)
    
    def test_report_validator_minimum_total_length(self) -> None:
        """Test report validator checks minimum total length."""
        from src.graph.validators.report_validator import ReportValidator
        
        # Each section exactly 50 chars, total = 200, but min_length = 500
        data = {
            "executive_summary": "A" * 50,
            "swot_breakdown": "B" * 50,
            "competitor_overview": "C" * 50,
            "recommendations": "D" * 50,
            "min_length": 500,
        }
        validator = ReportValidator()
        result = validator.validate(data)
        
        assert result.is_valid is False
        assert any("total" in err.lower() and "length" in err.lower() for err in result.errors)
    
    def test_report_validator_custom_min_length(self) -> None:
        """Test report validator respects custom min_length."""
        from src.graph.validators.report_validator import ReportValidator
        
        # Total length = 1100, custom min_length = 1000 (should pass)
        data = {
            "executive_summary": "A" * 200,
            "swot_breakdown": "B" * 300,
            "competitor_overview": "C" * 300,
            "recommendations": "D" * 300,
            "min_length": 1000,
        }
        validator = ReportValidator()
        result = validator.validate(data)
        
        assert result.is_valid is True
    
    def test_report_validator_short_summary_warning(self) -> None:
        """Test report validator warns about short summary."""
        from src.graph.validators.report_validator import ReportValidator
        
        # Summary exactly 200 chars (minimum, but short - should get warning)
        data = {
            "executive_summary": "A" * 200,
            "swot_breakdown": "B" * 300,
            "competitor_overview": "C" * 300,
            "recommendations": "D" * 300,
            "min_length": 1000,
        }
        validator = ReportValidator()
        result = validator.validate(data)
        
        # Should be valid but have warning
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("summary" in warn.lower() and "short" in warn.lower() for warn in result.warnings)
    
    def test_report_validator_repetitive_summary_warning(self) -> None:
        """Test report validator warns about repetitive summary."""
        from src.graph.validators.report_validator import ReportValidator
        
        # Repetitive summary (same word repeated)
        data = {
            "executive_summary": "word word word word word word word word word word " * 20,  # Repetitive, but meets 200 char minimum
            "swot_breakdown": "B" * 300,
            "competitor_overview": "C" * 300,
            "recommendations": "D" * 300,
            "min_length": 1000,
        }
        validator = ReportValidator()
        result = validator.validate(data)
        
        # Should be valid but have warning
        assert result.is_valid is True
        assert any("repetitive" in warn.lower() for warn in result.warnings)
    
    def test_report_validator_invalid_data_structure(self) -> None:
        """Test report validator handles invalid data structure."""
        from src.graph.validators.report_validator import ReportValidator
        
        validator = ReportValidator()
        
        # Not a dictionary
        result = validator.validate("not a dict")
        assert result.is_valid is False
        assert any("dictionary" in err.lower() for err in result.errors)
        
        # Section not a string
        result = validator.validate({
            "executive_summary": 123,  # Not a string
            "swot_breakdown": "B" * 100,
            "competitor_overview": "C" * 100,
            "recommendations": "D" * 100,
        })
        assert result.is_valid is False
        assert any("executive_summary" in err.lower() and "string" in err.lower() for err in result.errors)
    
    def test_report_validator_never_raises_exceptions(self) -> None:
        """Test that report validator never raises exceptions."""
        from src.graph.validators.report_validator import ReportValidator
        
        validator = ReportValidator()
        
        # Test with various invalid inputs - should never raise
        invalid_inputs = [
            None,
            [],
            "string",
            123,
            {"executive_summary": None},
            {"executive_summary": "", "swot_breakdown": None},
        ]
        
        for invalid_input in invalid_inputs:
            result = validator.validate(invalid_input)  # type: ignore
            assert isinstance(result, ValidationResult)
            # Should return ValidationResult, not raise exception
    
    def test_report_validator_all_sections_required(self) -> None:
        """Test report validator requires all sections."""
        from src.graph.validators.report_validator import ReportValidator
        
        validator = ReportValidator()
        
        # Missing multiple sections
        result = validator.validate({
            "executive_summary": "A" * 100,
            # Missing swot_breakdown, competitor_overview, recommendations
        })
        
        assert result.is_valid is False
        assert len(result.errors) >= 3  # At least 3 missing sections
    
    def test_report_validator_whitespace_handling(self) -> None:
        """Test report validator handles whitespace correctly."""
        from src.graph.validators.report_validator import ReportValidator
        
        # Sections with only whitespace should fail
        data = {
            "executive_summary": "   " * 20,  # Only whitespace
            "swot_breakdown": "B" * 100,
            "competitor_overview": "C" * 100,
            "recommendations": "D" * 100,
        }
        validator = ReportValidator()
        result = validator.validate(data)
        
        assert result.is_valid is False
        assert any("executive_summary" in err.lower() and "short" in err.lower() for err in result.errors)
    
    def test_report_validator_default_min_length(self) -> None:
        """Test report validator uses default min_length when not specified."""
        from src.graph.validators.report_validator import ReportValidator
        
        # Total length = 1100, default min_length = 1200 (should fail)
        data = {
            "executive_summary": "A" * 200,
            "swot_breakdown": "B" * 300,
            "competitor_overview": "C" * 300,
            "recommendations": "D" * 300,
            # No min_length specified, should use default 1200
        }
        validator = ReportValidator()
        result = validator.validate(data)
        
        assert result.is_valid is False
        assert any("1200" in err for err in result.errors)


class TestDataConsistencyValidator:
    """Tests for DataConsistencyValidator."""
    
    def test_data_consistency_validator_name(self) -> None:
        """Test validator name property."""
        from src.graph.validators.data_consistency_validator import DataConsistencyValidator
        
        validator = DataConsistencyValidator()
        assert validator.name == "data_consistency_validator"
    
    def test_data_consistency_validator_valid_data(self) -> None:
        """Test validator with valid, consistent data."""
        from src.graph.validators.data_consistency_validator import DataConsistencyValidator
        
        data = {
            "competitors": [
                {
                    "name": "Competitor A",
                    "market_share": 35.0,
                    "revenue": 2000000000,
                },
                {
                    "name": "Competitor B",
                    "market_share": 25.0,
                    "revenue": 1500000000,
                },
                {
                    "name": "Competitor C",
                    "market_share": 20.0,
                    "revenue": 1000000000,
                },
            ]
        }
        validator = DataConsistencyValidator()
        result = validator.validate(data)
        
        assert result.is_valid is True
        assert len(result.warnings) == 0
    
    def test_data_consistency_validator_low_market_share_sum(self) -> None:
        """Test validator detects low market share sum."""
        from src.graph.validators.data_consistency_validator import DataConsistencyValidator
        
        data = {
            "competitors": [
                {"name": "Competitor A", "market_share": 15.0},
                {"name": "Competitor B", "market_share": 10.0},
            ]
        }
        validator = DataConsistencyValidator()
        result = validator.validate(data)
        
        assert result.is_valid is True  # Warnings don't fail validation
        assert result.has_warnings() is True
        assert any("below the expected minimum" in w for w in result.warnings)
    
    def test_data_consistency_validator_high_market_share_sum(self) -> None:
        """Test validator detects high market share sum."""
        from src.graph.validators.data_consistency_validator import DataConsistencyValidator
        
        data = {
            "competitors": [
                {"name": "Competitor A", "market_share": 60.0},
                {"name": "Competitor B", "market_share": 70.0},
            ]
        }
        validator = DataConsistencyValidator()
        result = validator.validate(data)
        
        assert result.is_valid is True
        assert result.has_warnings() is True
        assert any("exceeds the expected maximum" in w for w in result.warnings)
    
    def test_data_consistency_validator_unrealistic_market_share(self) -> None:
        """Test validator detects unrealistic market share values."""
        from src.graph.validators.data_consistency_validator import DataConsistencyValidator
        
        data = {
            "competitors": [
                {"name": "Competitor A", "market_share": 120.0},  # > 100%
                {"name": "Competitor B", "market_share": -5.0},   # Negative
            ]
        }
        validator = DataConsistencyValidator()
        result = validator.validate(data)
        
        assert result.is_valid is True
        assert result.has_warnings() is True
        assert any("exceeds 100%" in w for w in result.warnings)
        assert any("cannot be negative" in w for w in result.warnings)
    
    def test_data_consistency_validator_revenue_inconsistency(self) -> None:
        """Test validator detects revenue/market share inconsistencies."""
        from src.graph.validators.data_consistency_validator import DataConsistencyValidator
        
        data = {
            "competitors": [
                {"name": "Competitor A", "market_share": 50.0, "revenue": 500000000},  # Higher share but much lower revenue
                {"name": "Competitor B", "market_share": 20.0, "revenue": 2000000000},  # Lower share but much higher revenue
            ]
        }
        validator = DataConsistencyValidator()
        result = validator.validate(data)
        
        assert result.is_valid is True
        # Should detect that Competitor A has higher market share but lower revenue
        assert result.has_warnings() is True
        assert any("revenue inconsistency" in w.lower() for w in result.warnings)
    
    def test_data_consistency_validator_conflicting_data(self) -> None:
        """Test validator detects conflicting data for same competitor."""
        from src.graph.validators.data_consistency_validator import DataConsistencyValidator
        
        data = {
            "competitors": [
                {"name": "Competitor A", "market_share": 35.0, "source_url": "https://source1.com"},
                {"name": "Competitor A", "market_share": 40.0, "source_url": "https://source2.com"},
            ]
        }
        validator = DataConsistencyValidator()
        result = validator.validate(data)
        
        assert result.is_valid is True
        assert result.has_warnings() is True
        assert any("conflicting" in w.lower() and "market share" in w.lower() for w in result.warnings)
    
    def test_data_consistency_validator_empty_data(self) -> None:
        """Test validator with empty competitor list."""
        from src.graph.validators.data_consistency_validator import DataConsistencyValidator
        
        data = {"competitors": []}
        validator = DataConsistencyValidator()
        result = validator.validate(data)
        
        assert result.is_valid is True
        assert len(result.warnings) == 0
    
    def test_data_consistency_validator_no_market_share_data(self) -> None:
        """Test validator with no market share data."""
        from src.graph.validators.data_consistency_validator import DataConsistencyValidator
        
        data = {
            "competitors": [
                {"name": "Competitor A", "revenue": 1000000000},
                {"name": "Competitor B", "revenue": 2000000000},
            ]
        }
        validator = DataConsistencyValidator()
        result = validator.validate(data)
        
        assert result.is_valid is True
        # Should not warn about market share sum if no market share data
        assert not any("market share percentages sum" in w for w in result.warnings)
    
    def test_data_consistency_validator_invalid_data_structure(self) -> None:
        """Test validator handles invalid data structure."""
        from src.graph.validators.data_consistency_validator import DataConsistencyValidator
        
        validator = DataConsistencyValidator()
        
        # Test with non-dict
        result = validator.validate("not a dict")
        assert result.is_valid is False
        assert any("must be a dictionary" in err for err in result.errors)
        
        # Test with non-list competitors
        result = validator.validate({"competitors": "not a list"})
        assert result.is_valid is False
        assert any("must be a list" in err for err in result.errors)
    
    def test_data_consistency_validator_revenue_string_parsing(self) -> None:
        """Test validator parses revenue strings correctly."""
        from src.graph.validators.data_consistency_validator import DataConsistencyValidator
        
        data = {
            "competitors": [
                {"name": "Competitor A", "market_share": 30.0, "revenue": "$1.5B"},
                {"name": "Competitor B", "market_share": 20.0, "revenue": "$500M"},
            ]
        }
        validator = DataConsistencyValidator()
        result = validator.validate(data)
        
        # Should not error on string revenue values
        assert result.is_valid is True