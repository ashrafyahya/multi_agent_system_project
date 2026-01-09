"""Tests for source quality validator.

This module contains unit tests for the SourceQualityValidator to verify
source quality validation logic and warning generation.
"""

from unittest.mock import Mock, patch

import pytest

from src.graph.validators.source_quality_validator import \
    SourceQualityValidator
from src.models.source_quality import SourceQuality


class TestSourceQualityValidator:
    """Tests for SourceQualityValidator class."""
    
    def test_validate_sufficient_primary_sources(self) -> None:
        """Test that validation passes when sufficient primary sources are present."""
        validator = SourceQualityValidator()
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_quality": SourceQuality.PRIMARY},
                {"name": "Comp2", "source_quality": SourceQuality.PRIMARY},
                {"name": "Comp3", "source_quality": SourceQuality.SECONDARY_HIGH},
            ]
        }
        
        result = validator.validate(data)
        assert result.is_valid is True
        assert len(result.warnings) == 0
    
    def test_validate_insufficient_primary_sources(self) -> None:
        """Test that validation generates warning when insufficient primary sources."""
        validator = SourceQualityValidator()
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_quality": SourceQuality.PRIMARY},
                {"name": "Comp2", "source_quality": SourceQuality.SECONDARY_HIGH},
                {"name": "Comp3", "source_quality": SourceQuality.SECONDARY_MEDIUM},
            ]
        }
        
        result = validator.validate(data)
        assert result.is_valid is False  # Now blocking for insufficient primary sources
        assert len(result.errors) > 0
        assert any("primary source" in error.lower() for error in result.errors)
    
    def test_validate_high_low_quality_ratio(self) -> None:
        """Test that validation generates warning when too many low-quality sources."""
        validator = SourceQualityValidator()
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_quality": SourceQuality.SECONDARY_LOW},
                {"name": "Comp2", "source_quality": SourceQuality.SECONDARY_LOW},
                {"name": "Comp3", "source_quality": SourceQuality.COMMUNITY},
                {"name": "Comp4", "source_quality": SourceQuality.SECONDARY_MEDIUM},
            ]
        }
        
        result = validator.validate(data)
        assert result.is_valid is False  # Blocking due to insufficient primary sources
        assert len(result.errors) > 0
        assert any("primary source" in error.lower() for error in result.errors)
        # Also check for low-quality warning
        assert len(result.warnings) > 0
        assert any("low-quality" in warning.lower() or "low quality" in warning.lower() for warning in result.warnings)
    
    def test_validate_string_source_quality(self) -> None:
        """Test that validation handles string source quality values."""
        validator = SourceQualityValidator()
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_quality": "primary"},
                {"name": "Comp2", "source_quality": "secondary_high"},
            ]
        }
        
        result = validator.validate(data)
        assert result.is_valid is False  # Insufficient primary sources (1 < 2)
    
    def test_validate_missing_source_quality(self) -> None:
        """Test that validation handles missing source quality (backward compatibility)."""
        validator = SourceQualityValidator()
        
        data = {
            "competitors": [
                {"name": "Comp1"},  # No source_quality
                {"name": "Comp2", "source_quality": SourceQuality.PRIMARY},
            ]
        }
        
        result = validator.validate(data)
        assert result.is_valid is False  # Insufficient primary sources (1 < 2)
        # Should only count sources with quality information
    
    def test_validate_empty_competitors(self) -> None:
        """Test that validation handles empty competitors list."""
        validator = SourceQualityValidator()
        
        data = {"competitors": []}
        
        result = validator.validate(data)
        assert result.is_valid is True
        assert len(result.warnings) == 0
    
    def test_validate_invalid_data_structure(self) -> None:
        """Test that validation handles invalid data structure."""
        validator = SourceQualityValidator()
        
        data = "not a dict"
        
        result = validator.validate(data)  # type: ignore
        assert result.is_valid is True  # Non-blocking
        assert len(result.warnings) == 0
    
    def test_validate_invalid_competitors_type(self) -> None:
        """Test that validation handles invalid competitors type."""
        validator = SourceQualityValidator()
        
        data = {"competitors": "not a list"}
        
        result = validator.validate(data)
        assert result.is_valid is True  # Non-blocking
        assert len(result.warnings) == 0
    
    def test_validate_invalid_source_quality_value(self) -> None:
        """Test that validation handles invalid source quality values."""
        validator = SourceQualityValidator()
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_quality": "invalid_quality"},
                {"name": "Comp2", "source_quality": SourceQuality.PRIMARY},
            ]
        }
        
        result = validator.validate(data)
        assert result.is_valid is False  # Insufficient primary sources (1 < 2)
        # Should skip invalid values and only count valid ones
    
    def test_validate_error_handling(self) -> None:
        """Test that validation handles errors gracefully."""
        validator = SourceQualityValidator()
        
        # Mock get_config to raise exception
        with patch("src.graph.validators.source_quality_validator.get_config", side_effect=Exception("Test error")):
            # Should still work if config was already loaded
            data = {
                "competitors": [
                    {"name": "Comp1", "source_quality": SourceQuality.PRIMARY},
                ]
            }
            result = validator.validate(data)
            assert result.is_valid is False  # Insufficient primary sources (1 < 2)
    
    def test_validate_all_quality_levels(self) -> None:
        """Test that validation correctly counts all quality levels."""
        validator = SourceQualityValidator()
        
        data = {
            "competitors": [
                {"name": "Comp1", "source_quality": SourceQuality.PRIMARY},
                {"name": "Comp2", "source_quality": SourceQuality.SECONDARY_HIGH},
                {"name": "Comp3", "source_quality": SourceQuality.SECONDARY_MEDIUM},
                {"name": "Comp4", "source_quality": SourceQuality.SECONDARY_LOW},
                {"name": "Comp5", "source_quality": SourceQuality.COMMUNITY},
            ]
        }
        
        result = validator.validate(data)
        assert result.is_valid is False  # Insufficient primary sources (1 < 2)
        # Should not generate warnings if we have sufficient primary sources
        # (assuming min_primary_sources is 2 or less)
    
    def test_validator_name(self) -> None:
        """Test that validator has correct name."""
        validator = SourceQualityValidator()
        assert validator.name == "SourceQualityValidator"
        validator = SourceQualityValidator()
        assert validator.name == "SourceQualityValidator"
        validator = SourceQualityValidator()
        assert validator.name == "SourceQualityValidator"
