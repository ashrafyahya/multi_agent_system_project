"""Tests for data verification service.

This module contains unit tests for the DataVerificationService to verify
data verification logic and pattern matching accuracy.
"""

import pytest
from unittest.mock import Mock, patch

from src.utils.data_verification_service import DataVerificationService


class TestDataVerificationService:
    """Tests for DataVerificationService class."""
    
    def test_verify_market_share_verified(self) -> None:
        """Test that market share is verified when found in source."""
        service = DataVerificationService()
        
        competitor_data = {"market_share": 35.0}
        source_content = "The company has a 35% market share in the industry."
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["market_share"] == "verified"
    
    def test_verify_market_share_percent_variations(self) -> None:
        """Test that market share with different percent formats is verified."""
        service = DataVerificationService()
        
        competitor_data = {"market_share": 35.0}
        
        # Test with "35 percent"
        source1 = "The company has 35 percent market share."
        status1 = service.verify_quantitative_data(competitor_data, source1)
        assert status1["market_share"] == "verified"
        
        # Test with "35%"
        source2 = "Market share: 35%"
        status2 = service.verify_quantitative_data(competitor_data, source2)
        assert status2["market_share"] == "verified"
    
    def test_verify_market_share_not_found(self) -> None:
        """Test that market share returns not_found when not in source."""
        service = DataVerificationService()
        
        competitor_data = {"market_share": 35.0}
        source_content = "The company is a market leader."
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["market_share"] == "not_found"
    
    def test_verify_revenue_verified(self) -> None:
        """Test that revenue is verified when found in source."""
        service = DataVerificationService()
        
        competitor_data = {"revenue": "$1.5B"}
        source_content = "The company reported revenue of $1.5B last year."
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["revenue"] == "verified"
    
    def test_verify_revenue_billion_format(self) -> None:
        """Test that revenue in billion format is verified."""
        service = DataVerificationService()
        
        competitor_data = {"revenue": "$2.3B"}
        source_content = "Revenue reached $2.3B in 2023."
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["revenue"] == "verified"
    
    def test_verify_revenue_not_found(self) -> None:
        """Test that revenue returns not_found when not in source."""
        service = DataVerificationService()
        
        competitor_data = {"revenue": "$1.5B"}
        source_content = "The company is profitable."
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["revenue"] == "not_found"
    
    def test_verify_user_count_verified(self) -> None:
        """Test that user count is verified when found in source."""
        service = DataVerificationService()
        
        competitor_data = {"user_count": 1000000}
        source_content = "The platform has 1,000,000 active users."
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["user_count"] == "verified"
    
    def test_verify_user_count_million_format(self) -> None:
        """Test that user count in million format is verified."""
        service = DataVerificationService()
        
        competitor_data = {"user_count": 5000000}
        source_content = "The service has 5 million users worldwide."
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["user_count"] == "verified"
    
    def test_verify_founded_year_verified(self) -> None:
        """Test that founded year is verified when found in source."""
        service = DataVerificationService()
        
        competitor_data = {"founded_year": 2010}
        source_content = "The company was founded in 2010."
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["founded_year"] == "verified"
    
    def test_verify_founded_year_established(self) -> None:
        """Test that founded year with 'established' is verified."""
        service = DataVerificationService()
        
        competitor_data = {"founded_year": 2015}
        source_content = "The company was established in 2015."
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["founded_year"] == "verified"
    
    def test_verify_multiple_metrics(self) -> None:
        """Test that multiple metrics are verified correctly."""
        service = DataVerificationService()
        
        competitor_data = {
            "market_share": 35.0,
            "revenue": "$1.5B",
            "user_count": 1000000,
        }
        source_content = (
            "The company has a 35% market share, "
            "reported revenue of $1.5B, "
            "and has 1,000,000 active users."
        )
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["market_share"] == "verified"
        assert status["revenue"] == "verified"
        assert status["user_count"] == "verified"
    
    def test_verify_partial_metrics(self) -> None:
        """Test that only found metrics are verified."""
        service = DataVerificationService()
        
        competitor_data = {
            "market_share": 35.0,
            "revenue": "$1.5B",
        }
        source_content = "The company has a 35% market share."
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["market_share"] == "verified"
        assert status["revenue"] == "not_found"
    
    def test_verify_empty_source_content(self) -> None:
        """Test that empty source content returns not_found."""
        service = DataVerificationService()
        
        competitor_data = {"market_share": 35.0}
        source_content = ""
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["market_share"] == "not_found"
    
    def test_verify_none_source_content(self) -> None:
        """Test that None source content returns not_found."""
        service = DataVerificationService()
        
        competitor_data = {"market_share": 35.0}
        source_content = None  # type: ignore
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["market_share"] == "not_found"
    
    def test_verify_no_quantitative_metrics(self) -> None:
        """Test that non-quantitative fields are ignored."""
        service = DataVerificationService()
        
        competitor_data = {
            "name": "Competitor Inc",
            "products": ["Product A"],
        }
        source_content = "Some content."
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert len(status) == 0
    
    def test_verify_error_handling(self) -> None:
        """Test that errors return not_found for all metrics."""
        service = DataVerificationService()
        
        competitor_data = {"market_share": 35.0}
        
        # Mock _verify_single_metric to raise exception
        with patch.object(
            service,
            "_verify_single_metric",
            side_effect=Exception("Test error")
        ):
            status = service.verify_quantitative_data(
                competitor_data,
                "Some content"
            )
            assert status["market_share"] == "not_found"
    
    def test_classify_data_origin_verified(self) -> None:
        """Test classify_data_origin convenience method."""
        service = DataVerificationService()
        
        source_content = "The company has a 35% market share."
        status = service.classify_data_origin("market_share", 35.0, source_content)
        assert status == "verified"
    
    def test_classify_data_origin_not_found(self) -> None:
        """Test classify_data_origin returns not_found when not found."""
        service = DataVerificationService()
        
        source_content = "The company is a market leader."
        status = service.classify_data_origin("market_share", 35.0, source_content)
        assert status == "not_found"
    
    def test_verify_disabled(self) -> None:
        """Test that verification returns not_found when disabled."""
        service = DataVerificationService()
        service.enabled = False
        
        competitor_data = {"market_share": 35.0}
        source_content = "The company has a 35% market share."
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["market_share"] == "not_found"
    
    def test_verify_case_insensitive(self) -> None:
        """Test that verification is case-insensitive."""
        service = DataVerificationService()
        
        competitor_data = {"market_share": 35.0}
        source_content = "The company has a 35% MARKET SHARE."
        
        status = service.verify_quantitative_data(competitor_data, source_content)
        assert status["market_share"] == "verified"
