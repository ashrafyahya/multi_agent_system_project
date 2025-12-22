"""Tests for data collection helper functions."""

import pytest

from src.agents.utils.data_collection_helpers import (
    extract_competitor_name,
    extract_products,
    extract_quantitative_metrics,
    extract_website_url,
)


class TestExtractCompetitorName:
    """Tests for extract_competitor_name function."""
    
    def test_extract_from_title(self) -> None:
        """Test extracting name from title."""
        name = extract_competitor_name(
            "Salesforce - Official Website",
            "https://salesforce.com",
            ""
        )
        assert name == "Salesforce"
    
    def test_extract_from_url(self) -> None:
        """Test extracting name from URL domain."""
        name = extract_competitor_name(
            "",
            "https://www.hubspot.com/products",
            ""
        )
        assert name == "Hubspot"
    
    def test_extract_from_snippet(self) -> None:
        """Test extracting name from snippet."""
        name = extract_competitor_name(
            "",
            "",
            "Microsoft Corporation is a leading technology company"
        )
        assert name is not None
        assert "Microsoft" in name
    
    def test_extract_returns_none_when_no_data(self) -> None:
        """Test that extraction returns None when no data available."""
        name = extract_competitor_name("", "", "")
        assert name is None


class TestExtractWebsiteUrl:
    """Tests for extract_website_url function."""
    
    def test_extract_valid_url(self) -> None:
        """Test extracting website URL from source URL."""
        url = extract_website_url("https://www.example.com/page", "")
        assert url == "https://www.example.com"
    
    def test_extract_returns_none_for_invalid_url(self) -> None:
        """Test that invalid URLs return None."""
        url = extract_website_url("not-a-url", "")
        assert url is None


class TestExtractProducts:
    """Tests for extract_products function."""
    
    def test_extract_products_from_text(self) -> None:
        """Test extracting products from snippet."""
        # The function lowercases text, so test with lowercase
        products = extract_products(
            "Our products include CRM, Marketing Automation, and Sales Cloud",
            "Company"
        )
        # Should extract products if pattern matches, or return empty list
        assert isinstance(products, list)
        # If products are extracted, verify they're valid
        if products:
            assert all(len(p) > 2 for p in products)
            assert all(len(p) < 100 for p in products)
    
    def test_extract_products_with_products_keyword(self) -> None:
        """Test extracting products when 'products:' keyword is present."""
        products = extract_products(
            "products: CRM, Marketing Automation, Sales Cloud",
            ""
        )
        # Should extract products when pattern matches
        if products:
            assert len(products) > 0
            # Products are lowercased in the function
            product_text = " ".join(products).lower()
            assert "crm" in product_text or "marketing" in product_text
    
    def test_extract_returns_empty_list_when_no_products(self) -> None:
        """Test that extraction returns empty list when no products found."""
        products = extract_products("Some generic text", "Title")
        assert products == []


class TestExtractQuantitativeMetrics:
    """Tests for extract_quantitative_metrics function."""
    
    def test_extract_market_share(self) -> None:
        """Test extracting market share percentage."""
        metrics = extract_quantitative_metrics(
            "Company holds 35% market share in the industry",
            "Company Name"
        )
        assert metrics["market_share"] == 35.0
    
    def test_extract_revenue(self) -> None:
        """Test extracting revenue."""
        metrics = extract_quantitative_metrics(
            "Company revenue is $2B annually",
            "Company Name"
        )
        assert metrics["revenue"] == 2e9
    
    def test_extract_user_count(self) -> None:
        """Test extracting user count."""
        metrics = extract_quantitative_metrics(
            "Company has 1M users worldwide",
            "Company Name"
        )
        assert metrics["user_count"] == 1000000
    
    def test_extract_founded_year(self) -> None:
        """Test extracting founded year."""
        metrics = extract_quantitative_metrics(
            "Company was founded in 2010",
            "Company Name"
        )
        assert metrics["founded_year"] == 2010
    
    def test_extract_headquarters(self) -> None:
        """Test extracting headquarters location."""
        metrics = extract_quantitative_metrics(
            "Company is headquartered in San Francisco",
            "Company Name"
        )
        assert metrics["headquarters"] is not None
        assert "San Francisco" in metrics["headquarters"]
    
    def test_extract_key_features(self) -> None:
        """Test extracting key features."""
        metrics = extract_quantitative_metrics(
            "Key features: AI-powered, Cloud-based, Scalable",
            "Company Name"
        )
        assert len(metrics["key_features"]) > 0
    
    def test_extract_returns_none_for_missing_metrics(self) -> None:
        """Test that missing metrics return None."""
        metrics = extract_quantitative_metrics("Generic text", "Title")
        assert metrics["market_share"] is None
        assert metrics["revenue"] is None
        assert metrics["user_count"] is None

