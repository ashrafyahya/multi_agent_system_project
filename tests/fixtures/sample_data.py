"""Sample test data fixtures.

This module contains sample data fixtures used in tests for realistic
but anonymized test scenarios.
"""

from src.models.competitor_profile import CompetitorProfile
from src.models.insight_model import Insight, SWOT
from src.models.plan_model import Plan


def sample_plan() -> dict:
    """Create a sample execution plan."""
    return {
        "tasks": [
            "Find top competitors in SaaS market",
            "Analyze pricing strategies",
            "Compare feature sets",
        ],
        "preferred_sources": ["web search", "official sites"],
        "minimum_results": 4,
        "search_strategy": "comprehensive",
    }


def sample_collected_data() -> dict:
    """Create sample collected competitor data."""
    return {
        "competitors": [
            {
                "name": "Competitor A",
                "source_url": "https://competitor-a.com",
                "website": "https://competitor-a.com",
                "description": "Leading SaaS provider",
                "products": ["Product 1", "Product 2"],
                "pricing": "Starting at $99/month",
                "market_share": 35.0,
                "revenue": 2000000000,
                "user_count": 1000000,
                "founded_year": 2010,
                "headquarters": "San Francisco",
                "key_features": ["Feature 1", "Feature 2", "Feature 3"],
            },
            {
                "name": "Competitor B",
                "source_url": "https://competitor-b.com",
                "website": "https://competitor-b.com",
                "description": "Enterprise SaaS solution",
                "products": ["Enterprise Suite"],
                "pricing": "Custom pricing",
                "market_share": 20.0,
                "revenue": 1500000000,
                "user_count": 500000,
                "founded_year": 2012,
                "headquarters": "New York",
                "key_features": ["Enterprise Feature 1", "Enterprise Feature 2"],
            },
            {
                "name": "Competitor C",
                "source_url": "https://competitor-c.com",
                "website": "https://competitor-c.com",
                "description": "Mid-market SaaS platform",
                "products": ["Platform"],
                "pricing": "Starting at $49/month",
                "market_share": 15.0,
                "revenue": 500000000,
                "user_count": 250000,
                "founded_year": 2015,
                "headquarters": "Austin",
                "key_features": ["Platform Feature 1"],
            },
            {
                "name": "Competitor D",
                "source_url": "https://competitor-d.com",
                "website": "https://competitor-d.com",
                "description": "Startup-focused SaaS",
                "products": ["Starter", "Pro"],
                "pricing": "Free tier available",
                "market_share": 10.0,
                "revenue": 200000000,
                "user_count": 100000,
                "founded_year": 2018,
                "headquarters": "Seattle",
                "key_features": ["Starter Feature", "Pro Feature"],
            },
        ]
    }


def sample_insights() -> dict:
    """Create sample business insights."""
    return {
        "swot": {
            "strengths": [
                "Strong market presence",
                "Comprehensive feature set",
                "Established customer base",
            ],
            "weaknesses": [
                "Higher pricing than competitors",
                "Complex onboarding process",
            ],
            "opportunities": [
                "Emerging markets",
                "AI integration",
                "Mobile expansion",
            ],
            "threats": [
                "New market entrants",
                "Price competition",
                "Technology disruption",
            ],
        },
        "positioning": "Premium enterprise-focused SaaS provider with comprehensive features, strong market presence, and established customer relationships in the competitive software-as-a-service industry",
        "trends": [
            "AI and automation integration",
            "Mobile-first approach",
            "Subscription model evolution",
        ],
        "opportunities": [
            "Expand into SMB market",
            "Develop AI-powered features",
            "Partner with integrators",
        ],
    }


def sample_report() -> str:
    """Create a sample formatted report."""
    return """## Executive Summary

This competitor analysis provides a comprehensive overview of the SaaS market landscape, identifying key competitors, their strategies, and market positioning. The analysis reveals a competitive market with opportunities for differentiation and growth.

## SWOT Breakdown

**Strengths:**
- Strong market presence and brand recognition
- Comprehensive feature set addressing enterprise needs
- Established customer base with high retention

**Weaknesses:**
- Higher pricing compared to competitors
- Complex onboarding process limiting adoption

**Opportunities:**
- Emerging markets with growing demand
- AI integration for competitive advantage
- Mobile expansion to reach new segments

**Threats:**
- New market entrants with innovative solutions
- Price competition from lower-cost alternatives
- Technology disruption from emerging platforms

## Competitor Overview

The market consists of four main competitors:
1. Competitor A - Leading SaaS provider with broad market coverage
2. Competitor B - Enterprise-focused solution with custom pricing
3. Competitor C - Mid-market platform with competitive pricing
4. Competitor D - Startup-focused solution with free tier

## Recommendations

1. **Pricing Strategy**: Consider introducing a mid-tier option to compete with Competitor C
2. **Market Expansion**: Target SMB segment currently served by Competitor D
3. **Innovation**: Invest in AI-powered features to differentiate from competitors
4. **Partnerships**: Develop integration partnerships to improve market reach
"""


def sample_llm_response_plan() -> str:
    """Sample LLM response for plan generation."""
    return """{
    "tasks": [
        "Find top competitors in SaaS market",
        "Analyze pricing strategies",
        "Compare feature sets"
    ],
    "preferred_sources": ["web search", "official sites"],
    "minimum_results": 4,
    "search_strategy": "comprehensive"
}"""


def sample_llm_response_insights() -> str:
    """Sample LLM response for insight generation."""
    return """{
    "swot": {
        "strengths": ["Strong market presence", "Comprehensive feature set"],
        "weaknesses": ["Higher pricing", "Complex onboarding"],
        "opportunities": ["Emerging markets", "AI integration"],
        "threats": ["New entrants", "Price competition"]
    },
    "positioning": "Premium enterprise-focused SaaS provider",
    "trends": ["AI integration", "Mobile-first approach"],
    "opportunities": ["SMB expansion", "AI features"]
}"""


def sample_llm_response_report() -> str:
    """Sample LLM response for report generation."""
    return """{
    "executive_summary": "This competitor analysis provides a comprehensive overview...",
    "swot_breakdown": "**Strengths:** Strong market presence...",
    "competitor_overview": "The market consists of four main competitors...",
    "recommendations": "1. Pricing Strategy: Consider introducing...",
    "methodology": "Data was collected through web search and scraping from multiple sources...",
    "sources": ["https://competitor-a.com", "https://competitor-b.com", "https://competitor-c.com", "https://competitor-d.com"]
}"""


def sample_collected_data_with_inconsistencies() -> dict:
    """Create sample collected data with data consistency issues for testing."""
    return {
        "competitors": [
            {
                "name": "Competitor A",
                "source_url": "https://competitor-a.com",
                "market_share": 50.0,  # High market share
                "revenue": 1000000000,  # But lower revenue
            },
            {
                "name": "Competitor B",
                "source_url": "https://competitor-b.com",
                "market_share": 30.0,
                "revenue": 2000000000,  # Higher revenue but lower market share
            },
            {
                "name": "Competitor C",
                "source_url": "https://competitor-c.com",
                "market_share": 120.0,  # Unrealistic (>100%)
            },
            {
                "name": "Competitor D",
                "source_url": "https://competitor-d.com",
                "market_share": -5.0,  # Negative value
            },
        ]
    }


def sample_collected_data_low_market_share_sum() -> dict:
    """Create sample data with low market share sum for testing."""
    return {
        "competitors": [
            {
                "name": "Competitor A",
                "source_url": "https://competitor-a.com",
                "market_share": 15.0,
            },
            {
                "name": "Competitor B",
                "source_url": "https://competitor-b.com",
                "market_share": 10.0,
            },
        ]
    }
