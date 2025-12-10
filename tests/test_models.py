"""Tests for data model implementations.

This module contains unit tests for all Pydantic models to verify
validation, serialization, and field constraints.
"""

import pytest
from pydantic import ValidationError

from src.models.competitor_profile import CompetitorProfile
from src.models.insight_model import Insight, SWOT
from src.models.plan_model import Plan
from src.models.report_model import Report


class TestPlanModel:
    """Tests for Plan model."""
    
    def test_plan_creation_with_all_fields(self) -> None:
        """Test creating a Plan with all fields."""
        plan = Plan(
            tasks=["task1", "task2", "task3"],
            preferred_sources=["web search", "official site"],
            minimum_results=6,
            search_strategy="comprehensive",
        )
        
        assert len(plan.tasks) == 3
        assert plan.tasks == ["task1", "task2", "task3"]
        assert plan.preferred_sources == ["web search", "official site"]
        assert plan.minimum_results == 6
        assert plan.search_strategy == "comprehensive"
    
    def test_plan_creation_with_defaults(self) -> None:
        """Test creating a Plan with default values."""
        plan = Plan(tasks=["task1"])
        
        assert plan.tasks == ["task1"]
        assert plan.preferred_sources == []
        assert plan.minimum_results == 4  # Default
        assert plan.search_strategy == "comprehensive"  # Default
    
    def test_plan_validation_minimum_results_valid(self) -> None:
        """Test that minimum_results validation works correctly."""
        plan = Plan(tasks=["task1"], minimum_results=1)
        assert plan.minimum_results == 1
        
        plan = Plan(tasks=["task1"], minimum_results=100)
        assert plan.minimum_results == 100
    
    def test_plan_validation_minimum_results_invalid(self) -> None:
        """Test that invalid minimum_results raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Plan(tasks=["task1"], minimum_results=0)
        
        assert "greater than or equal to 1" in str(exc_info.value).lower()
        
        with pytest.raises(ValidationError) as exc_info:
            Plan(tasks=["task1"], minimum_results=101)
        
        assert "less than or equal to 100" in str(exc_info.value).lower()
    
    def test_plan_validation_empty_tasks(self) -> None:
        """Test that empty tasks list raises ValidationError."""
        with pytest.raises(ValidationError):
            Plan(tasks=[])
    
    def test_plan_validation_tasks_with_empty_strings(self) -> None:
        """Test that tasks with only empty strings raises ValidationError."""
        with pytest.raises(ValidationError):
            Plan(tasks=["", "   ", "\t"])
    
    def test_plan_validation_tasks_stripped(self) -> None:
        """Test that task strings are stripped of whitespace."""
        plan = Plan(tasks=["  task1  ", "  task2  "])
        assert plan.tasks == ["task1", "task2"]
    
    def test_plan_serialization(self) -> None:
        """Test that Plan can be serialized to dict."""
        plan = Plan(tasks=["task1"], minimum_results=5)
        plan_dict = plan.model_dump()
        
        assert isinstance(plan_dict, dict)
        assert plan_dict["tasks"] == ["task1"]
        assert plan_dict["minimum_results"] == 5
    
    def test_plan_deserialization(self) -> None:
        """Test that Plan can be deserialized from dict."""
        plan_data = {
            "tasks": ["task1", "task2"],
            "minimum_results": 6,
        }
        plan = Plan.model_validate(plan_data)
        
        assert plan.tasks == ["task1", "task2"]
        assert plan.minimum_results == 6


class TestCompetitorProfileModel:
    """Tests for CompetitorProfile model."""
    
    def test_competitor_profile_creation(self) -> None:
        """Test creating a CompetitorProfile with all fields."""
        profile = CompetitorProfile(
            name="Competitor Inc",
            website="https://competitor.com",
            products=["Product A", "Product B"],
            pricing={"base": "$99/month"},
            market_presence="Global leader",
            source_url="https://source.com/article",
        )
        
        assert profile.name == "Competitor Inc"
        assert str(profile.website) == "https://competitor.com/"
        assert profile.products == ["Product A", "Product B"]
        assert profile.pricing == {"base": "$99/month"}
        assert profile.market_presence == "Global leader"
        assert str(profile.source_url) == "https://source.com/article"
    
    def test_competitor_profile_minimal(self) -> None:
        """Test creating a CompetitorProfile with minimal required fields."""
        profile = CompetitorProfile(
            name="Competitor Inc",
            source_url="https://source.com/article",
        )
        
        assert profile.name == "Competitor Inc"
        assert profile.website is None
        assert profile.products == []
        assert profile.pricing is None
        assert profile.market_presence is None
    
    def test_competitor_profile_validation_empty_name(self) -> None:
        """Test that empty name raises ValidationError."""
        with pytest.raises(ValidationError):
            CompetitorProfile(
                name="",
                source_url="https://source.com/article",
            )
    
    def test_competitor_profile_validation_invalid_url(self) -> None:
        """Test that invalid URL raises ValidationError."""
        with pytest.raises(ValidationError):
            CompetitorProfile(
                name="Competitor Inc",
                source_url="not-a-valid-url",
            )
    
    def test_competitor_profile_products_stripped(self) -> None:
        """Test that product strings are stripped."""
        profile = CompetitorProfile(
            name="Competitor Inc",
            products=["  Product A  ", "  Product B  ", ""],
            source_url="https://source.com/article",
        )
        
        assert profile.products == ["Product A", "Product B"]


class TestSWOTModel:
    """Tests for SWOT model."""
    
    def test_swot_creation(self) -> None:
        """Test creating a SWOT with all fields."""
        swot = SWOT(
            strengths=["Strong brand", "Market leader"],
            weaknesses=["High prices"],
            opportunities=["Emerging markets"],
            threats=["New competitors"],
        )
        
        assert swot.strengths == ["Strong brand", "Market leader"]
        assert swot.weaknesses == ["High prices"]
        assert swot.opportunities == ["Emerging markets"]
        assert swot.threats == ["New competitors"]
    
    def test_swot_creation_empty(self) -> None:
        """Test creating a SWOT with empty lists."""
        swot = SWOT()
        
        assert swot.strengths == []
        assert swot.weaknesses == []
        assert swot.opportunities == []
        assert swot.threats == []
    
    def test_swot_items_stripped(self) -> None:
        """Test that SWOT items are stripped."""
        swot = SWOT(
            strengths=["  Strength 1  ", "  Strength 2  ", ""],
            weaknesses=["  Weakness 1  "],
        )
        
        assert swot.strengths == ["Strength 1", "Strength 2"]
        assert swot.weaknesses == ["Weakness 1"]


class TestInsightModel:
    """Tests for Insight model."""
    
    def test_insight_creation(self) -> None:
        """Test creating an Insight with all fields."""
        swot = SWOT(
            strengths=["Strong brand"],
            weaknesses=["High prices"],
        )
        
        insight = Insight(
            swot=swot,
            positioning="Premium market leader",
            trends=["Digital transformation"],
            opportunities=["Expansion into Asia"],
        )
        
        assert insight.swot == swot
        assert insight.positioning == "Premium market leader"
        assert insight.trends == ["Digital transformation"]
        assert insight.opportunities == ["Expansion into Asia"]
    
    def test_insight_validation_empty_positioning(self) -> None:
        """Test that empty positioning raises ValidationError."""
        swot = SWOT()
        
        with pytest.raises(ValidationError):
            Insight(swot=swot, positioning="")
    
    def test_insight_trends_stripped(self) -> None:
        """Test that trend strings are stripped."""
        swot = SWOT()
        insight = Insight(
            swot=swot,
            positioning="Test positioning",
            trends=["  Trend 1  ", "  Trend 2  ", ""],
        )
        
        assert insight.trends == ["Trend 1", "Trend 2"]


class TestReportModel:
    """Tests for Report model."""
    
    def test_report_creation(self) -> None:
        """Test creating a Report with all fields."""
        report = Report(
            executive_summary="Summary of competitor analysis...",
            swot_breakdown="Detailed SWOT analysis...",
            competitor_overview="Overview of competitors...",
            recommendations="Strategic recommendations...",
            min_length=500,
        )
        
        assert len(report.executive_summary) >= 50
        assert len(report.swot_breakdown) >= 50
        assert len(report.competitor_overview) >= 50
        assert len(report.recommendations) >= 50
        assert report.min_length == 500
    
    def test_report_validation_min_length_default(self) -> None:
        """Test that min_length defaults to 500."""
        report = Report(
            executive_summary="x" * 200,
            swot_breakdown="x" * 200,
            competitor_overview="x" * 200,
            recommendations="x" * 200,
        )
        
        assert report.min_length == 500
    
    def test_report_validation_total_length_sufficient(self) -> None:
        """Test that report with sufficient total length passes."""
        report = Report(
            executive_summary="x" * 200,
            swot_breakdown="x" * 200,
            competitor_overview="x" * 200,
            recommendations="x" * 200,
            min_length=500,
        )
        
        # Should not raise error
        assert report.min_length == 500
    
    def test_report_validation_total_length_insufficient(self) -> None:
        """Test that report with insufficient total length raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Report(
                executive_summary="x" * 50,
                swot_breakdown="x" * 50,
                competitor_overview="x" * 50,
                recommendations="x" * 50,
                min_length=500,
            )
        
        assert "minimum required length" in str(exc_info.value).lower()
    
    def test_report_validation_empty_section(self) -> None:
        """Test that empty section raises ValidationError."""
        with pytest.raises(ValidationError):
            Report(
                executive_summary="",
                swot_breakdown="x" * 200,
                competitor_overview="x" * 200,
                recommendations="x" * 200,
            )
    
    def test_report_validation_min_length_range(self) -> None:
        """Test that min_length must be in valid range."""
        # Too low
        with pytest.raises(ValidationError):
            Report(
                executive_summary="x" * 200,
                swot_breakdown="x" * 200,
                competitor_overview="x" * 200,
                recommendations="x" * 200,
                min_length=50,  # Below minimum
            )
        
        # Too high
        with pytest.raises(ValidationError):
            Report(
                executive_summary="x" * 200,
                swot_breakdown="x" * 200,
                competitor_overview="x" * 200,
                recommendations="x" * 200,
                min_length=20000,  # Above maximum
            )


class TestModelSerialization:
    """Tests for model serialization and deserialization."""
    
    def test_all_models_serializable(self) -> None:
        """Test that all models can be serialized to dict."""
        plan = Plan(tasks=["task1"])
        plan_dict = plan.model_dump()
        assert isinstance(plan_dict, dict)
        
        profile = CompetitorProfile(
            name="Test",
            source_url="https://example.com",
        )
        profile_dict = profile.model_dump()
        assert isinstance(profile_dict, dict)
        
        swot = SWOT(strengths=["Strength"])
        swot_dict = swot.model_dump()
        assert isinstance(swot_dict, dict)
        
        insight = Insight(swot=swot, positioning="Test")
        insight_dict = insight.model_dump()
        assert isinstance(insight_dict, dict)
        
        report = Report(
            executive_summary="x" * 200,
            swot_breakdown="x" * 200,
            competitor_overview="x" * 200,
            recommendations="x" * 200,
        )
        report_dict = report.model_dump()
        assert isinstance(report_dict, dict)
    
    def test_all_models_deserializable(self) -> None:
        """Test that all models can be deserialized from dict."""
        plan_data = {"tasks": ["task1"], "minimum_results": 5}
        plan = Plan.model_validate(plan_data)
        assert isinstance(plan, Plan)
        
        profile_data = {
            "name": "Test",
            "source_url": "https://example.com",
        }
        profile = CompetitorProfile.model_validate(profile_data)
        assert isinstance(profile, CompetitorProfile)
        
        swot_data = {"strengths": ["Strength"]}
        swot = SWOT.model_validate(swot_data)
        assert isinstance(swot, SWOT)
        
        insight_data = {
            "swot": swot_data,
            "positioning": "Test",
        }
        insight = Insight.model_validate(insight_data)
        assert isinstance(insight, Insight)


