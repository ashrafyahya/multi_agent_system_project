"""Tests for data model implementations.

This module contains unit tests for all Pydantic models to verify
validation, serialization, and field constraints.
"""

import pytest
from pydantic import ValidationError

from pathlib import Path

from src.models.competitor_profile import CompetitorProfile
from src.models.insight_model import Insight, SWOT
from src.models.pdf_branding_config import PDFBrandingConfig
from src.models.pdf_layout_config import PDFLayoutConfig
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
            strengths=["Strong brand", "Market leader"],
            weaknesses=["High prices", "Limited distribution"],
        )
        
        insight = Insight(
            swot=swot,
            positioning="Premium market leader in the SaaS industry with strong brand recognition and customer loyalty",
            trends=["Digital transformation", "AI integration"],
            opportunities=["Expansion into Asia", "B2B market growth"],
        )
        
        assert insight.swot == swot
        assert len(insight.positioning) >= 50
        assert insight.trends == ["Digital transformation", "AI integration"]
        assert insight.opportunities == ["Expansion into Asia", "B2B market growth"]
    
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
            positioning="This is a comprehensive test positioning statement that meets the minimum length requirement of fifty characters for validation purposes",
            trends=["  Trend 1  ", "  Trend 2  ", ""],
        )
        
        assert insight.trends == ["Trend 1", "Trend 2"]


class TestReportModel:
    """Tests for Report model."""
    
    def test_report_creation(self) -> None:
        """Test creating a Report with all fields."""
        report = Report(
            executive_summary="This is a comprehensive executive summary of the competitor analysis findings with detailed insights and key observations. The analysis covers multiple market segments and provides strategic recommendations for business growth and competitive positioning in the marketplace.",
            swot_breakdown="The SWOT analysis reveals key strengths, weaknesses, opportunities, and threats in the competitive landscape. This detailed breakdown helps identify strategic priorities and areas for improvement. The analysis covers multiple dimensions including market position, product capabilities, customer relationships, and operational efficiency across different market segments.",
            competitor_overview="The competitor overview provides detailed information about market players and their positioning strategies. This section analyzes competitive dynamics and market structure. It examines how different competitors approach the market, their unique value propositions, target customer segments, and strategic initiatives that drive their competitive advantage.",
            recommendations="Based on the analysis, we recommend strategic actions for market positioning and competitive advantage. These recommendations are designed to enhance market performance and drive sustainable growth. The strategic roadmap includes product development initiatives, market expansion strategies, customer acquisition approaches, and operational improvements.",
            min_length=1200,
        )
        
        assert len(report.executive_summary) >= 200
        assert len(report.swot_breakdown) >= 300
        assert len(report.competitor_overview) >= 300
        assert len(report.recommendations) >= 300
        assert report.min_length == 1200
    
    def test_report_validation_min_length_default(self) -> None:
        """Test that min_length defaults to 1200."""
        report = Report(
            executive_summary="x" * 300,
            swot_breakdown="x" * 300,
            competitor_overview="x" * 300,
            recommendations="x" * 300,
        )
        
        assert report.min_length == 1200
    
    def test_report_validation_total_length_sufficient(self) -> None:
        """Test that report with sufficient total length passes."""
        report = Report(
            executive_summary="x" * 200,
            swot_breakdown="x" * 300,
            competitor_overview="x" * 300,
            recommendations="x" * 300,
            min_length=1000,
        )
        
        # Should not raise error
        assert report.min_length == 1000
    
    def test_report_validation_total_length_insufficient(self) -> None:
        """Test that report with insufficient total length raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Report(
                executive_summary="x" * 200,
                swot_breakdown="x" * 300,
                competitor_overview="x" * 300,
                recommendations="x" * 300,  # All sections meet minimums, but total: 1100 < 1200
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
        
        swot = SWOT(strengths=["Strength 1", "Strength 2"])
        swot_dict = swot.model_dump()
        assert isinstance(swot_dict, dict)
        
        insight = Insight(
            swot=swot,
            positioning="This is a comprehensive test positioning statement that meets the minimum length requirement of fifty characters for validation purposes"
        )
        insight_dict = insight.model_dump()
        assert isinstance(insight_dict, dict)
        
        report = Report(
            executive_summary="x" * 300,
            swot_breakdown="x" * 300,
            competitor_overview="x" * 300,
            recommendations="x" * 300,
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
        
        swot_data = {"strengths": ["Strength 1", "Strength 2"]}
        swot = SWOT.model_validate(swot_data)
        assert isinstance(swot, SWOT)
        
        insight_data = {
            "swot": swot_data,
            "positioning": "This is a comprehensive test positioning statement that meets the minimum length requirement of fifty characters for validation purposes",
        }
        insight = Insight.model_validate(insight_data)
        assert isinstance(insight, Insight)


class TestPDFBrandingConfig:
    """Tests for PDFBrandingConfig model."""
    
    def test_branding_config_creation_with_all_fields(self) -> None:
        """Test creating a PDFBrandingConfig with all fields."""
        branding = PDFBrandingConfig(
            company_name="Acme Corp",
            primary_color="#1a1a1a",
            secondary_color="#0066cc",
            accent_color="#ff6600",
            font_family="Helvetica",
            header_font="Helvetica-Bold",
            footer_text="Confidential",
            watermark="Draft",
            cover_page_template="executive",
        )
        
        assert branding.company_name == "Acme Corp"
        assert branding.primary_color == "#1a1a1a"
        assert branding.secondary_color == "#0066cc"
        assert branding.accent_color == "#ff6600"
        assert branding.font_family == "Helvetica"
        assert branding.header_font == "Helvetica-Bold"
        assert branding.footer_text == "Confidential"
        assert branding.watermark == "Draft"
        assert branding.cover_page_template == "executive"
    
    def test_branding_config_creation_with_defaults(self) -> None:
        """Test creating a PDFBrandingConfig with default values."""
        branding = PDFBrandingConfig(company_name="Test Company")
        
        assert branding.company_name == "Test Company"
        assert branding.company_logo_path is None
        assert branding.primary_color == "#1a1a1a"  # Default
        assert branding.secondary_color == "#0066cc"  # Default
        assert branding.accent_color == "#ff6600"  # Default
        assert branding.font_family == "Helvetica"  # Default
        assert branding.header_font == "Helvetica-Bold"  # Default
        assert branding.footer_text is None
        assert branding.watermark is None
        assert branding.cover_page_template == "default"  # Default
    
    def test_branding_config_validation_empty_company_name(self) -> None:
        """Test that empty company name raises ValidationError."""
        with pytest.raises(ValidationError):
            PDFBrandingConfig(company_name="")
        
        with pytest.raises(ValidationError):
            PDFBrandingConfig(company_name="   ")
    
    def test_branding_config_validation_hex_colors(self) -> None:
        """Test that hex color validation works correctly."""
        # Valid 6-digit hex colors
        branding = PDFBrandingConfig(
            company_name="Test",
            primary_color="#1a1a1a",
            secondary_color="#0066CC",
            accent_color="#FF6600",
        )
        assert branding.primary_color == "#1a1a1a"
        assert branding.secondary_color == "#0066cc"  # Lowercased
        assert branding.accent_color == "#ff6600"  # Lowercased
        
        # Valid 3-digit hex colors
        branding = PDFBrandingConfig(
            company_name="Test",
            primary_color="#1a1",
            secondary_color="#06c",
        )
        assert branding.primary_color == "#1a1"
        assert branding.secondary_color == "#06c"
    
    def test_branding_config_validation_invalid_hex_colors(self) -> None:
        """Test that invalid hex colors raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PDFBrandingConfig(company_name="Test", primary_color="not-a-color")
        assert "hex format" in str(exc_info.value).lower()
        
        with pytest.raises(ValidationError):
            PDFBrandingConfig(company_name="Test", primary_color="#gggggg")
        
        with pytest.raises(ValidationError):
            PDFBrandingConfig(company_name="Test", primary_color="#12345")  # Wrong length
        
        with pytest.raises(ValidationError):
            PDFBrandingConfig(company_name="Test", primary_color="123456")  # Missing #
    
    def test_branding_config_validation_logo_path_none(self) -> None:
        """Test that None logo path is allowed."""
        branding = PDFBrandingConfig(company_name="Test", company_logo_path=None)
        assert branding.company_logo_path is None
    
    def test_branding_config_validation_logo_path_string(self) -> None:
        """Test that logo path can be provided as string."""
        # Create a temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            branding = PDFBrandingConfig(
                company_name="Test",
                company_logo_path=str(tmp_path),
            )
            assert branding.company_logo_path == tmp_path
        finally:
            tmp_path.unlink()
    
    def test_branding_config_validation_logo_path_not_exists(self) -> None:
        """Test that non-existent logo path raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PDFBrandingConfig(
                company_name="Test",
                company_logo_path=Path("/nonexistent/path/logo.png"),
            )
        assert "does not exist" in str(exc_info.value).lower()
    
    def test_branding_config_validation_font_family_empty(self) -> None:
        """Test that empty font family raises ValidationError."""
        with pytest.raises(ValidationError):
            PDFBrandingConfig(company_name="Test", font_family="")
        
        with pytest.raises(ValidationError):
            PDFBrandingConfig(company_name="Test", font_family="   ")
    
    def test_branding_config_validation_template_options(self) -> None:
        """Test that template validation works correctly."""
        valid_templates = ["default", "executive", "minimal"]
        for template in valid_templates:
            branding = PDFBrandingConfig(
                company_name="Test",
                cover_page_template=template,
            )
            assert branding.cover_page_template == template
    
    def test_branding_config_validation_template_invalid(self) -> None:
        """Test that invalid template raises ValidationError."""
        with pytest.raises(ValidationError):
            PDFBrandingConfig(company_name="Test", cover_page_template="invalid")
    
    def test_branding_config_validation_footer_text_empty_string(self) -> None:
        """Test that empty string footer_text raises ValidationError."""
        with pytest.raises(ValidationError):
            PDFBrandingConfig(company_name="Test", footer_text="")
        
        with pytest.raises(ValidationError):
            PDFBrandingConfig(company_name="Test", footer_text="   ")
    
    def test_branding_config_validation_footer_text_none(self) -> None:
        """Test that None footer_text is allowed."""
        branding = PDFBrandingConfig(company_name="Test", footer_text=None)
        assert branding.footer_text is None
    
    def test_branding_config_company_name_stripped(self) -> None:
        """Test that company name is stripped."""
        branding = PDFBrandingConfig(company_name="  Test Company  ")
        assert branding.company_name == "Test Company"
    
    def test_branding_config_serialization(self) -> None:
        """Test that PDFBrandingConfig can be serialized to dict."""
        branding = PDFBrandingConfig(company_name="Test Company")
        branding_dict = branding.model_dump()
        
        assert isinstance(branding_dict, dict)
        assert branding_dict["company_name"] == "Test Company"
        assert branding_dict["primary_color"] == "#1a1a1a"
    
    def test_branding_config_deserialization(self) -> None:
        """Test that PDFBrandingConfig can be deserialized from dict."""
        branding_data = {
            "company_name": "Test Company",
            "primary_color": "#ff0000",
            "cover_page_template": "executive",
        }
        branding = PDFBrandingConfig.model_validate(branding_data)
        
        assert branding.company_name == "Test Company"
        assert branding.primary_color == "#ff0000"
        assert branding.cover_page_template == "executive"


class TestPDFLayoutConfig:
    """Tests for PDFLayoutConfig model."""
    
    def test_layout_config_creation_with_all_fields(self) -> None:
        """Test creating a PDFLayoutConfig with all fields."""
        layout = PDFLayoutConfig(
            page_size="Letter",
            orientation="landscape",
            margins={"top": 100.0, "bottom": 100.0, "left": 80.0, "right": 80.0},
            columns=2,
            header_height=0.6,
            footer_height=0.4,
        )
        
        assert layout.page_size == "Letter"
        assert layout.orientation == "landscape"
        assert layout.margins == {"top": 100.0, "bottom": 100.0, "left": 80.0, "right": 80.0}
        assert layout.columns == 2
        assert layout.header_height == 0.6
        assert layout.footer_height == 0.4
    
    def test_layout_config_creation_with_defaults(self) -> None:
        """Test creating a PDFLayoutConfig with default values."""
        layout = PDFLayoutConfig()
        
        assert layout.page_size == "A4"  # Default
        assert layout.orientation == "portrait"  # Default
        assert layout.margins == {"top": 72.0, "bottom": 72.0, "left": 72.0, "right": 72.0}
        assert layout.columns == 1  # Default
        assert layout.header_height == 0.5  # Default
        assert layout.footer_height == 0.3  # Default
    
    def test_layout_config_validation_page_size_options(self) -> None:
        """Test that page size validation works correctly."""
        valid_sizes = ["A4", "Letter", "Legal"]
        for size in valid_sizes:
            layout = PDFLayoutConfig(page_size=size)
            assert layout.page_size == size
    
    def test_layout_config_validation_page_size_invalid(self) -> None:
        """Test that invalid page size raises ValidationError."""
        with pytest.raises(ValidationError):
            PDFLayoutConfig(page_size="Invalid")
    
    def test_layout_config_validation_orientation_options(self) -> None:
        """Test that orientation validation works correctly."""
        layout = PDFLayoutConfig(orientation="portrait")
        assert layout.orientation == "portrait"
        
        layout = PDFLayoutConfig(orientation="landscape")
        assert layout.orientation == "landscape"
    
    def test_layout_config_validation_orientation_invalid(self) -> None:
        """Test that invalid orientation raises ValidationError."""
        with pytest.raises(ValidationError):
            PDFLayoutConfig(orientation="invalid")
    
    def test_layout_config_validation_margins_structure(self) -> None:
        """Test that margins validation works correctly."""
        # Valid margins
        layout = PDFLayoutConfig(
            margins={"top": 100.0, "bottom": 80.0, "left": 72.0, "right": 72.0}
        )
        assert layout.margins["top"] == 100.0
        assert layout.margins["bottom"] == 80.0
        
        # Integer margins should be converted to float
        layout = PDFLayoutConfig(
            margins={"top": 100, "bottom": 80, "left": 72, "right": 72}
        )
        assert isinstance(layout.margins["top"], float)
    
    def test_layout_config_validation_margins_missing_keys(self) -> None:
        """Test that missing margin keys raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PDFLayoutConfig(margins={"top": 72.0, "bottom": 72.0})
        assert "missing required keys" in str(exc_info.value).lower()
    
    def test_layout_config_validation_margins_extra_keys(self) -> None:
        """Test that extra margin keys raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PDFLayoutConfig(
                margins={
                    "top": 72.0,
                    "bottom": 72.0,
                    "left": 72.0,
                    "right": 72.0,
                    "extra": 50.0,
                }
            )
        assert "invalid keys" in str(exc_info.value).lower()
    
    def test_layout_config_validation_margins_negative(self) -> None:
        """Test that negative margins raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PDFLayoutConfig(
                margins={"top": -10.0, "bottom": 72.0, "left": 72.0, "right": 72.0}
            )
        assert "non-negative" in str(exc_info.value).lower()
    
    def test_layout_config_validation_margins_invalid_type(self) -> None:
        """Test that invalid margin type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PDFLayoutConfig(
                margins={"top": "invalid", "bottom": 72.0, "left": 72.0, "right": 72.0}
            )
        assert "valid number" in str(exc_info.value).lower() or "unable to parse" in str(exc_info.value).lower()
    
    def test_layout_config_validation_columns_options(self) -> None:
        """Test that columns validation works correctly."""
        for cols in [1, 2, 3]:
            layout = PDFLayoutConfig(columns=cols)
            assert layout.columns == cols
    
    def test_layout_config_validation_columns_invalid(self) -> None:
        """Test that invalid column count raises ValidationError."""
        with pytest.raises(ValidationError):
            PDFLayoutConfig(columns=0)
        
        with pytest.raises(ValidationError):
            PDFLayoutConfig(columns=4)
    
    def test_layout_config_validation_header_height_range(self) -> None:
        """Test that header height validation works correctly."""
        # Valid heights
        layout = PDFLayoutConfig(header_height=0.1)
        assert layout.header_height == 0.1
        
        layout = PDFLayoutConfig(header_height=2.0)
        assert layout.header_height == 2.0
    
    def test_layout_config_validation_header_height_invalid(self) -> None:
        """Test that invalid header height raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PDFLayoutConfig(header_height=0.0)
        assert "greater than 0" in str(exc_info.value).lower() or "positive" in str(exc_info.value).lower()
        
        with pytest.raises(ValidationError) as exc_info:
            PDFLayoutConfig(header_height=3.0)
        assert "less than or equal to 2" in str(exc_info.value).lower() or "exceed" in str(exc_info.value).lower()
    
    def test_layout_config_validation_footer_height_range(self) -> None:
        """Test that footer height validation works correctly."""
        # Valid heights
        layout = PDFLayoutConfig(footer_height=0.1)
        assert layout.footer_height == 0.1
        
        layout = PDFLayoutConfig(footer_height=2.0)
        assert layout.footer_height == 2.0
    
    def test_layout_config_validation_footer_height_invalid(self) -> None:
        """Test that invalid footer height raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            PDFLayoutConfig(footer_height=0.0)
        assert "greater than 0" in str(exc_info.value).lower() or "positive" in str(exc_info.value).lower()
        
        with pytest.raises(ValidationError) as exc_info:
            PDFLayoutConfig(footer_height=3.0)
        assert "less than or equal to 2" in str(exc_info.value).lower() or "exceed" in str(exc_info.value).lower()
    
    def test_layout_config_serialization(self) -> None:
        """Test that PDFLayoutConfig can be serialized to dict."""
        layout = PDFLayoutConfig(page_size="Letter", columns=2)
        layout_dict = layout.model_dump()
        
        assert isinstance(layout_dict, dict)
        assert layout_dict["page_size"] == "Letter"
        assert layout_dict["columns"] == 2
    
    def test_layout_config_deserialization(self) -> None:
        """Test that PDFLayoutConfig can be deserialized from dict."""
        layout_data = {
            "page_size": "Legal",
            "orientation": "landscape",
            "columns": 3,
        }
        layout = PDFLayoutConfig.model_validate(layout_data)
        
        assert layout.page_size == "Legal"
        assert layout.orientation == "landscape"
        assert layout.columns == 3


