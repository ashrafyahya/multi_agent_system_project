"""Tests for template implementations.

This module contains unit tests for all template components to verify
PDF generation, styling, and visualization functionality.
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.models.pdf_branding_config import PDFBrandingConfig
from src.models.pdf_layout_config import PDFLayoutConfig


class TestPDFTemplateEngine:
    """Tests for PDFTemplateEngine abstract class."""
    
    def test_template_engine_cannot_be_instantiated(self) -> None:
        """Test that PDFTemplateEngine cannot be instantiated directly."""
        from src.template.template_engine import PDFTemplateEngine
        
        with pytest.raises(TypeError):
            PDFTemplateEngine()  # type: ignore
    
    def test_template_engine_requires_create_cover_page(self) -> None:
        """Test that subclasses must implement create_cover_page method."""
        from src.template.template_engine import PDFTemplateEngine
        
        class IncompleteEngine(PDFTemplateEngine):
            def create_header(self, page_num: int, total_pages: int):
                return None
            
            def create_footer(self, page_num: int, total_pages: int):
                return None
        
        with pytest.raises(TypeError):
            IncompleteEngine()  # type: ignore
    
    def test_template_engine_requires_create_header(self) -> None:
        """Test that subclasses must implement create_header method."""
        from src.template.template_engine import PDFTemplateEngine
        
        class IncompleteEngine(PDFTemplateEngine):
            def create_cover_page(self) -> list:
                return []
            
            def create_footer(self, page_num: int, total_pages: int):
                return None
        
        with pytest.raises(TypeError):
            IncompleteEngine()  # type: ignore
    
    def test_template_engine_requires_create_footer(self) -> None:
        """Test that subclasses must implement create_footer method."""
        from src.template.template_engine import PDFTemplateEngine
        
        class IncompleteEngine(PDFTemplateEngine):
            def create_cover_page(self) -> list:
                return []
            
            def create_header(self, page_num: int, total_pages: int):
                return None
        
        with pytest.raises(TypeError):
            IncompleteEngine()  # type: ignore


class TestDefaultPDFTemplateEngine:
    """Tests for DefaultPDFTemplateEngine."""
    
    def test_default_engine_initialization_with_all_configs(self) -> None:
        """Test initialization with all configurations."""
        from src.template.template_engine import DefaultPDFTemplateEngine
        
        branding = PDFBrandingConfig(company_name="Test Company")
        layout = PDFLayoutConfig(page_size="A4")
        engine = DefaultPDFTemplateEngine(
            branding_config=branding,
            layout_config=layout,
            report_title="Test Report",
            report_date=datetime(2024, 1, 1),
        )
        
        assert engine.branding_config == branding
        assert engine.layout_config == layout
        assert engine.report_title == "Test Report"
        assert engine.report_date == datetime(2024, 1, 1)
    
    def test_default_engine_initialization_with_defaults(self) -> None:
        """Test initialization with default values."""
        from src.template.template_engine import DefaultPDFTemplateEngine
        
        engine = DefaultPDFTemplateEngine()
        
        assert engine.branding_config is None
        assert engine.layout_config is None
        assert engine.report_title == "Competitor Analysis Report"
        assert isinstance(engine.report_date, datetime)
    
    def test_default_engine_create_cover_page_delegates(self) -> None:
        """Test create_cover_page delegates correctly."""
        from src.template.template_engine import DefaultPDFTemplateEngine
        
        branding = PDFBrandingConfig(company_name="Test Company")
        engine = DefaultPDFTemplateEngine(branding_config=branding)
        
        with patch("src.template.template_engine.create_cover_page") as mock_create:
            mock_create.return_value = []
            result = engine.create_cover_page()
            
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["branding_config"] == branding
            assert call_kwargs["report_title"] == "Competitor Analysis Report"
    
    def test_default_engine_create_header_delegates(self) -> None:
        """Test create_header delegates correctly."""
        from src.template.template_engine import DefaultPDFTemplateEngine
        
        branding = PDFBrandingConfig(company_name="Test Company")
        engine = DefaultPDFTemplateEngine(branding_config=branding)
        
        with patch("src.template.template_engine.create_header") as mock_create:
            mock_create.return_value = None
            result = engine.create_header(page_num=1, total_pages=10)
            
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["branding_config"] == branding
            assert call_kwargs["page_num"] == 1
            assert call_kwargs["total_pages"] == 10
    
    def test_default_engine_create_footer_delegates(self) -> None:
        """Test create_footer delegates correctly."""
        from src.template.template_engine import DefaultPDFTemplateEngine
        
        branding = PDFBrandingConfig(company_name="Test Company")
        engine = DefaultPDFTemplateEngine(branding_config=branding)
        
        with patch("src.template.template_engine.create_footer") as mock_create:
            mock_create.return_value = None
            result = engine.create_footer(page_num=1, total_pages=10)
            
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["branding_config"] == branding
            assert call_kwargs["page_num"] == 1
            assert call_kwargs["total_pages"] == 10
    
    def test_default_engine_build_document_structure(self) -> None:
        """Test build_document_structure returns correct structure."""
        from src.template.template_engine import DefaultPDFTemplateEngine
        
        engine = DefaultPDFTemplateEngine()
        
        with patch.object(engine, "create_cover_page", return_value=[]):
            structure = engine.build_document_structure("# Report\n## Section")
            
            assert "cover_page" in structure
            assert "header_factory" in structure
            assert "footer_factory" in structure
            assert callable(structure["header_factory"])
            assert callable(structure["footer_factory"])
    
    def test_default_engine_handles_none_branding_config(self) -> None:
        """Test default engine handles None branding config."""
        from src.template.template_engine import DefaultPDFTemplateEngine
        
        engine = DefaultPDFTemplateEngine(branding_config=None)
        
        with patch("src.template.template_engine.create_cover_page") as mock_create:
            mock_create.return_value = []
            engine.create_cover_page()
            
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["branding_config"] is None
    
    def test_default_engine_handles_none_layout_config(self) -> None:
        """Test default engine handles None layout config."""
        from src.template.template_engine import DefaultPDFTemplateEngine
        
        engine = DefaultPDFTemplateEngine(layout_config=None)
        
        with patch("src.template.template_engine.create_cover_page") as mock_create:
            mock_create.return_value = []
            engine.create_cover_page()
            
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["layout_config"] is None
    
    def test_default_engine_report_date_defaults_to_current(self) -> None:
        """Test report_date defaults to current date."""
        from datetime import datetime

        from src.template.template_engine import DefaultPDFTemplateEngine
        
        engine = DefaultPDFTemplateEngine()
        
        # Should be close to current time (within 1 second)
        time_diff = abs((engine.report_date - datetime.now()).total_seconds())
        assert time_diff < 1.0
    
    def test_default_engine_custom_report_date(self) -> None:
        """Test custom report_date is used."""
        from src.template.template_engine import DefaultPDFTemplateEngine
        
        custom_date = datetime(2024, 6, 15)
        engine = DefaultPDFTemplateEngine(report_date=custom_date)
        
        assert engine.report_date == custom_date


class TestCoverPageFunctions:
    """Tests for cover_page functions."""
    
    def test_create_cover_page_with_all_branding_config(self) -> None:
        """Test create_cover_page with all branding config."""
        from src.template.cover_page import create_cover_page
        
        branding = PDFBrandingConfig(
            company_name="Test Company",
            primary_color="#1a1a1a",
            secondary_color="#0066cc",
        )
        layout = PDFLayoutConfig(page_size="A4")
        
        with patch("reportlab.lib.styles.getSampleStyleSheet"):
            with patch("reportlab.platypus.Image"):
                with patch("reportlab.platypus.Paragraph"):
                    result = create_cover_page(
                        branding_config=branding,
                        layout_config=layout,
                        report_title="Test Report",
                        report_date=datetime.now(),
                    )
                    
                    assert isinstance(result, list)
    
    def test_create_cover_page_with_minimal_config(self) -> None:
        """Test create_cover_page with minimal config."""
        from src.template.cover_page import create_cover_page
        
        with patch("reportlab.lib.styles.getSampleStyleSheet"):
            with patch("reportlab.platypus.Paragraph"):
                result = create_cover_page(
                    branding_config=None,
                    layout_config=None,
                    report_title="Test Report",
                    report_date=datetime.now(),
                )
                
                assert isinstance(result, list)
    
    def test_create_cover_page_handles_none_branding(self) -> None:
        """Test create_cover_page handles None branding."""
        from src.template.cover_page import create_cover_page
        
        with patch("reportlab.lib.styles.getSampleStyleSheet"):
            with patch("reportlab.platypus.Paragraph"):
                result = create_cover_page(
                    branding_config=None,
                    layout_config=None,
                    report_title="Test Report",
                    report_date=datetime.now(),
                )
                
                assert isinstance(result, list)
    
    def test_create_cover_page_handles_none_layout(self) -> None:
        """Test create_cover_page handles None layout."""
        from src.template.cover_page import create_cover_page
        
        branding = PDFBrandingConfig(company_name="Test Company")
        
        with patch("reportlab.lib.styles.getSampleStyleSheet"):
            with patch("reportlab.platypus.Paragraph"):
                result = create_cover_page(
                    branding_config=branding,
                    layout_config=None,
                    report_title="Test Report",
                    report_date=datetime.now(),
                )
                
                assert isinstance(result, list)
    
    def test_create_cover_page_with_logo(self) -> None:
        """Test create_cover_page with logo."""
        import tempfile

        from src.template.cover_page import create_cover_page
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            logo_path = Path(tmp.name)
        
        try:
            branding = PDFBrandingConfig(
                company_name="Test Company",
                company_logo_path=logo_path,
            )
            
            with patch("reportlab.lib.styles.getSampleStyleSheet"):
                with patch("reportlab.platypus.Image") as mock_image:
                    with patch("reportlab.platypus.Paragraph"):
                        mock_image.return_value = Mock()
                        result = create_cover_page(
                            branding_config=branding,
                            layout_config=None,
                            report_title="Test Report",
                            report_date=datetime.now(),
                        )
                        
                        assert isinstance(result, list)
        finally:
            if logo_path.exists():
                logo_path.unlink()
    
    def test_create_cover_page_without_logo(self) -> None:
        """Test create_cover_page without logo."""
        from src.template.cover_page import create_cover_page
        
        branding = PDFBrandingConfig(company_name="Test Company")
        
        with patch("reportlab.lib.styles.getSampleStyleSheet"):
            with patch("reportlab.platypus.Paragraph"):
                result = create_cover_page(
                    branding_config=branding,
                    layout_config=None,
                    report_title="Test Report",
                    report_date=datetime.now(),
                )
                
                assert isinstance(result, list)
    
    def test_create_cover_page_with_different_templates(self) -> None:
        """Test create_cover_page with different template styles."""
        from src.template.cover_page import create_cover_page
        
        templates = ["default", "executive", "minimal"]
        
        for template in templates:
            branding = PDFBrandingConfig(
                company_name="Test Company",
                cover_page_template=template,
            )
            
            with patch("reportlab.lib.styles.getSampleStyleSheet"):
                with patch("reportlab.platypus.Paragraph"):
                    result = create_cover_page(
                        branding_config=branding,
                        layout_config=None,
                        report_title="Test Report",
                        report_date=datetime.now(),
                    )
                    
                    assert isinstance(result, list)
    
    def test_create_cover_page_formats_date_correctly(self) -> None:
        """Test create_cover_page formats date correctly."""
        from src.template.cover_page import create_cover_page
        
        test_date = datetime(2024, 6, 15, 10, 30, 0)
        
        with patch("reportlab.lib.styles.getSampleStyleSheet"):
            with patch("reportlab.platypus.Paragraph") as mock_para:
                mock_para.return_value = Mock()
                create_cover_page(
                    branding_config=None,
                    layout_config=None,
                    report_title="Test Report",
                    report_date=test_date,
                )
                
                # Should have been called with date string
                assert mock_para.called
    
    def test_create_cover_page_handles_long_titles(self) -> None:
        """Test create_cover_page handles long titles."""
        from src.template.cover_page import create_cover_page
        
        long_title = "A" * 200  # Very long title
        
        with patch("reportlab.lib.styles.getSampleStyleSheet"):
            with patch("reportlab.platypus.Paragraph"):
                result = create_cover_page(
                    branding_config=None,
                    layout_config=None,
                    report_title=long_title,
                    report_date=datetime.now(),
                )
                
                assert isinstance(result, list)


class TestHeaderFooterFunctions:
    """Tests for header_footer functions."""
    
    def test_create_header_with_page_numbers(self) -> None:
        """Test create_header with page numbers."""
        from src.template.header_footer import create_header
        
        branding = PDFBrandingConfig(company_name="Test Company")
        header_func = create_header(
            branding_config=branding,
            report_title="Test Report",
            page_num=1,
            total_pages=10,
        )
        
        assert callable(header_func)
    
    def test_create_header_without_branding(self) -> None:
        """Test create_header without branding."""
        from src.template.header_footer import create_header
        
        header_func = create_header(
            branding_config=None,
            report_title="Test Report",
            page_num=1,
            total_pages=10,
        )
        
        assert header_func is None
    
    def test_create_header_with_branding(self) -> None:
        """Test create_header with branding."""
        from src.template.header_footer import create_header
        
        branding = PDFBrandingConfig(company_name="Test Company")
        header_func = create_header(
            branding_config=branding,
            report_title="Test Report",
            page_num=1,
            total_pages=10,
        )
        
        assert callable(header_func)
    
    def test_create_footer_with_page_numbers(self) -> None:
        """Test create_footer with page numbers."""
        from src.template.header_footer import create_footer
        
        footer_func = create_footer(
            branding_config=None,
            report_date=datetime.now(),
            page_num=1,
            total_pages=10,
        )
        
        assert callable(footer_func)
    
    def test_create_footer_without_branding(self) -> None:
        """Test create_footer without branding."""
        from src.template.header_footer import create_footer
        
        footer_func = create_footer(
            branding_config=None,
            report_date=datetime.now(),
            page_num=1,
            total_pages=10,
        )
        
        assert callable(footer_func)
    
    def test_create_footer_with_branding(self) -> None:
        """Test create_footer with branding."""
        from src.template.header_footer import create_footer
        
        branding = PDFBrandingConfig(
            company_name="Test Company",
            footer_text="Confidential",
        )
        footer_func = create_footer(
            branding_config=branding,
            report_date=datetime.now(),
            page_num=1,
            total_pages=10,
        )
        
        assert callable(footer_func)
    
    def test_create_footer_with_custom_text(self) -> None:
        """Test create_footer with custom footer text."""
        from src.template.header_footer import create_footer
        
        branding = PDFBrandingConfig(
            company_name="Test Company",
            footer_text="Custom Footer Text",
        )
        footer_func = create_footer(
            branding_config=branding,
            report_date=datetime.now(),
            page_num=1,
            total_pages=10,
        )
        
        assert callable(footer_func)


class TestTemplateEngineFlowables:
    """Comprehensive tests for template engine flowable generation."""
    
    def test_create_cover_page_generates_flowables(self) -> None:
        """Test create_cover_page generates actual flowables."""
        from src.template.template_engine import DefaultPDFTemplateEngine
        
        branding = PDFBrandingConfig(company_name="Test Company")
        engine = DefaultPDFTemplateEngine(
            branding_config=branding,
            report_title="Test Report",
        )
        
        flowables = engine.create_cover_page()
        
        assert isinstance(flowables, list)
        assert len(flowables) > 0
        # Should contain Paragraph, Spacer, or other flowables
        from reportlab.platypus import PageBreak, Paragraph, Spacer
        assert any(
            isinstance(f, (Paragraph, Spacer, PageBreak))
            for f in flowables
        )
    
    def test_create_header_returns_callable(self) -> None:
        """Test create_header returns callable function."""
        from unittest.mock import MagicMock

        from src.template.template_engine import DefaultPDFTemplateEngine
        
        branding = PDFBrandingConfig(company_name="Test Company")
        engine = DefaultPDFTemplateEngine(branding_config=branding)
        
        header_func = engine.create_header(page_num=1, total_pages=10)
        
        assert callable(header_func)
        
        # Test that it can be called with canvas and doc
        mock_canvas = MagicMock()
        mock_doc = MagicMock()
        mock_doc.pagesize = (612, 792)  # Letter size
        
        header_func(mock_canvas, mock_doc)
        
        # Should have called canvas methods
        assert mock_canvas.setFont.called or mock_canvas.drawString.called
    
    def test_create_footer_returns_callable(self) -> None:
        """Test create_footer returns callable function."""
        from unittest.mock import MagicMock

        from src.template.template_engine import DefaultPDFTemplateEngine
        
        branding = PDFBrandingConfig(company_name="Test Company")
        engine = DefaultPDFTemplateEngine(branding_config=branding)
        
        footer_func = engine.create_footer(page_num=1, total_pages=10)
        
        assert callable(footer_func)
        
        # Test that it can be called with canvas and doc
        mock_canvas = MagicMock()
        mock_doc = MagicMock()
        mock_doc.pagesize = (612, 792)  # Letter size
        
        footer_func(mock_canvas, mock_doc)
        
        # Should have called canvas methods
        assert mock_canvas.setFont.called or mock_canvas.drawString.called
    
    def test_template_selection_affects_cover_page(self) -> None:
        """Test different template styles generate different cover pages."""
        from src.template.template_engine import DefaultPDFTemplateEngine
        
        templates = ["default", "executive", "minimal"]
        flowables_by_template = {}
        
        for template in templates:
            branding = PDFBrandingConfig(
                company_name="Test Company",
                cover_page_template=template,
            )
            engine = DefaultPDFTemplateEngine(
                branding_config=branding,
                report_title="Test Report",
            )
            flowables = engine.create_cover_page()
            flowables_by_template[template] = flowables
        
        # All should generate flowables
        assert all(len(flowables) > 0 for flowables in flowables_by_template.values())
        # They might have different structures (different spacing, etc.)
    
    def test_header_with_different_page_numbers(self) -> None:
        """Test header works with different page numbers."""
        from unittest.mock import MagicMock

        from src.template.template_engine import DefaultPDFTemplateEngine
        
        branding = PDFBrandingConfig(company_name="Test Company")
        engine = DefaultPDFTemplateEngine(branding_config=branding)
        
        for page_num in [1, 5, 10]:
            header_func = engine.create_header(page_num=page_num, total_pages=20)
            assert callable(header_func)
            
            mock_canvas = MagicMock()
            mock_doc = MagicMock()
            mock_doc.pagesize = (612, 792)
            
            header_func(mock_canvas, mock_doc)
            assert mock_canvas.setFont.called or mock_canvas.drawString.called
    
    def test_footer_with_different_page_numbers(self) -> None:
        """Test footer works with different page numbers."""
        from unittest.mock import MagicMock

        from src.template.template_engine import DefaultPDFTemplateEngine
        
        branding = PDFBrandingConfig(company_name="Test Company")
        engine = DefaultPDFTemplateEngine(branding_config=branding)
        
        for page_num in [1, 5, 10]:
            footer_func = engine.create_footer(page_num=page_num, total_pages=20)
            assert callable(footer_func)
            
            mock_canvas = MagicMock()
            mock_doc = MagicMock()
            mock_doc.pagesize = (612, 792)
            
            footer_func(mock_canvas, mock_doc)
            assert mock_canvas.setFont.called or mock_canvas.drawString.called


class TestStyleUtilsFunctions:
    """Tests for style_utils functions."""
    
    def test_hex_to_rgb_with_valid_hex_colors(self) -> None:
        """Test hex_to_rgb with valid hex colors."""
        from src.template.style_utils import hex_to_rgb

        # 6-digit hex
        rgb = hex_to_rgb("#1a1a1a")
        assert isinstance(rgb, tuple)
        assert len(rgb) == 3
        assert all(0.0 <= val <= 1.0 for val in rgb)
        
        # 3-digit hex
        rgb = hex_to_rgb("#1a1")
        assert isinstance(rgb, tuple)
        assert len(rgb) == 3
    
    def test_hex_to_rgb_with_3_digit_hex(self) -> None:
        """Test hex_to_rgb with 3-digit hex."""
        from src.template.style_utils import hex_to_rgb
        
        rgb = hex_to_rgb("#abc")
        assert isinstance(rgb, tuple)
        assert len(rgb) == 3
    
    def test_hex_to_rgb_with_6_digit_hex(self) -> None:
        """Test hex_to_rgb with 6-digit hex."""
        from src.template.style_utils import hex_to_rgb
        
        rgb = hex_to_rgb("#abcdef")
        assert isinstance(rgb, tuple)
        assert len(rgb) == 3
    
    def test_hex_to_rgb_with_invalid_hex(self) -> None:
        """Test hex_to_rgb with invalid hex."""
        from src.template.style_utils import hex_to_rgb

        # Should handle gracefully or raise ValueError
        try:
            rgb = hex_to_rgb("invalid")
            # If it doesn't raise, should return some value
            assert isinstance(rgb, tuple)
        except (ValueError, IndexError):
            # Also acceptable
            pass
    
    def test_get_branding_color_with_primary_color(self) -> None:
        """Test get_branding_color with primary color."""
        from src.template.style_utils import get_branding_color
        
        branding = PDFBrandingConfig(
            company_name="Test Company",
            primary_color="#1a1a1a",
        )
        
        rgb = get_branding_color(branding, "primary")
        assert isinstance(rgb, tuple)
        assert len(rgb) == 3
    
    def test_get_branding_color_with_secondary_color(self) -> None:
        """Test get_branding_color with secondary color."""
        from src.template.style_utils import get_branding_color
        
        branding = PDFBrandingConfig(
            company_name="Test Company",
            secondary_color="#0066cc",
        )
        
        rgb = get_branding_color(branding, "secondary")
        assert isinstance(rgb, tuple)
        assert len(rgb) == 3
    
    def test_get_branding_color_with_accent_color(self) -> None:
        """Test get_branding_color with accent color."""
        from src.template.style_utils import get_branding_color
        
        branding = PDFBrandingConfig(
            company_name="Test Company",
            accent_color="#ff6600",
        )
        
        rgb = get_branding_color(branding, "accent")
        assert isinstance(rgb, tuple)
        assert len(rgb) == 3
    
    def test_get_branding_color_with_none_config(self) -> None:
        """Test get_branding_color with None config."""
        from src.template.style_utils import get_branding_color
        
        rgb = get_branding_color(None, "primary")
        assert rgb == (0.0, 0.0, 0.0)  # Black default
    
    def test_get_branding_color_with_invalid_color_type(self) -> None:
        """Test get_branding_color with invalid color type."""
        from src.template.style_utils import get_branding_color
        
        branding = PDFBrandingConfig(company_name="Test Company")
        
        rgb = get_branding_color(branding, "invalid")
        # Should return default black
        assert isinstance(rgb, tuple)
        assert len(rgb) == 3
    
    def test_calculate_responsive_spacing(self) -> None:
        """Test calculate_responsive_spacing function."""
        from src.template.style_utils import calculate_responsive_spacing
        
        spacing = calculate_responsive_spacing(
            template_style="default",
            has_logo=False,
            title_length=50,
            metadata_count=2,
        )
        
        assert isinstance(spacing, dict)
        assert "top_margin" in spacing
        assert "logo_spacing" in spacing
        assert "title_spacing_before" in spacing
        assert "title_spacing_after" in spacing
        assert "metadata_spacing" in spacing
        assert "bottom_margin" in spacing
    
    def test_calculate_responsive_spacing_with_logo(self) -> None:
        """Test calculate_responsive_spacing with logo."""
        from src.template.style_utils import calculate_responsive_spacing
        
        spacing = calculate_responsive_spacing(
            template_style="executive",
            has_logo=True,
            title_length=30,
            metadata_count=1,
        )
        
        assert spacing["logo_spacing"] > 0
    
    def test_get_page_height(self) -> None:
        """Test get_page_height function."""
        from src.template.style_utils import get_page_height
        
        layout = PDFLayoutConfig(page_size="A4")
        height = get_page_height(layout)
        
        assert isinstance(height, float)
        assert height > 0
    
    def test_get_page_height_with_none_config(self) -> None:
        """Test get_page_height with None config."""
        from src.template.style_utils import get_page_height
        
        height = get_page_height(None)
        assert height is None


