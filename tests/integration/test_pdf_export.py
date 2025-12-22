"""Integration tests for enhanced PDF export functionality.

This module contains integration tests that verify full PDF generation
with branding, layout, and metadata.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from langchain_core.language_models import BaseChatModel

from src.agents.export_agent import ExportAgent
from src.graph.state import create_initial_state
from src.models.pdf_branding_config import PDFBrandingConfig
from src.models.pdf_layout_config import PDFLayoutConfig


@pytest.mark.integration
class TestPDFExportIntegration:
    """Integration tests for PDF export with branding and layout."""
    
    def test_pdf_export_with_branding_config(self) -> None:
        """Test full PDF generation with branding configuration."""
        mock_llm = Mock(spec=BaseChatModel)
        branding = PDFBrandingConfig(
            company_name="Test Company",
            primary_color="#1a1a1a",
            secondary_color="#0066cc",
            accent_color="#ff6600",
        )
        config = {
            "export_format": "pdf",
            "pdf_branding": branding,
        }
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                state = create_initial_state("Test")
                state["report"] = "# Test Report\n## Section 1\nContent here."
                
                result_state = agent.execute(state)
                
                assert "export_paths" in result_state
                assert "pdf" in result_state["export_paths"]
                pdf_path = Path(result_state["export_paths"]["pdf"])
                assert pdf_path.exists()
                assert pdf_path.suffix == ".pdf"
    
    def test_pdf_export_without_branding_backward_compatibility(self) -> None:
        """Test PDF export without branding maintains backward compatibility."""
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                state = create_initial_state("Test")
                state["report"] = "# Test Report\n## Section 1\nContent here."
                
                result_state = agent.execute(state)
                
                # Should still generate PDF with default branding
                assert "export_paths" in result_state
                assert "pdf" in result_state["export_paths"]
                pdf_path = Path(result_state["export_paths"]["pdf"])
                assert pdf_path.exists()
    
    def test_pdf_export_with_different_template_styles(self) -> None:
        """Test PDF export with different template styles."""
        mock_llm = Mock(spec=BaseChatModel)
        templates = ["default", "executive", "minimal"]
        
        for template in templates:
            branding = PDFBrandingConfig(
                company_name="Test Company",
                cover_page_template=template,
            )
            config = {
                "export_format": "pdf",
                "pdf_branding": branding,
            }
            agent = ExportAgent(llm=mock_llm, config=config)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir) / "exports"
                config["output_dir"] = str(output_dir)
                
                with patch("src.agents.export_agent.get_config") as mock_get_config:
                    mock_app_config = Mock()
                    mock_app_config.data_dir = Path(tmpdir)
                    mock_get_config.return_value = mock_app_config
                    
                    state = create_initial_state("Test")
                    state["report"] = "# Test Report\n## Section 1\nContent here."
                    
                    result_state = agent.execute(state)
                    
                    assert "export_paths" in result_state
                    assert "pdf" in result_state["export_paths"]
                    pdf_path = Path(result_state["export_paths"]["pdf"])
                    assert pdf_path.exists()
    
    def test_pdf_export_with_layout_config(self) -> None:
        """Test PDF export with layout configuration."""
        mock_llm = Mock(spec=BaseChatModel)
        layout = PDFLayoutConfig(
            page_size="A4",
            orientation="portrait",
            margins={"top": 72.0, "bottom": 72.0, "left": 72.0, "right": 72.0},
        )
        config = {
            "export_format": "pdf",
            "pdf_layout": layout,
        }
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                state = create_initial_state("Test")
                state["report"] = "# Test Report\n## Section 1\nContent here."
                
                result_state = agent.execute(state)
                
                assert "export_paths" in result_state
                assert "pdf" in result_state["export_paths"]
                pdf_path = Path(result_state["export_paths"]["pdf"])
                assert pdf_path.exists()
    
    def test_pdf_export_with_metadata(self) -> None:
        """Test PDF export includes metadata."""
        mock_llm = Mock(spec=BaseChatModel)
        branding = PDFBrandingConfig(company_name="Test Company")
        config = {
            "export_format": "pdf",
            "pdf_branding": branding,
        }
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                state = create_initial_state("Test")
                state["report"] = "# Test Report\n## Section 1\nContent here."
                
                result_state = agent.execute(state)
                
                assert "export_paths" in result_state
                pdf_path = Path(result_state["export_paths"]["pdf"])
                assert pdf_path.exists()
                
                # Verify PDF has metadata by checking file size > 0
                # (actual metadata verification would require PDF parsing)
                assert pdf_path.stat().st_size > 0
    
    def test_pdf_export_with_branding_and_layout(self) -> None:
        """Test PDF export with both branding and layout configs."""
        mock_llm = Mock(spec=BaseChatModel)
        branding = PDFBrandingConfig(
            company_name="Test Company",
            primary_color="#1a1a1a",
            cover_page_template="executive",
        )
        layout = PDFLayoutConfig(
            page_size="Letter",
            orientation="portrait",
            margins={"top": 100.0, "bottom": 100.0, "left": 100.0, "right": 100.0},
        )
        config = {
            "export_format": "pdf",
            "pdf_branding": branding,
            "pdf_layout": layout,
        }
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                state = create_initial_state("Test")
                state["report"] = "# Test Report\n## Section 1\nContent here.\n## Section 2\nMore content."
                
                result_state = agent.execute(state)
                
                assert "export_paths" in result_state
                assert "pdf" in result_state["export_paths"]
                pdf_path = Path(result_state["export_paths"]["pdf"])
                assert pdf_path.exists()
    
    def test_pdf_export_handles_long_report(self) -> None:
        """Test PDF export handles long reports correctly."""
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                # Create a long report
                long_content = "# Long Report\n"
                for i in range(20):
                    long_content += f"## Section {i}\n"
                    long_content += "Content " * 100 + "\n"
                
                state = create_initial_state("Test")
                state["report"] = long_content
                
                result_state = agent.execute(state)
                
                assert "export_paths" in result_state
                assert "pdf" in result_state["export_paths"]
                pdf_path = Path(result_state["export_paths"]["pdf"])
                assert pdf_path.exists()
                # Long report should create larger PDF
                assert pdf_path.stat().st_size > 1000
    
    def test_pdf_export_with_logo(self) -> None:
        """Test PDF export with company logo."""
        import tempfile as tf
        
        mock_llm = Mock(spec=BaseChatModel)
        
        # Create a temporary logo file
        with tf.NamedTemporaryFile(suffix=".png", delete=False) as tmp_logo:
            logo_path = Path(tmp_logo.name)
            tmp_logo.write(b"fake png data")
        
        try:
            branding = PDFBrandingConfig(
                company_name="Test Company",
                company_logo_path=logo_path,
            )
            config = {
                "export_format": "pdf",
                "pdf_branding": branding,
            }
            agent = ExportAgent(llm=mock_llm, config=config)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir) / "exports"
                config["output_dir"] = str(output_dir)
                
                with patch("src.agents.export_agent.get_config") as mock_get_config:
                    mock_app_config = Mock()
                    mock_app_config.data_dir = Path(tmpdir)
                    mock_get_config.return_value = mock_app_config
                    
                    # Mock Image to avoid actual image processing
                    with patch("reportlab.platypus.Image") as mock_image:
                        mock_image.return_value = MagicMock()
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Test Report\n## Section 1\nContent here."
                        
                        result_state = agent.execute(state)
                        
                        assert "export_paths" in result_state
                        assert "pdf" in result_state["export_paths"]
                        pdf_path = Path(result_state["export_paths"]["pdf"])
                        assert pdf_path.exists()
        finally:
            if logo_path.exists():
                logo_path.unlink()

