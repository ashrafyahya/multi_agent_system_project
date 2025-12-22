"""Export agent for generating PDF exports from reports.

This module implements the ExportAgent that creates PDF documents from
competitor analysis reports. The agent supports professional PDF features including:

- **Branding Configuration**: Custom company logos, colors, fonts, and templates
- **Layout Configuration**: Page size, orientation, margins, and column layout
- **Cover Pages**: Professional cover pages with multiple template styles
- **Headers and Footers**: Branded headers and footers on all pages
- **PDF Metadata**: Document properties (title, author, keywords, etc.)
- **Bookmarks**: Clickable navigation bookmarks for major sections
    ```
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.config import get_config
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState
from src.graph.state_utils import update_state
from src.template.pdf_generator import export_to_pdf
from src.template.pdf_utils import get_pdf_configs

logger = logging.getLogger(__name__)


class ExportAgent(BaseAgent):
    """Agent that exports reports to PDF.

    This agent creates professional PDF documents from competitor analysis reports
    with support for branding and layout customization.

    **PDF Features:**
    - Professional cover pages with multiple template styles (default, executive, minimal)
    - Branded headers and footers on all pages
    - PDF metadata (title, author, keywords, creation date)
    - Clickable bookmarks for major sections
    - Custom branding (logos, colors, fonts)
    - Flexible layout options (page size, orientation, margins)

    **Configuration Options:**

    The agent accepts configuration through the `config` parameter:

    - `export_format`: Export format ("pdf", "both") - default: "pdf"
    - `output_dir`: Output directory path - default: "./data/exports"
    - `pdf_branding`: PDFBrandingConfig instance or dict with branding settings
    - `pdf_layout`: PDFLayoutConfig instance or dict with layout settings

    **Branding Configuration:**

    Pass a `PDFBrandingConfig` object or dictionary to customize PDF branding:

    ```python
    from src.models.pdf_branding_config import PDFBrandingConfig
    
    branding = PDFBrandingConfig(
        company_name="Your Company",
        company_logo_path=Path("./logo.png"),  # Optional
        primary_color="#1a1a1a",
        secondary_color="#0066cc",
        accent_color="#ff6600",
        cover_page_template="executive",  # Options: "default", "executive", "minimal"
        footer_text="Confidential",
        document_classification="Internal Use Only"
    )
    ```

    **Layout Configuration:**

    Pass a `PDFLayoutConfig` object or dictionary to customize PDF layout:

    ```python
    from src.models.pdf_layout_config import PDFLayoutConfig
    
    layout = PDFLayoutConfig(
        page_size="A4",  # Options: "A4", "Letter", "Legal"
        orientation="portrait",  # Options: "portrait", "landscape"
        margins={"top": 72, "bottom": 72, "left": 72, "right": 72},  # Points
        columns=1,  # Options: 1, 2, 3
        header_height=0.5,  # Inches
        footer_height=0.3  # Inches
    )
    ```

    **Backward Compatibility:**

    The agent maintains full backward compatibility. If no branding or layout
    configuration is provided, default professional settings are used automatically.

    Attributes:
        llm: Language model instance (injected, may be used for future enhancements)
        config: Configuration dictionary containing export settings and optional
            PDF branding/layout configurations

    """

    def _setup_export_environment(
        self, export_config: dict[str, Any]
    ) -> tuple[Path, str]:
        """Setup export environment (output directory and filename).

        Args:
            export_config: Export configuration dictionary

        Returns:
            Tuple of (output_dir, base_filename)
        """
        app_config = get_config()
        output_dir = Path(
            export_config.get("output_dir", app_config.data_dir / "exports")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"competitor_analysis_{timestamp}"

        return output_dir, base_filename

    def _handle_pdf_export_error(self, error: WorkflowError) -> None:
        """Handle PDF export errors, especially missing reportlab.

        Args:
            error: WorkflowError that occurred during PDF export
        """
        if "reportlab" in str(error).lower():
            logger.warning(
                f"PDF export skipped: {error}. "
                "Install reportlab to enable PDF export: pip install reportlab"
            )
        else:
            raise

    def _handle_pdf_export(
        self,
        report: str,
        export_config: dict[str, Any],
        output_dir: Path,
        base_filename: str,
        branding_config: Any,
        layout_config: Any,
    ) -> dict[str, str]:
        """Handle PDF export generation.

        Args:
            report: Markdown-formatted report string
            export_config: Export configuration dictionary
            output_dir: Output directory for PDF
            base_filename: Base filename for PDF
            branding_config: PDF branding configuration
            layout_config: PDF layout configuration

        Returns:
            Dictionary with PDF export path
        """
        export_paths: dict[str, str] = {}
        export_format = export_config.get("export_format", "pdf")

        if export_format not in ["pdf", "both"]:
            return export_paths

        try:
            pdf_path = export_to_pdf(
                report,
                output_dir,
                base_filename,
                branding_config=branding_config,
                layout_config=layout_config,
            )
            export_paths["pdf"] = str(pdf_path)
            logger.info(f"PDF exported to: {pdf_path}")
        except WorkflowError as e:
            self._handle_pdf_export_error(e)

        return export_paths

    def _finalize_export_state(
        self,
        state: WorkflowState,
        export_paths: dict[str, str],
        output_dir: Path,
    ) -> WorkflowState:
        """Finalize export state with all paths.

        Args:
            state: Current workflow state
            export_paths: Dictionary of export paths
            output_dir: Output directory

        Returns:
            Updated WorkflowState with export_paths populated
        """
        new_state = update_state(
            state,
            export_paths=export_paths,
            current_task="Export completed successfully"
        )

        logger.info(
            f"Export completed: {len(export_paths)} file(s) generated in {output_dir}"
        )

        return new_state

    def _validate_report(self, state: WorkflowState) -> str:
        """Validate and extract report from state.

        Args:
            state: Current workflow state

        Returns:
            Report string

        Raises:
            WorkflowError: If report is missing
        """
        report = state.get("report")
        if not report:
            raise WorkflowError(
                "Cannot export without a report",
                context={"state_keys": list(state.keys())}
            )
        return report

    def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute export generation from report.

        Args:
            state: Current workflow state with report (required)

        Returns:
            Updated WorkflowState with export_paths populated

        Raises:
            WorkflowError: If report is missing or export generation fails
        """
        try:
            report = self._validate_report(state)
            logger.info("Starting export generation")

            export_config = self.config.copy()
            output_dir, base_filename = self._setup_export_environment(export_config)
            branding_config, layout_config = get_pdf_configs(export_config)

            export_paths = self._handle_pdf_export(
                report,
                export_config,
                output_dir,
                base_filename,
                branding_config,
                layout_config,
            )

            return self._finalize_export_state(state, export_paths, output_dir)

        except WorkflowError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in export agent: {e}", exc_info=True)
            raise WorkflowError(
                "Export generation failed unexpectedly",
                context={"error": str(e)}
            ) from e

    @property
    def name(self) -> str:
        """Return agent name.

        Returns:
            String identifier for this agent
        """
        return "export_agent"
