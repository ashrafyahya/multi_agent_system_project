"""Export agent for generating PDF and image exports from reports.

This module implements the ExportAgent that creates PDF documents and optional
visualizations from competitor analysis reports.

Example:
    ```python
    from src.agents.export_agent import ExportAgent
    from langchain_groq import ChatGroq
    from src.graph.state import create_initial_state
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    config = {"export_format": "pdf", "include_visualizations": True}
    agent = ExportAgent(llm=llm, config=config)
    
    state = create_initial_state("Analyze competitors")
    state["report"] = "# Report\n## Summary\n..."
    updated_state = agent.execute(state)
    ```
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.config import get_config
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState

logger = logging.getLogger(__name__)


class ExportAgent(BaseAgent):
    """Agent that exports reports to PDF and generates visualizations.
    
    This agent creates export files from competitor analysis reports by:
    1. Extracting report from workflow state
    2. Converting markdown report to PDF
    3. Optionally generating visualizations (SWOT diagrams, charts)
    4. Saving files to configured data directory
    5. Storing export paths in state
    
    The agent uses external libraries for PDF generation and visualization.
    It follows the Single Responsibility Principle by only handling export
    operations, not report generation.
    
    Attributes:
        llm: Language model instance (injected, may be used for future enhancements)
        config: Configuration dictionary (injected)
    
    Example:
        ```python
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
        config = {
            "export_format": "pdf",
            "include_visualizations": True,
            "output_dir": "./data/exports"
        }
        agent = ExportAgent(llm=llm, config=config)
        
        state = create_initial_state("Analyze competitors")
        state["report"] = "# Report\n## Summary\n..."
        updated_state = agent.execute(state)
        ```
    """
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute export generation from report.
        
        Generates export files (PDF, images) by:
        1. Extracting report from state
        2. Creating output directory if needed
        3. Converting report to PDF
        4. Optionally generating visualizations
        5. Storing export paths in state
        
        Args:
            state: Current workflow state containing report
        
        Returns:
            Updated WorkflowState with export_paths field populated
        
        Raises:
            WorkflowError: If report is missing or export generation fails
        
        Example:
            ```python
            state = create_initial_state("Analyze competitors")
            state["report"] = "# Report\n## Summary\n..."
            updated_state = agent.execute(state)
            assert updated_state["export_paths"] is not None
            ```
        """
        try:
            # Extract report
            report = state.get("report")
            if not report:
                raise WorkflowError(
                    "Cannot export without a report",
                    context={"state_keys": list(state.keys())}
                )
            
            logger.info("Starting export generation")
            
            # Get configuration
            app_config = get_config()
            export_config = self.config.copy()
            
            # Determine output directory
            output_dir = Path(
                export_config.get("output_dir", app_config.data_dir / "exports")
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"competitor_analysis_{timestamp}"
            
            export_paths: dict[str, str] = {}
            
            # Generate visualizations first (needed for PDF embedding)
            viz_paths: dict[str, str] = {}
            if export_config.get("include_visualizations", False):
                insights = state.get("insights")
                if insights:
                    viz_paths = self._generate_visualizations(
                        insights, output_dir, base_filename
                    )
            
            # Export to PDF (with embedded diagrams if available)
            export_format = export_config.get("export_format", "pdf")
            if export_format in ["pdf", "both"]:
                try:
                    pdf_path = self._export_to_pdf(
                        report, output_dir, base_filename, viz_paths
                    )
                    export_paths["pdf"] = str(pdf_path)
                    logger.info(f"PDF exported to: {pdf_path}")
                except WorkflowError as e:
                    # If PDF export fails due to missing libraries, log warning but continue
                    if "reportlab" in str(e).lower():
                        logger.warning(
                            f"PDF export skipped: {e}. "
                            "Install reportlab to enable PDF export: pip install reportlab"
                        )
                    else:
                        raise
            
            # Update export_paths with visualization paths
            export_paths.update(viz_paths)
            if viz_paths:
                logger.info(f"Visualizations generated: {len(viz_paths)} files")
            
            # Update state
            new_state = state.copy()
            new_state["export_paths"] = export_paths
            new_state["current_task"] = "Export completed successfully"
            
            logger.info(
                f"Export completed: {len(export_paths)} file(s) generated in {output_dir}"
            )
            
            return new_state
            
        except WorkflowError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in export agent: {e}", exc_info=True)
            raise WorkflowError(
                "Export generation failed unexpectedly",
                context={"error": str(e)}
            ) from e
    
    def _export_to_pdf(
        self, report: str, output_dir: Path, base_filename: str,
        viz_paths: dict[str, str] | None = None
    ) -> Path:
        """Export report to PDF format.
        
        Converts markdown report to PDF using markdown and PDF libraries.
        Handles markdown formatting and creates a well-formatted PDF document.
        
        Args:
            report: Markdown-formatted report string
            output_dir: Directory to save PDF file
            base_filename: Base filename (without extension)
            viz_paths: Optional dictionary mapping diagram types to file paths
        
        Returns:
            Path to generated PDF file
        
        Raises:
            WorkflowError: If PDF generation fails
        """
        try:
            pdf_path = output_dir / f"{base_filename}.pdf"
            
            # Use reportlab to create PDF
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib.units import inch
                from reportlab.platypus import (Paragraph, SimpleDocTemplate,
                                                Spacer)
            except ImportError as e:
                raise WorkflowError(
                    "reportlab library is required for PDF export. "
                    "Please install it: pip install reportlab",
                    context={"error": str(e)}
                ) from e
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18,
            )
            
            # Build PDF content
            styles = getSampleStyleSheet()
            story = []
            
            # Parse markdown with comprehensive handling
            lines = report.split("\n")
            in_list = False
            list_items: list[str] = []
            in_table = False
            table_rows: list[list[str]] = []
            
            i = 0
            while i < len(lines):
                line_stripped = lines[i].strip()
                
                # Handle empty lines
                if not line_stripped:
                    if in_list and list_items:
                        story.extend(self._format_list_items(list_items, styles))
                        list_items = []
                        in_list = False
                    if in_table and table_rows:
                        story.extend(self._format_table(table_rows, styles))
                        table_rows = []
                        in_table = False
                    story.append(Spacer(1, 0.2 * inch))
                    i += 1
                    continue
                
                # Handle horizontal rules
                if line_stripped.startswith("---") or line_stripped == "---":
                    if in_list and list_items:
                        story.extend(self._format_list_items(list_items, styles))
                        list_items = []
                        in_list = False
                    if in_table and table_rows:
                        story.extend(self._format_table(table_rows, styles))
                        table_rows = []
                        in_table = False
                    story.append(Spacer(1, 0.3 * inch))
                    # Add a line separator
                    from reportlab.platypus import HRFlowable
                    story.append(HRFlowable(
                        width="100%",
                        thickness=1,
                        spaceBefore=0.1 * inch,
                        spaceAfter=0.1 * inch,
                        color=(0.5, 0.5, 0.5)
                    ))
                    story.append(Spacer(1, 0.3 * inch))
                    i += 1
                    continue
                
                # Handle markdown tables
                if "|" in line_stripped and not line_stripped.startswith("#"):
                    # Check if it's a table separator line (---)
                    if re.match(r'^\|[\s\-:]+\|', line_stripped):
                        i += 1
                        continue
                    
                    # Parse table row
                    cells = [cell.strip() for cell in line_stripped.split("|") if cell.strip()]
                    if len(cells) > 1:  # Valid table row
                        in_table = True
                        table_rows.append(cells)
                        i += 1
                        continue
                
                # If we were in a table and now we're not, render the table
                if in_table and table_rows:
                    story.extend(self._format_table(table_rows, styles))
                    table_rows = []
                    in_table = False
                
                # Handle headings
                if line_stripped.startswith("# "):
                    if in_list and list_items:
                        story.extend(self._format_list_items(list_items, styles))
                        list_items = []
                        in_list = False
                    story.append(Paragraph(self._convert_markdown_to_html(line_stripped[2:]), styles["Title"]))
                    story.append(Spacer(1, 0.3 * inch))
                    i += 1
                    continue
                elif line_stripped.startswith("## "):
                    if in_list and list_items:
                        story.extend(self._format_list_items(list_items, styles))
                        list_items = []
                        in_list = False
                    story.append(Paragraph(self._convert_markdown_to_html(line_stripped[3:]), styles["Heading1"]))
                    story.append(Spacer(1, 0.2 * inch))
                    i += 1
                    continue
                elif line_stripped.startswith("### "):
                    if in_list and list_items:
                        story.extend(self._format_list_items(list_items, styles))
                        list_items = []
                        in_list = False
                    story.append(Paragraph(self._convert_markdown_to_html(line_stripped[4:]), styles["Heading2"]))
                    story.append(Spacer(1, 0.15 * inch))
                    i += 1
                    continue
                
                # Handle unordered lists (-, *, •)
                if line_stripped.startswith("- ") or line_stripped.startswith("* ") or line_stripped.startswith("• "):
                    in_list = True
                    list_item = line_stripped[2:].strip()
                    list_item = self._convert_markdown_to_html(list_item)
                    if list_item:
                        list_items.append(list_item)
                    i += 1
                    continue
                
                # Handle numbered lists
                if self._is_numbered_list_item(line_stripped):
                    in_list = True
                    parts = line_stripped.split(".", 1)
                    if len(parts) > 1:
                        list_item = parts[1].strip()
                        list_item = self._convert_markdown_to_html(list_item)
                        if list_item:
                            list_items.append(list_item)
                    i += 1
                    continue
                
                # Regular paragraph - check for inline lists
                if in_list and list_items:
                    story.extend(self._format_list_items(list_items, styles))
                    list_items = []
                    in_list = False
                
                # Check if paragraph contains inline lists (e.g., "Strengths: - item1 - item2")
                para_text = line_stripped
                if self._has_inline_list(para_text):
                    # Split paragraph and extract lists
                    parts = self._split_inline_lists(para_text)
                    for part in parts:
                        if isinstance(part, list):
                            # It's a list
                            story.extend(self._format_list_items(part, styles))
                        else:
                            # It's regular text
                            if part.strip():
                                story.append(Paragraph(self._convert_markdown_to_html(part), styles["Normal"]))
                                story.append(Spacer(1, 0.1 * inch))
                elif self._has_inline_numbered_list(para_text):
                    # Split paragraph and extract numbered lists
                    parts = self._split_inline_numbered_lists(para_text)
                    for part in parts:
                        if isinstance(part, list):
                            # It's a numbered list - format as numbered list
                            story.extend(self._format_numbered_list_items(part, styles))
                        else:
                            # It's regular text
                            if part.strip():
                                story.append(Paragraph(self._convert_markdown_to_html(part), styles["Normal"]))
                                story.append(Spacer(1, 0.1 * inch))
                else:
                    # Regular paragraph
                    para_text = self._convert_markdown_to_html(para_text)
                    if para_text.strip():
                        story.append(Paragraph(para_text, styles["Normal"]))
                        story.append(Spacer(1, 0.1 * inch))
                
                i += 1
            
            # Handle trailing list or table
            if in_list and list_items:
                story.extend(self._format_list_items(list_items, styles))
            if in_table and table_rows:
                story.extend(self._format_table(table_rows, styles))
            
            # Embed diagrams if available
            if viz_paths:
                story.append(Spacer(1, 0.3 * inch))
                story.append(Paragraph("Visualizations", styles["Heading1"]))
                story.append(Spacer(1, 0.2 * inch))
                
                # Embed SWOT diagram
                if "swot_diagram" in viz_paths:
                    self._embed_image(story, viz_paths["swot_diagram"], "SWOT Analysis")
                
                # Embed trends diagram
                if "trends_diagram" in viz_paths:
                    self._embed_image(story, viz_paths["trends_diagram"], "Market Trends")
                
                # Embed opportunities diagram
                if "opportunities_diagram" in viz_paths:
                    self._embed_image(story, viz_paths["opportunities_diagram"], "Business Opportunities")
            
            # Build PDF
            doc.build(story)
            
            return pdf_path
                
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}", exc_info=True)
            raise WorkflowError(
                "PDF generation failed",
                context={"error": str(e), "output_dir": str(output_dir)}
            ) from e
    
    def _is_numbered_list_item(self, line: str) -> bool:
        """Check if line is a numbered list item (e.g., '1. Item', '2. Item').
        
        Args:
            line: Line to check
            
        Returns:
            True if line is a numbered list item
        """
        pattern = r'^\d+\.\s+'
        return bool(re.match(pattern, line))
    
    def _format_list_items(
        self, items: list[str], styles: Any
    ) -> list[Any]:
        """Format list items for PDF using reportlab.
        
        Args:
            items: List of item strings
            styles: ReportLab stylesheet
            
        Returns:
            List of reportlab flowables (Paragraphs and Spacers)
        """
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, Spacer
        
        flowables = []
        for item in items:
            # Use bullet style for list items
            flowables.append(Paragraph(f"• {item}", styles["Normal"]))
            flowables.append(Spacer(1, 0.05 * inch))
        flowables.append(Spacer(1, 0.1 * inch))
        return flowables
    
    def _convert_markdown_to_html(self, text: str) -> str:
        """Convert markdown formatting to HTML for reportlab Paragraph.
        
        Converts **bold**, *italic*, and other markdown to HTML tags.
        
        Args:
            text: Markdown text to convert
            
        Returns:
            HTML-formatted text
        """
        if not text:
            return ""
        
        # Escape HTML special characters first
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # Convert bold **text** to <b>text</b>
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        
        # Convert italic *text* to <i>text</i> (but not if it's part of **text**)
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', text)
        
        return text
    
    def _has_inline_list(self, text: str) -> bool:
        """Check if text contains inline list patterns.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains inline list patterns
        """
        # Pattern: "Label: - item1 - item2" or "Label: * item1 * item2"
        patterns = [
            r':\s*-\s+[^-]',  # Colon followed by dash list
            r':\s*\*\s+[^*]',  # Colon followed by asterisk list
        ]
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _split_inline_lists(self, text: str) -> list[str | list[str]]:
        """Split text containing inline lists into text and list parts.
        
        Args:
            text: Text that may contain inline lists
            
        Returns:
            List of strings (text) and lists (list items)
        """
        result: list[str | list[str]] = []
        
        # Pattern to match "Label: - item1 - item2 - item3"
        pattern = r'([^:]+):\s*((?:-\s+[^-]+(?:$|\s+(?=-)))+)'
        match = re.search(pattern, text)
        
        if match:
            # Text before the list
            before = text[:match.start()].strip()
            if before:
                result.append(before)
            
            # Label
            label = match.group(1).strip()
            if label:
                result.append(f"{label}:")
            
            # Extract list items
            list_text = match.group(2)
            list_items = re.findall(r'-\s+([^-]+)', list_text)
            list_items = [item.strip() for item in list_items if item.strip()]
            if list_items:
                # Convert markdown to HTML for each item
                list_items = [self._convert_markdown_to_html(item) for item in list_items]
                result.append(list_items)
            
            # Text after the list
            after = text[match.end():].strip()
            if after:
                result.append(after)
        else:
            # No inline list found, return text as-is
            result.append(text)
        
        return result
    
    def _has_inline_numbered_list(self, text: str) -> bool:
        """Check if text contains inline numbered list patterns.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains inline numbered list patterns
        """
        # Pattern: "Text: 1. item1 2. item2 3. item3" or similar
        pattern = r'\d+\.\s+[^\d]+(?:\s+\d+\.\s+[^\d]+)+'
        return bool(re.search(pattern, text))
    
    def _split_inline_numbered_lists(self, text: str) -> list[str | list[str]]:
        """Split text containing inline numbered lists into text and list parts.
        
        Args:
            text: Text that may contain inline numbered lists
            
        Returns:
            List of strings (text) and lists (list items)
        """
        result: list[str | list[str]] = []
        
        # Pattern to match numbered lists: "1. item1 2. item2 3. item3"
        # Find the start of the first numbered item
        pattern = r'(.+?)(\d+\.\s+[^\d]+(?:\s+\d+\.\s+[^\d]+)+)'
        match = re.search(pattern, text)
        
        if match:
            # Text before the list
            before = match.group(1).strip()
            if before:
                result.append(before)
            
            # Extract numbered list items
            list_text = match.group(2)
            # Match pattern: "1. text 2. text 3. text"
            list_items = re.findall(r'(\d+)\.\s+([^\d]+?)(?=\s+\d+\.|$)', list_text)
            
            if list_items:
                # Extract just the text part (without numbers)
                formatted_items = []
                for num, item in list_items:
                    item_clean = item.strip()
                    if item_clean:
                        formatted_items.append(self._convert_markdown_to_html(item_clean))
                
                if formatted_items:
                    result.append(formatted_items)
            
            # Text after the list
            after = text[match.end():].strip()
            if after:
                result.append(after)
        else:
            # No inline numbered list found, return text as-is
            result.append(text)
        
        return result
    
    def _format_numbered_list_items(
        self, items: list[str], styles: Any
    ) -> list[Any]:
        """Format numbered list items for PDF using reportlab.
        
        Args:
            items: List of item strings
            styles: ReportLab stylesheet
            
        Returns:
            List of reportlab flowables (Paragraphs and Spacers)
        """
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, Spacer
        
        flowables = []
        for idx, item in enumerate(items, start=1):
            # Use numbered format: "1. item", "2. item", etc.
            flowables.append(Paragraph(f"{idx}. {item}", styles["Normal"]))
            flowables.append(Spacer(1, 0.05 * inch))
        flowables.append(Spacer(1, 0.1 * inch))
        return flowables
    
    def _format_table(
        self, rows: list[list[str]], styles: Any
    ) -> list[Any]:
        """Format markdown table as PDF table.
        
        Args:
            rows: List of table rows, each row is a list of cells
            styles: ReportLab stylesheet
            
        Returns:
            List of reportlab flowables
        """
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        
        if not rows or len(rows) == 0:
            return []
        
        flowables = []
        
        # Convert cells to Paragraphs with HTML formatting
        table_data = []
        for row in rows:
            formatted_row = []
            for cell in row:
                cell_html = self._convert_markdown_to_html(cell)
                formatted_row.append(Paragraph(cell_html, styles["Normal"]))
            table_data.append(formatted_row)
        
        # Create table
        table = Table(table_data)
        
        # Apply table style
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),  # Header row
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ])
        table.setStyle(table_style)
        
        flowables.append(table)
        flowables.append(Spacer(1, 0.2 * inch))
        
        return flowables
    
    def _embed_image(
        self, story: list[Any], image_path: str, caption: str
    ) -> None:
        """Embed an image in the PDF story.
        
        Args:
            story: List of reportlab flowables
            image_path: Path to image file
            caption: Caption text for the image
        """
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Image, Paragraph, Spacer
        
        try:
            styles = getSampleStyleSheet()
            img_path = Path(image_path)
            
            if img_path.exists():
                # Add caption
                story.append(Paragraph(caption, styles["Heading2"]))
                story.append(Spacer(1, 0.1 * inch))
                
                # Add image (scale to fit page width)
                img = Image(str(img_path), width=6 * inch, height=4.5 * inch)
                story.append(img)
                story.append(Spacer(1, 0.2 * inch))
            else:
                logger.warning(f"Image not found: {image_path}")
        except Exception as e:
            logger.warning(f"Failed to embed image {image_path}: {e}")
    
    def _generate_visualizations(
        self, insights: dict[str, Any], output_dir: Path, base_filename: str
    ) -> dict[str, str]:
        """Generate visualization images from insights.
        
        Creates visualizations such as SWOT diagrams, trend charts, etc.
        from the insights data.
        
        Args:
            insights: Insights dictionary containing SWOT, trends, etc.
            output_dir: Directory to save visualization files
            base_filename: Base filename (without extension)
        
        Returns:
            Dictionary mapping visualization type to file path
        
        Raises:
            WorkflowError: If visualization generation fails
        """
        viz_paths: dict[str, str] = {}
        
        try:
            # Try to generate diagrams if matplotlib is available
            try:
                import matplotlib.patches as mpatches
                import matplotlib.pyplot as plt

                # Generate SWOT diagram
                swot_data = insights.get("swot", {})
                if swot_data:
                    swot_path = self._create_swot_diagram(
                        swot_data, output_dir, base_filename
                    )
                    viz_paths["swot_diagram"] = str(swot_path)
                
                # Generate trends diagram if trends are available
                trends = insights.get("trends", [])
                if trends and len(trends) > 0:
                    trends_path = self._create_trends_diagram(
                        trends, output_dir, base_filename
                    )
                    viz_paths["trends_diagram"] = str(trends_path)
                
                # Generate opportunities diagram if opportunities are available
                opportunities = insights.get("opportunities", [])
                if opportunities and len(opportunities) > 0:
                    opp_path = self._create_opportunities_diagram(
                        opportunities, output_dir, base_filename
                    )
                    viz_paths["opportunities_diagram"] = str(opp_path)
                    
            except ImportError:
                logger.warning("matplotlib not available, skipping visualizations")
            except Exception as e:
                logger.warning(f"Failed to generate diagrams: {e}")
            
            return viz_paths
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}", exc_info=True)
            # Don't fail the entire export if visualizations fail
            return viz_paths
    
    def _create_swot_diagram(
        self, swot_data: dict[str, Any], output_dir: Path, base_filename: str
    ) -> Path:
        """Create a SWOT analysis diagram.
        
        Generates a 2x2 grid diagram showing Strengths, Weaknesses,
        Opportunities, and Threats.
        
        Args:
            swot_data: Dictionary with strengths, weaknesses, opportunities, threats
            output_dir: Directory to save the diagram
            base_filename: Base filename (without extension)
        
        Returns:
            Path to generated diagram image file
        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("SWOT Analysis", fontsize=16, fontweight="bold")
        
        # Define colors for each quadrant
        colors = {
            "strengths": "#90EE90",  # Light green
            "weaknesses": "#FFB6C1",  # Light pink
            "opportunities": "#87CEEB",  # Sky blue
            "threats": "#FFA07A",  # Light salmon
        }
        
        # Plot each SWOT category
        categories = [
            ("strengths", "Strengths", 0, 0),
            ("weaknesses", "Weaknesses", 0, 1),
            ("opportunities", "Opportunities", 1, 0),
            ("threats", "Threats", 1, 1),
        ]
        
        for category_key, category_title, row, col in categories:
            ax = axes[row, col]
            ax.set_facecolor(colors[category_key])
            ax.set_title(category_title, fontsize=14, fontweight="bold", pad=10)
            ax.axis("off")
            
            # Get items for this category
            items = swot_data.get(category_key, [])
            if items:
                y_pos = 0.9
                for item in items[:8]:  # Limit to 8 items for readability
                    ax.text(
                        0.05,
                        y_pos,
                        f"• {item}",
                        fontsize=10,
                        verticalalignment="top",
                        wrap=True,
                    )
                    y_pos -= 0.12
                    if y_pos < 0.1:
                        break
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    fontsize=12,
                    ha="center",
                    va="center",
                    style="italic",
                )
        
        plt.tight_layout()
        
        diagram_path = output_dir / f"{base_filename}_swot.png"
        plt.savefig(diagram_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        return diagram_path
    
    def _create_trends_diagram(
        self, trends: list[str], output_dir: Path, base_filename: str
    ) -> Path:
        """Create a trends visualization diagram.
        
        Generates a horizontal bar chart showing identified market trends.
        
        Args:
            trends: List of trend strings
            output_dir: Directory to save the diagram
            base_filename: Base filename for the diagram
            
        Returns:
            Path to the generated trends diagram image file
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(trends) * 0.5)))
        fig.suptitle("Market Trends", fontsize=16, fontweight="bold")
        
        # Create horizontal bar chart
        y_pos = np.arange(len(trends))
        colors = plt.cm.viridis(np.linspace(0, 1, len(trends)))
        
        bars = ax.barh(y_pos, [1] * len(trends), color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(trends, fontsize=11)
        ax.set_xlabel("Trend Importance", fontsize=12)
        ax.set_xlim(0, 1.2)
        ax.set_xticks([])
        ax.grid(axis="x", alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, trend) in enumerate(zip(bars, trends)):
            width = bar.get_width()
            ax.text(
                width + 0.05,
                bar.get_y() + bar.get_height() / 2,
                trend,
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold"
            )
        
        plt.tight_layout()
        trends_path = output_dir / f"{base_filename}_trends.png"
        plt.savefig(trends_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        return trends_path
    
    def _create_opportunities_diagram(
        self, opportunities: list[str], output_dir: Path, base_filename: str
    ) -> Path:
        """Create an opportunities visualization diagram.
        
        Generates a pie chart or bar chart showing business opportunities.
        
        Args:
            opportunities: List of opportunity strings
            output_dir: Directory to save the diagram
            base_filename: Base filename for the diagram
            
        Returns:
            Path to the generated opportunities diagram image file
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.suptitle("Business Opportunities", fontsize=16, fontweight="bold")
        
        # Use pie chart for opportunities
        if len(opportunities) <= 8:
            # Pie chart for small number of opportunities
            colors = plt.cm.Set3(np.linspace(0, 1, len(opportunities)))
            wedges, texts, autotexts = ax.pie(
                [1] * len(opportunities),
                labels=opportunities,
                autopct="",
                colors=colors,
                startangle=90,
                textprops={"fontsize": 10}
            )
            # Make labels more readable
            for text in texts:
                text.set_fontsize(9)
                text.set_fontweight("bold")
        else:
            # Bar chart for many opportunities
            y_pos = np.arange(len(opportunities))
            colors = plt.cm.plasma(np.linspace(0, 1, len(opportunities)))
            
            bars = ax.barh(y_pos, [1] * len(opportunities), color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(opportunities, fontsize=10)
            ax.set_xlabel("Opportunity Priority", fontsize=12)
            ax.set_xlim(0, 1.2)
            ax.set_xticks([])
            ax.grid(axis="x", alpha=0.3)
        
        plt.tight_layout()
        opp_path = output_dir / f"{base_filename}_opportunities.png"
        plt.savefig(opp_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        return opp_path
    
    @property
    def name(self) -> str:
        """Return agent name.
        
        Returns:
            String identifier for this agent
        """
        return "export_agent"

