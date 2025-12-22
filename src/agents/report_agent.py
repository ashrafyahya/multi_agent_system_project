"""Report agent for generating final formatted reports.

This module implements the ReportAgent that generates formatted competitor
analysis reports with all required sections including executive summary,
SWOT breakdown, competitor overview, and recommendations.
"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from src.agents.base_agent import BaseAgent, agent_error_handler
from src.agents.prompts.report_agent_prompts import SYSTEM_PROMPT
from src.agents.utils.report_parser import parse_report_response
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState
from src.graph.state_utils import update_state
from src.models.insight_model import Insight
from src.models.report_model import Report

logger = logging.getLogger(__name__)


class ReportAgent(BaseAgent):
    """Agent that generates final formatted competitor analysis reports.
    
    This agent creates comprehensive reports from business insights by:
    1. Extracting insights from workflow state
    2. Using LLM to generate structured report sections
    3. Formatting sections into a complete report
    4. Validating report against Report model
    5. Storing formatted report string in state
    
    The agent uses a higher temperature (> 0) for creative, well-written reports.
    
    Attributes:
        llm: Language model instance (injected)
        config: Configuration dictionary (injected)
    """

    def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute report generation from insights.
        
        Generates a formatted competitor analysis report by:
        1. Extracting insights from state
        2. Using LLM to generate report sections
        3. Validating report structure
        4. Formatting complete report string
        5. Storing report in state
        
        Args:
            state: Current workflow state containing insights
        
        Returns:
            Updated WorkflowState with report field populated
        
        Raises:
            WorkflowError: If insights are missing or report generation fails
        
        """
        try:
            insights_data = state.get("insights")
            if not insights_data:
                raise WorkflowError(
                    "Cannot generate report without insights",
                    context={"state_keys": list(state.keys())}
                )
            
            try:
                insights = Insight(**insights_data)
            except Exception as e:
                raise WorkflowError(
                    "Invalid insights structure",
                    context={"error": str(e), "insights_data": insights_data}
                ) from e
            
            logger.info("Generating report from insights")
            
            temperature = self.config.get("temperature", 0.7)
            if temperature <= 0:
                logger.warning(
                    f"Report agent should use temperature > 0 for creative output, "
                    f"got temperature={temperature}. Consider updating config."
                )
            
            report_data = self._generate_report_sections(insights, state)
            
            try:
                report = Report(**report_data)
                report_dict = report.model_dump()
            except ValidationError as e:
                logger.error(f"Report validation failed: {e}")
                raise WorkflowError(
                    "Generated report failed validation",
                    context={                    "validation_errors": str(e), "report_data": report_data}
                ) from e
            
            formatted_report = self._format_report(report_dict)
            
            new_state = update_state(
                state,
                report=formatted_report,
                current_task="Report generated successfully"
            )
            
            total_length = len(formatted_report)
            logger.info(
                f"Report generated successfully: {total_length} characters, "
                f"all sections included"
            )
            
            return new_state
            
        except WorkflowError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in report agent: {e}", exc_info=True)
            raise WorkflowError(
                "Report generation failed unexpectedly",
                context={"error": str(e)}
            ) from e
    
    @agent_error_handler("report_agent", "report")
    def _generate_report_sections(self, insights: Insight, state: WorkflowState) -> dict[str, Any]:
        """Generate report sections using LLM.
        
        Args:
            insights: Insight model containing SWOT, positioning, trends, etc.
            state: Workflow state that may contain collected_data with quantitative information
        
        Returns:
            Dictionary containing report sections
        
        Raises:
            WorkflowError: If LLM invocation fails or response cannot be parsed
        """
        insights_summary = self._prepare_insights_summary(insights, state)
        
        # Use HumanMessage and SystemMessage directly to avoid template parsing issues
        # when insights_summary or SYSTEM_PROMPT contains curly braces (e.g., JSON examples)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=insights_summary)
        ]
        
        response = self.invoke_llm(messages)
        
        content = response.content if hasattr(response, "content") else str(response)
        
        if not content:
            raise WorkflowError("LLM returned empty response")
        
        logger.debug(f"LLM response length: {len(content)} characters")
        logger.debug(f"LLM response preview: {content[:300]}...")
        
        try:
            report_data = parse_report_response(content)
        except WorkflowError as e:
            # Log more context when parsing fails
            logger.error(
                f"Failed to parse report response. "
                f"Response length: {len(content)}, "
                f"First 500 chars: {content[:500]}"
            )
            raise
        
        return report_data
    
    def _prepare_insights_summary(self, insights: Insight, state: WorkflowState) -> str:
        """Prepare insights summary for LLM report generation.
        
        Args:
            insights: Insight model
            state: Workflow state that may contain collected_data with quantitative information
        
        Returns:
            Formatted string summary of insights including quantitative data
        """
        summary_parts = ["Business Insights for Report Generation:\n"]
        
        summary_parts.append("\nSWOT Analysis:")
        summary_parts.append(f"  Strengths: {', '.join(insights.swot.strengths[:5])}")
        summary_parts.append(f"  Weaknesses: {', '.join(insights.swot.weaknesses[:5])}")
        summary_parts.append(f"  Opportunities: {', '.join(insights.swot.opportunities[:5])}")
        summary_parts.append(f"  Threats: {', '.join(insights.swot.threats[:5])}")
        
        summary_parts.append(f"\nMarket Positioning: {insights.positioning}")
        
        if insights.trends:
            summary_parts.append(f"\nMarket Trends: {', '.join(insights.trends[:5])}")
        
        if insights.opportunities:
            summary_parts.append(f"\nBusiness Opportunities: {', '.join(insights.opportunities[:5])}")
        
        collected_data = state.get("collected_data")
        source_urls: list[str] = []
        
        if collected_data and isinstance(collected_data, dict):
            competitors = collected_data.get("competitors", [])
            if competitors:
                quantitative_info, source_urls = self._format_quantitative_data(
                    competitors, source_urls
                )
                if quantitative_info:
                    summary_parts.append("\n\nQuantitative Data from Competitor Analysis:")
                    summary_parts.extend(quantitative_info)
                    summary_parts.append(
                        "\nNote: Please incorporate these quantitative details (pricing, market share, "
                        "revenue, user counts, etc.) into your report with specific numbers, percentages, "
                        "and monetary amounts. Include source citations for all quantitative claims using "
                        "the source URLs provided above."
                    )
        
        self._format_validation_warnings(state, summary_parts)
        self._format_source_urls(source_urls, summary_parts)
        
        return "\n".join(summary_parts)
    
    def _format_quantitative_data(
        self, competitors: list[dict[str, Any]], source_urls: list[str]
    ) -> tuple[list[str], list[str]]:
        """Format quantitative data from competitor information.
        
        Args:
            competitors: List of competitor dictionaries
            source_urls: List to collect source URLs (modified in place)
        
        Returns:
            Tuple of (quantitative_info_lines, updated_source_urls)
        """
        quantitative_info: list[str] = []
        
        for comp in competitors[:10]:  # Include more competitors for better data
            comp_info = self._format_competitor_data(comp, source_urls)
            if len(comp_info) > 1:
                quantitative_info.extend(comp_info)
        
        return quantitative_info, source_urls
    
    def _format_competitor_data(
        self, comp: dict[str, Any], source_urls: list[str]
    ) -> list[str]:
        """Format individual competitor data.
        
        Args:
            comp: Competitor dictionary
            source_urls: List to collect source URLs (modified in place)
        
        Returns:
            List of formatted competitor information lines
        """
        comp_info: list[str] = []
        
        if not isinstance(comp, dict):
            return comp_info
        
        name = comp.get("name", "Unknown")
        comp_info.append(f"  {name}:")
        
        source_url = comp.get("source_url")
        if source_url:
            source_url_str = str(source_url)
            if source_url_str not in source_urls:
                source_urls.append(source_url_str)
            comp_info.append(f"    - Source: {source_url_str}")
        
        pricing = comp.get("pricing")
        if pricing:
            comp_info.append(f"    - Pricing: {pricing}")
        
        market_share = comp.get("market_share")
        if market_share is not None:
            comp_info.append(f"    - Market Share: {market_share}%")
        
        revenue = comp.get("revenue")
        if revenue is not None:
            if isinstance(revenue, (int, float)):
                comp_info.append(f"    - Revenue: ${revenue:,.0f}")
            else:
                comp_info.append(f"    - Revenue: {revenue}")
        
        user_count = comp.get("user_count")
        if user_count is not None:
            comp_info.append(f"    - User Count: {user_count}")
        
        founded_year = comp.get("founded_year")
        if founded_year is not None:
            comp_info.append(f"    - Founded: {founded_year}")
        
        headquarters = comp.get("headquarters")
        if headquarters:
            comp_info.append(f"    - Headquarters: {headquarters}")
        
        key_features = comp.get("key_features")
        if key_features:
            comp_info.append(f"    - Key Features: {', '.join(key_features[:5])}")
        
        market_presence = comp.get("market_presence")
        if market_presence:
            comp_info.append(f"    - Market Presence: {market_presence[:150]}")
        
        excluded_keys = [
            "name", "website", "source_url", "pricing", "market_presence",
            "products", "market_share", "revenue", "user_count",
            "founded_year", "headquarters", "key_features"
        ]
        
        for key, value in comp.items():
            if key not in excluded_keys:
                if isinstance(value, (int, float)) or (
                    isinstance(value, str) and any(char.isdigit() for char in value)
                ):
                    comp_info.append(f"    - {key}: {value}")
        
        return comp_info
    
    def _format_validation_warnings(
        self, state: WorkflowState, summary_parts: list[str]
    ) -> None:
        """Format validation warnings for the summary.
        
        Args:
            state: Workflow state containing validation warnings
            summary_parts: List to append formatted warnings (modified in place)
        """
        validation_warnings = state.get("validation_warnings", [])
        if validation_warnings:
            summary_parts.append("\n\nData Validation Warnings:")
            summary_parts.append("The following data consistency issues were detected:")
            for warning in validation_warnings:
                summary_parts.append(f"  - {warning}")
            summary_parts.append(
                "\nIMPORTANT: Acknowledge these validation warnings in the methodology section. "
                "Use conservative estimates when data conflicts are detected. Clearly distinguish "
                "between verified data and estimates."
            )
    
    def _format_source_urls(
        self, source_urls: list[str], summary_parts: list[str]
    ) -> None:
        """Format source URLs for the summary.
        
        Args:
            source_urls: List of source URLs
            summary_parts: List to append formatted URLs (modified in place)
        """
        if not source_urls:
            return
        
        summary_parts.append(
            f"\n\nSource URLs ({len(source_urls)} sources) - Use these numbers in citations:"
        )
        source_mapping: dict[str, int] = {}
        for i, url in enumerate(source_urls[:20], 1):
            source_mapping[url] = i
            summary_parts.append(f"  [{i}] {url}")
        
        if len(source_urls) > 20:
            summary_parts.append(f"  ... and {len(source_urls) - 20} more sources")
        
        summary_parts.append(
            "\nCRITICAL: When citing sources in your report, use the numbered format [1], [2], [3], etc. "
            "Each number corresponds to the source URL listed above. For example: "
            "'Market leader with 35% market share [1]' where [1] refers to the first source URL. "
            "Include these source URLs in the 'sources' field of your report JSON output in the same order."
        )
    
    def _format_report(self, report_dict: dict[str, Any]) -> str:
        """Format report dictionary into a complete report string.
        
        Cleans and normalizes markdown content to ensure proper structure:
        - Removes any ## headings from section content (sections already have H2 headings)
        - Ensures proper spacing between sections
        - Normalizes line breaks
        
        Args:
            report_dict: Dictionary containing report sections
        
        Returns:
            Formatted report string with proper markdown structure
        """
        # Clean each section
        executive_summary = self._clean_section_content(
            report_dict.get("executive_summary", ""), "executive_summary"
        )
        swot_breakdown_raw = report_dict.get("swot_breakdown", "")
        swot_breakdown = self._fix_swot_formatting(swot_breakdown_raw)
        swot_breakdown = self._clean_section_content(
            swot_breakdown, "swot_breakdown"
        )
        competitor_overview_raw = report_dict.get("competitor_overview", "")
        # Fix competitor overview if competitors are on same line (common LLM issue)
        competitor_overview = self._fix_competitor_overview_formatting(competitor_overview_raw)
        competitor_overview = self._clean_section_content(competitor_overview, "competitor_overview")
        recommendations_raw = report_dict.get("recommendations", "")
        # Fix recommendations if items are on same line (common LLM issue)
        logger.debug(f"Raw recommendations content (first 200 chars): {recommendations_raw[:200]}")
        recommendations = self._fix_recommendations_formatting(recommendations_raw)
        logger.debug(f"Fixed recommendations content (first 200 chars): {recommendations[:200]}")
        recommendations = self._clean_section_content(recommendations, "recommendations")
        methodology_raw = report_dict.get("methodology", "") or ""
        methodology = self._clean_methodology_content(methodology_raw)
        
        sources_section = self._format_sources_section(report_dict.get("sources"))
        
        return self._assemble_report_sections(
            executive_summary,
            swot_breakdown,
            competitor_overview,
            recommendations,
            methodology,
            sources_section,
        )
    
    def _fix_swot_formatting(self, content: str) -> str:
        """Fix SWOT formatting by splitting items on same line and removing trailing "-".
        
        Handles three issues:
        1. Multiple SWOT categories on same line separated by "  ### " (two spaces + ###)
        2. SWOT items on same line separated by " - " (e.g., "### Strengths - Item 1 - Item 2")
        3. Trailing "-" characters on bullet points
        
        Args:
            content: Raw SWOT breakdown content
        
        Returns:
            Fixed content with proper formatting
        """
        if not content:
            return content
        
        # CRITICAL FIX: First, split by "  ### " or "\n  ### " (two spaces + ###) to separate multiple SWOT categories
        # This handles the case where LLM generates: "### Strengths - Item 1  ### Weaknesses - Item 2"
        # Also handles: "### Strengths\n- Item 1  ### Weaknesses\n- Item 2"
        # Use regex to handle both cases: "  ### " (with or without leading newline)
        import re
        if '  ### ' in content or '\n  ### ' in content:
            # Split by "  ### " or "\n  ### " to separate SWOT categories
            # Pattern: (optional newline) + two spaces + ###
            category_parts = re.split(r'\n?\s{2,}###\s+', content)
            fixed_parts: list[str] = []
            
            for i, part in enumerate(category_parts):
                part = part.strip()
                if not part:
                    continue
                
                # If this is the first part and doesn't start with "###", it might be text before first category
                if i == 0 and not part.startswith('###'):
                    # Check if it contains SWOT category name - might be a category line without ###
                    if any(cat in part for cat in ['Strengths', 'Weaknesses', 'Opportunities', 'Threats']):
                        # Add ### prefix
                        part = '### ' + part
                    fixed_parts.append(part)
                else:
                    # Add ### prefix if not present
                    if not part.startswith('###'):
                        part = '### ' + part
                    fixed_parts.append(part)
            
            # Join with newlines to put each category on its own line
            content = '\n\n'.join(fixed_parts)
        
        lines = content.split('\n')
        fixed_lines: list[str] = []
        
        for line in lines:
            stripped = line.strip()
            
            # Check if line starts with ### (SWOT category heading)
            if stripped.startswith('###'):
                # Check if heading has items on same line (separated by " - ")
                # Pattern: "### Strengths - Item 1 - Item 2 - Item 3"
                if ' - ' in stripped:
                    # Split heading from items
                    parts = stripped.split(' - ', 1)
                    heading = parts[0]  # "### Strengths"
                    items_str = parts[1] if len(parts) > 1 else ""
                    
                    # Add the heading
                    fixed_lines.append(heading)
                    
                    # Split items by " - " and format as bullet points
                    if items_str:
                        items = items_str.split(' - ')
                        for item in items:
                            item = item.strip()
                            # Remove trailing "-" if present (handle multiple patterns)
                            # Pattern 1: " - " (space-dash-space)
                            while item.rstrip().endswith(' -'):
                                item = item.rstrip()[:-2].rstrip()
                            # Pattern 2: Single trailing "-" after space
                            if item.rstrip().endswith('-') and len(item.rstrip()) > 1:
                                # Check if it's a trailing dash (not part of a word or citation)
                                # Only remove if preceded by space (not part of citation like "[1]-")
                                if item.rstrip()[-2] == ' ':
                                    item = item.rstrip()[:-1].rstrip()
                            # Pattern 3: Remove any trailing " -" (space-dash without trailing space)
                            if item.rstrip().endswith(' -'):
                                item = item.rstrip()[:-2].rstrip()
                            if item:
                                fixed_lines.append(f"- {item}")
                    continue
            
            # Check if line is a bullet point (starts with - or •)
            if stripped.startswith(('-', '•')):
                # Remove trailing " - " or "-" from bullet points
                leading_spaces = len(line) - len(line.lstrip())
                bullet_marker = stripped[0] if stripped else ''
                content_after_bullet = stripped[1:].strip() if len(stripped) > 1 else ''
                
                # Remove trailing " - " or "-" from content
                while content_after_bullet.rstrip().endswith(' -'):
                    content_after_bullet = content_after_bullet.rstrip()[:-2].rstrip()
                if content_after_bullet.rstrip().endswith('-') and len(content_after_bullet.rstrip()) > 1:
                    if content_after_bullet.rstrip()[-2] == ' ':
                        content_after_bullet = content_after_bullet.rstrip()[:-1].rstrip()
                
                # Reconstruct line
                if content_after_bullet:
                    fixed_line = ' ' * leading_spaces + bullet_marker + ' ' + content_after_bullet
                else:
                    fixed_line = line
                fixed_lines.append(fixed_line)
            else:
                # Keep non-bullet lines as-is
                fixed_lines.append(line)
        
        result = '\n'.join(fixed_lines)
        return result
    
    def _clean_section_content(self, content: str, section_name: str) -> str:
        """Clean section content to remove conflicting headings and normalize formatting.
        
        Args:
            content: Raw section content
            section_name: Name of the section for logging
        
        Returns:
            Cleaned content with proper formatting
        """
        if not content:
            return ""
        
        # Remove any ## (H2) headings from content (sections already have H2 headings)
        # Replace ## with ### to make them H3 subheadings
        content = re.sub(r'^##\s+', '### ', content, flags=re.MULTILINE)
        
        # Normalize multiple consecutive newlines to double newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Ensure content doesn't start with newlines
        content = content.strip()
        
        # Ensure content ends with proper spacing
        if content and not content.endswith('\n'):
            content += '\n'
        
        return content
    
    def _clean_methodology_content(self, content: str) -> str:
        """Clean methodology section content by removing URLs and links.
        
        Methodology should not contain URLs - those belong in the Sources section.
        However, citation references like [1], [2], etc. MUST be preserved.
        
        This method removes:
        - Full URLs (http://, https://)
        - URLs that follow citations like "[1] https://..." (but keeps the [1])
        - Any text patterns that look like URLs (www.xxx.com)
        
        This method preserves:
        - Citation references like [1], [2], [3] without URLs
        
        Args:
            content: Raw methodology content
            
        Returns:
            Cleaned methodology content without URLs but with citation references preserved
        """
        if not content:
            return ""
        
        # First apply standard cleaning
        content = self._clean_section_content(content, "methodology")
        
        # Remove citation patterns with URLs like "[1] https://..." or "[1]https://..."
        # But preserve the citation number by replacing with just the citation
        # Pattern: [1] https://... becomes [1]
        content = re.sub(r'(\[\d+\])\s*https?://[^\s]+', r'\1', content)
        content = re.sub(r'(\[\d+\])https?://[^\s]+', r'\1', content)
        
        # Remove standalone URLs (http://, https://) that are not part of citations
        # Pattern matches: http://... or https://... followed by whitespace or end of line
        content = re.sub(r'https?://[^\s]+', '', content)
        
        # Remove any remaining URL-like patterns (www.xxx.com)
        content = re.sub(r'www\.[^\s]+', '', content)
        
        # Clean up multiple spaces that might result from URL removal
        # But preserve spaces around citations like "data [1] and" -> "data [1] and"
        content = re.sub(r' +', ' ', content)
        
        # Remove trailing spaces and normalize line breaks
        content = re.sub(r'\n\s+', '\n', content)
        content = content.strip()
        
        # Ensure content ends with proper spacing
        if content and not content.endswith('\n'):
            content += '\n'
        
        return content
    
    def _fix_competitor_overview_formatting(self, content: str) -> str:
        """Fix competitor overview formatting when competitors are on same line.
        
        Detects when competitors are separated by " ### " on the same line
        or when a competitor has " - " separators instead of bullet points,
        and splits them into separate lines with proper formatting.
        
        Args:
            content: Raw competitor overview content
        
        Returns:
            Fixed content with competitors on separate lines
        """
        if not content:
            return content
        
        # First, handle competitors separated by " ### "
        if ' ### ' in content:
            # Split by " ### " to separate competitors
            parts = content.split(' ### ')
            fixed_parts: list[str] = []
            
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                
                # Format this competitor part
                formatted = self._format_single_competitor(part)
                if formatted:
                    fixed_parts.append(formatted)
                
                # Add blank line between competitors (except after last one)
                if i < len(parts) - 1:
                    fixed_parts.append('')
            
            content = '\n'.join(fixed_parts)
        
        # Also check if first competitor (or any competitor) has " - " format on one line
        # Pattern: "Competitor Name - detail1 - detail2 - detail3" (all on one line)
        lines = content.split('\n')
        fixed_lines: list[str] = []
        
        for line in lines:
            line = line.strip()
            if not line:
                fixed_lines.append('')
                continue
            
            # Check if line looks like "Competitor Name - detail - detail - detail"
            # but doesn't start with ### or - or •
            if not line.startswith(('###', '-', '•', '#')) and ' - ' in line:
                # Count how many " - " separators (should be multiple for this pattern)
                if line.count(' - ') >= 2:
                    # This looks like a competitor with details on one line
                    formatted = self._format_single_competitor(line)
                    if formatted:
                        fixed_lines.append(formatted)
                        continue
            
            # Line is already properly formatted, keep it
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _format_single_competitor(self, competitor_text: str) -> str:
        """Format a single competitor from text with " - " separators.
        
        Args:
            competitor_text: Text like "Competitor Name - detail1 - detail2 - detail3"
        
        Returns:
            Formatted competitor with ### heading and bullet points
        """
        competitor_text = competitor_text.strip()
        if not competitor_text:
            return ""
        
        # Remove leading ### if present
        if competitor_text.startswith('###'):
            competitor_text = competitor_text[3:].strip()
        
        # Check if it has " - " pattern
        if ' - ' not in competitor_text:
            # No details, just competitor name
            return f"### {competitor_text}"
        
        # Split into competitor name and details
        parts = competitor_text.split(' - ', 1)
        competitor_name = parts[0].strip()
        details_str = parts[1] if len(parts) > 1 else ""
        
        formatted_parts = [f"### {competitor_name}"]
        
        # Split details by " - " to create bullet points
        if details_str:
            detail_items = details_str.split(' - ')
            for detail in detail_items:
                detail = detail.strip()
                if detail:
                    formatted_parts.append(f"- {detail}")
        
        return '\n'.join(formatted_parts)
    
    def _fix_recommendations_formatting(self, content: str) -> str:
        """Fix recommendations formatting when items are on same line.
        
        Detects when recommendation items are separated by " - " on the same line
        and splits them into separate bullet points. Handles multi-line Priority items.
        Also handles cases where multiple Priority items are on the same line separated by "  ### ".
        
        Args:
            content: Raw recommendations content
        
        Returns:
            Fixed content with proper formatting
        """
        if not content:
            return content
        
        # CRITICAL FIX: First, split by "  ### " (two spaces + ###) to separate multiple Priority items on same line
        # This handles the case where LLM generates: "### Priority 1: ...  ### Priority 2: ...  ### Priority 3: ..."
        original_content_length = len(content) if content else 0
        if '  ### ' in content:
            # Split by "  ### " to separate Priority items
            priority_parts = content.split('  ### ')
            fixed_parts: list[str] = []
            
            for i, part in enumerate(priority_parts):
                part = part.strip()
                if not part:
                    continue
                
                # If this is the first part and doesn't start with "###", it might be text before first Priority
                if i == 0 and not part.startswith('###'):
                    # Check if it contains "Priority" - might be a Priority line without ###
                    if 'Priority' in part and ':' in part:
                        # Add ### prefix
                        part = '### ' + part
                    fixed_parts.append(part)
                else:
                    # Add ### prefix if not present
                    if not part.startswith('###'):
                        part = '### ' + part
                    fixed_parts.append(part)
            
            # Join with newlines to put each Priority on its own line
            content = '\n\n'.join(fixed_parts)
        
        # First, merge lines that are part of the same Priority item
        # This handles cases where Priority items are split across multiple lines
        lines = content.split('\n')
        merged_lines: list[str] = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this starts a Priority item (may have leading whitespace)
            if line.startswith('Priority') and ':' in line:
                # Collect all continuation lines until we hit a blank line, bullet point, or another Priority
                priority_parts = [line]
                i += 1
                
                while i < len(lines):
                    next_line = lines[i].strip()
                    # Stop if we hit a blank line, formatted heading, or another Priority
                    if (not next_line or 
                        next_line.startswith('### Priority') or
                        (next_line.startswith('Priority') and ':' in next_line)):
                        break
                    # Stop if we hit a bullet point that's not part of the Priority line
                    # (bullet points that are part of Priority will have " - " in them)
                    if next_line.startswith(('-', '•')) and ' - ' not in next_line:
                        break
                    # Continue collecting if it looks like part of the Priority line
                    # (contains " - " or doesn't start with formatting markers)
                    if ' - ' in next_line or not next_line.startswith(('###', '-', '•')):
                        priority_parts.append(next_line)
                        i += 1
                    else:
                        break
                
                # Join the priority parts into one line
                merged_lines.append(' '.join(priority_parts))
                continue
            else:
                # Regular line, add as is
                merged_lines.append(lines[i])
                i += 1
        
        merged_content = '\n'.join(merged_lines)
        logger.debug(f"Merged recommendations content (first 300 chars): {merged_content[:300]}")
        
        # Now process each line to format Priority items
        fixed_lines: list[str] = []
        processed_lines = merged_content.split('\n')
        
        for line in processed_lines:
            line = line.strip()
            if not line:
                fixed_lines.append('')
                continue
            
            # Check if line starting with ### Priority has details that need to be split
            if line.startswith('### Priority') and ' - ' in line:
                # Extract the heading and details
                # Pattern: "### Priority X: Title - detail1 - detail2 - detail3"
                # Find the colon after "Priority X:"
                colon_pos = line.find(':')
                if colon_pos != -1:
                    # Get text after colon (skip the "### Priority X:" part)
                    after_colon = line[colon_pos + 1:].strip()
                    
                    # Find first " - " after colon to separate title from details
                    first_dash = after_colon.find(' - ')
                    if first_dash != -1:
                        # Extract title (everything after colon up to first " - ")
                        title = after_colon[:first_dash].strip()
                        # Reconstruct heading: "### Priority X:" + " " + title
                        heading = f"{line[:colon_pos + 1]} {title}".strip()
                        # Extract details (everything after first " - ")
                        details_str = after_colon[first_dash + 3:].strip()
                        
                        # Format as ### Priority heading
                        fixed_lines.append(heading)
                        
                        # Split details by " - " to create bullet points
                        if details_str:
                            detail_items = details_str.split(' - ')
                            for detail in detail_items:
                                detail = detail.strip()
                                # Remove trailing " - " if present
                                detail = detail.rstrip(' -').strip()
                                if detail:
                                    fixed_lines.append(f"- {detail}")
                        continue
                # If parsing failed, just add the line as-is
                fixed_lines.append(line)
                continue
            
            # Already properly formatted (no details to split)
            if line.startswith('### Priority'):
                fixed_lines.append(line)
                continue
            
            # Check if line is a Priority item that needs formatting (without ### prefix)
            if line.startswith('Priority') and ':' in line:
                # Pattern: "Priority X: Title - detail1 - detail2 - detail3"
                colon_pos = line.find(':')
                if colon_pos == -1:
                    fixed_lines.append(line)
                    continue
                
                # Get text after colon
                after_colon = line[colon_pos + 1:].strip()
                
                # Find first " - " after colon to separate title from details
                first_dash = after_colon.find(' - ')
                if first_dash != -1:
                    # Extract title (everything after colon up to first " - ")
                    title = after_colon[:first_dash].strip()
                    # Reconstruct heading: "Priority X:" + " " + title
                    # line[:colon_pos + 1] gives us "Priority 1:" (everything up to and including colon)
                    heading = f"{line[:colon_pos + 1]} {title}".strip()
                    # Extract details (everything after first " - ")
                    details_str = after_colon[first_dash + 3:].strip()
                    
                    logger.debug(f"Priority line detected - heading: '{heading}', details_str length: {len(details_str)}")
                else:
                    # No " - " found, entire line is the heading
                    heading = line
                    details_str = ""
                    logger.debug(f"Priority line with no details - heading: '{heading}'")
                
                # Format as ### Priority heading
                fixed_lines.append(f"### {heading}")
                
                # Split details by " - " to create bullet points
                if details_str:
                    detail_items = details_str.split(' - ')
                    for detail in detail_items:
                        detail = detail.strip()
                        # Remove trailing " - " if present
                        detail = detail.rstrip(' -').strip()
                        if detail:
                            fixed_lines.append(f"- {detail}")
                continue
            
            # Check if line has multiple " - " but doesn't start with ### or -
            if not line.startswith(('###', '-', '•', '#')) and line.count(' - ') >= 2:
                # Split by " - " to create bullet points
                detail_items = line.split(' - ')
                for detail in detail_items:
                    detail = detail.strip()
                    detail = detail.rstrip(' -').strip()
                    if detail:
                        fixed_lines.append(f"- {detail}")
                continue
            
            # Handle lines that already have bullet points but might have trailing " - "
            if line.startswith(('•', '-')):
                line = line.rstrip(' -').strip()
                if line.startswith('•'):
                    line = line.replace('•', '-', 1)
                fixed_lines.append(line)
                continue
            
            # Line is already properly formatted, keep it
            fixed_lines.append(line)
        
        # Remove duplicate consecutive lines
        result_lines: list[str] = []
        prev_line = ""
        for line in fixed_lines:
            line_stripped = line.strip()
            if line_stripped != prev_line or not line_stripped:
                result_lines.append(line)
                prev_line = line_stripped
        
        final_result = '\n'.join(result_lines)
        return final_result
    
    def _format_sources_section(self, sources_list: list[str] | None) -> str:
        """Format sources section as numbered list.
        
        Sources must be in the same order as provided to LLM to match citation numbers.
        
        Args:
            sources_list: List of source URLs or None
        
        Returns:
            Formatted sources section string, empty if no sources
        """
        if not sources_list or not isinstance(sources_list, list) or len(sources_list) == 0:
            return ""
        
        sources_lines = ["## Sources"]
        # Format as numbered list items (1. source, 2. source, etc.)
        for i, source in enumerate(sources_list, 1):
            # Remove any existing numbering from source if present
            source_clean = str(source).strip()
            # Remove leading numbers if already present (e.g., "1. https://..." -> "https://...")
            source_clean = re.sub(r'^\d+\.\s*', '', source_clean)
            # Format as numbered list item: "1. URL" (one per line)
            sources_lines.append(f"{i}. {source_clean}")
        
        return "\n".join(sources_lines)
    
    def _assemble_report_sections(
        self,
        executive_summary: str,
        swot_breakdown: str,
        competitor_overview: str,
        recommendations: str,
        methodology: str,
        sources_section: str,
    ) -> str:
        """Assemble report sections into complete report string.
        
        Args:
            executive_summary: Cleaned executive summary section
            swot_breakdown: Cleaned SWOT breakdown section
            competitor_overview: Cleaned competitor overview section
            recommendations: Cleaned recommendations section
            methodology: Cleaned methodology section (may be empty)
            sources_section: Formatted sources section (may be empty)
        
        Returns:
            Complete formatted report string
        """
        sections = [
            "# Competitor Analysis Report\n",
            "## Executive Summary\n",
            executive_summary,
        ]
        
        # Add methodology after executive summary if present
        if methodology:
            sections.extend([
                "\n\n## Methodology\n",
                methodology,
            ])
        
        sections.extend([
            "\n\n## SWOT Analysis Breakdown\n",
            swot_breakdown,
            "\n\n## Competitor Overview\n",
            competitor_overview,
            "\n\n## Strategic Recommendations\n",
            recommendations,
        ])
        
        # Add sources section at the end if present
        if sources_section:
            sections.extend([
                "\n\n",
                sources_section,
            ])
        
        final_report = "".join(sections)
        return final_report
    
    @property
    def name(self) -> str:
        """Return agent name.
        
        Returns:
            String identifier for this agent
        """
        return "report_agent"
