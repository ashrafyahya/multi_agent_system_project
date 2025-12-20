"""Report agent for generating final formatted reports.

This module implements the ReportAgent that generates formatted competitor
analysis reports with all required sections including executive summary,
SWOT breakdown, competitor overview, and recommendations.
"""

import json
import logging
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from src.agents.base_agent import BaseAgent
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState
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
    
    SYSTEM_PROMPT = """You are a senior business analyst and strategic consultant specializing in competitive intelligence and market analysis reports for C-level executives and strategic decision-makers.

Your task is to create a comprehensive, executive-ready competitor analysis report that combines strategic insights with quantitative data to inform critical business decisions.

**Report Structure & Requirements:**

1. **Executive Summary** (minimum 200 characters, recommended 300-500):
   - Start with a compelling one-sentence summary of the competitive landscape
   - Highlight 3-5 key findings with quantitative data
   - Include market size, growth rate, competitive intensity metrics
   - Summarize strategic implications and recommended actions
   - Write for busy executives: clear, concise, data-driven
   - Example opening: "The CRM market is highly competitive with 5 major players controlling 70% market share, growing at 15% CAGR. Our analysis reveals..."

2. **SWOT Breakdown** (minimum 300 characters, recommended 500-800):
   - Organize by SWOT category (Strengths, Weaknesses, Opportunities, Threats)
   - For each SWOT item:
     * State the insight clearly
     * Provide quantitative evidence (market share %, revenue, growth rates, user metrics)
     * Explain business implications
     * Connect to strategic opportunities or risks
   - Include comparative analysis when relevant
   - Use data to support every major point
   - **CRITICAL FORMATTING**: Use ONLY bullet points (-) or numbered lists. DO NOT use markdown headings (##, ###) inside this section. Format as:
     * ### Strengths
     * - Item 1 with quantitative data
     * - Item 2 with quantitative data
     * ### Weaknesses
     * - Item 1
     * - Item 2
     * (Use ### for subheadings within the section, not ##)

3. **Competitor Overview** (minimum 300 characters, recommended 500-1000):
   - Provide detailed overview of each major competitor analyzed
   - For each competitor include:
     * Market position (leader, challenger, follower, nicher)
     * Key metrics: market share %, revenue, user base, growth rate
     * Pricing strategy and tiers (with specific $ amounts)
     * Key strengths and differentiators
     * Target market segments
   - Include comparative analysis: side-by-side comparison of key metrics
   - Highlight competitive gaps and white spaces
   - **CRITICAL FORMATTING**: Use markdown tables (|) for comparisons. Use ### for competitor names as subheadings. Use bullet points for details. DO NOT use ## headings.

4. **Strategic Recommendations** (minimum 300 characters, recommended 400-600):
   - Prioritize recommendations by impact and feasibility
   - Structure as: Priority 1 (High Impact, High Feasibility), Priority 2, Priority 3
   - For each recommendation:
     * State the recommendation clearly
     * Provide quantitative targets or benchmarks
     * Explain expected impact (with metrics when possible)
     * Include implementation considerations
   - Focus on actionable, measurable recommendations
   - Connect recommendations directly to SWOT findings
   - **CRITICAL FORMATTING**: Use ### for Priority headings. Use bullet points (-) for details. DO NOT use ## headings.
   - Example: "### Priority 1: Target SMB Market Segment\n- Develop software targeting SMB segment worth $500M, growing at 25% YoY\n- Offer competitive pricing: $49/month vs. competitor average $79/month\n- Target: Capture 5% market share within 12 months, revenue target $25M"

**Writing Style & Quality Standards:**
- Professional, executive-level business writing
- Data-driven: Include numbers, percentages, dollar amounts in every major section
- Clear and concise: Avoid jargon, use plain business language
- Actionable: Every insight should lead to a decision or action
- Structured: Use headings, bullet points, and clear organization
- Evidence-based: Support claims with quantitative data from the analysis

**Quantitative Data Requirements:**
- Executive Summary: Include at least 3-5 key metrics with source citations when available
- SWOT Breakdown: Include quantitative data in at least 50% of SWOT items with source citations
- Competitor Overview: Include metrics for each competitor (pricing, market share, revenue, growth) with source citations
- Recommendations: Include quantitative targets for at least 70% of recommendations
- Methodology: Include data quality assessment and validation notes

**Total Report Length:**
- Minimum: 1,200 characters (approximately 200-250 words)
- Recommended: 2,000-3,000 characters (approximately 400-600 words)
- Maximum: 5,000 characters to maintain readability

**CRITICAL FORMATTING RULES:**
- DO NOT use ## (H2) headings inside section content - sections already have H2 headings
- Use ### (H3) for subheadings within sections (e.g., ### Strengths, ### Priority 1)
- Use bullet points (-) for lists, not numbered lists unless specifically needed
- Use markdown tables (|) for comparative data
- Keep paragraphs concise (2-4 sentences max)
- Use line breaks (\n\n) between major subsections
- Ensure proper markdown syntax: no orphaned headings, properly closed lists

5. **Methodology** (minimum 200 characters, recommended 300-500):
   - Describe the data collection approach (web search, scraping, sources analyzed)
   - Include number of sources analyzed and data collection date/time if available
   - Explain validation approach and data quality assessment
   - Note any limitations, assumptions, or data quality issues
   - Acknowledge data inconsistencies if validation warnings were provided
   - Clearly distinguish between verified data and estimates
   - Use conservative estimates when data conflicts
   - **CRITICAL FORMATTING**: Plain text, no headings. Use bullet points for lists.

6. **Sources** (optional but recommended):
   - List all source URLs used in the analysis
   - Format as a simple list or bullet points
   - Include source URLs for key quantitative claims when available

**Source Citation Requirements:**
- **CRITICAL**: All quantitative claims MUST include source references when available
- Format citations as numbered references: "Market leader with 35% market share [1]"
- Use square brackets with numbers: [1], [2], [3], etc.
- Each number corresponds to a source URL that will be listed at the end of the report
- The source numbers will be provided to you in the input - use the exact number assigned to each URL
- If sources are not available for specific claims, note this in the methodology section
- Clearly distinguish between verified data (with sources) and estimates (without sources)
- Example: "Competitor A holds 35% market share [1] with revenue of $2B [2]"

**Data Validation Requirements:**
- If validation warnings are provided, acknowledge them in the methodology section
- Use conservative estimates when data conflicts are detected
- Clearly state data quality confidence levels (high/medium/low) when appropriate
- Note any data inconsistencies and their potential impact on conclusions

**Output Format:**
Return your report as a valid JSON object with this exact structure:
{{
    "executive_summary": "Executive summary text with quantitative data and numbered source citations like [1], [2] (plain text, no headings)...",
    "swot_breakdown": "SWOT analysis with ### subheadings, bullet points, and numbered source citations like [1], [2]...",
    "competitor_overview": "Competitor overview with ### subheadings, tables, and numbered source citations like [1], [2]...",
    "recommendations": "Strategic recommendations with ### Priority subheadings and bullet points...",
    "methodology": "Methodology section describing data collection, validation, and limitations (minimum 200 characters)...",
    "sources": ["https://source1.com", "https://source2.com", ...],
    "min_length": 1200
}}

**CRITICAL: Source Numbering:**
- The 'sources' array in your JSON output must contain the source URLs in the EXACT SAME ORDER as provided in the input
- When you cite a source in the text, use [1] for the first source, [2] for the second source, etc.
- The numbers in your citations [1], [2], [3] must match the position of the URL in the 'sources' array (1-based indexing)
- Example: If sources = ["https://example.com", "https://test.com"], then [1] refers to "https://example.com" and [2] refers to "https://test.com"

**Quality Checklist:**
- ✓ All sections meet minimum length requirements
- ✓ Quantitative data included throughout (percentages, $ amounts, metrics)
- ✓ Numbered source citations [1], [2], [3] included for all quantitative claims when sources are available
- ✓ Source numbers in citations match the position in the 'sources' array (1-based indexing)
- ✓ Methodology section included (minimum 200 characters) describing data collection and validation
- ✓ Data validation warnings acknowledged in methodology if provided
- ✓ Executive summary is compelling and data-driven (plain text, no headings)
- ✓ SWOT analysis uses ### subheadings and bullet points with numbered source citations (no ## headings)
- ✓ Competitor overview includes tables and structured comparisons with numbered source citations (### subheadings)
- ✓ Recommendations use ### Priority subheadings and bullet points (no ## headings)
- ✓ Professional writing style suitable for executive audience
- ✓ Proper markdown formatting (no nested ## headings, proper list syntax)
- ✓ JSON is valid and properly formatted
- ✓ Sources list included when source URLs are available, in the same order as provided in input"""

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
            
            new_state = state.copy()
            new_state["report"] = formatted_report
            new_state["current_task"] = "Report generated successfully"
            
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
        try:
            insights_summary = self._prepare_insights_summary(insights, state)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_PROMPT),
                ("human", insights_summary)
            ])
            
            messages = prompt.format_messages()
            response = self.llm.invoke(messages)
            
            content = response.content if hasattr(response, "content") else str(response)
            
            if not content:
                raise WorkflowError("LLM returned empty response")
            
            logger.debug(f"LLM response: {content[:200]}...")
            
            report_data = self._parse_report_response(content)
            
            return report_data
            
        except WorkflowError:
            raise
        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            raise WorkflowError(
                "Failed to generate report from LLM",
                context={"error": str(e)}
            ) from e
    
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
                summary_parts.append("\n\nQuantitative Data from Competitor Analysis:")
                quantitative_info = []
                
                for comp in competitors[:10]:  # Include more competitors for better data
                    comp_info = []
                    if isinstance(comp, dict):
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
                            if isinstance(pricing, dict):
                                comp_info.append(f"    - Pricing: {pricing}")
                            else:
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
                        
                        for key, value in comp.items():
                            if key not in [
                                "name", "website", "source_url", "pricing", "market_presence",
                                "products", "market_share", "revenue", "user_count",
                                "founded_year", "headquarters", "key_features"
                            ]:
                                if isinstance(value, (int, float)) or (isinstance(value, str) and any(char.isdigit() for char in value)):
                                    comp_info.append(f"    - {key}: {value}")
                    
                    if len(comp_info) > 1:
                        quantitative_info.extend(comp_info)
                
                if quantitative_info:
                    summary_parts.extend(quantitative_info)
                    summary_parts.append(
                        "\nNote: Please incorporate these quantitative details (pricing, market share, "
                        "revenue, user counts, etc.) into your report with specific numbers, percentages, "
                        "and monetary amounts. Include source citations for all quantitative claims using "
                        "the source URLs provided above."
                    )
        
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
        
        if source_urls:
            summary_parts.append(f"\n\nSource URLs ({len(source_urls)} sources) - Use these numbers in citations:")
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
        
        return "\n".join(summary_parts)
    
    def _parse_report_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response into report dictionary.
        
        Args:
            content: LLM response content (may contain JSON or markdown)
        
        Returns:
            Dictionary containing report sections
        
        Raises:
            WorkflowError: If JSON cannot be parsed or structure is invalid
        """
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content
        
        try:
            json_str_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
            json_str_clean = re.sub(r',\s*([}\]])', r'\1', json_str_clean)
            report_data = json.loads(json_str_clean)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Response content: {content[:500]}")
            try:
                json_str_fixed = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                report_data = json.loads(json_str_fixed)
                logger.info("Successfully parsed JSON after fixing control characters")
            except json.JSONDecodeError:
                raise WorkflowError(
                    "Failed to parse report from LLM response",
                    context={"error": str(e), "content_preview": content[:500]}
                ) from e
        
        required_fields = [
            "executive_summary",
            "swot_breakdown",
            "competitor_overview",
            "recommendations",
        ]
        missing_fields = [field for field in required_fields if field not in report_data]
        
        if missing_fields:
            raise WorkflowError(
                f"Report missing required sections: {missing_fields}",
                context={"report_data": report_data}
            )
        
        for field in required_fields:
            if not isinstance(report_data.get(field), str):
                raise WorkflowError(
                    f"Report section '{field}' must be a string",
                    context={"report_data": report_data}
                )
        
        if "methodology" in report_data:
            if not isinstance(report_data.get("methodology"), str):
                raise WorkflowError(
                    "Report section 'methodology' must be a string if provided",
                    context={"report_data": report_data}
                )
        else:
            report_data["methodology"] = None
        
        if "sources" in report_data:
            if not isinstance(report_data.get("sources"), list):
                raise WorkflowError(
                    "Report field 'sources' must be a list if provided",
                    context={"report_data": report_data}
                )
            report_data["sources"] = [str(s) for s in report_data["sources"] if s]
        else:
            report_data["sources"] = None
        
        report_data.setdefault("min_length", 1200)
        
        return report_data
    
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
        import re
        
        def clean_section_content(content: str, section_name: str) -> str:
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
        
        # Clean each section
        executive_summary = clean_section_content(
            report_dict.get("executive_summary", ""), "executive_summary"
        )
        swot_breakdown = clean_section_content(
            report_dict.get("swot_breakdown", ""), "swot_breakdown"
        )
        competitor_overview = clean_section_content(
            report_dict.get("competitor_overview", ""), "competitor_overview"
        )
        recommendations = clean_section_content(
            report_dict.get("recommendations", ""), "recommendations"
        )
        methodology = clean_section_content(
            report_dict.get("methodology", "") or "", "methodology"
        )
        
        # Format sources section as numbered list
        # Sources must be in the same order as provided to LLM to match citation numbers
        sources_list = report_dict.get("sources")
        sources_section = ""
        if sources_list and isinstance(sources_list, list) and len(sources_list) > 0:
            sources_lines = ["## Sources"]
            # Format as numbered list items (1. source, 2. source, etc.)
            # This format will be detected by markdown parser as numbered list
            # Each line should be a separate numbered list item
            for i, source in enumerate(sources_list, 1):
                # Remove any existing numbering from source if present
                source_clean = str(source).strip()
                # Remove leading numbers if already present (e.g., "1. https://..." -> "https://...")
                source_clean = re.sub(r'^\d+\.\s*', '', source_clean)
                # Format as numbered list item: "1. URL" (one per line)
                sources_lines.append(f"{i}. {source_clean}")
            sources_section = "\n".join(sources_lines)
        
        # Build report with proper structure
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
        
        return "".join(sections)
    
    @property
    def name(self) -> str:
        """Return agent name.
        
        Returns:
            String identifier for this agent
        """
        return "report_agent"
