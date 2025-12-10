"""Report agent for generating final formatted reports.

This module implements the ReportAgent that generates formatted competitor
analysis reports with all required sections including executive summary,
SWOT breakdown, competitor overview, and recommendations.

Example:
    ```python
    from src.agents.report_agent import ReportAgent
    from langchain_groq import ChatGroq
    from src.graph.state import create_initial_state
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    config = {"temperature": 0.7}
    agent = ReportAgent(llm=llm, config=config)
    
    state = create_initial_state("Analyze competitors")
    state["insights"] = {...}
    updated_state = agent.execute(state)
    ```
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
    
    Example:
        ```python
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=llm, config=config)
        
        state = create_initial_state("Analyze competitors")
        state["insights"] = {...}
        updated_state = agent.execute(state)
        ```
    """
    
    SYSTEM_PROMPT = """You are an expert business analyst and report writer specializing in competitor analysis.

Your task is to create a comprehensive, well-written competitor analysis report based on business insights.

Generate a professional report with the following sections:

1. **Executive Summary** (minimum 50 characters):
   - Provide a concise overview of the key findings
   - Highlight the most important insights
   - Summarize the competitive landscape

2. **SWOT Breakdown** (minimum 50 characters):
   - Detailed analysis of strengths, weaknesses, opportunities, and threats
   - Explain each SWOT component with context
   - Connect SWOT items to business implications

3. **Competitor Overview** (minimum 50 characters):
   - Overview of the competitors analyzed
   - Key characteristics and market positions
   - Comparative analysis where relevant

4. **Recommendations** (minimum 50 characters):
   - Strategic recommendations based on the analysis
   - Actionable insights for decision-making
   - Prioritized suggestions

The total report should be at least 500 characters. Write in a professional, clear, and engaging style.

Return your report as a JSON object with this exact structure:
{{
    "executive_summary": "Executive summary text...",
    "swot_breakdown": "SWOT analysis breakdown...",
    "competitor_overview": "Competitor overview...",
    "recommendations": "Strategic recommendations...",
    "min_length": 500
}}

Ensure all sections are well-written, informative, and meet minimum length requirements."""

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
        
        Example:
            ```python
            state = create_initial_state("Analyze competitors")
            state["insights"] = {...}
            updated_state = agent.execute(state)
            assert updated_state["report"] is not None
            ```
        """
        try:
            # Extract insights
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
            
            # Check temperature (should be > 0 for creativity)
            temperature = self.config.get("temperature", 0.7)
            if temperature <= 0:
                logger.warning(
                    f"Report agent should use temperature > 0 for creative output, "
                    f"got temperature={temperature}. Consider updating config."
                )
            
            # Generate report sections using LLM
            report_data = self._generate_report_sections(insights)
            
            # Validate report against Report model
            try:
                report = Report(**report_data)
                report_dict = report.model_dump()
            except ValidationError as e:
                logger.error(f"Report validation failed: {e}")
                raise WorkflowError(
                    "Generated report failed validation",
                    context={"validation_errors": str(e), "report_data": report_data}
                ) from e
            
            # Format complete report string
            formatted_report = self._format_report(report_dict)
            
            # Update state
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
    
    def _generate_report_sections(self, insights: Insight) -> dict[str, Any]:
        """Generate report sections using LLM.
        
        Args:
            insights: Insight model containing SWOT, positioning, trends, etc.
        
        Returns:
            Dictionary containing report sections
        
        Raises:
            WorkflowError: If LLM invocation fails or response cannot be parsed
        """
        try:
            # Prepare insights summary for LLM
            insights_summary = self._prepare_insights_summary(insights)
            
            # Create prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_PROMPT),
                ("human", insights_summary)
            ])
            
            # Invoke LLM
            messages = prompt.format_messages()
            response = self.llm.invoke(messages)
            
            # Extract content
            content = response.content if hasattr(response, "content") else str(response)
            
            if not content:
                raise WorkflowError("LLM returned empty response")
            
            logger.debug(f"LLM response: {content[:200]}...")
            
            # Parse report from response
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
    
    def _prepare_insights_summary(self, insights: Insight) -> str:
        """Prepare insights summary for LLM report generation.
        
        Args:
            insights: Insight model
        
        Returns:
            Formatted string summary of insights
        """
        summary_parts = ["Business Insights for Report Generation:\n"]
        
        # SWOT Analysis
        summary_parts.append("\nSWOT Analysis:")
        summary_parts.append(f"  Strengths: {', '.join(insights.swot.strengths[:5])}")
        summary_parts.append(f"  Weaknesses: {', '.join(insights.swot.weaknesses[:5])}")
        summary_parts.append(f"  Opportunities: {', '.join(insights.swot.opportunities[:5])}")
        summary_parts.append(f"  Threats: {', '.join(insights.swot.threats[:5])}")
        
        # Positioning
        summary_parts.append(f"\nMarket Positioning: {insights.positioning}")
        
        # Trends
        if insights.trends:
            summary_parts.append(f"\nMarket Trends: {', '.join(insights.trends[:5])}")
        
        # Opportunities
        if insights.opportunities:
            summary_parts.append(f"\nBusiness Opportunities: {', '.join(insights.opportunities[:5])}")
        
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
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content
        
        try:
            # Clean JSON string: remove control characters that can break JSON parsing
            # Replace common problematic control characters with spaces
            json_str_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
            # Remove trailing commas before closing braces/brackets
            json_str_clean = re.sub(r',\s*([}\]])', r'\1', json_str_clean)
            report_data = json.loads(json_str_clean)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Response content: {content[:500]}")
            # Try to fix common JSON issues and retry
            try:
                # Try escaping newlines and other control characters in strings
                json_str_fixed = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                report_data = json.loads(json_str_fixed)
                logger.info("Successfully parsed JSON after fixing control characters")
            except json.JSONDecodeError:
                raise WorkflowError(
                    "Failed to parse report from LLM response",
                    context={"error": str(e), "content_preview": content[:500]}
                ) from e
        
        # Validate required fields
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
        
        # Ensure all sections are strings
        for field in required_fields:
            if not isinstance(report_data.get(field), str):
                raise WorkflowError(
                    f"Report section '{field}' must be a string",
                    context={"report_data": report_data}
                )
        
        # Set default min_length if not provided
        report_data.setdefault("min_length", 500)
        
        return report_data
    
    def _format_report(self, report_dict: dict[str, Any]) -> str:
        """Format report dictionary into a complete report string.
        
        Args:
            report_dict: Dictionary containing report sections
        
        Returns:
            Formatted report string
        """
        sections = [
            "# Competitor Analysis Report\n",
            "## Executive Summary\n",
            report_dict.get("executive_summary", ""),
            "\n\n## SWOT Analysis Breakdown\n",
            report_dict.get("swot_breakdown", ""),
            "\n\n## Competitor Overview\n",
            report_dict.get("competitor_overview", ""),
            "\n\n## Strategic Recommendations\n",
            report_dict.get("recommendations", ""),
        ]
        
        return "".join(sections)
    
    @property
    def name(self) -> str:
        """Return agent name.
        
        Returns:
            String identifier for this agent
        """
        return "report_agent"
