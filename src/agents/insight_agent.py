"""Insight agent for transforming data into business insights.

This module implements the InsightAgent that transforms collected competitor
data into business insights, generates SWOT analysis, and extracts trends
and opportunities.
"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from src.agents.base_agent import BaseAgent, agent_error_handler
from src.agents.prompts.insight_agent_prompts import SYSTEM_PROMPT
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState
from src.graph.state_utils import update_state
from src.models.competitor_profile import CompetitorProfile
from src.models.insight_model import SWOT, Insight
from src.utils.metrics import track_execution_time

logger = logging.getLogger(__name__)


class InsightAgent(BaseAgent):
    """Agent that transforms competitor data into business insights.
    
    This agent analyzes collected competitor data and generates:
    1. SWOT analysis (strengths, weaknesses, opportunities, threats)
    2. Market positioning analysis
    3. Market trends identification
    4. Business opportunities extraction
    
    The agent uses an LLM to analyze the data and extract meaningful insights,
    then structures the output as an Insight model.
    
    Attributes:
        llm: Language model instance (injected)
        config: Configuration dictionary (injected)
    
    """

    @track_execution_time("insight_agent")
    def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute insight generation from collected data.
        
        Transforms collected competitor data into business insights by:
        1. Extracting competitor data from state
        2. Analyzing data using LLM
        3. Generating SWOT analysis
        4. Extracting trends and opportunities
        5. Structuring output as Insight model
        
        Args:
            state: Current workflow state containing collected_data
        
        Returns:
            Updated WorkflowState with insights field populated
        
        Raises:
            WorkflowError: If collected_data is missing or insight generation fails
        """
        try:
            # Extract collected data
            collected_data = state.get("collected_data")
            if not collected_data:
                raise WorkflowError(
                    "Cannot generate insights without collected data",
                    context={"state_keys": list(state.keys())}
                )
            
            competitors_data = collected_data.get("competitors", [])
            if not competitors_data:
                # Handle empty competitor data gracefully
                logger.warning(
                    "No competitor data available for analysis. "
                    "Generating minimal insights to allow workflow continuation."
                )
                insight_data = self._generate_minimal_insights()
            else:
                logger.info(f"Generating insights from {len(competitors_data)} competitors")
                
                # Generate insights using LLM
                insight_data = self._generate_insights(competitors_data)
            
            # Clean and normalize insight data before validation
            insight_data = self._clean_insight_data(insight_data)
            
            # Validate and structure as Insight model
            try:
                insight = Insight(**insight_data)
                insight_dict = insight.model_dump()
            except ValidationError as e:
                logger.error(f"Insight validation failed: {e}")
                raise WorkflowError(
                    "Generated insights failed validation",
                    context={"validation_errors": str(e), "insight_data": insight_data}
                ) from e
            
            # Update state
            new_state = update_state(
                state,
                insights=insight_dict,
                current_task="Insights generated successfully"
            )
            
            # Log insight generation result
            strengths_count = len(insight_dict.get('swot', {}).get('strengths', []))
            weaknesses_count = len(insight_dict.get('swot', {}).get('weaknesses', []))
            trends_count = len(insight_dict.get('trends', []))
            logger.info(
                f"Insights generated: {strengths_count} strengths, "
                f"{weaknesses_count} weaknesses, {trends_count} trends"
            )
            
            return new_state
            
        except WorkflowError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in insight agent: {e}", exc_info=True)
            raise WorkflowError(
                "Insight generation failed unexpectedly",
                context={"error": str(e)}
            ) from e
    
    @agent_error_handler("insight_agent", "insights")
    def _generate_insights(self, competitors_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate insights from competitor data using LLM.
        
        Args:
            competitors_data: List of competitor dictionaries
        
        Returns:
            Dictionary containing insight data (swot, positioning, trends, opportunities)
        
        Raises:
            WorkflowError: If LLM invocation fails or response cannot be parsed
        """
        competitor_summary = self._prepare_competitor_summary(competitors_data)
        
        # Use HumanMessage and SystemMessage directly to avoid template parsing issues
        # when competitor_summary contains curly braces (e.g., JSON data)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=competitor_summary)
        ]
        
        response = self.invoke_llm(messages)
        
        content = response.content if hasattr(response, "content") else str(response)
        
        if not content:
            raise WorkflowError("LLM returned empty response")
        
        logger.debug(f"LLM response: {content[:200]}...")
        
        insight_data = self._parse_insight_response(content)
        
        return insight_data
    
    def _prepare_competitor_summary(self, competitors_data: list[dict[str, Any]]) -> str:
        """Prepare competitor data summary for LLM analysis.
        
        Args:
            competitors_data: List of competitor dictionaries
        
        Returns:
            Formatted string summary of competitor data
        """
        summary_parts = ["Competitor Analysis Data:\n"]
        
        for i, comp in enumerate(competitors_data[:10], 1):  # Limit to 10 for prompt size
            summary_parts.append(f"\nCompetitor {i}:")
            summary_parts.append(f"  Name: {comp.get('name', 'Unknown')}")
            
            if comp.get("website"):
                summary_parts.append(f"  Website: {comp.get('website')}")
            
            if comp.get("products"):
                products = ", ".join(comp.get("products", [])[:5])
                summary_parts.append(f"  Products: {products}")
            
            if comp.get("market_presence"):
                presence = comp.get("market_presence", "")[:200]
                summary_parts.append(f"  Market Presence: {presence}")
            
            if comp.get("pricing"):
                summary_parts.append(f"  Pricing: {comp.get('pricing')}")
        
        if len(competitors_data) > 10:
            summary_parts.append(f"\n... and {len(competitors_data) - 10} more competitors")
        
        return "\n".join(summary_parts)
    
    def _generate_minimal_insights(self) -> dict[str, Any]:
        """Generate minimal insights when no competitor data is available.
        
        Returns a basic insight structure that explains the data collection
        issue, allowing the workflow to continue and generate a report.
        Ensures all validation requirements are met based on configuration
        (minimum items per SWOT category, minimum trends, minimum opportunities).
        
        Returns:
            Dictionary containing minimal insight data
        """
        # Get minimum requirements from config to ensure we meet validation
        from src.config import get_config
        config = get_config()
        min_swot = config.min_swot_items_per_category
        min_trends = config.min_trends
        min_opportunities = config.min_opportunities
        
        # Generate items to meet minimum requirements
        # Base items - ensure we have at least min_swot items per category
        strengths_pool = [
            "Data collection infrastructure is in place and functional",
            "System architecture supports multiple data sources and search strategies",
            "Modular design allows for easy integration of new data sources"
        ]
        weaknesses_pool = [
            "Unable to collect competitor data due to missing API keys or search failures",
            "Limited data availability prevents comprehensive competitive analysis",
            "Search queries may not be optimized for the target market segment"
        ]
        opportunities_swot_pool = [
            "Improve data collection strategy to gather competitor information",
            "Review search queries and data sources for better results",
            "Implement alternative data collection methods and sources"
        ]
        threats_pool = [
            "Unable to assess competitive landscape due to insufficient data",
            "Missing competitive intelligence may lead to strategic blind spots",
            "Incomplete market analysis could result in suboptimal business decisions"
        ]
        trends_pool = [
            "Data collection challenges may indicate need for improved search strategies",
            "Market analysis requires reliable data sources and robust collection mechanisms",
            "Industry trends suggest increasing importance of competitive intelligence"
        ]
        opportunities_pool = [
            "Enhance data collection methodology",
            "Review and refine search queries",
            "Consider alternative data sources",
            "Implement automated data validation and quality checks"
        ]
        
        # Return items up to the minimum required (we have enough items in each pool)
        return {
            "swot": {
                "strengths": strengths_pool[:min_swot],
                "weaknesses": weaknesses_pool[:min_swot],
                "opportunities": opportunities_swot_pool[:min_swot],
                "threats": threats_pool[:min_swot]
            },
            "positioning": (
                "Unable to determine market positioning due to lack of competitor data. "
                "Data collection process should be reviewed to ensure adequate information gathering."
            ),
            "trends": trends_pool[:min_trends],
            "opportunities": opportunities_pool[:min_opportunities]
        }
    
    def _parse_insight_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response into insight dictionary.
        
        Args:
            content: LLM response content (may contain JSON or markdown)
        
        Returns:
            Dictionary containing insight data
        
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
            json_str_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
            json_str_clean = re.sub(r',\s*([}\]])', r'\1', json_str_clean)
            insight_data = json.loads(json_str_clean)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Response content: {content[:500]}")
            try:
                json_str_fixed = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                insight_data = json.loads(json_str_fixed)
                logger.info("Successfully parsed JSON after fixing control characters")
            except json.JSONDecodeError:
                raise WorkflowError(
                    "Failed to parse insights from LLM response",
                    context={"error": str(e), "content_preview": content[:500]}
                ) from e
        
        required_fields = ["swot", "positioning"]
        missing_fields = [field for field in required_fields if field not in insight_data]
        
        if missing_fields:
            raise WorkflowError(
                f"Insights missing required fields: {missing_fields}",
                context={"insight_data": insight_data}
            )
        
        swot = insight_data.get("swot", {})
        if not isinstance(swot, dict):
            raise WorkflowError(
                "SWOT field must be a dictionary",
                context={"insight_data": insight_data}
            )
        
        # Ensure SWOT has all categories
        for category in ["strengths", "weaknesses", "opportunities", "threats"]:
            if category not in swot:
                swot[category] = []
            elif not isinstance(swot[category], list):
                swot[category] = []
        
        insight_data.setdefault("trends", [])
        insight_data.setdefault("opportunities", [])
        
        if not isinstance(insight_data.get("trends"), list):
            insight_data["trends"] = []
        if not isinstance(insight_data.get("opportunities"), list):
            insight_data["opportunities"] = []
        
        return insight_data
    
    def _clean_insight_data(self, insight_data: dict[str, Any]) -> dict[str, Any]:
        """Clean and normalize insight data before validation.
        
        Performs data cleaning operations:
        - Truncates positioning if it exceeds max_length (500 chars)
        - Ensures all fields meet validation requirements
        - Normalizes string fields
        
        Args:
            insight_data: Raw insight data dictionary from LLM
        
        Returns:
            Cleaned insight data dictionary ready for validation
        """
        cleaned_data = insight_data.copy()
        
        # Clean positioning field - truncate if exceeds 500 characters
        positioning = cleaned_data.get("positioning", "")
        if positioning:
            positioning = str(positioning).strip()
            max_length = 500  # Match Insight model max_length
            if len(positioning) > max_length:
                logger.warning(
                    f"Positioning field exceeds {max_length} characters ({len(positioning)}). "
                    f"Truncating to {max_length} characters."
                )
                # Truncate at word boundary if possible
                truncated = positioning[:max_length]
                # Try to cut at a sentence boundary
                last_period = truncated.rfind('.')
                last_space = truncated.rfind(' ')
                if last_period > max_length * 0.8:  # If period is in last 20%, use it
                    truncated = truncated[:last_period + 1]
                elif last_space > max_length * 0.9:  # If space is in last 10%, use it
                    truncated = truncated[:last_space]
                else:
                    truncated = truncated[:max_length]
                cleaned_data["positioning"] = truncated
            else:
                cleaned_data["positioning"] = positioning
        
        # Ensure positioning meets minimum length
        if cleaned_data.get("positioning") and len(cleaned_data["positioning"]) < 50:
            logger.warning(
                f"Positioning field is too short ({len(cleaned_data['positioning'])} chars). "
                "Minimum is 50 characters."
            )
            # Pad with generic text if too short (shouldn't happen with updated prompt)
            if len(cleaned_data["positioning"]) < 50:
                cleaned_data["positioning"] = (
                    cleaned_data["positioning"] + 
                    " Market positioning analysis based on competitor data."
                )[:500]  # Ensure it doesn't exceed max
        
        return cleaned_data
    
    @property
    def name(self) -> str:
        """Return agent name.
        
        Returns:
            String identifier for this agent
        """
        return "insight_agent"
