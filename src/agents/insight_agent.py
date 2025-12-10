"""Insight agent for transforming data into business insights.

This module implements the InsightAgent that transforms collected competitor
data into business insights, generates SWOT analysis, and extracts trends
and opportunities.

Example:
    ```python
    from src.agents.insight_agent import InsightAgent
    from langchain_groq import ChatGroq
    from src.graph.state import create_initial_state
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    config = {"temperature": 0.7}
    agent = InsightAgent(llm=llm, config=config)
    
    state = create_initial_state("Analyze competitors")
    state["collected_data"] = {"competitors": [...]}
    updated_state = agent.execute(state)
    ```
"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from src.agents.base_agent import BaseAgent
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState
from src.models.competitor_profile import CompetitorProfile
from src.models.insight_model import SWOT, Insight

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
    
    Example:
        ```python
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
        config = {"temperature": 0.7}
        agent = InsightAgent(llm=llm, config=config)
        
        state = create_initial_state("Analyze competitors")
        state["collected_data"] = {"competitors": [...]}
        updated_state = agent.execute(state)
        ```
    """
    
    SYSTEM_PROMPT = """You are a business intelligence analyst specializing in competitor analysis.

Your task is to analyze competitor data and extract meaningful business insights.

Given competitor information, you must:
1. Generate a comprehensive SWOT analysis:
   - Strengths: What advantages do these competitors have?
   - Weaknesses: What are their limitations or vulnerabilities?
   - Opportunities: What market opportunities exist?
   - Threats: What threats do they face?

2. Determine market positioning: How do these competitors position themselves in the market?

3. Identify market trends: What trends are evident from the competitor data?

4. Extract business opportunities: What opportunities can be identified for our business?

Return your analysis as a JSON object with this exact structure:
{{
    "swot": {{
        "strengths": ["strength1", "strength2", ...],
        "weaknesses": ["weakness1", "weakness2", ...],
        "opportunities": ["opportunity1", "opportunity2", ...],
        "threats": ["threat1", "threat2", ...]
    }},
    "positioning": "Description of market positioning (at least 20 characters)",
    "trends": ["trend1", "trend2", ...],
    "opportunities": ["opportunity1", "opportunity2", ...]
}}

Ensure:
- Each SWOT category has at least 1 item
- Positioning is descriptive (at least 20 characters)
- Trends are specific and actionable
- Opportunities are business-focused
- All fields are present and valid JSON"""

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
        
        Example:
            ```python
            state = create_initial_state("Analyze competitors")
            state["collected_data"] = {"competitors": [...]}
            updated_state = agent.execute(state)
            assert updated_state["insights"] is not None
            ```
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
            new_state = state.copy()
            new_state["insights"] = insight_dict
            new_state["current_task"] = "Insights generated successfully"
            
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
    
    def _generate_insights(self, competitors_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate insights from competitor data using LLM.
        
        Args:
            competitors_data: List of competitor dictionaries
        
        Returns:
            Dictionary containing insight data (swot, positioning, trends, opportunities)
        
        Raises:
            WorkflowError: If LLM invocation fails or response cannot be parsed
        """
        try:
            # Prepare competitor data summary for LLM
            competitor_summary = self._prepare_competitor_summary(competitors_data)
            
            # Create prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_PROMPT),
                ("human", competitor_summary)
            ])
            
            # Invoke LLM
            messages = prompt.format_messages()
            response = self.llm.invoke(messages)
            
            # Extract content
            content = response.content if hasattr(response, "content") else str(response)
            
            if not content:
                raise WorkflowError("LLM returned empty response")
            
            logger.debug(f"LLM response: {content[:200]}...")
            
            # Parse insights from response
            insight_data = self._parse_insight_response(content)
            
            return insight_data
            
        except WorkflowError:
            raise
        except Exception as e:
            logger.error(f"Error generating insights: {e}", exc_info=True)
            raise WorkflowError(
                "Failed to generate insights from LLM",
                context={"error": str(e), "competitors_count": len(competitors_data)}
            ) from e
    
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
        Ensures all validation requirements are met (minimum 1 item per SWOT category,
        minimum 3 total insights).
        
        Returns:
            Dictionary containing minimal insight data
        """
        return {
            "swot": {
                "strengths": [
                    "Data collection infrastructure is in place and functional"
                ],
                "weaknesses": [
                    "Unable to collect competitor data due to missing API keys or search failures"
                ],
                "opportunities": [
                    "Improve data collection strategy to gather competitor information",
                    "Review search queries and data sources for better results"
                ],
                "threats": [
                    "Unable to assess competitive landscape due to insufficient data"
                ]
            },
            "positioning": (
                "Unable to determine market positioning due to lack of competitor data. "
                "Data collection process should be reviewed to ensure adequate information gathering."
            ),
            "trends": [
                "Data collection challenges may indicate need for improved search strategies"
            ],
            "opportunities": [
                "Enhance data collection methodology",
                "Review and refine search queries",
                "Consider alternative data sources"
            ]
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
            # Clean JSON string: remove control characters that can break JSON parsing
            # Replace common problematic control characters with spaces
            json_str_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
            # Remove trailing commas before closing braces/brackets
            json_str_clean = re.sub(r',\s*([}\]])', r'\1', json_str_clean)
            insight_data = json.loads(json_str_clean)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Response content: {content[:500]}")
            # Try to fix common JSON issues and retry
            try:
                # Try escaping newlines and other control characters in strings
                json_str_fixed = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                insight_data = json.loads(json_str_fixed)
                logger.info("Successfully parsed JSON after fixing control characters")
            except json.JSONDecodeError:
                raise WorkflowError(
                    "Failed to parse insights from LLM response",
                    context={"error": str(e), "content_preview": content[:500]}
                ) from e
        
        # Validate required fields
        required_fields = ["swot", "positioning"]
        missing_fields = [field for field in required_fields if field not in insight_data]
        
        if missing_fields:
            raise WorkflowError(
                f"Insights missing required fields: {missing_fields}",
                context={"insight_data": insight_data}
            )
        
        # Validate SWOT structure
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
        
        # Set defaults for optional fields
        insight_data.setdefault("trends", [])
        insight_data.setdefault("opportunities", [])
        
        # Ensure lists are lists
        if not isinstance(insight_data.get("trends"), list):
            insight_data["trends"] = []
        if not isinstance(insight_data.get("opportunities"), list):
            insight_data["opportunities"] = []
        
        return insight_data
    
    @property
    def name(self) -> str:
        """Return agent name.
        
        Returns:
            String identifier for this agent
        """
        return "insight_agent"
