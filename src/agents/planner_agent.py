"""Planner agent for generating execution plans.

This module implements the PlannerAgent that breaks down user requests into
actionable tasks and generates execution plans. It uses LLM with structured
output to generate Plan models.

Example:
    ```python
    from src.agents.planner_agent import PlannerAgent
    from langchain_groq import ChatGroq
    from src.graph.state import create_initial_state
    
    from src.config import get_config
    from src.main import initialize_llms_for_agents
    
    config = get_config()
    agent_llms = initialize_llms_for_agents(config)
    llm = agent_llms["planner"]
    agent_config = {"temperature": 0}
    agent = PlannerAgent(llm=llm, config=agent_config)
    
    state = create_initial_state("Analyze competitors in SaaS market")
    updated_state = agent.execute(state)
    plan = updated_state["plan"]
    ```
"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from src.agents.base_agent import BaseAgent
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState
from src.models.plan_model import Plan

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """Agent that creates execution plans from user requests.
    
    This agent breaks down competitor analysis requests into actionable tasks
    and generates structured execution plans. It uses an LLM with deterministic
    settings (temperature=0) to ensure consistent plan generation.
    
    The planner:
    1. Extracts user request from workflow state messages
    2. Uses LLM to generate a structured plan
    3. Parses and validates the plan against Plan model
    4. Updates workflow state with the generated plan
    
    Attributes:
        llm: Language model instance (injected)
        config: Configuration dictionary (injected)
    """
    
    SYSTEM_PROMPT = """You are an expert strategic planning consultant specializing in competitive intelligence and market analysis.

Your role is to transform business requests into actionable, data-driven execution plans for comprehensive competitor analysis.

**Core Principles:**
- Prioritize actionable, measurable tasks that lead to strategic insights
- Consider industry context, market dynamics, and business objectives
- Balance comprehensiveness with efficiency
- Focus on data quality over quantity

**When analyzing a user request, create a strategic execution plan:**

1. **Tasks** (3-8 specific, prioritized tasks):
   - Start with market/industry context gathering
   - Focus on direct competitors first, then indirect competitors
   - Include quantitative data collection (pricing, market share, revenue, user metrics)
   - Cover product/service features, positioning, and go-to-market strategies
   - Consider recent news, funding, partnerships, and strategic moves
   - Tasks should be SMART: Specific, Measurable, Achievable, Relevant, Time-bound
   - Example: "Collect pricing tiers and feature comparison for top 5 SaaS competitors in the CRM space"

2. **Preferred Sources** (prioritized list):
   - Primary: Official websites, product pages, pricing pages, investor relations
   - Secondary: Industry reports (Gartner, Forrester, IDC), market research firms
   - Tertiary: News articles, press releases, social media, review sites (G2, Capterra)
   - Consider: Financial filings (for public companies), patent databases, job postings
   - Specify: "official website", "industry reports", "news articles", "review platforms", "financial filings"

3. **Minimum Results** (intelligent determination):
   - Base minimum: 4-6 competitors for comprehensive analysis
   - Adjust based on request scope:
     * Narrow market/niche: 3-5 competitors
     * Broad market: 6-10 competitors
     * Enterprise/strategic analysis: 8-12 competitors
   - Consider market concentration (oligopoly vs. fragmented market)

4. **Search Strategy** (context-aware selection):
   - "comprehensive": Use for strategic planning, market entry, investment decisions
     * Broad market coverage, multiple data sources, deep analysis
   - "focused": Use for quick competitive checks, feature comparisons, pricing analysis
     * Targeted search, specific competitors, time-sensitive decisions
   - Choose based on: request urgency, decision timeline, analysis depth needed

**Output Format:**
Return ONLY a valid JSON object with this exact structure (no markdown, no explanations):
{{
    "tasks": ["task1", "task2", "task3"],
    "preferred_sources": ["source1", "source2"],
    "minimum_results": 4,
    "search_strategy": "comprehensive"
}}

**Quality Requirements:**
- All tasks must be actionable and specific
- Minimum 3 tasks, maximum 8 tasks
- At least 3 different source types
- minimum_results must be between 3 and 15
- search_strategy must be exactly "comprehensive" or "focused"
- JSON must be valid and parseable

**Error Prevention:**
- If request is ambiguous, infer reasonable scope based on context
- If industry is unclear, include tasks to identify market segment
- Always include at least one quantitative data collection task"""

    def execute(self, state: WorkflowState) -> WorkflowState:
        """Generate execution plan from user request.
        
        Extracts the user request from workflow state messages, uses the LLM
        to generate a structured plan, validates it against the Plan model,
        and updates the state with the plan.
        
        Args:
            state: Current workflow state containing messages with user request
        
        Returns:
            Updated WorkflowState with plan field populated
        
        Raises:
            WorkflowError: If plan generation fails critically after retries
                or if user request cannot be extracted
        
        """
        try:
            # Extract user request
            user_request = self._extract_user_request(state)
            if not user_request:
                raise WorkflowError(
                    "Cannot extract user request from workflow state",
                    context={"messages_count": len(state.get("messages", []))}
                )
            
            logger.info(f"Generating plan for request: {user_request[:100]}...")
            
            temperature = self.config.get("temperature", 0)
            if temperature != 0:
                logger.warning(
                    f"Planner agent should use temperature=0 for deterministic output, "
                    f"got temperature={temperature}. Consider updating config."
                )
            
            plan_data = self._generate_plan(user_request)
            
            try:
                plan_model = Plan(**plan_data)
                plan_dict = plan_model.model_dump()
            except ValidationError as e:
                logger.error(f"Plan validation failed: {e}")
                raise WorkflowError(
                    "Generated plan failed validation",
                    context={"validation_errors": str(e), "plan_data": plan_data}
                ) from e
            
            new_state = state.copy()
            new_state["plan"] = plan_dict
            new_state["current_task"] = "Planning completed"
            
            logger.info(
                f"Plan generated successfully: {len(plan_dict.get('tasks', []))} tasks, "
                f"minimum_results={plan_dict.get('minimum_results', 0)}"
            )
            
            return new_state
            
        except WorkflowError:
            # Re-raise workflow errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error in planner agent: {e}", exc_info=True)
            raise WorkflowError(
                "Plan generation failed unexpectedly",
                context={"error": str(e)}
            ) from e
    
    def _extract_user_request(self, state: WorkflowState) -> str:
        """Extract user request from workflow state messages.
        
        Searches through messages to find the most recent user message
        (HumanMessage) and extracts its content.
        
        Args:
            state: Workflow state containing messages
        
        Returns:
            User request string, or empty string if no user message found
        
        """
        messages = state.get("messages", [])
        
        # Find the most recent HumanMessage
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                content = message.content
                if isinstance(content, str) and content.strip():
                    return content.strip()
        
        # Fallback: try to extract from any message
        for message in reversed(messages):
            if hasattr(message, "content"):
                content = message.content
                if isinstance(content, str) and content.strip():
                    return content.strip()
        
        return ""
    
    def _generate_plan(self, user_request: str) -> dict[str, Any]:
        """Generate plan using LLM.
        
        Creates a prompt with the user request, invokes the LLM, and parses
        the response into a plan dictionary.
        
        Args:
            user_request: User's competitor analysis request
        
        Returns:
            Dictionary containing plan data (tasks, preferred_sources, etc.)
        
        Raises:
            WorkflowError: If LLM invocation fails or response cannot be parsed
        
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_PROMPT),
                ("human", user_request)
            ])
            
            messages = prompt.format_messages()
            response = self.llm.invoke(messages)
            
            content = response.content if hasattr(response, "content") else str(response)
            
            if not content:
                raise WorkflowError("LLM returned empty response")
            
            logger.debug(f"LLM response: {content[:200]}...")
            
            plan_data = self._parse_plan_response(content)
            
            return plan_data
            
        except WorkflowError:
            raise
        except Exception as e:
            logger.error(f"Error generating plan: {e}", exc_info=True)
            raise WorkflowError(
                "Failed to generate plan from LLM",
                context={"error": str(e), "user_request": user_request[:100]}
            ) from e
    
    def _parse_plan_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response into plan dictionary.
        
        Attempts to extract JSON from the LLM response. Handles cases where
        the response may contain markdown code blocks or extra text.
        
        Args:
            content: LLM response content (may contain JSON or markdown)
        
        Returns:
            Dictionary containing plan data
        
        Raises:
            WorkflowError: If JSON cannot be parsed or plan structure is invalid
        
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
            plan_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Response content: {content}")
            raise WorkflowError(
                "Failed to parse plan from LLM response",
                context={"error": str(e), "content_preview": content[:200]}
            ) from e
        
        required_fields = ["tasks"]
        missing_fields = [field for field in required_fields if field not in plan_data]
        
        if missing_fields:
            raise WorkflowError(
                f"Plan missing required fields: {missing_fields}",
                context={"plan_data": plan_data}
            )
        
        if not isinstance(plan_data.get("tasks"), list):
            raise WorkflowError(
                "Plan 'tasks' field must be a list",
                context={"plan_data": plan_data}
            )
        
        plan_data.setdefault("preferred_sources", [])
        plan_data.setdefault("minimum_results", 4)
        plan_data.setdefault("search_strategy", "comprehensive")
        
        return plan_data
    
    @property
    def name(self) -> str:
        """Return agent name.
        
        Returns:
            String identifier for this agent
        """
        return "planner_agent"
