"""Planner agent for generating execution plans.

This module implements the PlannerAgent that breaks down user requests into
actionable tasks and generates execution plans. It uses LLM with structured
output to generate Plan models.

Example:
    ```python
    from src.agents.planner_agent import PlannerAgent
    from langchain_groq import ChatGroq
    from src.graph.state import create_initial_state
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    config = {"temperature": 0}
    agent = PlannerAgent(llm=llm, config=config)
    
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
    
    Example:
        ```python
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        config = {"temperature": 0}
        agent = PlannerAgent(llm=llm, config=config)
        
        state = create_initial_state("Analyze competitors")
        updated_state = agent.execute(state)
        plan_data = updated_state["plan"]
        ```
    """
    
    SYSTEM_PROMPT = """You are a strategic planning agent specialized in competitor analysis.

Your task is to break down competitor analysis requests into actionable, structured plans.

When given a user request, analyze it and create a comprehensive execution plan that includes:

1. **Tasks**: A list of specific, actionable tasks to accomplish the analysis
   - Each task should be clear and specific
   - Tasks should cover: data collection, analysis, and insights generation
   - Example: "Collect pricing information for top 5 competitors"

2. **Preferred Sources**: List of preferred data sources
   - Common sources: "web search", "official website", "industry reports"
   - Specify sources that would be most relevant for the analysis

3. **Minimum Results**: The minimum number of competitor sources/data points needed
   - Should be at least 4 for comprehensive analysis
   - Consider the scope of the request when determining this number

4. **Search Strategy**: Overall approach to gathering information
   - Options: "comprehensive" (broad, thorough search), "focused" (targeted, specific search)
   - Choose based on the request scope and urgency

Return your response as a valid JSON object with this exact structure:
{{
    "tasks": ["task1", "task2", "task3"],
    "preferred_sources": ["web search", "official website"],
    "minimum_results": 4,
    "search_strategy": "comprehensive"
}}

Ensure the JSON is valid and all fields are present. The tasks list must contain at least one task."""

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
        
        Example:
            ```python
            state = create_initial_state("Analyze competitors in SaaS market")
            updated_state = agent.execute(state)
            plan = updated_state["plan"]
            assert plan["tasks"] is not None
            ```
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
            
            # Ensure LLM uses deterministic temperature
            temperature = self.config.get("temperature", 0)
            if temperature != 0:
                logger.warning(
                    f"Planner agent should use temperature=0 for deterministic output, "
                    f"got temperature={temperature}. Consider updating config."
                )
            
            # Generate plan using LLM
            plan_data = self._generate_plan(user_request)
            
            # Validate plan against Plan model
            try:
                plan_model = Plan(**plan_data)
                plan_dict = plan_model.model_dump()
            except ValidationError as e:
                logger.error(f"Plan validation failed: {e}")
                raise WorkflowError(
                    "Generated plan failed validation",
                    context={"validation_errors": str(e), "plan_data": plan_data}
                ) from e
            
            # Update state
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
        
        Example:
            ```python
            state = create_initial_state("Analyze competitors")
            request = agent._extract_user_request(state)
            assert request == "Analyze competitors"
            ```
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
        
        Example:
            ```python
            plan_data = agent._generate_plan("Analyze competitors")
            assert "tasks" in plan_data
            assert isinstance(plan_data["tasks"], list)
            ```
        """
        try:
            # Create prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_PROMPT),
                ("human", user_request)
            ])
            
            # Invoke LLM
            messages = prompt.format_messages()
            response = self.llm.invoke(messages)
            
            # Extract content
            content = response.content if hasattr(response, "content") else str(response)
            
            if not content:
                raise WorkflowError("LLM returned empty response")
            
            logger.debug(f"LLM response: {content[:200]}...")
            
            # Parse plan from response
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
        
        Example:
            ```python
            response = '{"tasks": ["task1"], "minimum_results": 4}'
            plan_data = agent._parse_plan_response(response)
            assert plan_data["tasks"] == ["task1"]
            ```
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
            plan_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Response content: {content}")
            raise WorkflowError(
                "Failed to parse plan from LLM response",
                context={"error": str(e), "content_preview": content[:200]}
            ) from e
        
        # Validate required fields
        required_fields = ["tasks"]
        missing_fields = [field for field in required_fields if field not in plan_data]
        
        if missing_fields:
            raise WorkflowError(
                f"Plan missing required fields: {missing_fields}",
                context={"plan_data": plan_data}
            )
        
        # Ensure tasks is a list
        if not isinstance(plan_data.get("tasks"), list):
            raise WorkflowError(
                "Plan 'tasks' field must be a list",
                context={"plan_data": plan_data}
            )
        
        # Set defaults for optional fields
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
