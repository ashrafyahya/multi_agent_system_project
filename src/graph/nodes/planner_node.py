"""Planner node for the workflow.

Pure function node that generates execution plans using PlannerAgent
and updates the workflow state.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.planner_agent import PlannerAgent
from src.graph.nodes.base_node import node_error_handler
from src.graph.state import WorkflowState
from src.utils.agent_logger import AgentLogger
from src.utils.metrics import track_execution_time

logger = logging.getLogger(__name__)


def create_planner_node(
    llm: BaseChatModel,
    config: dict[str, Any],
    agent_logger: AgentLogger | None = None
) -> Any:
    """Create a planner node function.
    
    This factory function creates a pure function node that generates execution
    plans. The node is a closure that captures the LLM and config dependencies,
    allowing it to remain a pure function (State -> State) while having access
    to required dependencies.
    
    Args:
        llm: Language model instance for the PlannerAgent
        config: Configuration dictionary for the PlannerAgent
    
    Returns:
        Pure function that takes WorkflowState and returns updated WorkflowState
    """
    
    @node_error_handler("planner_node")
    @track_execution_time("planner_node")
    def planner_node(state: WorkflowState) -> WorkflowState:
        """Node that generates execution plans.
        
        Pure function that takes state and returns updated state.
        No side effects except state mutations. This node wraps the
        PlannerAgent execution and handles errors gracefully.
        
        Args:
            state: Current workflow state containing user message
        
        Returns:
            Updated state with plan field populated, or original
            state with validation_errors if planning fails
        """
        # Create agent instance
        agent = PlannerAgent(llm=llm, config=config)
        
        # Execute agent
        updated_state = agent.execute(state)
        
        # Log agent output
        if agent_logger and agent_logger.enabled:
            try:
                plan_output = updated_state.get("plan")
                agent_logger.log_agent_output(agent.name, plan_output, updated_state)
            except Exception as e:
                # Don't disrupt workflow if logging fails
                logger.warning(f"Failed to log planner agent output: {e}")
        
        logger.info(
            f"Planner node completed: "
            f"{len(updated_state.get('plan', {}).get('tasks', []))} tasks planned"
        )
        
        return updated_state
    
    return planner_node


