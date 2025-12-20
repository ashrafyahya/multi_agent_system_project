"""Planner node for the workflow.

Pure function node that generates execution plans using PlannerAgent
and updates the workflow state.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.planner_agent import PlannerAgent
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState
from src.utils.agent_logger import AgentLogger

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
        try:
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
            
        except WorkflowError as e:
            logger.error(f"Workflow error in planner node: {e}", exc_info=True)
            # Handle workflow errors gracefully
            new_state = state.copy()
            error_list = list(new_state.get("validation_errors", []))
            error_list.append(f"Planning failed: {str(e)}")
            new_state["validation_errors"] = error_list
            new_state["current_task"] = "Planning failed"
            return new_state
            
        except Exception as e:
            logger.error(
                f"Unexpected error in planner node: {e}",
                exc_info=True
            )
            # Handle unexpected errors gracefully
            new_state = state.copy()
            error_list = list(new_state.get("validation_errors", []))
            error_list.append(f"Unexpected error in planning: {str(e)}")
            new_state["validation_errors"] = error_list
            new_state["current_task"] = "Planning failed"
            return new_state
    
    return planner_node


# For backward compatibility, also provide a direct function
def planner_node(state: WorkflowState) -> WorkflowState:
    """Node that generates execution plans (direct function version).
    
    This version expects llm and config to be in the state dictionary.
    For better type safety, use create_planner_node() instead.
    
    Args:
        state: Current workflow state containing:
            - messages: User message
            - llm: Language model instance (not in TypedDict)
            - config: Configuration dictionary (not in TypedDict)
    
    Returns:
        Updated state with plan field populated
    
    Raises:
        WorkflowError: If llm or config are missing from state
    """
    llm = state.get("llm")  # type: ignore
    config = state.get("config", {})  # type: ignore
    
    if llm is None:
        raise WorkflowError(
            "LLM instance required in state for planner node",
            context={"state_keys": list(state.keys())}
        )
    
    if not isinstance(config, dict):
        raise WorkflowError(
            "Config must be a dictionary",
            context={"config_type": type(config).__name__}
        )
    
    # Create node using factory function
    node_func = create_planner_node(llm=llm, config=config)
    return node_func(state)


