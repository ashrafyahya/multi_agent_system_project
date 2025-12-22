"""Supervisor node for the workflow.

Pure function node that controls workflow flow using SupervisorAgent
and updates the workflow state.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.supervisor_agent import SupervisorAgent
from src.graph.nodes.base_node import node_error_handler
from src.graph.state import WorkflowState
from src.utils.agent_logger import AgentLogger

logger = logging.getLogger(__name__)


def create_supervisor_node(
    llm: BaseChatModel,
    config: dict[str, Any],
    agent_logger: AgentLogger | None = None
) -> Any:
    """Create a supervisor node function.
    
    This factory function creates a pure function node that controls workflow
    flow. The node is a closure that captures the LLM and config dependencies,
    allowing it to remain a pure function (State -> State) while having access
    to required dependencies.
    
    Args:
        llm: Language model instance for the SupervisorAgent
        config: Configuration dictionary for the SupervisorAgent
    
    Returns:
        Pure function that takes WorkflowState and returns updated WorkflowState
    """
    @node_error_handler("supervisor_node")
    def supervisor_node(state: WorkflowState) -> WorkflowState:
        """Node that controls workflow flow.
        
        Pure function that takes state and returns updated state.
        No side effects except state mutations. This node wraps the
        SupervisorAgent execution and handles errors gracefully.
        
        Args:
            state: Current workflow state containing plan and current stage data
        
        Returns:
            Updated state with current_task and validation_errors updated
        """
        # Create agent instance
        agent = SupervisorAgent(llm=llm, config=config)
        
        # Execute agent
        updated_state = agent.execute(state)
        
        # Log agent output (supervisor updates current_task and validation decisions)
        if agent_logger and agent_logger.enabled:
            try:
                supervisor_output = {
                    "current_task": updated_state.get("current_task"),
                    "validation_errors": updated_state.get("validation_errors", []),
                    "retry_count": updated_state.get("retry_count", 0),
                }
                agent_logger.log_agent_output(agent.name, supervisor_output, updated_state)
            except Exception as e:
                # Don't disrupt workflow if logging fails
                logger.warning(f"Failed to log supervisor agent output: {e}")
        
        logger.info(
            f"Supervisor node completed: "
            f"Current task = {updated_state.get('current_task', 'Unknown')}"
        )
        
        return updated_state
    
    return supervisor_node


