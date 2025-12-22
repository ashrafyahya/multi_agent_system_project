"""Data collector node for the workflow.

Pure function node that collects competitor data using DataCollectorAgent
and updates the workflow state.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.data_collector import DataCollectorAgent
from src.graph.nodes.base_node import node_error_handler
from src.graph.state import WorkflowState
from src.utils.agent_logger import AgentLogger

logger = logging.getLogger(__name__)


def create_data_collector_node(
    llm: BaseChatModel,
    config: dict[str, Any],
    agent_logger: AgentLogger | None = None
) -> Any:
    """Create a data collector node function.
    
    This factory function creates a pure function node that collects competitor
    data. The node is a closure that captures the LLM and config dependencies,
    allowing it to remain a pure function (State -> State) while having access
    to required dependencies.
    
    Args:
        llm: Language model instance for the DataCollectorAgent
        config: Configuration dictionary for the DataCollectorAgent
    
    Returns:
            Pure function that takes WorkflowState and returns updated WorkflowState
    """
    @node_error_handler("data_collector_node")
    def data_collector_node(state: WorkflowState) -> WorkflowState:
        """Node that collects competitor data.
        
        Pure function that takes state and returns updated state.
        No side effects except state mutations. This node wraps the
        DataCollectorAgent execution and handles errors gracefully.
        
        Args:
            state: Current workflow state containing plan with tasks
        
        Returns:
            Updated state with collected_data field populated, or original
            state with validation_errors if collection fails
        """
        # Create agent instance
        agent = DataCollectorAgent(llm=llm, config=config)
        
        # Execute agent
        updated_state = agent.execute(state)
        
        # Log agent output
        if agent_logger and agent_logger.enabled:
            try:
                collected_data = updated_state.get("collected_data")
                agent_logger.log_agent_output(agent.name, collected_data, updated_state)
            except Exception as e:
                # Don't disrupt workflow if logging fails
                logger.warning(f"Failed to log data collector agent output: {e}")
        
        logger.info(
            f"Data collector node completed: "
            f"{len(updated_state.get('collected_data', {}).get('competitors', []))} "
            f"competitors collected"
        )
        
        return updated_state
    
    return data_collector_node

