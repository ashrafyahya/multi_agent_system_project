"""Insight node for the workflow.

Pure function node that transforms collected data into business insights
using InsightAgent and updates the workflow state.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.insight_agent import InsightAgent
from src.graph.nodes.base_node import node_error_handler
from src.graph.state import WorkflowState
from src.utils.agent_logger import AgentLogger
from src.utils.metrics import track_execution_time

logger = logging.getLogger(__name__)


def create_insight_node(
    llm: BaseChatModel,
    config: dict[str, Any],
    agent_logger: AgentLogger | None = None
) -> Any:
    """Create an insight node function.
    
    This factory function creates a pure function node that generates business
    insights from collected data. The node is a closure that captures the LLM
    and config dependencies, allowing it to remain a pure function (State -> State)
    while having access to required dependencies.
    
    Args:
        llm: Language model instance for the InsightAgent
        config: Configuration dictionary for the InsightAgent
    
    Returns:
        Pure function that takes WorkflowState and returns updated WorkflowState
    """
    @node_error_handler("insight_node")
    @track_execution_time("insight_node")
    def insight_node(state: WorkflowState) -> WorkflowState:
        """Node that generates business insights from collected data.
        
        Pure function that takes state and returns updated state.
        No side effects except state mutations. This node wraps the
        InsightAgent execution and handles errors gracefully.
        
        Args:
            state: Current workflow state containing collected_data
        
        Returns:
            Updated state with insights field populated, or original
            state with validation_errors if insight generation fails
        """
        # Create agent instance
        agent = InsightAgent(llm=llm, config=config)
        
        # Execute agent
        updated_state = agent.execute(state)
        
        # Log agent output
        if agent_logger and agent_logger.enabled:
            try:
                insights = updated_state.get("insights")
                agent_logger.log_agent_output(agent.name, insights, updated_state)
            except Exception as e:
                # Don't disrupt workflow if logging fails
                logger.warning(f"Failed to log insight agent output: {e}")
        
        logger.info(
            f"Insight node completed: "
            f"{len(updated_state.get('insights', {}).get('swot', {}).get('strengths', []))} "
            f"strengths identified"
        )
        
        return updated_state
    
    return insight_node

