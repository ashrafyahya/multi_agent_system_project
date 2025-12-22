"""Export node for the workflow.

Pure function node that generates PDF and image exports from reports using
ExportAgent and updates the workflow state.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.export_agent import ExportAgent
from src.graph.nodes.base_node import node_error_handler
from src.graph.state import WorkflowState
from src.utils.agent_logger import AgentLogger

logger = logging.getLogger(__name__)


def create_export_node(
    llm: BaseChatModel,
    config: dict[str, Any],
    agent_logger: AgentLogger | None = None
) -> Any:
    """Create an export node function.
    
    This factory function creates a pure function node that generates export
    files (PDF, images) from reports. The node is a closure that captures the
    LLM and config dependencies, allowing it to remain a pure function
    (State -> State) while having access to required dependencies.
    
    Args:
        llm: Language model instance for the ExportAgent
        config: Configuration dictionary for the ExportAgent
    
    Returns:
        Pure function that takes WorkflowState and returns updated WorkflowState
    """
    @node_error_handler("export_node")
    def export_node(state: WorkflowState) -> WorkflowState:
        """Node that generates export files from report.
        
        Pure function that takes state and returns updated state.
        No side effects except state mutations. This node wraps the
        ExportAgent execution and handles errors gracefully.
        
        Args:
            state: Current workflow state containing report
        
        Returns:
            Updated state with export_paths field populated, or original
            state with validation_errors if export generation fails
        """
        # Create agent instance
        agent = ExportAgent(llm=llm, config=config)
        
        # Execute agent
        updated_state = agent.execute(state)
        
        # Log agent output
        if agent_logger and agent_logger.enabled:
            try:
                export_paths = updated_state.get("export_paths")
                agent_logger.log_agent_output(agent.name, export_paths, updated_state)
            except Exception as e:
                # Don't disrupt workflow if logging fails
                logger.warning(f"Failed to log export agent output: {e}")
        
        export_count = len(updated_state.get("export_paths", {}))
        logger.info(
            f"Export node completed: {export_count} file(s) generated"
        )
        
        return updated_state
    
    return export_node


