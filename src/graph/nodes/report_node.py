"""Report node for the workflow.

Pure function node that generates the final formatted report using
ReportAgent and updates the workflow state.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.report_agent import ReportAgent
from src.graph.nodes.base_node import node_error_handler
from src.graph.state import WorkflowState
from src.utils.agent_logger import AgentLogger

logger = logging.getLogger(__name__)


def create_report_node(
    llm: BaseChatModel,
    config: dict[str, Any],
    agent_logger: AgentLogger | None = None
) -> Any:
    """Create a report node function.
    
    This factory function creates a pure function node that generates final
    formatted reports from business insights. The node is a closure that captures
    the LLM and config dependencies, allowing it to remain a pure function
    (State -> State) while having access to required dependencies.
    
    Args:
        llm: Language model instance for the ReportAgent
        config: Configuration dictionary for the ReportAgent
    
    Returns:
        Pure function that takes WorkflowState and returns updated WorkflowState
    """
    @node_error_handler("report_node")
    def report_node(state: WorkflowState) -> WorkflowState:
        """Node that generates the final formatted report.
        
        Pure function that takes state and returns updated state.
        No side effects except state mutations. This node wraps the
        ReportAgent execution and handles errors gracefully.
        
        Args:
            state: Current workflow state containing insights
        
        Returns:
            Updated state with report field populated, or original
            state with validation_errors if report generation fails
        """
        # Create agent instance
        agent = ReportAgent(llm=llm, config=config)
        
        # Execute agent
        updated_state = agent.execute(state)
        
        # Log agent output
        if agent_logger and agent_logger.enabled:
            try:
                report = updated_state.get("report")
                agent_logger.log_agent_output(agent.name, report, updated_state)
            except Exception as e:
                # Don't disrupt workflow if logging fails
                logger.warning(f"Failed to log report agent output: {e}")
        
        report_length = len(updated_state.get("report", "") or "")
        logger.info(
            f"Report node completed: "
            f"Report generated ({report_length} characters)"
        )
        
        return updated_state
    
    return report_node

