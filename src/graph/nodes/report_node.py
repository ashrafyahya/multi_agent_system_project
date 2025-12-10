"""Report node for the workflow.

Pure function node that generates the final formatted report using
ReportAgent and updates the workflow state.

Example:
    ```python
    from src.graph.nodes.report_node import create_report_node
    from langchain_groq import ChatGroq
    from src.graph.state import create_initial_state
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    config = {"temperature": 0.7}
    node = create_report_node(llm=llm, config=config)
    
    state = create_initial_state("Analyze competitors")
    state["insights"] = {...}
    updated_state = node(state)
    ```
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.report_agent import ReportAgent
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState

logger = logging.getLogger(__name__)


def create_report_node(
    llm: BaseChatModel,
    config: dict[str, Any]
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
    
    Example:
        ```python
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
        config = {"temperature": 0.7}
        node = create_report_node(llm=llm, config=config)
        
        state = create_initial_state("Analyze competitors")
        updated_state = node(state)
        ```
    """
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
        
        Example:
            ```python
            state = create_initial_state("Analyze competitors")
            state["insights"] = {
                "swot": {...},
                "positioning": "...",
                "trends": [...],
                "opportunities": [...]
            }
            updated_state = report_node(state)
            assert "report" in updated_state
            ```
        """
        try:
            # Create agent instance
            agent = ReportAgent(llm=llm, config=config)
            
            # Execute agent
            updated_state = agent.execute(state)
            
            report_length = len(updated_state.get("report", "") or "")
            logger.info(
                f"Report node completed: "
                f"Report generated ({report_length} characters)"
            )
            
            return updated_state
            
        except WorkflowError as e:
            logger.error(f"Workflow error in report node: {e}", exc_info=True)
            # Handle workflow errors gracefully
            new_state = state.copy()
            error_list = list(new_state.get("validation_errors", []))
            error_list.append(f"Report generation failed: {str(e)}")
            new_state["validation_errors"] = error_list
            new_state["current_task"] = "Report generation failed"
            return new_state
            
        except Exception as e:
            logger.error(
                f"Unexpected error in report node: {e}",
                exc_info=True
            )
            # Handle unexpected errors gracefully
            new_state = state.copy()
            error_list = list(new_state.get("validation_errors", []))
            error_list.append(f"Unexpected error in report generation: {str(e)}")
            new_state["validation_errors"] = error_list
            new_state["current_task"] = "Report generation failed"
            return new_state
    
    return report_node


# For backward compatibility and simpler usage, also provide a direct function
# that expects llm and config in state (though not in TypedDict)
def report_node(state: WorkflowState) -> WorkflowState:
    """Node that generates the final formatted report (direct function version).
    
    This version expects llm and config to be in the state dictionary
    (though not part of the TypedDict). For better type safety, use
    create_report_node() instead.
    
    Args:
        state: Current workflow state containing:
            - insights: Business insights and SWOT analysis
            - llm: Language model instance (not in TypedDict)
            - config: Configuration dictionary (not in TypedDict)
    
    Returns:
        Updated state with report field populated
    
    Raises:
        WorkflowError: If llm or config are missing from state
    """
    llm = state.get("llm")  # type: ignore
    config = state.get("config", {})  # type: ignore
    
    if llm is None:
        raise WorkflowError(
            "LLM instance required in state for report node",
            context={"state_keys": list(state.keys())}
        )
    
    if not isinstance(config, dict):
        raise WorkflowError(
            "Config must be a dictionary",
            context={"config_type": type(config).__name__}
        )
    
    # Create node using factory function
    node_func = create_report_node(llm=llm, config=config)
    return node_func(state)
