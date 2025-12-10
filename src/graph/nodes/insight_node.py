"""Insight node for the workflow.

Pure function node that transforms collected data into business insights
using InsightAgent and updates the workflow state.

Example:
    ```python
    from src.graph.nodes.insight_node import create_insight_node
    from langchain_groq import ChatGroq
    from src.graph.state import create_initial_state
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    config = {"temperature": 0.7}
    node = create_insight_node(llm=llm, config=config)
    
    state = create_initial_state("Analyze competitors")
    state["collected_data"] = {"competitors": [...]}
    updated_state = node(state)
    ```
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.insight_agent import InsightAgent
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState

logger = logging.getLogger(__name__)


def create_insight_node(
    llm: BaseChatModel,
    config: dict[str, Any]
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
    
    Example:
        ```python
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
        config = {"temperature": 0.7}
        node = create_insight_node(llm=llm, config=config)
        
        state = create_initial_state("Analyze competitors")
        updated_state = node(state)
        ```
    """
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
        
        Example:
            ```python
            state = create_initial_state("Analyze competitors")
            state["collected_data"] = {"competitors": [...]}
            updated_state = insight_node(state)
            assert "insights" in updated_state
            ```
        """
        try:
            # Create agent instance
            agent = InsightAgent(llm=llm, config=config)
            
            # Execute agent
            updated_state = agent.execute(state)
            
            logger.info(
                f"Insight node completed: "
                f"{len(updated_state.get('insights', {}).get('swot', {}).get('strengths', []))} "
                f"strengths identified"
            )
            
            return updated_state
            
        except WorkflowError as e:
            logger.error(f"Workflow error in insight node: {e}", exc_info=True)
            # Handle workflow errors gracefully
            new_state = state.copy()
            error_list = list(new_state.get("validation_errors", []))
            error_list.append(f"Insight generation failed: {str(e)}")
            new_state["validation_errors"] = error_list
            new_state["current_task"] = "Insight generation failed"
            return new_state
            
        except Exception as e:
            logger.error(
                f"Unexpected error in insight node: {e}",
                exc_info=True
            )
            # Handle unexpected errors gracefully
            new_state = state.copy()
            error_list = list(new_state.get("validation_errors", []))
            error_list.append(f"Unexpected error in insight generation: {str(e)}")
            new_state["validation_errors"] = error_list
            new_state["current_task"] = "Insight generation failed"
            return new_state
    
    return insight_node


# For backward compatibility and simpler usage, also provide a direct function
# that expects llm and config in state (though not in TypedDict)
def insight_node(state: WorkflowState) -> WorkflowState:
    """Node that generates business insights (direct function version).
    
    This version expects llm and config to be in the state dictionary
    (though not part of the TypedDict). For better type safety, use
    create_insight_node() instead.
    
    Args:
        state: Current workflow state containing:
            - collected_data: Collected competitor data
            - llm: Language model instance (not in TypedDict)
            - config: Configuration dictionary (not in TypedDict)
    
    Returns:
        Updated state with insights field populated
    
    Raises:
        WorkflowError: If llm or config are missing from state
    """
    llm = state.get("llm")  # type: ignore
    config = state.get("config", {})  # type: ignore
    
    if llm is None:
        raise WorkflowError(
            "LLM instance required in state for insight node",
            context={"state_keys": list(state.keys())}
        )
    
    if not isinstance(config, dict):
        raise WorkflowError(
            "Config must be a dictionary",
            context={"config_type": type(config).__name__}
        )
    
    # Create node using factory function
    node_func = create_insight_node(llm=llm, config=config)
    return node_func(state)
