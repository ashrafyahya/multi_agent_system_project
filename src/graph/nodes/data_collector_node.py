"""Data collector node for the workflow.

Pure function node that collects competitor data using DataCollectorAgent
and updates the workflow state.

Example:
    ```python
    from src.graph.nodes.data_collector_node import create_data_collector_node
    from langchain_groq import ChatGroq
    from src.graph.state import create_initial_state
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    config = {"max_results": 10}
    node = create_data_collector_node(llm=llm, config=config)
    
    state = create_initial_state("Analyze competitors")
    state["plan"] = {"tasks": ["Find competitors"], "minimum_results": 4}
    updated_state = node(state)
    ```
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.data_collector import DataCollectorAgent
from src.exceptions.collector_error import CollectorError
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState

logger = logging.getLogger(__name__)


def create_data_collector_node(
    llm: BaseChatModel,
    config: dict[str, Any]
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
    
    Example:
        ```python
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        config = {"max_results": 10}
        node = create_data_collector_node(llm=llm, config=config)
        
        state = create_initial_state("Analyze competitors")
        updated_state = node(state)
        ```
    """
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
        
        Example:
            ```python
            state = create_initial_state("Analyze competitors")
            state["plan"] = {"tasks": ["Find competitors"], "minimum_results": 4}
            updated_state = data_collector_node(state)
            assert "collected_data" in updated_state
            ```
        """
        try:
            # Create agent instance
            agent = DataCollectorAgent(llm=llm, config=config)
            
            # Execute agent
            updated_state = agent.execute(state)
            
            logger.info(
                f"Data collector node completed: "
                f"{len(updated_state.get('collected_data', {}).get('competitors', []))} "
                f"competitors collected"
            )
            
            return updated_state
            
        except CollectorError as e:
            logger.error(f"Data collection failed: {e}", exc_info=True)
            # Handle error gracefully - add to validation_errors
            new_state = state.copy()
            error_list = list(new_state.get("validation_errors", []))
            error_list.append(f"Data collection failed: {str(e)}")
            new_state["validation_errors"] = error_list
            new_state["current_task"] = "Data collection failed"
            return new_state
            
        except WorkflowError as e:
            logger.error(f"Workflow error in data collector node: {e}", exc_info=True)
            # Handle workflow errors gracefully
            new_state = state.copy()
            error_list = list(new_state.get("validation_errors", []))
            error_list.append(f"Workflow error: {str(e)}")
            new_state["validation_errors"] = error_list
            new_state["current_task"] = "Data collection failed"
            return new_state
            
        except Exception as e:
            logger.error(
                f"Unexpected error in data collector node: {e}",
                exc_info=True
            )
            # Handle unexpected errors gracefully
            new_state = state.copy()
            error_list = list(new_state.get("validation_errors", []))
            error_list.append(f"Unexpected error in data collection: {str(e)}")
            new_state["validation_errors"] = error_list
            new_state["current_task"] = "Data collection failed"
            return new_state
    
    return data_collector_node


# For backward compatibility and simpler usage, also provide a direct function
# that expects llm and config in state (though not in TypedDict)
def data_collector_node(state: WorkflowState) -> WorkflowState:
    """Node that collects competitor data (direct function version).
    
    This version expects llm and config to be in the state dictionary
    (though not part of the TypedDict). For better type safety, use
    create_data_collector_node() instead.
    
    Args:
        state: Current workflow state containing:
            - plan: Execution plan with tasks
            - llm: Language model instance (not in TypedDict)
            - config: Configuration dictionary (not in TypedDict)
    
    Returns:
        Updated state with collected_data field populated
    
    Raises:
        WorkflowError: If llm or config are missing from state
    """
    llm = state.get("llm")  # type: ignore
    config = state.get("config", {})  # type: ignore
    
    if llm is None:
        raise WorkflowError(
            "LLM instance required in state for data collector node",
            context={"state_keys": list(state.keys())}
        )
    
    if not isinstance(config, dict):
        raise WorkflowError(
            "Config must be a dictionary",
            context={"config_type": type(config).__name__}
        )
    
    # Create node using factory function
    node_func = create_data_collector_node(llm=llm, config=config)
    return node_func(state)
