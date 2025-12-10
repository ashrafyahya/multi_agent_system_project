"""Supervisor node for the workflow.

Pure function node that controls workflow flow using SupervisorAgent
and updates the workflow state.

Example:
    ```python
    from src.graph.nodes.supervisor_node import create_supervisor_node
    from langchain_groq import ChatGroq
    from src.graph.state import create_initial_state
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    config = {"max_retries": 3}
    node = create_supervisor_node(llm=llm, config=config)
    
    state = create_initial_state("Analyze competitors")
    state["plan"] = {"tasks": ["Find competitors"]}
    updated_state = node(state)
    ```
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.supervisor_agent import SupervisorAgent
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState

logger = logging.getLogger(__name__)


def create_supervisor_node(
    llm: BaseChatModel,
    config: dict[str, Any]
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
    
    Example:
        ```python
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        config = {"max_retries": 3}
        node = create_supervisor_node(llm=llm, config=config)
        
        state = create_initial_state("Analyze competitors")
        updated_state = node(state)
        ```
    """
    def supervisor_node(state: WorkflowState) -> WorkflowState:
        """Node that controls workflow flow.
        
        Pure function that takes state and returns updated state.
        No side effects except state mutations. This node wraps the
        SupervisorAgent execution and handles errors gracefully.
        
        Args:
            state: Current workflow state containing plan and current stage data
        
        Returns:
            Updated state with current_task and validation_errors updated
        
        Example:
            ```python
            state = create_initial_state("Analyze competitors")
            state["plan"] = {"tasks": ["Find competitors"]}
            updated_state = supervisor_node(state)
            assert "current_task" in updated_state
            ```
        """
        try:
            # Create agent instance
            agent = SupervisorAgent(llm=llm, config=config)
            
            # Execute agent
            updated_state = agent.execute(state)
            
            logger.info(
                f"Supervisor node completed: "
                f"Current task = {updated_state.get('current_task', 'Unknown')}"
            )
            
            return updated_state
            
        except WorkflowError as e:
            logger.error(f"Workflow error in supervisor node: {e}", exc_info=True)
            # Handle workflow errors gracefully
            new_state = state.copy()
            error_list = list(new_state.get("validation_errors", []))
            error_list.append(f"Supervisor failed: {str(e)}")
            new_state["validation_errors"] = error_list
            new_state["current_task"] = "Supervisor failed"
            return new_state
            
        except Exception as e:
            logger.error(
                f"Unexpected error in supervisor node: {e}",
                exc_info=True
            )
            # Handle unexpected errors gracefully
            new_state = state.copy()
            error_list = list(new_state.get("validation_errors", []))
            error_list.append(f"Unexpected error in supervisor: {str(e)}")
            new_state["validation_errors"] = error_list
            new_state["current_task"] = "Supervisor failed"
            return new_state
    
    return supervisor_node


# For backward compatibility, also provide a direct function
def supervisor_node(state: WorkflowState) -> WorkflowState:
    """Node that controls workflow flow (direct function version).
    
    This version expects llm and config to be in the state dictionary.
    For better type safety, use create_supervisor_node() instead.
    
    Args:
        state: Current workflow state containing:
            - plan: Execution plan
            - llm: Language model instance (not in TypedDict)
            - config: Configuration dictionary (not in TypedDict)
    
    Returns:
        Updated state with current_task and validation_errors updated
    
    Raises:
        WorkflowError: If llm or config are missing from state
    """
    llm = state.get("llm")  # type: ignore
    config = state.get("config", {})  # type: ignore
    
    if llm is None:
        raise WorkflowError(
            "LLM instance required in state for supervisor node",
            context={"state_keys": list(state.keys())}
        )
    
    if not isinstance(config, dict):
        raise WorkflowError(
            "Config must be a dictionary",
            context={"config_type": type(config).__name__}
        )
    
    # Create node using factory function
    node_func = create_supervisor_node(llm=llm, config=config)
    return node_func(state)


