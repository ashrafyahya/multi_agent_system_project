"""Retry node for handling retry logic in the workflow.

Pure function node that handles retry logic, modifies search queries,
and increments retry count.

Example:
    ```python
    from src.graph.nodes.retry_node import create_retry_node
    from src.graph.state import create_initial_state
    
    node = create_retry_node(max_retries=3)
    
    state = create_initial_state("Analyze competitors")
    state["plan"] = {"tasks": ["Find competitors"]}
    state["retry_count"] = 1
    state["validation_errors"] = ["Validation failed"]
    updated_state = node(state)
    ```
"""

import logging
from typing import Any

from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState

logger = logging.getLogger(__name__)


def create_retry_node(max_retries: int = 3) -> Any:
    """Create a retry node function.
    
    This factory function creates a pure function node that handles retry logic.
    The node increments retry count, modifies search queries to improve them,
    and clears validation errors to allow retry.
    
    Args:
        max_retries: Maximum number of retry attempts allowed (default: 3)
    
    Returns:
        Pure function that takes WorkflowState and returns updated WorkflowState
    
    Example:
        ```python
        node = create_retry_node(max_retries=3)
        
        state = create_initial_state("Analyze competitors")
        state["plan"] = {"tasks": ["Find competitors"]}
        updated_state = node(state)
        ```
    """
    def retry_node(state: WorkflowState) -> WorkflowState:
        """Node that handles retry logic.
        
        Pure function that takes state and returns updated state.
        No side effects except state mutations. This node:
        1. Increments retry_count
        2. Modifies search queries in plan to improve them
        3. Clears validation_errors to allow retry
        4. Updates current_task to indicate retry
        
        Args:
            state: Current workflow state containing plan and retry_count
        
        Returns:
            Updated state with:
            - retry_count incremented
            - plan modified with improved queries
            - validation_errors cleared
            - current_task updated
        
        Raises:
            WorkflowError: If max retries exceeded or plan is missing
        
        Example:
            ```python
            state = create_initial_state("Analyze competitors")
            state["plan"] = {"tasks": ["Find competitors"]}
            state["retry_count"] = 1
            updated_state = retry_node(state)
            assert updated_state["retry_count"] == 2
            ```
        """
        # Check if plan exists
        plan = state.get("plan")
        if not plan:
            raise WorkflowError(
                "Cannot retry without a plan",
                context={"state_keys": list(state.keys())}
            )
        
        # Get current retry count
        current_retry_count = state.get("retry_count", 0)
        new_retry_count = current_retry_count + 1
        
        # Check if max retries exceeded
        if new_retry_count > max_retries:
            logger.warning(
                f"Max retries ({max_retries}) exceeded. "
                f"Current retry count: {current_retry_count}"
            )
            new_state = state.copy()
            new_state["retry_count"] = new_retry_count
            new_state["current_task"] = f"Max retries ({max_retries}) exceeded"
            # Don't clear validation_errors if max retries exceeded
            return new_state
        
        logger.info(
            f"Retry node: Incrementing retry count from {current_retry_count} to {new_retry_count}"
        )
        
        # Create new state with updates
        new_state = state.copy()
        
        # Increment retry count
        new_state["retry_count"] = new_retry_count
        
        # Modify plan to improve queries
        modified_plan = _modify_plan_for_retry(plan.copy(), new_retry_count)
        new_state["plan"] = modified_plan
        
        # Clear validation errors to allow retry
        new_state["validation_errors"] = []
        
        # Update current task
        new_state["current_task"] = f"Retry attempt {new_retry_count}/{max_retries}"
        
        logger.info(
            f"Retry node completed: Modified plan with {len(modified_plan.get('tasks', []))} tasks"
        )
        
        return new_state
    
    return retry_node


def _modify_plan_for_retry(plan: dict[str, Any], retry_count: int) -> dict[str, Any]:
    """Modify plan to improve search queries for retry.
    
    This function enhances the plan's tasks and queries to improve search
    results on retry. It adds more context, refines queries, and may
    add additional search terms.
    
    Args:
        plan: Plan dictionary to modify
        retry_count: Current retry count (used to determine modification intensity)
    
    Returns:
        Modified plan dictionary with improved queries
    """
    modified_plan = plan.copy()
    tasks = modified_plan.get("tasks", [])
    
    if not tasks:
        return modified_plan
    
    # Modify tasks to improve search queries
    modified_tasks = []
    for task in tasks:
        if isinstance(task, str):
            # Enhance task with more context and specificity
            enhanced_task = _enhance_task_query(task, retry_count)
            modified_tasks.append(enhanced_task)
        else:
            # Keep non-string tasks as-is
            modified_tasks.append(task)
    
    modified_plan["tasks"] = modified_tasks
    
    # Increase minimum_results if specified (to get more data on retry)
    if "minimum_results" in modified_plan:
        current_min = modified_plan.get("minimum_results", 4)
        # Increase by 20% per retry, rounded up
        modified_plan["minimum_results"] = int(current_min * (1 + 0.2 * retry_count))
    
    return modified_plan


def _enhance_task_query(task: str, retry_count: int) -> str:
    """Enhance a task query with additional context for retry.
    
    Adds more specific terms, context keywords, and refinements to improve
    search query effectiveness on retry.
    
    Args:
        task: Original task string
        retry_count: Current retry count (higher = more aggressive enhancement)
    
    Returns:
        Enhanced task string with improved query terms
    """
    task = task.strip()
    
    # Add context keywords based on retry count
    enhancements = []
    
    if retry_count >= 2:
        # On second retry, add more specific terms
        if "competitor" not in task.lower():
            enhancements.append("competitor")
        if "analysis" not in task.lower():
            enhancements.append("analysis")
    
    if retry_count >= 3:
        # On third retry, add comparison and market terms
        if "comparison" not in task.lower():
            enhancements.append("comparison")
        if "market" not in task.lower():
            enhancements.append("market")
    
    # Build enhanced query
    if enhancements:
        enhanced = f"{task} {' '.join(enhancements)}"
    else:
        enhanced = task
    
    # Add specificity indicators
    if retry_count > 1:
        enhanced = f"{enhanced} detailed comprehensive"
    
    return enhanced.strip()


# For backward compatibility, provide a direct function that gets max_retries from config
def retry_node(state: WorkflowState) -> WorkflowState:
    """Node that handles retry logic (direct function version).
    
    This version gets max_retries from config. For better control, use
    create_retry_node() instead.
    
    Args:
        state: Current workflow state containing:
            - plan: Execution plan with tasks
            - retry_count: Current retry count
            - validation_errors: List of validation errors
    
    Returns:
        Updated state with retry_count incremented and plan modified
    
    Raises:
        WorkflowError: If plan is missing or max retries exceeded
    """
    from src.config import get_config
    
    config = get_config()
    max_retries = config.max_retries
    
    # Create node using factory function
    node_func = create_retry_node(max_retries=max_retries)
    return node_func(state)
