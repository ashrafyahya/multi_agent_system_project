"""Workflow error exception.

This module defines the WorkflowError exception raised when workflow
execution fails. This is a general error for workflow-level failures
that don't fit into more specific exception categories.
"""

from src.exceptions.base import BaseWorkflowError


class WorkflowError(BaseWorkflowError):
    """Raised when workflow execution fails.
    
    This exception is raised for general workflow execution failures
    that don't fit into more specific exception categories. Use this
    for workflow-level errors such as state transition failures,
    node execution failures, or other workflow orchestration issues.
    
    Example:
        ```python
        if state.get("retry_count", 0) >= MAX_RETRIES:
            raise WorkflowError(
                "Maximum retry count exceeded",
                context={"retry_count": state.get("retry_count")}
            )
        ```
    """
    
    pass
