"""Custom exception classes for the competitor analysis system.

This package contains the exception hierarchy:
- BaseWorkflowError: Base exception for all workflow errors
- ValidationError: Raised when validation fails
- CollectorError: Raised when data collection fails
- WorkflowError: Raised when workflow execution fails
"""

from src.exceptions.base import BaseWorkflowError
from src.exceptions.validation_error import ValidationError
from src.exceptions.collector_error import CollectorError
from src.exceptions.workflow_error import WorkflowError

__all__ = [
    "BaseWorkflowError",
    "ValidationError",
    "CollectorError",
    "WorkflowError",
]
