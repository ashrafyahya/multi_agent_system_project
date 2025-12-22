"""Validation error exception.

This module defines the ValidationError exception raised when validation
fails in the workflow. This exception is used when data validation fails
at workflow boundaries (input/output validation).
"""

from src.exceptions.base import BaseWorkflowError


class ValidationError(BaseWorkflowError):
    """Raised when validation fails.
    
    This exception is raised when data validation fails at workflow
    boundaries. It should be used for validation failures that require
    immediate error handling, not for business rule violations that
    should return ValidationResult objects.
    """
    
    pass
