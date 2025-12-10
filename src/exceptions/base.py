"""Base exception class for all workflow errors.

This module defines the BaseWorkflowError class that serves as the base
for all custom exceptions in the system. All workflow-specific exceptions
should inherit from this class to enable proper error handling hierarchy.
"""


class BaseWorkflowError(Exception):
    """Base exception for all workflow errors.
    
    This exception serves as the base class for all custom exceptions
    in the competitor analysis workflow. It provides a common interface
    for error handling and allows catching all workflow-related errors
    with a single exception type.
    
    Attributes:
        message: Error message describing what went wrong
        context: Optional dictionary with additional error context
    """
    
    def __init__(
        self,
        message: str,
        context: dict | None = None
    ) -> None:
        """Initialize base workflow error.
        
        Args:
            message: Human-readable error message
            context: Optional dictionary with additional error context
                (e.g., state information, input parameters)
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        """Return string representation of the error.
        
        Returns:
            Error message string
        """
        return self.message
    
    def __repr__(self) -> str:
        """Return detailed representation of the error.
        
        Returns:
            Detailed error representation including context
        """
        context_str = f", context={self.context}" if self.context else ""
        return f"{self.__class__.__name__}({self.message!r}{context_str})"
