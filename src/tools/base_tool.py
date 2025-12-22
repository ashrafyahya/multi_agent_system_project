"""Base tool interface for all tools in the system.

This module defines the abstract base class that tools can optionally
inherit from. However, tools should primarily be implemented as stateless
functions decorated with `@tool` from LangChain, following the Tool Pattern.

The BaseTool class provides a common interface for tools that need to
be implemented as classes rather than functions.
"""

from abc import ABC, abstractmethod
from typing import Any

from src.exceptions.collector_error import CollectorError


class BaseTool(ABC):
    """Base class for all tools.
    
    This abstract base class defines the interface that all tools must
    implement. However, most tools should be implemented as stateless
    functions decorated with `@tool` from LangChain rather than classes.
    
    This class is useful for tools that require more complex state management
    or dependency injection, but should be used sparingly. Prefer function-based
    tools decorated with `@tool` for simplicity and statelessness.
    
    Attributes:
        name: Name of the tool (must be implemented by subclasses)
    """
    
    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool with given parameters.
        
        This method performs the tool's operation and returns structured
        results. Tools should handle errors gracefully and return error
        information in the result dictionary rather than raising exceptions
        when possible.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Dictionary containing tool results. Should include:
            - 'success': Boolean indicating if operation succeeded
            - 'data': Tool-specific result data
            - 'error': Optional error message if operation failed
            
        Raises:
            CollectorError: If tool execution fails critically and cannot
                be represented in the return value
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool name.
        
        Returns:
            String name of the tool, used for identification and logging
        """
        pass
    
    def validate_inputs(self, **kwargs: Any) -> None:
        """Validate tool inputs before execution.
        
        This method can be overridden by subclasses to perform input
        validation. By default, it does nothing. Subclasses should raise
        ValueError for invalid inputs.
        
        Args:
            **kwargs: Tool inputs to validate
            
        Raises:
            ValueError: If inputs are invalid
        """
        pass
    
    def handle_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Handle errors during tool execution.
        
        This method provides a standard way to handle errors and return
        them in the tool result format. Subclasses can override this to
        customize error handling.
        
        Args:
            error: Exception that occurred
            context: Optional context dictionary with additional information
            
        Returns:
            Dictionary with error information:
            - 'success': False
            - 'error': Error message string
            - 'context': Optional context information
        """
        error_msg = str(error)
        if context:
            error_msg += f" (Context: {context})"
        
        return {
            "success": False,
            "error": error_msg,
            "context": context or {},
        }


