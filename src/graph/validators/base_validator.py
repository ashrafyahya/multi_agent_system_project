"""Base validator class for all validators in the system.

This module defines the abstract base class and ValidationResult model
that all validators must implement, following the Validator Pattern.

The Validator Pattern ensures that validators return ValidationResult
objects instead of raising exceptions for validation failures, making
them composable and allowing graceful error handling.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Result of validation operation.
    
    This model represents the result of a validation operation. Validators
    should return ValidationResult objects instead of raising exceptions
    for validation failures, following the Validator Pattern.
    
    Attributes:
        is_valid: Boolean indicating whether validation passed
        errors: List of error messages encountered during validation
        warnings: List of warning messages (non-blocking issues)
    """
    
    model_config = {"extra": "forbid"}
    
    is_valid: bool = Field(
        ...,
        description="Whether validation passed",
    )
    
    errors: list[str] = Field(
        default_factory=list,
        description="List of error messages",
    )
    
    warnings: list[str] = Field(
        default_factory=list,
        description="List of warning messages",
    )
    
    def add_error(self, message: str) -> None:
        """Add an error message and mark validation as invalid.
        
        This method adds an error message to the errors list and
        automatically sets is_valid to False. Errors indicate
        validation failures that prevent proceeding.
        
        Args:
            message: Error message describing the validation failure
        """
        if message and message.strip():
            self.errors.append(message.strip())
            self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message without affecting validation status.
        
        This method adds a warning message to the warnings list.
        Warnings indicate non-blocking issues that don't prevent
        validation from passing but should be noted.
        
        Args:
            message: Warning message describing a non-blocking issue
        """
        if message and message.strip():
            self.warnings.append(message.strip())
    
    def has_errors(self) -> bool:
        """Check if validation result has any errors.
        
        Returns:
            True if there are any errors, False otherwise
        """
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if validation result has any warnings.
        
        Returns:
            True if there are any warnings, False otherwise
        """
        return len(self.warnings) > 0
    
    def get_summary(self) -> str:
        """Get a summary of the validation result.
        
        Returns a human-readable summary of the validation result,
        including error and warning counts.
        
        Returns:
            Summary string describing the validation result
        """
        error_count = len(self.errors)
        warning_count = len(self.warnings)
        
        if self.is_valid:
            if warning_count > 0:
                return f"Validation passed with {warning_count} warning(s)"
            return "Validation passed"
        
        parts = [f"Validation failed: {error_count} error(s)"]
        if warning_count > 0:
            parts.append(f"{warning_count} warning(s)")
        
        return ", ".join(parts)
    
    @classmethod
    def success(cls) -> "ValidationResult":
        """Create a successful validation result.
        
        Factory method to create a ValidationResult indicating
        successful validation with no errors or warnings.
        
        Returns:
            ValidationResult with is_valid=True and empty errors/warnings
        """
        return cls(is_valid=True)
    
    @classmethod
    def failure(cls, *error_messages: str) -> "ValidationResult":
        """Create a failed validation result with error messages.
        
        Factory method to create a ValidationResult indicating
        failed validation with one or more error messages.
        
        Args:
            *error_messages: One or more error message strings
            
        Returns:
            ValidationResult with is_valid=False and provided errors
        """
        result = cls(is_valid=False)
        for message in error_messages:
            result.add_error(message)
        return result


class BaseValidator(ABC):
    """Base class for all validators in the system.
    
    This abstract base class defines the interface that all validators
    must implement. Validators follow the Validator Pattern, returning
    ValidationResult objects instead of raising exceptions for validation
    failures. This makes validators composable and allows graceful error
    handling in the workflow.
    
    All validators must implement:
    - validate(): Perform validation and return ValidationResult
    - name: Property returning the validator's name
    """
    
    @abstractmethod
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate data and return validation result.
        
        This method performs validation on the provided data and returns
        a ValidationResult object indicating success or failure. Validators
        should never raise exceptions for validation failures; instead,
        they should add errors to the ValidationResult.
        
        Args:
            data: Dictionary containing data to validate. The structure
                depends on the specific validator implementation.
        
        Returns:
            ValidationResult object with validation status, errors, and warnings
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the validator name.
        
        Returns:
            String name of the validator, used for identification and logging
        """
        pass
