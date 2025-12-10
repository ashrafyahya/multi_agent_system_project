"""Tests for exception implementations.

This module contains unit tests for all exception classes to verify
inheritance, error messages, and context handling.
"""

import pytest

from src.exceptions.base import BaseWorkflowError
from src.exceptions.collector_error import CollectorError
from src.exceptions.validation_error import ValidationError
from src.exceptions.workflow_error import WorkflowError


class TestBaseWorkflowError:
    """Tests for BaseWorkflowError."""
    
    def test_base_error_creation(self) -> None:
        """Test that BaseWorkflowError can be created with a message."""
        error = BaseWorkflowError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.context == {}
    
    def test_base_error_with_context(self) -> None:
        """Test that BaseWorkflowError can include context."""
        context = {"key": "value", "count": 42}
        error = BaseWorkflowError("Test error", context=context)
        assert error.context == context
        assert error.context["key"] == "value"
        assert error.context["count"] == 42
    
    def test_base_error_str_representation(self) -> None:
        """Test string representation of BaseWorkflowError."""
        error = BaseWorkflowError("Test message")
        assert str(error) == "Test message"
    
    def test_base_error_repr_representation(self) -> None:
        """Test detailed representation of BaseWorkflowError."""
        error = BaseWorkflowError("Test message", context={"key": "value"})
        repr_str = repr(error)
        assert "BaseWorkflowError" in repr_str
        assert "Test message" in repr_str
        assert "context" in repr_str


class TestValidationError:
    """Tests for ValidationError."""
    
    def test_validation_error_inheritance(self) -> None:
        """Test that ValidationError inherits from BaseWorkflowError."""
        assert issubclass(ValidationError, BaseWorkflowError)
        assert issubclass(ValidationError, Exception)
    
    def test_validation_error_message(self) -> None:
        """Test that ValidationError includes error message."""
        error = ValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert error.message == "Validation failed"
    
    def test_validation_error_with_context(self) -> None:
        """Test that ValidationError can include context."""
        context = {"field": "email", "value": "invalid"}
        error = ValidationError("Invalid email", context=context)
        assert error.context == context
        assert error.context["field"] == "email"
    
    def test_validation_error_can_be_caught_specifically(self) -> None:
        """Test that ValidationError can be caught specifically."""
        try:
            raise ValidationError("Test validation error")
        except ValidationError as e:
            assert str(e) == "Test validation error"
        except Exception:
            pytest.fail("ValidationError should be caught specifically")
    
    def test_validation_error_can_be_caught_by_base(self) -> None:
        """Test that ValidationError can be caught by base exception."""
        try:
            raise ValidationError("Test validation error")
        except BaseWorkflowError as e:
            assert isinstance(e, ValidationError)
            assert str(e) == "Test validation error"
        except Exception:
            pytest.fail("ValidationError should be caught by BaseWorkflowError")


class TestCollectorError:
    """Tests for CollectorError."""
    
    def test_collector_error_inheritance(self) -> None:
        """Test that CollectorError inherits from BaseWorkflowError."""
        assert issubclass(CollectorError, BaseWorkflowError)
        assert issubclass(CollectorError, Exception)
    
    def test_collector_error_message(self) -> None:
        """Test that CollectorError includes error message."""
        error = CollectorError("Collection failed")
        assert str(error) == "Collection failed"
        assert error.message == "Collection failed"
    
    def test_collector_error_with_context(self) -> None:
        """Test that CollectorError can include context."""
        context = {"query": "test query", "source": "web_search"}
        error = CollectorError("Search failed", context=context)
        assert error.context == context
        assert error.context["query"] == "test query"
    
    def test_collector_error_can_be_caught_specifically(self) -> None:
        """Test that CollectorError can be caught specifically."""
        try:
            raise CollectorError("Test collector error")
        except CollectorError as e:
            assert str(e) == "Test collector error"
        except Exception:
            pytest.fail("CollectorError should be caught specifically")
    
    def test_collector_error_can_be_caught_by_base(self) -> None:
        """Test that CollectorError can be caught by base exception."""
        try:
            raise CollectorError("Test collector error")
        except BaseWorkflowError as e:
            assert isinstance(e, CollectorError)
            assert str(e) == "Test collector error"
        except Exception:
            pytest.fail("CollectorError should be caught by BaseWorkflowError")


class TestWorkflowError:
    """Tests for WorkflowError."""
    
    def test_workflow_error_inheritance(self) -> None:
        """Test that WorkflowError inherits from BaseWorkflowError."""
        assert issubclass(WorkflowError, BaseWorkflowError)
        assert issubclass(WorkflowError, Exception)
    
    def test_workflow_error_message(self) -> None:
        """Test that WorkflowError includes error message."""
        error = WorkflowError("Workflow failed")
        assert str(error) == "Workflow failed"
        assert error.message == "Workflow failed"
    
    def test_workflow_error_with_context(self) -> None:
        """Test that WorkflowError can include context."""
        context = {"node": "collector", "retry_count": 3}
        error = WorkflowError("Max retries exceeded", context=context)
        assert error.context == context
        assert error.context["retry_count"] == 3
    
    def test_workflow_error_can_be_caught_specifically(self) -> None:
        """Test that WorkflowError can be caught specifically."""
        try:
            raise WorkflowError("Test workflow error")
        except WorkflowError as e:
            assert str(e) == "Test workflow error"
        except Exception:
            pytest.fail("WorkflowError should be caught specifically")
    
    def test_workflow_error_can_be_caught_by_base(self) -> None:
        """Test that WorkflowError can be caught by base exception."""
        try:
            raise WorkflowError("Test workflow error")
        except BaseWorkflowError as e:
            assert isinstance(e, WorkflowError)
            assert str(e) == "Test workflow error"
        except Exception:
            pytest.fail("WorkflowError should be caught by BaseWorkflowError")


class TestExceptionHierarchy:
    """Tests for exception hierarchy and relationships."""
    
    def test_all_exceptions_inherit_from_base(self) -> None:
        """Test that all custom exceptions inherit from BaseWorkflowError."""
        exceptions = [ValidationError, CollectorError, WorkflowError]
        for exc_class in exceptions:
            assert issubclass(exc_class, BaseWorkflowError), (
                f"{exc_class.__name__} should inherit from BaseWorkflowError"
            )
    
    def test_exception_specificity(self) -> None:
        """Test that exceptions can be caught in order of specificity."""
        try:
            raise ValidationError("Test")
        except ValidationError:
            # Should catch most specific first
            pass
        except BaseWorkflowError:
            pytest.fail("Should catch ValidationError before BaseWorkflowError")
        except Exception:
            pytest.fail("Should catch ValidationError before generic Exception")
    
    def test_exception_chaining(self) -> None:
        """Test that exceptions can be chained properly."""
        original_error = ValueError("Original error")
        try:
            raise CollectorError("Collection failed") from original_error
        except CollectorError as e:
            assert e.__cause__ == original_error
            assert isinstance(e.__cause__, ValueError)


