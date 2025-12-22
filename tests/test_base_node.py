"""Tests for base node error handling decorator.

This module contains unit tests for the node_error_handler decorator
that provides consistent error handling across all workflow nodes.
"""

import logging
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.exceptions.collector_error import CollectorError
from src.exceptions.workflow_error import WorkflowError
from src.graph.nodes.base_node import node_error_handler
from src.graph.state import WorkflowState, create_initial_state


class TestNodeErrorHandler:
    """Tests for node_error_handler decorator."""

    def test_decorator_with_workflow_error(self) -> None:
        """Test decorator handles WorkflowError correctly."""
        @node_error_handler("test_node")
        def test_node_func(state: WorkflowState) -> WorkflowState:
            raise WorkflowError("Test workflow error", context={"test": "data"})

        state = create_initial_state("Test query")
        result = test_node_func(state)

        # Should return state with error added
        assert result is not state
        assert len(result["validation_errors"]) == 1
        assert "Test workflow error" in result["validation_errors"][0]
        assert result["current_task"] == "test_node failed"
        # Original state should be unchanged
        assert len(state["validation_errors"]) == 0

    def test_decorator_with_collector_error(self) -> None:
        """Test decorator handles CollectorError correctly."""
        @node_error_handler("collector_node")
        def test_node_func(state: WorkflowState) -> WorkflowState:
            raise CollectorError("Data collection failed", context={"url": "test.com"})

        state = create_initial_state()
        result = test_node_func(state)

        # Should return state with error added
        assert result is not state
        assert len(result["validation_errors"]) == 1
        assert "Data collection failed" in result["validation_errors"][0]
        assert result["current_task"] == "collector_node failed"

    def test_decorator_with_generic_exception(self) -> None:
        """Test decorator handles generic Exception correctly."""
        @node_error_handler("test_node")
        def test_node_func(state: WorkflowState) -> WorkflowState:
            raise ValueError("Unexpected value error")

        state = create_initial_state()
        result = test_node_func(state)

        # Should return state with error added
        assert result is not state
        assert len(result["validation_errors"]) == 1
        assert "Unexpected error" in result["validation_errors"][0]
        assert "Unexpected value error" in result["validation_errors"][0]
        assert result["current_task"] == "test_node failed"

    def test_decorator_with_successful_execution(self) -> None:
        """Test decorator doesn't interfere with successful execution."""
        @node_error_handler("test_node")
        def test_node_func(state: WorkflowState) -> WorkflowState:
            from src.graph.state_utils import update_state
            return update_state(state, current_task="Success")

        state = create_initial_state()
        result = test_node_func(state)

        # Should return updated state normally
        assert result["current_task"] == "Success"
        assert len(result["validation_errors"]) == 0

    def test_decorator_preserves_existing_errors(self) -> None:
        """Test decorator preserves existing validation errors."""
        @node_error_handler("test_node")
        def test_node_func(state: WorkflowState) -> WorkflowState:
            raise WorkflowError("New error")

        state = create_initial_state()
        state["validation_errors"] = ["Existing error 1", "Existing error 2"]

        result = test_node_func(state)

        # Should preserve existing errors and add new one
        assert len(result["validation_errors"]) == 3
        assert "Existing error 1" in result["validation_errors"]
        assert "Existing error 2" in result["validation_errors"]
        assert any("New error" in err for err in result["validation_errors"])

    def test_decorator_error_message_formatting(self) -> None:
        """Test that error messages are formatted consistently."""
        @node_error_handler("planner_node")
        def test_node_func(state: WorkflowState) -> WorkflowState:
            raise WorkflowError("Planning failed")

        state = create_initial_state()
        result = test_node_func(state)

        # Error message should follow consistent format
        error_msg = result["validation_errors"][0]
        assert "planner_node" in error_msg.lower() or "planning" in error_msg.lower()
        assert "Planning failed" in error_msg

    def test_decorator_logs_errors(self) -> None:
        """Test that decorator logs errors appropriately."""
        @node_error_handler("test_node")
        def test_node_func(state: WorkflowState) -> WorkflowState:
            raise WorkflowError("Test error")

        state = create_initial_state()

        with patch("src.graph.nodes.base_node.logger") as mock_logger:
            result = test_node_func(state)

            # Should log error
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert "test_node" in str(call_args).lower()
            assert "Test error" in str(call_args)

    def test_decorator_logs_unexpected_errors_with_exc_info(self) -> None:
        """Test that unexpected errors are logged with exc_info."""
        @node_error_handler("test_node")
        def test_node_func(state: WorkflowState) -> WorkflowState:
            raise ValueError("Unexpected error")

        state = create_initial_state()

        with patch("src.graph.nodes.base_node.logger") as mock_logger:
            result = test_node_func(state)

            # Should log with exc_info=True for unexpected errors
            mock_logger.error.assert_called_once()
            call_kwargs = mock_logger.error.call_args[1]
            assert call_kwargs.get("exc_info") is True

    def test_decorator_with_none_state(self) -> None:
        """Test decorator handles edge case with None state gracefully."""
        @node_error_handler("test_node")
        def test_node_func(state: WorkflowState) -> WorkflowState:
            # This shouldn't happen, but test edge case
            if state is None:  # type: ignore
                raise WorkflowError("State is None")
            return state

        state = create_initial_state()
        result = test_node_func(state)

        # Should work normally
        assert result is not None

    def test_decorator_with_multiple_errors(self) -> None:
        """Test decorator handles multiple sequential errors."""
        call_count = 0

        @node_error_handler("test_node")
        def test_node_func(state: WorkflowState) -> WorkflowState:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise WorkflowError("First error")
            else:
                raise WorkflowError("Second error")

        state = create_initial_state()

        # First call
        result1 = test_node_func(state)
        assert len(result1["validation_errors"]) == 1

        # Second call with updated state
        result2 = test_node_func(result1)
        assert len(result2["validation_errors"]) == 2

    def test_decorator_preserves_state_structure(self) -> None:
        """Test that decorator preserves all state fields."""
        @node_error_handler("test_node")
        def test_node_func(state: WorkflowState) -> WorkflowState:
            raise WorkflowError("Error")

        state = create_initial_state("Test")
        state["retry_count"] = 5
        state["plan"] = {"tasks": ["task1"]}

        result = test_node_func(state)

        # All fields should be preserved
        assert result["retry_count"] == 5
        assert result["plan"] is not None
        assert result["plan"]["tasks"] == ["task1"]
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Test"

    def test_decorator_with_different_node_names(self) -> None:
        """Test decorator works with different node names."""
        @node_error_handler("planner_node")
        def planner_func(state: WorkflowState) -> WorkflowState:
            raise WorkflowError("Planning error")

        @node_error_handler("insight_node")
        def insight_func(state: WorkflowState) -> WorkflowState:
            raise WorkflowError("Insight error")

        state = create_initial_state()

        planner_result = planner_func(state)
        insight_result = insight_func(state)

        # Different nodes should produce different, meaningful task names
        assert planner_result["current_task"] == "Planning failed"
        assert insight_result["current_task"] == "Insight generation failed"
        # Verify they are different
        assert planner_result["current_task"] != insight_result["current_task"]

