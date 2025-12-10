"""Tests for workflow state management.

This module contains unit tests for WorkflowState TypedDict to verify
type safety, state creation, and state transitions.
"""

from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import WorkflowState, create_initial_state


class TestWorkflowStateCreation:
    """Tests for WorkflowState creation."""
    
    def test_state_creation_with_all_fields(self) -> None:
        """Test that WorkflowState can be created with all fields."""
        state: WorkflowState = {
            "messages": [HumanMessage(content="Test message")],
            "plan": {"tasks": ["task1", "task2"]},
            "collected_data": {"competitors": []},
            "insights": {"swot": {}},
            "report": "Test report",
            "retry_count": 0,
            "current_task": "collecting",
            "validation_errors": [],
        }
        
        assert state["retry_count"] == 0
        assert state["current_task"] == "collecting"
        assert len(state["messages"]) == 1
        assert state["plan"] is not None
        assert state["plan"]["tasks"] == ["task1", "task2"]
    
    def test_state_creation_with_minimal_fields(self) -> None:
        """Test that WorkflowState can be created with minimal required fields."""
        state: WorkflowState = {
            "messages": [],
            "retry_count": 0,
            "validation_errors": [],
        }
        
        assert state["retry_count"] == 0
        assert state["messages"] == []
        assert state["validation_errors"] == []
        assert state.get("plan") is None
        assert state.get("collected_data") is None
    
    def test_state_with_optional_fields_none(self) -> None:
        """Test that optional fields can be explicitly set to None."""
        state: WorkflowState = {
            "messages": [],
            "plan": None,
            "collected_data": None,
            "insights": None,
            "report": None,
            "retry_count": 0,
            "current_task": None,
            "validation_errors": [],
        }
        
        assert state["plan"] is None
        assert state["collected_data"] is None
        assert state["insights"] is None
        assert state["report"] is None
        assert state["current_task"] is None


class TestCreateInitialState:
    """Tests for create_initial_state helper function."""
    
    def test_create_initial_state_without_message(self) -> None:
        """Test creating initial state without a message."""
        state = create_initial_state()
        
        assert state["messages"] == []
        assert state["retry_count"] == 0
        assert state["plan"] is None
        assert state["collected_data"] is None
        assert state["insights"] is None
        assert state["report"] is None
        assert state["current_task"] is None
        assert state["validation_errors"] == []
    
    def test_create_initial_state_with_string_message(self) -> None:
        """Test creating initial state with a string message."""
        message_text = "Analyze competitors in the SaaS market"
        state = create_initial_state(message_text)
        
        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)
        assert state["messages"][0].content == message_text
        assert state["retry_count"] == 0
    
    def test_create_initial_state_with_base_message(self) -> None:
        """Test creating initial state with a BaseMessage instance."""
        message = SystemMessage(content="You are a competitor analysis agent")
        state = create_initial_state(message)
        
        assert len(state["messages"]) == 1
        assert state["messages"][0] == message
        assert isinstance(state["messages"][0], SystemMessage)
    
    def test_create_initial_state_defaults(self) -> None:
        """Test that create_initial_state sets all defaults correctly."""
        state = create_initial_state()
        
        # Required fields with defaults
        assert state["retry_count"] == 0
        assert isinstance(state["messages"], list)
        assert isinstance(state["validation_errors"], list)
        
        # Optional fields should be None
        assert state["plan"] is None
        assert state["collected_data"] is None
        assert state["insights"] is None
        assert state["report"] is None
        assert state["current_task"] is None


class TestStateTypeSafety:
    """Tests for state type safety."""
    
    def test_state_retry_count_is_int(self) -> None:
        """Test that retry_count must be an integer."""
        state: WorkflowState = {
            "messages": [],
            "retry_count": 5,
            "validation_errors": [],
        }
        
        assert isinstance(state["retry_count"], int)
        assert state["retry_count"] == 5
    
    def test_state_messages_is_list(self) -> None:
        """Test that messages must be a list of BaseMessage."""
        state: WorkflowState = {
            "messages": [HumanMessage(content="Test")],
            "retry_count": 0,
            "validation_errors": [],
        }
        
        assert isinstance(state["messages"], list)
        assert all(isinstance(msg, HumanMessage) for msg in state["messages"])
    
    def test_state_validation_errors_is_list(self) -> None:
        """Test that validation_errors must be a list of strings."""
        state: WorkflowState = {
            "messages": [],
            "retry_count": 0,
            "validation_errors": ["Error 1", "Error 2"],
        }
        
        assert isinstance(state["validation_errors"], list)
        assert all(isinstance(err, str) for err in state["validation_errors"])
        assert len(state["validation_errors"]) == 2
    
    def test_state_plan_is_dict_or_none(self) -> None:
        """Test that plan can be a dict or None."""
        # Test with dict
        state_with_plan: WorkflowState = {
            "messages": [],
            "plan": {"tasks": ["task1"]},
            "retry_count": 0,
            "validation_errors": [],
        }
        assert isinstance(state_with_plan["plan"], dict)
        
        # Test with None
        state_without_plan: WorkflowState = {
            "messages": [],
            "plan": None,
            "retry_count": 0,
            "validation_errors": [],
        }
        assert state_without_plan["plan"] is None


class TestStateTransitions:
    """Tests for state transitions and updates."""
    
    def test_state_can_be_updated(self) -> None:
        """Test that state fields can be updated."""
        state: WorkflowState = create_initial_state()
        
        # Update retry count
        state["retry_count"] = 1
        assert state["retry_count"] == 1
        
        # Update current task
        state["current_task"] = "collecting"
        assert state["current_task"] == "collecting"
        
        # Update plan
        state["plan"] = {"tasks": ["new_task"]}
        assert state["plan"] is not None
        assert state["plan"]["tasks"] == ["new_task"]
    
    def test_state_validation_errors_can_be_appended(self) -> None:
        """Test that validation errors can be appended."""
        state: WorkflowState = create_initial_state()
        
        state["validation_errors"].append("First error")
        state["validation_errors"].append("Second error")
        
        assert len(state["validation_errors"]) == 2
        assert "First error" in state["validation_errors"]
        assert "Second error" in state["validation_errors"]
    
    def test_state_messages_can_be_added(self) -> None:
        """Test that messages can be added to state."""
        state: WorkflowState = create_initial_state()
        
        initial_count = len(state["messages"])
        state["messages"].append(HumanMessage(content="New message"))
        
        assert len(state["messages"]) == initial_count + 1
        assert state["messages"][-1].content == "New message"


class TestStateImmutability:
    """Tests for state immutability considerations."""
    
    def test_state_dict_can_be_copied(self) -> None:
        """Test that state can be safely copied."""
        original_state: WorkflowState = create_initial_state("Test")
        original_state["retry_count"] = 5
        
        # Create a copy (in real workflow, LangGraph handles this)
        copied_state: WorkflowState = {
            "messages": original_state["messages"].copy(),
            "plan": original_state["plan"],
            "collected_data": original_state["collected_data"],
            "insights": original_state["insights"],
            "report": original_state["report"],
            "retry_count": original_state["retry_count"],
            "current_task": original_state["current_task"],
            "validation_errors": original_state["validation_errors"].copy(),
        }
        
        # Modify copy
        copied_state["retry_count"] = 10
        
        # Original should be unchanged
        assert original_state["retry_count"] == 5
        assert copied_state["retry_count"] == 10


