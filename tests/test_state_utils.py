"""Tests for state utility functions.

This module contains unit tests for state utility functions that ensure
immutable state updates using deep copy operations.
"""

import copy
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from src.graph.state import WorkflowState, create_initial_state
from src.graph.state_utils import (merge_state_updates, update_state,
                                   update_state_field)


class TestUpdateState:
    """Tests for update_state function."""

    def test_update_state_creates_new_state(self) -> None:
        """Test that update_state creates a new state object."""
        original_state = create_initial_state("Test query")
        original_state["retry_count"] = 5

        updated_state = update_state(original_state, retry_count=10)

        # Should be different objects
        assert updated_state is not original_state
        # Original should be unchanged
        assert original_state["retry_count"] == 5
        # Updated should have new value
        assert updated_state["retry_count"] == 10

    def test_update_state_with_single_field(self) -> None:
        """Test updating a single field in state."""
        state = create_initial_state()
        state["current_task"] = "planning"

        updated = update_state(state, current_task="collecting")

        assert updated["current_task"] == "collecting"
        assert state["current_task"] == "planning"  # Original unchanged

    def test_update_state_with_multiple_fields(self) -> None:
        """Test updating multiple fields at once."""
        state = create_initial_state()
        state["retry_count"] = 0
        state["current_task"] = "initial"

        updated = update_state(
            state,
            retry_count=3,
            current_task="retrying",
            validation_errors=["Error 1", "Error 2"],
        )

        assert updated["retry_count"] == 3
        assert updated["current_task"] == "retrying"
        assert updated["validation_errors"] == ["Error 1", "Error 2"]
        # Original unchanged
        assert state["retry_count"] == 0
        assert state["current_task"] == "initial"
        assert state["validation_errors"] == []

    def test_update_state_with_nested_dict(self) -> None:
        """Test that nested dictionaries are deep copied."""
        state = create_initial_state()
        state["plan"] = {"tasks": ["task1", "task2"], "sources": ["source1"]}

        updated = update_state(state, plan={"tasks": ["task3"], "sources": ["source2"]})

        # Updated state should have new plan
        assert updated["plan"] is not None
        assert updated["plan"]["tasks"] == ["task3"]
        # Original should be unchanged
        assert state["plan"] is not None
        assert state["plan"]["tasks"] == ["task1", "task2"]

    def test_update_state_preserves_other_fields(self) -> None:
        """Test that fields not being updated are preserved."""
        state = create_initial_state("Original query")
        state["retry_count"] = 5
        state["current_task"] = "processing"

        updated = update_state(state, retry_count=10)

        # Updated field changed
        assert updated["retry_count"] == 10
        # Other fields preserved
        assert updated["current_task"] == "processing"
        assert len(updated["messages"]) == 1
        assert updated["messages"][0].content == "Original query"

    def test_update_state_with_none_value(self) -> None:
        """Test updating a field to None."""
        state = create_initial_state()
        state["current_task"] = "processing"

        updated = update_state(state, current_task=None)

        assert updated["current_task"] is None
        assert state["current_task"] == "processing"  # Original unchanged

    def test_update_state_with_list_field(self) -> None:
        """Test updating list fields creates new list."""
        state = create_initial_state()
        state["validation_errors"] = ["Error 1"]

        updated = update_state(state, validation_errors=["Error 2", "Error 3"])

        assert updated["validation_errors"] == ["Error 2", "Error 3"]
        assert state["validation_errors"] == ["Error 1"]  # Original unchanged
        # Should be different list objects
        assert updated["validation_errors"] is not state["validation_errors"]


class TestUpdateStateField:
    """Tests for update_state_field function."""

    def test_update_state_field_single_update(self) -> None:
        """Test updating a single field using update_state_field."""
        state = create_initial_state()
        state["retry_count"] = 0

        updated = update_state_field(state, "retry_count", 5)

        assert updated["retry_count"] == 5
        assert state["retry_count"] == 0  # Original unchanged
        assert updated is not state

    def test_update_state_field_with_nested_dict(self) -> None:
        """Test that nested dict updates are deep copied."""
        state = create_initial_state()
        original_plan = {"tasks": ["task1"], "sources": ["source1"]}
        state["plan"] = original_plan

        new_plan = {"tasks": ["task2"], "sources": ["source2"]}
        updated = update_state_field(state, "plan", new_plan)

        # Updated state has new plan
        assert updated["plan"] is not None
        assert updated["plan"]["tasks"] == ["task2"]
        # Original state unchanged
        assert state["plan"] is not None
        assert state["plan"]["tasks"] == ["task1"]
        # Should be different dict objects
        assert updated["plan"] is not state["plan"]
        assert updated["plan"] is not new_plan  # Should be a copy

    def test_update_state_field_with_list(self) -> None:
        """Test that list updates create new list."""
        state = create_initial_state()
        state["validation_errors"] = ["Error 1"]

        new_errors = ["Error 2", "Error 3"]
        updated = update_state_field(state, "validation_errors", new_errors)

        assert updated["validation_errors"] == ["Error 2", "Error 3"]
        assert state["validation_errors"] == ["Error 1"]  # Original unchanged
        # Should be different list objects
        assert updated["validation_errors"] is not state["validation_errors"]
        assert updated["validation_errors"] is not new_errors  # Should be a copy

    def test_update_state_field_preserves_other_fields(self) -> None:
        """Test that other fields are preserved when updating one field."""
        state = create_initial_state("Test")
        state["retry_count"] = 3
        state["current_task"] = "processing"

        updated = update_state_field(state, "retry_count", 5)

        # Updated field changed
        assert updated["retry_count"] == 5
        # Other fields preserved
        assert updated["current_task"] == "processing"
        assert updated["retry_count"] != state["retry_count"]
        assert updated["current_task"] == state["current_task"]


class TestMergeStateUpdates:
    """Tests for merge_state_updates function."""

    def test_merge_state_updates_creates_new_state(self) -> None:
        """Test that merge_state_updates creates a new state."""
        state = create_initial_state()
        state["retry_count"] = 0

        updates = {"retry_count": 5, "current_task": "processing"}
        merged = merge_state_updates(state, updates)

        assert merged is not state
        assert merged["retry_count"] == 5
        assert merged["current_task"] == "processing"
        assert state["retry_count"] == 0  # Original unchanged

    def test_merge_state_updates_with_empty_dict(self) -> None:
        """Test merging with empty updates dict."""
        state = create_initial_state("Test")
        state["retry_count"] = 5

        merged = merge_state_updates(state, {})

        # Should create new state but with same values
        assert merged is not state
        assert merged["retry_count"] == 5
        assert len(merged["messages"]) == 1

    def test_merge_state_updates_with_nested_dicts(self) -> None:
        """Test that nested dicts are properly deep copied."""
        state = create_initial_state()
        state["plan"] = {"tasks": ["task1"], "sources": ["source1"]}

        updates = {"plan": {"tasks": ["task2"], "sources": ["source2"]}}
        merged = merge_state_updates(state, updates)

        assert merged["plan"] is not None
        assert merged["plan"]["tasks"] == ["task2"]
        assert state["plan"] is not None
        assert state["plan"]["tasks"] == ["task1"]  # Original unchanged
        # Should be different dict objects
        assert merged["plan"] is not state["plan"]
        assert merged["plan"] is not updates["plan"]  # Should be a copy

    def test_merge_state_updates_with_lists(self) -> None:
        """Test that lists are properly copied."""
        state = create_initial_state()
        state["validation_errors"] = ["Error 1"]

        updates = {"validation_errors": ["Error 2", "Error 3"]}
        merged = merge_state_updates(state, updates)

        assert merged["validation_errors"] == ["Error 2", "Error 3"]
        assert state["validation_errors"] == ["Error 1"]  # Original unchanged
        # Should be different list objects
        assert merged["validation_errors"] is not state["validation_errors"]
        assert merged["validation_errors"] is not updates["validation_errors"]


class TestStateImmutability:
    """Tests to verify state immutability is maintained."""

    def test_nested_dict_mutation_does_not_affect_original(self) -> None:
        """Test that mutating nested dict in updated state doesn't affect original."""
        state = create_initial_state()
        state["plan"] = {"tasks": ["task1"], "sources": ["source1"]}

        updated = update_state(state, plan={"tasks": ["task2"], "sources": ["source2"]})

        # Mutate nested dict in updated state
        if updated["plan"] is not None:
            updated["plan"]["tasks"].append("task3")

        # Original should be unaffected
        assert state["plan"] is not None
        assert state["plan"]["tasks"] == ["task1"]
        # Updated state should have the mutation
        assert updated["plan"] is not None
        assert updated["plan"]["tasks"] == ["task2", "task3"]

    def test_list_mutation_does_not_affect_original(self) -> None:
        """Test that mutating list in updated state doesn't affect original."""
        state = create_initial_state()
        state["validation_errors"] = ["Error 1"]

        updated = update_state(state, validation_errors=["Error 2"])

        # Mutate list in updated state
        updated["validation_errors"].append("Error 3")

        # Original should be unaffected
        assert state["validation_errors"] == ["Error 1"]
        # Updated state should have the mutation
        assert updated["validation_errors"] == ["Error 2", "Error 3"]

    def test_deep_nested_structure_immutability(self) -> None:
        """Test immutability with deeply nested structures."""
        state = create_initial_state()
        state["collected_data"] = {
            "competitors": [
                {"name": "Comp1", "data": {"revenue": 1000}},
                {"name": "Comp2", "data": {"revenue": 2000}},
            ]
        }

        new_competitors = [
            {"name": "Comp3", "data": {"revenue": 3000}},
        ]
        updated = update_state(
            state, collected_data={"competitors": new_competitors}
        )

        # Mutate deeply nested structure in updated state
        if updated["collected_data"] is not None:
            updated["collected_data"]["competitors"][0]["data"]["revenue"] = 4000

        # Original should be completely unaffected
        assert state["collected_data"] is not None
        assert state["collected_data"]["competitors"][0]["data"]["revenue"] == 1000
        # Updated should have the mutation
        assert updated["collected_data"] is not None
        assert updated["collected_data"]["competitors"][0]["data"]["revenue"] == 4000


class TestStateUtilsErrorHandling:
    """Tests for error handling in state utility functions."""

    def test_update_state_with_invalid_field(self) -> None:
        """Test that update_state accepts any field (TypedDict allows extra fields)."""
        state = create_initial_state()

        # TypedDict with total=False allows extra fields
        # This should not raise an error, but the field won't be type-checked
        updated = update_state(state, invalid_field="test")  # type: ignore

        # Function should still work (TypedDict is permissive)
        assert updated is not state

    def test_update_state_field_with_nonexistent_field(self) -> None:
        """Test updating a field that doesn't exist in TypedDict."""
        state = create_initial_state()

        # Should work but field won't be type-checked
        updated = update_state_field(state, "nonexistent_field", "value")  # type: ignore

        assert updated is not state

    def test_merge_state_updates_with_invalid_fields(self) -> None:
        """Test merging with fields not in TypedDict."""
        state = create_initial_state()

        updates = {"invalid_field": "value"}  # type: ignore
        merged = merge_state_updates(state, updates)

        # Should work (TypedDict is permissive)
        assert merged is not state

