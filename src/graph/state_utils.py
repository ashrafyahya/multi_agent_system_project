"""State utility functions for immutable state updates.

This module provides helper functions for updating WorkflowState in an
immutable manner using deep copy operations. This prevents accidental
state mutations that can cause bugs in the workflow.

The functions in this module ensure that:
- State updates create new state objects (immutability)
- Nested dictionaries and lists are deep copied
- Original state is never modified
- Type safety is maintained where possible
- Large data can be stored externally when enabled
"""

import copy
import logging
from typing import Any

from src.config import get_config
from src.graph.state import WorkflowState
from src.utils.state_storage import (is_storage_reference, retrieve_large_data,
                                     store_large_data)

logger = logging.getLogger(__name__)


def update_state(state: WorkflowState, **updates: Any) -> WorkflowState:
    """Create new state with updates, ensuring immutability.
    
    This function creates a deep copy of the state and applies the
    provided updates. The original state is never modified, ensuring
    immutability. Nested dictionaries and lists are deep copied to
    prevent accidental mutations.
    
    Args:
        state: Current workflow state to update
        **updates: Keyword arguments representing fields to update.
            Keys should match WorkflowState field names.
            Values will be deep copied before assignment.
    
    Returns:
        New WorkflowState with updates applied. The original state
        remains unchanged.
    """
    # Create deep copy of state to ensure immutability
    new_state = copy.deepcopy(state)
    
    # Apply updates (values are already in updates dict, but we deep copy them too)
    for key, value in updates.items():
        # Deep copy the value to ensure nested structures are also immutable
        new_state[key] = copy.deepcopy(value)  # type: ignore
    
    return new_state


def update_state_field(
    state: WorkflowState,
    field_name: str,
    value: Any
) -> WorkflowState:
    """Update a single field in state immutably.
    
    This is a convenience function for updating a single field. It
    creates a deep copy of the state and updates only the specified
    field. The original state is never modified.
    
    Args:
        state: Current workflow state to update
        field_name: Name of the field to update (must be a valid
            WorkflowState field name)
        value: New value for the field. Will be deep copied before
            assignment to ensure immutability.
    
    Returns:
        New WorkflowState with the specified field updated. The
        original state remains unchanged.
    """
    # Create deep copy of state
    new_state = copy.deepcopy(state)
    
    # Update the specified field with deep copy of value
    new_state[field_name] = copy.deepcopy(value)  # type: ignore
    
    return new_state


def merge_state_updates(
    state: WorkflowState,
    updates: dict[str, Any]
) -> WorkflowState:
    """Merge multiple updates into state immutably.
    
    This function creates a deep copy of the state and applies all
    updates from the provided dictionary. Useful when you have a
    dictionary of updates rather than keyword arguments.
    
    Args:
        state: Current workflow state to update
        updates: Dictionary mapping field names to new values.
            All values will be deep copied before assignment.
    
    Returns:
        New WorkflowState with all updates applied. The original
        state remains unchanged.
    """
    # Create deep copy of state
    new_state = copy.deepcopy(state)
    
    # Apply all updates with deep copy
    for key, value in updates.items():
        new_state[key] = copy.deepcopy(value)  # type: ignore
    
    return new_state


def update_state_with_storage(
    state: WorkflowState,
    **updates: Any
) -> WorkflowState:
    """Create new state with updates, using external storage for large data.
    
    This function is similar to update_state() but automatically stores large
    data externally when state_storage_enabled is True. Large fields like
    'report' and 'collected_data' are stored externally and replaced with
    references in the state.
    
    Args:
        state: Current workflow state to update
        **updates: Keyword arguments representing fields to update.
            Keys should match WorkflowState field names.
            Large fields (report, collected_data) will be stored externally
            if state_storage_enabled is True.
    
    Returns:
        New WorkflowState with updates applied. Large data may be stored
        externally and replaced with references.
    
    Note:
        This function requires state_storage_enabled to be True in config
        to actually use external storage. Otherwise, it behaves like update_state().
    """
    config = get_config()
    
    # Process updates to store large data if enabled
    processed_updates: dict[str, Any] = {}
    
    if config.state_storage_enabled:
        # Fields that should be stored externally if large
        large_fields = {"report", "collected_data", "insights"}
        size_threshold = 10000  # Store if larger than 10KB (approximate)
        
        for key, value in updates.items():
            if key in large_fields and value is not None:
                # Check if value is large enough to store
                try:
                    value_str = str(value)
                    if len(value_str) > size_threshold:
                        # Store externally
                        data_type = key
                        reference = store_large_data(value, data_type=data_type)
                        processed_updates[key] = reference
                        continue
                except Exception as e:
                    # If we can't determine size, don't store externally
                    logger.warning(f"Could not determine size for {key}, storing inline: {e}")
            
            # Store as-is (not large or not a large field)
            processed_updates[key] = value
    else:
        # Storage disabled, use updates as-is
        processed_updates = updates
    
    # Use regular update_state for the actual update
    return update_state(state, **processed_updates)


def retrieve_state_data(
    state: WorkflowState,
    field_name: str
) -> Any:
    """Retrieve data from state, handling storage references.
    
    This function retrieves data from state, automatically resolving
    storage references if the field contains a reference string.
    
    Args:
        state: Current workflow state
        field_name: Name of the field to retrieve
    
    Returns:
        Field value, with storage references automatically resolved
    """
    value = state.get(field_name)
    
    if value is None:
        return None
    
    # Check if value is a storage reference
    if is_storage_reference(value):
        try:
            return retrieve_large_data(value)
        except Exception as e:
            logger.warning(f"Failed to retrieve stored data for {field_name}: {e}")
            return value  # Return reference as fallback
    
    return value

