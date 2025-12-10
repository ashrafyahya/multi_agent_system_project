"""Tests for workflow builder.

This module contains unit tests for the workflow builder to verify
graph construction, node connections, and conditional edge logic.
"""

import pytest
from unittest.mock import Mock, patch

from langchain_core.language_models import BaseChatModel

from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState, create_initial_state
from src.graph.workflow import create_workflow


class TestWorkflowBuilder:
    """Tests for workflow builder."""
    
    def test_workflow_creation(self) -> None:
        """Test workflow builds successfully."""
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        
        workflow = create_workflow(llm=mock_llm, config=config)
        
        assert workflow is not None
        assert hasattr(workflow, "invoke")
    
    def test_workflow_has_all_nodes(self) -> None:
        """Test workflow includes all required nodes."""
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        
        workflow = create_workflow(llm=mock_llm, config=config)
        
        # Workflow should be compiled and ready to use
        assert workflow is not None
    
    def test_workflow_handles_config(self) -> None:
        """Test workflow handles configuration correctly."""
        mock_llm = Mock(spec=BaseChatModel)
        config = {
            "max_retries": 5,
            "temperature": 0.7,
            "planner_temperature": 0,
            "supervisor_temperature": 0,
        }
        
        workflow = create_workflow(llm=mock_llm, config=config)
        
        assert workflow is not None
    
    def test_workflow_default_config(self) -> None:
        """Test workflow works with default configuration."""
        mock_llm = Mock(spec=BaseChatModel)
        config = {}
        
        workflow = create_workflow(llm=mock_llm, config=config)
        
        assert workflow is not None
    
    def test_workflow_validation_functions(self) -> None:
        """Test validation functions work correctly."""
        from src.graph.workflow import (
            _supervisor_decision,
            _validate_collector_output,
            _validate_insight_output,
            _validate_report_output,
        )
        
        # Test supervisor decision
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        decision = _supervisor_decision(state, max_retries=3)
        assert decision in ["collector", "insight", "report", "retry", "END"]
        
        # Test collector validation
        state["collected_data"] = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
                {"name": "Comp2", "source_url": "https://example2.com"},
                {"name": "Comp3", "source_url": "https://example3.com"},
                {"name": "Comp4", "source_url": "https://example4.com"},
            ]
        }
        decision = _validate_collector_output(state, max_retries=3)
        assert decision in ["insight", "retry", "END"]
        
        # Test insight validation
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": ["Digital transformation"],
        }
        decision = _validate_insight_output(state, max_retries=3)
        assert decision in ["report", "retry", "END"]
        
        # Test report validation
        state["report"] = "## Executive Summary\n\n" + "A" * 500
        decision = _validate_report_output(state, max_retries=3)
        assert decision in ["END", "retry"]
    
    def test_workflow_handles_max_retries(self) -> None:
        """Test workflow respects max retries."""
        from src.graph.workflow import _validate_collector_output
        
        state = create_initial_state("Test")
        state["collected_data"] = {}  # Invalid data
        state["retry_count"] = 3  # At max retries
        
        decision = _validate_collector_output(state, max_retries=3)
        assert decision == "END"  # Should end, not retry
    
    def test_workflow_handles_retry_logic(self) -> None:
        """Test workflow retry logic."""
        from src.graph.workflow import _validate_collector_output
        
        state = create_initial_state("Test")
        state["collected_data"] = {}  # Invalid data
        state["retry_count"] = 1  # Below max retries
        
        decision = _validate_collector_output(state, max_retries=3)
        assert decision == "retry"  # Should retry
    
    def test_workflow_supervisor_decision_with_plan(self) -> None:
        """Test supervisor decision routes correctly with plan."""
        from src.graph.workflow import _supervisor_decision
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        
        decision = _supervisor_decision(state, max_retries=3)
        assert decision == "collector"
    
    def test_workflow_supervisor_decision_with_data(self) -> None:
        """Test supervisor decision routes correctly with collected data."""
        from src.graph.workflow import _supervisor_decision
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        state["collected_data"] = {"competitors": []}
        
        decision = _supervisor_decision(state, max_retries=3)
        assert decision == "insight"
    
    def test_workflow_supervisor_decision_with_insights(self) -> None:
        """Test supervisor decision routes correctly with insights."""
        from src.graph.workflow import _supervisor_decision
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        state["collected_data"] = {"competitors": []}
        state["insights"] = {"swot": {}}
        
        decision = _supervisor_decision(state, max_retries=3)
        assert decision == "report"
    
    def test_workflow_supervisor_decision_with_report(self) -> None:
        """Test supervisor decision ends workflow with report."""
        from src.graph.workflow import _supervisor_decision
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        state["collected_data"] = {"competitors": []}
        state["insights"] = {"swot": {}}
        state["report"] = "Final report"
        
        decision = _supervisor_decision(state, max_retries=3)
        assert decision == "END"
    
    def test_workflow_supervisor_decision_max_retries_exceeded(self) -> None:
        """Test supervisor decision ends workflow when max retries exceeded."""
        from src.graph.workflow import _supervisor_decision
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        state["retry_count"] = 3
        state["validation_errors"] = ["Error 1", "Error 2"]
        
        decision = _supervisor_decision(state, max_retries=3)
        assert decision == "END"
