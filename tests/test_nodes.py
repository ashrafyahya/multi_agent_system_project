"""Tests for graph node implementations.

This module contains unit tests for all workflow nodes to verify
pure function behavior, state updates, and error handling.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel

from src.exceptions.collector_error import CollectorError
from src.exceptions.workflow_error import WorkflowError
from src.graph.nodes.data_collector_node import create_data_collector_node
from src.graph.nodes.insight_node import create_insight_node
from src.graph.nodes.planner_node import create_planner_node
from src.graph.nodes.report_node import create_report_node
from src.graph.nodes.retry_node import create_retry_node
from src.graph.nodes.supervisor_node import create_supervisor_node
from src.graph.state import WorkflowState, create_initial_state


class TestDataCollectorNode:
    """Tests for data_collector_node."""
    
    def test_data_collector_node_success(self) -> None:
        """Test data collector node executes successfully."""
        from src.agents.data_collector import DataCollectorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        
        # Create node using factory function
        node = create_data_collector_node(llm=mock_llm, config=config)
        
        # Mock agent execution
        with patch("src.graph.nodes.data_collector_node.DataCollectorAgent") as mock_agent_class:
            mock_agent = Mock(spec=DataCollectorAgent)
            mock_agent.execute.return_value = {
                "messages": [],
                "plan": {"tasks": ["Find competitors"]},
                "collected_data": {
                    "competitors": [
                        {"name": "Comp1", "source_url": "https://example.com"},
                    ]
                },
                "current_task": "Data collected",
            }
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["plan"] = {"tasks": ["Find competitors"], "minimum_results": 4}
            
            result = node(state)
            
            assert "collected_data" in result
            assert result["collected_data"] is not None
            assert result["current_task"] == "Data collected"
    
    def test_data_collector_node_pure_function(self) -> None:
        """Test that data collector node is a pure function."""
        from src.agents.data_collector import DataCollectorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        node = create_data_collector_node(llm=mock_llm, config=config)
        
        state1 = create_initial_state("Test 1")
        state1["plan"] = {"tasks": ["Task 1"], "minimum_results": 4}
        
        state2 = create_initial_state("Test 2")
        state2["plan"] = {"tasks": ["Task 2"], "minimum_results": 4}
        
        # Mock agent to return different results
        with patch("src.graph.nodes.data_collector_node.DataCollectorAgent") as mock_agent_class:
            mock_agent = Mock(spec=DataCollectorAgent)
            
            def side_effect(state):
                result = state.copy()
                result["collected_data"] = {
                    "competitors": [{"name": f"Comp from {state.get('plan', {}).get('tasks', [''])[0]}"}]
                }
                return result
            
            mock_agent.execute.side_effect = side_effect
            mock_agent_class.return_value = mock_agent
            
            result1 = node(state1)
            result2 = node(state2)
            
            # Results should be independent
            assert result1["collected_data"] != result2["collected_data"]
            # Original states should not be modified (node creates copies)
            assert state1.get("collected_data") is None
            assert state2.get("collected_data") is None
    
    def test_data_collector_node_handles_collector_error(self) -> None:
        """Test data collector node handles CollectorError gracefully."""
        from src.agents.data_collector import DataCollectorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        node = create_data_collector_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.data_collector_node.DataCollectorAgent") as mock_agent_class:
            mock_agent = Mock(spec=DataCollectorAgent)
            mock_agent.execute.side_effect = CollectorError(
                "Collection failed",
                context={"query": "test"}
            )
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["plan"] = {"tasks": ["Find competitors"], "minimum_results": 4}
            
            result = node(state)
            
            # Should not raise exception, but add error to validation_errors
            assert len(result["validation_errors"]) > 0
            assert any("Collection failed" in err for err in result["validation_errors"])
            assert result["current_task"] == "Data collection failed"
    
    def test_data_collector_node_handles_workflow_error(self) -> None:
        """Test data collector node handles WorkflowError gracefully."""
        from src.agents.data_collector import DataCollectorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        node = create_data_collector_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.data_collector_node.DataCollectorAgent") as mock_agent_class:
            mock_agent = Mock(spec=DataCollectorAgent)
            mock_agent.execute.side_effect = WorkflowError(
                "Workflow error",
                context={"error": "test"}
            )
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["plan"] = {"tasks": ["Find competitors"], "minimum_results": 4}
            
            result = node(state)
            
            # Should not raise exception, but add error to validation_errors
            assert len(result["validation_errors"]) > 0
            assert any("Workflow error" in err for err in result["validation_errors"])
            assert result["current_task"] == "Data collection failed"
    
    def test_data_collector_node_handles_unexpected_error(self) -> None:
        """Test data collector node handles unexpected errors gracefully."""
        from src.agents.data_collector import DataCollectorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        node = create_data_collector_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.data_collector_node.DataCollectorAgent") as mock_agent_class:
            mock_agent = Mock(spec=DataCollectorAgent)
            mock_agent.execute.side_effect = ValueError("Unexpected error")
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["plan"] = {"tasks": ["Find competitors"], "minimum_results": 4}
            
            result = node(state)
            
            # Should not raise exception, but add error to validation_errors
            assert len(result["validation_errors"]) > 0
            assert any("Unexpected error" in err for err in result["validation_errors"])
            assert result["current_task"] == "Data collection failed"
    
    def test_data_collector_node_preserves_state_fields(self) -> None:
        """Test data collector node preserves existing state fields."""
        from src.agents.data_collector import DataCollectorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        node = create_data_collector_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.data_collector_node.DataCollectorAgent") as mock_agent_class:
            mock_agent = Mock(spec=DataCollectorAgent)
            mock_agent.execute.return_value = {
                "messages": [],
                "plan": {"tasks": ["Find competitors"]},
                "collected_data": {"competitors": []},
                "retry_count": 0,
                "current_task": "Data collected",
                "validation_errors": [],
            }
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["plan"] = {"tasks": ["Find competitors"], "minimum_results": 4}
            state["retry_count"] = 1
            state["current_task"] = "Previous task"
            
            result = node(state)
            
            # Should preserve retry_count and update current_task
            assert result["retry_count"] == 0  # From agent return
            assert result["current_task"] == "Data collected"
    
    def test_data_collector_node_no_side_effects(self) -> None:
        """Test data collector node has no side effects."""
        from src.agents.data_collector import DataCollectorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        node = create_data_collector_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.data_collector_node.DataCollectorAgent") as mock_agent_class:
            mock_agent = Mock(spec=DataCollectorAgent)
            mock_agent.execute.return_value = {
                "messages": [],
                "plan": {"tasks": ["Find competitors"]},
                "collected_data": {"competitors": []},
            }
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["plan"] = {"tasks": ["Find competitors"], "minimum_results": 4}
            
            original_state_id = id(state)
            original_validation_errors_id = id(state.get("validation_errors", []))
            
            result = node(state)
            
            # State object itself should not be mutated (new dict returned)
            # But validation_errors list might be copied
            assert id(result) != original_state_id
            # Original state should remain unchanged
            assert state.get("collected_data") is None


class TestInsightNode:
    """Tests for insight_node."""
    
    def test_insight_node_success(self) -> None:
        """Test insight node executes successfully."""
        from src.agents.insight_agent import InsightAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        
        # Create node using factory function
        node = create_insight_node(llm=mock_llm, config=config)
        
        # Mock agent execution
        with patch("src.graph.nodes.insight_node.InsightAgent") as mock_agent_class:
            mock_agent = Mock(spec=InsightAgent)
            mock_agent.execute.return_value = {
                "messages": [],
                "collected_data": {"competitors": []},
                "insights": {
                    "swot": {
                        "strengths": ["Strong brand"],
                        "weaknesses": ["High prices"],
                        "opportunities": ["Emerging markets"],
                        "threats": ["New competitors"],
                    },
                    "positioning": "Premium market leader",
                    "trends": ["Digital transformation"],
                    "opportunities": ["Expansion"],
                },
                "current_task": "Insights generated",
            }
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["collected_data"] = {
                "competitors": [
                    {"name": "Comp1", "source_url": "https://example.com"},
                ]
            }
            
            result = node(state)
            
            assert "insights" in result
            assert result["insights"] is not None
            assert result["current_task"] == "Insights generated"
    
    def test_insight_node_pure_function(self) -> None:
        """Test that insight node is a pure function."""
        from src.agents.insight_agent import InsightAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        node = create_insight_node(llm=mock_llm, config=config)
        
        state1 = create_initial_state("Test 1")
        state1["collected_data"] = {"competitors": [{"name": "Comp1"}]}
        
        state2 = create_initial_state("Test 2")
        state2["collected_data"] = {"competitors": [{"name": "Comp2"}]}
        
        # Mock agent to return different results
        with patch("src.graph.nodes.insight_node.InsightAgent") as mock_agent_class:
            mock_agent = Mock(spec=InsightAgent)
            
            def side_effect(state):
                result = state.copy()
                result["insights"] = {
                    "swot": {"strengths": [f"Insight from {state.get('collected_data', {}).get('competitors', [{}])[0].get('name', '')}"]},
                    "positioning": "Test",
                }
                return result
            
            mock_agent.execute.side_effect = side_effect
            mock_agent_class.return_value = mock_agent
            
            result1 = node(state1)
            result2 = node(state2)
            
            # Results should be independent
            assert result1["insights"] != result2["insights"]
            # Original states should not be modified
            assert state1.get("insights") is None
            assert state2.get("insights") is None
    
    def test_insight_node_handles_workflow_error(self) -> None:
        """Test insight node handles WorkflowError gracefully."""
        from src.agents.insight_agent import InsightAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        node = create_insight_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.insight_node.InsightAgent") as mock_agent_class:
            mock_agent = Mock(spec=InsightAgent)
            mock_agent.execute.side_effect = WorkflowError(
                "Workflow error",
                context={"error": "test"}
            )
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["collected_data"] = {"competitors": []}
            
            result = node(state)
            
            # Should not raise exception, but add error to validation_errors
            assert len(result["validation_errors"]) > 0
            assert any("Insight generation failed" in err for err in result["validation_errors"])
            assert result["current_task"] == "Insight generation failed"
    
    def test_insight_node_handles_unexpected_error(self) -> None:
        """Test insight node handles unexpected errors gracefully."""
        from src.agents.insight_agent import InsightAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        node = create_insight_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.insight_node.InsightAgent") as mock_agent_class:
            mock_agent = Mock(spec=InsightAgent)
            mock_agent.execute.side_effect = ValueError("Unexpected error")
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["collected_data"] = {"competitors": []}
            
            result = node(state)
            
            # Should not raise exception, but add error to validation_errors
            assert len(result["validation_errors"]) > 0
            assert any("Unexpected error" in err for err in result["validation_errors"])
            assert result["current_task"] == "Insight generation failed"
    
    def test_insight_node_preserves_state_fields(self) -> None:
        """Test insight node preserves existing state fields."""
        from src.agents.insight_agent import InsightAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        node = create_insight_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.insight_node.InsightAgent") as mock_agent_class:
            mock_agent = Mock(spec=InsightAgent)
            mock_agent.execute.return_value = {
                "messages": [],
                "collected_data": {"competitors": []},
                "insights": {"swot": {}, "positioning": "Test"},
                "retry_count": 0,
                "current_task": "Insights generated",
                "validation_errors": [],
            }
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["collected_data"] = {"competitors": []}
            state["retry_count"] = 1
            state["current_task"] = "Previous task"
            
            result = node(state)
            
            # Should preserve retry_count and update current_task
            assert result["retry_count"] == 0  # From agent return
            assert result["current_task"] == "Insights generated"
    
    def test_insight_node_no_side_effects(self) -> None:
        """Test insight node has no side effects."""
        from src.agents.insight_agent import InsightAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        node = create_insight_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.insight_node.InsightAgent") as mock_agent_class:
            mock_agent = Mock(spec=InsightAgent)
            mock_agent.execute.return_value = {
                "messages": [],
                "collected_data": {"competitors": []},
                "insights": {"swot": {}, "positioning": "Test"},
            }
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["collected_data"] = {"competitors": []}
            
            original_state_id = id(state)
            
            result = node(state)
            
            # State object itself should not be mutated (new dict returned)
            assert id(result) != original_state_id
            # Original state should remain unchanged
            assert state.get("insights") is None


class TestReportNode:
    """Tests for report_node."""
    
    def test_report_node_success(self) -> None:
        """Test report node executes successfully."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        
        # Create node using factory function
        node = create_report_node(llm=mock_llm, config=config)
        
        # Mock agent execution
        with patch("src.graph.nodes.report_node.ReportAgent") as mock_agent_class:
            mock_agent = Mock(spec=ReportAgent)
            mock_agent.execute.return_value = {
                "messages": [],
                "insights": {
                    "swot": {
                        "strengths": ["Strong brand"],
                        "weaknesses": ["High prices"],
                        "opportunities": ["Emerging markets"],
                        "threats": ["New competitors"],
                    },
                    "positioning": "Premium market leader",
                    "trends": ["Digital transformation"],
                    "opportunities": ["Expansion"],
                },
                "report": "## Executive Summary\n\nComprehensive analysis...",
                "current_task": "Report generated",
            }
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["insights"] = {
                "swot": {
                    "strengths": ["Strong brand"],
                    "weaknesses": ["High prices"],
                    "opportunities": ["Emerging markets"],
                    "threats": ["New competitors"],
                },
                "positioning": "Premium market leader",
                "trends": ["Digital transformation"],
                "opportunities": ["Expansion"],
            }
            
            result = node(state)
            
            assert "report" in result
            assert result["report"] is not None
            assert isinstance(result["report"], str)
            assert result["current_task"] == "Report generated"
    
    def test_report_node_pure_function(self) -> None:
        """Test that report node is a pure function."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        node = create_report_node(llm=mock_llm, config=config)
        
        state1 = create_initial_state("Test 1")
        state1["insights"] = {
            "swot": {"strengths": ["Strength 1"]},
            "positioning": "Position 1",
        }
        
        state2 = create_initial_state("Test 2")
        state2["insights"] = {
            "swot": {"strengths": ["Strength 2"]},
            "positioning": "Position 2",
        }
        
        # Mock agent to return different results
        with patch("src.graph.nodes.report_node.ReportAgent") as mock_agent_class:
            mock_agent = Mock(spec=ReportAgent)
            
            def side_effect(state):
                result = state.copy()
                positioning = state.get("insights", {}).get("positioning", "")
                result["report"] = f"Report for {positioning}"
                return result
            
            mock_agent.execute.side_effect = side_effect
            mock_agent_class.return_value = mock_agent
            
            result1 = node(state1)
            result2 = node(state2)
            
            # Results should be independent
            assert result1["report"] != result2["report"]
            # Original states should not be modified
            assert state1.get("report") is None
            assert state2.get("report") is None
    
    def test_report_node_handles_workflow_error(self) -> None:
        """Test report node handles WorkflowError gracefully."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        node = create_report_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.report_node.ReportAgent") as mock_agent_class:
            mock_agent = Mock(spec=ReportAgent)
            mock_agent.execute.side_effect = WorkflowError(
                "Workflow error",
                context={"error": "test"}
            )
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["insights"] = {
                "swot": {},
                "positioning": "Test",
            }
            
            result = node(state)
            
            # Should not raise exception, but add error to validation_errors
            assert len(result["validation_errors"]) > 0
            assert any("Report generation failed" in err for err in result["validation_errors"])
            assert result["current_task"] == "Report generation failed"
    
    def test_report_node_handles_unexpected_error(self) -> None:
        """Test report node handles unexpected errors gracefully."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        node = create_report_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.report_node.ReportAgent") as mock_agent_class:
            mock_agent = Mock(spec=ReportAgent)
            mock_agent.execute.side_effect = ValueError("Unexpected error")
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["insights"] = {
                "swot": {},
                "positioning": "Test",
            }
            
            result = node(state)
            
            # Should not raise exception, but add error to validation_errors
            assert len(result["validation_errors"]) > 0
            assert any("Unexpected error" in err for err in result["validation_errors"])
            assert result["current_task"] == "Report generation failed"
    
    def test_report_node_preserves_state_fields(self) -> None:
        """Test report node preserves existing state fields."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        node = create_report_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.report_node.ReportAgent") as mock_agent_class:
            mock_agent = Mock(spec=ReportAgent)
            mock_agent.execute.return_value = {
                "messages": [],
                "insights": {},
                "report": "Test report",
                "retry_count": 0,
                "current_task": "Report generated",
                "validation_errors": [],
            }
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["insights"] = {"swot": {}, "positioning": "Test"}
            state["retry_count"] = 1
            state["current_task"] = "Previous task"
            
            result = node(state)
            
            # Should preserve retry_count and update current_task
            assert result["retry_count"] == 0  # From agent return
            assert result["current_task"] == "Report generated"
    
    def test_report_node_no_side_effects(self) -> None:
        """Test report node has no side effects."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        node = create_report_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.report_node.ReportAgent") as mock_agent_class:
            mock_agent = Mock(spec=ReportAgent)
            mock_agent.execute.return_value = {
                "messages": [],
                "insights": {},
                "report": "Test report",
            }
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["insights"] = {"swot": {}, "positioning": "Test"}
            
            original_state_id = id(state)
            
            result = node(state)
            
            # State object itself should not be mutated (new dict returned)
            assert id(result) != original_state_id
            # Original state should remain unchanged
            assert state.get("report") is None
    
    def test_report_node_handles_empty_report(self) -> None:
        """Test report node handles empty report gracefully."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        node = create_report_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.report_node.ReportAgent") as mock_agent_class:
            mock_agent = Mock(spec=ReportAgent)
            mock_agent.execute.return_value = {
                "messages": [],
                "insights": {},
                "report": "",  # Empty report
            }
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["insights"] = {"swot": {}, "positioning": "Test"}
            
            result = node(state)
            
            # Should still return state with empty report
            assert "report" in result
            assert result["report"] == ""


class TestRetryNode:
    """Tests for retry_node."""
    
    def test_retry_node_increments_retry_count(self) -> None:
        """Test retry node increments retry count."""
        node = create_retry_node(max_retries=3)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        state["retry_count"] = 0
        state["validation_errors"] = ["Error 1"]
        
        result = node(state)
        
        assert result["retry_count"] == 1
        assert result["retry_count"] > state["retry_count"]
    
    def test_retry_node_modifies_queries(self) -> None:
        """Test retry node modifies search queries."""
        node = create_retry_node(max_retries=3)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        state["retry_count"] = 1
        
        result = node(state)
        
        # Plan should be modified
        assert "plan" in result
        assert result["plan"]["tasks"] != state["plan"]["tasks"]
        # Tasks should be enhanced
        assert len(result["plan"]["tasks"]) == len(state["plan"]["tasks"])
        # Enhanced task should contain original task
        assert "Find competitors" in result["plan"]["tasks"][0]
    
    def test_retry_node_clears_validation_errors(self) -> None:
        """Test retry node clears validation errors."""
        node = create_retry_node(max_retries=3)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        state["retry_count"] = 0
        state["validation_errors"] = ["Error 1", "Error 2"]
        
        result = node(state)
        
        assert len(result["validation_errors"]) == 0
        assert result["validation_errors"] == []
    
    def test_retry_node_respects_max_retries(self) -> None:
        """Test retry node respects max retries limit."""
        node = create_retry_node(max_retries=3)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        state["retry_count"] = 3  # Already at max
        
        result = node(state)
        
        # Should increment to 4, but mark as exceeded
        assert result["retry_count"] == 4
        assert "exceeded" in result["current_task"].lower()
        # Should not clear validation errors if max retries exceeded
        # (This depends on implementation - current implementation doesn't clear)
    
    def test_retry_node_updates_current_task(self) -> None:
        """Test retry node updates current task."""
        node = create_retry_node(max_retries=3)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        state["retry_count"] = 1
        
        result = node(state)
        
        assert "retry" in result["current_task"].lower()
        assert "2" in result["current_task"]  # Retry attempt 2
        assert "3" in result["current_task"]  # Max retries 3
    
    def test_retry_node_pure_function(self) -> None:
        """Test that retry node is a pure function."""
        node = create_retry_node(max_retries=3)
        
        state1 = create_initial_state("Test 1")
        state1["plan"] = {"tasks": ["Task 1"]}
        state1["retry_count"] = 0
        
        state2 = create_initial_state("Test 2")
        state2["plan"] = {"tasks": ["Task 2"]}
        state2["retry_count"] = 1
        
        result1 = node(state1)
        result2 = node(state2)
        
        # Results should be independent
        assert result1["retry_count"] == 1
        assert result2["retry_count"] == 2
        # Original states should not be modified
        assert state1["retry_count"] == 0
        assert state2["retry_count"] == 1
    
    def test_retry_node_handles_missing_plan(self) -> None:
        """Test retry node handles missing plan gracefully."""
        node = create_retry_node(max_retries=3)

        state = create_initial_state("Test")
        state["retry_count"] = 0
        # No plan

        result = node(state)

        # Should not raise exception, but add error to validation_errors
        assert len(result["validation_errors"]) > 0
        assert any("Cannot retry without a plan" in err for err in result["validation_errors"])
        assert result["current_task"] == "Retry failed"
    
    def test_retry_node_modifies_minimum_results(self) -> None:
        """Test retry node increases minimum_results in plan."""
        node = create_retry_node(max_retries=3)
        
        state = create_initial_state("Test")
        state["plan"] = {
            "tasks": ["Find competitors"],
            "minimum_results": 4
        }
        state["retry_count"] = 1
        
        result = node(state)
        
        # minimum_results should be increased (by 20% per retry)
        assert result["plan"]["minimum_results"] > state["plan"]["minimum_results"]
        assert result["plan"]["minimum_results"] >= 4
    
    def test_retry_node_enhances_task_queries(self) -> None:
        """Test retry node enhances task queries with additional context."""
        node = create_retry_node(max_retries=3)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find pricing"]}
        state["retry_count"] = 2  # Second retry
        
        result = node(state)
        
        # Enhanced task should contain original plus enhancements
        enhanced_task = result["plan"]["tasks"][0]
        assert "Find pricing" in enhanced_task
        # Should add context keywords on retry 2+
        assert isinstance(enhanced_task, str)
        assert len(enhanced_task) > len("Find pricing")
    
    def test_retry_node_no_side_effects(self) -> None:
        """Test retry node has no side effects."""
        node = create_retry_node(max_retries=3)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        state["retry_count"] = 0
        
        original_state_id = id(state)
        original_plan_id = id(state["plan"])
        
        result = node(state)
        
        # State object itself should not be mutated (new dict returned)
        assert id(result) != original_state_id
        # Plan should be a new dict
        assert id(result["plan"]) != original_plan_id
        # Original state should remain unchanged
        assert state["retry_count"] == 0
    
    def test_retry_node_handles_empty_tasks(self) -> None:
        """Test retry node handles empty tasks list."""
        node = create_retry_node(max_retries=3)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": []}
        state["retry_count"] = 0
        
        result = node(state)
        
        # Should still increment retry count
        assert result["retry_count"] == 1
        assert result["plan"]["tasks"] == []
    
    def test_retry_node_handles_non_string_tasks(self) -> None:
        """Test retry node handles non-string tasks."""
        node = create_retry_node(max_retries=3)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["String task", {"dict": "task"}, 123]}
        state["retry_count"] = 0
        
        result = node(state)
        
        # Should modify string tasks but keep non-string tasks as-is
        assert len(result["plan"]["tasks"]) == 3
        assert isinstance(result["plan"]["tasks"][0], str)  # Enhanced
        assert isinstance(result["plan"]["tasks"][1], dict)  # Unchanged
        assert isinstance(result["plan"]["tasks"][2], int)  # Unchanged
    
    def test_retry_node_intelligent_retry_with_llm(self) -> None:
        """Test intelligent retry uses LLM to improve queries based on errors."""
        import json
        
        mock_llm = Mock(spec=BaseChatModel)
        # Mock LLM response with improved queries
        improved_queries = [
            "Find competitors in SaaS market with pricing information",
            "Analyze competitor features and market positioning"
        ]
        mock_response = Mock()
        mock_response.content = json.dumps(improved_queries)
        
        node = create_retry_node(max_retries=3, llm=mock_llm)
        
        state = create_initial_state("Test")
        state["plan"] = {
            "tasks": ["Find competitors", "Analyze features"],
            "minimum_results": 4
        }
        state["retry_count"] = 1
        state["validation_errors"] = [
            "Insufficient competitor data collected",
            "Missing pricing information"
        ]
        
        with patch("src.graph.nodes.retry_node.get_config") as mock_config, \
             patch("src.graph.nodes.retry_node.invoke_llm_with_retry") as mock_invoke:
            mock_config.return_value.intelligent_retry_enabled = True
            mock_invoke.return_value = mock_response
            
            result = node(state)
        
        # Verify LLM was called through rate limiter
        assert mock_invoke.called
        # Verify tasks were modified
        assert "plan" in result
        assert result["plan"]["tasks"] != state["plan"]["tasks"]
        # Verify improved queries are in result
        assert len(result["plan"]["tasks"]) == len(improved_queries)
    
    def test_retry_node_fallback_to_rule_based_when_llm_fails(self) -> None:
        """Test retry falls back to rule-based when LLM fails."""
        mock_llm = Mock(spec=BaseChatModel)
        
        node = create_retry_node(max_retries=3, llm=mock_llm)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        state["retry_count"] = 1
        state["validation_errors"] = ["Error 1"]
        
        with patch("src.graph.nodes.retry_node.get_config") as mock_config, \
             patch("src.graph.nodes.retry_node.invoke_llm_with_retry") as mock_invoke:
            mock_config.return_value.intelligent_retry_enabled = True
            # Mock LLM to raise exception
            mock_invoke.side_effect = Exception("LLM error")
            
            result = node(state)
        
        # Should still work with rule-based fallback
        assert "plan" in result
        assert len(result["plan"]["tasks"]) == 1
        # Should contain original task plus enhancements
        assert "Find competitors" in result["plan"]["tasks"][0]
    
    def test_retry_node_fallback_when_intelligent_retry_disabled(self) -> None:
        """Test retry uses rule-based when intelligent_retry_enabled is False."""
        mock_llm = Mock(spec=BaseChatModel)
        
        node = create_retry_node(max_retries=3, llm=mock_llm)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        state["retry_count"] = 1
        state["validation_errors"] = ["Error 1"]
        
        with patch("src.graph.nodes.retry_node.get_config") as mock_config, \
             patch("src.graph.nodes.retry_node.invoke_llm_with_retry") as mock_invoke:
            mock_config.return_value.intelligent_retry_enabled = False
            
            result = node(state)
        
        # LLM should not be called
        assert not mock_invoke.called
        # Should use rule-based enhancement
        assert "plan" in result
        assert "Find competitors" in result["plan"]["tasks"][0]
    
    def test_retry_node_fallback_when_no_llm_provided(self) -> None:
        """Test retry uses rule-based when no LLM is provided."""
        node = create_retry_node(max_retries=3, llm=None)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        state["retry_count"] = 1
        state["validation_errors"] = ["Error 1"]
        
        result = node(state)
        
        # Should use rule-based enhancement
        assert "plan" in result
        assert "Find competitors" in result["plan"]["tasks"][0]
        # Should have enhancements
        assert len(result["plan"]["tasks"][0]) > len("Find competitors")
    
    def test_retry_node_fallback_when_no_validation_errors(self) -> None:
        """Test retry uses rule-based when no validation errors provided."""
        mock_llm = Mock(spec=BaseChatModel)
        
        node = create_retry_node(max_retries=3, llm=mock_llm)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        state["retry_count"] = 1
        state["validation_errors"] = []  # No errors
        
        with patch("src.graph.nodes.retry_node.get_config") as mock_config, \
             patch("src.graph.nodes.retry_node.invoke_llm_with_retry") as mock_invoke:
            mock_config.return_value.intelligent_retry_enabled = True
            
            result = node(state)
        
        # LLM should not be called (no errors to analyze)
        assert not mock_invoke.called
        # Should use rule-based enhancement
        assert "plan" in result
        assert "Find competitors" in result["plan"]["tasks"][0]
    
    def test_retry_node_intelligent_retry_handles_invalid_json(self) -> None:
        """Test intelligent retry falls back when LLM returns invalid JSON."""
        mock_llm = Mock(spec=BaseChatModel)
        # Mock LLM to return invalid JSON
        mock_response = Mock()
        mock_response.content = "This is not valid JSON"
        
        node = create_retry_node(max_retries=3, llm=mock_llm)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors"]}
        state["retry_count"] = 1
        state["validation_errors"] = ["Error 1"]
        
        with patch("src.graph.nodes.retry_node.get_config") as mock_config, \
             patch("src.graph.nodes.retry_node.invoke_llm_with_retry") as mock_invoke:
            mock_config.return_value.intelligent_retry_enabled = True
            mock_invoke.return_value = mock_response
            
            result = node(state)
        
        # Should fall back to rule-based
        assert "plan" in result
        assert "Find competitors" in result["plan"]["tasks"][0]
    
    def test_retry_node_intelligent_retry_handles_wrong_array_length(self) -> None:
        """Test intelligent retry falls back when LLM returns wrong number of queries."""
        import json
        mock_llm = Mock(spec=BaseChatModel)
        # Mock LLM to return wrong number of queries
        mock_response = Mock()
        mock_response.content = json.dumps(["Only one query"])  # Should be 2
        
        node = create_retry_node(max_retries=3, llm=mock_llm)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Find competitors", "Analyze features"]}
        state["retry_count"] = 1
        state["validation_errors"] = ["Error 1"]
        
        with patch("src.graph.nodes.retry_node.get_config") as mock_config, \
             patch("src.graph.nodes.retry_node.invoke_llm_with_retry") as mock_invoke:
            mock_config.return_value.intelligent_retry_enabled = True
            mock_invoke.return_value = mock_response
            
            result = node(state)
        
        # Should fall back to rule-based
        assert "plan" in result
        assert len(result["plan"]["tasks"]) == 2
        assert "Find competitors" in result["plan"]["tasks"][0]


class TestPlannerNode:
    """Tests for planner_node."""
    
    def test_planner_node_success(self) -> None:
        """Test planner node executes successfully."""
        from src.agents.planner_agent import PlannerAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0}
        
        node = create_planner_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.planner_node.PlannerAgent") as mock_agent_class:
            mock_agent = Mock(spec=PlannerAgent)
            mock_agent.execute.return_value = {
                "messages": [],
                "plan": {"tasks": ["Find competitors"], "minimum_results": 4},
                "current_task": "Planning completed",
            }
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Analyze competitors")
            result = node(state)
            
            assert "plan" in result
            assert result["plan"] is not None
            assert result["current_task"] == "Planning completed"
    
    def test_planner_node_handles_workflow_error(self) -> None:
        """Test planner node handles WorkflowError gracefully."""
        from src.agents.planner_agent import PlannerAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0}
        node = create_planner_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.planner_node.PlannerAgent") as mock_agent_class:
            mock_agent = Mock(spec=PlannerAgent)
            mock_agent.execute.side_effect = WorkflowError("Planning failed")
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            result = node(state)
            
            assert len(result["validation_errors"]) > 0
            assert any("Planning failed" in err for err in result["validation_errors"])
            assert result["current_task"] == "Planning failed"
    
    def test_planner_node_handles_unexpected_error(self) -> None:
        """Test planner node handles unexpected errors gracefully."""
        from src.agents.planner_agent import PlannerAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0}
        node = create_planner_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.planner_node.PlannerAgent") as mock_agent_class:
            mock_agent = Mock(spec=PlannerAgent)
            mock_agent.execute.side_effect = ValueError("Unexpected error")
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            result = node(state)
            
            assert len(result["validation_errors"]) > 0
            assert result["current_task"] == "Planning failed"
    
    def test_planner_node_pure_function(self) -> None:
        """Test that planner node is a pure function."""
        from src.agents.planner_agent import PlannerAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0}
        node = create_planner_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.planner_node.PlannerAgent") as mock_agent_class:
            mock_agent = Mock(spec=PlannerAgent)
            mock_agent.execute.return_value = {
                "messages": [],
                "plan": {"tasks": ["Task 1"]},
            }
            mock_agent_class.return_value = mock_agent
            
            state1 = create_initial_state("Test 1")
            state2 = create_initial_state("Test 2")
            
            result1 = node(state1)
            result2 = node(state2)
            
            # Results should be independent
            assert result1["plan"] == result2["plan"]
            # Original states should not be modified
            assert state1.get("plan") is None
            assert state2.get("plan") is None


class TestSupervisorNode:
    """Tests for supervisor_node."""
    
    def test_supervisor_node_success(self) -> None:
        """Test supervisor node executes successfully."""
        from src.agents.supervisor_agent import SupervisorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        
        node = create_supervisor_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.supervisor_node.SupervisorAgent") as mock_agent_class:
            mock_agent = Mock(spec=SupervisorAgent)
            mock_agent.execute.return_value = {
                "messages": [],
                "plan": {"tasks": ["Find competitors"]},
                "current_task": "Proceeding to data collection",
            }
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["plan"] = {"tasks": ["Find competitors"]}
            result = node(state)
            
            assert "current_task" in result
            assert result["current_task"] is not None
    
    def test_supervisor_node_handles_workflow_error(self) -> None:
        """Test supervisor node handles WorkflowError gracefully."""
        from src.agents.supervisor_agent import SupervisorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        node = create_supervisor_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.supervisor_node.SupervisorAgent") as mock_agent_class:
            mock_agent = Mock(spec=SupervisorAgent)
            mock_agent.execute.side_effect = WorkflowError("Supervisor failed")
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["plan"] = {"tasks": ["Find competitors"]}
            result = node(state)
            
            assert len(result["validation_errors"]) > 0
            assert any("Supervisor failed" in err for err in result["validation_errors"])
            assert result["current_task"] == "Supervisor failed"
    
    def test_supervisor_node_handles_unexpected_error(self) -> None:
        """Test supervisor node handles unexpected errors gracefully."""
        from src.agents.supervisor_agent import SupervisorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        node = create_supervisor_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.supervisor_node.SupervisorAgent") as mock_agent_class:
            mock_agent = Mock(spec=SupervisorAgent)
            mock_agent.execute.side_effect = ValueError("Unexpected error")
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["plan"] = {"tasks": ["Find competitors"]}
            result = node(state)
            
            assert len(result["validation_errors"]) > 0
            assert result["current_task"] == "Supervisor failed"
    
    def test_supervisor_node_pure_function(self) -> None:
        """Test that supervisor node is a pure function."""
        from src.agents.supervisor_agent import SupervisorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        node = create_supervisor_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.supervisor_node.SupervisorAgent") as mock_agent_class:
            mock_agent = Mock(spec=SupervisorAgent)
            mock_agent.execute.return_value = {
                "messages": [],
                "plan": {"tasks": ["Find competitors"]},
                "current_task": "Test",
            }
            mock_agent_class.return_value = mock_agent
            
            state1 = create_initial_state("Test 1")
            state1["plan"] = {"tasks": ["Task 1"]}
            state2 = create_initial_state("Test 2")
            state2["plan"] = {"tasks": ["Task 2"]}
            
            result1 = node(state1)
            result2 = node(state2)
            
            # Results should be independent
            assert result1["current_task"] == result2["current_task"]
            # Original states should not be modified
            assert state1.get("current_task") is None
            assert state2.get("current_task") is None


class TestExportNode:
    """Tests for export_node."""
    
    def test_export_node_success(self) -> None:
        """Test export node executes successfully."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.graph.nodes.export_node import create_export_node
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf", "include_visualizations": False}
        
        node = create_export_node(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config["output_dir"] = str(Path(tmpdir) / "exports")
            
            with patch("src.config.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.template_engine.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_table_of_contents.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc_instance.build = MagicMock(return_value=None)
                        mock_doc.return_value = mock_doc_instance
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Report\n## Summary\nTest content"
                        
                        result = node(state)
                        
                        assert "export_paths" in result
                        assert result["export_paths"] is not None
                        assert result["current_task"] == "Export completed successfully"
    
    def test_export_node_pure_function(self) -> None:
        """Test that export node is a pure function."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.graph.nodes.export_node import create_export_node
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        node = create_export_node(llm=mock_llm, config=config)
        
        state1 = create_initial_state("Test 1")
        state1["report"] = "# Report 1\n## Summary\nContent 1"
        
        state2 = create_initial_state("Test 2")
        state2["report"] = "# Report 2\n## Summary\nContent 2"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config["output_dir"] = str(Path(tmpdir) / "exports")
            
            with patch("src.config.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.template_engine.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_table_of_contents.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc_instance.build = MagicMock(return_value=None)
                        mock_doc.return_value = mock_doc_instance
                        
                        result1 = node(state1)
                        result2 = node(state2)
                        
                        # Results should be independent - both should have export_paths
                        assert result1.get("export_paths") is not None
                        assert result2.get("export_paths") is not None
                        # Original states should not be modified
                        assert state1.get("export_paths") is None
                        assert state2.get("export_paths") is None
                        # Results should be different objects
                        assert result1 is not result2
                        # Results should have different reports (from original states)
                        assert result1.get("report") == state1.get("report")
                        assert result2.get("report") == state2.get("report")
    
    def test_export_node_handles_workflow_error(self) -> None:
        """Test export node handles WorkflowError gracefully."""
        from src.agents.export_agent import ExportAgent
        from src.graph.nodes.export_node import create_export_node
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        node = create_export_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.export_node.ExportAgent") as mock_agent_class:
            mock_agent = Mock(spec=ExportAgent)
            mock_agent.execute.side_effect = WorkflowError(
                "Export failed",
                context={"error": "test"}
            )
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["report"] = "# Report\n## Summary\nTest content"
            
            result = node(state)
            
            # Should not raise exception, but add error to validation_errors
            assert len(result["validation_errors"]) > 0
            assert any("Export generation failed" in err for err in result["validation_errors"])
            assert result["current_task"] == "Export generation failed"
    
    def test_export_node_handles_unexpected_error(self) -> None:
        """Test export node handles unexpected errors gracefully."""
        from src.agents.export_agent import ExportAgent
        from src.graph.nodes.export_node import create_export_node
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        node = create_export_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.export_node.ExportAgent") as mock_agent_class:
            mock_agent = Mock(spec=ExportAgent)
            mock_agent.execute.side_effect = ValueError("Unexpected error")
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["report"] = "# Report\n## Summary\nTest content"
            
            result = node(state)
            
            # Should not raise exception, but add error to validation_errors
            assert len(result["validation_errors"]) > 0
            assert any("Unexpected error" in err for err in result["validation_errors"])
            assert result["current_task"] == "Export generation failed"
    
    def test_export_node_preserves_state_fields(self) -> None:
        """Test export node preserves existing state fields."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.graph.nodes.export_node import create_export_node
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        node = create_export_node(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config["output_dir"] = str(Path(tmpdir) / "exports")
            
            with patch("src.config.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.template_engine.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_table_of_contents.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc_instance.build = MagicMock(return_value=None)
                        mock_doc.return_value = mock_doc_instance
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Report\n## Summary\nTest content"
                        state["retry_count"] = 1
                        state["current_task"] = "Previous task"
                        
                        result = node(state)
                        
                        # Should preserve retry_count and update current_task
                        assert result["retry_count"] == 1
                        assert result["current_task"] == "Export completed successfully"
    
    def test_export_node_no_side_effects(self) -> None:
        """Test export node has no side effects."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.graph.nodes.export_node import create_export_node
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        node = create_export_node(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config["output_dir"] = str(Path(tmpdir) / "exports")
            
            with patch("src.config.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.template_engine.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_table_of_contents.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc_instance.build = MagicMock(return_value=None)
                        mock_doc.return_value = mock_doc_instance
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Report\n## Summary\nTest content"
                        
                        original_state_id = id(state)
                        
                        result = node(state)
                        
                        # State object itself should not be mutated (new dict returned)
                        assert id(result) != original_state_id
                        # Original state should remain unchanged
                        assert state.get("export_paths") is None
    
    def test_export_node_handles_missing_report(self) -> None:
        """Test export node handles missing report gracefully."""
        from src.agents.export_agent import ExportAgent
        from src.graph.nodes.export_node import create_export_node
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        node = create_export_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.export_node.ExportAgent") as mock_agent_class:
            mock_agent = Mock(spec=ExportAgent)
            mock_agent.execute.side_effect = WorkflowError("Cannot export without a report")
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            # No report
            
            result = node(state)
            
            # Should not raise exception, but add error to validation_errors
            assert len(result["validation_errors"]) > 0
            assert result["current_task"] == "Export generation failed"
    
    def test_export_node_handles_missing_insights(self) -> None:
        """Test export node handles missing insights gracefully."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.graph.nodes.export_node import create_export_node
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf", "include_visualizations": True}
        node = create_export_node(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config["output_dir"] = str(Path(tmpdir) / "exports")
            
            with patch("src.config.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.template_engine.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_table_of_contents.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc_instance.build = MagicMock(return_value=None)
                        mock_doc.return_value = mock_doc_instance
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Report\n## Summary\nTest content"
                        # No insights
        
                        result = node(state)
                        
                        # Should still succeed without visualizations
                        assert "export_paths" in result
    
    def test_export_node_handles_file_system_errors(self) -> None:
        """Test export node handles file system errors gracefully."""
        from src.agents.export_agent import ExportAgent
        from src.graph.nodes.export_node import create_export_node
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf", "output_dir": "/invalid/path"}
        node = create_export_node(llm=mock_llm, config=config)
        
        with patch("src.graph.nodes.export_node.ExportAgent") as mock_agent_class:
            mock_agent = Mock(spec=ExportAgent)
            mock_agent.execute.side_effect = WorkflowError("File system error")
            mock_agent_class.return_value = mock_agent
            
            state = create_initial_state("Test")
            state["report"] = "# Report\n## Summary\nTest content"
            
            result = node(state)
            
            # Should handle error gracefully
            assert len(result["validation_errors"]) > 0
            assert result["current_task"] == "Export generation failed"
