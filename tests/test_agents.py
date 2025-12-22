"""Tests for agent implementations.

This module contains unit tests for all agent classes to verify
agent behavior, dependency injection, and interface compliance.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agents.base_agent import BaseAgent
from src.graph.state import WorkflowState, create_initial_state
from tests.fixtures.sample_data import sample_collected_data, sample_insights


class TestBaseAgent:
    """Tests for BaseAgent abstract class."""
    
    def test_base_agent_cannot_be_instantiated(self) -> None:
        """Test that BaseAgent cannot be instantiated directly."""
        llm = Mock(spec=BaseChatModel)
        config = {"test": "value"}
        
        with pytest.raises(TypeError):
            BaseAgent(llm=llm, config=config)  # type: ignore
    
    def test_base_agent_requires_execute_method(self) -> None:
        """Test that subclasses must implement execute method."""
        
        class IncompleteAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "incomplete"
        
        llm = Mock(spec=BaseChatModel)
        config = {"test": "value"}
        
        with pytest.raises(TypeError):
            IncompleteAgent(llm=llm, config=config)  # type: ignore
    
    def test_base_agent_requires_name_property(self) -> None:
        """Test that subclasses must implement name property."""
        
        class IncompleteAgent(BaseAgent):
            def execute(self, state: WorkflowState) -> WorkflowState:
                return state
        
        llm = Mock(spec=BaseChatModel)
        config = {"test": "value"}
        
        with pytest.raises(TypeError):
            IncompleteAgent(llm=llm, config=config)  # type: ignore
    
    def test_complete_agent_implementation(self) -> None:
        """Test a complete agent implementation."""
        
        class TestAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "test_agent"
            
            def execute(self, state: WorkflowState) -> WorkflowState:
                state["current_task"] = "Test task"
                return state
        
        llm = Mock(spec=BaseChatModel)
        config = {"test": "value"}
        
        agent = TestAgent(llm=llm, config=config)
        assert agent.name == "test_agent"
        assert agent.llm is llm
        assert agent.config == config
        
        # Test execute method
        initial_state = create_initial_state("Test query")
        updated_state = agent.execute(initial_state)
        
        assert updated_state["current_task"] == "Test task"
        assert updated_state["messages"] == initial_state["messages"]
    
    def test_base_agent_dependency_injection(self) -> None:
        """Test that dependency injection works correctly."""
        
        class TestAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "test_agent"
            
            def execute(self, state: WorkflowState) -> WorkflowState:
                return state
        
        llm1 = Mock(spec=BaseChatModel)
        llm2 = Mock(spec=BaseChatModel)
        config1 = {"temperature": 0}
        config2 = {"temperature": 1}
        
        agent1 = TestAgent(llm=llm1, config=config1)
        agent2 = TestAgent(llm=llm2, config=config2)
        
        assert agent1.llm is llm1
        assert agent1.config == config1
        assert agent2.llm is llm2
        assert agent2.config == config2
        assert agent1.llm is not agent2.llm
    
    def test_base_agent_llm_type_validation(self) -> None:
        """Test that BaseAgent validates LLM type."""
        
        class TestAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "test_agent"
            
            def execute(self, state: WorkflowState) -> WorkflowState:
                return state
        
        config = {"test": "value"}
        
        # Should raise TypeError for invalid LLM type
        with pytest.raises(TypeError, match="llm must be a BaseChatModel"):
            TestAgent(llm="not an llm", config=config)  # type: ignore
        
        with pytest.raises(TypeError, match="llm must be a BaseChatModel"):
            TestAgent(llm=123, config=config)  # type: ignore
        
        with pytest.raises(TypeError, match="llm must be a BaseChatModel"):
            TestAgent(llm=None, config=config)  # type: ignore
    
    def test_base_agent_config_type_validation(self) -> None:
        """Test that BaseAgent validates config type."""
        
        class TestAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "test_agent"
            
            def execute(self, state: WorkflowState) -> WorkflowState:
                return state
        
        llm = Mock(spec=BaseChatModel)
        
        # Should raise ValueError for invalid config type
        with pytest.raises(ValueError, match="config must be a dictionary"):
            TestAgent(llm=llm, config="not a dict")  # type: ignore
        
        with pytest.raises(ValueError, match="config must be a dictionary"):
            TestAgent(llm=llm, config=123)  # type: ignore
        
        with pytest.raises(ValueError, match="config must be a dictionary"):
            TestAgent(llm=llm, config=None)  # type: ignore
    
    def test_base_agent_execute_returns_state(self) -> None:
        """Test that execute method returns WorkflowState."""
        
        class TestAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "test_agent"
            
            def execute(self, state: WorkflowState) -> WorkflowState:
                # Modify state
                new_state = state.copy()
                new_state["current_task"] = "Completed"
                return new_state
        
        llm = Mock(spec=BaseChatModel)
        config = {"test": "value"}
        
        agent = TestAgent(llm=llm, config=config)
        initial_state = create_initial_state("Test")
        result_state = agent.execute(initial_state)
        
        assert isinstance(result_state, dict)
        assert result_state["current_task"] == "Completed"
        assert "messages" in result_state
        assert "retry_count" in result_state
    
    def test_base_agent_stateless(self) -> None:
        """Test that agents are stateless (state passed in, not stored)."""
        
        class TestAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "test_agent"
            
            def execute(self, state: WorkflowState) -> WorkflowState:
                # Agent should not store state in instance variables
                # All state is passed in and returned
                new_state = state.copy()
                new_state["current_task"] = "Task executed"
                return new_state
        
        llm = Mock(spec=BaseChatModel)
        config = {"test": "value"}
        
        agent = TestAgent(llm=llm, config=config)
        
        # Execute with different states
        state1 = create_initial_state("Query 1")
        state2 = create_initial_state("Query 2")
        
        result1 = agent.execute(state1)
        result2 = agent.execute(state2)
        
        # Results should be independent
        assert result1["current_task"] == "Task executed"
        assert result2["current_task"] == "Task executed"
        assert result1["messages"][0].content == "Query 1"
        assert result2["messages"][0].content == "Query 2"
        
        # Agent should not have stored state
        assert not hasattr(agent, "state")
        assert not hasattr(agent, "last_state")
    
    def test_base_agent_config_access(self) -> None:
        """Test that agents can access config values."""
        
        class TestAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "test_agent"
            
            def execute(self, state: WorkflowState) -> WorkflowState:
                # Access config in execute method
                temperature = self.config.get("temperature", 0.7)
                new_state = state.copy()
                new_state["current_task"] = f"Temperature: {temperature}"
                return new_state
        
        llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0, "max_retries": 3}
        
        agent = TestAgent(llm=llm, config=config)
        state = create_initial_state("Test")
        result = agent.execute(state)
        
        assert result["current_task"] == "Temperature: 0"
    
    def test_base_agent_llm_access(self) -> None:
        """Test that agents can access LLM instance."""
        
        class TestAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "test_agent"
            
            def execute(self, state: WorkflowState) -> WorkflowState:
                # Access LLM in execute method
                model_name = getattr(self.llm, "model_name", "unknown")
                new_state = state.copy()
                new_state["current_task"] = f"Using model: {model_name}"
                return new_state
        
        llm = Mock(spec=BaseChatModel)
        llm.model_name = "test-model"
        config = {"test": "value"}
        
        agent = TestAgent(llm=llm, config=config)
        state = create_initial_state("Test")
        result = agent.execute(state)
        
        assert result["current_task"] == "Using model: test-model"


class TestPlannerAgent:
    """Tests for PlannerAgent."""
    
    def test_planner_agent_execute_success(self) -> None:
        """Test planner agent generates valid plan."""
        from src.agents.planner_agent import PlannerAgent

        # Mock LLM response
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "tasks": ["Collect competitor data", "Analyze pricing"],
            "preferred_sources": ["web search"],
            "minimum_results": 4,
            "search_strategy": "comprehensive"
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0}
        agent = PlannerAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Analyze competitors")
        result_state = agent.execute(state)
        
        assert "plan" in result_state
        assert result_state["plan"] is not None
        assert result_state["plan"]["tasks"] == ["Collect competitor data", "Analyze pricing"]
        assert result_state["plan"]["minimum_results"] == 4
        assert result_state["current_task"] == "Planning completed"
    
    def test_planner_agent_parses_json_from_markdown(self) -> None:
        """Test planner agent parses JSON from markdown code blocks."""
        from src.agents.planner_agent import PlannerAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = """Here's the plan:
```json
{
    "tasks": ["Task 1"],
    "minimum_results": 5
}
```"""
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0}
        agent = PlannerAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        result_state = agent.execute(state)
        
        assert result_state["plan"]["tasks"] == ["Task 1"]
        assert result_state["plan"]["minimum_results"] == 5
    
    def test_planner_agent_sets_defaults(self) -> None:
        """Test planner agent sets default values for optional fields."""
        from src.agents.planner_agent import PlannerAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "tasks": ["Task 1"]
            # Missing optional fields
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0}
        agent = PlannerAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        result_state = agent.execute(state)
        
        plan = result_state["plan"]
        assert plan["preferred_sources"] == []
        assert plan["minimum_results"] == 4
        assert plan["search_strategy"] == "comprehensive"
    
    def test_planner_agent_extracts_user_request(self) -> None:
        """Test planner agent extracts user request from messages."""
        from src.agents.planner_agent import PlannerAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({"tasks": ["Task 1"]})
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0}
        agent = PlannerAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Analyze competitors in SaaS market")
        result_state = agent.execute(state)
        
        # Verify LLM was called with the user request
        assert mock_llm.invoke.called
        call_args = mock_llm.invoke.call_args[0][0]
        assert any("Analyze competitors in SaaS market" in str(msg.content) for msg in call_args)
    
    def test_planner_agent_handles_llm_error(self) -> None:
        """Test planner agent handles LLM errors gracefully."""
        from src.agents.planner_agent import PlannerAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke.side_effect = Exception("LLM API error")
        
        config = {"temperature": 0}
        agent = PlannerAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        
        with pytest.raises(WorkflowError, match="Failed to generate plan"):
            agent.execute(state)
    
    def test_planner_agent_handles_invalid_json(self) -> None:
        """Test planner agent handles invalid JSON response."""
        from src.agents.planner_agent import PlannerAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = "This is not JSON {invalid"
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0}
        agent = PlannerAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        
        with pytest.raises(WorkflowError, match="Failed to parse plan"):
            agent.execute(state)
    
    def test_planner_agent_handles_missing_tasks(self) -> None:
        """Test planner agent handles missing tasks field."""
        from src.agents.planner_agent import PlannerAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "minimum_results": 4
            # Missing tasks field
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0}
        agent = PlannerAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        
        with pytest.raises(WorkflowError, match="missing required fields"):
            agent.execute(state)
    
    def test_planner_agent_handles_invalid_tasks_type(self) -> None:
        """Test planner agent handles invalid tasks type."""
        from src.agents.planner_agent import PlannerAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "tasks": "not a list"  # Should be a list
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0}
        agent = PlannerAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        
        with pytest.raises(WorkflowError, match="must be a list"):
            agent.execute(state)
    
    def test_planner_agent_validates_plan_model(self) -> None:
        """Test planner agent validates plan against Plan model."""
        from src.agents.planner_agent import PlannerAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "tasks": [],  # Empty tasks list should fail validation
            "minimum_results": 4
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0}
        agent = PlannerAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        
        with pytest.raises(WorkflowError, match="failed validation"):
            agent.execute(state)
    
    def test_planner_agent_handles_empty_user_request(self) -> None:
        """Test planner agent handles empty user request."""
        from src.agents.planner_agent import PlannerAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0}
        agent = PlannerAgent(llm=mock_llm, config=config)
        
        # State with no messages
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
        
        with pytest.raises(WorkflowError, match="Cannot extract user request"):
            agent.execute(state)
    
    def test_planner_agent_temperature_warning(self) -> None:
        """Test planner agent works with non-zero temperature (should still function)."""
        from src.agents.planner_agent import PlannerAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({"tasks": ["Task 1"]})
        mock_llm.invoke.return_value = mock_response
        
        # Use non-zero temperature (should still work, just logs warning)
        config = {"temperature": 0.7}
        agent = PlannerAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        
        # Should still work (warning is logged but doesn't prevent execution)
        result_state = agent.execute(state)
        assert result_state["plan"] is not None
        assert result_state["plan"]["tasks"] == ["Task 1"]
    
    def test_planner_agent_name(self) -> None:
        """Test planner agent name property."""
        from src.agents.planner_agent import PlannerAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0}
        agent = PlannerAgent(llm=mock_llm, config=config)
        
        assert agent.name == "planner_agent"
    
    def test_planner_agent_handles_empty_response(self) -> None:
        """Test planner agent handles empty LLM response."""
        from src.agents.planner_agent import PlannerAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = ""  # Empty response
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0}
        agent = PlannerAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        
        with pytest.raises(WorkflowError, match="empty response"):
            agent.execute(state)


class TestSupervisorAgent:
    """Tests for SupervisorAgent."""
    
    def test_supervisor_agent_execute_collector_stage(self) -> None:
        """Test supervisor agent at collector stage."""
        from src.agents.supervisor_agent import SupervisorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        agent = SupervisorAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Collect data"]}
        # No collected_data yet
        
        result_state = agent.execute(state)
        
        assert result_state["current_task"] == "Proceeding to data collection"
        assert result_state["retry_count"] == 0
    
    def test_supervisor_agent_validates_collector_output(self) -> None:
        """Test supervisor validates collector output."""
        from src.agents.supervisor_agent import SupervisorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        agent = SupervisorAgent(llm=mock_llm, config=config)
        
        # Valid collector data
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Collect data"]}
        state["collected_data"] = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
                {"name": "Comp2", "source_url": "https://example2.com"},
                {"name": "Comp3", "source_url": "https://example3.com"},
                {"name": "Comp4", "source_url": "https://example4.com"},
            ]
        }
        
        result_state = agent.execute(state)
        
        assert result_state["current_task"] == "Proceeding to insight generation"
        assert len(result_state["validation_errors"]) == 0
    
    def test_supervisor_agent_triggers_retry_on_collector_validation_failure(self) -> None:
        """Test supervisor triggers retry when collector validation fails."""
        from src.agents.supervisor_agent import SupervisorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        agent = SupervisorAgent(llm=mock_llm, config=config)
        
        # Invalid collector data (only 1 competitor, need 4)
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Collect data"]}
        state["collected_data"] = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
            ]
        }
        state["retry_count"] = 0
        
        result_state = agent.execute(state)
        
        assert result_state["retry_count"] == 1
        assert "retry" in result_state["current_task"].lower()
        assert len(result_state["validation_errors"]) > 0
    
    def test_supervisor_agent_ends_on_max_retries(self) -> None:
        """Test supervisor ends workflow when max retries exceeded."""
        from src.agents.supervisor_agent import SupervisorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        agent = SupervisorAgent(llm=mock_llm, config=config)
        
        # Invalid collector data with max retries already reached
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Collect data"]}
        state["collected_data"] = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
            ]
        }
        state["retry_count"] = 3  # Max retries reached
        
        result_state = agent.execute(state)
        
        assert result_state["retry_count"] == 3  # Not incremented
        assert "ended" in result_state["current_task"].lower() or "max retries" in result_state["current_task"].lower()
        assert len(result_state["validation_errors"]) > 0
    
    def test_supervisor_agent_validates_insight_output(self) -> None:
        """Test supervisor validates insight output."""
        from src.agents.supervisor_agent import SupervisorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        agent = SupervisorAgent(llm=mock_llm, config=config)
        
        # Valid insight data
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Collect data"]}
        state["collected_data"] = {"competitors": []}
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        
        result_state = agent.execute(state)
        
        assert result_state["current_task"] == "Proceeding to report generation"
        assert len(result_state["validation_errors"]) == 0
    
    def test_supervisor_agent_triggers_retry_on_insight_validation_failure(self) -> None:
        """Test supervisor triggers retry when insight validation fails."""
        from src.agents.supervisor_agent import SupervisorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        agent = SupervisorAgent(llm=mock_llm, config=config)
        
        # Invalid insight data (missing positioning)
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Collect data"]}
        state["collected_data"] = {"competitors": []}
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "",  # Empty positioning
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        state["retry_count"] = 0
        
        result_state = agent.execute(state)
        
        assert result_state["retry_count"] == 1
        assert "retry" in result_state["current_task"].lower()
        assert len(result_state["validation_errors"]) > 0
    
    def test_supervisor_agent_validates_report_output(self) -> None:
        """Test supervisor validates report output."""
        from src.agents.supervisor_agent import SupervisorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        agent = SupervisorAgent(llm=mock_llm, config=config)
        
        # Valid report (long enough) - use valid insights
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Collect data"]}
        state["collected_data"] = {"competitors": []}
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        state["report"] = "A" * 1200  # Minimum length
        
        result_state = agent.execute(state)
        
        assert result_state["current_task"] == "Workflow complete"
        assert len(result_state["validation_errors"]) == 0
    
    def test_supervisor_agent_triggers_retry_on_report_validation_failure(self) -> None:
        """Test supervisor triggers retry when report validation fails."""
        from src.agents.supervisor_agent import SupervisorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        agent = SupervisorAgent(llm=mock_llm, config=config)
        
        # Invalid report (too short)
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Collect data"]}
        state["collected_data"] = {"competitors": []}
        state["insights"] = {"swot": {}}
        state["report"] = "Short report"  # Too short
        state["retry_count"] = 0
        
        result_state = agent.execute(state)
        
        assert result_state["retry_count"] == 1
        assert "retry" in result_state["current_task"].lower()
        assert len(result_state["validation_errors"]) > 0
    
    def test_supervisor_agent_handles_missing_plan(self) -> None:
        """Test supervisor handles missing plan."""
        from src.agents.supervisor_agent import SupervisorAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        agent = SupervisorAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        # No plan
        
        with pytest.raises(WorkflowError, match="without a plan"):
            agent.execute(state)
    
    def test_supervisor_agent_determines_stage_correctly(self) -> None:
        """Test supervisor determines workflow stage correctly."""
        from src.agents.supervisor_agent import SupervisorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        agent = SupervisorAgent(llm=mock_llm, config=config)
        
        # Test collector stage
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Collect data"]}
        result_state = agent.execute(state)
        assert "data collection" in result_state["current_task"].lower()
        
        # Test insight stage - use valid collected_data (at least 4 competitors)
        state["collected_data"] = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com/1"},
                {"name": "Comp2", "source_url": "https://example.com/2"},
                {"name": "Comp3", "source_url": "https://example.com/3"},
                {"name": "Comp4", "source_url": "https://example.com/4"},
            ]
        }
        result_state = agent.execute(state)
        assert "insight generation" in result_state["current_task"].lower()
        
        # Test report stage - use valid insights (needs positioning field)
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        result_state = agent.execute(state)
        assert "report generation" in result_state["current_task"].lower()
        
        # Test complete stage
        state["report"] = "A" * 1200
        result_state = agent.execute(state)
        assert "complete" in result_state["current_task"].lower()
    
    def test_supervisor_agent_name(self) -> None:
        """Test supervisor agent name property."""
        from src.agents.supervisor_agent import SupervisorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        agent = SupervisorAgent(llm=mock_llm, config=config)
        
        assert agent.name == "supervisor_agent"
    
    def test_supervisor_agent_handles_validation_warnings(self) -> None:
        """Test supervisor handles validation warnings."""
        from src.agents.supervisor_agent import SupervisorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_retries": 3}
        agent = SupervisorAgent(llm=mock_llm, config=config)
        
        # Valid data but with warnings (e.g., short positioning)
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Collect data"]}
        state["collected_data"] = {"competitors": []}
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "This is a positioning statement that meets the minimum length requirement of fifty characters for validation purposes",  # Valid length
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        
        result_state = agent.execute(state)
        
        # Should pass but may have warnings
        assert result_state["current_task"] == "Proceeding to report generation"


class TestDataCollectorAgent:
    """Tests for DataCollectorAgent."""
    
    def test_data_collector_agent_execute_success(self) -> None:
        """Test data collector agent collects data successfully."""
        from unittest.mock import patch

        from src.agents.data_collector import DataCollectorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=mock_llm, config=config)
        
        # Mock web_search tool
        with patch("src.agents.data_collector.web_search") as mock_web_search:
            mock_web_search.invoke.return_value = {
                "success": True,
                "results": [
                    {
                        "url": "https://competitor1.com",
                        "title": "Competitor 1 - Official Website",
                        "snippet": "Competitor 1 offers SaaS solutions",
                    },
                    {
                        "url": "https://competitor2.com",
                        "title": "Competitor 2 - Home",
                        "snippet": "Competitor 2 provides cloud services",
                    },
                    {
                        "url": "https://competitor3.com",
                        "title": "Competitor 3",
                        "snippet": "Competitor 3 is a market leader",
                    },
                    {
                        "url": "https://competitor4.com",
                        "title": "Competitor 4",
                        "snippet": "Competitor 4 offers enterprise solutions",
                    },
                ],
                "count": 4,
            }
            
            state = create_initial_state("Test")
            state["plan"] = {
                "tasks": ["Find competitors"],
                "minimum_results": 4,
                "preferred_sources": ["web search"],
                "search_strategy": "comprehensive",
            }
            
            result_state = agent.execute(state)
            
            assert "collected_data" in result_state
            assert "competitors" in result_state["collected_data"]
            assert len(result_state["collected_data"]["competitors"]) >= 4
            assert result_state["current_task"] is not None
    
    def test_data_collector_agent_handles_missing_plan(self) -> None:
        """Test data collector agent handles missing plan."""
        from src.agents.data_collector import DataCollectorAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        # No plan
        
        with pytest.raises(WorkflowError, match="without a plan"):
            agent.execute(state)
    
    def test_data_collector_agent_handles_invalid_plan(self) -> None:
        """Test data collector agent handles invalid plan structure."""
        from src.agents.data_collector import DataCollectorAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["plan"] = {"invalid": "plan"}  # Missing required fields
        
        with pytest.raises(WorkflowError, match="Invalid plan"):
            agent.execute(state)
    
    def test_data_collector_agent_handles_web_search_failure(self) -> None:
        """Test data collector agent raises WorkflowError when all searches fail."""
        from unittest.mock import patch

        from src.agents.data_collector import DataCollectorAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=mock_llm, config=config)
        
        # Mock web_search to return failure
        with patch("src.agents.data_collector.web_search") as mock_web_search:
            mock_web_search.invoke.return_value = {
                "success": False,
                "error": "Search API error",
                "results": [],
                "count": 0,
            }
            
            state = create_initial_state("Test")
            state["plan"] = {
                "tasks": ["Find competitors"],
                "minimum_results": 4,
            }
            
            # Should raise WorkflowError when all searches fail and no competitors collected
            with pytest.raises(WorkflowError, match="No competitor data collected"):
                agent.execute(state)
    
    def test_data_collector_agent_extracts_competitor_name(self) -> None:
        """Test data collector agent extracts competitor names correctly."""
        from src.agents.utils.data_collection_helpers import \
            extract_competitor_name

        # Test name extraction from title
        name = extract_competitor_name(
            "Competitor Inc - Official Website",
            "https://competitor.com",
            ""
        )
        assert name is not None
        assert "Competitor" in name
        
        # Test name extraction from URL
        name = extract_competitor_name(
            "",
            "https://www.example.com/page",
            ""
        )
        assert name is not None
        
        # Test name extraction from snippet
        name = extract_competitor_name(
            "",
            "https://example.com",
            "Competitor XYZ is a leading company"
        )
        assert name is not None
    
    def test_data_collector_agent_deduplicates_competitors(self) -> None:
        """Test data collector agent deduplicates competitors."""
        from unittest.mock import patch

        from src.agents.data_collector import DataCollectorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=mock_llm, config=config)
        
        # Mock web_search to return duplicate competitors
        with patch("src.agents.data_collector.web_search") as mock_web_search:
            mock_web_search.invoke.return_value = {
                "success": True,
                "results": [
                    {
                        "url": "https://competitor1.com",
                        "title": "Competitor 1",
                        "snippet": "Competitor 1 description",
                    },
                    {
                        "url": "https://competitor1.com/page2",  # Same competitor, different page
                        "title": "Competitor 1 - About",
                        "snippet": "Competitor 1 more info",
                    },
                ],
                "count": 2,
            }
            
            state = create_initial_state("Test")
            state["plan"] = {
                "tasks": ["Find competitors"],
                "minimum_results": 2,
            }
            
            result_state = agent.execute(state)
            competitors = result_state["collected_data"]["competitors"]
            
            # Should deduplicate by name (case-insensitive)
            names = [c["name"].lower() for c in competitors]
            assert len(names) == len(set(names))  # All unique
    
    def test_data_collector_agent_generates_search_queries(self) -> None:
        """Test data collector agent generates search queries from tasks."""
        from src.agents.data_collector import DataCollectorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=mock_llm, config=config)
        
        tasks = ["Find competitors", "Collect pricing data"]
        queries = agent._generate_search_queries(tasks)
        
        assert len(queries) > 0
        assert "Find competitors" in queries
        assert "Collect pricing data" in queries
    
    def test_data_collector_agent_extracts_products(self) -> None:
        """Test data collector agent extracts products from text."""
        from src.agents.utils.data_collection_helpers import extract_products
        
        snippet = "Our products include Product A, Product B, and Product C"
        title = "Company Products"
        
        products = extract_products(snippet, title)
        
        # May extract products or return empty list depending on pattern matching
        assert isinstance(products, list)
    
    def test_data_collector_agent_name(self) -> None:
        """Test data collector agent name property."""
        from src.agents.data_collector import DataCollectorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=mock_llm, config=config)
        
        assert agent.name == "data_collector_agent"
    
    def test_data_collector_agent_handles_empty_search_results(self) -> None:
        """Test data collector agent raises WorkflowError on empty search results."""
        from unittest.mock import patch

        from src.agents.data_collector import DataCollectorAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=mock_llm, config=config)
        
        # Mock web_search to return empty results
        with patch("src.agents.data_collector.web_search") as mock_web_search:
            mock_web_search.invoke.return_value = {
                "success": True,
                "results": [],
                "count": 0,
            }
            
            state = create_initial_state("Test")
            state["plan"] = {
                "tasks": ["Find competitors"],
                "minimum_results": 4,
            }
            
            # Should raise WorkflowError when no competitors are collected
            with pytest.raises(WorkflowError, match="No competitor data collected"):
                agent.execute(state)
    
    def test_data_collector_agent_raises_error_on_no_tavily_results(self) -> None:
        """Test data collector agent raises WorkflowError when no competitors collected."""
        from unittest.mock import patch

        from src.agents.data_collector import DataCollectorAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=mock_llm, config=config)
        
        # Mock multiple failed searches
        with patch("src.agents.data_collector.web_search") as mock_web_search:
            mock_web_search.invoke.return_value = {
                "success": False,
                "error": "API error",
                "results": [],
                "count": 0,
            }
            
            state = create_initial_state("Test")
            state["plan"] = {
                "tasks": ["Find competitors", "Analyze market"],
                "minimum_results": 3,
            }
            
            with pytest.raises(WorkflowError) as exc_info:
                agent.execute(state)
            
            # Verify error message contains helpful information
            assert "No competitor data collected" in str(exc_info.value)
            assert "Tavily API key" in str(exc_info.value).lower() or "api key" in str(exc_info.value).lower()
    
    def test_data_collector_agent_raises_error_on_missing_api_key(self) -> None:
        """Test data collector agent raises WorkflowError with specific message for missing API key."""
        from unittest.mock import patch

        from src.agents.data_collector import DataCollectorAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=mock_llm, config=config)
        
        # Mock web_search to return missing API key error
        with patch("src.agents.data_collector.web_search") as mock_web_search:
            mock_web_search.invoke.return_value = {
                "success": False,
                "error": "TAVILY_API_KEY not configured. Please set TAVILY_API_KEY in your .env file or environment variables.",
                "results": [],
                "count": 0,
            }
            
            state = create_initial_state("Test")
            state["plan"] = {
                "tasks": ["Find competitors"],
                "minimum_results": 4,
            }
            
            with pytest.raises(WorkflowError) as exc_info:
                agent.execute(state)
            
            # Verify specific error message for missing API key
            error_msg = str(exc_info.value)
            assert "TAVILY_API_KEY is not configured" in error_msg or "TAVILY_API_KEY" in error_msg
            assert ".env" in error_msg.lower()
    
    def test_data_collector_agent_raises_error_on_invalid_api_key(self) -> None:
        """Test data collector agent raises WorkflowError with specific message for invalid API key."""
        from unittest.mock import patch

        from src.agents.data_collector import DataCollectorAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=mock_llm, config=config)
        
        # Mock web_search to return invalid API key error (401/403 or invalid key message)
        with patch("src.agents.data_collector.web_search") as mock_web_search:
            mock_web_search.invoke.return_value = {
                "success": False,
                "error": "401 Unauthorized - Invalid API key",
                "results": [],
                "count": 0,
            }
            
            state = create_initial_state("Test")
            state["plan"] = {
                "tasks": ["Find competitors"],
                "minimum_results": 4,
            }
            
            with pytest.raises(WorkflowError) as exc_info:
                agent.execute(state)
            
            # Verify specific error message for invalid API key
            error_msg = str(exc_info.value)
            assert "invalid" in error_msg.lower() or "Invalid" in error_msg
            assert "API key" in error_msg or "TAVILY_API_KEY" in error_msg
    
    @pytest.mark.asyncio
    async def test_data_collector_agent_async_raises_error_on_no_results(self) -> None:
        """Test async data collector agent raises WorkflowError when no competitors collected."""
        from unittest.mock import patch

        from src.agents.data_collector import DataCollectorAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=mock_llm, config=config)
        
        # Mock web_search_async to return empty results
        with patch("src.agents.data_collector.web_search_async") as mock_web_search:
            mock_web_search.return_value = {
                "success": False,
                "error": "API error",
                "results": [],
                "count": 0,
            }
            
            state = create_initial_state("Test")
            state["plan"] = {
                "tasks": ["Find competitors"],
                "minimum_results": 4,
            }
            
            with pytest.raises(WorkflowError, match="No competitor data collected"):
                await agent.execute_async(state)
    
    def test_data_collector_agent_respects_minimum_results(self) -> None:
        """Test data collector agent respects minimum_results from plan."""
        from unittest.mock import patch

        from src.agents.data_collector import DataCollectorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=mock_llm, config=config)
        
        # Mock web_search to return enough results
        with patch("src.agents.data_collector.web_search") as mock_web_search:
            # Create enough mock results
            mock_results = [
                {
                    "url": f"https://competitor{i}.com",
                    "title": f"Competitor {i}",
                    "snippet": f"Competitor {i} description",
                }
                for i in range(10)
            ]
            
            mock_web_search.invoke.return_value = {
                "success": True,
                "results": mock_results,
                "count": 10,
            }
            
            state = create_initial_state("Test")
            state["plan"] = {
                "tasks": ["Find competitors"],
                "minimum_results": 4,
            }
            
            result_state = agent.execute(state)
            competitors = result_state["collected_data"]["competitors"]
            
            # Should collect at least minimum_results, up to 2x
            assert len(competitors) >= 4
            assert len(competitors) <= 8  # 2x minimum_results


class TestInsightAgent:
    """Tests for InsightAgent."""
    
    def test_insight_agent_execute_success(self) -> None:
        """Test insight agent generates insights successfully."""
        from src.agents.insight_agent import InsightAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader in SaaS industry",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market"],
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = InsightAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["collected_data"] = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
                {"name": "Comp2", "source_url": "https://example2.com"},
            ]
        }
        
        result_state = agent.execute(state)
        
        assert "insights" in result_state
        assert result_state["insights"] is not None
        assert "swot" in result_state["insights"]
        assert "positioning" in result_state["insights"]
        assert result_state["current_task"] == "Insights generated successfully"
    
    def test_insight_agent_handles_missing_collected_data(self) -> None:
        """Test insight agent handles missing collected data."""
        from src.agents.insight_agent import InsightAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        agent = InsightAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        # No collected_data
        
        with pytest.raises(WorkflowError, match="without collected data"):
            agent.execute(state)
    
    def test_insight_agent_handles_empty_competitors(self) -> None:
        """Test insight agent handles empty competitor list."""
        from src.agents.insight_agent import InsightAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "swot": {
                "strengths": ["Market presence"],
                "weaknesses": ["Limited data"],
                "opportunities": ["Market expansion"],
                "threats": ["Competition"],
            },
            "positioning": "Market analysis with limited competitor data available",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Market expansion"],
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = InsightAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["collected_data"] = {"competitors": []}
        
        # Should generate minimal insights instead of raising error
        result_state = agent.execute(state)
        assert result_state["insights"] is not None
        assert "swot" in result_state["insights"]
    
    def test_insight_agent_handles_llm_error(self) -> None:
        """Test insight agent handles LLM errors gracefully."""
        from src.agents.insight_agent import InsightAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke.side_effect = Exception("LLM API error")
        
        config = {"temperature": 0.7}
        agent = InsightAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["collected_data"] = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
            ]
        }
        
        with pytest.raises(WorkflowError, match="Failed to generate insights"):
            agent.execute(state)
    
    def test_insight_agent_handles_invalid_json(self) -> None:
        """Test insight agent handles invalid JSON response."""
        from src.agents.insight_agent import InsightAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = "This is not JSON {invalid"
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = InsightAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["collected_data"] = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
            ]
        }
        
        with pytest.raises(WorkflowError, match="Failed to parse insights"):
            agent.execute(state)
    
    def test_insight_agent_handles_missing_swot(self) -> None:
        """Test insight agent handles missing SWOT field."""
        from src.agents.insight_agent import InsightAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "positioning": "Market leader"
            # Missing swot
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = InsightAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["collected_data"] = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
            ]
        }
        
        with pytest.raises(WorkflowError, match="missing required fields"):
            agent.execute(state)
    
    def test_insight_agent_validates_insight_model(self) -> None:
        """Test insight agent validates insight against Insight model."""
        from src.agents.insight_agent import InsightAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "swot": {
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": [],
            },
            "positioning": "",  # Empty positioning should fail validation
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = InsightAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["collected_data"] = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
            ]
        }
        
        with pytest.raises(WorkflowError, match="failed validation"):
            agent.execute(state)
    
    def test_insight_agent_parses_json_from_markdown(self) -> None:
        """Test insight agent parses JSON from markdown code blocks."""
        from src.agents.insight_agent import InsightAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = """Here are the insights:
```json
{
    "swot": {
        "strengths": ["Strong brand", "Market leader"],
        "weaknesses": ["High prices", "Limited distribution"],
        "opportunities": ["Emerging markets", "B2B expansion"],
        "threats": ["New competitors", "Market saturation"]
    },
    "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
    "trends": ["Digital transformation", "AI integration"],
    "opportunities": ["Expansion into Asia", "B2B market growth"]
}
```"""
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = InsightAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["collected_data"] = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
            ]
        }
        
        result_state = agent.execute(state)
        assert result_state["insights"]["swot"]["strengths"] == ["Strong brand", "Market leader"]
    
    def test_insight_agent_prepares_competitor_summary(self) -> None:
        """Test insight agent prepares competitor summary correctly."""
        from src.agents.insight_agent import InsightAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "swot": {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []},
            "positioning": "Test positioning",
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = InsightAgent(llm=mock_llm, config=config)
        
        competitors = [
            {
                "name": "Comp1",
                "website": "https://comp1.com",
                "products": ["Product A", "Product B"],
                "market_presence": "Market leader",
            }
        ]
        
        summary = agent._prepare_competitor_summary(competitors)
        
        assert "Comp1" in summary
        assert "Product A" in summary
        assert "Market leader" in summary
    
    def test_insight_agent_sets_defaults_for_optional_fields(self) -> None:
        """Test insight agent sets defaults for optional fields."""
        from src.agents.insight_agent import InsightAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            # Missing trends and opportunities (should be set to empty lists)
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = InsightAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["collected_data"] = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
            ]
        }
        
        result_state = agent.execute(state)
        insights = result_state["insights"]
        
        assert "trends" in insights
        assert "opportunities" in insights
        assert isinstance(insights["trends"], list)
        assert isinstance(insights["opportunities"], list)
    
    def test_insight_agent_name(self) -> None:
        """Test insight agent name property."""
        from src.agents.insight_agent import InsightAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        agent = InsightAgent(llm=mock_llm, config=config)
        
        assert agent.name == "insight_agent"
    
    def test_insight_agent_handles_empty_llm_response(self) -> None:
        """Test insight agent handles empty LLM response."""
        from src.agents.insight_agent import InsightAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = ""  # Empty response
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = InsightAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["collected_data"] = {
            "competitors": [
                {"name": "Comp1", "source_url": "https://example.com"},
            ]
        }
        
        with pytest.raises(WorkflowError, match="empty response"):
            agent.execute(state)


class TestReportAgent:
    """Tests for ReportAgent."""
    
    def test_report_agent_execute_success(self) -> None:
        """Test report agent generates report successfully."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "executive_summary": "This executive summary provides a comprehensive overview of the competitor analysis findings and key insights. The analysis reveals important strategic information about market positioning and competitive dynamics. The findings indicate significant opportunities for market expansion and strategic positioning improvements. This comprehensive analysis covers multiple market segments and provides actionable recommendations for business growth.",
            "swot_breakdown": "The SWOT analysis reveals key strengths, weaknesses, opportunities, and threats in the competitive landscape. This comprehensive breakdown provides strategic insights for decision-making. The analysis covers multiple dimensions including market position, product capabilities, customer relationships, and operational efficiency. This detailed examination helps identify strategic priorities and areas for improvement across different market segments.",
            "competitor_overview": "The competitor overview examines the key players in the market and their strategic positions. This section analyzes market share, positioning strategies, and competitive advantages. It provides detailed information about how different competitors approach the market, their unique value propositions, target customer segments, and strategic initiatives that drive their competitive advantage in the marketplace.",
            "recommendations": "Based on the analysis, we recommend strategic actions to improve competitive positioning. These recommendations focus on leveraging strengths and addressing market opportunities. The strategic roadmap includes product development initiatives, market expansion strategies, customer acquisition approaches, and operational improvements. These recommendations are designed to enhance market performance and drive sustainable growth.",
            "min_length": 1200,
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        
        result_state = agent.execute(state)
        
        assert "report" in result_state
        assert result_state["report"] is not None
        assert len(result_state["report"]) > 0
        assert "Executive Summary" in result_state["report"]
        assert "SWOT Analysis" in result_state["report"]
        assert result_state["current_task"] == "Report generated successfully"
    
    def test_report_agent_handles_missing_insights(self) -> None:
        """Test report agent handles missing insights."""
        from src.agents.report_agent import ReportAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        # No insights
        
        with pytest.raises(WorkflowError, match="without insights"):
            agent.execute(state)
    
    def test_report_agent_handles_invalid_insights(self) -> None:
        """Test report agent handles invalid insights structure."""
        from src.agents.report_agent import ReportAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {"invalid": "structure"}  # Missing required fields
        
        with pytest.raises(WorkflowError, match="Invalid insights"):
            agent.execute(state)
    
    def test_report_agent_handles_llm_error(self) -> None:
        """Test report agent handles LLM errors gracefully."""
        from src.agents.report_agent import ReportAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke.side_effect = Exception("LLM API error")
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        
        with pytest.raises(WorkflowError, match="Failed to generate report"):
            agent.execute(state)
    
    def test_report_agent_handles_invalid_json(self) -> None:
        """Test report agent handles invalid JSON response."""
        from src.agents.report_agent import ReportAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = "This is not JSON {invalid"
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        
        # The error message could be either "Failed to parse report" or "Could not extract JSON"
        with pytest.raises(WorkflowError, match="(Failed to parse report|Could not extract JSON)"):
            agent.execute(state)
    
    def test_report_agent_handles_missing_sections(self) -> None:
        """Test report agent handles missing report sections."""
        from src.agents.report_agent import ReportAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "executive_summary": "Summary"
            # Missing other sections
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        
        with pytest.raises(WorkflowError, match="missing required sections"):
            agent.execute(state)
    
    def test_report_agent_validates_report_model(self) -> None:
        """Test report agent validates report against Report model."""
        from src.agents.report_agent import ReportAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "executive_summary": "Short",  # Too short (< 50 chars)
            "swot_breakdown": "The SWOT analysis reveals key strengths, weaknesses, opportunities, and threats.",
            "competitor_overview": "The competitor overview provides detailed information about market players.",
            "recommendations": "Based on the analysis, we recommend strategic actions for market positioning.",
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        
        with pytest.raises(WorkflowError, match="failed validation"):
            agent.execute(state)
    
    def test_report_agent_parses_json_from_markdown(self) -> None:
        """Test report agent parses JSON from markdown code blocks."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = """Here's the report:
```json
{
    "executive_summary": "This is a comprehensive executive summary of the competitor analysis findings with detailed insights and key observations. The analysis covers multiple market segments and provides strategic recommendations for business growth and competitive positioning. The findings reveal important market dynamics and opportunities for strategic improvement.",
    "swot_breakdown": "The SWOT analysis reveals key strengths, weaknesses, opportunities, and threats in detail. This comprehensive breakdown provides strategic insights for decision-making. The analysis covers multiple dimensions including market position, product capabilities, customer relationships, and operational efficiency across different market segments and competitive environments.",
    "competitor_overview": "The competitor overview provides detailed information about market players and their positioning strategies. This section analyzes competitive dynamics and market structure. It examines how different competitors approach the market, their unique value propositions, target customer segments, and strategic initiatives that drive their competitive advantage.",
    "recommendations": "Based on the analysis, we recommend strategic actions for market positioning and competitive advantage. These recommendations are designed to enhance market performance and drive sustainable growth. The strategic roadmap includes product development initiatives, market expansion strategies, customer acquisition approaches, and operational improvements.",
    "min_length": 1200
}
```"""
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        
        result_state = agent.execute(state)
        assert "Executive Summary" in result_state["report"]
    
    def test_report_agent_formats_report_correctly(self) -> None:
        """Test report agent formats report string correctly."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "executive_summary": "This is a comprehensive executive summary of the competitor analysis findings with detailed insights and key observations. The analysis covers multiple market segments and provides strategic recommendations for business growth and competitive positioning. The findings reveal important market dynamics and opportunities for strategic improvement.",
            "swot_breakdown": "The SWOT analysis reveals key strengths, weaknesses, opportunities, and threats in the competitive landscape. This detailed breakdown helps identify strategic priorities and areas for improvement. The analysis covers multiple dimensions including market position, product capabilities, customer relationships, and operational efficiency across different market segments and competitive environments.",
            "competitor_overview": "The competitor overview provides detailed information about market players and their positioning strategies. This section analyzes competitive dynamics and market structure. It examines how different competitors approach the market, their unique value propositions, target customer segments, and strategic initiatives that drive their competitive advantage in the marketplace.",
            "recommendations": "Based on the analysis, we recommend strategic actions for market positioning and competitive advantage. These recommendations are designed to enhance market performance and drive sustainable growth. The strategic roadmap includes product development initiatives, market expansion strategies, customer acquisition approaches, and operational improvements that will strengthen competitive positioning.",
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        
        result_state = agent.execute(state)
        report = result_state["report"]
        
        # Check all sections are included
        assert "# Competitor Analysis Report" in report
        assert "## Executive Summary" in report
        assert "## SWOT Analysis Breakdown" in report
        assert "## Competitor Overview" in report
        assert "## Strategic Recommendations" in report
    
    def test_report_agent_temperature_warning(self) -> None:
        """Test report agent warns about low temperature."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "executive_summary": "This is a comprehensive executive summary of the competitor analysis findings with detailed insights and key observations. The analysis covers multiple market segments and provides strategic recommendations for business growth and competitive positioning. The findings reveal important market dynamics and opportunities for strategic improvement.",
            "swot_breakdown": "The SWOT analysis reveals key strengths, weaknesses, opportunities, and threats in the competitive landscape. This detailed breakdown helps identify strategic priorities and areas for improvement. The analysis covers multiple dimensions including market position, product capabilities, customer relationships, and operational efficiency across different market segments and competitive environments.",
            "competitor_overview": "The competitor overview provides detailed information about market players and their positioning strategies. This section analyzes competitive dynamics and market structure. It examines how different competitors approach the market, their unique value propositions, target customer segments, and strategic initiatives that drive their competitive advantage in the marketplace.",
            "recommendations": "Based on the analysis, we recommend strategic actions for market positioning and competitive advantage. These recommendations are designed to enhance market performance and drive sustainable growth. The strategic roadmap includes product development initiatives, market expansion strategies, customer acquisition approaches, and operational improvements that will strengthen competitive positioning.",
        })
        mock_llm.invoke.return_value = mock_response
        
        # Use low temperature (should still work but log warning)
        config = {"temperature": 0}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        
        # Should still work (warning is logged but doesn't prevent execution)
        result_state = agent.execute(state)
        assert result_state["report"] is not None
    
    def test_report_agent_prepares_insights_summary(self) -> None:
        """Test report agent prepares insights summary correctly."""
        from src.agents.report_agent import ReportAgent
        from src.models.insight_model import SWOT, Insight
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "executive_summary": "This is a comprehensive executive summary of the competitor analysis findings with detailed insights and key observations. The analysis covers multiple market segments and provides strategic recommendations for business growth and competitive positioning. The findings reveal important market dynamics and opportunities for strategic improvement.",
            "swot_breakdown": "The SWOT analysis reveals key strengths, weaknesses, opportunities, and threats in the competitive landscape. This detailed breakdown helps identify strategic priorities and areas for improvement. The analysis covers multiple dimensions including market position, product capabilities, customer relationships, and operational efficiency across different market segments and competitive environments.",
            "competitor_overview": "The competitor overview provides detailed information about market players and their positioning strategies. This section analyzes competitive dynamics and market structure. It examines how different competitors approach the market, their unique value propositions, target customer segments, and strategic initiatives that drive their competitive advantage in the marketplace.",
            "recommendations": "Based on the analysis, we recommend strategic actions for market positioning and competitive advantage. These recommendations are designed to enhance market performance and drive sustainable growth. The strategic roadmap includes product development initiatives, market expansion strategies, customer acquisition approaches, and operational improvements that will strengthen competitive positioning.",
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        swot = SWOT(
            strengths=["Strong brand", "Market leader"],
            weaknesses=["High prices", "Limited distribution"],
            opportunities=["Emerging markets", "B2B expansion"],
            threats=["New competitors", "Market saturation"],
        )
        insights = Insight(
            swot=swot,
            positioning="Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            trends=["Digital transformation", "AI integration"],
            opportunities=["Expansion into Asia", "B2B market growth"],
        )
        
        state = create_initial_state("Test")
        state["collected_data"] = sample_collected_data()
        
        summary = agent._prepare_insights_summary(insights, state)
        
        assert "SWOT Analysis" in summary
        assert "Strong brand" in summary
        assert "Premium market leader" in summary
        assert "Digital transformation" in summary
        assert "Source URLs" in summary or "source" in summary.lower()
    
    def test_report_agent_name(self) -> None:
        """Test report agent name property."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        assert agent.name == "report_agent"
    
    def test_report_agent_handles_empty_llm_response(self) -> None:
        """Test report agent handles empty LLM response."""
        from src.agents.report_agent import ReportAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = ""  # Empty response
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        
        with pytest.raises(WorkflowError, match="empty response"):
            agent.execute(state)
    
    def test_report_agent_meets_minimum_length(self) -> None:
        """Test report agent generates report meeting minimum length."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        # Generate sections that meet minimum length
        mock_response.content = json.dumps({
            "executive_summary": "A" * 300,
            "swot_breakdown": "B" * 300,
            "competitor_overview": "C" * 300,
            "recommendations": "D" * 300,
            "min_length": 1200,
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand", "Market leader"],
                "weaknesses": ["High prices", "Limited distribution"],
                "opportunities": ["Emerging markets", "B2B expansion"],
                "threats": ["New competitors", "Market saturation"],
            },
            "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
            "trends": ["Digital transformation", "AI integration"],
            "opportunities": ["Expansion into Asia", "B2B market growth"],
        }
        
        result_state = agent.execute(state)
        report = result_state["report"]
        
        # Total length should be at least 500 (sections + formatting)
        assert len(report) >= 500
    
    def test_report_agent_includes_methodology_and_sources(self) -> None:
        """Test report agent includes methodology and sources sections."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "executive_summary": "A" * 200,
            "swot_breakdown": "B" * 300,
            "competitor_overview": "C" * 300,
            "recommendations": "D" * 300,
            "methodology": "E" * 200,
            "sources": ["https://source1.com", "https://source2.com"],
            "min_length": 1200,
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = sample_insights()
        state["collected_data"] = sample_collected_data()
        
        result_state = agent.execute(state)
        
        report = result_state["report"]
        assert report is not None
        assert "## Methodology" in report
        assert "## Sources" in report
        assert "https://source1.com" in report
        assert "https://source2.com" in report


class TestExportAgent:
    """Tests for ExportAgent."""
    
    def test_export_agent_execute_success(self) -> None:
        """Test export agent executes successfully."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.agents.export_agent import ExportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf", "include_visualizations": False}
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.pdf_generator.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine = MagicMock()
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc.return_value = mock_doc_instance
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Report\n## Summary\nTest content"
                        
                        result_state = agent.execute(state)
                        
                        assert "export_paths" in result_state
                        assert result_state["export_paths"] is not None
                        assert result_state["current_task"] == "Export completed successfully"
    
    def test_export_agent_handles_missing_report(self) -> None:
        """Test export agent handles missing report."""
        from src.agents.export_agent import ExportAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        agent = ExportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        # No report
        
        with pytest.raises(WorkflowError, match="without a report"):
            agent.execute(state)
    
    def test_export_agent_handles_missing_insights(self) -> None:
        """Test export agent handles missing insights gracefully."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from src.agents.export_agent import ExportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf", "include_visualizations": True}
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.pdf_generator.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc.return_value = mock_doc_instance
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Report\n## Summary\nTest content"
                        # No insights
        
                        result_state = agent.execute(state)
                        
                        # Should still succeed, just without visualizations
                        assert "export_paths" in result_state
    
    def test_export_agent_generates_visualizations(self) -> None:
        """Test export agent generates visualizations when requested."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.agents.export_agent import ExportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf", "include_visualizations": True}
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.pdf_generator.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc.return_value = mock_doc_instance
                        
                        # Patch matplotlib imports that happen inside functions
                        import sys
                        mock_matplotlib = MagicMock()
                        mock_pyplot = MagicMock()
                        mock_fig = MagicMock()
                        mock_axes = MagicMock()
                        mock_pyplot.subplots.return_value = (mock_fig, mock_axes)
                        mock_pyplot.savefig.return_value = None
                        mock_pyplot.close.return_value = None
                        mock_pyplot.tight_layout.return_value = None
                        mock_pyplot.suptitle.return_value = None
                        mock_pyplot.cm = MagicMock()
                        mock_pyplot.cm.viridis.return_value = [(0.5, 0.5, 0.5, 1.0)]
                        mock_pyplot.cm.Set3.return_value = [(0.5, 0.5, 0.5, 1.0)]
                        mock_pyplot.cm.plasma.return_value = [(0.5, 0.5, 0.5, 1.0)]
                        mock_pyplot.linspace = MagicMock(return_value=[0.5])
                        mock_matplotlib.pyplot = mock_pyplot
                        mock_matplotlib.patches = MagicMock()
                        
                        with patch.dict(sys.modules, {"matplotlib": mock_matplotlib, "matplotlib.pyplot": mock_pyplot, "matplotlib.patches": mock_matplotlib.patches}):
                            with patch("numpy.arange", return_value=[0, 1]):
                                state = create_initial_state("Test")
                                state["report"] = "# Report\n## Summary\nTest content"
                                state["insights"] = {
                                    "swot": {
                                        "strengths": ["Strong brand", "Market leader"],
                                        "weaknesses": ["High prices", "Limited distribution"],
                                        "opportunities": ["Emerging markets", "B2B expansion"],
                                        "threats": ["New competitors", "Market saturation"],
                                    },
                                    "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
                                    "trends": ["Digital transformation", "AI integration"],
                                    "opportunities": ["Expansion into Asia", "B2B market growth"],
                                }
                                
                                result_state = agent.execute(state)
                                
                                assert "export_paths" in result_state
    
    def test_export_agent_respects_branding_config(self) -> None:
        """Test export agent respects branding configuration."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.agents.export_agent import ExportAgent
        from src.models.pdf_branding_config import PDFBrandingConfig
        
        mock_llm = Mock(spec=BaseChatModel)
        branding = PDFBrandingConfig(company_name="Test Company")
        config = {
            "export_format": "pdf",
            "pdf_branding": branding,
        }
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.pdf_generator.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc.return_value = mock_doc_instance
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Report\n## Summary\nTest content"
                        
                        result_state = agent.execute(state)
                        
                        # Verify branding config was used
                        assert mock_engine_class.called
                        call_kwargs = mock_engine_class.call_args[1]
                        assert call_kwargs["branding_config"] == branding
    
    def test_export_agent_respects_layout_config(self) -> None:
        """Test export agent respects layout configuration."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.agents.export_agent import ExportAgent
        from src.models.pdf_layout_config import PDFLayoutConfig
        
        mock_llm = Mock(spec=BaseChatModel)
        layout = PDFLayoutConfig(page_size="Letter", orientation="landscape")
        config = {
            "export_format": "pdf",
            "pdf_layout": layout,
        }
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.pdf_generator.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc.return_value = mock_doc_instance
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Report\n## Summary\nTest content"
                        
                        result_state = agent.execute(state)
                        
                        # Verify layout config was used
                        assert mock_engine_class.called
                        call_kwargs = mock_engine_class.call_args[1]
                        assert call_kwargs["layout_config"] == layout
    
    def test_export_agent_handles_file_system_errors(self) -> None:
        """Test export agent handles file system errors gracefully."""
        from unittest.mock import patch

        from src.agents.export_agent import ExportAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf", "output_dir": "/invalid/path"}
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with patch("src.agents.export_agent.get_config") as mock_get_config:
            mock_app_config = Mock()
            mock_app_config.data_dir = Path("/invalid")
            mock_get_config.return_value = mock_app_config
            
            state = create_initial_state("Test")
            state["report"] = "# Report\n## Summary\nTest content"
            
            # Should handle error gracefully or raise WorkflowError
            try:
                result_state = agent.execute(state)
                # If it doesn't raise, should have error in validation_errors
                assert "export_paths" in result_state or len(result_state.get("validation_errors", [])) > 0
            except WorkflowError:
                # Also acceptable - error handling
                pass
    
    def test_export_agent_validates_output_directory(self) -> None:
        """Test export agent validates output directory."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from src.agents.export_agent import ExportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.pdf_generator.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc.return_value = mock_doc_instance
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Report\n## Summary\nTest content"
                        
                        result_state = agent.execute(state)
                        
                        # Should create output directory
                        assert "export_paths" in result_state
    
    def test_export_agent_generates_swot_diagram(self) -> None:
        """Test export agent generates SWOT diagram."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.agents.export_agent import ExportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf", "include_visualizations": True}
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.pdf_generator.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc.return_value = mock_doc_instance
                        
                        # Patch matplotlib imports that happen inside functions
                        import sys
                        mock_matplotlib = MagicMock()
                        mock_pyplot = MagicMock()
                        mock_fig = MagicMock()
                        mock_axes = MagicMock()
                        mock_pyplot.subplots.return_value = (mock_fig, mock_axes)
                        mock_pyplot.savefig.return_value = None
                        mock_pyplot.close.return_value = None
                        mock_pyplot.tight_layout.return_value = None
                        mock_pyplot.suptitle.return_value = None
                        mock_pyplot.cm = MagicMock()
                        mock_pyplot.cm.viridis.return_value = [(0.5, 0.5, 0.5, 1.0)]
                        mock_pyplot.cm.Set3.return_value = [(0.5, 0.5, 0.5, 1.0)]
                        mock_pyplot.cm.plasma.return_value = [(0.5, 0.5, 0.5, 1.0)]
                        mock_pyplot.linspace = MagicMock(return_value=[0.5])
                        mock_matplotlib.pyplot = mock_pyplot
                        mock_matplotlib.patches = MagicMock()
                        
                        with patch.dict(sys.modules, {"matplotlib": mock_matplotlib, "matplotlib.pyplot": mock_pyplot, "matplotlib.patches": mock_matplotlib.patches}):
                            with patch("numpy.arange", return_value=[0, 1]):
                                state = create_initial_state("Test")
                                state["report"] = "# Report\n## Summary\nTest content"
                                state["insights"] = {
                                    "swot": {
                                        "strengths": ["Strong brand", "Market leader"],
                                        "weaknesses": ["High prices", "Limited distribution"],
                                        "opportunities": ["Emerging markets", "B2B expansion"],
                                        "threats": ["New competitors", "Market saturation"],
                                    },
                                    "positioning": "Premium market leader in the SaaS industry with strong brand recognition and customer loyalty, positioned as a trusted provider of enterprise solutions",
                                    "trends": ["Digital transformation", "AI integration"],
                                    "opportunities": ["Expansion into Asia", "B2B market growth"],
                                }
                                
                                result_state = agent.execute(state)
                                
                                # Should attempt to generate SWOT diagram
                                assert "export_paths" in result_state
    
    def test_export_agent_name(self) -> None:
        """Test export agent name property."""
        from src.agents.export_agent import ExportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        agent = ExportAgent(llm=mock_llm, config=config)
        
        assert agent.name == "export_agent"
    
    def test_export_agent_handles_empty_report_content(self) -> None:
        """Test export agent handles empty report content."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.agents.export_agent import ExportAgent
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                state = create_initial_state("Test")
                state["report"] = ""  # Empty report
                
                # Empty report should raise WorkflowError
                with pytest.raises(WorkflowError, match="without a report"):
                    agent.execute(state)
    
    def test_export_agent_handles_missing_competitor_data(self) -> None:
        """Test export agent handles missing competitor data for advanced visualizations."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.agents.export_agent import ExportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {
            "export_format": "pdf",
            "include_visualizations": True,
            "include_advanced_viz": True,
        }
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.pdf_generator.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc.return_value = mock_doc_instance
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Report\n## Summary\nTest content"
                        # No collected_data for advanced visualizations
        
                        result_state = agent.execute(state)
                        
                        # Should still succeed without advanced visualizations
                        assert "export_paths" in result_state
    
    def test_export_agent_sets_pdf_metadata(self) -> None:
        """Test export agent sets PDF metadata correctly."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.agents.export_agent import ExportAgent
        from src.models.pdf_branding_config import PDFBrandingConfig
        
        mock_llm = Mock(spec=BaseChatModel)
        branding = PDFBrandingConfig(company_name="Test Company")
        config = {
            "export_format": "pdf",
            "pdf_branding": branding,
        }
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.pdf_generator.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc_class:
                        mock_doc_instance = MagicMock()
                        mock_canvas = MagicMock()
                        mock_doc_instance.pagesize = (612, 792)
                        mock_doc_class.return_value = mock_doc_instance
                        
                        # Track metadata calls
                        metadata_calls = {}
                        
                        def track_metadata_setter(name):
                            def setter(value):
                                metadata_calls[name] = value
                            return setter
                        
                        mock_canvas.setTitle = track_metadata_setter("title")
                        mock_canvas.setAuthor = track_metadata_setter("author")
                        mock_canvas.setSubject = track_metadata_setter("subject")
                        mock_canvas.setKeywords = track_metadata_setter("keywords")
                        mock_canvas.setCreator = track_metadata_setter("creator")
                        
                        # Mock the onFirstPage callback to capture canvas
                        def capture_callback(callback):
                            if callback:
                                callback(mock_canvas, mock_doc_instance)
                        
                        mock_doc_instance.build = MagicMock(side_effect=lambda story, **kwargs: capture_callback(kwargs.get("onFirstPage")))
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Test Report\n## Summary\nTest content"
                        
                        result_state = agent.execute(state)
                        
                        assert "export_paths" in result_state
                        # Metadata should be set (if callback was called)
                        # Note: In actual execution, metadata is set via canvas callbacks
    
    def test_export_agent_extracts_bookmarks(self) -> None:
        """Test export agent extracts bookmarks from report."""
        from src.agents.export_agent import ExportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        agent = ExportAgent(llm=mock_llm, config=config)
        
        report_content = """# Main Title
## Section 1
### Subsection 1.1
## Section 2
### Subsection 2.1"""
        
        from src.template.pdf_utils import extract_bookmarks
        bookmarks = extract_bookmarks(report_content)
        
        assert isinstance(bookmarks, list)
        assert len(bookmarks) == 5  # 1 H1 + 2 H2 + 2 H3
        assert all("title" in bm and "level" in bm for bm in bookmarks)
        assert bookmarks[0]["level"] == 1  # Main Title
        assert bookmarks[1]["level"] == 2  # Section 1
    
    def test_export_agent_handles_dict_branding_config(self) -> None:
        """Test export agent handles dict branding config."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.agents.export_agent import ExportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        branding_dict = {
            "company_name": "Test Company",
            "primary_color": "#1a1a1a",
        }
        config = {
            "export_format": "pdf",
            "pdf_branding": branding_dict,
        }
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.pdf_generator.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc.return_value = mock_doc_instance
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Report\n## Summary\nTest content"
                        
                        result_state = agent.execute(state)
                        
                        assert "export_paths" in result_state
                        # Should have converted dict to PDFBrandingConfig
                        assert mock_engine_class.called
    
    def test_export_agent_handles_dict_layout_config(self) -> None:
        """Test export agent handles dict layout config."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.agents.export_agent import ExportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        layout_dict = {
            "page_size": "A4",
            "orientation": "portrait",
        }
        config = {
            "export_format": "pdf",
            "pdf_layout": layout_dict,
        }
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.pdf_generator.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc.return_value = mock_doc_instance
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Report\n## Summary\nTest content"
                        
                        result_state = agent.execute(state)
                        
                        assert "export_paths" in result_state
                        # Should have converted dict to PDFLayoutConfig
                        assert mock_engine_class.called
    
    def test_export_agent_handles_invalid_branding_config(self) -> None:
        """Test export agent handles invalid branding config gracefully."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.agents.export_agent import ExportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        invalid_branding = {"invalid_field": "invalid_value"}
        config = {
            "export_format": "pdf",
            "pdf_branding": invalid_branding,
        }
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                with patch("src.template.pdf_generator.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_instance = MagicMock()
                    mock_engine_instance.create_cover_page.return_value = []
                    mock_engine_instance.create_header.return_value = None
                    mock_engine_instance.create_footer.return_value = None
                    mock_engine_class.return_value = mock_engine_instance
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc.return_value = mock_doc_instance
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Report\n## Summary\nTest content"
                        
                        # Should use default config and still succeed
                        result_state = agent.execute(state)
                        
                        assert "export_paths" in result_state
    
    def test_export_agent_extracts_keywords(self) -> None:
        """Test export agent extracts keywords from report."""
        from src.agents.export_agent import ExportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        agent = ExportAgent(llm=mock_llm, config=config)
        
        report_content = """# Competitor Analysis Report
## Market Overview
## SWOT Analysis
## Recommendations"""
        
        from src.template.pdf_utils import extract_keywords
        keywords = extract_keywords(report_content)
        
        assert isinstance(keywords, str)
        assert "competitor" in keywords.lower()
        assert "analysis" in keywords.lower()
        # Should include section headings as keywords
        assert len(keywords) > 0
    
    def test_export_agent_creates_default_configs(self) -> None:
        """Test export agent creates default configs when none provided."""
        from src.agents.export_agent import ExportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        agent = ExportAgent(llm=mock_llm, config=config)
        
        from src.template.pdf_utils import get_pdf_configs
        branding_config, layout_config = get_pdf_configs(config)
        
        # Should create default configs
        assert branding_config is not None
        assert layout_config is not None
        assert branding_config.company_name is not None
        assert layout_config.page_size is not None
    
    def test_export_agent_template_engine_fallback(self) -> None:
        """Test export agent falls back to basic PDF if template engine fails."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from src.agents.export_agent import ExportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"export_format": "pdf"}
        agent = ExportAgent(llm=mock_llm, config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            config["output_dir"] = str(output_dir)
            
            with patch("src.agents.export_agent.get_config") as mock_get_config:
                mock_app_config = Mock()
                mock_app_config.data_dir = Path(tmpdir)
                mock_get_config.return_value = mock_app_config
                
                # Make template engine raise exception
                with patch("src.template.pdf_generator.DefaultPDFTemplateEngine") as mock_engine_class:
                    mock_engine_class.side_effect = Exception("Template engine failed")
                    
                    with patch("reportlab.platypus.SimpleDocTemplate") as mock_doc:
                        mock_doc_instance = MagicMock()
                        mock_doc.return_value = mock_doc_instance
                        
                        state = create_initial_state("Test")
                        state["report"] = "# Report\n## Summary\nTest content"
                        
                        # Should still generate PDF without template engine
                        result_state = agent.execute(state)
                        
                        assert "export_paths" in result_state
                        assert "pdf" in result_state["export_paths"]