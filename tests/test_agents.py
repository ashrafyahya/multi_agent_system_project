"""Tests for agent implementations.

This module contains unit tests for all agent classes to verify
agent behavior, dependency injection, and interface compliance.
"""

import json
import pytest
from unittest.mock import Mock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agents.base_agent import BaseAgent
from src.graph.state import WorkflowState, create_initial_state


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
        
        with pytest.raises(WorkflowError, match="validation failed"):
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
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": ["Digital transformation"],
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
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "",  # Empty positioning
            "trends": ["Digital transformation"],
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
        
        # Valid report (long enough)
        state = create_initial_state("Test")
        state["plan"] = {"tasks": ["Collect data"]}
        state["collected_data"] = {"competitors": []}
        state["insights"] = {"swot": {}}
        state["report"] = "A" * 500  # Minimum length
        
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
        
        # Test insight stage
        state["collected_data"] = {"competitors": []}
        result_state = agent.execute(state)
        assert "insight generation" in result_state["current_task"].lower()
        
        # Test report stage
        state["insights"] = {"swot": {}}
        result_state = agent.execute(state)
        assert "report generation" in result_state["current_task"].lower()
        
        # Test complete stage
        state["report"] = "A" * 500
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
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Short",  # Short but valid
            "trends": ["Digital transformation"],
        }
        
        result_state = agent.execute(state)
        
        # Should pass but may have warnings
        assert result_state["current_task"] == "Proceeding to report generation"


class TestDataCollectorAgent:
    """Tests for DataCollectorAgent."""
    
    def test_data_collector_agent_execute_success(self) -> None:
        """Test data collector agent collects data successfully."""
        from src.agents.data_collector import DataCollectorAgent
        from unittest.mock import patch
        
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
        """Test data collector agent handles web search failures gracefully."""
        from src.agents.data_collector import DataCollectorAgent
        from unittest.mock import patch
        
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
            
            # Should not raise exception, but return empty or partial results
            result_state = agent.execute(state)
            assert "collected_data" in result_state
            # May have 0 competitors if all searches fail
    
    def test_data_collector_agent_extracts_competitor_name(self) -> None:
        """Test data collector agent extracts competitor names correctly."""
        from src.agents.data_collector import DataCollectorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=mock_llm, config=config)
        
        # Test name extraction from title
        name = agent._extract_competitor_name(
            "Competitor Inc - Official Website",
            "https://competitor.com",
            ""
        )
        assert name is not None
        assert "Competitor" in name
        
        # Test name extraction from URL
        name = agent._extract_competitor_name(
            "",
            "https://www.example.com/page",
            ""
        )
        assert name is not None
        
        # Test name extraction from snippet
        name = agent._extract_competitor_name(
            "",
            "https://example.com",
            "Competitor XYZ is a leading company"
        )
        assert name is not None
    
    def test_data_collector_agent_deduplicates_competitors(self) -> None:
        """Test data collector agent deduplicates competitors."""
        from src.agents.data_collector import DataCollectorAgent
        from unittest.mock import patch
        
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
        from src.agents.data_collector import DataCollectorAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=mock_llm, config=config)
        
        snippet = "Our products include Product A, Product B, and Product C"
        title = "Company Products"
        
        products = agent._extract_products(snippet, title)
        
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
        """Test data collector agent handles empty search results."""
        from src.agents.data_collector import DataCollectorAgent
        from unittest.mock import patch
        
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
            
            result_state = agent.execute(state)
            assert "collected_data" in result_state
            assert len(result_state["collected_data"]["competitors"]) == 0
    
    def test_data_collector_agent_respects_minimum_results(self) -> None:
        """Test data collector agent respects minimum_results from plan."""
        from src.agents.data_collector import DataCollectorAgent
        from unittest.mock import patch
        
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
        from src.exceptions.workflow_error import WorkflowError
        
        mock_llm = Mock(spec=BaseChatModel)
        config = {"temperature": 0.7}
        agent = InsightAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["collected_data"] = {"competitors": []}
        
        with pytest.raises(WorkflowError, match="No competitor data"):
            agent.execute(state)
    
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
        
        with pytest.raises(WorkflowError, match="validation failed"):
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
        "strengths": ["Strong brand"],
        "weaknesses": ["High prices"],
        "opportunities": ["Emerging markets"],
        "threats": ["New competitors"]
    },
    "positioning": "Premium market leader",
    "trends": ["Digital transformation"],
    "opportunities": ["Expansion"]
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
        assert result_state["insights"]["swot"]["strengths"] == ["Strong brand"]
    
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
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
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
            "executive_summary": "This executive summary provides a comprehensive overview of the competitor analysis findings and key insights.",
            "swot_breakdown": "The SWOT analysis reveals key strengths, weaknesses, opportunities, and threats in the competitive landscape.",
            "competitor_overview": "The competitor overview examines the key players in the market and their strategic positions.",
            "recommendations": "Based on the analysis, we recommend strategic actions to improve competitive positioning.",
            "min_length": 500,
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
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
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": [],
            "opportunities": [],
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
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": [],
            "opportunities": [],
        }
        
        with pytest.raises(WorkflowError, match="Failed to parse report"):
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
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": [],
            "opportunities": [],
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
            "swot_breakdown": "SWOT breakdown details",
            "competitor_overview": "Competitor overview details",
            "recommendations": "Recommendations details",
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": [],
            "opportunities": [],
        }
        
        with pytest.raises(WorkflowError, match="validation failed"):
            agent.execute(state)
    
    def test_report_agent_parses_json_from_markdown(self) -> None:
        """Test report agent parses JSON from markdown code blocks."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = """Here's the report:
```json
{
    "executive_summary": "Executive summary text with sufficient length",
    "swot_breakdown": "SWOT breakdown details",
    "competitor_overview": "Competitor overview details",
    "recommendations": "Recommendations details"
}
```"""
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": [],
            "opportunities": [],
        }
        
        result_state = agent.execute(state)
        assert "Executive Summary" in result_state["report"]
    
    def test_report_agent_formats_report_correctly(self) -> None:
        """Test report agent formats report string correctly."""
        from src.agents.report_agent import ReportAgent
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "executive_summary": "Executive summary text with sufficient length",
            "swot_breakdown": "SWOT breakdown details",
            "competitor_overview": "Competitor overview details",
            "recommendations": "Recommendations details",
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": [],
            "opportunities": [],
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
            "executive_summary": "Executive summary text with sufficient length",
            "swot_breakdown": "SWOT breakdown details",
            "competitor_overview": "Competitor overview details",
            "recommendations": "Recommendations details",
        })
        mock_llm.invoke.return_value = mock_response
        
        # Use low temperature (should still work but log warning)
        config = {"temperature": 0}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": [],
            "opportunities": [],
        }
        
        # Should still work (warning is logged but doesn't prevent execution)
        result_state = agent.execute(state)
        assert result_state["report"] is not None
    
    def test_report_agent_prepares_insights_summary(self) -> None:
        """Test report agent prepares insights summary correctly."""
        from src.agents.report_agent import ReportAgent
        from src.models.insight_model import Insight, SWOT
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_response = Mock()
        mock_response.content = json.dumps({
            "executive_summary": "Executive summary text with sufficient length",
            "swot_breakdown": "SWOT breakdown details",
            "competitor_overview": "Competitor overview details",
            "recommendations": "Recommendations details",
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        swot = SWOT(
            strengths=["Strong brand", "Market leader"],
            weaknesses=["High prices"],
            opportunities=["Emerging markets"],
            threats=["New competitors"],
        )
        insights = Insight(
            swot=swot,
            positioning="Premium market leader",
            trends=["Digital transformation"],
            opportunities=["Expansion"],
        )
        
        summary = agent._prepare_insights_summary(insights)
        
        assert "SWOT Analysis" in summary
        assert "Strong brand" in summary
        assert "Premium market leader" in summary
        assert "Digital transformation" in summary
    
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
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": [],
            "opportunities": [],
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
            "executive_summary": "A" * 100,
            "swot_breakdown": "B" * 100,
            "competitor_overview": "C" * 100,
            "recommendations": "D" * 200,
            "min_length": 500,
        })
        mock_llm.invoke.return_value = mock_response
        
        config = {"temperature": 0.7}
        agent = ReportAgent(llm=mock_llm, config=config)
        
        state = create_initial_state("Test")
        state["insights"] = {
            "swot": {
                "strengths": ["Strong brand"],
                "weaknesses": ["High prices"],
                "opportunities": ["Emerging markets"],
                "threats": ["New competitors"],
            },
            "positioning": "Premium market leader",
            "trends": [],
            "opportunities": [],
        }
        
        result_state = agent.execute(state)
        report = result_state["report"]
        
        # Total length should be at least 500 (sections + formatting)
        assert len(report) >= 500
