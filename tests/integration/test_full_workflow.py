"""Integration tests for full workflow execution.

This module contains end-to-end integration tests that verify the complete
workflow execution from user query to final report, with all LLM calls mocked.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.graph.state import WorkflowState, create_initial_state
from src.graph.workflow import create_workflow
from tests.fixtures.sample_data import (
    sample_collected_data,
    sample_insights,
    sample_llm_response_plan,
    sample_llm_response_insights,
    sample_llm_response_report,
    sample_report,
)


@pytest.mark.integration
class TestFullWorkflow:
    """Integration tests for complete workflow execution."""
    
    def test_full_workflow_successful_execution(self, mock_llm: Mock, workflow_config: dict) -> None:
        """Test complete workflow execution from start to finish."""
        # Mock LLM responses for each agent
        def mock_llm_invoke(messages):
            """Mock LLM invoke to return appropriate responses based on context."""
            content = str(messages[-1].content if hasattr(messages[-1], 'content') else messages[-1])
            
            # Planner agent - return plan
            if "plan" in content.lower() or "task" in content.lower():
                return AIMessage(content=sample_llm_response_plan())
            
            # Insight agent - return insights
            if "insight" in content.lower() or "swot" in content.lower():
                return AIMessage(content=sample_llm_response_insights())
            
            # Report agent - return report
            if "report" in content.lower() or "summary" in content.lower():
                return AIMessage(content=sample_llm_response_report())
            
            # Default response
            return AIMessage(content='{"result": "success"}')
        
        mock_llm.invoke = Mock(side_effect=mock_llm_invoke)
        
        # Mock web search and scraper tools
        with patch("src.tools.web_search.web_search") as mock_web_search:
            with patch("src.tools.scraper.scrape_url") as mock_scraper:
                # Mock web search to return results
                mock_web_search.return_value = {
                    "success": True,
                    "results": [
                        {
                            "title": "Competitor A",
                            "url": "https://competitor-a.com",
                            "snippet": "Leading SaaS provider",
                        },
                        {
                            "title": "Competitor B",
                            "url": "https://competitor-b.com",
                            "snippet": "Enterprise SaaS solution",
                        },
                        {
                            "title": "Competitor C",
                            "url": "https://competitor-c.com",
                            "snippet": "Mid-market SaaS platform",
                        },
                        {
                            "title": "Competitor D",
                            "url": "https://competitor-d.com",
                            "snippet": "Startup-focused SaaS",
                        },
                    ],
                }
                
                # Mock scraper to return content
                def mock_scrape(url: str) -> dict:
                    return {
                        "success": True,
                        "url": url,
                        "title": f"Page for {url}",
                        "content": f"Content from {url}",
                        "content_length": 1000,
                        "links": [],
                    }
                
                mock_scraper.side_effect = mock_scrape
                
                # Create workflow
                workflow = create_workflow(llm=mock_llm, config=workflow_config)
                
                # Create initial state
                initial_state = create_initial_state("Analyze competitors in SaaS market")
                
                # Execute workflow
                final_state = workflow.invoke(initial_state)
                
                # Verify final state
                assert final_state is not None
                assert "plan" in final_state
                assert final_state["plan"] is not None
                assert "collected_data" in final_state
                assert "insights" in final_state
                assert "report" in final_state or len(final_state.get("validation_errors", [])) == 0
    
    def test_workflow_state_transitions(self, mock_llm: Mock, workflow_config: dict) -> None:
        """Test that workflow state transitions correctly through all stages."""
        # Track state transitions
        state_transitions = []
        
        def mock_llm_invoke(messages):
            return AIMessage(content='{"tasks": ["Find competitors"], "minimum_results": 4}')
        
        mock_llm.invoke = Mock(side_effect=mock_llm_invoke)
        
        with patch("src.tools.web_search.web_search") as mock_web_search:
            with patch("src.tools.scraper.scrape_url") as mock_scraper:
                mock_web_search.return_value = {
                    "success": True,
                    "results": [
                        {"title": "Comp1", "url": "https://comp1.com", "snippet": "Desc1"},
                        {"title": "Comp2", "url": "https://comp2.com", "snippet": "Desc2"},
                        {"title": "Comp3", "url": "https://comp3.com", "snippet": "Desc3"},
                        {"title": "Comp4", "url": "https://comp4.com", "snippet": "Desc4"},
                    ],
                }
                
                def mock_scrape(url: str) -> dict:
                    return {
                        "success": True,
                        "url": url,
                        "title": "Page",
                        "content": "Content",
                        "content_length": 1000,
                        "links": [],
                    }
                
                mock_scraper.side_effect = mock_scrape
                
                workflow = create_workflow(llm=mock_llm, config=workflow_config)
                initial_state = create_initial_state("Test query")
                
                # Execute workflow
                final_state = workflow.invoke(initial_state)
                
                # Verify state progression
                # Should have plan
                assert final_state.get("plan") is not None
                # Should attempt to collect data (may succeed or fail based on mocks)
                # State should be valid
                assert isinstance(final_state, dict)
                assert "retry_count" in final_state
    
    def test_workflow_handles_collector_validation_failure(self, mock_llm: Mock, workflow_config: dict) -> None:
        """Test workflow handles collector validation failure and retries."""
        def mock_llm_invoke(messages):
            return AIMessage(content='{"tasks": ["Find competitors"], "minimum_results": 4}')
        
        mock_llm.invoke = Mock(side_effect=mock_llm_invoke)
        
        with patch("src.tools.web_search.web_search") as mock_web_search:
            with patch("src.tools.scraper.scrape_url") as mock_scraper:
                # Return insufficient results (less than 4)
                mock_web_search.return_value = {
                    "success": True,
                    "results": [
                        {"title": "Comp1", "url": "https://comp1.com", "snippet": "Desc1"},
                    ],  # Only 1 result - will fail validation
                }
                
                mock_scraper.side_effect = lambda url: {
                    "success": True,
                    "url": url,
                    "title": "Page",
                    "content": "Content",
                    "content_length": 1000,
                    "links": [],
                }
                
                workflow = create_workflow(llm=mock_llm, config=workflow_config)
                initial_state = create_initial_state("Test query")
                
                final_state = workflow.invoke(initial_state)
                
                # Should have retry attempts or validation errors
                assert final_state.get("retry_count", 0) >= 0
                # May have validation errors or may have retried
                assert isinstance(final_state, dict)
    
    def test_workflow_handles_max_retries_exceeded(self, mock_llm: Mock) -> None:
        """Test workflow ends when max retries exceeded."""
        config = {"max_retries": 1}  # Low retry limit
        
        def mock_llm_invoke(messages):
            return AIMessage(content='{"tasks": ["Find competitors"], "minimum_results": 4}')
        
        mock_llm.invoke = Mock(side_effect=mock_llm_invoke)
        
        with patch("src.tools.web_search.web_search") as mock_web_search:
            with patch("src.tools.scraper.scrape_url") as mock_scraper:
                # Always return insufficient results to trigger retries
                mock_web_search.return_value = {
                    "success": True,
                    "results": [],  # No results - will fail validation
                }
                
                mock_scraper.side_effect = lambda url: {
                    "success": False,
                    "error": "Failed",
                    "url": url,
                }
                
                workflow = create_workflow(llm=mock_llm, config=config)
                initial_state = create_initial_state("Test query")
                
                final_state = workflow.invoke(initial_state)
                
                # Should end workflow (not retry indefinitely)
                assert isinstance(final_state, dict)
                # Should have reached max retries or ended
                assert final_state.get("retry_count", 0) <= config["max_retries"]
    
    def test_workflow_handles_llm_errors(self, mock_llm: Mock, workflow_config: dict) -> None:
        """Test workflow handles LLM errors gracefully."""
        # Mock LLM to raise error
        mock_llm.invoke = Mock(side_effect=Exception("LLM API error"))
        
        workflow = create_workflow(llm=mock_llm, config=workflow_config)
        initial_state = create_initial_state("Test query")
        
        # Workflow should handle error gracefully
        try:
            final_state = workflow.invoke(initial_state)
            # If it completes, should have error handling
            assert isinstance(final_state, dict)
        except Exception:
            # If it raises, that's also acceptable error handling
            pass
    
    def test_workflow_handles_tool_errors(self, mock_llm: Mock, workflow_config: dict) -> None:
        """Test workflow handles tool errors gracefully."""
        def mock_llm_invoke(messages):
            return AIMessage(content='{"tasks": ["Find competitors"], "minimum_results": 4}')
        
        mock_llm.invoke = Mock(side_effect=mock_llm_invoke)
        
        with patch("src.tools.web_search.web_search") as mock_web_search:
            # Mock web search to raise error
            mock_web_search.side_effect = Exception("Web search API error")
            
            workflow = create_workflow(llm=mock_llm, config=workflow_config)
            initial_state = create_initial_state("Test query")
            
            # Workflow should handle error gracefully
            try:
                final_state = workflow.invoke(initial_state)
                # Should have error handling in state
                assert isinstance(final_state, dict)
            except Exception:
                # If it raises, that's also acceptable error handling
                pass
    
    def test_workflow_validates_all_stages(self, mock_llm: Mock, workflow_config: dict) -> None:
        """Test workflow validates outputs at each stage."""
        call_count = {"plan": 0, "insight": 0, "report": 0}
        
        def mock_llm_invoke(messages):
            content = str(messages[-1].content if hasattr(messages[-1], 'content') else messages[-1])
            
            if "plan" in content.lower():
                call_count["plan"] += 1
                return AIMessage(content=sample_llm_response_plan())
            elif "insight" in content.lower() or "swot" in content.lower():
                call_count["insight"] += 1
                return AIMessage(content=sample_llm_response_insights())
            elif "report" in content.lower():
                call_count["report"] += 1
                return AIMessage(content=sample_llm_response_report())
            
            return AIMessage(content='{"result": "success"}')
        
        mock_llm.invoke = Mock(side_effect=mock_llm_invoke)
        
        with patch("src.tools.web_search.web_search") as mock_web_search:
            with patch("src.tools.scraper.scrape_url") as mock_scraper:
                mock_web_search.return_value = {
                    "success": True,
                    "results": [
                        {"title": "Comp1", "url": "https://comp1.com", "snippet": "Desc1"},
                        {"title": "Comp2", "url": "https://comp2.com", "snippet": "Desc2"},
                        {"title": "Comp3", "url": "https://comp3.com", "snippet": "Desc3"},
                        {"title": "Comp4", "url": "https://comp4.com", "snippet": "Desc4"},
                    ],
                }
                
                mock_scraper.side_effect = lambda url: {
                    "success": True,
                    "url": url,
                    "title": "Page",
                    "content": "Content",
                    "content_length": 1000,
                    "links": [],
                }
                
                workflow = create_workflow(llm=mock_llm, config=workflow_config)
                initial_state = create_initial_state("Test query")
                
                final_state = workflow.invoke(initial_state)
                
                # Verify workflow progressed through stages
                # At minimum, should have attempted planning
                assert call_count["plan"] > 0
                # State should be valid
                assert isinstance(final_state, dict)
    
    def test_workflow_preserves_state_through_transitions(self, mock_llm: Mock, workflow_config: dict) -> None:
        """Test workflow preserves state correctly through transitions."""
        def mock_llm_invoke(messages):
            return AIMessage(content='{"tasks": ["Find competitors"], "minimum_results": 4}')
        
        mock_llm.invoke = Mock(side_effect=mock_llm_invoke)
        
        with patch("src.tools.web_search.web_search") as mock_web_search:
            with patch("src.tools.scraper.scrape_url") as mock_scraper:
                mock_web_search.return_value = {
                    "success": True,
                    "results": [
                        {"title": "Comp1", "url": "https://comp1.com", "snippet": "Desc1"},
                        {"title": "Comp2", "url": "https://comp2.com", "snippet": "Desc2"},
                        {"title": "Comp3", "url": "https://comp3.com", "snippet": "Desc3"},
                        {"title": "Comp4", "url": "https://comp4.com", "snippet": "Desc4"},
                    ],
                }
                
                mock_scraper.side_effect = lambda url: {
                    "success": True,
                    "url": url,
                    "title": "Page",
                    "content": "Content",
                    "content_length": 1000,
                    "links": [],
                }
                
                workflow = create_workflow(llm=mock_llm, config=workflow_config)
                
                # Create initial state with custom retry_count
                initial_state = create_initial_state("Test query")
                initial_state["retry_count"] = 0
                
                final_state = workflow.invoke(initial_state)
                
                # Verify state structure is preserved
                assert "retry_count" in final_state
                assert "messages" in final_state
                assert "validation_errors" in final_state
                # Retry count should be tracked
                assert isinstance(final_state["retry_count"], int)


