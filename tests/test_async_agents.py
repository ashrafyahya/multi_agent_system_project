"""Tests for async agent functionality.

This module tests the async capabilities of agents, including:
- Async LLM invocation
- Async web search and scraping
- Async data collection
- Fallback behavior when async is not available
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agents.base_agent import BaseAgent
from src.agents.data_collector import DataCollectorAgent
from src.config import Config
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState
from src.models.plan_model import Plan
from src.tools.scraper import scrape_url_async
from src.tools.web_search import web_search_async
from src.utils.rate_limiter import invoke_llm_with_retry_async


class TestAsyncAgent(BaseAgent):
    """Test agent for async testing."""
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        """Sync execute implementation."""
        return state
    
    @property
    def name(self) -> str:
        """Return agent name."""
        return "test_async_agent"


class TestAsyncLLMInvocation:
    """Tests for async LLM invocation."""
    
    @pytest.mark.asyncio
    async def test_invoke_llm_async_with_ainvoke(self):
        """Test async LLM invocation when LLM supports ainvoke."""
        # Create mock LLM with ainvoke
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Test response"))
        
        agent = TestAsyncAgent(mock_llm, {})
        
        messages = [HumanMessage(content="Test")]
        response = await agent.invoke_llm_async(messages)
        
        assert response.content == "Test response"
        mock_llm.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_invoke_llm_async_fallback_to_sync(self):
        """Test async LLM invocation falls back to sync when ainvoke not available."""
        # Create mock LLM without ainvoke
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_llm.invoke = MagicMock(return_value=MagicMock(content="Test response"))
        del mock_llm.ainvoke  # Remove ainvoke attribute
        
        agent = TestAsyncAgent(mock_llm, {})
        
        messages = [HumanMessage(content="Test")]
        response = await agent.invoke_llm_async(messages)
        
        assert response.content == "Test response"
        mock_llm.invoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_async_default_implementation(self):
        """Test default execute_async runs sync execute in executor."""
        mock_llm = MagicMock(spec=BaseChatModel)
        agent = TestAsyncAgent(mock_llm, {})
        
        state = WorkflowState()
        result = await agent.execute_async(state)
        
        assert result == state


class TestAsyncWebSearch:
    """Tests for async web search."""
    
    @pytest.mark.asyncio
    @patch("src.tools.web_search._perform_tavily_search_async")
    async def test_web_search_async_success(self, mock_search):
        """Test successful async web search."""
        mock_search.return_value = [
            {
                "url": "https://example.com",
                "title": "Example",
                "snippet": "Example content",
                "source": "tavily",
            }
        ]
        
        with patch("src.tools.web_search.get_config") as mock_config:
            mock_config.return_value.tavily_api_key = "test_key"
            result = await web_search_async("test query", max_results=10)
        
        assert result["success"] is True
        assert result["count"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["url"] == "https://example.com"
    
    @pytest.mark.asyncio
    async def test_web_search_async_invalid_query(self):
        """Test async web search with invalid query."""
        result = await web_search_async("", max_results=10)
        
        assert result["success"] is False
        assert "empty" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_web_search_async_invalid_max_results(self):
        """Test async web search with invalid max_results."""
        result = await web_search_async("test", max_results=0)
        
        assert result["success"] is False
        assert "max_results" in result["error"].lower()
    
    @pytest.mark.asyncio
    @patch("src.tools.web_search.get_config")
    async def test_web_search_async_no_api_key(self, mock_config):
        """Test async web search without API key."""
        mock_config.return_value.tavily_api_key = None
        
        result = await web_search_async("test query", max_results=10)
        
        assert result["success"] is False
        assert "tavily_api_key" in result["error"].lower() or "api key" in result["error"].lower()


class TestAsyncScraper:
    """Tests for async web scraper."""
    
    @pytest.mark.asyncio
    @patch("src.tools.scraper._fetch_url_content_async")
    async def test_scrape_url_async_success(self, mock_fetch):
        """Test successful async URL scraping."""
        mock_fetch.return_value = (
            "text/html",
            "<html><head><title>Test</title></head><body>Content</body></html>"
        )
        
        result = await scrape_url_async("https://example.com", timeout=10)
        
        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert "Content" in result["content"]
        assert result["title"] == "Test"
    
    @pytest.mark.asyncio
    async def test_scrape_url_async_invalid_url(self):
        """Test async scraping with invalid URL."""
        result = await scrape_url_async("not-a-url", timeout=10)
        
        assert result["success"] is False
        assert "invalid" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_scrape_url_async_invalid_timeout(self):
        """Test async scraping with invalid timeout."""
        result = await scrape_url_async("https://example.com", timeout=1)
        
        assert result["success"] is False
        assert "timeout" in result["error"].lower()
    
    @pytest.mark.asyncio
    @patch("src.tools.scraper.AIOHTTP_AVAILABLE", False)
    async def test_scrape_url_async_fallback_to_sync(self):
        """Test async scraping falls back to sync when aiohttp not available."""
        with patch("src.tools.scraper._fetch_url_content") as mock_fetch:
            mock_fetch.return_value = (
                "text/html",
                "<html><head><title>Test</title></head><body>Content</body></html>"
            )
            
            result = await scrape_url_async("https://example.com", timeout=10)
            
            assert result["success"] is True
            mock_fetch.assert_called_once()


class TestAsyncDataCollector:
    """Tests for async data collector agent."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        return MagicMock(spec=BaseChatModel)
    
    @pytest.fixture
    def agent_config(self):
        """Create agent configuration."""
        return {"max_results": 5}
    
    @pytest.fixture
    def plan_data(self):
        """Create plan data for testing."""
        return {
            "tasks": ["Find competitors in SaaS", "Analyze pricing"],
            "minimum_results": 3,
            "preferred_sources": ["web search"],
            "search_strategy": "comprehensive",
        }
    
    @pytest.mark.asyncio
    @patch("src.agents.data_collector.web_search_async")
    async def test_collect_competitor_data_async(self, mock_search, mock_llm, agent_config, plan_data):
        """Test async competitor data collection."""
        # Mock search results
        mock_search.return_value = {
            "success": True,
            "results": [
                {
                    "url": "https://competitor1.com",
                    "title": "Competitor 1",
                    "snippet": "Competitor 1 description",
                },
                {
                    "url": "https://competitor2.com",
                    "title": "Competitor 2",
                    "snippet": "Competitor 2 description",
                },
            ],
            "count": 2,
        }
        
        agent = DataCollectorAgent(mock_llm, agent_config)
        plan = Plan(**plan_data)
        
        competitors = await agent._collect_competitor_data_async(plan)
        
        assert len(competitors) > 0
        assert all(isinstance(c, type(competitors[0])) for c in competitors)
        # Verify parallel execution (should be called multiple times)
        assert mock_search.call_count == len(agent._generate_search_queries(plan.tasks))
    
    @pytest.mark.asyncio
    @patch("src.agents.data_collector.web_search_async")
    async def test_execute_async_success(self, mock_search, mock_llm, agent_config, plan_data):
        """Test async execute method."""
        mock_search.return_value = {
            "success": True,
            "results": [
                {
                    "url": "https://competitor1.com",
                    "title": "Competitor 1",
                    "snippet": "Competitor 1 description",
                },
            ],
            "count": 1,
        }
        
        agent = DataCollectorAgent(mock_llm, agent_config)
        state = WorkflowState(plan=plan_data)
        
        result = await agent.execute_async(state)
        
        assert "collected_data" in result
        assert "competitors" in result["collected_data"]
        assert len(result["collected_data"]["competitors"]) > 0
    
    @pytest.mark.asyncio
    async def test_execute_async_no_plan(self, mock_llm, agent_config):
        """Test async execute fails without plan."""
        agent = DataCollectorAgent(mock_llm, agent_config)
        state = WorkflowState()
        
        with pytest.raises(WorkflowError, match="plan"):
            await agent.execute_async(state)
    
    @pytest.mark.asyncio
    @patch("src.agents.data_collector.web_search_async")
    @patch("src.agents.data_collector.scrape_url_async")
    async def test_collect_competitor_data_async_with_scraping(
        self, mock_scrape, mock_search, mock_llm, agent_config, plan_data
    ):
        """Test async data collection with URL scraping."""
        mock_search.return_value = {
            "success": True,
            "results": [
                {
                    "url": "https://competitor1.com",
                    "title": "Competitor 1",
                    "snippet": "Competitor 1 description",
                },
            ],
            "count": 1,
        }
        
        mock_scrape.return_value = {
            "success": True,
            "url": "https://competitor1.com",
            "title": "Competitor 1",
            "content": "Detailed content",
            "content_length": 100,
            "links": [],
        }
        
        agent = DataCollectorAgent(mock_llm, agent_config)
        plan = Plan(**plan_data)
        
        competitors = await agent._collect_competitor_data_async(plan)
        
        # Should have collected competitors
        assert len(competitors) > 0
        # Scraping should be called if we need more data
        # (exact behavior depends on minimum_results)


class TestAsyncRateLimiter:
    """Tests for async rate limiter."""
    
    @pytest.mark.asyncio
    async def test_invoke_llm_with_retry_async_success(self):
        """Test successful async LLM invocation with retry."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Test response"))
        
        messages = [HumanMessage(content="Test")]
        response = await invoke_llm_with_retry_async(mock_llm, messages)
        
        assert response.content == "Test response"
        mock_llm.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_invoke_llm_with_retry_async_fallback(self):
        """Test async LLM invocation falls back to sync when ainvoke not available."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_llm.invoke = MagicMock(return_value=MagicMock(content="Test response"))
        del mock_llm.ainvoke
        
        messages = [HumanMessage(content="Test")]
        response = await invoke_llm_with_retry_async(mock_llm, messages)
        
        assert response.content == "Test response"
        mock_llm.invoke.assert_called_once()


class TestAsyncParallelExecution:
    """Tests for parallel async execution."""
    
    @pytest.mark.asyncio
    async def test_parallel_web_searches(self):
        """Test running multiple web searches in parallel."""
        queries = ["query1", "query2", "query3"]
        
        with patch("src.tools.web_search._perform_tavily_search_async") as mock_search:
            mock_search.return_value = []
            
            with patch("src.tools.web_search.get_config") as mock_config:
                mock_config.return_value.tavily_api_key = "test_key"
                
                results = await asyncio.gather(*[
                    web_search_async(q, max_results=10)
                    for q in queries
                ])
        
        assert len(results) == len(queries)
        assert all(isinstance(r, dict) for r in results)
    
    @pytest.mark.asyncio
    async def test_parallel_url_scraping(self):
        """Test running multiple URL scrapes in parallel."""
        urls = [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ]
        
        with patch("src.tools.scraper._fetch_url_content_async") as mock_fetch:
            mock_fetch.return_value = (
                "text/html",
                "<html><body>Content</body></html>"
            )
            
            results = await asyncio.gather(*[
                scrape_url_async(url, timeout=10)
                for url in urls
            ])
        
        assert len(results) == len(urls)
        assert all(r["success"] for r in results)

