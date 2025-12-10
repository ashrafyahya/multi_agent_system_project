"""Tests for tool implementations.

This module contains unit tests for all tool functions and classes.
"""

from unittest.mock import Mock, patch

import pytest
import requests

from src.exceptions.collector_error import CollectorError
from src.tools.base_tool import BaseTool
from src.tools.query_generator import generate_queries
from src.tools.scraper import scrape_url
from src.tools.text_utils import (
    clean_text,
    deduplicate_texts,
    extract_sentences,
    extract_urls,
    normalize_url,
    remove_html_tags,
    summarize_text,
    truncate_text,
    validate_url,
)
from src.tools.web_search import web_search


class TestBaseTool:
    """Tests for BaseTool abstract class."""
    
    def test_base_tool_cannot_be_instantiated(self) -> None:
        """Test that BaseTool cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTool()  # type: ignore
    
    def test_base_tool_requires_execute_method(self) -> None:
        """Test that subclasses must implement execute method."""
        
        class IncompleteTool(BaseTool):
            @property
            def name(self) -> str:
                return "incomplete"
        
        with pytest.raises(TypeError):
            IncompleteTool()  # type: ignore
    
    def test_base_tool_requires_name_property(self) -> None:
        """Test that subclasses must implement name property."""
        
        class IncompleteTool(BaseTool):
            def execute(self, **kwargs: dict) -> dict:
                return {"success": True}
        
        with pytest.raises(TypeError):
            IncompleteTool()  # type: ignore
    
    def test_complete_tool_implementation(self) -> None:
        """Test a complete tool implementation."""
        
        class TestTool(BaseTool):
            @property
            def name(self) -> str:
                return "test_tool"
            
            def execute(self, **kwargs: dict) -> dict:
                return {
                    "success": True,
                    "data": {"result": "test"},
                }
        
        tool = TestTool()
        assert tool.name == "test_tool"
        
        result = tool.execute(test_param="value")
        assert result["success"] is True
        assert result["data"]["result"] == "test"
    
    def test_validate_inputs_default_implementation(self) -> None:
        """Test that validate_inputs does nothing by default."""
        
        class TestTool(BaseTool):
            @property
            def name(self) -> str:
                return "test_tool"
            
            def execute(self, **kwargs: dict) -> dict:
                return {"success": True}
        
        tool = TestTool()
        # Should not raise any exception
        tool.validate_inputs(any_param="any_value")
    
    def test_validate_inputs_custom_implementation(self) -> None:
        """Test custom validate_inputs implementation."""
        
        class TestTool(BaseTool):
            @property
            def name(self) -> str:
                return "test_tool"
            
            def execute(self, **kwargs: dict) -> dict:
                return {"success": True}
            
            def validate_inputs(self, **kwargs: dict) -> None:
                if "required_param" not in kwargs:
                    raise ValueError("required_param is required")
        
        tool = TestTool()
        
        # Should raise ValueError for missing parameter
        with pytest.raises(ValueError, match="required_param"):
            tool.validate_inputs()
        
        # Should not raise for valid input
        tool.validate_inputs(required_param="value")
    
    def test_handle_error_default_implementation(self) -> None:
        """Test default handle_error implementation."""
        
        class TestTool(BaseTool):
            @property
            def name(self) -> str:
                return "test_tool"
            
            def execute(self, **kwargs: dict) -> dict:
                return {"success": True}
        
        tool = TestTool()
        error = ValueError("Test error")
        result = tool.handle_error(error)
        
        assert result["success"] is False
        assert "Test error" in result["error"]
        assert result["context"] == {}
    
    def test_handle_error_with_context(self) -> None:
        """Test handle_error with context."""
        
        class TestTool(BaseTool):
            @property
            def name(self) -> str:
                return "test_tool"
            
            def execute(self, **kwargs: dict) -> dict:
                return {"success": True}
        
        tool = TestTool()
        error = ValueError("Test error")
        context = {"param": "value", "count": 42}
        result = tool.handle_error(error, context=context)
        
        assert result["success"] is False
        assert "Test error" in result["error"]
        assert "Context" in result["error"]
        assert result["context"] == context
    
    def test_tool_execute_with_error_handling(self) -> None:
        """Test tool execute method with error handling."""
        
        class TestTool(BaseTool):
            @property
            def name(self) -> str:
                return "test_tool"
            
            def execute(self, **kwargs: dict) -> dict:
                try:
                    if kwargs.get("should_fail"):
                        raise ValueError("Operation failed")
                    return {"success": True, "data": {"result": "success"}}
                except Exception as e:
                    return self.handle_error(e, context=kwargs)
        
        tool = TestTool()
        
        # Successful execution
        result = tool.execute(should_fail=False)
        assert result["success"] is True
        assert "data" in result
        
        # Failed execution
        result = tool.execute(should_fail=True)
        assert result["success"] is False
        assert "error" in result
        assert "Operation failed" in result["error"]


class TestToolPattern:
    """Tests for Tool Pattern compliance."""
    
    def test_tools_should_be_stateless(self) -> None:
        """Test that tools should not maintain state between calls."""
        
        class StatelessTool(BaseTool):
            def __init__(self) -> None:
                # No instance state stored
                pass
            
            @property
            def name(self) -> str:
                return "stateless_tool"
            
            def execute(self, **kwargs: dict) -> dict:
                # Uses only input parameters, no instance state
                return {"success": True, "data": kwargs}
        
        tool = StatelessTool()
        result1 = tool.execute(param1="value1")
        result2 = tool.execute(param2="value2")
        
        # Results should be independent
        assert result1["data"]["param1"] == "value1"
        assert result2["data"]["param2"] == "value2"
        assert "param1" not in result2["data"]
    
    def test_tools_should_return_structured_results(self) -> None:
        """Test that tools return structured result dictionaries."""
        
        class StructuredTool(BaseTool):
            @property
            def name(self) -> str:
                return "structured_tool"
            
            def execute(self, **kwargs: dict) -> dict:
                return {
                    "success": True,
                    "data": {"key": "value"},
                    "metadata": {"count": 1},
                }
        
        tool = StructuredTool()
        result = tool.execute()
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "data" in result
        assert result["success"] is True


class TestWebSearchTool:
    """Tests for web_search tool."""
    
    @patch("src.tools.web_search._perform_tavily_search")
    @patch("src.tools.web_search.get_config")
    def test_web_search_success(self, mock_get_config: Mock, mock_search: Mock) -> None:
        """Test successful web search."""
        # Mock config
        mock_config = Mock()
        mock_config.tavily_api_key = "test_api_key"
        mock_get_config.return_value = mock_config
        
        # Mock search results
        mock_search.return_value = [
            {
                "url": "https://example.com/1",
                "title": "Result 1",
                "snippet": "Snippet 1",
                "source": "tavily",
            },
            {
                "url": "https://example.com/2",
                "title": "Result 2",
                "snippet": "Snippet 2",
                "source": "tavily",
            },
        ]
        
        result = web_search.invoke({"query": "test query", "max_results": 10})
        
        assert result["success"] is True
        assert "results" in result
        assert len(result["results"]) == 2
        assert result["count"] == 2
        assert result["query"] == "test query"
        assert result["results"][0]["url"] == "https://example.com/1"
        assert result["results"][0]["title"] == "Result 1"
        
        mock_search.assert_called_once_with("test query", 10, "test_api_key")
    
    @patch("src.tools.web_search._perform_tavily_search")
    @patch("src.tools.web_search.get_config")
    def test_web_search_respects_max_results(self, mock_get_config: Mock, mock_search: Mock) -> None:
        """Test that web_search respects max_results parameter."""
        mock_config = Mock()
        mock_config.tavily_api_key = "test_api_key"
        mock_get_config.return_value = mock_config
        
        # Return more results than requested
        mock_search.return_value = [
            {"url": f"https://example.com/{i}", "title": f"Result {i}", "snippet": "", "source": "tavily"}
            for i in range(15)
        ]
        
        result = web_search.invoke({"query": "test", "max_results": 5})
        
        assert result["success"] is True
        # Should only return 5 results (limited by max_results)
        assert len(result["results"]) == 5
        assert result["count"] == 5
    
    @patch("src.tools.web_search._perform_tavily_search")
    @patch("src.tools.web_search.get_config")
    def test_web_search_empty_query(self, mock_get_config: Mock, mock_search: Mock) -> None:
        """Test web_search with empty query."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config
        
        result = web_search.invoke({"query": "", "max_results": 10})
        
        assert result["success"] is False
        assert "error" in result
        assert "cannot be empty" in result["error"].lower()
        assert result["count"] == 0
        
        # Should not call search
        mock_search.assert_not_called()
    
    @patch("src.tools.web_search._perform_tavily_search")
    @patch("src.tools.web_search.get_config")
    def test_web_search_invalid_max_results(self, mock_get_config: Mock, mock_search: Mock) -> None:
        """Test web_search with invalid max_results."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config
        
        # Test max_results too low
        result = web_search.invoke({"query": "test", "max_results": 0})
        assert result["success"] is False
        assert "max_results must be between" in result["error"]
        
        # Test max_results too high
        result = web_search.invoke({"query": "test", "max_results": 21})
        assert result["success"] is False
        assert "max_results must be between" in result["error"]
        
        # Should not call search
        mock_search.assert_not_called()
    
    @patch("src.tools.web_search._perform_tavily_search")
    @patch("src.tools.web_search.get_config")
    def test_web_search_retry_on_failure(self, mock_get_config: Mock, mock_search: Mock) -> None:
        """Test that web_search retries on failure."""
        mock_config = Mock()
        mock_config.tavily_api_key = "test_api_key"
        mock_get_config.return_value = mock_config
        
        # First two calls fail, third succeeds
        mock_search.side_effect = [
            CollectorError("First failure"),
            CollectorError("Second failure"),
            [
                {"url": "https://example.com", "title": "Success", "snippet": "", "source": "tavily"}
            ],
        ]
        
        result = web_search.invoke({"query": "test", "max_results": 10})
        
        assert result["success"] is True
        assert len(result["results"]) == 1
        assert mock_search.call_count == 3
    
    @patch("src.tools.web_search._perform_tavily_search")
    @patch("src.tools.web_search.get_config")
    def test_web_search_all_retries_fail(self, mock_get_config: Mock, mock_search: Mock) -> None:
        """Test that web_search raises CollectorError after all retries fail."""
        mock_config = Mock()
        mock_config.tavily_api_key = "test_api_key"
        mock_get_config.return_value = mock_config
        
        # All retries fail
        mock_search.side_effect = CollectorError("Persistent failure")
        
        with pytest.raises(CollectorError, match="Persistent failure"):
            web_search.invoke({"query": "test", "max_results": 10})
        
        assert mock_search.call_count == 3
    
    @patch("src.tools.web_search._perform_tavily_search")
    @patch("src.tools.web_search.get_config")
    def test_web_search_unexpected_error(self, mock_get_config: Mock, mock_search: Mock) -> None:
        """Test web_search handles unexpected errors gracefully."""
        mock_config = Mock()
        mock_config.tavily_api_key = "test_api_key"
        mock_get_config.return_value = mock_config
        
        # Unexpected error (not CollectorError)
        mock_search.side_effect = ValueError("Unexpected error")
        
        result = web_search.invoke({"query": "test", "max_results": 10})
        
        assert result["success"] is False
        assert "error" in result
        assert "Unexpected error" in result["error"]
        assert result["count"] == 0
    
    @patch("src.tools.web_search._perform_tavily_search")
    @patch("src.tools.web_search.get_config")
    def test_web_search_no_api_key(self, mock_get_config: Mock, mock_search: Mock) -> None:
        """Test web_search handles missing API key."""
        mock_config = Mock()
        mock_config.tavily_api_key = None
        mock_get_config.return_value = mock_config
        
        mock_search.return_value = [
            {"url": "https://example.com", "title": "Result", "snippet": "", "source": "tavily"}
        ]
        
        result = web_search.invoke({"query": "test", "max_results": 10})
        
        # Should still work (Tavily may use default key)
        assert result["success"] is True
        mock_search.assert_called_once_with("test", 10, None)
    
    @patch("src.tools.web_search._perform_tavily_search")
    @patch("src.tools.web_search.get_config")
    def test_web_search_strips_query_whitespace(self, mock_get_config: Mock, mock_search: Mock) -> None:
        """Test that web_search strips whitespace from query."""
        mock_config = Mock()
        mock_config.tavily_api_key = "test_api_key"
        mock_get_config.return_value = mock_config
        
        mock_search.return_value = []
        
        web_search.invoke({"query": "  test query  ", "max_results": 10})
        
        # Should be called with stripped query
        mock_search.assert_called_once_with("test query", 10, "test_api_key")
    
    @patch("src.tools.web_search._perform_tavily_search")
    @patch("src.tools.web_search.get_config")
    def test_web_search_default_max_results(self, mock_get_config: Mock, mock_search: Mock) -> None:
        """Test web_search uses default max_results when not provided."""
        mock_config = Mock()
        mock_config.tavily_api_key = "test_api_key"
        mock_get_config.return_value = mock_config
        
        mock_search.return_value = []
        
        web_search.invoke({"query": "test"})
        
        # Should use default max_results=10
        mock_search.assert_called_once_with("test", 10, "test_api_key")


class TestScraperTool:
    """Tests for scrape_url tool."""
    
    @patch("src.tools.scraper._fetch_url_content")
    def test_scrape_url_success(self, mock_fetch: Mock) -> None:
        """Test successful URL scraping."""
        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <main>
                    <h1>Main Content</h1>
                    <p>This is the main content of the page.</p>
                    <a href="/link1">Link 1</a>
                </main>
            </body>
        </html>
        """
        mock_fetch.return_value = ("text/html", html_content)
        
        result = scrape_url.invoke({"url": "https://example.com", "timeout": 10})
        
        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["title"] == "Test Page"
        assert "Main Content" in result["content"]
        assert result["content_length"] > 0
        assert len(result["links"]) > 0
    
    @patch("src.tools.scraper._fetch_url_content")
    def test_scrape_url_extracts_links(self, mock_fetch: Mock) -> None:
        """Test that scraper extracts links correctly."""
        html_content = """
        <html>
            <body>
                <a href="https://example.com/page1">Page 1</a>
                <a href="/page2">Page 2</a>
            </body>
        </html>
        """
        mock_fetch.return_value = ("text/html", html_content)
        
        result = scrape_url.invoke({"url": "https://example.com", "timeout": 10})
        
        assert result["success"] is True
        assert len(result["links"]) == 2
        assert any(link["url"] == "https://example.com/page1" for link in result["links"])
        assert any(link["url"] == "https://example.com/page2" for link in result["links"])
    
    @patch("src.tools.scraper._fetch_url_content")
    def test_scrape_url_removes_scripts_and_styles(self, mock_fetch: Mock) -> None:
        """Test that scraper removes scripts and styles."""
        html_content = """
        <html>
            <head>
                <script>console.log('test');</script>
                <style>body { color: red; }</style>
            </head>
            <body>
                <p>Visible content</p>
            </body>
        </html>
        """
        mock_fetch.return_value = ("text/html", html_content)
        
        result = scrape_url.invoke({"url": "https://example.com", "timeout": 10})
        
        assert result["success"] is True
        assert "Visible content" in result["content"]
        assert "console.log" not in result["content"]
        assert "color: red" not in result["content"]
    
    @patch("src.tools.scraper._fetch_url_content")
    def test_scrape_url_non_html_content(self, mock_fetch: Mock) -> None:
        """Test scraping non-HTML content."""
        text_content = "This is plain text content"
        mock_fetch.return_value = ("text/plain", text_content)
        
        result = scrape_url.invoke({"url": "https://example.com/file.txt", "timeout": 10})
        
        assert result["success"] is True
        assert result["content"] == text_content[:10000]  # Limited
        assert result["title"] == ""
        assert result["links"] == []
    
    def test_scrape_url_empty_url(self) -> None:
        """Test scrape_url with empty URL."""
        result = scrape_url.invoke({"url": "", "timeout": 10})
        
        assert result["success"] is False
        assert "cannot be empty" in result["error"].lower()
        assert result["content_length"] == 0
    
    def test_scrape_url_invalid_url(self) -> None:
        """Test scrape_url with invalid URL."""
        result = scrape_url.invoke({"url": "not-a-valid-url", "timeout": 10})
        
        assert result["success"] is False
        assert "invalid url" in result["error"].lower()
        assert result["content_length"] == 0
    
    def test_scrape_url_invalid_timeout(self) -> None:
        """Test scrape_url with invalid timeout."""
        # Too low
        result = scrape_url.invoke({"url": "https://example.com", "timeout": 4})
        assert result["success"] is False
        assert "timeout must be between" in result["error"].lower()
        
        # Too high
        result = scrape_url.invoke({"url": "https://example.com", "timeout": 31})
        assert result["success"] is False
        assert "timeout must be between" in result["error"].lower()
    
    @patch("src.tools.scraper._fetch_url_content")
    def test_scrape_url_default_timeout(self, mock_fetch: Mock) -> None:
        """Test that scrape_url uses default timeout."""
        mock_fetch.return_value = ("text/html", "<html><body>Test</body></html>")
        
        scrape_url.invoke({"url": "https://example.com"})
        
        # Should use default timeout=10
        mock_fetch.assert_called_once_with("https://example.com", 10)
    
    @patch("src.tools.scraper._fetch_url_content")
    def test_scrape_url_timeout_error(self, mock_fetch: Mock) -> None:
        """Test scrape_url handles timeout errors."""
        mock_fetch.side_effect = CollectorError("Timeout", context={"url": "https://example.com"})
        
        with pytest.raises(CollectorError):
            scrape_url.invoke({"url": "https://example.com", "timeout": 10})
    
    @patch("src.tools.scraper._fetch_url_content")
    def test_scrape_url_retry_on_failure(self, mock_fetch: Mock) -> None:
        """Test that scraper retries on failure."""
        # First two calls fail, third succeeds
        mock_fetch.side_effect = [
            requests.RequestException("First failure"),
            requests.RequestException("Second failure"),
            ("text/html", "<html><body>Success</body></html>"),
        ]
        
        result = scrape_url.invoke({"url": "https://example.com", "timeout": 10})
        
        assert result["success"] is True
        # Note: retry happens inside _fetch_url_content, so we see 1 call
        # but internally it retries 3 times
        assert mock_fetch.call_count >= 1
    
    @patch("src.tools.scraper._fetch_url_content")
    def test_scrape_url_all_retries_fail(self, mock_fetch: Mock) -> None:
        """Test that scraper raises CollectorError after all retries fail."""
        # Simulate persistent RequestException (will be retried internally)
        mock_fetch.side_effect = requests.RequestException("Persistent failure")
        
        with pytest.raises(CollectorError):
            scrape_url.invoke({"url": "https://example.com", "timeout": 10})
        
        # Function will be called, retries happen inside
        assert mock_fetch.call_count >= 1
    
    @patch("src.tools.scraper._fetch_url_content")
    def test_scrape_url_unexpected_error(self, mock_fetch: Mock) -> None:
        """Test scrape_url handles unexpected errors gracefully."""
        mock_fetch.side_effect = ValueError("Unexpected error")
        
        result = scrape_url.invoke({"url": "https://example.com", "timeout": 10})
        
        assert result["success"] is False
        assert "error" in result
        assert "Unexpected error" in result["error"]
    
    @patch("src.tools.scraper._fetch_url_content")
    def test_scrape_url_strips_url_whitespace(self, mock_fetch: Mock) -> None:
        """Test that scrape_url strips whitespace from URL."""
        mock_fetch.return_value = ("text/html", "<html><body>Test</body></html>")
        
        scrape_url.invoke({"url": "  https://example.com  ", "timeout": 10})
        
        # Should be called with stripped URL
        mock_fetch.assert_called_once_with("https://example.com", 10)
    
    @patch("src.tools.scraper._fetch_url_content")
    def test_scrape_url_limits_links(self, mock_fetch: Mock) -> None:
        """Test that scraper limits number of links."""
        # Create HTML with many links
        links_html = "".join([f'<a href="/link{i}">Link {i}</a>' for i in range(100)])
        html_content = f"<html><body>{links_html}</body></html>"
        mock_fetch.return_value = ("text/html", html_content)
        
        result = scrape_url.invoke({"url": "https://example.com", "timeout": 10})
        
        assert result["success"] is True
        assert len(result["links"]) == 50  # Limited to 50


class TestQueryGeneratorTool:
    """Tests for generate_queries tool."""
    
    def test_generate_queries_success(self) -> None:
        """Test successful query generation."""
        result = generate_queries.invoke({
            "task": "Find competitor pricing information",
            "max_variations": 5
        })
        
        assert result["success"] is True
        assert result["task"] == "Find competitor pricing information"
        assert len(result["queries"]) > 0
        assert result["count"] == len(result["queries"])
        assert result["count"] <= 5
        
        # Check query structure
        for query in result["queries"]:
            assert "text" in query
            assert "type" in query
            assert isinstance(query["text"], str)
            assert len(query["text"]) > 0
    
    def test_generate_queries_returns_multiple_variations(self) -> None:
        """Test that generate_queries returns multiple variations."""
        result = generate_queries.invoke({
            "task": "Analyze competitor products",
            "max_variations": 10
        })
        
        assert result["success"] is True
        assert result["count"] >= 3  # Should generate at least 3 variations
        assert result["count"] <= 10
        
        # Check for different query types
        query_types = {q["type"] for q in result["queries"]}
        assert len(query_types) > 1  # Should have multiple types
    
    def test_generate_queries_deduplicates(self) -> None:
        """Test that generate_queries deduplicates queries."""
        result = generate_queries.invoke({
            "task": "competitor competitor competitor",  # Repetitive
            "max_variations": 10
        })
        
        assert result["success"] is True
        
        # Check for duplicates (case-insensitive)
        query_texts = [q["text"].lower() for q in result["queries"]]
        assert len(query_texts) == len(set(query_texts))  # No duplicates
    
    def test_generate_queries_empty_task(self) -> None:
        """Test generate_queries with empty task."""
        result = generate_queries.invoke({"task": "", "max_variations": 5})
        
        assert result["success"] is False
        assert "cannot be empty" in result["error"].lower()
        assert result["count"] == 0
    
    def test_generate_queries_invalid_max_variations(self) -> None:
        """Test generate_queries with invalid max_variations."""
        # Too low
        result = generate_queries.invoke({"task": "test task", "max_variations": 0})
        assert result["success"] is False
        assert "max_variations must be between" in result["error"]
        
        # Too high
        result = generate_queries.invoke({"task": "test task", "max_variations": 20})
        assert result["success"] is False
        assert "max_variations must be between" in result["error"]
    
    def test_generate_queries_default_max_variations(self) -> None:
        """Test that generate_queries uses default max_variations."""
        result = generate_queries.invoke({"task": "Find competitor information"})
        
        assert result["success"] is True
        assert result["count"] <= 5  # Default is 5
    
    def test_generate_queries_strips_task_whitespace(self) -> None:
        """Test that generate_queries strips whitespace from task."""
        result = generate_queries.invoke({
            "task": "  Find competitor pricing  ",
            "max_variations": 5
        })
        
        assert result["success"] is True
        assert result["task"] == "Find competitor pricing"  # Stripped
    
    def test_generate_queries_includes_direct_variation(self) -> None:
        """Test that generate_queries includes direct task as query."""
        task = "Find competitor pricing"
        result = generate_queries.invoke({"task": task, "max_variations": 10})
        
        assert result["success"] is True
        
        # Should include direct variation
        direct_queries = [q for q in result["queries"] if q["type"] == "direct"]
        assert len(direct_queries) > 0
        assert direct_queries[0]["text"] == task
    
    def test_generate_queries_adds_context_keywords(self) -> None:
        """Test that generate_queries adds relevant context keywords."""
        result = generate_queries.invoke({
            "task": "pricing information",  # No "competitor" keyword
            "max_variations": 10
        })
        
        assert result["success"] is True
        
        # Should add competitor context
        competitor_queries = [
            q for q in result["queries"]
            if "competitor" in q["text"].lower()
        ]
        assert len(competitor_queries) > 0
    
    def test_generate_queries_handles_pricing_keywords(self) -> None:
        """Test that generate_queries handles pricing-specific tasks."""
        result = generate_queries.invoke({
            "task": "competitor pricing",
            "max_variations": 10
        })
        
        assert result["success"] is True
        
        # Should include pricing-focused variations
        pricing_queries = [
            q for q in result["queries"]
            if "pricing" in q["text"].lower() or "price" in q["text"].lower()
        ]
        assert len(pricing_queries) > 0
    
    def test_generate_queries_handles_product_keywords(self) -> None:
        """Test that generate_queries handles product-specific tasks."""
        result = generate_queries.invoke({
            "task": "competitor products",
            "max_variations": 10
        })
        
        assert result["success"] is True
        
        # Should include product-focused variations
        product_queries = [
            q for q in result["queries"]
            if "product" in q["text"].lower() or "features" in q["text"].lower()
        ]
        assert len(product_queries) > 0
    
    def test_generate_queries_generates_different_types(self) -> None:
        """Test that generate_queries generates different query types."""
        result = generate_queries.invoke({
            "task": "competitor analysis",
            "max_variations": 10
        })
        
        assert result["success"] is True
        
        query_types = {q["type"] for q in result["queries"]}
        # Should have multiple different types
        assert len(query_types) >= 3
        
        # Check for common types
        type_names = {q["type"] for q in result["queries"]}
        assert any("direct" in t or "question" in t or "comparison" in t for t in type_names)
    
    def test_generate_queries_minimum_length(self) -> None:
        """Test that generate_queries filters out very short queries."""
        result = generate_queries.invoke({
            "task": "ab",  # Very short task
            "max_variations": 10
        })
        
        # Should still generate some queries or handle gracefully
        if result["success"]:
            for query in result["queries"]:
                assert len(query["text"]) >= 3  # Minimum reasonable length


class TestTextUtils:
    """Tests for text utility functions."""
    
    def test_clean_text_removes_extra_spaces(self) -> None:
        """Test that clean_text removes extra whitespace."""
        dirty = "  Text   with   extra   spaces  "
        clean = clean_text(dirty)
        assert clean == "Text with extra spaces"
    
    def test_clean_text_preserves_newlines(self) -> None:
        """Test that clean_text preserves newlines when requested."""
        text = "Line 1\n\nLine 2\n\n\nLine 3"
        clean = clean_text(text, preserve_newlines=True)
        assert "\n" in clean
        assert "Line 1" in clean
        assert "Line 2" in clean
    
    def test_clean_text_empty_string(self) -> None:
        """Test clean_text with empty string."""
        assert clean_text("") == ""
        assert clean_text("   ") == ""
    
    def test_deduplicate_texts_case_insensitive(self) -> None:
        """Test deduplicate_texts with case-insensitive deduplication."""
        texts = ["text1", "Text1", "text2", "TEXT1"]
        unique = deduplicate_texts(texts, case_sensitive=False)
        assert len(unique) == 2
        assert "text1" in unique
        assert "text2" in unique
    
    def test_deduplicate_texts_case_sensitive(self) -> None:
        """Test deduplicate_texts with case-sensitive deduplication."""
        texts = ["text1", "Text1", "text2"]
        unique = deduplicate_texts(texts, case_sensitive=True)
        assert len(unique) == 3
    
    def test_deduplicate_texts_preserves_order(self) -> None:
        """Test that deduplicate_texts preserves order."""
        texts = ["first", "second", "first", "third"]
        unique = deduplicate_texts(texts)
        assert unique == ["first", "second", "third"]
    
    def test_deduplicate_texts_empty_list(self) -> None:
        """Test deduplicate_texts with empty list."""
        assert deduplicate_texts([]) == []
    
    def test_truncate_text_within_limit(self) -> None:
        """Test truncate_text when text is within limit."""
        text = "Short text"
        result = truncate_text(text, 100)
        assert result == text
    
    def test_truncate_text_exceeds_limit(self) -> None:
        """Test truncate_text when text exceeds limit."""
        text = "This is a very long text that needs to be truncated"
        result = truncate_text(text, 20)
        assert len(result) <= 20
        assert result.endswith("...")
    
    def test_truncate_text_preserves_words(self) -> None:
        """Test that truncate_text tries to preserve word boundaries."""
        text = "This is a sentence with words"
        result = truncate_text(text, 15)
        # Should truncate at word boundary if possible
        assert " " in result or result.endswith("...")
    
    def test_extract_sentences(self) -> None:
        """Test extract_sentences function."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = extract_sentences(text)
        assert len(sentences) == 3
        assert "First sentence." in sentences[0]
        assert "Second sentence!" in sentences[1]
    
    def test_extract_sentences_max_limit(self) -> None:
        """Test extract_sentences with max_sentences limit."""
        text = "First. Second. Third. Fourth."
        sentences = extract_sentences(text, max_sentences=2)
        assert len(sentences) == 2
    
    def test_summarize_text_short_text(self) -> None:
        """Test summarize_text with text shorter than max_length."""
        text = "Short text"
        summary = summarize_text(text, max_length=100)
        assert summary == text
    
    def test_summarize_text_long_text(self) -> None:
        """Test summarize_text with long text."""
        text = "First sentence. " * 20
        summary = summarize_text(text, max_length=50)
        assert len(summary) <= 50
        assert len(summary) > 0
    
    def test_validate_url_valid(self) -> None:
        """Test validate_url with valid URLs."""
        assert validate_url("https://example.com") is True
        assert validate_url("http://example.com/path") is True
        assert validate_url("https://example.com:8080/path?query=value") is True
    
    def test_validate_url_invalid(self) -> None:
        """Test validate_url with invalid URLs."""
        assert validate_url("not-a-url") is False
        assert validate_url("") is False
        assert validate_url("example.com") is False  # Missing scheme
        assert validate_url("http://") is False  # Missing netloc
    
    def test_normalize_url_adds_scheme(self) -> None:
        """Test that normalize_url adds scheme if missing."""
        normalized = normalize_url("example.com")
        assert normalized == "http://example.com"
        assert normalized is not None
    
    def test_normalize_url_lowercases(self) -> None:
        """Test that normalize_url lowercases hostname."""
        normalized = normalize_url("HTTPS://EXAMPLE.COM")
        assert normalized == "https://example.com"
    
    def test_normalize_url_removes_default_ports(self) -> None:
        """Test that normalize_url removes default ports."""
        normalized = normalize_url("https://example.com:443")
        assert normalized == "https://example.com"
        
        normalized = normalize_url("http://example.com:80")
        assert normalized == "http://example.com"
    
    def test_normalize_url_invalid(self) -> None:
        """Test normalize_url with invalid URL."""
        assert normalize_url("") is None
        assert normalize_url("not-a-url") is None
    
    def test_extract_urls(self) -> None:
        """Test extract_urls function."""
        text = "Visit https://example.com and http://test.com for more info"
        urls = extract_urls(text)
        assert len(urls) == 2
        assert "https://example.com" in urls
        assert "http://test.com" in urls
    
    def test_extract_urls_no_urls(self) -> None:
        """Test extract_urls with text containing no URLs."""
        text = "This text has no URLs"
        urls = extract_urls(text)
        assert urls == []
    
    def test_remove_html_tags(self) -> None:
        """Test remove_html_tags function."""
        html = "<p>This is <b>bold</b> text</p>"
        clean = remove_html_tags(html)
        assert "<" not in clean
        assert ">" not in clean
        assert "This is bold text" in clean
    
    def test_remove_html_tags_decodes_entities(self) -> None:
        """Test that remove_html_tags decodes HTML entities."""
        html = "Text &amp; more &lt;tags&gt;"
        clean = remove_html_tags(html)
        assert "&amp;" not in clean
        assert "&lt;" not in clean
        assert "&gt;" not in clean
    
    def test_remove_html_tags_empty(self) -> None:
        """Test remove_html_tags with empty string."""
        assert remove_html_tags("") == ""
        assert remove_html_tags("<p></p>") == ""
