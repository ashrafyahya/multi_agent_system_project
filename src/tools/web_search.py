"""Web search tool for gathering competitor information.

This tool performs web searches using Tavily search API through LangChain
and returns structured results with retry logic. It's designed to gather
information about competitors, their products, pricing, and market presence.

Example:
    ```python
    from src.tools.web_search import web_search
    
    result = web_search.invoke({
        "query": "competitor pricing analysis SaaS",
        "max_results": 10
    })
    
    if result["success"]:
        for item in result["results"]:
            print(f"{item['title']}: {item['url']}")
    ```
"""

import asyncio
import logging
import re
from typing import Any

try:
    # Try to use the new TavilySearch from langchain_tavily (recommended)
    # This eliminates the deprecation warning
    from langchain_tavily import TavilySearch
    TAVILY_SEARCH_CLASS = TavilySearch
except ImportError:
    try:
        # Fallback to TavilySearchResults from langchain_tavily
        from langchain_tavily import TavilySearchResults
        TAVILY_SEARCH_CLASS = TavilySearchResults
    except ImportError:
        # Final fallback to langchain_community (deprecated)
        from langchain_community.tools.tavily_search import TavilySearchResults
        TAVILY_SEARCH_CLASS = TavilySearchResults
from langchain_core.tools import tool
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)
from tenacity.asyncio import AsyncRetrying

from src.config import get_config
from src.exceptions.collector_error import CollectorError

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((CollectorError,)),
    reraise=True,
)
def _perform_tavily_search(query: str, max_results: int, api_key: str | None) -> list[dict[str, Any]]:
    """Perform Tavily search with retry logic.
    
    This internal function performs the actual Tavily search with retry
    logic using tenacity. It's separated from the tool function to allow
    proper retry decoration.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        api_key: Tavily API key (optional, can use default)
        
    Returns:
        List of search result dictionaries
        
    Raises:
        CollectorError: If search fails after all retries
        Exception: Other exceptions from Tavily API
    """
    try:
        # Validate API key before attempting search
        if not api_key:
            raise CollectorError(
                "Tavily API key is required but not provided. "
                "Please set TAVILY_API_KEY in your .env file or environment variables.",
                context={"query": query, "max_results": max_results}
            )
        
        # Create Tavily search tool
        # Both TavilySearch and TavilySearchResults use 'tavily_api_key'
        search_tool = TAVILY_SEARCH_CLASS(
            max_results=max_results,
            tavily_api_key=api_key,
        )
        
        # Perform search
        raw_response = search_tool.invoke(query)
        
        # Normalize response format: Tavily API evolution introduced breaking changes
        # Modern API (langchain_tavily >= 0.1.0): Returns dict with 'results' key containing array of result objects
        # Legacy API (langchain_community): Returns list of result objects directly
        # This abstraction ensures compatibility across API versions
        if isinstance(raw_response, dict) and "results" in raw_response:
            # Extract results array from modern API response wrapper
            results = raw_response.get("results", [])
        elif isinstance(raw_response, list):
            # Legacy format: response is already the results array
            results = raw_response
        else:
            # Edge case: handle unexpected response types by wrapping in list
            results = [raw_response] if raw_response else []
        
        # Handle different return formats
        formatted_results = []
        for result in results:
            if isinstance(result, str):
                # If result is a string, try to extract URL and content
                # Look for URLs in the string
                url_match = re.search(r'https?://[^\s]+', result)
                url = url_match.group(0) if url_match else ""
                # Use the string as snippet, extract title from first line
                lines = result.split('\n')
                title = lines[0][:100] if lines else result[:100]
                snippet = result
                
                formatted_results.append({
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "source": "tavily",
                })
            elif isinstance(result, dict):
                # Standard dictionary format from TavilySearchResults or TavilySearch
                # TavilySearch uses 'content', TavilySearchResults may use 'snippet'
                formatted_results.append({
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "snippet": result.get("content", result.get("snippet", result.get("raw_content", ""))),
                    "source": "tavily",
                })
            else:
                # Fallback for unexpected formats
                logger.warning(f"Unexpected result format: {type(result)}, value: {result}")
                formatted_results.append({
                    "url": "",
                    "title": str(result)[:100],
                    "snippet": str(result),
                    "source": "tavily",
                })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Tavily search failed: {e}", exc_info=True)
        raise CollectorError(
            f"Web search failed for query: {query}",
            context={"query": query, "max_results": max_results, "error": str(e)}
        ) from e


@tool
def web_search(query: str, max_results: int = 10) -> dict[str, Any]:
    """Search the web for information about competitors.
    
    This tool performs a web search using Tavily search API and returns
    structured results including URLs, titles, and snippets. Use this tool
    to gather information about competitor products, pricing, market presence,
    and other relevant business intelligence.
    
    The tool automatically handles retries for transient failures and rate
    limits. It returns structured data that can be easily processed by
    other agents in the workflow.
    
    Args:
        query: Search query string describing what to search for.
            Examples:
            - "competitor pricing analysis SaaS"
            - "top competitors in project management software"
            - "competitor product features comparison"
        max_results: Maximum number of search results to return.
            Default is 10, maximum recommended is 20 for best performance.
            Range: 1-20.
    
    Returns:
        Dictionary with the following structure:
        {
            "success": bool,  # True if search succeeded
            "results": [      # List of search results (if successful)
                {
                    "url": str,        # Result URL
                    "title": str,      # Result title
                    "snippet": str,    # Result snippet/content
                    "source": str      # Source name ("tavily")
                },
                ...
            ],
            "error": str,     # Error message (if failed)
            "query": str,     # Original query
            "count": int      # Number of results returned
        }
    
    Raises:
        CollectorError: If search fails after all retry attempts.
            This indicates a persistent failure that cannot be recovered.
    
    Example:
        ```python
        # Search for competitor information
        result = web_search.invoke({
            "query": "competitor pricing analysis",
            "max_results": 5
        })
        
        if result["success"]:
            print(f"Found {result['count']} results")
            for item in result["results"]:
                print(f"{item['title']}: {item['url']}")
        else:
            print(f"Search failed: {result['error']}")
        ```
    
    Note:
        - Requires TAVILY_API_KEY environment variable or config
        - Automatically retries up to 3 times on failure
        - Uses exponential backoff for retries
        - Results are limited to max_results parameter
    """
    # Validate inputs
    if not query or not query.strip():
        return {
            "success": False,
            "error": "Query cannot be empty",
            "results": [],
            "query": query,
            "count": 0,
        }
    
    if max_results < 1 or max_results > 20:
        return {
            "success": False,
            "error": f"max_results must be between 1 and 20, got {max_results}",
            "results": [],
            "query": query,
            "count": 0,
        }
    
    query = query.strip()
    
    try:
        # Get configuration
        config = get_config()
        api_key = config.tavily_api_key
        
        # Clean and validate API key
        if api_key:
            api_key = api_key.strip()
            if not api_key:
                api_key = None
        
        if not api_key:
            error_msg = (
                "TAVILY_API_KEY not configured. "
                "Please set TAVILY_API_KEY in your .env file or environment variables. "
                "The .env file should be in the project root directory. "
                "Format: TAVILY_API_KEY=your_key_here (no quotes, no spaces around =)"
            )
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "results": [],
                "query": query,
                "count": 0,
            }
        
        logger.info(f"Performing web search: query='{query}', max_results={max_results}")
        results = _perform_tavily_search(query, max_results, api_key)
        results = results[:max_results]
        
        logger.info(f"Web search successful: found {len(results)} results")
        
        return {
            "success": True,
            "results": results,
            "query": query,
            "count": len(results),
        }
        
    except CollectorError:
        raise
    except Exception as e:
        error_msg = f"Unexpected error during web search: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            "success": False,
            "error": error_msg,
            "results": [],
            "query": query,
            "count": 0,
        }


async def _perform_tavily_search_async(query: str, max_results: int, api_key: str | None) -> list[dict[str, Any]]:
    """Perform Tavily search asynchronously with retry logic.
    
    This is the async version of _perform_tavily_search. It uses async/await
    for non-blocking execution, allowing multiple searches to run concurrently.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        api_key: Tavily API key (optional, can use default)
        
    Returns:
        List of search result dictionaries
        
    Raises:
        CollectorError: If search fails after all retries
        Exception: Other exceptions from Tavily API
    """
    try:
        # Validate API key before attempting search
        if not api_key:
            raise CollectorError(
                "Tavily API key is required but not provided. "
                "Please set TAVILY_API_KEY in your .env file or environment variables.",
                context={"query": query, "max_results": max_results}
            )
        
        # Create Tavily search tool
        # Both TavilySearch and TavilySearchResults use 'tavily_api_key'
        search_tool = TAVILY_SEARCH_CLASS(
            max_results=max_results,
            tavily_api_key=api_key,
        )
        
        # Check if tool supports async
        if hasattr(search_tool, "ainvoke"):
            # Use async invoke if available
            raw_response = await search_tool.ainvoke(query)
        else:
            # Fall back to sync invoke in executor
            loop = asyncio.get_event_loop()
            raw_response = await loop.run_in_executor(None, search_tool.invoke, query)
        
        # Normalize response format: Tavily API evolution introduced breaking changes
        # Modern API (langchain_tavily >= 0.1.0): Returns dict with 'results' key containing array of result objects
        # Legacy API (langchain_community): Returns list of result objects directly
        # This abstraction ensures compatibility across API versions
        if isinstance(raw_response, dict) and "results" in raw_response:
            # Extract results array from modern API response wrapper
            results = raw_response.get("results", [])
        elif isinstance(raw_response, list):
            # Legacy format: response is already the results array
            results = raw_response
        else:
            # Edge case: handle unexpected response types by wrapping in list
            results = [raw_response] if raw_response else []
        
        formatted_results = []
        for result in results:
            if isinstance(result, str):
                # If result is a string, try to extract URL and content
                url_match = re.search(r'https?://[^\s]+', result)
                url = url_match.group(0) if url_match else ""
                lines = result.split('\n')
                title = lines[0][:100] if lines else result[:100]
                snippet = result
                
                formatted_results.append({
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "source": "tavily",
                })
            elif isinstance(result, dict):
                # Standard dictionary format from TavilySearchResults or TavilySearch
                formatted_results.append({
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "snippet": result.get("content", result.get("snippet", result.get("raw_content", ""))),
                    "source": "tavily",
                })
            else:
                logger.warning(f"Unexpected result format: {type(result)}, value: {result}")
                formatted_results.append({
                    "url": "",
                    "title": str(result)[:100],
                    "snippet": str(result),
                    "source": "tavily",
                })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Tavily search failed: {e}", exc_info=True)
        raise CollectorError(
            f"Web search failed for query: {query}",
            context={"query": query, "max_results": max_results, "error": str(e)}
        ) from e


async def web_search_async(query: str, max_results: int = 10) -> dict[str, Any]:
    """Search the web asynchronously for information about competitors.
    
    This is the async version of web_search. It performs a web search using
    Tavily search API and returns structured results. Multiple searches can
    be run concurrently using asyncio.gather().
    
    Args:
        query: Search query string describing what to search for
        max_results: Maximum number of search results to return (default: 10)
    
    Returns:
        Dictionary with the same structure as web_search():
        {
            "success": bool,
            "results": list[dict],
            "error": str,
            "query": str,
            "count": int
        }
    
    Example:
        ```python
        # Run multiple searches concurrently
        queries = ["query1", "query2", "query3"]
        results = await asyncio.gather(*[
            web_search_async(q, max_results=10)
            for q in queries
        ])
        ```
    """
    # Validate inputs (same as sync version)
    if not query or not query.strip():
        return {
            "success": False,
            "error": "Query cannot be empty",
            "results": [],
            "query": query,
            "count": 0,
        }
    
    if max_results < 1 or max_results > 20:
        return {
            "success": False,
            "error": f"max_results must be between 1 and 20, got {max_results}",
            "results": [],
            "query": query,
            "count": 0,
        }
    
    query = query.strip()
    
    try:
        # Get configuration
        config = get_config()
        api_key = config.tavily_api_key
        
        # Clean and validate API key
        if api_key:
            api_key = api_key.strip()
            if not api_key:
                api_key = None
        
        if not api_key:
            error_msg = (
                "TAVILY_API_KEY not configured. "
                "Please set TAVILY_API_KEY in your .env file or environment variables."
            )
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "results": [],
                "query": query,
                "count": 0,
            }
        
        logger.info(f"Performing async web search: query='{query}', max_results={max_results}")
        
        # Use async retry logic
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((CollectorError,)),
            reraise=True,
        ):
            with attempt:
                results = await _perform_tavily_search_async(query, max_results, api_key)
        
        results = results[:max_results]
        logger.info(f"Async web search successful: found {len(results)} results")
        
        return {
            "success": True,
            "results": results,
            "query": query,
            "count": len(results),
        }
        
    except CollectorError:
        raise
    except Exception as e:
        error_msg = f"Unexpected error during async web search: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            "success": False,
            "error": error_msg,
            "results": [],
            "query": query,
            "count": 0,
        }
