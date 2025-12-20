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

import logging
from typing import Any

try:
    # Use TavilySearchResults (works reliably)
    # Note: TavilySearch has different return format, so we stick with TavilySearchResults
    from langchain_tavily import TavilySearchResults
    TAVILY_SEARCH_CLASS = TavilySearchResults
except ImportError:
    # Fallback to langchain_community if langchain_tavily is not available
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_SEARCH_CLASS = TavilySearchResults
from langchain_core.tools import tool
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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
        # Use the appropriate class (TavilySearch or TavilySearchResults)
        # Both expect 'tavily_api_key' parameter
        search_tool = TAVILY_SEARCH_CLASS(
            max_results=max_results,
            tavily_api_key=api_key,
        )
        
        # Perform search
        results = search_tool.invoke(query)
        
        # Handle different return formats from TavilySearch vs TavilySearchResults
        # TavilySearch may return a single string or list of strings
        # TavilySearchResults returns a list of dicts
        if not isinstance(results, list):
            # If results is not a list, convert it to a list
            results = [results] if results else []
        
        # Handle different return formats
        formatted_results = []
        for result in results:
            if isinstance(result, str):
                # If result is a string, try to parse it or create a basic entry
                # This handles the case where TavilySearch returns strings
                formatted_results.append({
                    "url": "",
                    "title": result[:100] if len(result) > 100 else result,
                    "snippet": result,
                    "source": "tavily",
                })
            elif isinstance(result, dict):
                # Standard dictionary format from TavilySearchResults
                formatted_results.append({
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "snippet": result.get("content", result.get("snippet", "")),
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
