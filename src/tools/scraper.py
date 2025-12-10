"""Web scraper tool for extracting content from URLs.

This tool extracts text content from web pages, handles different content
types, and provides error handling for invalid URLs and timeouts. It's
designed to extract clean, readable text from competitor websites and
other sources.

Example:
    ```python
    from src.tools.scraper import scrape_url
    
    result = scrape_url.invoke({
        "url": "https://example.com/article",
        "timeout": 10
    })
    
    if result["success"]:
        print(f"Extracted {len(result['content'])} characters")
        print(result["title"])
    ```
"""

import logging
from typing import Any
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_config
from src.exceptions.collector_error import CollectorError

logger = logging.getLogger(__name__)


def _is_valid_url(url: str) -> bool:
    """Check if URL is valid and well-formed.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def _extract_text_from_html(html_content: str, url: str) -> dict[str, Any]:
    """Extract text content from HTML.
    
    Args:
        html_content: HTML content string
        url: Source URL for reference
        
    Returns:
        Dictionary with extracted content:
        - title: Page title
        - content: Main text content
        - links: List of links found on page
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
        script.decompose()
    
    # Extract main content
    # Try to find main content area
    main_content = soup.find("main") or soup.find("article") or soup.find("body")
    
    if main_content:
        content = main_content.get_text(separator=" ", strip=True)
    else:
        content = soup.get_text(separator=" ", strip=True)
    
    # Clean up whitespace
    content = " ".join(content.split())
    
    # Extract links
    links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        # Convert relative URLs to absolute
        absolute_url = urljoin(url, href)
        link_text = link.get_text(strip=True)
        if link_text and absolute_url:
            links.append({"url": absolute_url, "text": link_text})
    
    return {
        "title": title,
        "content": content,
        "links": links[:50],  # Limit to first 50 links
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.RequestException,)),
    reraise=True,
)
def _fetch_url_content(url: str, timeout: int) -> tuple[str, str]:
    """Fetch content from URL with retry logic.
    
    This internal function performs the actual HTTP request with retry
    logic using tenacity. It's separated from the tool function to allow
    proper retry decoration. Retries on requests.RequestException.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (content_type, content) where content_type is the
        Content-Type header value and content is the response body
        
    Raises:
        CollectorError: If request fails after all retries
        requests.RequestException: For HTTP-related errors (will be retried)
    """
    try:
        config = get_config()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        
        content_type = response.headers.get("Content-Type", "").lower()
        content = response.text
        
        return content_type, content
        
    except requests.Timeout as e:
        logger.warning(f"Timeout fetching URL {url} (attempt will be retried): {e}")
        # Re-raise to allow retry
        raise
    except requests.RequestException as e:
        logger.warning(f"Request failed for URL {url} (attempt will be retried): {e}")
        # Re-raise to allow retry
        raise
    except Exception as e:
        # Non-retryable errors - convert to CollectorError immediately
        logger.error(f"Unexpected error fetching URL {url}: {e}", exc_info=True)
        raise CollectorError(
            f"Unexpected error fetching URL: {url}",
            context={"url": url, "error": str(e)}
        ) from e


@tool
def scrape_url(url: str, timeout: int = 10) -> dict[str, Any]:
    """Scrape text content from a web page URL.
    
    This tool fetches a web page and extracts clean, readable text content
    from it. It handles HTML parsing, removes scripts and styles, and
    extracts the main content along with page title and links.
    
    The tool automatically handles retries for transient failures and timeouts.
    It returns structured data that can be easily processed by other agents
    in the workflow.
    
    Args:
        url: URL of the web page to scrape. Must be a valid HTTP/HTTPS URL.
            Examples:
            - "https://competitor.com/products"
            - "https://example.com/pricing"
        timeout: Request timeout in seconds. Default is 10 seconds.
            Range: 5-30 seconds.
    
    Returns:
        Dictionary with the following structure:
        {
            "success": bool,      # True if scraping succeeded
            "url": str,           # Original URL
            "title": str,         # Page title (if successful)
            "content": str,       # Extracted text content (if successful)
            "content_length": int, # Length of content in characters
            "links": [            # List of links found on page
                {
                    "url": str,   # Link URL
                    "text": str   # Link text
                },
                ...
            ],
            "error": str          # Error message (if failed)
        }
    
    Raises:
        CollectorError: If scraping fails after all retry attempts.
            This indicates a persistent failure that cannot be recovered.
    
    Example:
        ```python
        # Scrape a competitor's pricing page
        result = scrape_url.invoke({
            "url": "https://competitor.com/pricing",
            "timeout": 15
        })
        
        if result["success"]:
            print(f"Title: {result['title']}")
            print(f"Content length: {result['content_length']} characters")
            print(f"Found {len(result['links'])} links")
        else:
            print(f"Scraping failed: {result['error']}")
        ```
    
    Note:
        - Automatically retries up to 3 times on failure
        - Uses exponential backoff for retries
        - Respects timeout settings
        - Extracts clean text by removing scripts, styles, and navigation
        - Limits links to first 50 for performance
        - Handles both absolute and relative URLs
    """
    # Validate inputs
    if not url or not url.strip():
        return {
            "success": False,
            "error": "URL cannot be empty",
            "url": url,
            "title": "",
            "content": "",
            "content_length": 0,
            "links": [],
        }
    
    url = url.strip()
    
    if not _is_valid_url(url):
        return {
            "success": False,
            "error": f"Invalid URL format: {url}",
            "url": url,
            "title": "",
            "content": "",
            "content_length": 0,
            "links": [],
        }
    
    # Validate timeout
    if timeout < 5 or timeout > 30:
        return {
            "success": False,
            "error": f"Timeout must be between 5 and 30 seconds, got {timeout}",
            "url": url,
            "title": "",
            "content": "",
            "content_length": 0,
            "links": [],
        }
    
    try:
        logger.info(f"Scraping URL: {url}")
        
        # Fetch content with retry logic
        content_type, html_content = _fetch_url_content(url, timeout)
        
        # Check if content is HTML
        if "text/html" not in content_type:
            logger.warning(f"Content type is not HTML: {content_type}")
            # For non-HTML content, return as plain text
            return {
                "success": True,
                "url": url,
                "title": "",
                "content": html_content[:10000],  # Limit non-HTML content
                "content_length": len(html_content),
                "links": [],
            }
        
        # Extract text from HTML
        extracted = _extract_text_from_html(html_content, url)
        
        content_length = len(extracted["content"])
        logger.info(
            f"Successfully scraped URL {url}: "
            f"{content_length} characters, "
            f"{len(extracted['links'])} links"
        )
        
        return {
            "success": True,
            "url": url,
            "title": extracted["title"],
            "content": extracted["content"],
            "content_length": len(extracted["content"]),
            "links": extracted["links"],
        }
        
    except requests.Timeout as e:
        # Timeout after all retries
        error_msg = f"Timeout fetching URL after retries: {url}"
        logger.error(error_msg)
        raise CollectorError(
            error_msg,
            context={"url": url, "timeout": timeout, "error": str(e)}
        ) from e
    except requests.RequestException as e:
        # Request failed after all retries
        error_msg = f"Failed to fetch URL after retries: {url}"
        logger.error(error_msg)
        raise CollectorError(
            error_msg,
            context={
                "url": url,
                "error": str(e),
                "status_code": getattr(e.response, "status_code", None)
            }
        ) from e
    except CollectorError:
        # Re-raise CollectorError as-is
        raise
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error during scraping: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            "success": False,
            "error": error_msg,
            "url": url,
            "title": "",
            "content": "",
            "content_length": 0,
            "links": [],
        }
