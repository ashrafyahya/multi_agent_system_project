"""Text processing utilities.

This module provides text cleaning, deduplication, summarization helpers,
and URL validation functions. These utilities are used throughout the
system to process and clean text data from various sources.

Example:
    ```python
    from src.tools.text_utils import clean_text, deduplicate_texts, validate_url
    
    # Clean text
    cleaned = clean_text("  Dirty   text  with   extra   spaces  ")
    # Returns: "Dirty text with extra spaces"
    
    # Deduplicate
    texts = ["text1", "text1", "text2"]
    unique = deduplicate_texts(texts)
    # Returns: ["text1", "text2"]
    
    # Validate URL
    is_valid = validate_url("https://example.com")
    # Returns: True
    ```
"""

import re
from typing import Any
from urllib.parse import urlparse


def clean_text(text: str, preserve_newlines: bool = False) -> str:
    """Clean and normalize text content.
    
    Removes extra whitespace, normalizes line breaks, and cleans up
    common text artifacts. This function is useful for cleaning scraped
    content or user input.
    
    Args:
        text: Text string to clean
        preserve_newlines: If True, preserves newline characters.
            If False, converts newlines to spaces. Default: False
    
    Returns:
        Cleaned text string
        
    Example:
        ```python
        dirty = "  Text   with   extra   spaces  \n\n  "
        clean = clean_text(dirty)
        # Returns: "Text with extra spaces"
        ```
    """
    if not text:
        return ""
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    if preserve_newlines:
        # Normalize line breaks (multiple newlines -> single newline)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Clean up spaces around newlines
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n +', '\n', text)
        # Normalize spaces within lines
        text = re.sub(r'[ \t]+', ' ', text)
    else:
        # Convert all whitespace to single spaces
        text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def deduplicate_texts(texts: list[str], case_sensitive: bool = False) -> list[str]:
    """Remove duplicate text entries from a list.
    
    Removes duplicate strings from a list while preserving order.
    Optionally performs case-insensitive deduplication.
    
    Args:
        texts: List of text strings to deduplicate
        case_sensitive: If False, treats "Text" and "text" as duplicates.
            Default: False
    
    Returns:
        List of unique text strings in original order
        
    Example:
        ```python
        texts = ["text1", "Text1", "text2", "text1"]
        unique = deduplicate_texts(texts)
        # Returns: ["text1", "text2"] (case-insensitive)
        ```
    """
    if not texts:
        return []
    
    seen: set[str] = set()
    unique_texts: list[str] = []
    
    for text in texts:
        if not text:
            continue
        
        # Create key for comparison
        key = text if case_sensitive else text.lower()
        
        if key not in seen:
            seen.add(key)
            unique_texts.append(text)
    
    return unique_texts


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to a maximum length.
    
    Truncates text to the specified maximum length and appends a suffix
    if truncation occurred. Preserves word boundaries when possible.
    
    Args:
        text: Text string to truncate
        max_length: Maximum length of the returned string (including suffix)
        suffix: Suffix to append if text is truncated. Default: "..."
    
    Returns:
        Truncated text string
        
    Example:
        ```python
        long_text = "This is a very long text that needs to be truncated"
        short = truncate_text(long_text, 20)
        # Returns: "This is a very lo..."
        ```
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    # Try to truncate at word boundary
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.7:  # If space is reasonably close
        truncated = truncated[:last_space]
    
    return truncated + suffix


def extract_sentences(text: str, max_sentences: int | None = None) -> list[str]:
    """Extract sentences from text.
    
    Splits text into sentences using simple sentence boundary detection.
    Handles common sentence endings (. ! ?) and filters empty sentences.
    
    Args:
        text: Text string to extract sentences from
        max_sentences: Maximum number of sentences to return.
            If None, returns all sentences. Default: None
    
    Returns:
        List of sentence strings
        
    Example:
        ```python
        text = "First sentence. Second sentence! Third sentence?"
        sentences = extract_sentences(text, max_sentences=2)
        # Returns: ["First sentence.", "Second sentence!"]
        ```
    """
    if not text:
        return []
    
    # Simple sentence splitting (can be improved with NLP libraries)
    # Split on sentence endings followed by space or end of string
    sentences = re.split(r'([.!?]+(?:\s+|$))', text)
    
    # Combine sentence with its punctuation
    result: list[str] = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = (sentences[i] + sentences[i + 1]).strip()
        else:
            sentence = sentences[i].strip()
        
        if sentence:
            result.append(sentence)
    
    # Handle last sentence if no punctuation
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())
    
    if max_sentences is not None:
        result = result[:max_sentences]
    
    return result


def summarize_text(text: str, max_length: int = 200) -> str:
    """Create a simple summary of text.
    
    Creates a summary by truncating text to a maximum length, attempting
    to preserve sentence boundaries. This is a simple summarization
    method; for better results, use LLM-based summarization.
    
    Args:
        text: Text string to summarize
        max_length: Maximum length of the summary. Default: 200
    
    Returns:
        Summarized text string
        
    Example:
        ```python
        long_text = "This is a very long article with many sentences..."
        summary = summarize_text(long_text, max_length=100)
        # Returns: First ~100 characters preserving sentences
        ```
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    # Extract sentences and build summary
    sentences = extract_sentences(text)
    
    summary_parts: list[str] = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence) + 1  # +1 for space
        
        if current_length + sentence_length <= max_length:
            summary_parts.append(sentence)
            current_length += sentence_length
        else:
            break
    
    if summary_parts:
        summary = " ".join(summary_parts)
        # If we have room, add a bit more
        if len(summary) < max_length * 0.8 and len(sentences) > len(summary_parts):
            # Try to add part of next sentence
            remaining = max_length - len(summary) - 4  # Reserve for "..."
            if remaining > 20:
                next_sentence = sentences[len(summary_parts)]
                summary += " " + truncate_text(next_sentence, remaining, "...")
        return summary
    
    # Fallback to simple truncation
    return truncate_text(text, max_length)


def validate_url(url: str) -> bool:
    """Validate URL format.
    
    Checks if a string is a valid URL format. Validates scheme and netloc.
    Does not check if the URL is accessible.
    
    Args:
        url: URL string to validate
    
    Returns:
        True if URL format is valid, False otherwise
        
    Example:
        ```python
        is_valid = validate_url("https://example.com")
        # Returns: True
        
        is_valid = validate_url("not-a-url")
        # Returns: False
        ```
    """
    if not url or not url.strip():
        return False
    
    try:
        result = urlparse(url.strip())
        # Must have scheme and netloc
        return bool(result.scheme and result.netloc)
    except Exception:
        return False


def normalize_url(url: str) -> str | None:
    """Normalize URL format.
    
    Normalizes a URL by adding scheme if missing, lowercasing hostname,
    and removing default ports. Returns None if URL is invalid.
    
    Args:
        url: URL string to normalize
    
    Returns:
        Normalized URL string, or None if URL is invalid
        
    Example:
        ```python
        normalized = normalize_url("example.com")
        # Returns: "http://example.com"
        
        normalized = normalize_url("HTTPS://EXAMPLE.COM:443")
        # Returns: "https://example.com"
        ```
    """
    if not url or not url.strip():
        return None
    
    url = url.strip()
    
    # Check if URL already has a scheme (case-insensitive)
    url_lower = url.lower()
    has_scheme = url_lower.startswith(('http://', 'https://'))
    
    # Add scheme if missing
    if not has_scheme:
        url = 'http://' + url
    
    try:
        parsed = urlparse(url)
        
        # Validate URL - must have scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            return None
        
        # Additional validation: if we added a scheme, check if netloc is valid
        # If netloc doesn't look like a valid hostname, return None
        if not has_scheme:
            # Check if netloc looks like a valid hostname (contains at least one dot or is localhost)
            if '.' not in parsed.netloc and parsed.netloc.lower() not in ('localhost', 'localhost.localdomain'):
                # If it's not a valid hostname, return None
                return None
        
        # Normalize scheme and hostname
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        
        # Remove default ports
        if ':' in netloc:
            host, port = netloc.rsplit(':', 1)
            if (scheme == 'http' and port == '80') or (scheme == 'https' and port == '443'):
                netloc = host
        
        # Reconstruct URL
        normalized = f"{scheme}://{netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        if parsed.fragment:
            normalized += f"#{parsed.fragment}"
        
        return normalized
        
    except Exception:
        return None


def extract_urls(text: str) -> list[str]:
    """Extract URLs from text.
    
    Finds and extracts URLs from a text string using regex pattern matching.
    Returns URLs in the order they appear in the text.
    
    Args:
        text: Text string to extract URLs from
    
    Returns:
        List of URL strings found in the text
        
    Example:
        ```python
        text = "Visit https://example.com and http://test.com for more info"
        urls = extract_urls(text)
        # Returns: ["https://example.com", "http://test.com"]
        ```
    """
    if not text:
        return []
    
    # URL pattern: http:// or https:// followed by valid URL characters
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    
    urls = re.findall(url_pattern, text)
    
    # Filter to valid URLs
    valid_urls = [url for url in urls if validate_url(url)]
    
    return valid_urls


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text.
    
    Removes HTML/XML tags from text using regex. This is a simple
    implementation; for better results with malformed HTML, use BeautifulSoup.
    
    Args:
        text: Text string containing HTML tags
    
    Returns:
        Text string with HTML tags removed
        
    Example:
        ```python
        html = "<p>This is <b>bold</b> text</p>"
        clean = remove_html_tags(html)
        # Returns: "This is bold text"
        ```
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode common HTML entities
    html_entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&nbsp;': ' ',
    }
    
    for entity, char in html_entities.items():
        text = text.replace(entity, char)
    
    return text.strip()
