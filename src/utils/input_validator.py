"""Input validation and sanitization utilities.

This module provides security-focused input validation and sanitization
functions to prevent security issues and improve robustness. It includes:
- User query validation and sanitization
- URL validation and sanitization
- File path validation (path traversal prevention)
- Character validation and length limits

All functions follow security best practices to prevent:
- Path traversal attacks
- Injection attacks
- Buffer overflow issues
- Invalid input handling
"""

import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from src.config import get_config
from src.exceptions.workflow_error import WorkflowError

logger = logging.getLogger(__name__)

# Dangerous characters that should be removed or escaped
DANGEROUS_CHARS = re.compile(r'[<>"\'\\\x00-\x1f\x7f-\x9f]')

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS = [
    r'\.\./',
    r'\.\.\\',
    r'\.\.',
    r'//',
    r'\\\\',
    r'%2e%2e',
    r'%2E%2E',
    r'\.\.%2f',
    r'\.\.%2F',
]

# Allowed URL schemes
ALLOWED_URL_SCHEMES = {'http', 'https'}


def sanitize_user_query(query: str, config: Any | None = None) -> str:
    """Validate and sanitize user input query.
    
    Validates user query against length limits and sanitizes dangerous
    characters. Raises WorkflowError if validation fails.
    
    Args:
        query: User input query string
        config: Optional Config instance. If not provided, uses get_config()
    
    Returns:
        Sanitized query string (trimmed, with dangerous chars removed)
    
    Raises:
        WorkflowError: If query is empty, too short, or too long
        ValueError: If config validation fails
    
    Example:
        ```python
        sanitized = sanitize_user_query("Analyze competitors in SaaS market")
        # Returns: "Analyze competitors in SaaS market"
        
        sanitized = sanitize_user_query("  Test query  ")
        # Returns: "Test query" (trimmed)
        ```
    """
    if config is None:
        config = get_config()
    
    # Trim whitespace first
    query = query.strip()
    
    # Check if empty after trimming
    if not query:
        raise WorkflowError(
            "User query cannot be empty",
            context={"query_length": 0}
        )
    
    # Check minimum length
    if len(query) < config.min_query_length:
        raise WorkflowError(
            f"User query is too short. Minimum length: {config.min_query_length} characters",
            context={
                "query_length": len(query),
                "min_length": config.min_query_length
            }
        )
    
    # Check maximum length
    if len(query) > config.max_query_length:
        raise WorkflowError(
            f"User query is too long. Maximum length: {config.max_query_length} characters",
            context={
                "query_length": len(query),
                "max_length": config.max_query_length
            }
        )
    
    # Remove dangerous characters (but preserve basic punctuation)
    # Remove control characters and potentially dangerous chars
    sanitized = DANGEROUS_CHARS.sub('', query)
    
    # Normalize whitespace (replace multiple spaces with single space)
    sanitized = re.sub(r'\s+', ' ', sanitized)
    sanitized = sanitized.strip()
    
    # Final length check after sanitization
    if len(sanitized) < config.min_query_length:
        raise WorkflowError(
            f"User query is too short after sanitization. Minimum length: {config.min_query_length} characters",
            context={
                "original_length": len(query),
                "sanitized_length": len(sanitized),
                "min_length": config.min_query_length
            }
        )
    
    logger.debug(
        f"Query sanitized: original_length={len(query)}, "
        f"sanitized_length={len(sanitized)}"
    )
    
    return sanitized


def validate_url(url: str, allow_localhost: bool = False) -> tuple[bool, str | None]:
    """Validate and sanitize URL.
    
    Validates URL format, scheme, and checks for potentially dangerous patterns.
    Returns validation result and sanitized URL.
    
    Args:
        url: URL string to validate
        allow_localhost: If True, allows localhost URLs (default: False)
    
    Returns:
        Tuple of (is_valid: bool, sanitized_url: str | None)
        - is_valid: True if URL is valid and safe
        - sanitized_url: Sanitized URL string, or None if invalid
    
    Example:
        ```python
        is_valid, sanitized = validate_url("https://example.com")
        # Returns: (True, "https://example.com")
        
        is_valid, sanitized = validate_url("javascript:alert(1)")
        # Returns: (False, None)
        ```
    """
    if not url or not url.strip():
        return False, None
    
    url = url.strip()
    
    try:
        parsed = urlparse(url)
        
        # Must have scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            logger.debug(f"URL validation failed: missing scheme or netloc: {url}")
            return False, None
        
        # Check allowed schemes only
        scheme = parsed.scheme.lower()
        if scheme not in ALLOWED_URL_SCHEMES:
            logger.debug(f"URL validation failed: disallowed scheme '{scheme}': {url}")
            return False, None
        
        # Check for localhost (if not allowed)
        if not allow_localhost:
            netloc_lower = parsed.netloc.lower()
            if netloc_lower.startswith('localhost') or netloc_lower.startswith('127.0.0.1'):
                logger.debug(f"URL validation failed: localhost not allowed: {url}")
                return False, None
        
        # Check for path traversal in path
        if parsed.path:
            for pattern in PATH_TRAVERSAL_PATTERNS:
                if re.search(pattern, parsed.path, re.IGNORECASE):
                    logger.debug(f"URL validation failed: path traversal detected: {url}")
                    return False, None
        
        # Reconstruct sanitized URL
        sanitized = urlunparse((
            scheme,
            parsed.netloc.lower(),  # Normalize hostname to lowercase
            parsed.path,
            parsed.params,
            parsed.query,
            ''  # Remove fragment for security
        ))
        
        # Additional length check (prevent extremely long URLs)
        if len(sanitized) > 2048:
            logger.debug(f"URL validation failed: URL too long: {len(sanitized)} characters")
            return False, None
        
        return True, sanitized
        
    except Exception as e:
        logger.debug(f"URL validation failed with exception: {e}, URL: {url}")
        return False, None


def validate_file_path(
    file_path: str | Path,
    base_dir: Path | None = None,
    allow_absolute: bool = False
) -> tuple[bool, Path | None, str | None]:
    """Validate file path and prevent path traversal attacks.
    
    Validates that a file path is safe and doesn't contain path traversal
    sequences. Optionally resolves paths relative to a base directory.
    
    Args:
        file_path: File path string or Path object to validate
        base_dir: Optional base directory. If provided, path is resolved relative to this.
            If None, uses current working directory.
        allow_absolute: If True, allows absolute paths (default: False)
    
    Returns:
        Tuple of (is_valid: bool, resolved_path: Path | None, error_message: str | None)
        - is_valid: True if path is valid and safe
        - resolved_path: Resolved Path object, or None if invalid
        - error_message: Error message if validation failed, or None if valid
    
    Example:
        ```python
        is_valid, path, error = validate_file_path("../etc/passwd")
        # Returns: (False, None, "Path traversal detected")
        
        is_valid, path, error = validate_file_path("data/report.pdf", base_dir=Path("/safe"))
        # Returns: (True, Path("/safe/data/report.pdf"), None)
        ```
    """
    if not file_path:
        return False, None, "File path cannot be empty"
    
    try:
        path = Path(file_path)
        
        # Check for path traversal patterns in string representation
        path_str = str(path)
        for pattern in PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, path_str, re.IGNORECASE):
                return False, None, f"Path traversal detected: {pattern}"
        
        # Check for absolute paths (if not allowed)
        if not allow_absolute and path.is_absolute():
            return False, None, "Absolute paths are not allowed"
        
        # Resolve relative to base directory if provided
        if base_dir:
            if not base_dir.is_dir():
                return False, None, f"Base directory does not exist: {base_dir}"
            
            # Resolve path relative to base directory
            resolved = (base_dir / path).resolve()
            
            # Ensure resolved path is still within base directory (prevent traversal)
            try:
                resolved.relative_to(base_dir.resolve())
            except ValueError:
                return False, None, "Path resolves outside base directory"
            
            return True, resolved, None
        else:
            # Resolve relative to current working directory
            resolved = path.resolve()
            
            # Check if resolved path is still relative (if absolute not allowed)
            if not allow_absolute and resolved.is_absolute():
                # Check if it's within current working directory
                try:
                    resolved.relative_to(Path.cwd().resolve())
                except ValueError:
                    return False, None, "Path resolves outside current directory"
            
            return True, resolved, None
            
    except Exception as e:
        logger.debug(f"File path validation failed with exception: {e}, path: {file_path}")
        return False, None, f"Invalid file path: {str(e)}"


def validate_and_sanitize_url(url: str, allow_localhost: bool = False) -> str:
    """Validate and sanitize URL, raising exception if invalid.
    
    Convenience function that validates URL and raises WorkflowError if invalid.
    Returns sanitized URL if valid.
    
    Args:
        url: URL string to validate and sanitize
        allow_localhost: If True, allows localhost URLs (default: False)
    
    Returns:
        Sanitized URL string
    
    Raises:
        WorkflowError: If URL is invalid or unsafe
    
    Example:
        ```python
        sanitized = validate_and_sanitize_url("https://example.com")
        # Returns: "https://example.com"
        
        validate_and_sanitize_url("javascript:alert(1)")
        # Raises: WorkflowError
        ```
    """
    is_valid, sanitized = validate_url(url, allow_localhost=allow_localhost)
    
    if not is_valid:
        raise WorkflowError(
            f"Invalid or unsafe URL: {url}",
            context={"url": url, "allow_localhost": allow_localhost}
        )
    
    return sanitized

