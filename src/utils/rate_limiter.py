"""Rate limiting and retry logic for LLM API calls.

This module provides decorators and utilities for handling rate limits
and transient failures in LLM API calls using exponential backoff.

The rate limiter uses the tenacity library to implement retry logic with
configurable attempts and backoff strategies. It detects rate limit errors
and handles them gracefully with exponential backoff.
"""

import logging
from functools import wraps
from typing import Any, Callable, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    after_log,
    RetryCallState,
)

from src.config import get_config
from src.exceptions.workflow_error import WorkflowError
from src.utils.llm_cache import cached_llm_call

logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar("T")


def _is_rate_limit_error(exception: Exception) -> bool:
    """Check if exception is a rate limit error.
    
    Detects various forms of rate limit errors from LLM APIs:
    - HTTP 429 (Too Many Requests)
    - Rate limit exceptions from LangChain/Groq
    - Quota exceeded errors
    
    Args:
        exception: Exception to check
        
    Returns:
        True if exception appears to be a rate limit error, False otherwise
    """
    error_str = str(exception).lower()
    error_type = type(exception).__name__.lower()
    
    # Check for common rate limit indicators
    rate_limit_indicators = [
        "rate limit",
        "rate_limit",
        "429",
        "too many requests",
        "quota exceeded",
        "quota_exceeded",
        "throttle",
        "throttled",
    ]
    
    # Check error string
    if any(indicator in error_str for indicator in rate_limit_indicators):
        return True
    
    # Check error type name
    if any(indicator in error_type for indicator in rate_limit_indicators):
        return True
    
    # Check for HTTP status code 429
    if hasattr(exception, "status_code") and exception.status_code == 429:
        return True
    
    if hasattr(exception, "response"):
        response = exception.response
        if hasattr(response, "status_code") and response.status_code == 429:
            return True
    
    return False


class RateLimitError(Exception):
    """Exception raised when rate limit is detected.
    
    This exception is used internally to trigger retry logic for rate limit errors.
    It wraps the original exception to provide consistent handling.
    """
    
    def __init__(self, original_exception: Exception) -> None:
        """Initialize rate limit error.
        
        Args:
            original_exception: The original exception that was identified as a rate limit error
        """
        # Use 'from' to preserve exception chain
        super().__init__(f"Rate limit error: {original_exception}")
        self.original_exception = original_exception


def retry_llm_call(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for retrying LLM API calls with exponential backoff.
    
    This decorator wraps LLM API calls with retry logic using tenacity.
    It handles rate limit errors and transient failures with exponential backoff.
    
    The decorator:
    - Retries on rate limit errors and general exceptions
    - Uses exponential backoff with configurable min/max wait times
    - Logs retry attempts for debugging
    - Raises WorkflowError after max retries exceeded
    
    Configuration is loaded from Config:
    - llm_retry_attempts: Maximum number of retry attempts (default: 3)
    - llm_retry_backoff_min: Minimum backoff time in seconds (default: 1.0)
    - llm_retry_backoff_max: Maximum backoff time in seconds (default: 30.0)
    
    Example:
        @retry_llm_call
        def call_llm(messages):
            return llm.invoke(messages)
    
    Args:
        func: Function to wrap with retry logic. Should be a function that
            makes LLM API calls and may raise exceptions on rate limits or failures.
    
    Returns:
        Wrapped function with retry logic applied
    """
    config = get_config()
    
    def retry_condition(retry_state: RetryCallState) -> bool:
        """Custom retry condition that handles rate limit errors."""
        if retry_state.outcome is None:
            return False
        
        exception = retry_state.outcome.exception()
        if exception is None:
            return False
        
        # Always retry on exceptions (tenacity will stop after max attempts)
        # Log rate limit errors specifically
        if isinstance(exception, RateLimitError):
            logger.warning(
                f"Rate limit error in {func.__name__}, retrying (attempt {retry_state.attempt_number})"
            )
            return True
        
        if _is_rate_limit_error(exception):
            logger.warning(
                f"Rate limit detected in {func.__name__}, retrying (attempt {retry_state.attempt_number})"
            )
            return True
        
        # Retry on any exception (tenacity will stop after max attempts)
        logger.debug(
            f"Exception in {func.__name__}, retrying (attempt {retry_state.attempt_number}): "
            f"{type(exception).__name__}: {str(exception)[:200]}"
        )
        return True
    
    @wraps(func)
    @retry(
        stop=stop_after_attempt(config.llm_retry_attempts),
        wait=wait_exponential(
            multiplier=1,
            min=config.llm_retry_backoff_min,
            max=config.llm_retry_backoff_max,
        ),
        retry=retry_condition,
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.DEBUG),
    )
    def wrapper(*args: Any, **kwargs: Any) -> T:
        """Wrapper function with retry logic.
        
        This wrapper attempts to call the original function and handles
        exceptions by checking if they are rate limit errors. If so,
        it allows tenacity to retry with exponential backoff.
        
        Args:
            *args: Positional arguments to pass to the original function
            **kwargs: Keyword arguments to pass to the original function
        
        Returns:
            Return value from the original function
        
        Raises:
            WorkflowError: If all retries are exhausted and the last attempt failed
            Exception: The original exception if it's not a rate limit error
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Check if this is a rate limit error and wrap it
            if _is_rate_limit_error(e) and not isinstance(e, RateLimitError):
                # Wrap rate limit errors for consistent handling
                raise RateLimitError(e) from e
            # Re-raise to trigger tenacity retry
            raise
    
    return wrapper


def invoke_llm_with_retry(
    llm: Any,
    messages: list[Any],
    **kwargs: Any
) -> Any:
    """Invoke LLM with automatic retry logic and caching.
    
    This is a convenience function that wraps llm.invoke() calls with
    retry logic and caching. It's useful when you want to use retry logic
    without decorating the entire function.
    
    The function first checks the cache (if enabled) before making an API call.
    If a cache hit occurs, the cached response is returned immediately without
    retry logic. If a cache miss occurs, the LLM is called with retry logic.
    
    Args:
        llm: Language model instance (BaseChatModel)
        messages: List of messages to send to the LLM
        **kwargs: Additional keyword arguments to pass to llm.invoke()
    
    Returns:
        Response from LLM (either from cache or new API call)
    
    Raises:
        WorkflowError: If all retries are exhausted
        Exception: Original exception if not retryable
    """
    config = get_config()
    
    # Create a retry-wrapped invoke function
    def retry_invoke(wrapped_llm: Any, msgs: list[Any], kw: dict[str, Any]) -> Any:
        """Invoke LLM with retry logic."""
        @retry_llm_call
        def _invoke() -> Any:
            return wrapped_llm.invoke(msgs, **kw)
        
        try:
            return _invoke()
        except RateLimitError as e:
            # After all retries exhausted, wrap in WorkflowError
            raise WorkflowError(
                "LLM API rate limit exceeded after retries",
                context={
                    "error": str(e.original_exception),
                    "retry_attempts": config.llm_retry_attempts,
                }
            ) from e.original_exception
        except Exception as e:
            # For other exceptions after retries, wrap in WorkflowError
            raise WorkflowError(
                "LLM API call failed after retries",
                context={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "retry_attempts": config.llm_retry_attempts,
                }
            ) from e
    
    # Use cached_llm_call with retry-wrapped invoke function
    # This will check cache first, and if miss, use retry_invoke
    return cached_llm_call(llm, messages, invoke_func=retry_invoke, **kwargs)


async def invoke_llm_with_retry_async(
    llm: Any,
    messages: list[Any],
    **kwargs: Any
) -> Any:
    """Invoke LLM asynchronously with automatic retry logic.
    
    This is the async version of invoke_llm_with_retry. It uses llm.ainvoke()
    instead of llm.invoke() for non-blocking execution.
    
    Note: Caching is not yet supported for async calls. This will be added
    in a future update when async cache support is implemented.
    
    Args:
        llm: Language model instance (BaseChatModel)
        messages: List of messages to send to the LLM
        **kwargs: Additional keyword arguments to pass to llm.ainvoke()
    
    Returns:
        Response from LLM
    
    Raises:
        WorkflowError: If all retries are exhausted
        Exception: Original exception if not retryable
    
    Note:
        This function requires the LLM to support async operations (ainvoke method).
        If async is not supported, it falls back to sync execution in an executor.
    """
    import asyncio
    from tenacity.asyncio import AsyncRetrying
    from tenacity import (
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
    
    config = get_config()
    
    # Check if LLM supports async
    if not hasattr(llm, "ainvoke"):
        # Fall back to sync version wrapped in async
        logger.warning("LLM does not support async, falling back to sync execution")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: invoke_llm_with_retry(llm, messages, **kwargs)
        )
    
    # Create async retry wrapper
    async def _async_invoke() -> Any:
        """Async invoke with retry logic."""
        try:
            # Use ainvoke for async execution
            return await llm.ainvoke(messages, **kwargs)
        except Exception as e:
            # Check if this is a rate limit error and wrap it
            if _is_rate_limit_error(e) and not isinstance(e, RateLimitError):
                raise RateLimitError(e) from e
            raise
    
    try:
        # Use AsyncRetrying for async retry logic
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(config.llm_retry_attempts),
            wait=wait_exponential(
                multiplier=1,
                min=config.llm_retry_backoff_min,
                max=config.llm_retry_backoff_max,
            ),
            retry=retry_if_exception_type((RateLimitError, Exception)),
            reraise=True,
        ):
            with attempt:
                try:
                    return await _async_invoke()
                except Exception as e:
                    # Check if this is a rate limit error
                    if _is_rate_limit_error(e) and not isinstance(e, RateLimitError):
                        raise RateLimitError(e) from e
                    # Re-raise to trigger retry logic
                    raise
        
        # This should never be reached, but type checker needs it
        raise WorkflowError("Unexpected error in async retry logic")
        
    except RateLimitError as e:
        # After all retries exhausted, wrap in WorkflowError
        raise WorkflowError(
            "LLM API rate limit exceeded after retries",
            context={
                "error": str(e.original_exception),
                "retry_attempts": config.llm_retry_attempts,
            }
        ) from e.original_exception
    except Exception as e:
        # For other exceptions after retries, wrap in WorkflowError
        raise WorkflowError(
            "LLM API call failed after retries",
            context={
                "error": str(e),
                "error_type": type(e).__name__,
                "retry_attempts": config.llm_retry_attempts,
            }
        ) from e

