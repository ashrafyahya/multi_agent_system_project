"""In-memory caching for LLM responses.

This module provides caching functionality for LLM API calls to avoid redundant
API calls for similar inputs. It uses functools.lru_cache with a custom cache
key generation function that creates hashable keys from messages and kwargs.

The cache is designed to:
- Cache LLM responses based on message content and parameters
- Provide cache statistics (hits, misses, size)
- Support cache invalidation
- Be configurable via Config (enabled/disabled, cache size)
"""

import hashlib
import json
import logging
from functools import lru_cache, wraps
from typing import Any, Callable, TypeVar

from src.config import get_config

logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar("T")

# Global cache statistics
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "size": 0,
}


def _serialize_message(message: Any) -> str:
    """Serialize a message object to a string representation.
    
    Handles different message types from LangChain:
    - HumanMessage, AIMessage, SystemMessage
    - Messages with content attribute
    - Plain strings
    
    Args:
        message: Message object to serialize
        
    Returns:
        String representation of the message
    """
    # If it's already a string, return it
    if isinstance(message, str):
        return message
    
    # Try to get content attribute
    if hasattr(message, "content"):
        content = message.content
        if isinstance(content, str):
            return content
        # If content is a list (e.g., for multimodal), serialize it
        if isinstance(content, list):
            return json.dumps(content, sort_keys=True)
    
    # Try to get type and content
    message_type = getattr(message, "type", None) or type(message).__name__
    content = getattr(message, "content", None) or str(message)
    
    # Create a serializable representation
    return json.dumps(
        {
            "type": message_type,
            "content": content if isinstance(content, str) else str(content),
        },
        sort_keys=True,
    )


def get_cache_key(messages: list[Any], **kwargs: Any) -> str:
    """Generate a cache key from messages and kwargs.
    
    Creates a deterministic hash key from the message content and any
    relevant kwargs (like temperature, model name, etc.) that affect
    the LLM response.
    
    The cache key is based on:
    - Message content (serialized)
    - Relevant kwargs that affect output (temperature, max_tokens, etc.)
    - Model name if provided in kwargs
    
    Args:
        messages: List of messages to send to the LLM
        **kwargs: Additional keyword arguments that affect the response
        
    Returns:
        SHA256 hash string representing the cache key
    """
    # Serialize messages
    serialized_messages = [_serialize_message(msg) for msg in messages]
    
    # Filter kwargs to only include those that affect LLM output
    # Common parameters: temperature, max_tokens, top_p, etc.
    relevant_kwargs = {}
    relevant_params = {
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "model",
    }
    
    for key, value in kwargs.items():
        if key in relevant_params:
            # Convert value to string for hashing
            if isinstance(value, (list, tuple)):
                relevant_kwargs[key] = tuple(sorted(str(v) for v in value))
            else:
                relevant_kwargs[key] = str(value)
    
    # Create a deterministic string representation
    cache_data = {
        "messages": serialized_messages,
        "kwargs": sorted(relevant_kwargs.items()),
    }
    
    # Convert to JSON string and hash
    cache_string = json.dumps(cache_data, sort_keys=True)
    cache_key = hashlib.sha256(cache_string.encode("utf-8")).hexdigest()
    
    return cache_key


def cached_llm_call(
    llm: Any,
    messages: list[Any],
    invoke_func: Callable[[Any, list[Any], dict[str, Any]], Any] | None = None,
    **kwargs: Any
) -> Any:
    """Cached wrapper for LLM calls.
    
    This function wraps LLM invoke calls with caching. If caching is enabled
    in config, it will check the cache first before making an API call.
    
    The cache key is generated from messages and relevant kwargs. Cache hits
    return the cached response immediately without making an API call.
    
    Args:
        llm: Language model instance (BaseChatModel)
        messages: List of messages to send to the LLM
        invoke_func: Optional callable to use for actual LLM invocation.
            If provided, this function will be called instead of llm.invoke().
            Signature: invoke_func(llm, messages, kwargs) -> response
            If None, defaults to llm.invoke(messages, **kwargs)
        **kwargs: Additional keyword arguments to pass to llm.invoke()
    
    Returns:
        Response from LLM (either from cache or new API call)
    
    Note:
        This function uses a global LRU cache. The cache size is configured
        via Config.llm_cache_size. Cache statistics are tracked globally.
    """
    config = get_config()
    
    # Default invoke function
    if invoke_func is None:
        def default_invoke(llm: Any, msgs: list[Any], kw: dict[str, Any]) -> Any:
            return llm.invoke(msgs, **kw)
        invoke_func = default_invoke
    
    # If caching is disabled, call LLM directly
    if not config.llm_cache_enabled:
        logger.debug("LLM cache is disabled, making direct API call")
        return invoke_func(llm, messages, kwargs)
    
    # Generate cache key
    cache_key = get_cache_key(messages, **kwargs)
    
    # Store call parameters for the cached function to use
    _llm_storage[cache_key] = llm
    _messages_storage[cache_key] = messages
    _kwargs_storage[cache_key] = kwargs
    # Store invoke function
    if "_invoke_func_storage" not in globals():
        globals()["_invoke_func_storage"] = {}
    globals()["_invoke_func_storage"][cache_key] = invoke_func
    
    # Get cached function (created on first call)
    cached_func = _get_cached_llm_func(config.llm_cache_size)
    
    # Check if this is a cache hit by checking cache_info
    cache_info_before = cached_func.cache_info()
    
    try:
        # Call cached function - it will use the stored parameters
        # If this raises an exception, it will propagate (not cached)
        result = cached_func(cache_key)
        
        # Check if it was a cache hit
        cache_info_after = cached_func.cache_info()
        if cache_info_after.hits > cache_info_before.hits:
            _cache_stats["hits"] += 1
            logger.debug(f"LLM cache hit for key: {cache_key[:16]}...")
        else:
            _cache_stats["misses"] += 1
            logger.debug(f"LLM cache miss for key: {cache_key[:16]}...")
        
        return result
    except (KeyError, Exception):
        # Cache miss or exception - make direct call
        # Exceptions from invoke_func will propagate normally
        # KeyError means cache entry not found (shouldn't happen, but handle it)
        _cache_stats["misses"] += 1
        logger.debug(f"LLM cache miss for key: {cache_key[:16]}...")
        # Re-raise if it's not a KeyError (let exceptions propagate)
        try:
            return invoke_func(llm, messages, kwargs)
        except Exception:
            # Exception occurred - don't cache it, just propagate
            raise


# Module-level storage for call parameters
# These are keyed by cache_key and used by the cached function
_llm_storage: dict[str, Any] = {}
_messages_storage: dict[str, list[Any]] = {}
_kwargs_storage: dict[str, dict[str, Any]] = {}
_invoke_func_storage: dict[str, Callable[[Any, list[Any], dict[str, Any]], Any]] = {}

# Global variable to store the cached function
_cached_llm_func: Callable[[str], Any] | None = None
_cache_size: int = 128


def _get_cached_llm_func(cache_size: int) -> Callable[[str], Any]:
    """Get or create the cached LLM function.
    
    This function creates a cached wrapper for LLM calls. The cache is
    keyed by cache_key (string), which is a hash of messages and kwargs.
    
    Args:
        cache_size: Maximum size of the LRU cache
        
    Returns:
        Cached function that wraps LLM invoke calls
    """
    global _cached_llm_func, _cache_size
    
    # Recreate cache if size changed
    if _cached_llm_func is None or _cache_size != cache_size:
        _cache_size = cache_size
        
        @lru_cache(maxsize=cache_size)
        def _cached_invoke(cache_key: str) -> Any:
            """Cached LLM invoke function.
            
            This function is wrapped with lru_cache. The cache key is
            the cache_key string, which uniquely identifies the call.
            
            Note: Only successful responses are cached. Exceptions are
            not cached and will propagate normally.
            """
            # Retrieve llm, messages, kwargs, and invoke_func from module-level storage
            llm = _llm_storage.get(cache_key)
            messages = _messages_storage.get(cache_key)
            kwargs = _kwargs_storage.get(cache_key, {})
            invoke_func = _invoke_func_storage.get(cache_key)
            
            if llm is None or messages is None:
                # This shouldn't happen, but handle it gracefully
                raise KeyError(f"Cache entry not found for key: {cache_key[:16]}...")
            
            # Use stored invoke_func if available, otherwise default to llm.invoke
            # If invoke_func raises an exception, it will propagate (not cached)
            if invoke_func is not None:
                result = invoke_func(llm, messages, kwargs)
            else:
                result = llm.invoke(messages, **kwargs)
            
            # Only return result if no exception was raised
            # Exceptions will propagate and prevent caching
            return result
        
        _cached_llm_func = _cached_invoke
    
    return _cached_llm_func


def get_cache_stats() -> dict[str, int]:
    """Get cache statistics.
    
    Returns statistics about cache usage:
    - hits: Number of cache hits
    - misses: Number of cache misses
    - size: Current cache size (number of entries)
    
    Returns:
        Dictionary with cache statistics
    """
    global _cache_stats, _cached_llm_func
    
    # Get actual cache size from lru_cache
    if _cached_llm_func is not None:
        cache_info = _cached_llm_func.cache_info()
        _cache_stats["size"] = cache_info.currsize
        _cache_stats["hits"] = cache_info.hits
        _cache_stats["misses"] = cache_info.misses
    
    return _cache_stats.copy()


def clear_cache() -> None:
    """Clear the LLM response cache.
    
    This function clears all cached LLM responses and resets cache statistics.
    Useful for testing or when you want to force fresh API calls.
    """
    global _cached_llm_func, _cache_stats, _llm_storage, _messages_storage, _kwargs_storage, _invoke_func_storage
    
    if _cached_llm_func is not None:
        _cached_llm_func.cache_clear()
    
    # Clear storage dictionaries
    _llm_storage.clear()
    _messages_storage.clear()
    _kwargs_storage.clear()
    _invoke_func_storage.clear()
    
    _cache_stats = {
        "hits": 0,
        "misses": 0,
        "size": 0,
    }
    
    logger.info("LLM cache cleared")


def invalidate_cache_entry(messages: list[Any], **kwargs: Any) -> bool:
    """Invalidate a specific cache entry.
    
    This function attempts to invalidate a specific cache entry by generating
    its cache key. However, since lru_cache doesn't support selective invalidation,
    this function clears the entire cache if the entry exists.
    
    Note:
        Due to limitations of lru_cache, this function clears the entire cache
        rather than a single entry. For selective invalidation, consider using
        a custom cache implementation.
    
    Args:
        messages: List of messages that were used for the cached call
        **kwargs: Keyword arguments that were used for the cached call
        
    Returns:
        True if cache was cleared, False otherwise
    """
    global _cached_llm_func
    
    if _cached_llm_func is None:
        return False
    
    # Since lru_cache doesn't support selective invalidation,
    # we clear the entire cache
    clear_cache()
    logger.debug("Cache invalidated (full clear)")
    return True

