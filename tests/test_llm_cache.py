"""Tests for LLM response caching functionality.

This module tests the caching system for LLM API calls, including:
- Cache key generation
- Cache hit/miss behavior
- Cache invalidation
- Cache size limits
- Cache statistics
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.config import Config, reload_config
from src.utils.llm_cache import (cached_llm_call, clear_cache, get_cache_key,
                                 get_cache_stats, invalidate_cache_entry)


class TestCacheKeyGeneration:
    """Test cache key generation from messages and kwargs."""
    
    def test_get_cache_key_with_simple_messages(self):
        """Test cache key generation with simple string messages."""
        messages = ["Hello", "World"]
        key1 = get_cache_key(messages)
        key2 = get_cache_key(messages)
        
        # Same messages should generate same key
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 64  # SHA256 hex digest length
    
    def test_get_cache_key_with_different_messages(self):
        """Test that different messages generate different keys."""
        messages1 = ["Hello"]
        messages2 = ["World"]
        
        key1 = get_cache_key(messages1)
        key2 = get_cache_key(messages2)
        
        assert key1 != key2
    
    def test_get_cache_key_with_message_objects(self):
        """Test cache key generation with message objects."""
        # Create mock message objects
        msg1 = Mock()
        msg1.content = "Hello"
        msg1.type = "human"
        
        msg2 = Mock()
        msg2.content = "World"
        msg2.type = "ai"
        
        messages = [msg1, msg2]
        key = get_cache_key(messages)
        
        assert isinstance(key, str)
        assert len(key) == 64
    
    def test_get_cache_key_with_kwargs(self):
        """Test cache key includes relevant kwargs."""
        messages = ["Test"]
        
        key1 = get_cache_key(messages, temperature=0.7)
        key2 = get_cache_key(messages, temperature=0.8)
        key3 = get_cache_key(messages, temperature=0.7)
        
        # Different temperatures should generate different keys
        assert key1 != key2
        # Same temperature should generate same key
        assert key1 == key3
    
    def test_get_cache_key_ignores_irrelevant_kwargs(self):
        """Test that irrelevant kwargs are ignored in cache key."""
        messages = ["Test"]
        
        key1 = get_cache_key(messages, temperature=0.7, irrelevant_param="value")
        key2 = get_cache_key(messages, temperature=0.7, another_irrelevant="value")
        
        # Keys should be the same despite different irrelevant params
        assert key1 == key2
    
    def test_get_cache_key_with_multiple_relevant_kwargs(self):
        """Test cache key with multiple relevant parameters."""
        messages = ["Test"]
        
        key1 = get_cache_key(
            messages,
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )
        key2 = get_cache_key(
            messages,
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )
        key3 = get_cache_key(
            messages,
            temperature=0.7,
            max_tokens=200,  # Different value
            top_p=0.9
        )
        
        # Same params should generate same key
        assert key1 == key2
        # Different params should generate different key
        assert key1 != key3
    
    def test_get_cache_key_deterministic(self):
        """Test that cache key generation is deterministic."""
        messages = ["Test message"]
        kwargs = {"temperature": 0.7, "max_tokens": 100}
        
        # Generate keys multiple times
        keys = [get_cache_key(messages, **kwargs) for _ in range(10)]
        
        # All keys should be identical
        assert len(set(keys)) == 1


class TestCachedLLMCall:
    """Test cached LLM call functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        clear_cache()
    
    def teardown_method(self):
        """Clean up after tests."""
        clear_cache()
    
    @patch("src.utils.llm_cache.get_config")
    def test_cached_llm_call_disabled(self, mock_get_config):
        """Test that caching is bypassed when disabled in config."""
        # Create mock config with caching disabled
        mock_config = Mock()
        mock_config.llm_cache_enabled = False
        mock_get_config.return_value = mock_config
        
        # Create mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_llm.invoke.return_value = mock_response
        
        messages = ["Test"]
        result = cached_llm_call(mock_llm, messages)
        
        # Should call LLM directly
        mock_llm.invoke.assert_called_once_with(messages)
        assert result == mock_response
    
    @patch("src.utils.llm_cache.get_config")
    def test_cached_llm_call_cache_hit(self, mock_get_config):
        """Test cache hit behavior."""
        # Create mock config with caching enabled
        mock_config = Mock()
        mock_config.llm_cache_enabled = True
        mock_config.llm_cache_size = 128
        mock_get_config.return_value = mock_config
        
        # Create mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Cached response"
        mock_llm.invoke.return_value = mock_response
        
        messages = ["Test message"]
        
        # First call - should be a miss
        result1 = cached_llm_call(mock_llm, messages)
        assert mock_llm.invoke.call_count == 1
        
        # Second call with same messages - should be a hit
        result2 = cached_llm_call(mock_llm, messages)
        
        # Should only be called once (cached on second call)
        assert mock_llm.invoke.call_count == 1
        assert result1 == result2
    
    @patch("src.utils.llm_cache.get_config")
    def test_cached_llm_call_cache_miss(self, mock_get_config):
        """Test cache miss behavior."""
        # Create mock config with caching enabled
        mock_config = Mock()
        mock_config.llm_cache_enabled = True
        mock_config.llm_cache_size = 128
        mock_get_config.return_value = mock_config
        
        # Create mock LLM
        mock_llm = Mock()
        mock_response1 = Mock()
        mock_response1.content = "Response 1"
        mock_response2 = Mock()
        mock_response2.content = "Response 2"
        mock_llm.invoke.side_effect = [mock_response1, mock_response2]
        
        messages1 = ["Test message 1"]
        messages2 = ["Test message 2"]
        
        # First call
        result1 = cached_llm_call(mock_llm, messages1)
        assert mock_llm.invoke.call_count == 1
        
        # Second call with different messages - should be a miss
        result2 = cached_llm_call(mock_llm, messages2)
        
        # Should be called twice (different messages)
        assert mock_llm.invoke.call_count == 2
        assert result1 != result2
    
    @patch("src.utils.llm_cache.get_config")
    def test_cached_llm_call_with_kwargs(self, mock_get_config):
        """Test caching with kwargs."""
        # Create mock config with caching enabled
        mock_config = Mock()
        mock_config.llm_cache_enabled = True
        mock_config.llm_cache_size = 128
        mock_get_config.return_value = mock_config
        
        # Create mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_llm.invoke.return_value = mock_response
        
        messages = ["Test"]
        
        # First call with temperature
        cached_llm_call(mock_llm, messages, temperature=0.7)
        assert mock_llm.invoke.call_count == 1
        
        # Second call with same temperature - should be cached
        cached_llm_call(mock_llm, messages, temperature=0.7)
        assert mock_llm.invoke.call_count == 1
        
        # Third call with different temperature - should not be cached
        cached_llm_call(mock_llm, messages, temperature=0.8)
        assert mock_llm.invoke.call_count == 2
    
    @patch("src.utils.llm_cache.get_config")
    def test_cached_llm_call_cache_size_limit(self, mock_get_config):
        """Test that cache respects size limits."""
        # Create mock config with small cache size
        mock_config = Mock()
        mock_config.llm_cache_enabled = True
        mock_config.llm_cache_size = 2  # Very small cache
        mock_get_config.return_value = mock_config
        
        # Create mock LLM
        mock_llm = Mock()
        mock_llm.invoke.side_effect = lambda msgs, **kw: Mock(content=f"Response for {msgs[0]}")
        
        # Fill cache with 2 entries
        cached_llm_call(mock_llm, ["Message 1"])
        cached_llm_call(mock_llm, ["Message 2"])
        assert mock_llm.invoke.call_count == 2
        
        # Add third entry - should evict first entry (LRU)
        cached_llm_call(mock_llm, ["Message 3"])
        assert mock_llm.invoke.call_count == 3
        
        # First message should be evicted, so it's a miss
        cached_llm_call(mock_llm, ["Message 1"])
        assert mock_llm.invoke.call_count == 4


class TestCacheStatistics:
    """Test cache statistics functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        clear_cache()
    
    def teardown_method(self):
        """Clean up after tests."""
        clear_cache()
    
    def test_get_cache_stats_initial(self):
        """Test initial cache statistics."""
        stats = get_cache_stats()
        
        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
        assert stats["size"] == 0
    
    @patch("src.utils.llm_cache.get_config")
    def test_get_cache_stats_after_calls(self, mock_get_config):
        """Test cache statistics after cache operations."""
        # Create mock config
        mock_config = Mock()
        mock_config.llm_cache_enabled = True
        mock_config.llm_cache_size = 128
        mock_get_config.return_value = mock_config
        
        # Create mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="Response")
        
        messages = ["Test"]
        
        # Make some calls
        cached_llm_call(mock_llm, messages)  # Miss
        cached_llm_call(mock_llm, messages)  # Hit
        cached_llm_call(mock_llm, messages)  # Hit
        
        stats = get_cache_stats()
        
        # Should have at least 1 hit and 1 miss
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["size"] >= 1


class TestCacheInvalidation:
    """Test cache invalidation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        clear_cache()
    
    def teardown_method(self):
        """Clean up after tests."""
        clear_cache()
    
    @patch("src.utils.llm_cache.get_config")
    def test_clear_cache(self, mock_get_config):
        """Test clearing the entire cache."""
        # Create mock config
        mock_config = Mock()
        mock_config.llm_cache_enabled = True
        mock_config.llm_cache_size = 128
        mock_get_config.return_value = mock_config
        
        # Create mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="Response")
        
        messages = ["Test"]
        
        # Make a call to populate cache
        cached_llm_call(mock_llm, messages)
        
        # Clear cache
        clear_cache()
        
        # Verify cache is cleared
        stats = get_cache_stats()
        assert stats["size"] == 0
        
        # Next call should be a miss
        cached_llm_call(mock_llm, messages)
        assert mock_llm.invoke.call_count == 2
    
    @patch("src.utils.llm_cache.get_config")
    def test_invalidate_cache_entry(self, mock_get_config):
        """Test invalidating a specific cache entry."""
        # Create mock config
        mock_config = Mock()
        mock_config.llm_cache_enabled = True
        mock_config.llm_cache_size = 128
        mock_get_config.return_value = mock_config
        
        # Create mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="Response")
        
        messages = ["Test"]
        
        # Make a call to populate cache
        cached_llm_call(mock_llm, messages)
        
        # Invalidate entry (clears entire cache)
        result = invalidate_cache_entry(messages)
        
        assert result is True
        
        # Next call should be a miss
        cached_llm_call(mock_llm, messages)
        assert mock_llm.invoke.call_count == 2


class TestCacheIntegration:
    """Integration tests for cache functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        clear_cache()
    
    def teardown_method(self):
        """Clean up after tests."""
        clear_cache()
    
    @patch("src.utils.llm_cache.get_config")
    def test_cache_with_message_objects(self, mock_get_config):
        """Test caching with actual message-like objects."""
        # Create mock config
        mock_config = Mock()
        mock_config.llm_cache_enabled = True
        mock_config.llm_cache_size = 128
        mock_get_config.return_value = mock_config
        
        # Create mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Response"
        mock_llm.invoke.return_value = mock_response
        
        # Create message objects
        msg1 = Mock()
        msg1.content = "Hello"
        msg1.type = "human"
        
        msg2 = Mock()
        msg2.content = "World"
        msg2.type = "ai"
        
        messages = [msg1, msg2]
        
        # First call
        result1 = cached_llm_call(mock_llm, messages)
        assert mock_llm.invoke.call_count == 1
        
        # Second call with same messages - should be cached
        result2 = cached_llm_call(mock_llm, messages)
        assert mock_llm.invoke.call_count == 1
        assert result1 == result2
    
    @patch("src.utils.llm_cache.get_config")
    def test_cache_preserves_response_object(self, mock_get_config):
        """Test that cached responses preserve the response object."""
        # Create mock config
        mock_config = Mock()
        mock_config.llm_cache_enabled = True
        mock_config.llm_cache_size = 128
        mock_get_config.return_value = mock_config
        
        # Create mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_response.other_attr = "value"
        mock_llm.invoke.return_value = mock_response
        
        messages = ["Test"]
        
        # First call
        result1 = cached_llm_call(mock_llm, messages)
        
        # Second call - should return same object
        result2 = cached_llm_call(mock_llm, messages)
        
        # Should be the same object (cached)
        assert result1 is result2
        assert result2.content == "Test response"
        assert result2.other_attr == "value"

