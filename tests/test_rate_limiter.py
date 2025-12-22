"""Tests for rate limiter and retry logic.

This module contains unit tests for the rate limiter to verify
exponential backoff, retry attempts, rate limit detection, and error handling.
"""

import time
from unittest.mock import Mock, patch
from typing import Any

import pytest

from src.config import Config, reload_config
from src.exceptions.workflow_error import WorkflowError
from src.utils.rate_limiter import (
    RateLimitError,
    _is_rate_limit_error,
    invoke_llm_with_retry,
    retry_llm_call,
)


class TestRateLimitDetection:
    """Tests for rate limit error detection."""
    
    def test_detects_rate_limit_in_error_message(self) -> None:
        """Test rate limit detection from error message."""
        error = Exception("Rate limit exceeded")
        assert _is_rate_limit_error(error) is True
    
    def test_detects_429_status_code(self) -> None:
        """Test rate limit detection from HTTP 429 status code."""
        error = Mock()
        error.status_code = 429
        assert _is_rate_limit_error(error) is True
    
    def test_detects_quota_exceeded(self) -> None:
        """Test rate limit detection from quota exceeded message."""
        error = Exception("Quota exceeded for this API")
        assert _is_rate_limit_error(error) is True
    
    def test_detects_throttled_error(self) -> None:
        """Test rate limit detection from throttled message."""
        error = Exception("Request throttled")
        assert _is_rate_limit_error(error) is True
    
    def test_detects_rate_limit_in_response(self) -> None:
        """Test rate limit detection from response status code."""
        error = Mock()
        error.response = Mock()
        error.response.status_code = 429
        assert _is_rate_limit_error(error) is True
    
    def test_does_not_detect_non_rate_limit_error(self) -> None:
        """Test that non-rate-limit errors are not detected."""
        error = Exception("Invalid API key")
        assert _is_rate_limit_error(error) is False
    
    def test_does_not_detect_other_http_errors(self) -> None:
        """Test that other HTTP errors are not detected as rate limits."""
        error = Mock()
        error.status_code = 500
        assert _is_rate_limit_error(error) is False


class TestRetryLlmCallDecorator:
    """Tests for retry_llm_call decorator."""
    
    def test_successful_call_no_retry(self) -> None:
        """Test that successful calls don't trigger retries."""
        call_count = 0
        
        @retry_llm_call
        def successful_call() -> str:
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_call()
        
        assert result == "success"
        assert call_count == 1
    
    def test_retries_on_rate_limit_error(self) -> None:
        """Test that rate limit errors trigger retries."""
        call_count = 0
        
        @retry_llm_call
        def rate_limited_call() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Rate limit exceeded")
            return "success"
        
        with patch("src.utils.rate_limiter.get_config") as mock_config:
            mock_config.return_value.llm_retry_attempts = 3
            mock_config.return_value.llm_retry_backoff_min = 0.01
            mock_config.return_value.llm_retry_backoff_max = 0.1
            
            result = rate_limited_call()
        
        assert result == "success"
        assert call_count == 2
    
    def test_max_retry_attempts(self) -> None:
        """Test that max retry attempts are respected."""
        call_count = 0
        
        @retry_llm_call
        def always_failing_call() -> str:
            nonlocal call_count
            call_count += 1
            raise Exception("Rate limit exceeded")
        
        with patch("src.utils.rate_limiter.get_config") as mock_config:
            mock_config.return_value.llm_retry_attempts = 3
            mock_config.return_value.llm_retry_backoff_min = 0.01
            mock_config.return_value.llm_retry_backoff_max = 0.1
            
            with pytest.raises(Exception, match="Rate limit exceeded"):
                always_failing_call()
        
        assert call_count == 3
    
    def test_exponential_backoff_timing(self) -> None:
        """Test that exponential backoff timing is used."""
        call_times: list[float] = []
        
        @retry_llm_call
        def timing_test_call() -> str:
            call_times.append(time.time())
            if len(call_times) < 3:
                raise Exception("Rate limit exceeded")
            return "success"
        
        with patch("src.utils.rate_limiter.get_config") as mock_config:
            mock_config.return_value.llm_retry_attempts = 3
            mock_config.return_value.llm_retry_backoff_min = 0.1
            mock_config.return_value.llm_retry_backoff_max = 1.0
            
            start_time = time.time()
            result = timing_test_call()
            end_time = time.time()
        
        assert result == "success"
        assert len(call_times) == 3
        
        # Check that backoff time increases (allowing some tolerance)
        if len(call_times) >= 2:
            first_backoff = call_times[1] - call_times[0]
            second_backoff = call_times[2] - call_times[1]
            # Second backoff should be longer (exponential)
            assert second_backoff >= first_backoff * 0.8  # Allow 20% tolerance
    
    def test_retries_on_general_exception(self) -> None:
        """Test that general exceptions also trigger retries."""
        call_count = 0
        
        @retry_llm_call
        def general_error_call() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Network error")
            return "success"
        
        with patch("src.utils.rate_limiter.get_config") as mock_config:
            mock_config.return_value.llm_retry_attempts = 3
            mock_config.return_value.llm_retry_backoff_min = 0.01
            mock_config.return_value.llm_retry_backoff_max = 0.1
            
            result = general_error_call()
        
        assert result == "success"
        assert call_count == 2
    
    def test_wraps_rate_limit_errors(self) -> None:
        """Test that rate limit errors are wrapped in RateLimitError."""
        @retry_llm_call
        def rate_limited_call() -> str:
            error = Exception("Rate limit exceeded")
            raise error
        
        with patch("src.utils.rate_limiter.get_config") as mock_config:
            mock_config.return_value.llm_retry_attempts = 1
            mock_config.return_value.llm_retry_backoff_min = 0.01
            mock_config.return_value.llm_retry_backoff_max = 0.1
            
            with pytest.raises(RateLimitError):
                rate_limited_call()


class TestInvokeLlmWithRetry:
    """Tests for invoke_llm_with_retry convenience function."""
    
    def test_successful_invoke(self) -> None:
        """Test successful LLM invocation."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_llm.invoke.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test"}]
        result = invoke_llm_with_retry(mock_llm, messages)
        
        assert result == mock_response
        mock_llm.invoke.assert_called_once_with(messages)
    
    def test_retries_on_rate_limit(self) -> None:
        """Test that rate limit errors trigger retries."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Success"
        
        call_count = 0
        def invoke_side_effect(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Rate limit exceeded")
            return mock_response
        
        mock_llm.invoke.side_effect = invoke_side_effect
        
        with patch("src.utils.rate_limiter.get_config") as mock_config:
            mock_config.return_value.llm_retry_attempts = 3
            mock_config.return_value.llm_retry_backoff_min = 0.01
            mock_config.return_value.llm_retry_backoff_max = 0.1
            
            messages = [{"role": "user", "content": "Test"}]
            result = invoke_llm_with_retry(mock_llm, messages)
        
        assert result == mock_response
        assert call_count == 2
    
    def test_raises_workflow_error_after_max_retries(self) -> None:
        """Test that WorkflowError is raised after max retries."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Rate limit exceeded")
        
        with patch("src.utils.rate_limiter.get_config") as mock_config:
            mock_config.return_value.llm_retry_attempts = 2
            mock_config.return_value.llm_retry_backoff_min = 0.01
            mock_config.return_value.llm_retry_backoff_max = 0.1
            
            messages = [{"role": "user", "content": "Test"}]
            
            with pytest.raises(WorkflowError, match="rate limit exceeded"):
                invoke_llm_with_retry(mock_llm, messages)
    
    def test_passes_kwargs_to_llm(self) -> None:
        """Test that kwargs are passed to LLM invoke."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test"
        mock_llm.invoke.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test"}]
        result = invoke_llm_with_retry(
            mock_llm,
            messages,
            temperature=0.7,
            max_tokens=100
        )
        
        assert result == mock_response
        mock_llm.invoke.assert_called_once_with(
            messages,
            temperature=0.7,
            max_tokens=100
        )


class TestRateLimitError:
    """Tests for RateLimitError exception."""
    
    def test_rate_limit_error_wraps_original(self) -> None:
        """Test that RateLimitError wraps original exception."""
        original = Exception("Rate limit exceeded")
        error = RateLimitError(original)
        
        assert error.original_exception == original
        assert "Rate limit error" in str(error)
    
    def test_rate_limit_error_preserves_exception_chain(self) -> None:
        """Test that exception chain is preserved."""
        original = Exception("Rate limit exceeded")
        try:
            raise original
        except Exception:
            error = RateLimitError(original)
            # Check that original_exception attribute is set
            assert error.original_exception == original
            # Check that the error message includes the original
            assert "Rate limit exceeded" in str(error)


class TestConfigIntegration:
    """Tests for configuration integration."""
    
    def test_uses_config_values(self) -> None:
        """Test that decorator uses config values."""
        call_count = 0
        
        @retry_llm_call
        def test_call() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Error")
            return "success"
        
        # Reload config to ensure fresh values
        with patch.dict(
            "os.environ",
            {
                "GROQ_API_KEY": "test_key",
                "LLM_RETRY_ATTEMPTS": "5",
                "LLM_RETRY_BACKOFF_MIN": "0.5",
                "LLM_RETRY_BACKOFF_MAX": "10.0",
            },
            clear=False,
        ):
            import os
            original_env = os.environ.copy()
            try:
                os.environ.update({
                    "GROQ_API_KEY": "test_key",
                    "LLM_RETRY_ATTEMPTS": "5",
                    "LLM_RETRY_BACKOFF_MIN": "0.5",
                    "LLM_RETRY_BACKOFF_MAX": "10.0",
                })
                reload_config()
                result = test_call()
                assert result == "success"
            finally:
                os.environ.clear()
                os.environ.update(original_env)
                reload_config()

