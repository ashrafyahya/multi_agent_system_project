"""Tests for input validation and sanitization utilities.

This module tests all input validation functions to ensure they:
- Properly validate input lengths
- Sanitize dangerous characters
- Prevent path traversal attacks
- Validate URLs correctly
- Handle edge cases gracefully
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.config import Config
from src.exceptions.workflow_error import WorkflowError
from src.utils.input_validator import (sanitize_user_query,
                                       validate_and_sanitize_url,
                                       validate_file_path, validate_url)


class TestSanitizeUserQuery:
    """Tests for sanitize_user_query function."""
    
    def test_valid_query(self):
        """Test that valid queries pass validation."""
        query = "Analyze competitors in the SaaS market"
        result = sanitize_user_query(query)
        assert result == query
        assert len(result) == len(query)
    
    def test_query_with_whitespace(self):
        """Test that whitespace is trimmed."""
        query = "  Test query with spaces  "
        result = sanitize_user_query(query)
        assert result == "Test query with spaces"
        assert result.strip() == result
    
    def test_query_too_short(self):
        """Test that queries below minimum length raise error."""
        config = Config(
            groq_api_key="test-key",
            min_query_length=10,
            max_query_length=5000
        )
        
        with pytest.raises(WorkflowError) as exc_info:
            sanitize_user_query("short", config=config)
        
        assert "too short" in str(exc_info.value).lower()
        assert exc_info.value.context["min_length"] == 10
    
    def test_query_too_long(self):
        """Test that queries above maximum length raise error."""
        config = Config(
            groq_api_key="test-key",
            min_query_length=10,
            max_query_length=100
        )
        
        long_query = "a" * 200
        
        with pytest.raises(WorkflowError) as exc_info:
            sanitize_user_query(long_query, config=config)
        
        assert "too long" in str(exc_info.value).lower()
        assert exc_info.value.context["max_length"] == 100
    
    def test_empty_query(self):
        """Test that empty queries raise error."""
        with pytest.raises(WorkflowError) as exc_info:
            sanitize_user_query("")
        
        assert "cannot be empty" in str(exc_info.value).lower()
    
    def test_query_with_dangerous_chars(self):
        """Test that dangerous characters are removed."""
        query = "Test query with <script>alert('xss')</script> dangerous chars"
        result = sanitize_user_query(query)
        
        assert "<script>" not in result
        assert "</script>" not in result
        assert "'" not in result  # Single quotes should be removed
        assert "Test query with" in result
        # Note: "alert" is not a dangerous character, only special chars are removed
    
    def test_query_with_control_chars(self):
        """Test that control characters are removed."""
        query = "Test query\x00\x1f with control chars"
        result = sanitize_user_query(query)
        
        assert "\x00" not in result
        assert "\x1f" not in result
        assert "Test query" in result
    
    def test_query_normalizes_whitespace(self):
        """Test that multiple spaces are normalized."""
        query = "Test    query   with    multiple    spaces"
        result = sanitize_user_query(query)
        
        assert "  " not in result  # No double spaces
        assert "Test query with multiple spaces" == result
    
    def test_query_after_sanitization_too_short(self):
        """Test that query too short after sanitization raises error."""
        config = Config(
            groq_api_key="test-key",
            min_query_length=10,
            max_query_length=5000
        )
        
        # Query that is long enough before sanitization but becomes too short after
        # "abcdefgh" is 8 chars, but we add control chars to make it 10+ before sanitization
        # After removing control chars, only "abcdefgh" remains (8 chars < 10)
        query = "abcdefgh\x00\x01"
        
        with pytest.raises(WorkflowError) as exc_info:
            sanitize_user_query(query, config=config)
        
        # The error should mention it's too short after sanitization
        error_msg = str(exc_info.value).lower()
        assert "too short after sanitization" in error_msg


class TestValidateUrl:
    """Tests for validate_url function."""
    
    def test_valid_http_url(self):
        """Test that valid HTTP URLs pass validation."""
        is_valid, sanitized = validate_url("http://example.com")
        
        assert is_valid is True
        assert sanitized == "http://example.com"
    
    def test_valid_https_url(self):
        """Test that valid HTTPS URLs pass validation."""
        is_valid, sanitized = validate_url("https://example.com/path?query=value")
        
        assert is_valid is True
        assert sanitized == "https://example.com/path?query=value"
        assert "#" not in sanitized  # Fragment should be removed
    
    def test_url_with_uppercase_scheme(self):
        """Test that uppercase schemes are normalized."""
        is_valid, sanitized = validate_url("HTTPS://EXAMPLE.COM")
        
        assert is_valid is True
        assert sanitized == "https://example.com"
    
    def test_url_without_scheme(self):
        """Test that URLs without scheme fail validation."""
        is_valid, sanitized = validate_url("example.com")
        
        assert is_valid is False
        assert sanitized is None
    
    def test_url_with_disallowed_scheme(self):
        """Test that disallowed schemes fail validation."""
        is_valid, sanitized = validate_url("javascript:alert(1)")
        
        assert is_valid is False
        assert sanitized is None
    
    def test_url_with_path_traversal(self):
        """Test that URLs with path traversal fail validation."""
        is_valid, sanitized = validate_url("https://example.com/../etc/passwd")
        
        assert is_valid is False
        assert sanitized is None
    
    def test_url_with_localhost(self):
        """Test that localhost URLs are rejected by default."""
        is_valid, sanitized = validate_url("http://localhost:8080")
        
        assert is_valid is False
        assert sanitized is None
    
    def test_url_with_localhost_allowed(self):
        """Test that localhost URLs pass when allowed."""
        is_valid, sanitized = validate_url("http://localhost:8080", allow_localhost=True)
        
        assert is_valid is True
        assert sanitized == "http://localhost:8080"
    
    def test_empty_url(self):
        """Test that empty URLs fail validation."""
        is_valid, sanitized = validate_url("")
        
        assert is_valid is False
        assert sanitized is None
    
    def test_url_with_whitespace(self):
        """Test that URLs with whitespace are trimmed."""
        is_valid, sanitized = validate_url("  https://example.com  ")
        
        assert is_valid is True
        assert sanitized == "https://example.com"
    
    def test_url_too_long(self):
        """Test that extremely long URLs fail validation."""
        long_url = "https://example.com/" + "a" * 3000
        is_valid, sanitized = validate_url(long_url)
        
        assert is_valid is False
        assert sanitized is None
    
    def test_url_fragment_removed(self):
        """Test that URL fragments are removed for security."""
        is_valid, sanitized = validate_url("https://example.com/page#fragment")
        
        assert is_valid is True
        assert "#" not in sanitized
        assert sanitized == "https://example.com/page"


class TestValidateAndSanitizeUrl:
    """Tests for validate_and_sanitize_url convenience function."""
    
    def test_valid_url_returns_sanitized(self):
        """Test that valid URLs return sanitized version."""
        result = validate_and_sanitize_url("https://example.com")
        assert result == "https://example.com"
    
    def test_invalid_url_raises_error(self):
        """Test that invalid URLs raise WorkflowError."""
        with pytest.raises(WorkflowError) as exc_info:
            validate_and_sanitize_url("javascript:alert(1)")
        
        assert "Invalid or unsafe URL" in str(exc_info.value)
        assert exc_info.value.context["url"] == "javascript:alert(1)"


class TestValidateFilePath:
    """Tests for validate_file_path function."""
    
    def test_valid_relative_path(self):
        """Test that valid relative paths pass validation."""
        is_valid, path, error = validate_file_path("data/report.pdf")
        
        assert is_valid is True
        assert path is not None
        assert isinstance(path, Path)
        assert error is None
    
    def test_path_with_traversal_detected(self):
        """Test that path traversal patterns are detected."""
        is_valid, path, error = validate_file_path("../etc/passwd")
        
        assert is_valid is False
        assert path is None
        assert "path traversal" in error.lower()
    
    def test_path_with_double_dot(self):
        """Test that double dot patterns are detected."""
        is_valid, path, error = validate_file_path("data/../../etc/passwd")
        
        assert is_valid is False
        assert path is None
        assert "path traversal" in error.lower()
    
    def test_path_with_backslash_traversal(self):
        """Test that backslash traversal is detected."""
        is_valid, path, error = validate_file_path("data\\..\\etc\\passwd")
        
        assert is_valid is False
        assert path is None
        assert "path traversal" in error.lower()
    
    def test_path_with_url_encoded_traversal(self):
        """Test that URL-encoded traversal is detected."""
        is_valid, path, error = validate_file_path("data/%2e%2e/etc/passwd")
        
        assert is_valid is False
        assert path is None
        assert "path traversal" in error.lower()
    
    def test_absolute_path_not_allowed(self):
        """Test that absolute paths are rejected by default."""
        is_valid, path, error = validate_file_path("/etc/passwd", allow_absolute=False)
        
        assert is_valid is False
        assert path is None
        # On Windows, absolute paths might resolve differently, so check for either error message
        assert "absolute" in error.lower() or "outside" in error.lower()
    
    def test_absolute_path_allowed(self):
        """Test that absolute paths pass when allowed."""
        is_valid, path, error = validate_file_path("/tmp/test.txt", allow_absolute=True)
        
        assert is_valid is True
        assert path is not None
        assert error is None
    
    def test_path_with_base_dir(self):
        """Test path resolution relative to base directory."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "test_base"
            base_dir.mkdir(parents=True, exist_ok=True)
            
            is_valid, path, error = validate_file_path(
                "subdir/file.txt",
                base_dir=base_dir
            )
            
            assert is_valid is True
            assert path is not None
            assert path.is_absolute()
            # Check that resolved path is within base_dir (using resolved paths for Windows compatibility)
            assert base_dir.resolve() in path.resolve().parents or path.resolve() == (base_dir / "subdir" / "file.txt").resolve()
            assert error is None
    
    def test_path_resolves_outside_base_dir(self):
        """Test that paths resolving outside base directory are rejected."""
        base_dir = Path("/tmp/test_base")
        base_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # This should fail because even though we check for .., 
            # we also verify the resolved path is within base_dir
            is_valid, path, error = validate_file_path(
                "../../etc/passwd",
                base_dir=base_dir
            )
            
            # Should fail due to path traversal detection
            assert is_valid is False
            assert path is None
        finally:
            base_dir.rmdir()
    
    def test_empty_path(self):
        """Test that empty paths fail validation."""
        is_valid, path, error = validate_file_path("")
        
        assert is_valid is False
        assert path is None
        assert "cannot be empty" in error.lower()
    
    def test_path_with_base_dir_not_existing(self):
        """Test that non-existent base directory fails validation."""
        base_dir = Path("/nonexistent/directory")
        
        is_valid, path, error = validate_file_path(
            "file.txt",
            base_dir=base_dir
        )
        
        assert is_valid is False
        assert path is None
        assert "does not exist" in error.lower()
    
    def test_path_object_input(self):
        """Test that Path objects are accepted."""
        path_obj = Path("data/report.pdf")
        is_valid, path, error = validate_file_path(path_obj)
        
        assert is_valid is True
        assert path is not None
        assert error is None


class TestInputValidatorIntegration:
    """Integration tests for input validator with real config."""
    
    @patch('src.utils.input_validator.get_config')
    def test_sanitize_user_query_uses_config(self, mock_get_config):
        """Test that sanitize_user_query uses config from get_config()."""
        config = Config(
            groq_api_key="test-key",
            min_query_length=5,
            max_query_length=100
        )
        mock_get_config.return_value = config
        
        # Should pass
        result = sanitize_user_query("Valid query")
        assert result == "Valid query"
        
        # Should fail (too short)
        with pytest.raises(WorkflowError):
            sanitize_user_query("abc")
        
        # Should fail (too long)
        with pytest.raises(WorkflowError):
            sanitize_user_query("a" * 200)

