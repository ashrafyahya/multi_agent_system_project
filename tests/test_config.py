"""Tests for configuration management.

This module contains unit tests for the configuration system to verify
environment variable loading, validation, and type safety.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from src.config import Config, get_config, reload_config


class TestConfig:
    """Tests for Config class."""
    
    def test_config_loads_from_environment(self) -> None:
        """Test config loads values from environment variables."""
        with patch.dict(
            os.environ,
            {
                "GROQ_API_KEY": "test_api_key",
                "GROQ_MODEL": "llama-3.1-8b-instant",
                "MAX_RETRIES": "5",
                "LOG_LEVEL": "DEBUG",
                "DATA_DIR": "./test_data",
            },
            clear=False,
        ):
            config = Config()
            
            assert config.groq_api_key == "test_api_key"
            assert config.groq_model == "llama-3.1-8b-instant"
            assert config.max_retries == 5
            assert config.log_level == "DEBUG"
            assert isinstance(config.data_dir, Path)
    
    def test_config_uses_defaults(self) -> None:
        """Test config uses default values when env vars not set."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}, clear=False):
            # Remove other env vars
            env_vars_to_remove = [
                "GROQ_MODEL",
                "MAX_RETRIES",
                "LOG_LEVEL",
                "DATA_DIR",
                "TAVILY_API_KEY",
            ]
            for var in env_vars_to_remove:
                os.environ.pop(var, None)
            
            config = Config()
            
            assert config.groq_model == "llama-3.1-8b-instant"  # Default
            assert config.max_retries == 3  # Default
            assert config.log_level == "INFO"  # Default
            assert isinstance(config.data_dir, Path)
    
    def test_config_validates_groq_api_key_required(self) -> None:
        """Test config requires groq_api_key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError):
                Config()
    
    def test_config_validates_max_retries_range(self) -> None:
        """Test config validates max_retries is in valid range."""
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "MAX_RETRIES": "0"},
            clear=False,
        ):
            with pytest.raises(ValidationError):
                Config()
        
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "MAX_RETRIES": "11"},
            clear=False,
        ):
            with pytest.raises(ValidationError):
                Config()
    
    def test_config_validates_log_level(self) -> None:
        """Test config validates log level is one of allowed values."""
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "LOG_LEVEL": "INVALID"},
            clear=False,
        ):
            with pytest.raises(ValueError, match="log_level must be one of"):
                Config()
    
    def test_config_normalizes_log_level_to_uppercase(self) -> None:
        """Test config normalizes log level to uppercase."""
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "LOG_LEVEL": "debug"},
            clear=False,
        ):
            config = Config()
            assert config.log_level == "DEBUG"
    
    def test_config_creates_data_dir(self) -> None:
        """Test config creates data directory if it doesn't exist."""
        test_dir = Path("./test_data_dir_config")
        
        # Clean up if exists
        if test_dir.exists():
            test_dir.rmdir()
        
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "DATA_DIR": str(test_dir)},
            clear=False,
        ):
            config = Config()
            
            assert config.data_dir == test_dir
            assert test_dir.exists()
            
            # Cleanup
            test_dir.rmdir()
    
    def test_config_handles_optional_tavily_api_key(self) -> None:
        """Test config handles optional tavily_api_key."""
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key"},
            clear=False,
        ):
            os.environ.pop("TAVILY_API_KEY", None)
            config = Config()
            assert config.tavily_api_key is None
        
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "TAVILY_API_KEY": "tavily_key"},
            clear=False,
        ):
            config = Config()
            assert config.tavily_api_key == "tavily_key"


class TestGetConfig:
    """Tests for get_config function."""
    
    def test_get_config_returns_singleton(self) -> None:
        """Test get_config returns the same instance on multiple calls."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}, clear=False):
            config1 = get_config()
            config2 = get_config()
            
            assert config1 is config2
    
    def test_get_config_loads_config_on_first_call(self) -> None:
        """Test get_config loads config on first call."""
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "GROQ_MODEL": "test-model"},
            clear=False,
        ):
            config = get_config()
            assert config.groq_api_key == "test_key"
            assert config.groq_model == "test-model"


class TestReloadConfig:
    """Tests for reload_config function."""
    
    def test_reload_config_creates_new_instance(self) -> None:
        """Test reload_config creates a new config instance."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}, clear=False):
            config1 = get_config()
            config2 = reload_config()
            
            # Should be different instances
            assert config1 is not config2
    
    def test_reload_config_loads_new_values(self) -> None:
        """Test reload_config loads new values from environment."""
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "GROQ_MODEL": "model1"},
            clear=False,
        ):
            config1 = get_config()
            assert config1.groq_model == "model1"
            
            # Change environment
            os.environ["GROQ_MODEL"] = "model2"
            config2 = reload_config()
            
            assert config2.groq_model == "model2"
            # Old config should still have old value
            assert config1.groq_model == "model1"


