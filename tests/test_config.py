"""Tests for configuration management.

This module contains unit tests for the configuration system to verify
environment variable loading, validation, and type safety.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
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
                "LLM_MODEL": "llama-3.1-8b-instant",
                "MAX_RETRIES": "5",
                "LOG_LEVEL": "DEBUG",
                "DATA_DIR": "./test_data",
            },
            clear=False,
        ):
            config = Config()
            
            assert config.groq_api_key == "test_api_key"
            assert config.llm_model == "llama-3.1-8b-instant"
            assert config.max_retries == 5
            assert config.log_level == "DEBUG"
            assert isinstance(config.data_dir, Path)
    
    def test_config_uses_defaults(self) -> None:
        """Test config uses default values when env vars not set."""
        # Store original values to restore later
        original_values = {}
        env_vars_to_check = [
            "LLM_MODEL",
            "LLM_MODEL_PLANNER",
            "LLM_MODEL_SUPERVISOR",
            "LLM_MODEL_INSIGHT",
            "LLM_MODEL_REPORT",
            "LLM_MODEL_COLLECTOR",
            "LLM_MODEL_EXPORT",
            "MAX_RETRIES",
            "LOG_LEVEL",
            "DATA_DIR",
            "TAVILY_API_KEY",
        ]
        for var in env_vars_to_check:
            original_values[var] = os.environ.get(var)
        
        # Use _env_file=None to disable .env file loading and test pure defaults
        # This ensures we test the actual default values, not values from .env file
        env_overrides = {
            "GROQ_API_KEY": "test_key",
            "TAVILY_API_KEY": "test_tavily_key",  # Required field
        }
        
        # Remove other env vars that might be set
        for var in env_vars_to_check:
            os.environ.pop(var, None)
        
        # Reload config with overridden environment, ignoring .env file
        with patch.dict(os.environ, env_overrides, clear=False):
            # Use _env_file=None to disable .env file loading
            config = Config(_env_file=None)
            
            assert config.llm_model == "llama-3.1-8b-instant"  # Default
            assert config.max_retries == 3  # Default
            assert config.log_level == "INFO"  # Default
            assert isinstance(config.data_dir, Path)
        
        # Restore original values
        for var, value in original_values.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                os.environ.pop(var, None)
    
    def test_config_validates_groq_api_key_required(self) -> None:
        """Test config requires groq_api_key."""
        # Remove GROQ_API_KEY from environment
        original_key = os.environ.pop("GROQ_API_KEY", None)
        try:
            # Create Config without .env file loading
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValidationError):
                    # Use _env_file=None to disable .env file loading
                    Config(_env_file=None)
        finally:
            # Restore original key if it existed
            if original_key:
                os.environ["GROQ_API_KEY"] = original_key
    
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
    
    def test_config_requires_tavily_api_key(self) -> None:
        """Test config requires tavily_api_key."""
        original_tavily = os.environ.pop("TAVILY_API_KEY", None)
        try:
            # Use _env_file=None to disable .env file loading
            with patch.dict(
                os.environ,
                {"GROQ_API_KEY": "test_key"},
                clear=False,
            ):
                # Ensure TAVILY_API_KEY is not in environment
                os.environ.pop("TAVILY_API_KEY", None)
                # Should raise ValidationError when tavily_api_key is missing
                with pytest.raises(ValidationError):
                    Config(_env_file=None)
            
            with patch.dict(
                os.environ,
                {"GROQ_API_KEY": "test_key", "TAVILY_API_KEY": "tavily_key"},
                clear=False,
            ):
                config = Config(_env_file=None)
                assert config.tavily_api_key == "tavily_key"
        finally:
            # Restore original key if it existed
            if original_tavily:
                os.environ["TAVILY_API_KEY"] = original_tavily
        
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
        # Reset global config to ensure fresh load
        import src.config
        from src.config import _config, reload_config
        original_config = src.config._config
        src.config._config = None
        
        try:
            with patch.dict(
                os.environ,
                {"GROQ_API_KEY": "test_key", "LLM_MODEL": "test-model"},
                clear=False,
            ):
                # Use reload_config to ensure we get fresh config with test values
                config = reload_config()
                assert config.groq_api_key == "test_key"
                assert config.llm_model == "test-model"
        finally:
            # Restore original config
            src.config._config = original_config


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
        import src.config
        from src.config import reload_config

        # Reset global config to ensure fresh load
        original_config = src.config._config
        src.config._config = None
        
        try:
            with patch.dict(
                os.environ,
                {"GROQ_API_KEY": "test_key", "LLM_MODEL": "model1"},
                clear=False,
            ):
                config1 = reload_config()
                assert config1.llm_model == "model1"
                
                # Change environment
                os.environ["LLM_MODEL"] = "model2"
                config2 = reload_config()
                
                assert config2.llm_model == "model2"
                # Old config should still have old value
                assert config1.llm_model == "model1"
        finally:
            # Restore original config
            src.config._config = original_config


class TestValidationThresholds:
    """Tests for validation threshold configuration fields."""
    
    def test_validation_thresholds_use_defaults(self) -> None:
        """Test validation thresholds use default values when not set."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}, clear=False):
            config = Config()
            
            assert config.min_insights == 8
            assert config.min_positioning_length == 50
            assert config.min_report_length == 1200
            assert config.min_swot_items_per_category == 2
            assert config.min_trends == 2
            assert config.min_opportunities == 2
            assert config.min_collector_sources == 4
    
    def test_validation_thresholds_load_from_environment(self) -> None:
        """Test validation thresholds load from environment variables."""
        with patch.dict(
            os.environ,
            {
                "GROQ_API_KEY": "test_key",
                "MIN_INSIGHTS": "10",
                "MIN_POSITIONING_LENGTH": "100",
                "MIN_REPORT_LENGTH": "2000",
                "MIN_SWOT_ITEMS_PER_CATEGORY": "3",
                "MIN_TRENDS": "5",
                "MIN_OPPORTUNITIES": "4",
                "MIN_COLLECTOR_SOURCES": "6",
            },
            clear=False,
        ):
            config = Config()
            
            assert config.min_insights == 10
            assert config.min_positioning_length == 100
            assert config.min_report_length == 2000
            assert config.min_swot_items_per_category == 3
            assert config.min_trends == 5
            assert config.min_opportunities == 4
            assert config.min_collector_sources == 6
    
    def test_validation_thresholds_validate_ranges(self) -> None:
        """Test validation thresholds validate ranges correctly."""
        # Test min_insights range
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "MIN_INSIGHTS": "0"},
            clear=False,
        ):
            with pytest.raises(ValidationError):
                Config()
        
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "MIN_INSIGHTS": "101"},
            clear=False,
        ):
            with pytest.raises(ValidationError):
                Config()
        
        # Test min_positioning_length range
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "MIN_POSITIONING_LENGTH": "5"},
            clear=False,
        ):
            with pytest.raises(ValidationError):
                Config()
        
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "MIN_POSITIONING_LENGTH": "1001"},
            clear=False,
        ):
            with pytest.raises(ValidationError):
                Config()
        
        # Test min_report_length range
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "MIN_REPORT_LENGTH": "50"},
            clear=False,
        ):
            with pytest.raises(ValidationError):
                Config()
        
        # Test min_swot_items_per_category range
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "MIN_SWOT_ITEMS_PER_CATEGORY": "0"},
            clear=False,
        ):
            with pytest.raises(ValidationError):
                Config()
        
        # Test min_trends range
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "MIN_TRENDS": "0"},
            clear=False,
        ):
            with pytest.raises(ValidationError):
                Config()
        
        # Test min_opportunities range
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "MIN_OPPORTUNITIES": "0"},
            clear=False,
        ):
            with pytest.raises(ValidationError):
                Config()
        
        # Test min_collector_sources range
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test_key", "MIN_COLLECTOR_SOURCES": "0"},
            clear=False,
        ):
            with pytest.raises(ValidationError):
                Config()
    
    def test_validation_thresholds_backward_compatibility(self) -> None:
        """Test that validation thresholds maintain backward compatibility with defaults."""
        # Test that defaults match original hardcoded values
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}, clear=False):
            config = Config()
            
            # These defaults should match the original hardcoded values
            # from validators before this change
            assert config.min_insights == 8  # Original: MIN_INSIGHTS = 8
            assert config.min_positioning_length == 50  # Original: MIN_POSITIONING_LENGTH = 50
            assert config.min_report_length == 1200  # Original: min_length = 1200 in workflow.py
            assert config.min_swot_items_per_category == 2  # Original: MIN_SWOT_ITEMS_PER_CATEGORY = 2
            assert config.min_trends == 2  # Original: MIN_TRENDS = 2
            assert config.min_opportunities == 2  # Original: MIN_OPPORTUNITIES = 2
            assert config.min_collector_sources == 4  # Original: MIN_SOURCES = 4


