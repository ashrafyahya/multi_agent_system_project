"""Pytest configuration and shared fixtures.

This module contains pytest configuration and shared fixtures used
across all test files.
"""

import pytest
from unittest.mock import Mock, patch

from langchain_core.language_models import BaseChatModel

from src.config import Config
from src.utils.llm_cache import clear_cache


@pytest.fixture
def mock_llm() -> Mock:
    """Create a mock LLM instance."""
    llm = Mock(spec=BaseChatModel)
    return llm


@pytest.fixture
def mock_config() -> Mock:
    """Create a mock Config instance."""
    config = Mock(spec=Config)
    config.groq_api_key = "test_api_key"
    config.llm_model = "llama-3.1-8b-instant"
    config.max_retries = 3
    config.log_level = "INFO"
    config.data_dir = None
    config.tavily_api_key = "test_tavily_api_key"
    config.llm_model_planner = None
    config.llm_model_supervisor = None
    config.llm_model_insight = None
    config.llm_model_report = None
    config.llm_model_collector = None
    config.llm_model_export = None
    config.get_model_for_agent = Mock(return_value="llama-3.1-8b-instant")
    return config


@pytest.fixture
def workflow_config() -> dict:
    """Create a standard workflow configuration."""
    return {
        "max_retries": 3,
        "temperature": 0,
        "planner_temperature": 0,
        "supervisor_temperature": 0,
        "collector_temperature": 0,
        "insight_temperature": 0.7,
        "report_temperature": 0.7,
    }


@pytest.fixture(autouse=True)
def clear_llm_cache():
    """Clear LLM cache before and after each test.
    
    This ensures test isolation - each test starts with a clean cache
    and doesn't leave cached responses for other tests. Tests that
    specifically want to test caching behavior can manage the cache
    themselves.
    """
    # Clear cache before each test
    clear_cache()
    yield
    # Clear cache after each test
    clear_cache()
