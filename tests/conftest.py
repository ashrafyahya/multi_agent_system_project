"""Pytest configuration and shared fixtures.

This module contains pytest configuration and shared fixtures used
across all test files.
"""

import pytest
from unittest.mock import Mock

from langchain_core.language_models import BaseChatModel

from src.config import Config


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
    config.groq_model = "llama-3.1-8b-instant"
    config.max_retries = 3
    config.log_level = "INFO"
    config.data_dir = None
    config.tavily_api_key = None
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
