"""Collector error exception.

This module defines the CollectorError exception raised when data collection
fails in the workflow. This includes failures from web search, scraping,
or other data collection operations.
"""

from src.exceptions.base import BaseWorkflowError


class CollectorError(BaseWorkflowError):
    """Raised when data collection fails.
    
    This exception is raised when data collection operations fail,
    such as web search failures, scraping errors, or API call failures.
    It should include context about what was being collected and why
    it failed.
    """
    
    pass
