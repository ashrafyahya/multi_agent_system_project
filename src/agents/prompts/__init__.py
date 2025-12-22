"""Prompts module for agent system prompts.

This module contains system prompts used by various agents in the system.
Extracting prompts to separate modules improves code organization and
maintainability.
"""

from src.agents.prompts.insight_agent_prompts import \
    SYSTEM_PROMPT as INSIGHT_AGENT_SYSTEM_PROMPT
from src.agents.prompts.planner_agent_prompts import \
    SYSTEM_PROMPT as PLANNER_AGENT_SYSTEM_PROMPT
from src.agents.prompts.report_agent_prompts import \
    SYSTEM_PROMPT as REPORT_AGENT_SYSTEM_PROMPT

__all__ = [
    "INSIGHT_AGENT_SYSTEM_PROMPT",
    "PLANNER_AGENT_SYSTEM_PROMPT",
    "REPORT_AGENT_SYSTEM_PROMPT",
]

