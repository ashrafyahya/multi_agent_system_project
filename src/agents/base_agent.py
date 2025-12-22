"""Base agent class for all agents in the system.

This module defines the abstract base class that all agents must implement,
following the Agent Pattern with dependency injection.

The Agent Pattern ensures that:
- Agents are self-contained units with clear inputs/outputs
- Agents communicate through state objects, not direct method calls
- Agents are stateless (state passed in, not stored)
- Dependencies (LLM, config) are injected, not created internally
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, TypeVar

from langchain_core.language_models import BaseChatModel

from src.config import get_config
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState
from src.utils.rate_limiter import invoke_llm_with_retry, invoke_llm_with_retry_async

logger = logging.getLogger(__name__)

# Type variable for method return type
T = TypeVar("T")


def agent_error_handler(agent_name: str, operation_name: str) -> Callable:
    """Decorator for consistent agent method error handling.
    
    This decorator wraps agent methods (like _generate_plan, _generate_insights)
    to provide consistent error handling. It catches WorkflowErrors from rate
    limiters and wraps them with agent-specific messages while preserving context.
    
    The decorator handles:
    - WorkflowErrors from rate limiter (wraps with specific operation message)
    - Other WorkflowErrors (re-raises as-is)
    - Generic Exceptions (wraps in WorkflowError with context)
    
    Args:
        agent_name: Name of the agent (e.g., "planner_agent", "insight_agent")
        operation_name: Name of the operation (e.g., "plan", "insights", "report")
            Used in error messages like "Failed to generate {operation_name} from LLM"
    
    Returns:
        Decorator function that wraps the agent method
    
    Example:
        ```python
        @agent_error_handler("planner_agent", "plan")
        def _generate_plan(self, user_request: str) -> dict[str, Any]:
            # Method implementation
            # If rate limiter raises WorkflowError, it will be wrapped with
            # "Failed to generate plan from LLM"
            return plan_data
        ```
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        """Inner decorator that wraps the agent method."""
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            """Wrapper function that handles errors."""
            try:
                return func(*args, **kwargs)
            except WorkflowError as e:
                # If it's a WorkflowError from rate limiter, wrap with specific context
                error_msg = str(e).lower()
                if "llm api" in error_msg or "rate limit" in error_msg:
                    # Extract context from original error
                    original_context = getattr(e, "context", {})
                    
                    # Preserve original error message in context
                    operation_context = {
                        **original_context,
                        "agent": agent_name,
                        "operation": operation_name,
                        "original_error": str(e),
                    }
                    
                    # Wrap with agent-specific message
                    raise WorkflowError(
                        f"Failed to generate {operation_name} from LLM",
                        context=operation_context
                    ) from e
                # Re-raise other WorkflowErrors as-is
                raise
            except Exception as e:
                # Wrap unexpected errors in WorkflowError
                logger.error(
                    f"Unexpected error in {agent_name}.{func.__name__}: {e}",
                    exc_info=True
                )
                raise WorkflowError(
                    f"Failed to generate {operation_name} from LLM",
                    context={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "agent": agent_name,
                        "operation": func.__name__,
                    }
                ) from e
        
        return wrapper
    
    return decorator


class BaseAgent(ABC):
    """Base class for all agents in the system.
    
    This abstract base class defines the interface that all agents must
    implement. Agents follow the Agent Pattern:
    - Self-contained units with clear inputs/outputs
    - Communicate through state objects, not direct method calls
    - Stateless (state passed in, not stored)
    - Dependencies injected via constructor
    
    All concrete agent implementations must:
    1. Inherit from BaseAgent
    2. Implement the `execute` method
    3. Implement the `name` property
    4. Accept LLM and config via constructor
    
    Attributes:
        llm: Language model instance (injected dependency)
        config: Configuration dictionary (injected dependency)
    """
    
    def __init__(self, llm: BaseChatModel, config: dict[str, Any]) -> None:
        """Initialize agent with dependencies.
        
        This constructor uses dependency injection to provide the agent
        with its required dependencies. This allows for:
        - Easy testing (can inject mocks)
        - Flexible configuration (different LLMs/configs)
        - Loose coupling (agent doesn't create dependencies)
        
        Args:
            llm: Language model instance to use for agent operations.
                Must be a BaseChatModel instance (e.g., ChatGroq, ChatOpenAI).
            config: Configuration dictionary containing agent-specific settings.
                Common keys include:
                - "temperature": float (LLM temperature, default varies by agent)
                - "max_retries": int (retry attempts for operations)
                - Agent-specific configuration keys
        
        Raises:
            TypeError: If llm is not a BaseChatModel instance
            ValueError: If config is not a dictionary
        """
        if not isinstance(llm, BaseChatModel):
            raise TypeError(
                f"llm must be a BaseChatModel instance, got {type(llm).__name__}"
            )
        if not isinstance(config, dict):
            raise ValueError(
                f"config must be a dictionary, got {type(config).__name__}"
            )
        
        self.llm = llm
        self.config = config
    
    def invoke_llm(self, messages: list[Any], **kwargs: Any) -> Any:
        """Invoke LLM with automatic retry logic for rate limits.
        
        This method wraps llm.invoke() calls with retry logic using exponential
        backoff. It handles rate limit errors and transient failures gracefully.
        
        All agents should use this method instead of calling self.llm.invoke()
        directly to ensure consistent retry behavior across all LLM calls.
        
        Args:
            messages: List of messages to send to the LLM
            **kwargs: Additional keyword arguments to pass to llm.invoke()
        
        Returns:
            Response from LLM
        
        Raises:
            WorkflowError: If all retries are exhausted
        
        Example:
            messages = prompt.format_messages()
            response = self.invoke_llm(messages)
            content = response.content
        """
        return invoke_llm_with_retry(self.llm, messages, **kwargs)
    
    async def invoke_llm_async(self, messages: list[Any], **kwargs: Any) -> Any:
        """Invoke LLM asynchronously with automatic retry logic for rate limits.
        
        This is the async version of invoke_llm(). It uses llm.ainvoke() for
        non-blocking execution. This allows multiple LLM calls to run concurrently.
        
        All agents should use this method when async execution is enabled to ensure
        consistent retry behavior across all async LLM calls.
        
        Args:
            messages: List of messages to send to the LLM
            **kwargs: Additional keyword arguments to pass to llm.ainvoke()
        
        Returns:
            Response from LLM
        
        Raises:
            WorkflowError: If all retries are exhausted
        
        Example:
            messages = prompt.format_messages()
            response = await self.invoke_llm_async(messages)
            content = response.content
        
        Note:
            This method requires the LLM to support async operations (ainvoke method).
            If async is not supported, it will fall back to sync execution.
        """
        return await invoke_llm_with_retry_async(self.llm, messages, **kwargs)
    
    @abstractmethod
    def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute agent logic.
        
        This method contains the core logic for the agent. It receives
        the current workflow state, performs its operations, and returns
        an updated state. Agents should be stateless - all state is passed
        in and returned, not stored in instance variables.
        
        The agent should:
        1. Extract necessary information from the state
        2. Perform its operations (e.g., generate plan, collect data, analyze)
        3. Update the state with results
        4. Return the updated state
        
        Args:
            state: Current workflow state containing:
                - messages: List of conversation messages
                - plan: Optional execution plan (for collector/insight/report agents)
                - collected_data: Optional collected competitor data (for insight/report agents)
                - insights: Optional business insights (for report agent)
                - report: Optional final report
                - retry_count: Current retry count
                - current_task: Current task being executed
                - validation_errors: List of validation errors
        
        Returns:
            Updated WorkflowState with agent's results added. The agent should
            update the appropriate field (plan, collected_data, insights, or report)
            based on its role.
        
        Raises:
            WorkflowError: If agent execution fails critically and cannot be
                recovered. Most errors should be handled gracefully and added
                to validation_errors in the state.
        """
        pass
    
    async def execute_async(self, state: WorkflowState) -> WorkflowState:
        """Execute agent logic asynchronously.
        
        This is the async version of execute(). By default, it calls the sync
        execute() method wrapped in an async executor. Subclasses can override
        this method to provide true async implementations that can run I/O
        operations concurrently.
        
        When async is enabled in config, agents that perform multiple I/O
        operations (like DataCollectorAgent with multiple web searches) should
        override this method to use async tools and run operations in parallel.
        
        Args:
            state: Current workflow state (same as execute())
        
        Returns:
            Updated WorkflowState (same as execute())
        
        Raises:
            WorkflowError: If agent execution fails critically
        
        Note:
            Default implementation runs sync execute() in an executor.
            Override this method in subclasses for true async execution.
        """
        # Default: run sync execute in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, state)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return agent name.
        
        This property provides a unique identifier for the agent. It's used
        for logging, error messages, and identification purposes. Each agent
        type should have a distinct name.
        
        Returns:
            String identifier for this agent. Should be lowercase with underscores,
            e.g., "planner_agent", "data_collector_agent", "insight_agent"
        """
        pass
