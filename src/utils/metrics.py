"""Metrics tracking for execution time, token usage, and API calls.

This module provides decorators and utilities for tracking performance metrics
across nodes and agents. It follows the decorator pattern established in Task 2
and integrates with the configuration system from Task 3.

The metrics system tracks:
- Execution time for nodes and agents
- Token usage from LLM calls (if available)
- API call counts
- Metrics aggregation and export
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Any, Callable, TypeVar

from src.config import get_config

logger = logging.getLogger(__name__)

# Type variable for function return type
T = TypeVar("T")


@dataclass
class ExecutionMetrics:
    """Metrics for a single execution.
    
    Attributes:
        name: Name of the function/component being tracked
        execution_time: Execution time in seconds
        timestamp: Unix timestamp when execution started
        token_usage: Optional dictionary with token usage info
            (prompt_tokens, completion_tokens, total_tokens)
        api_calls: Number of API calls made during execution
        success: Whether execution completed successfully
        error: Optional error message if execution failed
    """
    name: str
    execution_time: float
    timestamp: float
    token_usage: dict[str, int] | None = None
    api_calls: int = 0
    success: bool = True
    error: str | None = None


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a component.
    
    Attributes:
        name: Name of the component
        total_executions: Total number of executions
        successful_executions: Number of successful executions
        failed_executions: Number of failed executions
        total_execution_time: Total execution time in seconds
        average_execution_time: Average execution time in seconds
        min_execution_time: Minimum execution time in seconds
        max_execution_time: Maximum execution time in seconds
        total_tokens: Total tokens used (if available)
        total_api_calls: Total API calls made
        metrics: List of individual execution metrics
    """
    name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    min_execution_time: float = float("inf")
    max_execution_time: float = 0.0
    total_tokens: int = 0
    total_api_calls: int = 0
    metrics: list[ExecutionMetrics] = field(default_factory=list)


class MetricsCollector:
    """Collects and aggregates metrics from nodes and agents.
    
    This class provides thread-safe metrics collection and aggregation.
    It stores metrics in memory and can export them to JSON files.
    
    Attributes:
        _metrics: Dictionary mapping component names to lists of ExecutionMetrics
        _lock: Thread lock for thread-safe operations
    """
    
    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._metrics: dict[str, list[ExecutionMetrics]] = defaultdict(list)
        self._lock = Lock()
    
    def record_metrics(self, metrics: ExecutionMetrics) -> None:
        """Record execution metrics.
        
        Thread-safe method to record metrics for a component.
        
        Args:
            metrics: ExecutionMetrics object to record
        """
        with self._lock:
            self._metrics[metrics.name].append(metrics)
    
    def get_aggregated_metrics(self, name: str | None = None) -> dict[str, AggregatedMetrics]:
        """Get aggregated metrics for one or all components.
        
        Args:
            name: Optional component name. If provided, returns metrics for
                that component only. If None, returns metrics for all components.
        
        Returns:
            Dictionary mapping component names to AggregatedMetrics
        """
        with self._lock:
            if name:
                if name not in self._metrics:
                    return {}
                metrics_list = self._metrics[name]
            else:
                # Get all metrics
                metrics_list = []
                for component_metrics in self._metrics.values():
                    metrics_list.extend(component_metrics)
            
            # Group by name if getting all metrics
            if name:
                components = {name: metrics_list}
            else:
                components = dict(self._metrics)
            
            aggregated = {}
            for component_name, component_metrics in components.items():
                if not component_metrics:
                    continue
                
                successful = [m for m in component_metrics if m.success]
                failed = [m for m in component_metrics if not m.success]
                
                execution_times = [m.execution_time for m in component_metrics]
                total_time = sum(execution_times)
                avg_time = total_time / len(execution_times) if execution_times else 0.0
                
                total_tokens = sum(
                    (m.token_usage or {}).get("total_tokens", 0)
                    for m in component_metrics
                )
                
                total_api_calls = sum(m.api_calls for m in component_metrics)
                
                aggregated[component_name] = AggregatedMetrics(
                    name=component_name,
                    total_executions=len(component_metrics),
                    successful_executions=len(successful),
                    failed_executions=len(failed),
                    total_execution_time=total_time,
                    average_execution_time=avg_time,
                    min_execution_time=min(execution_times) if execution_times else 0.0,
                    max_execution_time=max(execution_times) if execution_times else 0.0,
                    total_tokens=total_tokens,
                    total_api_calls=total_api_calls,
                    metrics=component_metrics,
                )
            
            return aggregated
    
    def clear_metrics(self, name: str | None = None) -> None:
        """Clear metrics for one or all components.
        
        Args:
            name: Optional component name. If provided, clears metrics for
                that component only. If None, clears all metrics.
        """
        with self._lock:
            if name:
                if name in self._metrics:
                    del self._metrics[name]
            else:
                self._metrics.clear()
    
    def export_metrics(self, export_path: Path, filename: str | None = None) -> Path:
        """Export metrics to a JSON file.
        
        Args:
            export_path: Directory path where metrics file should be saved
            filename: Optional filename. If None, uses timestamp-based filename.
        
        Returns:
            Path to the exported metrics file
        """
        export_path.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        file_path = export_path / filename
        
        # Get all aggregated metrics
        aggregated = self.get_aggregated_metrics()
        
        # Convert to JSON-serializable format
        export_data = {
            "export_timestamp": time.time(),
            "components": {}
        }
        
        for component_name, metrics in aggregated.items():
            # Convert ExecutionMetrics to dict
            metrics_dict = asdict(metrics)
            # Convert ExecutionMetrics in the list to dicts
            metrics_dict["metrics"] = [asdict(m) for m in metrics.metrics]
            export_data["components"][component_name] = metrics_dict
        
        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {file_path}")
        return file_path


# Global metrics collector instance
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.
    
    Returns:
        Global MetricsCollector instance (singleton)
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def track_execution_time(
    component_name: str | None = None,
    track_tokens: bool = True,
    track_api_calls: bool = True,
) -> Callable:
    """Decorator to track execution time and other metrics.
    
    This decorator measures execution time and optionally tracks token usage
    and API calls. It integrates with the global MetricsCollector to aggregate
    metrics across multiple executions.
    
    The decorator:
    - Measures execution time
    - Tracks token usage from LLM responses (if available)
    - Counts API calls (if track_api_calls is True)
    - Records success/failure status
    - Stores metrics in MetricsCollector
    
    Args:
        component_name: Name of the component being tracked. If None, uses
            the function's __name__ attribute.
        track_tokens: Whether to track token usage from LLM responses.
            Only works if the function returns an object with token usage info.
        track_api_calls: Whether to track API call counts. Currently counts
            LLM invocations.
    
    Returns:
        Decorator function
    
    Example:
        ```python
        @track_execution_time("planner_agent")
        def execute(self, state: WorkflowState) -> WorkflowState:
            # Agent implementation
            return updated_state
        ```
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        """Inner decorator that wraps the function."""
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            """Wrapper function that tracks metrics."""
            config = get_config()
            
            # If metrics are disabled, just call the function
            if not config.metrics_enabled:
                return func(*args, **kwargs)
            
            # Determine component name
            name = component_name or func.__name__
            
            # Track execution time
            start_time = time.time()
            timestamp = start_time
            
            # Track API calls (if enabled)
            api_calls = 0
            token_usage: dict[str, int] | None = None
            success = True
            error: str | None = None
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Try to extract token usage from result
                if track_tokens:
                    token_usage = _extract_token_usage(result)
                
                # Count API calls (simplified - counts LLM invocations)
                if track_api_calls:
                    api_calls = _count_api_calls(result, args, kwargs)
                
                return result
                
            except Exception as e:
                success = False
                error = str(e)
                raise
                
            finally:
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Create metrics object
                metrics = ExecutionMetrics(
                    name=name,
                    execution_time=execution_time,
                    timestamp=timestamp,
                    token_usage=token_usage,
                    api_calls=api_calls,
                    success=success,
                    error=error,
                )
                
                # Record metrics
                collector = get_metrics_collector()
                collector.record_metrics(metrics)
                
                # Log metrics (debug level)
                logger.debug(
                    f"Metrics for {name}: "
                    f"time={execution_time:.3f}s, "
                    f"tokens={token_usage.get('total_tokens', 'N/A') if token_usage else 'N/A'}, "
                    f"api_calls={api_calls}, "
                    f"success={success}"
                )
        
        return wrapper
    
    return decorator


def _extract_token_usage(result: Any) -> dict[str, int] | None:
    """Extract token usage information from LLM response.
    
    Checks common attributes where LLM responses store token usage:
    - response_metadata (LangChain)
    - usage (OpenAI-style)
    - token_usage (custom)
    
    Args:
        result: Result object that might contain token usage info
    
    Returns:
        Dictionary with token usage (prompt_tokens, completion_tokens, total_tokens)
        or None if not available
    """
    if result is None:
        return None
    
    # Check for response_metadata (LangChain)
    if hasattr(result, "response_metadata"):
        metadata = result.response_metadata
        if isinstance(metadata, dict):
            usage = metadata.get("token_usage") or metadata.get("usage")
            if isinstance(usage, dict):
                return {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
    
    # Check for usage attribute (OpenAI-style)
    if hasattr(result, "usage"):
        usage = result.usage
        if hasattr(usage, "prompt_tokens"):
            return {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
    
    # Check for token_usage attribute (custom)
    if hasattr(result, "token_usage"):
        usage = result.token_usage
        if isinstance(usage, dict):
            return {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
    
    return None


def _count_api_calls(result: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    """Count API calls made during execution.
    
    This is a simplified implementation that counts LLM invocations.
    It checks if the function is an agent's execute method and counts
    LLM calls made through invoke_llm.
    
    Args:
        result: Result from the function
        args: Function arguments
        kwargs: Function keyword arguments
    
    Returns:
        Number of API calls (currently always 1 for agent executions,
        or 0 if not detectable)
    """
    # For agents, we assume at least one LLM call per execution
    # This is a simplified approach - in a real implementation, we might
    # track this more accurately by instrumenting the LLM invoke calls
    if args and hasattr(args[0], "invoke_llm"):
        # This looks like an agent execution
        return 1
    
    return 0

