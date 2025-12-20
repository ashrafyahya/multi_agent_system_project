"""Agent output logger for capturing and storing agent outputs to files.

This module provides the AgentLogger class that logs agent outputs to
timestamped plain text files. Each agent's output is logged to a separate
file, making it easy to track what each agent produced during workflow execution.

Example:
    ```python
    from src.utils.agent_logger import AgentLogger
    from pathlib import Path
    
    logger = AgentLogger(log_dir=Path("./data/agent_logs"), enabled=True)
    logger.log_agent_output("planner_agent", {"plan": {...}}, state)
    ```
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.graph.state import WorkflowState

logger = logging.getLogger(__name__)


class AgentLogger:
    """Logger for capturing and storing agent outputs to files.
    
    This class provides functionality to log agent outputs to timestamped
    plain text files. Each agent's output is logged to a separate file
    with metadata including timestamp, agent name, and execution context.
    
    The logger handles file I/O errors gracefully and will not disrupt
    workflow execution if logging fails.
    
    Attributes:
        log_dir: Directory where log files will be stored
        enabled: Whether logging is enabled (if False, all logging operations are no-ops)
    
    Example:
        ```python
        from pathlib import Path
        from src.utils.agent_logger import AgentLogger
        
        logger = AgentLogger(
            log_dir=Path("./data/agent_logs"),
            enabled=True
        )
        
        # Log agent output
        output = {"plan": {"tasks": ["task1", "task2"]}}
        state = {...}  # WorkflowState
        logger.log_agent_output("planner_agent", output, state)
        ```
    """
    
    def __init__(self, log_dir: Path, enabled: bool = True) -> None:
        """Initialize the agent logger.
        
        Args:
            log_dir: Directory where log files will be stored. Will be created
                if it doesn't exist.
            enabled: Whether logging is enabled. If False, all logging
                operations become no-ops.
        
        Example:
            ```python
            from pathlib import Path
            
            logger = AgentLogger(
                log_dir=Path("./data/agent_logs"),
                enabled=True
            )
            ```
        """
        self.log_dir = log_dir
        self.enabled = enabled
        
        if self.enabled:
            self._ensure_log_directory()
    
    def log_agent_output(
        self,
        agent_name: str,
        output: dict[str, Any] | str | None,
        state: WorkflowState
    ) -> None:
        """Log agent output to a file.
        
        Creates a timestamped log file for the agent and writes the formatted
        output. If logging is disabled or an error occurs, the operation is
        silently skipped to avoid disrupting workflow execution.
        
        Args:
            agent_name: Name of the agent (e.g., "planner_agent", "insight_agent")
            output: Agent output to log. Can be a dictionary, string, or None.
            state: Current workflow state for context information
        
        Example:
            ```python
            logger.log_agent_output(
                "planner_agent",
                {"plan": {"tasks": ["task1"]}},
                state
            )
            ```
        """
        if not self.enabled:
            return
        
        try:
            log_file_path = self._get_log_file_path(agent_name)
            formatted_output = self._format_output(agent_name, output, state)
            
            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write(formatted_output)
            
            logger.debug(f"Agent output logged to: {log_file_path}")
            
        except Exception as e:
            # Log error but don't disrupt workflow
            logger.warning(
                f"Failed to log agent output for {agent_name}: {e}",
                exc_info=True
            )
    
    def _format_output(
        self,
        agent_name: str,
        output: dict[str, Any] | str | None,
        state: WorkflowState
    ) -> str:
        """Format agent output as readable plain text.
        
        Args:
            agent_name: Name of the agent
            output: Agent output to format
            state: Current workflow state for context
        
        Returns:
            Formatted plain text string ready to write to log file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        retry_count = state.get("retry_count", 0)
        current_task = state.get("current_task", "Unknown")
        validation_errors = state.get("validation_errors", [])
        
        # Determine workflow stage based on agent name
        stage_map = {
            "planner_agent": "Planning",
            "supervisor_agent": "Supervision",
            "data_collector_agent": "Data Collection",
            "insight_agent": "Insight Generation",
            "report_agent": "Report Generation",
            "export_agent": "Export Generation",
        }
        workflow_stage = stage_map.get(agent_name, "Unknown")
        
        # Format output section
        output_section = self._format_output_section(output, agent_name)
        
        # Build formatted text
        lines = [
            "=== Agent Output Log ===",
            f"Agent: {agent_name}",
            f"Timestamp: {timestamp}",
            f"Workflow Stage: {workflow_stage}",
            f"Retry Count: {retry_count}",
            "",
            "--- Output ---",
            output_section,
            "",
            "--- Context ---",
            f"Current Task: {current_task}",
            f"Validation Errors: {len(validation_errors)}",
        ]
        
        if validation_errors:
            lines.append("")
            lines.append("Validation Error Details:")
            for i, error in enumerate(validation_errors, 1):
                lines.append(f"  {i}. {error}")
        
        return "\n".join(lines)
    
    def _format_output_section(
        self,
        output: dict[str, Any] | str | None,
        agent_name: str
    ) -> str:
        """Format the output section based on output type.
        
        Args:
            output: Agent output to format
            agent_name: Name of the agent (for type-specific formatting)
        
        Returns:
            Formatted output string
        """
        if output is None:
            return "No output generated"
        
        # Handle string output (e.g., report)
        if isinstance(output, str):
            if agent_name == "report_agent":
                # For reports, include length and full content
                length = len(output)
                return f"Report Length: {length} characters\n\n{output}"
            return output
        
        # Handle dictionary output
        if isinstance(output, dict):
            # Format as JSON with indentation for readability
            try:
                json_str = json.dumps(output, indent=2, ensure_ascii=False)
                
                # Add summary statistics for structured data
                summary = self._generate_summary(output, agent_name)
                if summary:
                    return f"{summary}\n\n{json_str}"
                return json_str
            except (TypeError, ValueError) as e:
                # Fallback if JSON serialization fails
                return f"Output (non-serializable): {str(output)[:500]}"
        
        # Fallback for other types
        return str(output)
    
    def _generate_summary(
        self,
        output: dict[str, Any],
        agent_name: str
    ) -> str:
        """Generate summary statistics for structured output.
        
        Args:
            output: Agent output dictionary
            agent_name: Name of the agent
        
        Returns:
            Summary string or empty string if no summary available
        """
        summaries = []
        
        if agent_name == "planner_agent":
            tasks = output.get("tasks", [])
            if tasks:
                summaries.append(f"Tasks Planned: {len(tasks)}")
            minimum_results = output.get("minimum_results")
            if minimum_results is not None:
                summaries.append(f"Minimum Results Required: {minimum_results}")
        
        elif agent_name == "data_collector_agent":
            competitors = output.get("competitors", [])
            if competitors:
                summaries.append(f"Competitors Collected: {len(competitors)}")
        
        elif agent_name == "insight_agent":
            swot = output.get("swot", {})
            if swot:
                strengths = len(swot.get("strengths", []))
                weaknesses = len(swot.get("weaknesses", []))
                opportunities = len(swot.get("opportunities", []))
                threats = len(swot.get("threats", []))
                summaries.append(
                    f"SWOT Analysis: {strengths} strengths, {weaknesses} weaknesses, "
                    f"{opportunities} opportunities, {threats} threats"
                )
            trends = output.get("trends", [])
            if trends:
                summaries.append(f"Market Trends Identified: {len(trends)}")
        
        elif agent_name == "export_agent":
            # For export_agent, output is actually export_paths dict
            if isinstance(output, dict):
                export_count = len(output)
                summaries.append(f"Export Files Generated: {export_count}")
                for export_type, path in output.items():
                    summaries.append(f"  - {export_type}: {path}")
        
        elif agent_name == "supervisor_agent":
            # Supervisor output is typically in state, not separate output
            # But we can summarize from the output dict if it contains decisions
            if "next_action" in output:
                summaries.append(f"Next Action: {output['next_action']}")
            if "validation_passed" in output:
                summaries.append(
                    f"Validation Passed: {output['validation_passed']}"
                )
        
        return "\n".join(summaries) if summaries else ""
    
    def _get_log_file_path(self, agent_name: str) -> Path:
        """Get the log file path for an agent.
        
        Args:
            agent_name: Name of the agent
        
        Returns:
            Path object for the log file
        
        Example:
            ```python
            path = logger._get_log_file_path("planner_agent")
            # Returns: Path("./data/agent_logs/planner_agent_20241219_212734.log")
            ```
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Normalize agent name for filename (replace spaces with underscores)
        safe_agent_name = agent_name.replace(" ", "_").lower()
        filename = f"{safe_agent_name}_{timestamp}.log"
        return self.log_dir / filename
    
    def _ensure_log_directory(self) -> None:
        """Ensure the log directory exists, creating it if necessary.
        
        Raises:
            OSError: If directory creation fails
        """
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create log directory {self.log_dir}: {e}")
            raise

