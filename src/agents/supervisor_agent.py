"""Supervisor agent for workflow control and business rule enforcement.

This module implements the SupervisorAgent that controls workflow flow,
applies business rules, runs validation gates, and triggers retry loops.
"""

import logging
from typing import Any, Literal

from src.agents.base_agent import BaseAgent
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState
from src.graph.validators.collector_validator import CollectorValidator
from src.graph.validators.insight_validator import InsightValidator
from src.graph.validators.report_validator import ReportValidator

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Agent that controls workflow flow and enforces business rules.
    
    The supervisor agent is responsible for:
    1. Controlling workflow flow between nodes
    2. Applying business rules and quality standards
    3. Running validation gates at each stage
    4. Triggering retry loops when validation fails
    5. Managing task execution sequence
    
    The supervisor validates outputs at each stage:
    - Collector output → CollectorValidator
    - Insight output → InsightValidator
    - Report output → ReportValidator
    
    Based on validation results and retry count, the supervisor decides:
    - Continue to next stage (if validation passes)
    - Trigger retry (if validation fails and retries available)
    - End workflow (if max retries exceeded or workflow complete)
    
    Attributes:
        llm: Language model instance (injected, may be used for decision-making)
        config: Configuration dictionary (injected)
    """
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute supervisor logic to control workflow flow.
        
        Analyzes the current workflow state, validates outputs at each stage,
        and decides the next action based on validation results and business rules.
        
        The supervisor checks:
        1. If plan exists (required for workflow to proceed)
        2. If collected_data exists → validate with CollectorValidator
        3. If insights exist → validate with InsightValidator
        4. If report exists → validate with ReportValidator
        
        Based on validation results:
        - If validation passes → continue to next stage
        - If validation fails and retries available → trigger retry
        - If validation fails and max retries exceeded → end workflow
        
        Args:
            state: Current workflow state containing plan, collected_data,
                insights, report, retry_count, etc.
        
        Returns:
            Updated WorkflowState with:
            - current_task: Updated to reflect supervisor decision
            - retry_count: Incremented if retry triggered
            - validation_errors: Updated with validation errors if any
        
        Raises:
            WorkflowError: If workflow state is invalid or supervisor logic fails
        """
        try:
            new_state = state.copy()
            if "validation_errors" not in new_state:
                new_state["validation_errors"] = []
            max_retries = self.config.get("max_retries", 3)
            current_retry_count = state.get("retry_count", 0)
            
            # Check if plan exists (required for workflow)
            if not state.get("plan"):
                raise WorkflowError(
                    "Workflow cannot proceed without a plan",
                    context={"state_keys": list(state.keys())}
                )
            
            # Determine current workflow stage
            stage = self._determine_stage(state)
            logger.info(f"Supervisor: Current stage = {stage}, retry_count = {current_retry_count}")
            
            # Validate outputs based on stage
            validation_result = None
            next_action: Literal["continue", "retry", "end"] = "continue"
            
            # Get data from state
            collected_data = state.get("collected_data")
            insights = state.get("insights")
            report = state.get("report")
            
            if collected_data and not insights:
                validation_result = self._validate_collector_output(collected_data)
                if not validation_result.is_valid:
                    if current_retry_count < max_retries:
                        next_action = "retry"
                        new_state["retry_count"] = current_retry_count + 1
                        stage = "collector"  # Set stage to collector for proper message formatting
                    else:
                        next_action = "end"
                    new_state["validation_errors"].extend(validation_result.errors)
                    logger.warning(
                        f"Collector validation failed: {len(validation_result.errors)} errors"
                    )
                else:
                    logger.info("Collector validation passed")
                    next_action = "continue"
            
            if insights and next_action != "retry" and next_action != "end":
                validation_result = self._validate_insight_output(insights)
                if not validation_result.is_valid:
                    if current_retry_count < max_retries:
                        next_action = "retry"
                        new_state["retry_count"] = current_retry_count + 1
                        stage = "insight"  # Set stage to insight for proper message formatting
                    else:
                        next_action = "end"
                    new_state["validation_errors"].extend(validation_result.errors)
                    logger.warning(
                        f"Insight validation failed: {len(validation_result.errors)} errors"
                    )
                else:
                    logger.info("Insight validation passed")
                    if stage == "insight":
                        next_action = "continue"
            
            if report and next_action != "retry" and next_action != "end":
                validation_result = self._validate_report_output(report)
                if not validation_result.is_valid:
                    if current_retry_count < max_retries:
                        next_action = "retry"
                        new_state["retry_count"] = current_retry_count + 1
                        stage = "report"  # Set stage to report for proper message formatting
                    else:
                        next_action = "end"
                    new_state["validation_errors"].extend(validation_result.errors)
                    logger.warning(
                        f"Report validation failed: {len(validation_result.errors)} errors"
                    )
                else:
                    logger.info("Report validation passed")
                    next_action = "end"
            
            if next_action == "continue":
                if stage == "collector":
                    if not collected_data:
                        next_action = "continue"
                elif stage == "insight":
                    if not insights:
                        next_action = "continue"
                elif stage == "report":
                    if not report:
                        next_action = "continue"
            
            elif stage == "complete":
                next_action = "end"
            
            new_state["current_task"] = self._format_task_message(stage, next_action)
            
            if validation_result and validation_result.warnings:
                logger.info(f"Validation warnings: {len(validation_result.warnings)}")
                new_state["validation_errors"].extend(
                    [f"Warning: {w}" for w in validation_result.warnings]
                )
            
            logger.info(f"Supervisor decision: {next_action} (stage: {stage})")
            
            return new_state
            
        except WorkflowError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in supervisor agent: {e}", exc_info=True)
            raise WorkflowError(
                "Supervisor execution failed unexpectedly",
                context={"error": str(e)}
            ) from e
    
    def _determine_stage(self, state: WorkflowState) -> Literal["collector", "insight", "report", "complete"]:
        """Determine current workflow stage based on state.
        
        Args:
            state: Current workflow state
        
        Returns:
            Current stage: "collector", "insight", "report", or "complete"
        """
        if state.get("report"):
            return "complete"
        elif state.get("insights"):
            return "report"
        elif state.get("collected_data"):
            return "insight"
        else:
            return "collector"
    
    def _validate_collector_output(self, data: dict[str, Any]) -> Any:
        """Validate collector output using CollectorValidator.
        
        Args:
            data: Collected data dictionary
        
        Returns:
            ValidationResult from CollectorValidator
        """
        validator = CollectorValidator()
        return validator.validate(data)
    
    def _validate_insight_output(self, data: dict[str, Any]) -> Any:
        """Validate insight output using InsightValidator.
        
        Args:
            data: Insight data dictionary
        
        Returns:
            ValidationResult from InsightValidator
        """
        validator = InsightValidator()
        return validator.validate(data)
    
    def _validate_report_output(self, data: dict[str, Any] | str) -> Any:
        """Validate report output using ReportValidator.
        
        Args:
            data: Report data - can be a dict with sections or a string report
        
        Returns:
            ValidationResult from ReportValidator
        
        Note:
            If data is a string, it's assumed the report agent will provide
            structured data. For string reports, we do basic length validation.
        """
        validator = ReportValidator()
        
        # Handle string reports (basic validation)
        if isinstance(data, str):
            from src.graph.validators.base_validator import ValidationResult
            
            result = ValidationResult.success()
            if len(data.strip()) < 1200:
                result.add_error(
                    f"Report length ({len(data.strip())} chars) is less than "
                    "minimum required (1200 chars)"
                )
            return result
        
        # Handle dict reports (full validation)
        if isinstance(data, dict):
            # If dict has "report" key with string, extract it
            if "report" in data and isinstance(data["report"], str):
                report_str = data["report"]
                result = ValidationResult.success()
                if len(report_str.strip()) < 1200:
                    result.add_error(
                        f"Report length ({len(report_str.strip())} chars) is less than "
                        "minimum required (1200 chars)"
                    )
                return result
            # Otherwise, validate as structured report dict
            return validator.validate(data)
        
        # Invalid type
        from src.graph.validators.base_validator import ValidationResult
        result = ValidationResult.failure("Report must be a string or dictionary")
        return result
    
    def _format_task_message(
        self,
        stage: Literal["collector", "insight", "report", "complete"],
        action: Literal["continue", "retry", "end"]
    ) -> str:
        """Format task message based on stage and action.
        
        Args:
            stage: Current workflow stage
            action: Next action to take
        
        Returns:
            Formatted task message string
        """
        if action == "retry":
            return f"Retrying {stage} stage due to validation failures"
        elif action == "end":
            if stage == "complete":
                return "Workflow complete"
            else:
                return f"Workflow ended at {stage} stage (max retries exceeded)"
        else:
            if stage == "collector":
                return "Proceeding to data collection"
            elif stage == "insight":
                return "Proceeding to insight generation"
            elif stage == "report":
                return "Proceeding to report generation"
            else:
                return "Workflow in progress"
    
    @property
    def name(self) -> str:
        """Return agent name.
        
        Returns:
            String identifier for this agent
        """
        return "supervisor_agent"
