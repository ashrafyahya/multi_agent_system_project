"""LangGraph workflow builder for competitor analysis.

This module builds the complete StateGraph with all nodes, conditional edges,
validation gates, and retry logic.

Example:
    ```python
    from src.graph.workflow import create_workflow
    from langchain_groq import ChatGroq
    from src.graph.state import create_initial_state
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    config = {"max_retries": 3, "temperature": 0}
    workflow = create_workflow(llm=llm, config=config)
    
    state = create_initial_state("Analyze competitors in SaaS market")
    result = workflow.invoke(state)
    print(result["report"])
    ```
"""

import logging
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from src.graph.nodes.data_collector_node import create_data_collector_node
from src.graph.nodes.export_node import create_export_node
from src.graph.nodes.insight_node import create_insight_node
from src.graph.nodes.planner_node import create_planner_node
from src.graph.nodes.report_node import create_report_node
from src.graph.nodes.retry_node import create_retry_node
from src.graph.nodes.supervisor_node import create_supervisor_node
from src.graph.state import WorkflowState
from src.graph.validators.collector_validator import CollectorValidator
from src.graph.validators.insight_validator import InsightValidator
from src.graph.validators.report_validator import ReportValidator

logger = logging.getLogger(__name__)


def create_workflow(
    llm: BaseChatModel,
    config: dict[str, Any]
) -> Any:
    """Create and configure the competitor analysis workflow.
    
    This function builds a complete LangGraph StateGraph with all nodes,
    conditional edges, validation gates, and retry logic. The workflow
    follows this flow:
    
    1. planner: Generates execution plan from user request
    2. supervisor: Controls workflow flow and validates outputs
    3. collector: Collects competitor data
    4. insight: Transforms data into business insights
    5. report: Generates final formatted report
    6. export: Generates PDF and image exports from report
    7. retry: Handles retry logic when validation fails
    
    Conditional edges use validators to determine next steps:
    - If validation passes → continue to next stage
    - If validation fails and retries available → go to retry node
    - If validation fails and max retries exceeded → end workflow
    
    Args:
        llm: Language model instance for all agents
        config: Configuration dictionary containing:
            - max_retries: Maximum retry attempts (default: 3)
            - temperature: LLM temperature (default: 0 for planner/supervisor, 0.7 for others)
    
    Returns:
        Compiled StateGraph ready for execution
    
    Example:
        ```python
        from langchain_groq import ChatGroq
        from src.graph.workflow import create_workflow
        from src.graph.state import create_initial_state
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        config = {"max_retries": 3}
        workflow = create_workflow(llm=llm, config=config)
        
        state = create_initial_state("Analyze competitors")
        result = workflow.invoke(state)
        ```
    """
    # Extract configuration
    max_retries = config.get("max_retries", 3)
    planner_config = config.copy()
    planner_config["temperature"] = config.get("planner_temperature", 0)
    supervisor_config = config.copy()
    supervisor_config["temperature"] = config.get("supervisor_temperature", 0)
    collector_config = config.copy()
    collector_config["temperature"] = config.get("collector_temperature", 0)
    insight_config = config.copy()
    insight_config["temperature"] = config.get("insight_temperature", 0.7)
    report_config = config.copy()
    report_config["temperature"] = config.get("report_temperature", 0.7)
    export_config = config.copy()
    export_config["export_format"] = config.get("export_format", "pdf")
    export_config["include_visualizations"] = config.get("include_visualizations", True)
    
    # Create node functions
    planner_node_func = create_planner_node(llm=llm, config=planner_config)
    supervisor_node_func = create_supervisor_node(llm=llm, config=supervisor_config)
    collector_node_func = create_data_collector_node(llm=llm, config=collector_config)
    insight_node_func = create_insight_node(llm=llm, config=insight_config)
    report_node_func = create_report_node(llm=llm, config=report_config)
    export_node_func = create_export_node(llm=llm, config=export_config)
    retry_node_func = create_retry_node(max_retries=max_retries)
    
    # Build graph
    graph = StateGraph(WorkflowState)
    
    # Add nodes
    graph.add_node("planner", planner_node_func)
    graph.add_node("supervisor", supervisor_node_func)
    graph.add_node("collector", collector_node_func)
    graph.add_node("insight", insight_node_func)
    graph.add_node("report", report_node_func)
    graph.add_node("export", export_node_func)
    graph.add_node("retry", retry_node_func)
    
    # Set entry point
    graph.set_entry_point("planner")
    
    # Add edges
    # planner -> supervisor
    graph.add_edge("planner", "supervisor")
    
    # supervisor -> collector (if no data yet) or validate and decide
    graph.add_conditional_edges(
        "supervisor",
        lambda state: _supervisor_decision(state, max_retries),
        {
            "collector": "collector",
            "insight": "insight",
            "report": "report",
            "export": "export",
            "retry": "retry",
            END: END,
        }
    )
    
    # collector -> validate -> insight or retry or END
    graph.add_conditional_edges(
        "collector",
        lambda state: _validate_collector_output(state, max_retries),
        {
            "insight": "insight",
            "retry": "retry",
            END: END,
        }
    )
    
    # insight -> validate -> report or retry or END
    graph.add_conditional_edges(
        "insight",
        lambda state: _validate_insight_output(state, max_retries),
        {
            "report": "report",
            "retry": "retry",
            END: END,
        }
    )
    
    # report -> validate -> export or retry or END
    graph.add_conditional_edges(
        "report",
        lambda state: _validate_report_output(state, max_retries),
        {
            "export": "export",
            "retry": "retry",
            END: END,
        }
    )
    
    # export -> END (always succeeds, no validation needed)
    graph.add_edge("export", END)
    
    # retry -> supervisor (to re-evaluate)
    graph.add_edge("retry", "supervisor")
    
    logger.info("Workflow graph built successfully")
    
    return graph.compile()


def _supervisor_decision(
    state: WorkflowState,
    max_retries: int
) -> Literal["collector", "insight", "report", "export", "retry", END]:
    """Determine next step based on supervisor state.
    
    The supervisor node validates outputs and updates the state. This function
    determines the next step based on what data exists and validation results.
    
    Args:
        state: Current workflow state
        max_retries: Maximum retry attempts
    
    Returns:
        Next node name or END
    """
    retry_count = state.get("retry_count", 0)
    validation_errors = state.get("validation_errors", [])
    
    # Check if we have data at each stage
    has_plan = bool(state.get("plan"))
    has_collected_data = bool(state.get("collected_data"))
    has_insights = bool(state.get("insights"))
    has_report = bool(state.get("report"))
    has_export = bool(state.get("export_paths"))
    
    # If max retries exceeded and we have validation errors, end workflow
    if retry_count >= max_retries and validation_errors:
        logger.warning(
            f"Max retries ({max_retries}) exceeded with {len(validation_errors)} errors, "
            "ending workflow"
        )
        return END
    
    # Route based on what data we have and validation status
    # If we have exports, we're done
    if has_export:
        return END
    
    # If we have a report but no exports, go to export
    if has_report and not has_export:
        return "export"
    
    # If we have insights but no report, go to report
    if has_insights and not has_report:
        return "report"
    
    # If we have collected data but no insights, go to insight
    if has_collected_data and not has_insights:
        return "insight"
    
    # If we have a plan but no collected data, go to collector
    if has_plan and not has_collected_data:
        return "collector"
    
    # If we have validation errors and retries available, go to retry
    if validation_errors and retry_count < max_retries:
        return "retry"
    
    # Default: if we have a plan, try collector
    if has_plan:
        return "collector"
    
    # No plan means we can't proceed
    logger.error("No plan found in state, ending workflow")
    return END


def _validate_collector_output(
    state: WorkflowState,
    max_retries: int
) -> Literal["insight", "retry", END]:
    """Validate collector output and decide next step.
    
    Args:
        state: Current workflow state
        max_retries: Maximum retry attempts
    
    Returns:
        Next node name or END
    """
    validator = CollectorValidator()
    collected_data = state.get("collected_data", {})
    result = validator.validate(collected_data)
    
    retry_count = state.get("retry_count", 0)
    
    if result.is_valid:
        logger.info("Collector validation passed, proceeding to insight")
        return "insight"
    elif retry_count < max_retries:
        logger.warning(
            f"Collector validation failed ({len(result.errors)} errors), "
            f"retrying (attempt {retry_count + 1}/{max_retries})"
        )
        return "retry"
    else:
        logger.error(
            f"Collector validation failed and max retries ({max_retries}) exceeded, "
            "ending workflow"
        )
        return END


def _validate_insight_output(
    state: WorkflowState,
    max_retries: int
) -> Literal["report", "retry", END]:
    """Validate insight output and decide next step.
    
    Args:
        state: Current workflow state
        max_retries: Maximum retry attempts
    
    Returns:
        Next node name or END
    """
    validator = InsightValidator()
    insights = state.get("insights", {})
    result = validator.validate(insights)
    
    retry_count = state.get("retry_count", 0)
    
    if result.is_valid:
        logger.info("Insight validation passed, proceeding to report")
        return "report"
    elif retry_count < max_retries:
        logger.warning(
            f"Insight validation failed ({len(result.errors)} errors), "
            f"retrying (attempt {retry_count + 1}/{max_retries})"
        )
        return "retry"
    else:
        logger.error(
            f"Insight validation failed and max retries ({max_retries}) exceeded, "
            "ending workflow"
        )
        return END


def _validate_report_output(
    state: WorkflowState,
    max_retries: int
) -> Literal["export", "retry", END]:
    """Validate report output and decide next step.
    
    Args:
        state: Current workflow state
        max_retries: Maximum retry attempts
    
    Returns:
        "export", "retry", or END
    """
    report = state.get("report", "")
    
    # Report is stored as a formatted string in state
    # We perform basic validation: check if report exists and has minimum length
    min_length = 500
    
    if report and len(report.strip()) >= min_length:
        logger.info("Report validation passed, proceeding to export")
        return "export"
    
    retry_count = state.get("retry_count", 0)
    
    if retry_count < max_retries:
        logger.warning(
            f"Report validation failed (length: {len(report) if report else 0}, "
            f"required: {min_length}), retrying (attempt {retry_count + 1}/{max_retries})"
        )
        return "retry"
    else:
        logger.error(
            f"Report validation failed and max retries ({max_retries}) exceeded, "
            "ending workflow"
        )
        return END
