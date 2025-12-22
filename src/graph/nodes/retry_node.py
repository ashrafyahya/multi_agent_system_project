"""Retry node for handling retry logic in the workflow.

Pure function node that handles retry logic, modifies search queries,
and increments retry count. Supports intelligent retry using LLM to
analyze validation errors and improve queries.
"""

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.config import get_config
from src.exceptions.workflow_error import WorkflowError
from src.graph.nodes.base_node import node_error_handler
from src.graph.state import WorkflowState
from src.graph.state_utils import update_state
from src.utils.rate_limiter import invoke_llm_with_retry

logger = logging.getLogger(__name__)


def create_retry_node(
    max_retries: int = 3,
    llm: BaseChatModel | None = None
) -> Any:
    """Create a retry node function.
    
    This factory function creates a pure function node that handles retry logic.
    The node increments retry count, modifies search queries to improve them,
    and clears validation errors to allow retry.
    
    Args:
        max_retries: Maximum number of retry attempts allowed (default: 3)
        llm: Optional LLM instance for intelligent retry. If provided and
            intelligent_retry_enabled is True, uses LLM to analyze errors and
            improve queries. If None or disabled, uses rule-based enhancement.
    
    Returns:
        Pure function that takes WorkflowState and returns updated WorkflowState
    """
    @node_error_handler("retry_node")
    def retry_node(state: WorkflowState) -> WorkflowState:
        """Node that handles retry logic.
        
        Pure function that takes state and returns updated state.
        No side effects except state mutations. This node:
        1. Increments retry_count
        2. Modifies search queries in plan to improve them
        3. Clears validation_errors to allow retry
        4. Updates current_task to indicate retry
        
        Args:
            state: Current workflow state containing plan and retry_count
        
        Returns:
            Updated state with:
            - retry_count incremented
            - plan modified with improved queries
            - validation_errors cleared
            - current_task updated
        
        Raises:
            WorkflowError: If max retries exceeded or plan is missing
        """
        # Check if plan exists
        plan = state.get("plan")
        if not plan:
            raise WorkflowError(
                "Cannot retry without a plan",
                context={"state_keys": list(state.keys())}
            )
        
        # Get current retry count
        current_retry_count = state.get("retry_count", 0)
        new_retry_count = current_retry_count + 1
        
        # Check if max retries exceeded
        if new_retry_count > max_retries:
            logger.warning(
                f"Max retries ({max_retries}) exceeded. "
                f"Current retry count: {current_retry_count}"
            )
            new_state = update_state(
                state,
                retry_count=new_retry_count,
                current_task=f"Max retries ({max_retries}) exceeded"
            )
            # Don't clear validation_errors if max retries exceeded
            return new_state
        
        logger.info(
            f"Retry node: Incrementing retry count from {current_retry_count} to {new_retry_count}"
        )
        
        # Get validation errors for intelligent retry
        validation_errors = state.get("validation_errors", [])
        
        # Modify plan to improve queries (deep copy handled in helper)
        modified_plan = _modify_plan_for_retry(
            plan.copy(),
            new_retry_count,
            llm=llm,
            validation_errors=validation_errors
        )
        
        # Create new state with all updates
        new_state = update_state(
            state,
            retry_count=new_retry_count,
            plan=modified_plan,
            validation_errors=[],
            current_task=f"Retry attempt {new_retry_count}/{max_retries}"
        )
        
        logger.info(
            f"Retry node completed: Modified plan with {len(modified_plan.get('tasks', []))} tasks"
        )
        
        return new_state
    
    return retry_node


def _modify_plan_for_retry(
    plan: dict[str, Any],
    retry_count: int,
    llm: BaseChatModel | None = None,
    validation_errors: list[str] | None = None
) -> dict[str, Any]:
    """Modify plan to improve search queries for retry.
    
    This function enhances the plan's tasks and queries to improve search
    results on retry. It can use LLM for intelligent query improvement based
    on validation errors, or fall back to rule-based enhancement.
    
    Args:
        plan: Plan dictionary to modify
        retry_count: Current retry count (used to determine modification intensity)
        llm: Optional LLM instance for intelligent retry. If provided and
            intelligent_retry_enabled is True, uses LLM to analyze errors and
            improve queries.
        validation_errors: Optional list of validation error messages to analyze
    
    Returns:
        Modified plan dictionary with improved queries
    """
    config = get_config()
    modified_plan = plan.copy()
    tasks = modified_plan.get("tasks", [])
    
    if not tasks:
        return modified_plan
    
    # Try intelligent retry if enabled and LLM is available
    if (
        config.intelligent_retry_enabled
        and llm is not None
        and validation_errors
        and len(validation_errors) > 0
    ):
        try:
            logger.info(
                f"Using intelligent retry with LLM to improve queries "
                f"based on {len(validation_errors)} validation errors"
            )
            modified_tasks = _intelligent_enhance_tasks(
                tasks, retry_count, llm, validation_errors
            )
            modified_plan["tasks"] = modified_tasks
            logger.info("Intelligent retry completed successfully")
        except Exception as e:
            logger.warning(
                f"Intelligent retry failed, falling back to rule-based enhancement: {e}"
            )
            # Fall back to rule-based enhancement
            modified_tasks = _rule_based_enhance_tasks(tasks, retry_count)
            modified_plan["tasks"] = modified_tasks
    else:
        # Use rule-based enhancement
        logger.debug("Using rule-based query enhancement for retry")
        modified_tasks = _rule_based_enhance_tasks(tasks, retry_count)
        modified_plan["tasks"] = modified_tasks
    
    # Increase minimum_results if specified (to get more data on retry)
    if "minimum_results" in modified_plan:
        current_min = modified_plan.get("minimum_results", 4)
        # Increase by 20% per retry, rounded up
        modified_plan["minimum_results"] = int(current_min * (1 + 0.2 * retry_count))
    
    return modified_plan


def _intelligent_enhance_tasks(
    tasks: list[str],
    retry_count: int,
    llm: BaseChatModel,
    validation_errors: list[str]
) -> list[str]:
    """Use LLM to intelligently enhance tasks based on validation errors.
    
    This function uses an LLM to analyze validation errors and generate
    improved search queries that address the issues found.
    
    Args:
        tasks: List of original task strings
        retry_count: Current retry count
        llm: LLM instance for query improvement
        validation_errors: List of validation error messages
    
    Returns:
        List of enhanced task strings
    """
    # Create prompt for LLM
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a query improvement assistant. Your task is to analyze validation errors and improve search queries to address the issues.

Given the original search queries and validation errors, generate improved queries that:
1. Address the specific issues mentioned in the validation errors
2. Are more specific and targeted
3. Include relevant keywords that will yield better results
4. Maintain the original intent while improving clarity

Return a JSON array of improved queries, one for each original query. Each improved query should be a string."""),
        HumanMessage(content=f"""Original queries:
{json.dumps(tasks, indent=2)}

Validation errors:
{json.dumps(validation_errors, indent=2)}

Retry attempt: {retry_count}

Generate improved queries as a JSON array of strings. Return only the JSON array, no additional text.""")
    ])
    
    # Invoke LLM with retry logic
    messages = prompt.format_messages()
    response = invoke_llm_with_retry(llm, messages, temperature=0.3)
    
    # Parse response
    content = response.content.strip()
    
    # Try to extract JSON from response (handle markdown code blocks)
    if content.startswith("```"):
        # Extract JSON from code block
        lines = content.split("\n")
        json_start = None
        json_end = None
        for i, line in enumerate(lines):
            if line.strip().startswith("```json") or line.strip().startswith("```"):
                json_start = i + 1
            elif line.strip() == "```" and json_start is not None:
                json_end = i
                break
        if json_start is not None and json_end is not None:
            content = "\n".join(lines[json_start:json_end])
        elif json_start is not None:
            content = "\n".join(lines[json_start:])
    
    try:
        enhanced_tasks = json.loads(content)
        if not isinstance(enhanced_tasks, list):
            raise ValueError("Response is not a list")
        if len(enhanced_tasks) != len(tasks):
            logger.warning(
                f"LLM returned {len(enhanced_tasks)} queries but expected {len(tasks)}, "
                "using rule-based fallback"
            )
            return _rule_based_enhance_tasks(tasks, retry_count)
        # Validate all items are strings
        if not all(isinstance(task, str) for task in enhanced_tasks):
            logger.warning("LLM returned non-string items, using rule-based fallback")
            return _rule_based_enhance_tasks(tasks, retry_count)
        return enhanced_tasks
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}, using rule-based fallback")
        return _rule_based_enhance_tasks(tasks, retry_count)


def _rule_based_enhance_tasks(tasks: list[str], retry_count: int) -> list[str]:
    """Enhance tasks using rule-based approach (fallback).
    
    Args:
        tasks: List of original task strings
        retry_count: Current retry count
    
    Returns:
        List of enhanced task strings
    """
    modified_tasks = []
    for task in tasks:
        if isinstance(task, str):
            enhanced_task = _enhance_task_query(task, retry_count)
            modified_tasks.append(enhanced_task)
        else:
            # Keep non-string tasks as-is
            modified_tasks.append(task)
    return modified_tasks


def _enhance_task_query(task: str, retry_count: int) -> str:
    """Enhance a task query with additional context for retry.
    
    Adds more specific terms, context keywords, and refinements to improve
    search query effectiveness on retry.
    
    Args:
        task: Original task string
        retry_count: Current retry count (higher = more aggressive enhancement)
    
    Returns:
        Enhanced task string with improved query terms
    """
    task = task.strip()
    
    # Add context keywords based on retry count
    enhancements = []
    
    if retry_count >= 2:
        # On second retry, add more specific terms
        if "competitor" not in task.lower():
            enhancements.append("competitor")
        if "analysis" not in task.lower():
            enhancements.append("analysis")
    
    if retry_count >= 3:
        # On third retry, add comparison and market terms
        if "comparison" not in task.lower():
            enhancements.append("comparison")
        if "market" not in task.lower():
            enhancements.append("market")
    
    # Build enhanced query
    if enhancements:
        enhanced = f"{task} {' '.join(enhancements)}"
    else:
        enhanced = task
    
    # Add specificity indicators
    if retry_count > 1:
        enhanced = f"{enhanced} detailed comprehensive"
    
    return enhanced.strip()

