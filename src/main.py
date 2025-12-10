"""Main entry point for the Competitor Analysis Multi-Agent System.

This module provides the main entry point for running competitor analysis
workflows. It handles configuration loading, LLM initialization, workflow
creation, and execution.

Example:
    ```python
    from src.main import run_analysis
    
    result = run_analysis("Analyze competitors in the SaaS market")
    print(result["report"])
    ```
    
    Or as a command-line tool:
    ```bash
    python -m src.main "Analyze competitors in the SaaS market"
    ```
"""

import argparse
import logging
import sys
from typing import Any

from langchain_groq import ChatGroq

from src.config import get_config
from src.graph.state import WorkflowState, create_initial_state
from src.graph.workflow import create_workflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def initialize_llm(config: Any) -> ChatGroq:
    """Initialize Groq LLM with configuration.
    
    Args:
        config: Config instance from get_config()
    
    Returns:
        Initialized ChatGroq LLM instance
    
    Example:
        ```python
        from src.config import get_config
        from src.main import initialize_llm
        
        config = get_config()
        llm = initialize_llm(config)
        ```
    """
    logger.info(f"Initializing Groq LLM with model: {config.groq_model}")
    
    llm = ChatGroq(
        api_key=config.groq_api_key,
        model=config.groq_model,
        temperature=0,  # Default temperature, can be overridden per agent
    )
    
    logger.info("LLM initialized successfully")
    return llm


def run_analysis(
    user_query: str,
    config: Any | None = None,
    llm: ChatGroq | None = None,
) -> WorkflowState:
    """Run competitor analysis workflow.
    
    This is the main function for executing competitor analysis. It:
    1. Loads configuration (if not provided)
    2. Initializes LLM (if not provided)
    3. Creates workflow
    4. Executes workflow with user query
    5. Returns final workflow state with results
    
    Args:
        user_query: User's competitor analysis request/question
        config: Optional Config instance. If not provided, loads from environment
        llm: Optional ChatGroq instance. If not provided, creates from config
    
    Returns:
        Final WorkflowState containing:
        - report: Final formatted competitor analysis report
        - export_paths: Dictionary of export file paths (PDF, images)
        - insights: Business insights and SWOT analysis
        - collected_data: Collected competitor data
        - plan: Execution plan
        - validation_errors: Any validation errors encountered
    
    Raises:
        ValueError: If user_query is empty
        RuntimeError: If workflow execution fails
    
    Example:
        ```python
        from src.main import run_analysis
        
        result = run_analysis("Analyze competitors in the SaaS market")
        
        if result.get("report"):
            print(result["report"])
        else:
            print(f"Analysis failed: {result.get('validation_errors', [])}")
        ```
    """
    if not user_query or not user_query.strip():
        raise ValueError("User query cannot be empty")
    
    # Load configuration if not provided
    if config is None:
        try:
            config = get_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise RuntimeError(f"Configuration loading failed: {e}") from e
    
    # Initialize LLM if not provided
    if llm is None:
        try:
            llm = initialize_llm(config)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}") from e
    
    # Prepare workflow configuration
    workflow_config = {
        "max_retries": config.max_retries,
        "temperature": 0,  # Default for planner/supervisor
        "planner_temperature": 0,
        "supervisor_temperature": 0,
        "collector_temperature": 0,
        "insight_temperature": 0.7,
        "report_temperature": 0.7,
    }
    
    # Create workflow
    try:
        logger.info("Creating workflow...")
        workflow = create_workflow(llm=llm, config=workflow_config)
        logger.info("Workflow created successfully")
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}", exc_info=True)
        raise RuntimeError(f"Workflow creation failed: {e}") from e
    
    # Create initial state
    try:
        initial_state = create_initial_state(user_query)
        logger.info(f"Starting analysis for query: {user_query[:100]}...")
    except Exception as e:
        logger.error(f"Failed to create initial state: {e}")
        raise RuntimeError(f"State creation failed: {e}") from e
    
    # Execute workflow
    try:
        logger.info("Executing workflow...")
        final_state = workflow.invoke(initial_state)
        logger.info("Workflow execution completed")
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        raise RuntimeError(f"Workflow execution failed: {e}") from e
    
    # Log results
    if final_state.get("report"):
        logger.info(
            f"Analysis completed successfully. Report length: "
            f"{len(final_state['report'])} characters"
        )
    else:
        logger.warning(
            f"Analysis completed but no report generated. "
            f"Validation errors: {len(final_state.get('validation_errors', []))}"
        )
    
    return final_state


def main() -> int:
    """Main entry point for command-line usage.
    
    Parses command-line arguments and runs the competitor analysis workflow.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    
    Example:
        ```bash
        python -m src.main "Analyze competitors in the SaaS market"
        ```
    """
    parser = argparse.ArgumentParser(
        description="Competitor Analysis Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main "Analyze competitors in the SaaS market"
  python -m src.main "Compare pricing strategies of top 5 competitors"
        """,
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="Competitor analysis query or question",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        # Run analysis
        result = run_analysis(args.query)
        
        # Print results
        if result.get("report"):
            print("\n" + "=" * 80)
            print("COMPETITOR ANALYSIS REPORT")
            print("=" * 80 + "\n")
            print(result["report"])
            print("\n" + "=" * 80)
            
            # Print export paths if available
            export_paths = result.get("export_paths")
            if export_paths:
                print("\nExport Files Generated:")
                for export_type, file_path in export_paths.items():
                    print(f"  - {export_type}: {file_path}")
                print()
            
            return 0
        else:
            print("\nAnalysis completed but no report was generated.")
            if result.get("validation_errors"):
                print("\nValidation Errors:")
                for error in result["validation_errors"]:
                    print(f"  - {error}")
            return 1
            
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        print("\nAnalysis interrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
