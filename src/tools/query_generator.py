"""Query generator tool for optimizing search queries.

This tool generates search queries from tasks and returns multiple
query variations optimized for search engines. It helps improve search
coverage by generating different phrasings and perspectives.

Example:
    ```python
    from src.tools.query_generator import generate_queries
    
    result = generate_queries.invoke({
        "task": "Find competitor pricing information",
        "max_variations": 5
    })
    
    if result["success"]:
        for query in result["queries"]:
            print(f"Query: {query['text']} (type: {query['type']})")
    ```
"""

import logging
import re
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _generate_query_variations(task: str, max_variations: int) -> list[dict[str, str]]:
    """Generate multiple query variations from a task.
    
    This function creates different query variations by:
    - Adding context keywords (competitor, analysis, comparison)
    - Creating different phrasings (questions, statements, comparisons)
    - Adding industry-specific terms
    - Creating focused vs. broad queries
    
    Args:
        task: Task description to convert into queries
        max_variations: Maximum number of query variations to generate
        
    Returns:
        List of query dictionaries with 'text' and 'type' fields
    """
    task_lower = task.lower().strip()
    queries: list[dict[str, str]] = []
    
    # Extract key terms from task
    # Remove common stop words
    stop_words = {"find", "get", "collect", "gather", "analyze", "the", "a", "an", "and", "or", "for", "of", "in", "on", "at", "to", "from"}
    words = [w for w in re.findall(r'\b\w+\b', task_lower) if w not in stop_words and len(w) > 2]
    key_terms = " ".join(words[:5])  # Use first 5 meaningful words
    
    if not key_terms:
        # Fallback if no meaningful terms extracted
        key_terms = task_lower
    
    # Variation 1: Direct task as query
    queries.append({
        "text": task.strip(),
        "type": "direct",
    })
    
    # Variation 2: Add "competitor" context if not present
    if "competitor" not in task_lower:
        queries.append({
            "text": f"competitor {task.strip()}",
            "type": "competitor_focused",
        })
    
    # Variation 3: Question format
    if not task.strip().endswith("?"):
        queries.append({
            "text": f"what is {task.strip()}",
            "type": "question",
        })
    
    # Variation 4: Comparison format
    if "compare" not in task_lower and "comparison" not in task_lower:
        queries.append({
            "text": f"{task.strip()} comparison",
            "type": "comparison",
        })
    
    # Variation 5: Analysis format
    if "analysis" not in task_lower and "analyze" not in task_lower:
        queries.append({
            "text": f"{task.strip()} analysis",
            "type": "analysis",
        })
    
    # Variation 6: Best/top format
    if "best" not in task_lower and "top" not in task_lower:
        queries.append({
            "text": f"best {task.strip()}",
            "type": "best_top",
        })
    
    # Variation 7: Key terms only (focused)
    if len(words) > 2:
        queries.append({
            "text": key_terms,
            "type": "focused",
        })
    
    # Variation 8: Add "market" context
    if "market" not in task_lower:
        queries.append({
            "text": f"{task.strip()} market",
            "type": "market_context",
        })
    
    # Variation 9: Add "industry" context
    if "industry" not in task_lower:
        queries.append({
            "text": f"{task.strip()} industry",
            "type": "industry_context",
        })
    
    # Variation 10: Pricing-specific (if task mentions pricing)
    if "price" in task_lower or "cost" in task_lower or "pricing" in task_lower:
        queries.append({
            "text": f"{task.strip()} pricing strategy",
            "type": "pricing_focused",
        })
    
    # Variation 11: Product-specific (if task mentions product)
    if "product" in task_lower:
        queries.append({
            "text": f"{task.strip()} features",
            "type": "product_focused",
        })
    
    # Variation 12: Review/opinion format
    queries.append({
        "text": f"{task.strip()} review",
        "type": "review",
    })
    
    # Deduplicate and limit
    seen = set()
    unique_queries = []
    for query in queries:
        query_text = query["text"].lower().strip()
        if query_text not in seen and len(query_text) > 3:
            seen.add(query_text)
            unique_queries.append(query)
            if len(unique_queries) >= max_variations:
                break
    
    return unique_queries


@tool
def generate_queries(task: str, max_variations: int = 5) -> dict[str, Any]:
    """Generate optimized search queries from a task description.
    
    This tool takes a task description and generates multiple search query
    variations optimized for search engines. It creates different phrasings,
    adds relevant context keywords, and generates focused vs. broad queries
    to improve search coverage.
    
    The tool is useful for converting high-level tasks into specific search
    queries that can be used with the web_search tool to gather comprehensive
    information about competitors.
    
    Args:
        task: Task description to convert into search queries.
            Examples:
            - "Find competitor pricing information"
            - "Analyze competitor products"
            - "Get market share data"
            - "Compare pricing strategies"
        max_variations: Maximum number of query variations to generate.
            Default is 5, recommended range is 3-10. More variations
            provide better coverage but may increase search time.
    
    Returns:
        Dictionary with the following structure:
        {
            "success": bool,      # True if generation succeeded
            "task": str,          # Original task
            "queries": [          # List of generated queries
                {
                    "text": str,  # Query text
                    "type": str   # Query type (e.g., "direct", "competitor_focused")
                },
                ...
            ],
            "count": int,         # Number of queries generated
            "error": str          # Error message (if failed)
        }
    
    Example:
        ```python
        # Generate queries for competitor analysis
        result = generate_queries.invoke({
            "task": "Find competitor pricing",
            "max_variations": 5
        })
        
        if result["success"]:
            print(f"Generated {result['count']} query variations:")
            for query in result["queries"]:
                print(f"  - {query['text']} ({query['type']})")
        ```
    
    Note:
        - Generates multiple query types: direct, question, comparison, etc.
        - Automatically deduplicates queries
        - Optimizes queries for search engines
        - Adds relevant context keywords when appropriate
    """
    # Validate inputs
    if not task or not task.strip():
        return {
            "success": False,
            "error": "Task cannot be empty",
            "task": task,
            "queries": [],
            "count": 0,
        }
    
    if max_variations < 1 or max_variations > 15:
        return {
            "success": False,
            "error": f"max_variations must be between 1 and 15, got {max_variations}",
            "task": task,
            "queries": [],
            "count": 0,
        }
    
    task = task.strip()
    
    try:
        logger.info(f"Generating queries for task: {task}, max_variations: {max_variations}")
        
        # Generate query variations
        queries = _generate_query_variations(task, max_variations)
        
        if not queries:
            return {
                "success": False,
                "error": "Failed to generate any queries from task",
                "task": task,
                "queries": [],
                "count": 0,
            }
        
        logger.info(f"Generated {len(queries)} query variations for task: {task}")
        
        return {
            "success": True,
            "task": task,
            "queries": queries,
            "count": len(queries),
        }
        
    except Exception as e:
        error_msg = f"Unexpected error during query generation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            "success": False,
            "error": error_msg,
            "task": task,
            "queries": [],
            "count": 0,
        }
