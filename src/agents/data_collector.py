"""Data collector agent for gathering competitor information.

This module implements the DataCollectorAgent that uses web search and
scraper tools to collect competitor data and return structured CompetitorProfile
objects.
"""

import asyncio
import logging
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.utils.data_collection_helpers import (
    extract_competitor_name, extract_products, extract_quantitative_metrics,
    extract_website_url)
from src.config import get_config
from src.exceptions.collector_error import CollectorError
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState
from src.graph.state_utils import update_state
from src.models.competitor_profile import CompetitorProfile
from src.models.plan_model import Plan
from src.tools.scraper import scrape_url, scrape_url_async
from src.tools.web_search import web_search, web_search_async
from src.utils.input_validator import validate_and_sanitize_url, validate_url

logger = logging.getLogger(__name__)


class DataCollectorAgent(BaseAgent):
    """Agent that collects competitor data using web search and scraping tools.
    
    This agent uses web search and web scraping tools to gather competitor
    information based on the execution plan. It:
    1. Extracts tasks from the plan
    2. Generates search queries from tasks
    3. Performs web searches to find competitor information
    4. Scrapes relevant URLs to extract detailed data
    5. Structures data into CompetitorProfile objects
    
    The agent handles tool failures gracefully and returns structured data
    that can be validated by the CollectorValidator.
    
    Attributes:
        llm: Language model instance (injected, may be used for data extraction)
        config: Configuration dictionary (injected)
    """
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute data collection based on plan.
        
        Collects competitor data by:
        1. Extracting plan and tasks from state
        2. Generating search queries from tasks
        3. Performing web searches
        4. Scraping relevant URLs
        5. Extracting and structuring competitor information
        
        Args:
            state: Current workflow state containing plan with tasks
        
        Returns:
            Updated WorkflowState with collected_data field populated
        
        Raises:
            WorkflowError: If plan is missing or data collection fails critically
        """
        try:
            plan_data = state.get("plan")
            if not plan_data:
                raise WorkflowError(
                    "Cannot collect data without a plan",
                    context={"state_keys": list(state.keys())}
                )
            
            try:
                plan = Plan(**plan_data)
            except Exception as e:
                raise WorkflowError(
                    "Invalid plan structure",
                    context={"error": str(e), "plan_data": plan_data}
                ) from e
            
            logger.info(
                f"Starting data collection: {len(plan.tasks)} tasks, "
                f"minimum_results={plan.minimum_results}"
            )
            
            competitors = self._collect_competitor_data(plan)
            
            new_state = update_state(
                state,
                collected_data={
                    "competitors": [comp.model_dump() for comp in competitors]
                },
                current_task=f"Collected data for {len(competitors)} competitors"
            )
            
            logger.info(f"Data collection completed: {len(competitors)} competitors collected")
            
            return new_state
            
        except WorkflowError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in data collector agent: {e}", exc_info=True)
            raise WorkflowError(
                "Data collection failed unexpectedly",
                context={"error": str(e)}
            ) from e
    
    async def execute_async(self, state: WorkflowState) -> WorkflowState:
        """Execute data collection asynchronously based on plan.
        
        This is the async version of execute(). It uses async web search and
        scraping tools to run multiple operations in parallel for improved
        performance.
        
        Args:
            state: Current workflow state containing plan with tasks
        
        Returns:
            Updated WorkflowState with collected_data field populated
        
        Raises:
            WorkflowError: If plan is missing or data collection fails critically
        """
        try:
            plan_data = state.get("plan")
            if not plan_data:
                raise WorkflowError(
                    "Cannot collect data without a plan",
                    context={"state_keys": list(state.keys())}
                )
            
            try:
                plan = Plan(**plan_data)
            except Exception as e:
                raise WorkflowError(
                    "Invalid plan structure",
                    context={"error": str(e), "plan_data": plan_data}
                ) from e
            
            logger.info(
                f"Starting async data collection: {len(plan.tasks)} tasks, "
                f"minimum_results={plan.minimum_results}"
            )
            
            competitors = await self._collect_competitor_data_async(plan)
            
            new_state = update_state(
                state,
                collected_data={
                    "competitors": [comp.model_dump() for comp in competitors]
                },
                current_task=f"Collected data for {len(competitors)} competitors (async)"
            )
            
            logger.info(f"Async data collection completed: {len(competitors)} competitors collected")
            
            return new_state
            
        except WorkflowError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in async data collector agent: {e}", exc_info=True)
            raise WorkflowError(
                "Data collection failed unexpectedly",
                context={"error": str(e)}
            ) from e
    
    def _collect_competitor_data(self, plan: Plan) -> list[CompetitorProfile]:
        """Collect competitor data based on plan.
        
        Args:
            plan: Execution plan with tasks and requirements
        
        Returns:
            List of CompetitorProfile objects
        
        Raises:
            CollectorError: If data collection fails critically
        """
        competitors: list[CompetitorProfile] = []
        seen_urls: set[str] = set()
        seen_names: set[str] = set()
        failed_searches: list[dict[str, Any]] = []
        
        max_results = self.config.get("max_results", plan.minimum_results)
        
        search_queries = self._generate_search_queries(plan.tasks)
        
        for query in search_queries:
            try:
                search_result = web_search.invoke({
                    "query": query,
                    "max_results": max_results
                })
                
                if not search_result.get("success"):
                    error_msg = search_result.get("error", "Unknown error")
                    logger.warning(f"Search failed for query '{query}': {error_msg}")
                    failed_searches.append({
                        "query": query,
                        "error": error_msg,
                        "success": False
                    })
                    continue
                
                results_list = search_result.get("results", [])
                logger.debug(f"Processing {len(results_list)} results from query '{query}'")
                
                for idx, result in enumerate(results_list):
                    url = result.get("url", "")
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    
                    logger.debug(f"Result {idx+1}/{len(results_list)}: url='{url[:80]}', title='{title[:80]}'")
                    
                    # Skip if URL is already seen (but allow results without URLs)
                    if url and url in seen_urls:
                        logger.debug(f"Skipping duplicate URL: {url[:80]}")
                        continue
                    
                    # Validate and sanitize URL if present
                    if url:
                        try:
                            url = validate_and_sanitize_url(url, allow_localhost=False)
                            result["url"] = url  # Update result with sanitized URL
                        except WorkflowError as e:
                            logger.debug(f"Invalid URL, will try without URL: {url}, error: {e}")
                            result["url"] = ""  # Clear invalid URL but continue processing
                            url = ""
                    
                    competitor = self._extract_competitor_info(result, seen_names)
                    if competitor:
                        competitors.append(competitor)
                        if url:
                            seen_urls.add(url)
                        seen_names.add(competitor.name.lower())
                        logger.info(f"✓ Collected competitor {len(competitors)}: {competitor.name}")
                        
                        if len(competitors) >= plan.minimum_results * 2:
                            break
                    else:
                        logger.debug(f"✗ Failed to extract competitor from result {idx+1}")
                
                if len(competitors) >= plan.minimum_results * 2:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing query '{query}': {e}")
                failed_searches.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
                continue
        
        # Validate that we collected at least some competitors
        self._validate_collection_results(competitors, search_queries, failed_searches)
        
        return competitors[:plan.minimum_results * 2]  # Return up to 2x minimum
    
    async def _collect_competitor_data_async(self, plan: Plan) -> list[CompetitorProfile]:
        """Collect competitor data asynchronously based on plan.
        
        This is the async version of _collect_competitor_data. It runs multiple
        web searches and URL scrapes in parallel for improved performance.
        
        Args:
            plan: Execution plan with tasks and requirements
        
        Returns:
            List of CompetitorProfile objects
        
        Raises:
            CollectorError: If data collection fails critically
        """
        competitors: list[CompetitorProfile] = []
        seen_urls: set[str] = set()
        seen_names: set[str] = set()
        failed_searches: list[dict[str, Any]] = []
        
        max_results = self.config.get("max_results", plan.minimum_results)
        search_queries = self._generate_search_queries(plan.tasks)
        
        # Run all searches in parallel
        logger.info(f"Running {len(search_queries)} searches in parallel (async)")
        search_tasks = [
            web_search_async(query, max_results=max_results)
            for query in search_queries
        ]
        
        try:
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error running parallel searches: {e}", exc_info=True)
            search_results = []
        
        # Process search results and collect URLs to scrape
        urls_to_scrape: list[str] = []
        for i, search_result in enumerate(search_results):
            if isinstance(search_result, Exception):
                error_msg = str(search_result)
                logger.warning(f"Search failed for query '{search_queries[i]}': {error_msg}")
                failed_searches.append({
                    "query": search_queries[i],
                    "error": error_msg,
                    "success": False
                })
                continue
            
            if not search_result.get("success"):
                error_msg = search_result.get("error", "Unknown error")
                logger.warning(f"Search failed for query '{search_queries[i]}': {error_msg}")
                failed_searches.append({
                    "query": search_queries[i],
                    "error": error_msg,
                    "success": False
                })
                continue
            
            for result in search_result.get("results", []):
                url = result.get("url", "")
                if not url or url in seen_urls:
                    continue
                
                # Validate and sanitize URL
                try:
                    url = validate_and_sanitize_url(url, allow_localhost=False)
                except WorkflowError as e:
                    logger.debug(f"Invalid URL skipped: {url}, error: {e}")
                    continue
                
                competitor = self._extract_competitor_info(result, seen_names)
                if competitor:
                    competitors.append(competitor)
                    seen_urls.add(url)
                    seen_names.add(competitor.name.lower())
                    urls_to_scrape.append(url)
                    
                    if len(competitors) >= plan.minimum_results * 2:
                        break
            
            if len(competitors) >= plan.minimum_results * 2:
                break
        
        # Scrape URLs in parallel if we need more data
        if len(competitors) < plan.minimum_results and urls_to_scrape:
            logger.info(f"Scraping {len(urls_to_scrape)} URLs in parallel (async)")
            scrape_tasks = [
                scrape_url_async(url, timeout=10)
                for url in urls_to_scrape[:plan.minimum_results * 2]
            ]
            
            try:
                scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
                
                # Process scrape results (extract additional info if needed)
                for scrape_result in scrape_results:
                    if isinstance(scrape_result, Exception):
                        continue
                    if scrape_result.get("success") and scrape_result.get("content"):
                        # Could extract additional competitor info from scraped content
                        # For now, we just use the search results
                        pass
            except Exception as e:
                logger.warning(f"Error during parallel scraping: {e}")
        
        # Validate that we collected at least some competitors
        self._validate_collection_results(competitors, search_queries, failed_searches)
        
        return competitors[:plan.minimum_results * 2]  # Return up to 2x minimum
    
    def _generate_search_queries(self, tasks: list[str]) -> list[str]:
        """Generate search queries from tasks.
        
        Args:
            tasks: List of tasks from plan
        
        Returns:
            List of search query strings
        """
        queries: list[str] = []
        
        for task in tasks:
            # Add task as-is
            queries.append(task)
            
            # Add variations
            if "competitor" not in task.lower():
                queries.append(f"{task} competitors")
            
            if "pricing" in task.lower():
                queries.append(f"{task} comparison")
            elif "product" in task.lower():
                queries.append(f"{task} features")
        
        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower and q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q.strip())
        
        return unique_queries[:5]  # Limit to 5 queries
    
    def _extract_competitor_info(
        self,
        search_result: dict[str, Any],
        seen_names: set[str]
    ) -> CompetitorProfile | None:
        """Extract competitor information from search result.
        
        Args:
            search_result: Search result dictionary with url, title, snippet
            seen_names: Set of competitor names already collected (for deduplication)
        
        Returns:
            CompetitorProfile object or None if extraction fails
        """
        try:
            url = search_result.get("url", "")
            title = search_result.get("title", "")
            snippet = search_result.get("snippet", "")
            
            # Log the raw search result for debugging
            logger.debug(f"Processing search result: url='{url[:100]}', title='{title[:100]}', snippet='{snippet[:100] if snippet else None}'")
            
            # Require at least title or snippet to proceed
            if not title and not snippet:
                logger.debug("Skipping result: no title or snippet")
                return None
            
            # Extract competitor name from title, URL, or snippet
            name = extract_competitor_name(title, url, snippet)
            if not name:
                logger.warning(
                    f"Failed to extract competitor name - "
                    f"title='{title[:80] if title else 'None'}', "
                    f"url='{url[:80] if url else 'None'}', "
                    f"snippet='{snippet[:80] if snippet else 'None'}'"
                )
                return None
            
            # Check for duplicates (case-insensitive)
            name_lower = name.lower()
            if name_lower in seen_names:
                logger.debug(f"Skipping duplicate competitor: {name}")
                return None
            
            # Extract website URL (use URL if available, otherwise None)
            website = extract_website_url(url, snippet) if url else None
            
            # Extract products (basic extraction)
            products = extract_products(snippet, title)
            
            # Extract quantitative metrics from snippet
            metrics = extract_quantitative_metrics(snippet, title)
            
            # Create competitor profile
            # source_url is required, so we need a valid URL
            # If no URL is available, we can't create a valid profile
            if not url:
                # Try to extract a URL from snippet or use a placeholder
                # For now, if there's no URL, we'll skip this result
                # as source_url is required by the model
                logger.debug(f"Skipping result: no valid URL for competitor '{name}'")
                return None
            
            # Validate URL before using it
            is_valid, sanitized_url = validate_url(url, allow_localhost=False)
            if not is_valid or not sanitized_url:
                logger.debug(f"Invalid URL for competitor '{name}': {url}")
                return None
            source_url = sanitized_url
            
            competitor = CompetitorProfile(
                name=name,
                website=website,
                products=products,
                source_url=source_url,
                market_presence=snippet[:200] if snippet else title[:200] if title else None,  # Truncate
                market_share=metrics.get("market_share"),
                revenue=metrics.get("revenue"),
                user_count=metrics.get("user_count"),
                founded_year=metrics.get("founded_year"),
                headquarters=metrics.get("headquarters"),
                key_features=metrics.get("key_features", [])
            )
            
            logger.debug(f"Successfully extracted competitor: {name}")
            return competitor
            
        except Exception as e:
            logger.warning(
                f"Failed to extract competitor info from result - "
                f"url='{search_result.get('url', 'None')[:80]}', "
                f"title='{search_result.get('title', 'None')[:80]}', "
                f"error: {e}"
            )
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _validate_collection_results(
        self,
        competitors: list[CompetitorProfile],
        search_queries: list[str],
        failed_searches: list[dict[str, Any]]
    ) -> None:
        """Validate that competitor data was collected successfully.
        
        Raises WorkflowError if no competitors were collected, with a detailed
        error message based on the type of failure (missing API key, invalid
        API key, expired API key, or general API error).
        
        Args:
            competitors: List of collected CompetitorProfile objects
            search_queries: List of search queries that were attempted
            failed_searches: List of failed search results with error information
            
        Raises:
            WorkflowError: If no competitors were collected, with context about
                the failure type and search queries attempted
        """
        if len(competitors) > 0:
            return  # Validation passed
        
        # Determine error type from failed searches
        error_type = "api_error"
        error_details: list[str] = []
        
        for failed_search in failed_searches:
            error_msg = failed_search.get("error", "").lower()
            
            if "tavily_api_key not configured" in error_msg or "api key" in error_msg and "not" in error_msg:
                error_type = "missing_api_key"
                error_details.append("TAVILY_API_KEY not configured")
            elif "invalid" in error_msg and ("key" in error_msg or "api" in error_msg):
                error_type = "invalid_api_key"
                error_details.append("Invalid API key")
            elif "expired" in error_msg or "expiration" in error_msg:
                error_type = "expired_api_key"
                error_details.append("API key expired")
            elif "401" in error_msg or "403" in error_msg or "unauthorized" in error_msg or "forbidden" in error_msg:
                error_type = "invalid_api_key"
                error_details.append("API authentication failed")
            else:
                error_details.append(failed_search.get("error", "Unknown error"))
        
        # Create error message based on error type
        if error_type == "missing_api_key":
            error_message = (
                "No competitor data collected. TAVILY_API_KEY is not configured. "
                "Please set TAVILY_API_KEY in your .env file or environment variables. "
                "The .env file should be in the project root directory. "
                "Format: TAVILY_API_KEY=your_key_here (no quotes, no spaces around =)"
            )
        elif error_type == "invalid_api_key":
            error_message = (
                "No competitor data collected. TAVILY_API_KEY appears to be invalid. "
                "Please verify your API key is correct in your .env file or environment variables."
            )
        elif error_type == "expired_api_key":
            error_message = (
                "No competitor data collected. TAVILY_API_KEY appears to be expired. "
                "Please obtain a new API key from Tavily and update your configuration."
            )
        else:
            error_message = (
                "No competitor data collected. Tavily API key may be missing, invalid, or expired. "
                "Please check TAVILY_API_KEY configuration in your .env file or environment variables."
            )
        
        logger.error(
            f"Data collection failed: {len(competitors)} competitors collected, "
            f"{len(failed_searches)} searches failed. Error type: {error_type}"
        )
        
        raise WorkflowError(
            error_message,
            context={
                "competitors_count": len(competitors),
                "search_queries": search_queries,
                "failed_searches": failed_searches,
                "error_type": error_type,
                "error_details": error_details
            }
        )
    
    @property
    def name(self) -> str:
        """Return agent name.
        
        Returns:
            String identifier for this agent
        """
        return "data_collector_agent"
