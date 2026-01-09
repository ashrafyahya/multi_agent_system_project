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
from src.models.source_quality import SourceQuality
from src.tools.scraper import scrape_url, scrape_url_async
from src.tools.web_search import web_search, web_search_async
from src.utils.data_verification_service import DataVerificationService
from src.utils.input_validator import validate_and_sanitize_url, validate_url
from src.utils.source_quality_analyzer import SourceQualityAnalyzer

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
    
    def _get_global_config(self):
        """Get the global configuration object."""
        from src.config import get_config
        return get_config()
    
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
            
            # Add timestamp and source statistics
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            
            # Calculate source statistics
            source_stats = self._calculate_source_statistics(competitors)
            
            new_state = update_state(
                state,
                collected_data={
                    "competitors": [comp.model_dump() for comp in competitors]
                },
                data_collection_timestamp=timestamp,
                source_statistics=source_stats,
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
                    
                    skip_low_quality = self._should_skip_low_quality(competitors)
                    competitor = self._extract_competitor_info(result, seen_names, skip_low_quality)
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
        
        # Ensure we have minimum primary sources
        additional_competitors = self._ensure_primary_sources(plan, competitors)
        competitors.extend(additional_competitors)
        
        # Find primary sources for each competitor
        competitors = self._find_primary_sources_for_competitors(competitors)
        
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
                
                competitor = self._extract_competitor_info(result, seen_names, skip_low_quality=False)
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
        
        # Find primary sources for competitors
        competitors = self._find_primary_sources_for_competitors(competitors)
        
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
        
        # Add primary source queries if enabled
        from src.config import get_config
        global_config = get_config()
        if global_config.primary_source_search_enabled:
            primary_queries = self._generate_primary_source_queries(tasks)
            queries.extend(primary_queries)
        
        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower and q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q.strip())
        
        return unique_queries[:5]  # Limit to 5 queries
    
    def _generate_primary_source_queries(self, tasks: list[str]) -> list[str]:
        """Generate primary source-specific search queries.
        
        Args:
            tasks: List of tasks from plan
        
        Returns:
            List of primary source search queries
        """
        primary_queries: list[str] = []
        
        for task in tasks:
            # Extract potential company/product names from task
            # Simple heuristic: look for capitalized words or known patterns
            words = task.split()
            potential_names = []
            
            for word in words:
                # Skip common words
                if word.lower() in ['find', 'analyze', 'research', 'identify', 'competitors', 'competition', 'market', 'pricing', 'products', 'features']:
                    continue
                # Keep capitalized words or words that look like product names
                if word[0].isupper() or len(word) > 3:
                    potential_names.append(word)
            
            # If we found potential names, create primary source queries
            if potential_names:
                base_name = " ".join(potential_names[:2])  # Use first 1-2 names
                
                # SEC filings
                primary_queries.append(f'site:sec.gov {base_name} financial report')
                primary_queries.append(f'site:sec.gov {base_name} 10-K')
                primary_queries.append(f'site:sec.gov {base_name} annual report')
                
                # Company investor relations
                primary_queries.append(f'{base_name} site:investor.company.com')
                primary_queries.append(f'{base_name} investor relations')
                primary_queries.append(f'{base_name} official pricing page')
                primary_queries.append(f'{base_name} press release revenue')
                
                # Government/business registries
                primary_queries.append(f'{base_name} site:bbb.org')
                primary_queries.append(f'{base_name} site:crunchbase.com')
        
        return primary_queries
    
    def _prioritize_results_by_quality(self, search_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prioritize search results by source quality.
        
        Args:
            search_results: List of search result dictionaries
        
        Returns:
            Sorted list with higher quality sources first
        """
        if not search_results:
            return search_results
        
        # Analyze quality for each result
        results_with_quality = []
        analyzer = SourceQualityAnalyzer()
        
        for result in search_results:
            url = result.get("url", "")
            quality = None
            if url:
                try:
                    quality = analyzer.analyze(url)
                except Exception as e:
                    logger.debug(f"Failed to analyze quality for {url}: {e}")
            
            results_with_quality.append({
                "result": result,
                "quality": quality,
                "quality_value": quality.value if quality else "unknown"
            })
        
        # Sort by quality priority (PRIMARY first, then SECONDARY_HIGH, etc.)
        quality_order = {
            "primary": 0,
            "secondary_high": 1,
            "secondary_medium": 2,
            "secondary_low": 3,
            "community": 4,
            "unknown": 5
        }
        
        results_with_quality.sort(key=lambda x: quality_order.get(x["quality_value"], 5))
        
        # Return just the results
        return [item["result"] for item in results_with_quality]
    
    def _ensure_primary_sources(self, plan: Plan, existing_competitors: list[CompetitorProfile]) -> list[CompetitorProfile]:
        """Ensure minimum primary sources are found by expanding search if needed.
        
        Args:
            plan: Execution plan with tasks
            existing_competitors: Already collected competitors
        
        Returns:
            Additional competitors from expanded search
        """
        global_config = self._get_global_config()
        if not global_config.auto_expand_primary_sources:
            return []
        
        # Count current primary sources
        primary_count = sum(
            1 for comp in existing_competitors
            if comp.source_quality and comp.source_quality.value == "primary"
        )
        
        if primary_count >= global_config.min_primary_sources:
            logger.debug(f"Sufficient primary sources found ({primary_count})")
            return []
        
        logger.info(f"Only {primary_count} primary sources found, expanding search...")
        
        additional_competitors = []
        max_attempts = 3
        
        for attempt in range(max_attempts):
            if primary_count >= global_config.min_primary_sources:
                break
                
            # Generate additional primary-focused queries
            primary_queries = []
            for task in plan.tasks:
                primary_queries.extend(self._generate_primary_source_queries([task]))
            
            # Remove duplicates and limit
            seen_queries = set()
            unique_primary_queries = []
            for q in primary_queries:
                q_lower = q.lower().strip()
                if q_lower not in seen_queries:
                    seen_queries.add(q_lower)
                    unique_primary_queries.append(q)
            
            # Try a few more queries
            queries_to_try = unique_primary_queries[attempt*2:(attempt+1)*2]
            
            if not queries_to_try:
                break
                
            logger.debug(f"Attempt {attempt+1}: Trying {len(queries_to_try)} additional primary queries")
            
            for query in queries_to_try:
                try:
                    search_result = web_search.invoke({
                        "query": query,
                        "max_results": plan.minimum_results
                    })
                    
                    if not search_result.get("success"):
                        continue
                    
                    results_list = search_result.get("results", [])
                    
                    for result in results_list:
                        url = result.get("url", "")
                        if not url:
                            continue
                            
                        # Check if we already have this competitor
                        title = result.get("title", "")
                        snippet = result.get("snippet", "")
                        
                        name = extract_competitor_name(title, url, snippet)
                        if not name or any(comp.name.lower() == name.lower() for comp in existing_competitors + additional_competitors):
                            continue
                        
                        # Extract competitor info
                        competitor = self._extract_competitor_info(result, set(), skip_low_quality=False)
                        if competitor and competitor.source_quality and competitor.source_quality.value == "primary":
                            additional_competitors.append(competitor)
                            primary_count += 1
                            logger.info(f"✓ Found additional primary source competitor: {competitor.name}")
                            
                            if primary_count >= global_config.min_primary_sources:
                                break
                    
                    if primary_count >= global_config.min_primary_sources:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error in primary source expansion query '{query}': {e}")
                    continue
        
        logger.info(f"Primary source expansion complete. Found {len(additional_competitors)} additional competitors")
        return additional_competitors
    
    def _find_primary_sources_for_competitors(self, competitors: list[CompetitorProfile]) -> list[CompetitorProfile]:
        """Find and update primary sources for competitors by searching official websites.
        
        Performs targeted searches for each competitor's official website to ensure
        primary sources are available. Updates existing competitors with primary
        source quality when found.
        
        Args:
            competitors: List of collected competitors
        
        Returns:
            Updated list of competitors with primary sources where found
        """
        updated_competitors = []
        
        for comp in competitors:
            if comp.source_quality and comp.source_quality.value == "primary":
                updated_competitors.append(comp)
                continue
            
            # Search for official website
            query = f"official website {comp.name}"
            try:
                search_result = web_search.invoke({"query": query, "max_results": 3})
                
                if search_result.get("success"):
                    primary_found = False
                    for result in search_result.get("results", []):
                        url = result.get("url", "")
                        title = result.get("title", "")
                        
                        if url and self._is_likely_primary_source(url, title, comp.name):
                            # Update competitor with primary source
                            updated_comp = comp.model_copy()
                            updated_comp.source_quality = SourceQuality.PRIMARY
                            updated_comp.source_url = url
                            updated_comp.website = url
                            updated_comp.sources = [url]
                            updated_competitors.append(updated_comp)
                            logger.info(f"Updated {comp.name} with primary source: {url}")
                            primary_found = True
                            break
                    
                    if not primary_found:
                        updated_competitors.append(comp)
                else:
                    updated_competitors.append(comp)
                    
            except Exception as e:
                logger.warning(f"Error finding primary source for {comp.name}: {e}")
                updated_competitors.append(comp)
        
        return updated_competitors
    
    def _is_likely_primary_source(self, url: str, title: str, competitor_name: str) -> bool:
        """Check if URL is likely a primary source for the competitor.
        
        Args:
            url: The URL to check
            title: The page title
            competitor_name: Name of the competitor
        
        Returns:
            True if likely primary source, False otherwise
        """
        try:
            analyzer = SourceQualityAnalyzer()
            quality = analyzer.analyze(url)
            return quality.value == "primary"
        except Exception:
            return False
    
    def _calculate_source_statistics(self, competitors: list[CompetitorProfile]) -> dict[str, Any]:
        """Calculate detailed source statistics.
        
        Args:
            competitors: List of collected competitors
        
        Returns:
            Dictionary with detailed source statistics
        """
        from collections import Counter
        
        quality_counts = Counter()
        verification_counts = Counter()
        
        for comp in competitors:
            # Count source qualities
            if comp.source_quality:
                quality_counts[comp.source_quality.value] += 1
            
            # Count verification statuses
            if comp.data_verification_status:
                for metric, status in comp.data_verification_status.items():
                    verification_counts[f"{metric}_{status}"] += 1
        
        return {
            "quality_distribution": dict(quality_counts),
            "verification_distribution": dict(verification_counts),
            "total_competitors": len(competitors),
            "total_sources": len([c for c in competitors if c.source_url])
        }
    
    def _should_skip_low_quality(self, competitors: list[CompetitorProfile]) -> bool:
        """Check if we should skip low-quality sources based on existing high-quality sources.
        
        Args:
            competitors: Already collected competitors
        
        Returns:
            True if we should skip low-quality sources
        """
        if not competitors:
            return False
        
        # Count high-quality sources
        high_quality_count = sum(
            1 for comp in competitors
            if comp.source_quality and comp.source_quality.value in ["primary", "secondary_high", "secondary_medium"]
        )
        
        return high_quality_count >= self._get_global_config().max_low_quality_before_skip
    
    def _extract_competitor_info(
        self,
        search_result: dict[str, Any],
        seen_names: set[str],
        skip_low_quality: bool = False
    ) -> CompetitorProfile | None:
        """Extract competitor information from search result.
        
        Args:
            search_result: Search result dictionary with url, title, snippet
            seen_names: Set of competitor names already collected (for deduplication)
            skip_low_quality: Whether to skip low-quality sources (secondary_low, community)
        
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
            
            # Analyze source quality
            source_quality = None
            try:
                analyzer = SourceQualityAnalyzer()
                source_quality = analyzer.analyze(str(source_url))
                if source_quality.value in ["secondary_low", "community"]:
                    logger.debug(
                        f"Low quality source detected for competitor '{name}': "
                        f"{source_quality.value} - {source_url}"
                    )
                    if skip_low_quality:
                        logger.debug(f"Skipping low-quality source as requested: {source_url}")
                        return None
            except Exception as e:
                logger.warning(
                    f"Failed to analyze source quality for {source_url}: {e}. "
                    "Continuing without source quality classification."
                )
            
            # Verify quantitative data
            data_verification_status = None
            try:
                verification_service = DataVerificationService()
                source_content = f"{title} {snippet}" if title and snippet else (snippet or title or "")
                
                # Check if we should scrape for verification
                should_scrape = (
                    self._get_global_config().scrape_for_verification and
                    source_quality and
                    source_content and
                    any(metrics.values())  # Has quantitative data
                )
                
                if should_scrape:
                    # Check quality threshold
                    quality_levels = ["primary", "secondary_high", "secondary_medium", "secondary_low", "community"]
                    global_config = self._get_global_config()
                    min_quality_index = quality_levels.index(global_config.scrape_quality_threshold)
                    current_quality_index = quality_levels.index(source_quality.value)
                    
                    if current_quality_index <= min_quality_index:
                        try:
                            logger.debug(f"Scraping full page for verification: {source_url}")
                            scrape_result = scrape_url.invoke({"url": source_url})
                            if scrape_result.get("success"):
                                scraped_content = scrape_result.get("content", "")
                                if scraped_content:
                                    source_content = f"{source_content} {scraped_content}"
                                    logger.debug(f"Added scraped content ({len(scraped_content)} chars) for verification")
                        except Exception as e:
                            logger.warning(f"Failed to scrape {source_url} for verification: {e}")
                
                if source_content:
                    competitor_data = {
                        "market_share": metrics.get("market_share"),
                        "revenue": metrics.get("revenue"),
                        "user_count": metrics.get("user_count"),
                        "founded_year": metrics.get("founded_year"),
                    }
                    data_verification_status = verification_service.verify_quantitative_data(
                        competitor_data,
                        source_content
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to verify data for competitor '{name}': {e}. "
                    "Continuing without data verification."
                )
            
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
                key_features=metrics.get("key_features", []),
                source_quality=source_quality,
                data_verification_status=data_verification_status,
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
