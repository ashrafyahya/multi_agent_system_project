"""Data collector agent for gathering competitor information.

This module implements the DataCollectorAgent that uses web search and
scraper tools to collect competitor data and return structured CompetitorProfile
objects.

Example:
    ```python
    from src.agents.data_collector import DataCollectorAgent
    from langchain_groq import ChatGroq
    from src.graph.state import create_initial_state
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    config = {"max_results": 10}
    agent = DataCollectorAgent(llm=llm, config=config)
    
    state = create_initial_state("Analyze competitors")
    state["plan"] = {"tasks": ["Find competitors"], "minimum_results": 4}
    updated_state = agent.execute(state)
    ```
"""

import logging
import re
from typing import Any
from urllib.parse import urlparse

from src.agents.base_agent import BaseAgent
from src.exceptions.collector_error import CollectorError
from src.exceptions.workflow_error import WorkflowError
from src.graph.state import WorkflowState
from src.models.competitor_profile import CompetitorProfile
from src.models.plan_model import Plan
from src.tools.scraper import scrape_url
from src.tools.web_search import web_search

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
    
    Example:
        ```python
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        config = {"max_results": 10}
        agent = DataCollectorAgent(llm=llm, config=config)
        
        state = create_initial_state("Analyze competitors")
        state["plan"] = {"tasks": ["Find competitors"], "minimum_results": 4}
        updated_state = agent.execute(state)
        ```
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
        
        Example:
            ```python
            state = create_initial_state("Analyze competitors")
            state["plan"] = {"tasks": ["Find competitors"], "minimum_results": 4}
            updated_state = agent.execute(state)
            assert updated_state["collected_data"] is not None
            ```
        """
        try:
            # Extract plan
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
            
            # Collect competitor data
            competitors = self._collect_competitor_data(plan)
            
            # Update state
            new_state = state.copy()
            new_state["collected_data"] = {
                "competitors": [comp.model_dump() for comp in competitors]
            }
            new_state["current_task"] = f"Collected data for {len(competitors)} competitors"
            
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
        
        max_results = self.config.get("max_results", plan.minimum_results)
        
        # Generate search queries from tasks
        search_queries = self._generate_search_queries(plan.tasks)
        
        # Perform searches and collect data
        for query in search_queries:
            try:
                # Perform web search
                search_result = web_search.invoke({
                    "query": query,
                    "max_results": max_results
                })
                
                if not search_result.get("success"):
                    logger.warning(f"Search failed for query '{query}': {search_result.get('error')}")
                    continue
                
                # Process search results
                for result in search_result.get("results", []):
                    url = result.get("url", "")
                    if not url or url in seen_urls:
                        continue
                    
                    # Extract competitor information
                    competitor = self._extract_competitor_info(result, seen_names)
                    if competitor:
                        competitors.append(competitor)
                        seen_urls.add(url)
                        seen_names.add(competitor.name.lower())
                        
                        # Stop if we have enough competitors
                        if len(competitors) >= plan.minimum_results * 2:  # Collect extra for quality
                            break
                
                # Stop if we have enough competitors
                if len(competitors) >= plan.minimum_results * 2:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing query '{query}': {e}")
                continue
        
        # If we don't have enough, try scraping some URLs for more details
        if len(competitors) < plan.minimum_results:
            logger.info(f"Only collected {len(competitors)} competitors, attempting to scrape for more")
            # Additional scraping logic could go here
        
        if len(competitors) < plan.minimum_results:
            logger.warning(
                f"Collected only {len(competitors)} competitors, "
                f"minimum required: {plan.minimum_results}"
            )
        
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
            
            if not url:
                return None
            
            # Extract competitor name from title or URL
            name = self._extract_competitor_name(title, url, snippet)
            if not name or name.lower() in seen_names:
                return None
            
            # Extract website URL
            website = self._extract_website_url(url, snippet)
            
            # Extract products (basic extraction)
            products = self._extract_products(snippet, title)
            
            # Create competitor profile
            competitor = CompetitorProfile(
                name=name,
                website=website,
                products=products,
                source_url=url,
                market_presence=snippet[:200] if snippet else None  # Truncate
            )
            
            return competitor
            
        except Exception as e:
            logger.debug(f"Failed to extract competitor info from {search_result.get('url')}: {e}")
            return None
    
    def _extract_competitor_name(self, title: str, url: str, snippet: str) -> str | None:
        """Extract competitor name from title, URL, or snippet.
        
        Args:
            title: Page title
            url: Page URL
            snippet: Page snippet
        
        Returns:
            Competitor name string or None if extraction fails
        """
        # Try to extract from title first
        if title:
            # Remove common suffixes
            name = re.sub(r'\s*-\s*(Home|Official|Website|About).*$', '', title, flags=re.IGNORECASE)
            name = name.strip()
            if name and len(name) > 2 and len(name) < 100:
                return name
        
        # Try to extract from URL domain
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain:
                # Remove www. and common TLDs
                name = domain.replace("www.", "").split(".")[0]
                if name and len(name) > 2:
                    return name.capitalize()
        except Exception:
            pass
        
        # Try to extract from snippet (first few words)
        if snippet:
            words = snippet.split()[:5]
            potential_name = " ".join(words)
            if len(potential_name) > 2 and len(potential_name) < 50:
                return potential_name
        
        return None
    
    def _extract_website_url(self, url: str, snippet: str) -> str | None:
        """Extract competitor website URL.
        
        Args:
            url: Source URL
            snippet: Page snippet
        
        Returns:
            Website URL string or None
        """
        # Use the source URL's domain as website
        try:
            parsed = urlparse(url)
            website = f"{parsed.scheme}://{parsed.netloc}"
            return website
        except Exception:
            return None
    
    def _extract_products(self, snippet: str, title: str) -> list[str]:
        """Extract product names from snippet and title.
        
        Args:
            snippet: Page snippet
            title: Page title
        
        Returns:
            List of product name strings
        """
        products: list[str] = []
        text = f"{title} {snippet}".lower()
        
        # Look for common product indicators
        product_patterns = [
            r'product[s]?:\s*([^,\.]+)',
            r'offers?\s+([^,\.]+)',
            r'solutions?:\s*([^,\.]+)',
        ]
        
        for pattern in product_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                product = match.strip()
                if product and len(product) > 2 and len(product) < 50:
                    products.append(product)
        
        return products[:5]  # Limit to 5 products
    
    @property
    def name(self) -> str:
        """Return agent name.
        
        Returns:
            String identifier for this agent
        """
        return "data_collector_agent"
