"""Data collector agent for gathering competitor information.

This module implements the DataCollectorAgent that uses web search and
scraper tools to collect competitor data and return structured CompetitorProfile
objects.
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
        
        search_queries = self._generate_search_queries(plan.tasks)
        
        for query in search_queries:
            try:
                search_result = web_search.invoke({
                    "query": query,
                    "max_results": max_results
                })
                
                if not search_result.get("success"):
                    logger.warning(f"Search failed for query '{query}': {search_result.get('error')}")
                    continue
                
                for result in search_result.get("results", []):
                    url = result.get("url", "")
                    if not url or url in seen_urls:
                        continue
                    
                    competitor = self._extract_competitor_info(result, seen_names)
                    if competitor:
                        competitors.append(competitor)
                        seen_urls.add(url)
                        seen_names.add(competitor.name.lower())
                        
                        if len(competitors) >= plan.minimum_results * 2:
                            break
                
                if len(competitors) >= plan.minimum_results * 2:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing query '{query}': {e}")
                continue
        
        if len(competitors) < plan.minimum_results:
            logger.info(f"Only collected {len(competitors)} competitors, attempting to scrape for more")
        
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
            
            # Extract quantitative metrics from snippet
            metrics = self._extract_quantitative_metrics(snippet, title)
            
            # Create competitor profile
            competitor = CompetitorProfile(
                name=name,
                website=website,
                products=products,
                source_url=url,
                market_presence=snippet[:200] if snippet else None,  # Truncate
                market_share=metrics.get("market_share"),
                revenue=metrics.get("revenue"),
                user_count=metrics.get("user_count"),
                founded_year=metrics.get("founded_year"),
                headquarters=metrics.get("headquarters"),
                key_features=metrics.get("key_features", [])
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
    
    def _extract_quantitative_metrics(
        self,
        snippet: str,
        title: str
    ) -> dict[str, Any]:
        """Extract quantitative metrics from snippet and title.
        
        Attempts to extract structured quantitative data from text using
        pattern matching. Extracts market share, revenue, user count,
        founded year, headquarters, and key features.
        
        Args:
            snippet: Page snippet text
            title: Page title
        
        Returns:
            Dictionary containing extracted metrics:
            - market_share: float | None
            - revenue: float | str | None
            - user_count: int | str | None
            - founded_year: int | None
            - headquarters: str | None
            - key_features: list[str]
        """
        metrics: dict[str, Any] = {
            "market_share": None,
            "revenue": None,
            "user_count": None,
            "founded_year": None,
            "headquarters": None,
            "key_features": []
        }
        
        text = f"{title} {snippet}".lower()
        full_text = f"{title} {snippet}"
        
        # Extract market share (e.g., "35% market share", "20% of the market")
        market_share_patterns = [
            r'(\d+\.?\d*)\s*%\s*market\s*share',
            r'(\d+\.?\d*)\s*%\s*of\s*the\s*market',
            r'market\s*share\s*of\s*(\d+\.?\d*)\s*%',
        ]
        for pattern in market_share_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    share = float(match.group(1))
                    if 0 <= share <= 100:
                        metrics["market_share"] = share
                        break
                except (ValueError, TypeError):
                    continue
        
        # Extract revenue (e.g., "$2B", "$1.5 billion", "$500M")
        revenue_patterns = [
            r'\$(\d+\.?\d*)\s*([BMK]|billion|million|thousand)',
            r'revenue\s*(?:of|is|:)?\s*\$?(\d+\.?\d*)\s*([BMK]|billion|million)',
        ]
        for pattern in revenue_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2).upper()
                    if unit in ['B', 'BILLION']:
                        metrics["revenue"] = value * 1e9
                    elif unit in ['M', 'MILLION']:
                        metrics["revenue"] = value * 1e6
                    elif unit in ['K', 'THOUSAND']:
                        metrics["revenue"] = value * 1e3
                    else:
                        metrics["revenue"] = value
                    break
                except (ValueError, TypeError, IndexError):
                    continue
        
        # Extract user count (e.g., "1M users", "500K customers")
        user_count_patterns = [
            r'(\d+\.?\d*)\s*([BMK]|million|thousand)\s*(?:users?|customers?|subscribers?)',
            r'(?:users?|customers?|subscribers?):?\s*(\d+\.?\d*)\s*([BMK]|million|thousand)',
        ]
        for pattern in user_count_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    unit = match.group(2).upper()
                    if unit in ['M', 'MILLION']:
                        metrics["user_count"] = int(value * 1e6)
                    elif unit in ['K', 'THOUSAND']:
                        metrics["user_count"] = int(value * 1e3)
                    elif unit in ['B', 'BILLION']:
                        metrics["user_count"] = int(value * 1e9)
                    else:
                        metrics["user_count"] = int(value)
                    break
                except (ValueError, TypeError, IndexError):
                    continue
        
        # Extract founded year (e.g., "founded in 2010", "established 2005")
        founded_patterns = [
            r'founded\s+(?:in\s+)?(\d{4})',
            r'established\s+(?:in\s+)?(\d{4})',
            r'(\d{4})\s*\(founded\)',
        ]
        for pattern in founded_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    year = int(match.group(1))
                    if 1800 <= year <= 2100:
                        metrics["founded_year"] = year
                        break
                except (ValueError, TypeError):
                    continue
        
        # Extract headquarters (e.g., "headquartered in San Francisco", "based in New York")
        hq_patterns = [
            r'headquartered\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'based\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'hq:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        for pattern in hq_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                hq = match.group(1).strip()
                if len(hq) > 2 and len(hq) < 100:
                    metrics["headquarters"] = hq
                    break
        
        # Extract key features (look for feature lists)
        feature_patterns = [
            r'features?:\s*([^\.]+)',
            r'key\s+features?:\s*([^\.]+)',
            r'offers?\s+([^\.]+)',
        ]
        for pattern in feature_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Split by common separators
                features = re.split(r'[,;]|\sand\s', match)
                for feature in features:
                    feature = feature.strip()
                    if feature and len(feature) > 3 and len(feature) < 100:
                        metrics["key_features"].append(feature)
                if metrics["key_features"]:
                    break
            if metrics["key_features"]:
                break
        
        # Limit features
        metrics["key_features"] = metrics["key_features"][:10]
        
        return metrics
    
    @property
    def name(self) -> str:
        """Return agent name.
        
        Returns:
            String identifier for this agent
        """
        return "data_collector_agent"
