"""Configuration management for the Competitor Analysis system.

This module handles environment variable loading and provides type-safe
configuration access using Pydantic models.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration loaded from environment variables and .env file.
    
    All configuration values are automatically loaded from:
    1. `.env` file in the project root (if present)
    2. Environment variables (as fallback)
    
    Type validation is performed automatically by Pydantic. The configuration
    system uses Pydantic Settings with automatic environment variable loading.
    
    Attributes:
        groq_api_key: API key for Groq LLM service (required)
        data_dir: Directory for storing temporary data and downloads
        max_retries: Maximum number of retry attempts for failed operations
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        tavily_api_key: API key for Tavily search service (required)
        llm_model: Default LLM model name for all agents (fallback)
        llm_model_planner: Model for Planner agent (optional)
        llm_model_supervisor: Model for Supervisor agent (optional)
        llm_model_insight: Model for Insight agent (optional)
        llm_model_report: Model for Report agent (optional)
        llm_model_collector: Model for Data Collector agent (optional)
        llm_model_export: Model for Export agent (optional)
        agent_log_dir: Directory for storing agent output log files
        agent_log_enabled: Enable/disable agent output logging (default: True)
        min_insights: Minimum number of total insights required (default: 8)
        min_positioning_length: Minimum character length for positioning (default: 50)
        min_report_length: Minimum total character length for report (default: 1200)
        min_swot_items_per_category: Minimum items per SWOT category (default: 2)
        min_trends: Minimum number of trends required (default: 2)
        min_opportunities: Minimum number of opportunities required (default: 2)
        min_collector_sources: Minimum number of unique competitor sources (default: 4)
        llm_retry_attempts: Maximum retry attempts for LLM API calls (default: 3)
        llm_retry_backoff_min: Minimum backoff time in seconds (default: 1.0)
        llm_retry_backoff_max: Maximum backoff time in seconds (default: 30.0)
        max_query_length: Maximum character length for user queries (default: 5000)
        min_query_length: Minimum character length for user queries (default: 10)
        llm_cache_enabled: Enable/disable in-memory caching for LLM responses (default: True)
        llm_cache_size: Maximum number of cached LLM responses (default: 128)
        metrics_enabled: Enable/disable metrics tracking (default: True)
        metrics_export_path: Directory path for exporting metrics data (default: ./data/metrics)
        use_async: Enable/disable async execution for I/O operations (default: False)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    groq_api_key: str = Field(
        ...,
        description="Groq API key for LLM service",
        min_length=1,
    )
    
    data_dir: Path = Field(
        default=Path("./data"),
        description="Directory for storing temporary data",
    )
    
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts",
        ge=1,
        le=10,
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    
    tavily_api_key: str = Field(
        ...,
        description="Tavily API key for web search (required)",
        min_length=1,
    )
    
    llm_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Default LLM model name for all agents (fallback)",
    )
    
    llm_model_planner: Optional[str] = Field(
        default=None,
        description="Model for Planner agent",
    )
    
    llm_model_supervisor: Optional[str] = Field(
        default=None,
        description="Model for Supervisor agent",
    )
    
    llm_model_insight: Optional[str] = Field(
        default=None,
        description="Model for Insight agent",
    )
    
    llm_model_report: Optional[str] = Field(
        default=None,
        description="Model for Report agent",
    )
    
    llm_model_collector: Optional[str] = Field(
        default=None,
        description="Model for Data Collector agent",
    )
    
    llm_model_export: Optional[str] = Field(
        default=None,
        description="Model for Export agent",
    )
    
    agent_log_dir: Path = Field(
        default=Path("./data/agent_logs"),
        description="Directory for storing agent output log files",
    )
    
    agent_log_enabled: bool = Field(
        default=True,
        description="Enable/disable agent output logging",
    )
    
    # Validation thresholds
    min_insights: int = Field(
        default=8,
        description="Minimum number of total insights required (SWOT items + trends + opportunities + positioning)",
        ge=1,
        le=100,
    )
    
    min_positioning_length: int = Field(
        default=50,
        description="Minimum character length for positioning statement",
        ge=10,
        le=1000,
    )
    
    min_report_length: int = Field(
        default=1200,
        description="Minimum total character length for report",
        ge=100,
        le=100000,
    )
    
    min_swot_items_per_category: int = Field(
        default=2,
        description="Minimum number of items required per SWOT category (strengths, weaknesses, opportunities, threats)",
        ge=1,
        le=50,
    )
    
    min_trends: int = Field(
        default=2,
        description="Minimum number of trends required",
        ge=1,
        le=50,
    )
    
    min_opportunities: int = Field(
        default=2,
        description="Minimum number of opportunities required (beyond SWOT)",
        ge=1,
        le=50,
    )
    
    min_collector_sources: int = Field(
        default=4,
        description="Minimum number of unique competitor sources required",
        ge=1,
        le=100,
    )
    
    # Rate limiting and retry configuration
    llm_retry_attempts: int = Field(
        default=3,
        description="Maximum number of retry attempts for LLM API calls",
        ge=1,
        le=10,
    )
    
    llm_retry_backoff_min: float = Field(
        default=1.0,
        description="Minimum backoff time in seconds for exponential backoff",
        ge=0.1,
        le=60.0,
    )
    
    llm_retry_backoff_max: float = Field(
        default=30.0,
        description="Maximum backoff time in seconds for exponential backoff",
        ge=1.0,
        le=300.0,
    )
    
    # Input validation configuration
    max_query_length: int = Field(
        default=5000,
        description="Maximum character length for user queries",
        ge=100,
        le=50000,
    )
    
    min_query_length: int = Field(
        default=10,
        description="Minimum character length for user queries",
        ge=1,
        le=1000,
    )
    
    # LLM caching configuration
    llm_cache_enabled: bool = Field(
        default=True,
        description="Enable/disable in-memory caching for LLM responses",
    )
    
    llm_cache_size: int = Field(
        default=128,
        description="Maximum number of cached LLM responses (LRU cache size)",
        ge=1,
        le=10000,
    )
    
    # Metrics configuration
    metrics_enabled: bool = Field(
        default=True,
        description="Enable/disable metrics tracking for execution time, token usage, and API calls",
    )
    
    metrics_export_path: Path = Field(
        default=Path("./data/metrics"),
        description="Directory path for exporting metrics data (JSON files)",
    )
    
    # Async operations configuration
    use_async: bool = Field(
        default=False,
        description="Enable/disable async execution for I/O-bound operations (LLM calls, web requests). When enabled, agents and tools will use async versions for parallel execution.",
    )
    
    # Retry logic configuration
    intelligent_retry_enabled: bool = Field(
        default=True,
        description="Enable/disable intelligent retry using LLM to analyze validation errors and improve queries. When disabled, uses simple rule-based query enhancement.",
    )
    
    # State storage configuration
    state_storage_enabled: bool = Field(
        default=False,
        description="Enable/disable external storage for large state data (reports, collected data). When enabled, large data is stored externally and state contains references.",
    )
    
    state_storage_dir: Path = Field(
        default=Path("./data/state_storage"),
        description="Directory for storing large state data externally",
    )
    
    state_storage_ttl: int = Field(
        default=86400,
        description="Time to live for stored state data in seconds (default: 24 hours). Data older than TTL is automatically cleaned up.",
        ge=60,  # Minimum 1 minute
        le=604800,  # Maximum 7 days
    )
    
    # LangSmith observability configuration
    langsmith_enabled: bool = Field(
        default=False,
        description="Enable/disable LangSmith tracing and observability. When enabled, all LLM calls and agent operations will be traced to LangSmith.",
    )
    
    langsmith_api_key: Optional[str] = Field(
        default=None,
        description="LangSmith API key for authentication. Required if langsmith_enabled is True.",
    )
    
    langsmith_project: Optional[str] = Field(
        default="multi-agent-system",
        description="LangSmith project name for organizing traces. All traces will be grouped under this project name.",
    )
    
    langsmith_endpoint: Optional[str] = Field(
        default=None,
        description="Custom LangSmith endpoint URL. If not provided, uses default LangSmith cloud endpoint.",
    )
    
    @field_validator("state_storage_dir", mode="after")
    @classmethod
    def validate_state_storage_dir(cls, value: Path) -> Path:
        """Ensure state_storage_dir is a Path object."""
        # Directory will be created when storage is actually used
        return value
    
    def get_model_for_agent(self, agent_name: str) -> str:
        """Get the model name for a specific agent.
        
        This method maps agent names to their specific model configuration fields.
        Falls back to llm_model if agent-specific model is not set. 
        Returns appropriate default based on agent type.
        Model names are used directly from configuration without transformation.
        
        Args:
            agent_name: Name of the agent (e.g., "planner", "insight", "report")
        
        Returns:
            Model name string for the agent as configured in environment variables
        """
        # Normalize agent name to lowercase
        agent_name_lower = agent_name.lower()
        
        # Map agent names to their config fields and defaults
        agent_model_map = {
            "planner": (self.llm_model_planner, "llama-3.1-8b-instant"),
            "supervisor": (self.llm_model_supervisor, "llama-3.1-8b-instant"),
            "insight": (self.llm_model_insight, "llama-3.3-70b-versatile"),
            "report": (self.llm_model_report, "llama-3.3-70b-versatile"),
            "collector": (self.llm_model_collector, "llama-3.1-8b-instant"),
            "export": (self.llm_model_export, "llama-3.3-70b-versatile"),
        }
        
        # Get agent-specific model or None
        agent_model, default_model = agent_model_map.get(
            agent_name_lower, (None, "llama-3.1-8b-instant")
        )
        
        # Determine the model to use in priority order:
        # 1. Agent-specific model if set
        # 2. Default llm_model (fallback)
        # 3. Agent-specific default
        if agent_model:
            selected_model = agent_model
        elif self.llm_model:
            selected_model = self.llm_model
        else:
            selected_model = default_model
        
        return selected_model
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Validate log level is one of the allowed values.
        
        Args:
            value: Log level string to validate
            
        Returns:
            Uppercase log level string
            
        Raises:
            ValueError: If log level is not one of the allowed values
        """
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_value = value.upper()
        if upper_value not in allowed_levels:
            raise ValueError(
                f"log_level must be one of {allowed_levels}, got {value}"
            )
        return upper_value
    
    @field_validator("data_dir", mode="before")
    @classmethod
    def validate_data_dir(cls, value: str | Path) -> Path:
        """Convert data_dir to Path object and create if needed.
        
        Args:
            value: Data directory path as string or Path
            
        Returns:
            Path object for data directory
        """
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @field_validator("agent_log_dir", mode="before")
    @classmethod
    def validate_agent_log_dir(cls, value: str | Path) -> Path:
        """Convert agent_log_dir to Path object and create if needed.
        
        Args:
            value: Agent log directory path as string or Path
            
        Returns:
            Path object for agent log directory
        """
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @field_validator("metrics_export_path", mode="before")
    @classmethod
    def validate_metrics_export_path(cls, value: str | Path) -> Path:
        """Convert metrics_export_path to Path object and create if needed.
        
        Args:
            value: Metrics export directory path as string or Path
            
        Returns:
            Path object for metrics export directory
        """
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.
    
    Loads configuration from `.env` file (if present) and environment variables
    on first call and returns the same instance on subsequent calls (singleton pattern).
    
    The configuration is automatically loaded using Pydantic Settings, which:
    - Reads from `.env` file in the project root
    - Falls back to environment variables
    - Validates all values on load
    - Provides type-safe access to configuration
    
    Returns:
        Config instance with loaded configuration values
        
    Raises:
        ValueError: If required configuration values are missing or invalid
    """
    import logging
    logger = logging.getLogger(__name__)
    
    global _config
    if _config is None:
        _config = Config()
        # Log configuration status (without exposing sensitive values)
        logger.debug(
            f"Configuration loaded: "
            f"GROQ_API_KEY={'set' if _config.groq_api_key else 'missing'}, "
            f"TAVILY_API_KEY={'set' if _config.tavily_api_key else 'missing'}, "
            f"LLM_MODEL={_config.llm_model}, "
            f"MAX_RETRIES={_config.max_retries}"
        )
        # Log agent-specific models if set
        agent_models = {
            "planner": _config.llm_model_planner,
            "supervisor": _config.llm_model_supervisor,
            "insight": _config.llm_model_insight,
            "report": _config.llm_model_report,
            "collector": _config.llm_model_collector,
            "export": _config.llm_model_export,
        }
        configured_agents = {
            agent: model
            for agent, model in agent_models.items()
            if model is not None
        }
        if configured_agents:
            logger.debug(
                f"Agent-specific models configured: {configured_agents}"
            )
    return _config


def reload_config() -> Config:
    """Reload configuration from environment variables.
    
    Useful for testing or when configuration changes at runtime.
    
    Returns:
        New Config instance with reloaded configuration values
    """
    global _config
    _config = Config()
    return _config
