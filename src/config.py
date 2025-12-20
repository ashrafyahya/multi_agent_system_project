"""Configuration management for the Competitor Analysis system.

This module handles environment variable loading and provides type-safe
configuration access using Pydantic models.

Example:
    ```python
    from src.config import get_config
    
    config = get_config()
    api_key = config.groq_api_key
    max_retries = config.max_retries
    ```
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
        tavily_api_key: Optional API key for Tavily search service
        llm_model: Default LLM model name for all agents (fallback)
        llm_model_planner: Model for Planner agent (optional)
        llm_model_supervisor: Model for Supervisor agent (optional)
        llm_model_insight: Model for Insight agent (optional)
        llm_model_report: Model for Report agent (optional)
        llm_model_collector: Model for Data Collector agent (optional)
        llm_model_export: Model for Export agent (optional)
        agent_log_dir: Directory for storing agent output log files
        agent_log_enabled: Enable/disable agent output logging (default: True)
    
    Example:
        ```python
        from src.config import get_config
        
        config = get_config()
        # API keys are automatically loaded from .env file or environment
        api_key = config.groq_api_key
        tavily_key = config.tavily_api_key
        ```
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
    
    tavily_api_key: Optional[str] = Field(
        default=None,
        description="Optional Tavily API key for web search",
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
        
        Example:
            ```python
            config = get_config()
            planner_model = config.get_model_for_agent("planner")
            insight_model = config.get_model_for_agent("insight")
            ```
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
        
    Example:
        ```python
        from src.config import get_config
        
        config = get_config()
        # Access API keys (automatically loaded from .env or environment)
        groq_key = config.groq_api_key
        tavily_key = config.tavily_api_key  # May be None if not set
        ```
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
