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
    """Application configuration loaded from environment variables.
    
    All configuration values are loaded from environment variables with
    sensible defaults. Type validation is performed automatically by Pydantic.
    
    Attributes:
        groq_api_key: API key for Groq LLM service
        data_dir: Directory for storing temporary data and downloads
        max_retries: Maximum number of retry attempts for failed operations
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        tavily_api_key: Optional API key for Tavily search service
        groq_model: Groq model name to use
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
    
    groq_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model name to use",
    )
    
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


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.
    
    Loads configuration from environment variables on first call and
    returns the same instance on subsequent calls (singleton pattern).
    
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
            f"GROQ_MODEL={_config.groq_model}, "
            f"MAX_RETRIES={_config.max_retries}"
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
