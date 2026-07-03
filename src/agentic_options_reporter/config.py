from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AOR_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./agentic_options_reporter.db"
    risk_free_rate: float = 0.045
    min_open_interest: int = 50
    max_spread_pct: float = 0.15
    cache_ttl_seconds: int = 300
    llm_model: str = "claude-sonnet-5"
    llm_max_tokens: int = 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
