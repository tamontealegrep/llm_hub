from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    xai_api_key: str | None = None

    default_openai_model: str = "gpt-4o-mini"
    default_anthropic_model: str = "claude-3-5-sonnet-latest"
    default_google_model: str = "gemini-1.5-flash"
    default_xai_model: str = "grok-beta"

    default_temperature: float = 0.2
    default_top_p: float = 1.0
    default_max_output_tokens: int = 2048
    default_timeout_seconds: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()