from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    """
    Configuración cargada desde variables de entorno / .env.

    Aquí solo deben vivir:
    - secretos
    - selección de entorno
    - punteros a archivos de configuración
    """

    app_env: str = "local"
    app_config_file: str | None = None

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    xai_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )


@lru_cache
def get_env_settings() -> EnvSettings:
    return EnvSettings()