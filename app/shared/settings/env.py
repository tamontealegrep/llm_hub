from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    """
    Configuración cargada desde variables de entorno / .env.

    API Keys de providers cloud conocidos.
    Para providers locales (Ollama, HuggingFace pipeline), no se necesita key.
    Para agregar una nueva key: solo añade el campo aquí con `str | None = None`.
    """

    app_env: str = "local"
    app_config_file: str | None = None

    # ── API Keys de providers cloud ──────────────────────────────────────────
    # Para agregar un nuevo provider cloud: añade su key aquí.
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    xai_api_key: str | None = None

    # ── Providers locales / self-hosted ──────────────────────────────────────
    # Ollama no necesita API key por defecto.
    # HuggingFace Inference Endpoint sí necesita key.
    huggingface_api_key: str | None = None  # para HF Inference Endpoints
    # URL base de Ollama (default: http://localhost:11434)
    ollama_base_url: str = "http://localhost:11434"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignora variables no declaradas
        frozen=True,
    )

    def get_provider_key(self, env_key_name: str) -> str | None:
        """
        Obtiene la API key de un provider dado el nombre del campo de EnvSettings.
        Útil para el factory genérico.
        """
        return getattr(self, env_key_name, None)


@lru_cache
def get_env_settings() -> EnvSettings:
    return EnvSettings()