# app/runtime/providers/langchain/chat/core/meta.py
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderAdapterMeta:
    """
    Metadatos que cada adapter declara sobre sí mismo.
    El sistema los usa para auto-registro y para saber
    qué credencial necesita del entorno.
    """
    provider_code: str
    env_key_name: str | None = None
    required_packages: list[str] = field(default_factory=list)
    is_local: bool = False