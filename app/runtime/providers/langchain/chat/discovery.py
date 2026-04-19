"""
Auto-descubrimiento de adapters langchain.
Escanea el directorio de adapters y retorna todas las clases
que son subclases de BaseLangChainChatAdapter.
"""
from __future__ import annotations
import importlib
import logging
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.runtime.providers.langchain.chat.base import BaseLangChainChatAdapter

logger = logging.getLogger(__name__)

# Módulos a excluir del scan (no son adapters)
_EXCLUDED_MODULES = {"base", "discovery", "__init__"}


def discover_chat_adapters() -> list[type[BaseLangChainChatAdapter]]:
    """
    Escanea app/runtime/providers/langchain/chat/ y retorna todas las clases
    de adapter encontradas (subclases de BaseLangChainChatAdapter).
    """
    from app.runtime.providers.langchain.chat.base import BaseLangChainChatAdapter

    chat_package_dir = Path(__file__).parent
    package_name = "app.runtime.providers.langchain.chat"

    found: list[type[BaseLangChainChatAdapter]] = []

    for module_info in pkgutil.iter_modules([str(chat_package_dir)]):
        if module_info.name in _EXCLUDED_MODULES:
            continue

        module_full_name = f"{package_name}.{module_info.name}"
        try:
            module = importlib.import_module(module_full_name)
        except ImportError as e:
            logger.warning(
                "No se pudo importar el módulo de adapter '%s': %s",
                module_full_name,
                e,
            )
            continue

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseLangChainChatAdapter)
                and obj is not BaseLangChainChatAdapter
                and hasattr(obj, "adapter_meta")
            ):
                found.append(obj)
                logger.debug("Adapter descubierto: %s (%s)", obj.__name__, obj.provider_code)

    return found