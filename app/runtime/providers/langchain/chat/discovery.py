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
    from app.runtime.providers.langchain.chat.core import BaseLangChainChatAdapter

logger = logging.getLogger(__name__)


def discover_chat_adapters() -> list[type["BaseLangChainChatAdapter"]]:
    """
    Escanea app/runtime/providers/langchain/chat/adapters/ y retorna
    todas las clases de adapter encontradas.
    """
    from app.runtime.providers.langchain.chat.core import BaseLangChainChatAdapter

    # Apunta a la subcarpeta adapters/
    adapters_dir = Path(__file__).parent / "adapters"
    package_name = "app.runtime.providers.langchain.chat.adapters"

    found: list[type[BaseLangChainChatAdapter]] = []

    for module_info in pkgutil.iter_modules([str(adapters_dir)]):
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
                logger.debug(
                    "Adapter descubierto: %s (%s)", obj.__name__, obj.provider_code
                )

    return found