# app/tools/loader.py
"""
Loader automático de herramientas personalizadas.

Descubre y registra todas las tools en app/tools/custom/ sin necesidad
de modificar bootstrap.py cada vez que creas una nueva tool.

Protocolo de descubrimiento:
    Para cada archivo .py en app/tools/custom/ (excepto __init__.py):

    OPCIÓN A — Función register() (recomendada):
        Si el módulo expone una función `register(registry)`,
        el loader la llama directamente. Tú controlas qué registrar
        y con qué condiciones (ej: solo si hay API key).

        def register(registry) -> None:
            if not registry.has("web_search"):
                registry.register(WebSearchTool())

    OPCIÓN B — Auto-registro por convención:
        Si el módulo NO tiene función `register()`, el loader busca
        todas las clases que hereden de ToolHandler y las instancia
        automáticamente (requieren __init__ sin argumentos).

        class MySimpleTool(ToolHandler):
            ...   # sin __init__ → el loader la instancia solo

Configuración de exclusiones:
    Si tienes un archivo .py en custom/ que NO quieres que el loader
    toque (por ejemplo, un módulo de utilidades compartidas), agrégalo
    a la lista EXCLUDED_MODULES de este archivo.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from types import ModuleType

from app.tools.interfaces import ToolHandler
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Archivos en custom/ que el loader debe ignorar
EXCLUDED_MODULES: frozenset[str] = frozenset({
    "__init__",
    # Agrega aquí nombres de módulos que NO son tools:
    # "utils",
    # "base_http_tool",
})


def load_custom_tools(registry: ToolRegistry) -> int:
    """
    Descubre y registra todas las tools en app/tools/custom/.

    Returns:
        Número de tools efectivamente registradas en esta llamada.

    Raises:
        No lanza excepciones. Los errores por módulo se loguean como WARNING
        y el proceso continúa con el siguiente archivo.
    """
    custom_package = "app.tools.custom"
    custom_path = Path(__file__).parent / "custom"

    if not custom_path.is_dir():
        logger.warning("load_custom_tools: directorio custom/ no encontrado en %s", custom_path)
        return 0

    registered_count = 0

    for module_info in pkgutil.iter_modules([str(custom_path)]):
        module_name = module_info.name

        if module_name in EXCLUDED_MODULES:
            logger.debug("load_custom_tools: ignorando módulo excluido '%s'", module_name)
            continue

        full_module_name = f"{custom_package}.{module_name}"

        try:
            module = importlib.import_module(full_module_name)
        except ImportError as exc:
            logger.warning(
                "load_custom_tools: no se pudo importar '%s': %s",
                full_module_name,
                exc,
            )
            continue
        except Exception as exc:
            logger.warning(
                "load_custom_tools: error importando '%s': %s",
                full_module_name,
                exc,
            )
            continue

        before = _count_registered(registry)
        n = _register_from_module(module, registry, full_module_name)
        after = _count_registered(registry)
        newly_registered = after - before

        registered_count += newly_registered
        logger.debug(
            "load_custom_tools: módulo '%s' → %d tool(s) registrada(s)",
            module_name,
            newly_registered,
        )

    if registered_count:
        logger.info(
            "load_custom_tools: %d tool(s) custom registrada(s) desde %s",
            registered_count,
            custom_path,
        )
    else:
        logger.debug("load_custom_tools: ninguna tool custom nueva registrada")

    return registered_count


def _register_from_module(
    module: ModuleType,
    registry: ToolRegistry,
    full_name: str,
) -> int:
    """
    Intenta registrar tools desde un módulo usando Opción A o B.
    Retorna el número de tools que intentó registrar (no las efectivamente nuevas).
    """
    # ── Opción A: función register() explícita ────────────────────────────────
    register_fn = getattr(module, "register", None)
    if callable(register_fn):
        try:
            register_fn(registry)
            return 1  # no sabemos cuántas registró, pero al menos intentó
        except Exception as exc:
            logger.warning(
                "load_custom_tools: error en register() de '%s': %s",
                full_name,
                exc,
            )
            return 0

    # ── Opción B: auto-registro por clase ─────────────────────────────────────
    count = 0
    for attr_name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            obj.__module__ == module.__name__  # definida en este módulo (no importada)
            and issubclass(obj, ToolHandler)
            and obj is not ToolHandler
            and not inspect.isabstract(obj)
        ):
            try:
                instance = obj()
                tool_name = instance.definition.name
                if not registry.has(tool_name):
                    registry.register(instance)
                    count += 1
                    logger.debug(
                        "load_custom_tools: registrada '%s' desde clase %s",
                        tool_name,
                        attr_name,
                    )
                else:
                    logger.debug(
                        "load_custom_tools: tool '%s' ya registrada, omitida",
                        tool_name,
                    )
            except Exception as exc:
                logger.warning(
                    "load_custom_tools: error instanciando %s desde '%s': %s",
                    attr_name,
                    full_name,
                    exc,
                )

    return count


def _count_registered(registry: ToolRegistry) -> int:
    return len(registry.available_tool_names())