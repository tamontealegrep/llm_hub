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

DEFAULT_EXCLUDED_MODULES: frozenset[str] = frozenset({
    "__init__",
})


def load_builtin_tools(registry: ToolRegistry) -> int:
    return _load_tools_from_package(
        registry=registry,
        package_name="app.tools.builtin",
        package_path=Path(__file__).parent / "builtin",
        excluded_modules=DEFAULT_EXCLUDED_MODULES,
        source_label="builtin",
    )


def load_custom_tools(registry: ToolRegistry) -> int:
    return _load_tools_from_package(
        registry=registry,
        package_name="app.tools.custom",
        package_path=Path(__file__).parent / "custom",
        excluded_modules=DEFAULT_EXCLUDED_MODULES,
        source_label="custom",
    )


def _load_tools_from_package(
    *,
    registry: ToolRegistry,
    package_name: str,
    package_path: Path,
    excluded_modules: frozenset[str],
    source_label: str,
) -> int:
    if not package_path.is_dir():
        logger.warning(
            "load_%s_tools: directorio no encontrado en %s",
            source_label,
            package_path,
        )
        return 0

    registered_count = 0

    for module_info in pkgutil.iter_modules([str(package_path)]):
        module_name = module_info.name

        if module_name in excluded_modules:
            logger.debug(
                "load_%s_tools: ignorando módulo excluido '%s'",
                source_label,
                module_name,
            )
            continue

        full_module_name = f"{package_name}.{module_name}"

        try:
            module = importlib.import_module(full_module_name)
        except ImportError as exc:
            logger.warning(
                "load_%s_tools: no se pudo importar '%s': %s",
                source_label,
                full_module_name,
                exc,
            )
            continue
        except Exception as exc:
            logger.warning(
                "load_%s_tools: error importando '%s': %s",
                source_label,
                full_module_name,
                exc,
            )
            continue

        before = _count_registered(registry)
        _register_from_module(module, registry, full_module_name, source_label)
        after = _count_registered(registry)

        newly_registered = after - before
        registered_count += newly_registered

        logger.debug(
            "load_%s_tools: módulo '%s' → %d tool(s) registrada(s)",
            source_label,
            module_name,
            newly_registered,
        )

    if registered_count:
        logger.info(
            "load_%s_tools: %d tool(s) registradas desde %s",
            source_label,
            registered_count,
            package_path,
        )
    else:
        logger.debug(
            "load_%s_tools: ninguna tool nueva registrada",
            source_label,
        )

    return registered_count


def _register_from_module(
    module: ModuleType,
    registry: ToolRegistry,
    full_name: str,
    source_label: str,
) -> int:
    register_fn = getattr(module, "register", None)
    if callable(register_fn):
        try:
            register_fn(registry)
            return 1
        except Exception as exc:
            logger.warning(
                "load_%s_tools: error en register() de '%s': %s",
                source_label,
                full_name,
                exc,
            )
            return 0

    count = 0
    for attr_name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            obj.__module__ == module.__name__
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
                        "load_%s_tools: registrada '%s' desde clase %s",
                        source_label,
                        tool_name,
                        attr_name,
                    )
                else:
                    logger.debug(
                        "load_%s_tools: tool '%s' ya registrada, omitida",
                        source_label,
                        tool_name,
                    )
            except Exception as exc:
                logger.warning(
                    "load_%s_tools: error instanciando %s desde '%s': %s",
                    source_label,
                    attr_name,
                    full_name,
                    exc,
                )

    return count


def _count_registered(registry: ToolRegistry) -> int:
    return len(registry.available_tool_names())