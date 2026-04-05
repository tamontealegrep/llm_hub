# app/tools/custom/web_search.py
"""
WebSearchTool — búsqueda web sin API key mediante DuckDuckGo.

Dependencia:
    pip install duckduckgo-search

La librería duckduckgo-search no requiere registro ni API key.
Si prefieres usar SerpAPI, Brave Search o Tavily, consulta el manual
de tools (docs/MANUAL_TOOLS.md) para ver cómo adaptar esta misma clase.

Comportamiento:
    - Recibe una consulta de texto y un número máximo de resultados.
    - Devuelve una lista de resultados con título, URL y snippet.
    - Maneja timeouts, errores de red y ausencia de resultados de forma explícita.
    - Es async-safe: llama a la API sincrónica en un executor para no bloquear.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.tools.contracts import ToolDefinition, ToolExecutionContext
from app.tools.interfaces import ToolHandler

logger = logging.getLogger(__name__)

# Número máximo de resultados que el modelo puede solicitar
_MAX_RESULTS_LIMIT = 10
# Timeout en segundos para la llamada a DuckDuckGo
_SEARCH_TIMEOUT_SECONDS = 15


class WebSearchTool(ToolHandler):
    """
    Herramienta de búsqueda web usando DuckDuckGo.

    Retorna hasta `max_results` resultados con:
      - title   : título de la página
      - url     : URL del resultado
      - snippet : extracto de texto de la página
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description=(
                "Realiza una búsqueda en internet y devuelve los resultados más relevantes. "
                "Úsala cuando necesites información actual, datos recientes, noticias, "
                "precios, documentación externa o cualquier dato que no esté en tu contexto. "
                "Devuelve título, URL y un extracto de cada resultado."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Consulta de búsqueda en lenguaje natural o palabras clave. "
                            "Escribe como si buscaras en Google. "
                            "Ejemplos: 'Python asyncio tutorial 2024', "
                            "'precio iPhone 16 Colombia', "
                            "'últimas noticias inteligencia artificial'."
                        ),
                        "minLength": 1,
                        "maxLength": 400,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Número máximo de resultados a retornar. "
                            f"Entre 1 y {_MAX_RESULTS_LIMIT}. Por defecto 5."
                        ),
                        "minimum": 1,
                        "maximum": _MAX_RESULTS_LIMIT,
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        # ── 1. Validar y extraer argumentos ──────────────────────────────────
        query = str(arguments.get("query", "")).strip()
        if not query:
            return {
                "error": "El parámetro 'query' no puede estar vacío.",
                "results": [],
            }

        max_results = int(arguments.get("max_results", 5))
        max_results = max(1, min(max_results, _MAX_RESULTS_LIMIT))

        logger.info(
            "web_search query=%r max_results=%d conversation_id=%s",
            query,
            max_results,
            context.conversation_id,
        )

        # ── 2. Ejecutar búsqueda ──────────────────────────────────────────────
        try:
            results = await asyncio.wait_for(
                self._search(query, max_results),
                timeout=_SEARCH_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("web_search timeout query=%r", query)
            return {
                "error": f"La búsqueda tardó más de {_SEARCH_TIMEOUT_SECONDS}s y fue cancelada.",
                "query": query,
                "results": [],
            }
        except ImportError:
            logger.error("web_search: duckduckgo-search no está instalado")
            return {
                "error": (
                    "La librería 'duckduckgo-search' no está instalada. "
                    "Ejecuta: pip install duckduckgo-search"
                ),
                "results": [],
            }
        except Exception as exc:
            logger.error("web_search error query=%r exc=%s: %s", query, type(exc).__name__, exc)
            return {
                "error": f"Error al realizar la búsqueda: {exc}",
                "query": query,
                "results": [],
            }

        # ── 3. Normalizar y retornar ─────────────────────────────────────────
        if not results:
            return {
                "query": query,
                "total_results": 0,
                "results": [],
                "note": "No se encontraron resultados para esta consulta.",
            }

        return {
            "query": query,
            "total_results": len(results),
            "results": results,
        }

    # ── Método privado: llamada a DuckDuckGo en executor ─────────────────────

    async def _search(self, query: str, max_results: int) -> list[dict[str, str]]:
        """
        Llama a DuckDuckGo en un ThreadPoolExecutor para no bloquear el
        event loop (la librería duckduckgo-search es sincrónica).
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,  # executor por defecto del loop
            self._search_sync,
            query,
            max_results,
        )

    def _search_sync(self, query: str, max_results: int) -> list[dict[str, str]]:
        """
        Búsqueda sincrónica. Se ejecuta en un thread separado.
        Aislada aquí para facilitar el reemplazo por otro provider.
        """
        from ddgs import DDGS

        results: list[dict[str, str]] = []

        with DDGS() as ddgs:
            for raw in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title":   raw.get("title", "Sin título"),
                        "url":     raw.get("href", ""),
                        "snippet": raw.get("body", "Sin descripción"),
                    }
                )

        return results


# ── Función de registro ───────────────────────────────────────────────────────

def register(registry) -> None:
    """
    Función estándar de registro. Llamada automáticamente por el loader.
    Puedes registrar varias tools desde un mismo archivo si las defines aquí.
    """
    if not registry.has("web_search"):
        registry.register(WebSearchTool())