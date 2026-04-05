# Manual: Cómo crear herramientas (Tools) para el LLM Hub

## Tabla de contenidos

1. [Conceptos fundamentales](#1-conceptos-fundamentales)
2. [Estructura del sistema de tools](#2-estructura-del-sistema-de-tools)
3. [Tu primera tool en 5 minutos](#3-tu-primera-tool-en-5-minutos)
4. [Anatomía completa de una tool](#4-anatomía-completa-de-una-tool)
5. [El input_schema en detalle](#5-el-input_schema-en-detalle)
6. [Valores de retorno](#6-valores-de-retorno)
7. [Manejo de errores](#7-manejo-de-errores)
8. [Tools síncronas vs asíncronas](#8-tools-síncronas-vs-asíncronas)
9. [Tools con configuración externa (API keys)](#9-tools-con-configuración-externa-api-keys)
10. [Registro: función register() vs auto-registro](#10-registro-función-register-vs-auto-registro)
11. [Descripción efectiva: cómo el modelo decide usar tu tool](#11-descripción-efectiva-cómo-el-modelo-decide-usar-tu-tool)
12. [Receta completa: tool de búsqueda web con API key](#12-receta-completa-tool-de-búsqueda-web-con-api-key)
13. [Catálogo de patrones frecuentes](#13-catálogo-de-patrones-frecuentes)
14. [Checklist antes de publicar](#14-checklist-antes-de-publicar)
15. [Errores comunes y cómo evitarlos](#15-errores-comunes-y-cómo-evitarlos)

---

## 1. Conceptos fundamentales

### ¿Qué es una tool?

Una **tool** (herramienta) es una función que el modelo de lenguaje puede invocar
durante una conversación cuando necesita hacer algo que está más allá del texto puro:
consultar una API, hacer un cálculo, leer un archivo, buscar en una base de datos, etc.

El flujo es el siguiente:

```
Usuario: "¿Qué temperatura hace en Bogotá ahora?"

    → El modelo decide que necesita la tool "get_weather"
    → El modelo genera un tool_call con los argumentos: {"city": "Bogotá"}
    → El sistema ejecuta tu tool con esos argumentos
    → Tu tool llama a una API del clima y devuelve {"temp_c": 14, "condition": "Parcialmente nublado"}
    → El sistema le devuelve ese resultado al modelo
    → El modelo genera la respuesta final: "En Bogotá hay 14°C con cielo parcialmente nublado."
```

### ¿Quién controla qué?

| Componente | Responsabilidad |
|---|---|
| **Tu tool** | Ejecutar la acción y devolver datos |
| **`ToolDefinition`** | Describir al modelo qué hace la tool y qué argumentos acepta |
| **`ToolRegistry`** | Almacenar todas las tools disponibles |
| **`ToolExecutor`** | Invocar tu tool cuando el modelo la solicita |
| **`ConversationService`** | Orquestar el loop completo (modelo → tool → modelo) |

Tú solo tienes que preocuparte por **tu tool**. El resto lo hace el sistema.

---

## 2. Estructura del sistema de tools

```
app/tools/
├── __init__.py
├── interfaces.py      ← Contrato base: clase ToolHandler (solo leer, no modificar)
├── contracts.py       ← Tipos: ToolDefinition, ToolCall, ToolExecutionContext (solo leer)
├── registry.py        ← Almacén de tools (solo leer)
├── executor.py        ← Motor de ejecución (solo leer)
├── builtin.py         ← Tools internas del sistema (get_current_utc_time, sum_numbers)
├── loader.py          ← Auto-descubrimiento de tools en custom/ (solo leer)
└── custom/            ← ← ← AQUÍ CREAS TUS TOOLS
    ├── __init__.py
    ├── web_search.py  ← Tool de búsqueda web (incluida)
    ├── weather.py     ← Tu próxima tool (ejemplo)
    └── ...
```

**Regla de oro:** Solo tocas la carpeta `custom/`. El resto del sistema es
infraestructura que ya está resuelta.

---

## 3. Tu primera tool en 5 minutos

Vamos a crear una tool que convierte temperaturas de Celsius a Fahrenheit.

### Paso 1: Crear el archivo

```
app/tools/custom/temperature_converter.py
```

### Paso 2: Escribir la tool

```python
# app/tools/custom/temperature_converter.py
from typing import Any
from app.tools.contracts import ToolDefinition, ToolExecutionContext
from app.tools.interfaces import ToolHandler


class CelsiusToFahrenheitTool(ToolHandler):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="celsius_to_fahrenheit",
            description=(
                "Convierte una temperatura de grados Celsius a Fahrenheit. "
                "Úsala cuando el usuario pregunte por conversiones de temperatura."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "celsius": {
                        "type": "number",
                        "description": "Temperatura en grados Celsius a convertir.",
                    }
                },
                "required": ["celsius"],
                "additionalProperties": False,
            },
        )

    def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        celsius = float(arguments["celsius"])
        fahrenheit = (celsius * 9 / 5) + 32
        return {
            "celsius": celsius,
            "fahrenheit": round(fahrenheit, 2),
        }


def register(registry) -> None:
    if not registry.has("celsius_to_fahrenheit"):
        registry.register(CelsiusToFahrenheitTool())
```

### Paso 3: Ejecutar

```bash
python demo.py
# La tool aparece automáticamente en "Tools activas"
```

Eso es todo. No hay paso 4.

---

## 4. Anatomía completa de una tool

```python
from typing import Any
from app.tools.contracts import ToolDefinition, ToolExecutionContext
from app.tools.interfaces import ToolHandler


class MiTool(ToolHandler):
    """
    Descripción de la clase para los desarrolladores.
    No confundir con ToolDefinition.description, que es para el modelo.
    """

    # ── Propiedad obligatoria ─────────────────────────────────────────────────
    @property
    def definition(self) -> ToolDefinition:
        """
        Define cómo se presenta esta tool al modelo de lenguaje.
        El modelo lee esto para decidir cuándo y cómo usarla.
        """
        return ToolDefinition(
            name="mi_tool",                    # snake_case, único en el sistema
            description="...",                 # crucial — ver sección 11
            input_schema={...},                # JSON Schema — ver sección 5
        )

    # ── Método obligatorio ────────────────────────────────────────────────────
    def execute(
        self,
        arguments: dict[str, Any],             # argumentos enviados por el modelo
        context: ToolExecutionContext,          # metadata de la conversación
    ) -> Any:                                  # ver sección 6 para tipos válidos
        ...


# ── Función de registro (recomendada) ─────────────────────────────────────────
def register(registry) -> None:
    if not registry.has("mi_tool"):
        registry.register(MiTool())
```

### El objeto `context`

`ToolExecutionContext` contiene información sobre la conversación en curso.
Úsalo cuando tu tool necesite saber de dónde viene la llamada:

```python
context.conversation_id   # str — ID único de la conversación
context.provider_code     # str — "openai", "anthropic", "google", "xai"
context.model_key         # str — "gpt-4o-mini", "claude-3-5-sonnet-latest", etc.
context.metadata          # dict — metadata adicional (extensible en el futuro)
```

Ejemplo de uso: una tool que registra en logs qué modelo la invocó.

```python
def execute(self, arguments, context):
    logger.info(
        "tool invocada por provider=%s model=%s",
        context.provider_code,
        context.model_key,
    )
    ...
```

---

## 5. El input_schema en detalle

El `input_schema` es un objeto [JSON Schema](https://json-schema.org/) que define
los argumentos que el modelo puede enviarle a tu tool. Es lo más importante
para que el modelo genere llamadas correctas.

### Estructura mínima

```python
input_schema = {
    "type": "object",           # siempre "object"
    "properties": {},           # los argumentos (puede ser vacío)
    "additionalProperties": False,  # siempre False (rechaza argumentos no declarados)
}
```

### Sin argumentos (tool sin parámetros)

```python
input_schema = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}
```

Ejemplo: `get_current_utc_time` no necesita ningún argumento.

### Tipos de datos disponibles

#### String

```python
"query": {
    "type": "string",
    "description": "Término de búsqueda.",
    "minLength": 1,       # opcional: longitud mínima
    "maxLength": 500,     # opcional: longitud máxima
    "enum": ["a", "b"],   # opcional: valores permitidos
}
```

#### Número (entero o decimal)

```python
"temperature": {
    "type": "number",     # acepta int y float
    "description": "Temperatura en Celsius.",
    "minimum": -273.15,   # opcional
    "maximum": 1000,      # opcional
}

"count": {
    "type": "integer",    # solo enteros
    "description": "Número de resultados.",
    "minimum": 1,
    "maximum": 100,
    "default": 10,        # valor por defecto (informativo para el modelo)
}
```

#### Booleano

```python
"include_details": {
    "type": "boolean",
    "description": "Si es true, incluye información detallada en la respuesta.",
}
```

#### Array (lista)

```python
"tags": {
    "type": "array",
    "description": "Lista de etiquetas para filtrar.",
    "items": {"type": "string"},   # tipo de cada elemento
    "minItems": 1,                 # opcional
    "maxItems": 10,                # opcional
}
```

#### Enum (opciones fijas)

```python
"format": {
    "type": "string",
    "description": "Formato de salida. 'json' para datos estructurados, 'text' para texto plano.",
    "enum": ["json", "text", "markdown"],
}
```

#### Objeto anidado

```python
"location": {
    "type": "object",
    "description": "Coordenadas geográficas.",
    "properties": {
        "latitude":  {"type": "number", "description": "Latitud (-90 a 90)."},
        "longitude": {"type": "number", "description": "Longitud (-180 a 180)."},
    },
    "required": ["latitude", "longitude"],
    "additionalProperties": False,
}
```

### Campos obligatorios vs opcionales

```python
input_schema = {
    "type": "object",
    "properties": {
        "query":       {"type": "string", "description": "..."},  # obligatorio
        "max_results": {"type": "integer", "description": "...", "default": 5},  # opcional
    },
    "required": ["query"],              # solo los que son obligatorios
    "additionalProperties": False,
}
```

Si un argumento no está en `required`, el modelo puede o no enviarlo.
**Siempre usa `.get()` con valor por defecto al leerlo en `execute()`:**

```python
max_results = int(arguments.get("max_results", 5))  # ← seguro
max_results = int(arguments["max_results"])           # ← puede lanzar KeyError
```

---

## 6. Valores de retorno

Tu método `execute()` puede devolver varios tipos. El `ToolExecutor` los
serializa automáticamente a string para enviarlos al modelo.

### Recomendado: dict

```python
return {
    "temperature_c": 14,
    "temperature_f": 57.2,
    "condition": "Parcialmente nublado",
    "humidity_percent": 78,
}
```

El modelo recibe esto como JSON y puede usar cualquier campo.
**Prefiere dicts con nombres de clave claros y descriptivos.**

### También válido: lista de dicts

```python
return [
    {"title": "Resultado 1", "url": "https://...", "snippet": "..."},
    {"title": "Resultado 2", "url": "https://...", "snippet": "..."},
]
```

### También válido: string

```python
return "La conversión es 57.2°F"
```

Solo úsalo cuando el resultado sea inherentemente texto y no haya
otros campos que el modelo pueda necesitar.

### También válido: número

```python
return 57.2
```

Evítalo en favor de `{"result": 57.2}` para que el contexto sea claro.

### Nunca retornes None directamente

```python
# MAL
return None

# BIEN — el executor serializa None a "null", pero es confuso para el modelo
return {"result": None, "note": "No se encontró resultado"}
```

### Estructura de error en el retorno

Cuando algo falla de forma esperada (parámetro inválido, API sin resultados),
**no lances una excepción**: devuelve un dict con `"error"` y el contexto suficiente
para que el modelo lo entienda.

```python
return {
    "error": "La ciudad 'Xyzzy' no existe en la base de datos.",
    "suggestion": "Verifica el nombre de la ciudad e intenta de nuevo.",
}
```

El modelo puede entonces generar una respuesta informativa al usuario.

Para errores inesperados (bugs, excepciones de red), **sí puedes lanzar**:
el `ToolExecutor` los captura, los loguea, y devuelve al modelo un mensaje
de error estructurado con `is_error=True`.

---

## 7. Manejo de errores

### Captura explícita (errores esperados)

```python
def execute(self, arguments, context):
    city = arguments.get("city", "").strip()

    # Validación de argumentos
    if not city:
        return {"error": "El parámetro 'city' no puede estar vacío."}

    if len(city) > 100:
        return {"error": "El nombre de ciudad es demasiado largo (máx. 100 caracteres)."}

    # Error de negocio
    try:
        data = self._call_weather_api(city)
    except CityNotFoundError:
        return {
            "error": f"No se encontró la ciudad '{city}'.",
            "suggestion": "Prueba con el nombre en inglés o verifica la ortografía.",
        }
    except ApiRateLimitError:
        return {
            "error": "Se superó el límite de consultas a la API del clima.",
            "retry_after_seconds": 60,
        }

    return data
```

### Excepciones que debes dejar propagar

Las excepciones **inesperadas** (bugs, errores de red transitorios,
`ValueError` por datos corruptos) puedes dejarlas propagar.
El `ToolExecutor` las captura y las loguea:

```python
# ToolExecutor hace esto automáticamente:
try:
    raw_output = handler.execute(call.arguments, context)
except Exception as exc:
    logger.exception("Error ejecutando tool '%s'", call.name)
    return ToolExecutionResult(
        tool_call_id=call.id,
        name=call.name,
        content=json.dumps({"error": str(exc), "tool": call.name}),
        is_error=True,
    )
```

### Uso de logging en tu tool

```python
import logging
logger = logging.getLogger(__name__)

def execute(self, arguments, context):
    query = arguments.get("query")
    logger.info("web_search query=%r", query)        # operación normal
    logger.warning("web_search sin resultados q=%r", query)  # algo raro pero OK
    logger.error("web_search fallo de red: %s", exc)         # error importante
```

**No uses `print()`.** Usa siempre el logger.

---

## 8. Tools síncronas vs asíncronas

El sistema soporta ambas. El `ToolExecutor` detecta automáticamente si tu
método `execute()` es una coroutine y la awaita:

```python
# En executor.py — ya resuelto, solo para entender cómo funciona
raw_output = handler.execute(call.arguments, context)
if inspect.isawaitable(raw_output):
    raw_output = await raw_output
```

### ¿Cuándo usar async?

Usa `async def execute()` cuando tu tool haga **I/O**: llamadas HTTP,
consultas a base de datos, lectura de archivos grandes, etc.

```python
async def execute(self, arguments, context):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            data = await response.json()
    return data
```

### ¿Cuándo usar sync?

Usa `def execute()` cuando tu tool sea puramente **computacional** o use
una librería que solo tiene interfaz sincrónica.

```python
def execute(self, arguments, context):
    celsius = float(arguments["celsius"])
    return {"fahrenheit": (celsius * 9 / 5) + 32}
```

### Librería sincrónica dentro de async

Si necesitas usar una librería **sincrónica** dentro de un contexto async
(como `duckduckgo_search`), ejecútala en un executor para no bloquear
el event loop:

```python
import asyncio

async def execute(self, arguments, context):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,                        # usa el ThreadPoolExecutor por defecto
        self._sync_call,             # función sincrónica
        arguments["query"],          # argumentos
    )
    return result

def _sync_call(self, query: str) -> dict:
    # aquí va el código sincrónico
    ...
```

---

## 9. Tools con configuración externa (API keys)

### Patrón recomendado: inyección en __init__

```python
import os
from app.tools.contracts import ToolDefinition, ToolExecutionContext
from app.tools.interfaces import ToolHandler


class BraveSearchTool(ToolHandler):

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("BraveSearchTool requiere una API key de Brave Search.")
        self._api_key = api_key

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="brave_search",
            description="Busca en internet usando Brave Search API.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Consulta de búsqueda."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    async def execute(self, arguments, context):
        # self._api_key disponible aquí
        ...


def register(registry) -> None:
    api_key = os.getenv("BRAVE_API_KEY", "").strip()

    if not api_key:
        # Log informativo — no es un error fatal, la tool simplemente no se registra
        import logging
        logging.getLogger(__name__).info(
            "BraveSearchTool no registrada: BRAVE_API_KEY no está configurada."
        )
        return

    if not registry.has("brave_search"):
        registry.register(BraveSearchTool(api_key=api_key))
```

### Agregar la key al .env

```bash
# .env
BRAVE_API_KEY=BSA...tu_key_aqui...
```

### Agregar la key a Settings (opcional pero recomendado)

Si quieres que la key esté tipada y validada por pydantic-settings:

```python
# app/core/config.py — agregar campo
class Settings(BaseSettings):
    # ... campos existentes ...
    brave_api_key: str | None = None
```

Luego en register():

```python
def register(registry) -> None:
    from app.core.config import get_settings
    settings = get_settings()
    if not settings.brave_api_key:
        return
    if not registry.has("brave_search"):
        registry.register(BraveSearchTool(api_key=settings.brave_api_key))
```

---

## 10. Registro: función register() vs auto-registro

### Opción A: función register() — RECOMENDADA

Tienes control total. Puedes condicionar el registro, pasar argumentos
al constructor, registrar múltiples tools desde un mismo archivo, etc.

```python
def register(registry) -> None:
    # Condicionar registro a una API key
    api_key = os.getenv("MY_API_KEY")
    if not api_key:
        return

    # Registrar múltiples tools desde el mismo archivo
    if not registry.has("search_web"):
        registry.register(WebSearchTool(api_key=api_key))

    if not registry.has("search_images"):
        registry.register(ImageSearchTool(api_key=api_key))
```

### Opción B: auto-registro — para tools simples

Si tu tool no tiene `__init__` personalizado y no necesita condiciones,
el loader la instancia y registra automáticamente. Solo es necesario
que la clase herede de `ToolHandler` y esté en `custom/`.

```python
# El loader hace esto automáticamente:
class MySimpleTool(ToolHandler):
    # sin __init__ → se instancia con MySimpleTool()
    ...

# NO necesitas función register()
```

### Convivencia de ambas opciones

Si defines `register()`, el loader usa esa función y **no** hace el
auto-registro de las clases del mismo módulo. Son mutuamente excluyentes
por módulo.

---

## 11. Descripción efectiva: cómo el modelo decide usar tu tool

La `description` en `ToolDefinition` es **el texto más importante de tu tool**.
El modelo lo lee para decidir:

1. ¿Debo usar esta tool para responder esta pregunta?
2. ¿Qué argumentos tengo que pasarle?

### Qué incluir en la descripción de la tool

```python
description=(
    # 1. QUÉ hace la tool (frase principal, acción concreta)
    "Busca información actualizada en internet y devuelve los resultados más relevantes. "

    # 2. CUÁNDO usarla (casos de uso explícitos)
    "Úsala cuando necesites datos recientes, noticias, precios actuales, "
    "documentación de librerías, o cualquier información que pueda haber cambiado. "

    # 3. Qué devuelve (para que el modelo sepa qué esperar)
    "Devuelve título, URL y extracto de texto de cada resultado."
),
```

### Qué incluir en la descripción de cada argumento

```python
"query": {
    "type": "string",
    "description": (
        # Qué es
        "Consulta de búsqueda en lenguaje natural o palabras clave. "
        # Cómo formularla
        "Escribe como si buscaras en Google. "
        # Ejemplos concretos (muy útiles para el modelo)
        "Ejemplos: 'Python asyncio tutorial 2024', 'precio iPhone Colombia', "
        "'últimas noticias IA'."
    ),
},
```

### Errores comunes en las descripciones

| Error | Por qué es malo | Versión correcta |
|---|---|---|
| `"Busca cosas"` | Ambigua, el modelo no sabe cuándo usarla | `"Busca información en internet cuando el usuario pregunte por datos actuales"` |
| `"Temperatura"` | No describe la unidad ni el rango | `"Temperatura en grados Celsius. Acepta decimales. Ej: -5.0, 0, 100"` |
| No mencionar el retorno | El modelo no sabe qué esperar | Agrega: `"Devuelve {campo1}, {campo2} y {campo3}"` |
| Solo en inglés | Mezcla idiomas con las instrucciones | Usa el mismo idioma del system prompt |

### Ejemplo: descripción pobre vs. descripción efectiva

**Pobre:**
```python
description="Obtiene el clima."
input_schema={
    "properties": {
        "city": {"type": "string", "description": "Ciudad"}
    }
}
```

**Efectiva:**
```python
description=(
    "Obtiene las condiciones meteorológicas actuales para una ciudad. "
    "Úsala cuando el usuario pregunte por el tiempo, temperatura, lluvia o clima "
    "en cualquier ciudad del mundo. "
    "Devuelve temperatura en Celsius y Fahrenheit, condición (soleado, nublado, etc.), "
    "humedad y velocidad del viento."
),
input_schema={
    "properties": {
        "city": {
            "type": "string",
            "description": (
                "Nombre de la ciudad en español o inglés. "
                "Incluye el país si hay ambigüedad. "
                "Ejemplos: 'Bogotá', 'Madrid', 'Buenos Aires, Argentina', 'New York'."
            ),
        }
    }
}
```

---

## 12. Receta completa: tool de búsqueda web con API key

Esta es la plantilla más completa para una tool profesional con API key
externa, async, validación robusta y logging adecuado.

```python
# app/tools/custom/brave_search.py
"""
BraveSearchTool — búsqueda web usando Brave Search API.

Requiere:
    - pip install httpx
    - Variable de entorno: BRAVE_API_KEY

Documentación de la API:
    https://api.search.brave.com/app/documentation/web-search
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from app.tools.contracts import ToolDefinition, ToolExecutionContext
from app.tools.interfaces import ToolHandler

logger = logging.getLogger(__name__)

_API_URL = "https://api.search.brave.com/res/v1/web/search"
_DEFAULT_MAX_RESULTS = 5
_MAX_RESULTS_LIMIT = 10
_TIMEOUT_SECONDS = 15


class BraveSearchTool(ToolHandler):
    """
    Tool de búsqueda web usando Brave Search API.
    Requiere BRAVE_API_KEY en el entorno.
    """

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("BraveSearchTool requiere api_key.")
        self._api_key = api_key

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="brave_search",
            description=(
                "Busca información actualizada en internet usando Brave Search. "
                "Úsala cuando necesites datos recientes, noticias, precios, "
                "documentación, o cualquier información que pueda haber cambiado. "
                "Devuelve título, URL y extracto de los resultados más relevantes."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Consulta de búsqueda. Escribe como en Google. "
                            "Ejemplos: 'mejores frameworks Python 2024', "
                            "'precio dólar Colombia hoy'."
                        ),
                        "minLength": 1,
                        "maxLength": 400,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": f"Número de resultados (1-{_MAX_RESULTS_LIMIT}). Default: {_DEFAULT_MAX_RESULTS}.",
                        "minimum": 1,
                        "maximum": _MAX_RESULTS_LIMIT,
                        "default": _DEFAULT_MAX_RESULTS,
                    },
                    "country": {
                        "type": "string",
                        "description": (
                            "Código de país ISO 3166-1 alpha-2 para localizar resultados. "
                            "Ejemplos: 'CO' (Colombia), 'ES' (España), 'MX' (México), 'US' (EEUU)."
                        ),
                        "default": "CO",
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
        # ── Validar argumentos ────────────────────────────────────────────────
        query = str(arguments.get("query", "")).strip()
        if not query:
            return {"error": "El parámetro 'query' no puede estar vacío.", "results": []}

        max_results = int(arguments.get("max_results", _DEFAULT_MAX_RESULTS))
        max_results = max(1, min(max_results, _MAX_RESULTS_LIMIT))
        country = str(arguments.get("country", "CO")).upper()

        logger.info(
            "brave_search query=%r max=%d country=%s conv=%s",
            query, max_results, country, context.conversation_id,
        )

        # ── Llamar a la API ───────────────────────────────────────────────────
        try:
            results = await self._fetch(query, max_results, country)
        except httpx.TimeoutException:
            logger.warning("brave_search timeout query=%r", query)
            return {
                "error": f"La búsqueda tardó más de {_TIMEOUT_SECONDS}s.",
                "query": query,
                "results": [],
            }
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                logger.error("brave_search auth error — verifica BRAVE_API_KEY")
                return {"error": "Error de autenticación con Brave Search API.", "results": []}
            if exc.response.status_code == 429:
                logger.warning("brave_search rate limit")
                return {"error": "Límite de consultas a Brave Search superado.", "results": []}
            logger.error("brave_search HTTP %d: %s", exc.response.status_code, exc)
            return {"error": f"Error HTTP {exc.response.status_code} de Brave Search.", "results": []}
        except Exception as exc:
            logger.error("brave_search error inesperado: %s", exc)
            return {"error": f"Error inesperado: {exc}", "results": []}

        if not results:
            return {"query": query, "total_results": 0, "results": [], "note": "Sin resultados."}

        return {"query": query, "total_results": len(results), "results": results}

    async def _fetch(self, query: str, max_results: int, country: str) -> list[dict[str, str]]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        params = {"q": query, "count": max_results, "country": country}

        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            response = await client.get(_API_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append({
                "title":   item.get("title", "Sin título"),
                "url":     item.get("url", ""),
                "snippet": item.get("description", "Sin descripción"),
            })
        return results


def register(registry) -> None:
    api_key = os.getenv("BRAVE_API_KEY", "").strip()
    if not api_key:
        logger.info("BraveSearchTool no registrada: BRAVE_API_KEY no configurada.")
        return
    if not registry.has("brave_search"):
        registry.register(BraveSearchTool(api_key=api_key))
```

---

## 13. Catálogo de patrones frecuentes

### Tool sin argumentos

```python
class GetServerTimeTool(ToolHandler):
    @property
    def definition(self):
        return ToolDefinition(
            name="get_server_time",
            description="Devuelve la fecha y hora actuales del servidor en UTC.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        )

    def execute(self, arguments, context):
        from datetime import datetime, timezone
        return {"utc": datetime.now(timezone.utc).isoformat()}
```

### Tool con enum (opciones fijas)

```python
class CurrencyConverterTool(ToolHandler):
    @property
    def definition(self):
        return ToolDefinition(
            name="convert_currency",
            description="Convierte entre monedas. Solo soporta USD, EUR, COP y MXN.",
            input_schema={
                "type": "object",
                "properties": {
                    "amount":   {"type": "number", "description": "Monto a convertir."},
                    "from_currency": {
                        "type": "string",
                        "enum": ["USD", "EUR", "COP", "MXN"],
                        "description": "Moneda de origen.",
                    },
                    "to_currency": {
                        "type": "string",
                        "enum": ["USD", "EUR", "COP", "MXN"],
                        "description": "Moneda de destino.",
                    },
                },
                "required": ["amount", "from_currency", "to_currency"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments, context):
        # ... lógica de conversión
        pass
```

### Tool que devuelve múltiples items

```python
class ListFilesTool(ToolHandler):
    @property
    def definition(self):
        return ToolDefinition(
            name="list_files",
            description="Lista archivos en un directorio.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Ruta del directorio."},
                    "extension": {
                        "type": "string",
                        "description": "Filtrar por extensión. Ej: '.py', '.txt'. Opcional.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments, context):
        import os, pathlib
        path = pathlib.Path(arguments["path"])
        ext  = arguments.get("extension")

        if not path.is_dir():
            return {"error": f"'{path}' no es un directorio válido."}

        files = [f.name for f in path.iterdir() if f.is_file()]
        if ext:
            files = [f for f in files if f.endswith(ext)]

        return {"path": str(path), "count": len(files), "files": files}
```

### Múltiples tools en un mismo archivo

```python
# app/tools/custom/math_tools.py

class AddTool(ToolHandler):
    @property
    def definition(self):
        return ToolDefinition(name="math_add", description="Suma dos números.", ...)

    def execute(self, arguments, context):
        return {"result": arguments["a"] + arguments["b"]}


class MultiplyTool(ToolHandler):
    @property
    def definition(self):
        return ToolDefinition(name="math_multiply", description="Multiplica dos números.", ...)

    def execute(self, arguments, context):
        return {"result": arguments["a"] * arguments["b"]}


def register(registry) -> None:
    if not registry.has("math_add"):
        registry.register(AddTool())
    if not registry.has("math_multiply"):
        registry.register(MultiplyTool())
```

---

## 14. Checklist antes de publicar

Antes de dar por lista una tool, verifica:

### Definición

- [ ] `name` es único y en `snake_case`
- [ ] `description` explica **qué hace**, **cuándo usarla** y **qué devuelve**
- [ ] `input_schema` tiene `"type": "object"` y `"additionalProperties": False`
- [ ] Cada argumento tiene `"description"` con ejemplos
- [ ] Los argumentos opcionales tienen `"default"` documentado
- [ ] Solo los argumentos obligatorios están en `"required"`

### Implementación

- [ ] Los argumentos opcionales se leen con `.get("clave", valor_por_defecto)`
- [ ] Los argumentos de texto se hacen `.strip()` antes de validar
- [ ] Los errores esperados se devuelven como `{"error": "..."}`, no como excepciones
- [ ] Se usa `logger` (no `print()`)
- [ ] Si hay I/O → `async def execute()`
- [ ] Si usa librería sincrónica en async → `run_in_executor`
- [ ] Si necesita API key → se lee de `os.getenv()` y se valida en `register()`

### Registro

- [ ] Tiene función `register()` que verifica `registry.has()` antes de registrar
- [ ] Si requiere API key, `register()` retorna silenciosamente si no está configurada

### Prueba manual

- [ ] Ejecutar `python demo.py` y verificar que aparece en "Tools activas"
- [ ] Hacer una pregunta que debería invocar la tool y confirmar que funciona
- [ ] Hacer una pregunta que NO debería invocarla y confirmar que no se invoca
- [ ] Probar con `/stream on` y `/stream off` para verificar ambos modos

---

## 15. Errores comunes y cómo evitarlos

### Error 1: Argumento obligatorio sin `.get()`

```python
# MAL — lanza KeyError si el modelo no envía max_results
max_results = int(arguments["max_results"])

# BIEN
max_results = int(arguments.get("max_results", 5))
```

### Error 2: `additionalProperties` omitido

```python
# MAL — el modelo puede enviar argumentos no declarados
input_schema = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "..."}},
    "required": ["query"],
    # falta additionalProperties
}

# BIEN
input_schema = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "..."}},
    "required": ["query"],
    "additionalProperties": False,  # ← siempre
}
```

### Error 3: Lanzar excepción por un error esperado

```python
# MAL — el ToolExecutor captura la excepción pero la respuesta al modelo es genérica
def execute(self, arguments, context):
    if not arguments.get("city"):
        raise ValueError("city es obligatorio")  # ← MAL

# BIEN — el modelo puede usar esta información para responder al usuario
def execute(self, arguments, context):
    if not arguments.get("city"):
        return {"error": "El parámetro 'city' no fue proporcionado."}
```

### Error 4: Nombre de tool con espacios o camelCase

```python
# MAL
name="WebSearch"        # camelCase
name="web search"       # con espacio
name="Web_Search"       # mayúsculas

# BIEN
name="web_search"       # snake_case, todo minúscula
```

### Error 5: Olvidar `async` en una tool con I/O

```python
# MAL — bloquea el event loop durante la llamada HTTP
def execute(self, arguments, context):
    import requests
    response = requests.get(url)   # sincrónico, bloquea todo
    return response.json()

# BIEN — no bloquea el event loop
async def execute(self, arguments, context):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    return response.json()
```

### Error 6: No verificar si la tool ya existe antes de registrar

```python
# MAL — lanza ValueError si el sistema intenta registrar dos veces
def register(registry):
    registry.register(MyTool())   # ← ValueError si ya existe

# BIEN
def register(registry):
    if not registry.has("my_tool"):
        registry.register(MyTool())
```

### Error 7: Descripción que no activa la tool cuando debería

Si el modelo nunca usa tu tool aunque la pregunta sea obvia, el problema
suele estar en la `description`. Agrega:
- Casos de uso explícitos ("Úsala cuando el usuario pregunte por...")
- Sinónimos ("clima", "tiempo", "temperatura", "lluvia")
- Ejemplos concretos de queries

```python
# Descripción que el modelo suele ignorar
description="Obtiene datos del clima."

# Descripción que activa la tool correctamente
description=(
    "Obtiene las condiciones meteorológicas actuales para una ciudad. "
    "Úsala cuando el usuario pregunte por el tiempo, clima, temperatura, "
    "si va a llover, si hace frío o calor, o cualquier condición atmosférica. "
    "Funciona para cualquier ciudad del mundo."
)
```
