# Roadmap / TODO del proyecto `llm_hub`

## Objetivo final

Construir un **backend de chatbot multiproveedor y multiusuario** con:

- soporte para múltiples LLM providers:
  - OpenAI
  - Anthropic
  - Google Gemini
  - xAI / Grok
- conversaciones persistentes
- soporte para **tool calling**
- API con **FastAPI**
- base de datos **PostgreSQL**
- ejecución en **Docker**

---

# Fase 1 — Tool Calling MVP en memoria

## Objetivo

Agregar soporte real para herramientas en el flujo conversacional, manteniendo por ahora:

- repositorio en memoria
- demos por consola
- soporte de tools solo en `send_message()`
- sin persistencia SQL todavía

## TODO

### Contratos y tipos
- [x] Agregar `ToolDefinition`
- [x] Agregar `ToolCall`
- [x] Agregar `ToolExecutionContext`
- [x] Agregar `ToolExecutionResult`

### Capa LLM
- [x] Extender `NormalizedMessage` para soportar:
  - [x] mensajes `tool`
  - [x] `tool_calls` en mensajes `assistant`
- [x] Extender `LLMRequest` para incluir:
  - [x] `tools`
  - [x] `tool_choice` opcional
- [x] Extender `LLMCompletionResult` para incluir:
  - [x] `tool_calls`

> **Nota:** `tool_choice` ya existe en el contrato, pero todavía no se está explotando explícitamente desde el servicio o los demos.


### Adapters / Providers
- [x] Modificar `BaseProviderAdapter` para:
  - [x] bindear herramientas al modelo (`bind_tools`)
  - [x] convertir mensajes assistant con `tool_calls` a `AIMessage`
  - [x] convertir mensajes `tool` a `ToolMessage`
  - [x] extraer `tool_calls` desde la respuesta del provider
  - [x] mantener compatibilidad con respuestas sin tools
- [ ] Verificar compatibilidad de tool calling en cada provider/model:
  - [ ] OpenAI
  - [ ] Anthropic
  - [ ] Gemini
  - [ ] xAI / Grok

### Dominio conversacional
- [x] Extender `ConversationMessage` para guardar:
  - [x] `tool_call_id`
  - [x] `name`
  - [x] `tool_calls`
- [x] Extender `ConversationSession` para:
  - [x] agregar `add_tool_message(...)`
  - [x] permitir guardar assistant messages con `tool_calls`

### Construcción de contexto
- [x] Modificar `SimpleContextBuilder` para incluir:
  - [x] mensajes `assistant` con `tool_calls`
  - [x] mensajes `tool`

### Registro y ejecución de tools
- [x] Crear `ToolRegistry`
- [x] Crear `ToolExecutor`
- [x] Definir herramientas builtin de prueba:
  - [x] `get_current_utc_time`
  - [x] `sum_numbers`

### Servicio conversacional
- [x] Modificar `ConversationService.send_message()` para:
  - [x] enviar tools al modelo
  - [x] detectar `tool_calls`
  - [x] guardar assistant message con `tool_calls`
  - [x] ejecutar herramientas
  - [x] guardar mensajes `tool`
  - [x] volver a invocar al modelo hasta obtener respuesta final
  - [x] cortar el loop si supera `max_tool_iterations`
- [x] Agregar excepción `ToolLoopLimitExceededError`
- [x] Agregar excepción `ToolNotFoundError`

### Bootstrap
- [x] Registrar `ToolRegistry` y `ToolExecutor` en `build_container()`
- [x] Permitir `with_builtin_tools=True`
- [x] Agregar `DEFAULT_MAX_TOOL_ITERATIONS` a settings

### Demos
- [x] Crear `demo_chat_tools.py`
- [x] Actualizar `demo_chat.py` para awareness de tools
- [x] Actualizar `demo_chat_multi.py` para awareness de tools
- [x] Actualizar `demo_chat_stream.py` para aclarar que streaming con tools no está soportado todavía

## Limitaciones aceptadas en esta fase
- [x] `stream_message()` no soporta tool calling
- [x] no hay persistencia SQL
- [x] no hay multiusuario real
- [x] no hay endpoints FastAPI todavía

> Estas no son tareas hechas “positivas”, sino **limitaciones vigentes aceptadas** en la fase actual.

## Criterio de cierre de Fase 1
- [x] el chat no streaming puede usar tools
- [x] el historial guarda `tool_calls` y mensajes `tool`
- [ ] los demos funcionan
- [x] el sistema sigue soportando conversación normal sin tools

---

# Fase 2 — Persistencia SQL con PostgreSQL

## Objetivo

Reemplazar el repositorio en memoria por una persistencia real en PostgreSQL.

## TODO

### Dependencias e infraestructura
- [ ] Agregar dependencias para persistencia:
  - [ ] `sqlalchemy`
  - [ ] `asyncpg`
  - [ ] `alembic`
- [ ] Agregar configuración de base de datos en `Settings`
  - [ ] `DATABASE_URL`
- [ ] Crear módulo de conexión async a PostgreSQL
- [ ] Crear session factory / engine async

### Modelado de datos
- [ ] Diseñar esquema SQL alineado con el dominio conversacional y tool calling
- [ ] Crear modelos ORM
- [ ] Crear migraciones Alembic iniciales
- [ ] Definir índices necesarios

### Repositorios
- [ ] Implementar `SqlConversationRepository`
- [ ] Mantener la interfaz `ConversationRepository`
- [ ] Hacer que `build_container()` permita usar repositorio SQL
- [ ] Persistir correctamente:
  - [ ] conversación
  - [ ] mensajes user
  - [ ] mensajes assistant
  - [ ] mensajes tool
  - [ ] tool calls del assistant

### Adaptaciones al dominio
- [ ] Evaluar agregar `id` a `ConversationMessage`
- [ ] Evaluar agregar campos persistibles para usage y raw response
- [ ] Mapear correctamente `tool_calls` a JSON/JSONB

### Consultas necesarias
- [ ] Obtener una conversación por id
- [ ] Listar conversaciones de un usuario
- [ ] Obtener mensajes de una conversación en orden
- [ ] Guardar cambios de modelo/provider
- [ ] Guardar historial incremental durante el tool loop

### Migraciones
- [ ] Crear migración inicial
- [ ] Crear comandos de upgrade/downgrade
- [ ] Documentar cómo correr migraciones localmente

### Testing
- [ ] Tests de integración contra PostgreSQL
- [ ] Test de lectura/escritura de conversaciones
- [ ] Test de persistencia de `tool_calls`
- [ ] Test de orden cronológico de mensajes

## Tablas mínimas que hay que construir en PostgreSQL

### 1. `users`
Tabla de usuarios dueños de las conversaciones.

**Campos mínimos sugeridos:**
- [ ] `id` UUID PK
- [ ] `auth_subject` VARCHAR/TEXT UNIQUE NULL
- [ ] `email` VARCHAR UNIQUE NULL
- [ ] `display_name` VARCHAR NULL
- [ ] `created_at` TIMESTAMPTZ NOT NULL
- [ ] `updated_at` TIMESTAMPTZ NOT NULL

**Notas:**
- Si luego usas auth externa, `auth_subject` puede guardar el subject del proveedor.
- Si luego haces auth propia, aquí podrías sumar `password_hash`.

---

### 2. `conversations`
Tabla principal de conversaciones.

**Campos mínimos sugeridos:**
- [ ] `id` UUID PK
- [ ] `user_id` UUID FK -> `users.id`
- [ ] `current_provider_code` VARCHAR NOT NULL
- [ ] `current_model_key` VARCHAR NOT NULL
- [ ] `system_prompt` TEXT NULL
- [ ] `title` VARCHAR NULL
- [ ] `created_at` TIMESTAMPTZ NOT NULL
- [ ] `updated_at` TIMESTAMPTZ NOT NULL

**Índices sugeridos:**
- [ ] índice por `user_id`
- [ ] índice por `(user_id, updated_at DESC)`

---

### 3. `messages`
Tabla de mensajes de cada conversación.

**Campos mínimos sugeridos:**
- [ ] `id` UUID PK
- [ ] `conversation_id` UUID FK -> `conversations.id`
- [ ] `role` VARCHAR NOT NULL
- [ ] `content` TEXT NOT NULL
- [ ] `provider_code` VARCHAR NULL
- [ ] `model_key` VARCHAR NULL
- [ ] `tool_call_id` VARCHAR NULL
- [ ] `tool_name` VARCHAR NULL
- [ ] `tool_calls_json` JSONB NOT NULL DEFAULT `[]`
- [ ] `prompt_tokens` INTEGER NULL
- [ ] `completion_tokens` INTEGER NULL
- [ ] `total_tokens` INTEGER NULL
- [ ] `raw_response_json` JSONB NULL
- [ ] `created_at` TIMESTAMPTZ NOT NULL

**Índices sugeridos:**
- [ ] índice por `(conversation_id, created_at)`
- [ ] índice por `tool_call_id` si quieres rastrear tools más fácilmente

**Notas:**
- `tool_calls_json` sirve para guardar los `tool_calls` de mensajes `assistant`
- `tool_call_id` y `tool_name` sirven especialmente para mensajes `tool`
- Si prefieres simplificar el MVP, puedes dejar `raw_response_json` para una migración posterior

## Tablas opcionales para más adelante

### 4. `conversation_summaries` (opcional)
Para resumir conversaciones largas y hacer trimming por contexto.

**Campos sugeridos:**
- [ ] `id`
- [ ] `conversation_id`
- [ ] `summary_text`
- [ ] `from_message_id`
- [ ] `to_message_id`
- [ ] `created_at`

---

### 5. `tool_execution_logs` (opcional)
Si quieres trazabilidad detallada de herramientas.

**Campos sugeridos:**
- [ ] `id`
- [ ] `conversation_id`
- [ ] `message_id` o `tool_call_id`
- [ ] `tool_name`
- [ ] `arguments_json`
- [ ] `result_json`
- [ ] `is_error`
- [ ] `created_at`

> Para el MVP **no es obligatoria**, porque parte de esa información ya vive en `messages`.

## Criterio de cierre de Fase 2
- [ ] se puede levantar PostgreSQL y correr migraciones
- [ ] las conversaciones quedan persistidas
- [ ] los mensajes user/assistant/tool se almacenan correctamente
- [ ] al reiniciar la app no se pierde el historial
- [ ] el tool calling sigue funcionando con repositorio SQL

---

# Fase 3 — FastAPI + API REST + multiusuario

## Objetivo

Exponer el backend vía FastAPI para que lo pueda consumir un frontend o cliente externo.

## TODO

### Estructura API
- [ ] Crear módulo `app/api/`
- [ ] Crear routers separados por dominio:
  - [ ] `conversations`
  - [ ] `messages`
  - [ ] `providers`
  - [ ] `tools`
  - [ ] `health`

### Dependencias FastAPI
- [ ] Crear dependencia para obtener `ConversationService`
- [ ] Crear dependencia para obtener settings
- [ ] Crear dependencia para usuario autenticado (`current_user`)
- [ ] Integrar sesión SQL por request

### Schemas / DTOs
- [ ] Crear modelos Pydantic de request/response para:
  - [ ] crear conversación
  - [ ] cambiar modelo/provider
  - [ ] enviar mensaje
  - [ ] listar conversaciones
  - [ ] listar mensajes
  - [ ] listar providers
  - [ ] listar tools

### Endpoints mínimos
- [ ] `GET /health`
- [ ] `GET /providers`
- [ ] `GET /tools`
- [ ] `POST /conversations`
- [ ] `GET /conversations`
- [ ] `GET /conversations/{conversation_id}`
- [ ] `PATCH /conversations/{conversation_id}/model`
- [ ] `GET /conversations/{conversation_id}/messages`
- [ ] `POST /conversations/{conversation_id}/messages`

### Multiusuario
- [ ] Definir estrategia de autenticación:
  - [ ] JWT propio
  - [ ] proveedor externo
- [ ] Asociar cada conversación a un `user_id`
- [ ] Garantizar ownership:
  - [ ] un usuario solo puede ver sus conversaciones
  - [ ] un usuario solo puede escribir en sus conversaciones
- [ ] Filtrar siempre por `user_id`

### Manejo de errores
- [ ] Crear handlers globales para `AppError`
- [ ] Normalizar respuestas HTTP de error
- [ ] Mapear errores de dominio a status codes adecuados

### Observabilidad básica
- [ ] Logging de requests
- [ ] Logging de errores
- [ ] Logging de provider/model usados

### Testing
- [ ] Tests de endpoints
- [ ] Tests de ownership
- [ ] Tests de envío de mensajes
- [ ] Tests de tool calling vía API

## Criterio de cierre de Fase 3
- [ ] se puede crear conversación por API
- [ ] se puede enviar mensaje por API
- [ ] se puede listar historial por API
- [ ] hay aislamiento real por usuario
- [ ] el backend ya sirve como chatbot multiproveedor básico

---

# Fase 4 — Streaming productivo y tools en streaming

## Objetivo

Agregar streaming usable desde FastAPI y resolver la historia de tools + streaming.

## TODO

### Decisión de estrategia
- [ ] Elegir una estrategia inicial:
  - [ ] **híbrida recomendada**: resolver tools internamente y streamear solo la respuesta final
  - [ ] **streaming completo**: emitir también eventos de tool calling

> Recomendación: empezar por la estrategia **híbrida**.

### Contratos de streaming
- [ ] Revisar `LLMStreamEvent`
- [ ] Si haces streaming completo, agregar tipos de evento como:
  - [ ] `tool_call_start`
  - [ ] `tool_call_delta`
  - [ ] `tool_call_end`
  - [ ] `tool_result`
- [ ] Definir contrato estable para frontend

### API streaming
- [ ] Agregar endpoint streaming:
  - [ ] SSE (`text/event-stream`) o
  - [ ] WebSocket
- [ ] Permitir streaming por conversación
- [ ] Soportar cancelación de cliente

### Servicio
- [ ] Implementar `stream_message()` con estrategia elegida
- [ ] Si hay tools:
  - [ ] reconstruir tool calls
  - [ ] ejecutar tools
  - [ ] continuar la conversación
- [ ] Persistir la respuesta final cuando termine el stream

### Providers
- [ ] Normalizar diferencias de streaming entre providers
- [ ] Testear comportamiento de tool calling streameado por provider

### UX / protocolo cliente
- [ ] Definir qué eventos ve el frontend
- [ ] Definir si el frontend verá pasos internos de tools o solo respuesta final
- [ ] Documentar el protocolo de streaming

## Criterio de cierre de Fase 4
- [ ] existe endpoint de streaming funcional
- [ ] el frontend puede consumirlo
- [ ] no se rompen las conversaciones
- [ ] las tools y el historial siguen siendo consistentes

---

# Fase 5 — Dockerización

## Objetivo

Empaquetar todo el sistema para ejecución local y despliegue más simple.

## TODO

### Imagen de la app
- [ ] Crear `Dockerfile`
- [ ] Crear `.dockerignore`
- [ ] Instalar dependencias en imagen
- [ ] Definir comando de arranque de FastAPI / uvicorn

### Orquestación local
- [ ] Crear `docker-compose.yml`
- [ ] Incluir servicios:
  - [ ] `app`
  - [ ] `postgres`
- [ ] Configurar red interna
- [ ] Configurar volúmenes para PostgreSQL

### Variables de entorno
- [ ] Pasar API keys por env vars
- [ ] Pasar `DATABASE_URL`
- [ ] Pasar settings del backend

### Arranque
- [ ] Asegurar que la DB esté lista antes de iniciar app
- [ ] Correr migraciones al iniciar o documentar el paso manual
- [ ] Agregar healthchecks

### Documentación
- [ ] Documentar cómo levantar todo con Docker
- [ ] Documentar cómo apagarlo
- [ ] Documentar cómo resetear base local

## Criterio de cierre de Fase 5
- [ ] `docker compose up` levanta app + postgres
- [ ] la app queda accesible
- [ ] las migraciones corren
- [ ] el backend queda listo para pruebas integradas

---

# Fase 6 — Hardening / producción

## Objetivo

Convertir el MVP en un backend más robusto para uso real.

## TODO

### Seguridad
- [ ] Validar CORS
- [ ] Rate limiting por usuario
- [ ] Sanitización de inputs
- [ ] Validar acceso a tools sensibles
- [ ] Revisar secretos y gestión de API keys

### Calidad / Testing
- [ ] Tests unitarios
- [ ] Tests de integración
- [ ] Tests end-to-end
- [ ] Mocking de providers para CI

### Observabilidad
- [ ] Logs estructurados
- [ ] Métricas
- [ ] tracing opcional
- [ ] monitoreo de errores

### Rendimiento y contexto
- [ ] Token trimming real
- [ ] Resúmenes automáticos
- [ ] paginación de mensajes
- [ ] optimización de queries SQL

### Funcionalidad extra
- [ ] endpoint para modelos disponibles por provider
- [ ] títulos automáticos de conversación
- [ ] borrado de conversaciones
- [ ] archivado
- [ ] exportación de historial
- [ ] tool registry configurable por usuario/tenant

## Criterio de cierre de Fase 6
- [ ] backend estable para uso real
- [ ] observabilidad suficiente
- [ ] seguridad básica cubierta
- [ ] rendimiento aceptable en producción

---

# Orden recomendado de implementación

1. [ ] **Fase 1** — Tool calling MVP en memoria
2. [ ] **Fase 2** — PostgreSQL + persistencia
3. [ ] **Fase 3** — FastAPI + multiusuario
4. [ ] **Fase 4** — Streaming productivo
5. [ ] **Fase 5** — Docker
6. [ ] **Fase 6** — Hardening / producción

---

# MVP mínimo usable

Si quieres llegar lo antes posible a una versión usable por frontend, el camino mínimo sería:

- [ ] Fase 1 completa
- [ ] Fase 2 completa
- [ ] Fase 3 completa
- [ ] Fase 5 completa

Con eso ya tendrías:

- backend multiproveedor
- tool calling
- persistencia en Postgres
- multiusuario básico
- endpoints FastAPI
- dockerización