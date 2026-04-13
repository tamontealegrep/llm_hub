import copy
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, ClassVar
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.shared.exceptions import (
    AppError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.capabilities.common.models import TokenUsage
from app.capabilities.chat.contracts import (
    ChatCompletionResult,
    ChatRequest,
    ChatStreamEvent,
    ChatStreamEventType,
)
from app.runtime.providers.langchain.factory import LangChainProviderFactory
from app.runtime.providers.base import ChatProviderAdapter
from app.tools.contracts import ToolCall, ToolDefinition
from app.model_catalog.service import ModelCatalogService

logger = logging.getLogger(__name__)


class BaseLangChainChatAdapter(ChatProviderAdapter, ABC):

    provider_code: ClassVar[str]

    def __init__(
        self,
        factory: LangChainProviderFactory,
        model_catalog: ModelCatalogService,
    ) -> None:
        self._factory = factory
        self._model_catalog = model_catalog

    # ------------------------------------------------------------------
    # Método abstracto — cada adapter concreto lo implementa
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_chat_model(self, request: ChatRequest) -> BaseChatModel:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Preparación del modelo y los mensajes
    # ------------------------------------------------------------------

    def _prepare_model(self, request: ChatRequest) -> Any:
        model = self._build_chat_model(request)

        if not request.tools:
            return model

        bind_tools = getattr(model, "bind_tools", None)
        if bind_tools is None:
            raise ProviderError(
                f"El proveedor '{self.provider_code}' no soporta tool calling"
            )

        tool_schemas = [self._to_langchain_tool_schema(t) for t in request.tools]

        try:
            if request.tool_choice is not None:
                return bind_tools(tool_schemas, tool_choice=request.tool_choice)
            return bind_tools(tool_schemas)
        except TypeError:
            # Algunos providers no aceptan tool_choice → reintentar sin él
            if request.tool_choice is not None:
                return bind_tools(tool_schemas)
            raise
        except NotImplementedError as exc:
            raise ProviderError(
                f"El proveedor '{self.provider_code}' no soporta tool calling"
            ) from exc

    def _prepare_messages(self, request: ChatRequest) -> list:
        return self._to_langchain_messages(request.messages)

    # ------------------------------------------------------------------
    # Conversión de esquemas y mensajes
    # ------------------------------------------------------------------

    def _clean_schema_for_gemini(self, schema: Any) -> None:
        """
        Elimina recursivamente 'additionalProperties' del esquema, ya que Gemini
        no lo soporta y lanza advertencias o errores.
        """
        if isinstance(schema, dict):
            schema.pop("additionalProperties", None)
            for value in schema.values():
                self._clean_schema_for_gemini(value)
        elif isinstance(schema, list):
            for item in schema:
                self._clean_schema_for_gemini(item)

    def _to_langchain_tool_schema(self, tool: ToolDefinition) -> dict[str, Any]:
        parameters = copy.deepcopy(tool.input_schema)

        if self.provider_code == "google":
            self._clean_schema_for_gemini(parameters)

        if "type" not in parameters:
            parameters["type"] = "object"

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        }

    def _to_langchain_messages(self, messages: list) -> list:
        """
        Convierte ChatMessage a mensajes LangChain.
        Soporta roles: system, user, assistant (con o sin tool_calls), tool.
        """
        converted = []

        for message in messages:
            if message.role == "system":
                converted.append(SystemMessage(content=message.content))

            elif message.role == "user":
                converted.append(HumanMessage(content=message.content))

            elif message.role == "assistant":
                if message.tool_calls:
                    converted.append(
                        AIMessage(
                            content=message.content,
                            tool_calls=[
                                {
                                    "id": tc.id,
                                    "name": tc.name,
                                    "args": tc.arguments,
                                    "type": "tool_call",
                                }
                                for tc in message.tool_calls
                            ],
                        )
                    )
                else:
                    converted.append(AIMessage(content=message.content))

            elif message.role == "tool":
                if not message.tool_call_id:
                    raise ValueError(
                        "ChatMessage con role='tool' requiere tool_call_id. "
                        f"Contenido: {message.content!r}"
                    )
                converted.append(
                    ToolMessage(
                        content=message.content,
                        tool_call_id=message.tool_call_id,
                        name=message.name,
                    )
                )

            else:
                raise ValueError(
                    f"Rol de mensaje desconocido: {message.role!r}. "
                    "Roles válidos: 'system', 'user', 'assistant', 'tool'."
                )

        return converted

    # ------------------------------------------------------------------
    # Extracción de tool_calls de una respuesta completa (complete())
    # ------------------------------------------------------------------

    def _normalize_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                return {"raw": arguments}
        return {"value": arguments}

    def _extract_tool_calls(self, response: Any) -> list[ToolCall]:
        """Extrae tool_calls de una respuesta AIMessage completa."""
        raw_tool_calls = getattr(response, "tool_calls", None)

        if not raw_tool_calls:
            additional_kwargs = getattr(response, "additional_kwargs", None)
            if isinstance(additional_kwargs, dict):
                raw_tool_calls = additional_kwargs.get("tool_calls")

        if not raw_tool_calls:
            return []

        parsed: list[ToolCall] = []

        for raw_call in raw_tool_calls:
            call_id: Any = None
            name: Any = None
            arguments: Any = None

            if isinstance(raw_call, dict):
                function_payload = raw_call.get("function")
                call_id = raw_call.get("id")
                name = raw_call.get("name")
                if not name and isinstance(function_payload, dict):
                    name = function_payload.get("name")
                arguments = raw_call.get("args")
                if arguments is None and isinstance(function_payload, dict):
                    arguments = function_payload.get("arguments")
            else:
                call_id = getattr(raw_call, "id", None)
                name = getattr(raw_call, "name", None)
                arguments = getattr(raw_call, "args", None)

            if not name:
                logger.warning(
                    "Se ignoró un tool_call sin nombre provider=%s raw=%r",
                    self.provider_code,
                    raw_call,
                )
                continue

            try:
                parsed.append(
                    ToolCall(
                        id=str(call_id or f"toolcall_{uuid4().hex}"),
                        name=str(name),
                        arguments=self._normalize_tool_arguments(arguments),
                    )
                )
            except ValueError as exc:
                logger.warning(
                    "Tool call inválido provider=%s raw=%r error=%s",
                    self.provider_code,
                    raw_call,
                    exc,
                )

        return parsed

    # ------------------------------------------------------------------
    # Ensamblado de tool_call_chunks durante streaming
    # ------------------------------------------------------------------

    def _assemble_tool_calls_from_chunks(
        self,
        accumulated: dict[int, dict[str, str]],
    ) -> list[ToolCall]:
        """
        Convierte el dict acumulado de chunks de tool_calls en ToolCall completos.

        accumulated tiene la forma:
            { index: {"id": "...", "name": "...", "args": "..."} }

        donde cada string es la concatenación de todos los chunks recibidos.
        """
        result: list[ToolCall] = []

        for idx in sorted(accumulated.keys()):
            entry = accumulated[idx]
            name = entry.get("name", "").strip()
            if not name:
                logger.warning(
                    "Se ignoró tool_call_chunk sin nombre provider=%s idx=%d",
                    self.provider_code,
                    idx,
                )
                continue

            raw_args = entry.get("args", "").strip()
            arguments = self._normalize_tool_arguments(raw_args or "{}")

            call_id = entry.get("id", "").strip() or f"toolcall_{uuid4().hex}"

            try:
                result.append(
                    ToolCall(
                        id=call_id,
                        name=name,
                        arguments=arguments,
                    )
                )
            except ValueError as exc:
                logger.warning(
                    "Tool call inválido tras ensamblado provider=%s idx=%d error=%s",
                    self.provider_code,
                    idx,
                    exc,
                )

        return result

    def _process_stream_chunk_for_tool_calls(
        self,
        chunk: Any,
        accumulated: dict[int, dict[str, str]],
    ) -> None:
        """
        Lee los tool_call_chunks de un chunk de streaming y los acumula
        en el dict `accumulated`, mutándolo en lugar (para no crear objetos
        intermedios por cada chunk).

        LangChain normaliza los tool_call_chunks en chunk.tool_call_chunks
        para la mayoría de providers. Para providers que no siguen esta
        convención, también intentamos additional_kwargs.tool_calls.
        """
        # --- Ruta principal: chunk.tool_call_chunks (LangChain normalizado) ---
        tool_call_chunks = getattr(chunk, "tool_call_chunks", None) or []

        for tc_chunk in tool_call_chunks:
            if isinstance(tc_chunk, dict):
                idx = tc_chunk.get("index") or 0
                call_id = tc_chunk.get("id") or ""
                name = tc_chunk.get("name") or ""
                args = tc_chunk.get("args") or ""
            else:
                idx = getattr(tc_chunk, "index", 0) or 0
                call_id = getattr(tc_chunk, "id", "") or ""
                name = getattr(tc_chunk, "name", "") or ""
                args = getattr(tc_chunk, "args", "") or ""

            if idx not in accumulated:
                accumulated[idx] = {"id": "", "name": "", "args": ""}

            accumulated[idx]["id"] += call_id
            accumulated[idx]["name"] += name
            accumulated[idx]["args"] += args

        if tool_call_chunks:
            return  # ya procesado

        # --- Ruta de fallback: additional_kwargs.tool_calls (OpenAI legacy) ---
        additional_kwargs = getattr(chunk, "additional_kwargs", None)
        if not isinstance(additional_kwargs, dict):
            return

        raw_calls = additional_kwargs.get("tool_calls") or []
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue

            idx = raw_call.get("index") or 0
            call_id = raw_call.get("id") or ""

            function_payload = raw_call.get("function") or {}
            name = function_payload.get("name") or ""
            args = function_payload.get("arguments") or ""

            if idx not in accumulated:
                accumulated[idx] = {"id": "", "name": "", "args": ""}

            accumulated[idx]["id"] += call_id
            accumulated[idx]["name"] += name
            accumulated[idx]["args"] += args

    # ------------------------------------------------------------------
    # Extracción de texto y uso de tokens
    # ------------------------------------------------------------------

    def _extract_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    def _extract_usage(self, response: Any) -> TokenUsage | None:
        prompt: int | None = None
        completion: int | None = None
        total: int | None = None

        usage_metadata = getattr(response, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            prompt = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens")
            completion = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens")
            total = usage_metadata.get("total_tokens")
        else:
            response_metadata = getattr(response, "response_metadata", None)
            if isinstance(response_metadata, dict):
                token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
                if isinstance(token_usage, dict):
                    prompt = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
                    completion = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
                    total = token_usage.get("total_tokens")

        if prompt is None and completion is None:
            return None

        if total is None:
            total = (prompt or 0) + (completion or 0)

        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )

    # ------------------------------------------------------------------
    # Normalización de excepciones externas
    # ------------------------------------------------------------------

    def _normalize_exception(self, exc: Exception) -> Exception:
        try:
            import openai
            if isinstance(exc, openai.APITimeoutError):
                return ProviderTimeoutError(f"Timeout invocando proveedor '{self.provider_code}'")
            if isinstance(exc, openai.RateLimitError):
                return ProviderRateLimitError(f"Rate limit en proveedor '{self.provider_code}'")
            if isinstance(exc, openai.AuthenticationError):
                return ProviderAuthenticationError(f"Auth error en proveedor '{self.provider_code}'")
            if isinstance(exc, openai.APIStatusError) and exc.status_code == 503:
                return ProviderUnavailableError(f"Proveedor '{self.provider_code}' no disponible")
        except ImportError:
            pass

        try:
            import anthropic
            if isinstance(exc, anthropic.APITimeoutError):
                return ProviderTimeoutError(f"Timeout invocando proveedor '{self.provider_code}'")
            if isinstance(exc, anthropic.RateLimitError):
                return ProviderRateLimitError(f"Rate limit en proveedor '{self.provider_code}'")
            if isinstance(exc, anthropic.AuthenticationError):
                return ProviderAuthenticationError(f"Auth error en proveedor '{self.provider_code}'")
            if isinstance(exc, anthropic.APIStatusError) and exc.status_code == 503:
                return ProviderUnavailableError(f"Proveedor '{self.provider_code}' no disponible")
        except ImportError:
            pass

        try:
            from google.api_core import exceptions as google_exc
            if isinstance(exc, google_exc.DeadlineExceeded):
                return ProviderTimeoutError(f"Timeout invocando proveedor '{self.provider_code}'")
            if isinstance(exc, google_exc.ResourceExhausted):
                return ProviderRateLimitError(f"Rate limit en proveedor '{self.provider_code}'")
            if isinstance(exc, google_exc.Unauthenticated):
                return ProviderAuthenticationError(f"Auth error en proveedor '{self.provider_code}'")
            if isinstance(exc, google_exc.ServiceUnavailable):
                return ProviderUnavailableError(f"Proveedor '{self.provider_code}' no disponible")
        except ImportError:
            pass

        # Fallback por string matching
        name = exc.__class__.__name__.lower()
        message = str(exc).lower()

        if "timeout" in name or "timeout" in message:
            return ProviderTimeoutError(f"Timeout invocando proveedor '{self.provider_code}'")
        if "rate" in name or "429" in message:
            return ProviderRateLimitError(f"Rate limit en proveedor '{self.provider_code}'")
        if "auth" in name or "401" in message or "api key" in message or "permission" in message:
            return ProviderAuthenticationError(f"Auth error en proveedor '{self.provider_code}'")
        if "503" in message or "unavailable" in message:
            return ProviderUnavailableError(f"Proveedor '{self.provider_code}' no disponible")

        return ProviderError(f"Error invocando proveedor '{self.provider_code}'")
    
    # ------------------------------------------------------------------
    # Validar con el catalogo
    # ------------------------------------------------------------------
    def _iter_request_params(self, request: ChatRequest) -> dict[str, Any]:
        params = dict(request.config.to_param_dict())

        if request.tool_choice is not None:
            params["tool_choice"] = request.tool_choice

        return params

    def _validate_request_config_against_catalog(self, request: ChatRequest) -> None:
        for param_name, value in self._iter_request_params(request).items():
            self._model_catalog.validate_param_value(
                request.provider_code,
                request.model_key,
                param_name,
                value,
            )

    def _validate_request_against_catalog(
        self,
        request: ChatRequest,
        *,
        streaming: bool,
    ) -> None:
        self._model_catalog.require_model(request.provider_code, request.model_key)
        self._model_catalog.require_capability(
            request.provider_code,
            request.model_key,
            "chat",
        )

        if request.tools:
            self._model_catalog.require_capability(
                request.provider_code,
                request.model_key,
                "tools",
            )

        if streaming:
            self._model_catalog.require_capability(
                request.provider_code,
                request.model_key,
                "streaming",
            )

        self._validate_request_config_against_catalog(request)

    # ------------------------------------------------------------------
    # complete() — invocación sin streaming
    # ------------------------------------------------------------------

    async def complete(self, request: ChatRequest) -> ChatCompletionResult:
        self._validate_request_against_catalog(request, streaming=False)

        model = self._prepare_model(request)
        messages = self._prepare_messages(request)

        logger.debug(
            "complete() provider=%s model=%s mensajes=%d tools=%d",
            self.provider_code,
            request.model_key,
            len(messages),
            len(request.tools),
        )

        try:
            response = await model.ainvoke(messages)
            response_metadata = getattr(response, "response_metadata", None)
            additional_kwargs = getattr(response, "additional_kwargs", None)

            usage = self._extract_usage(response)
            tool_calls = self._extract_tool_calls(response)

            raw_response: dict[str, Any] | None = None
            if isinstance(response_metadata, dict) or isinstance(additional_kwargs, dict):
                raw_response = {}
                if isinstance(response_metadata, dict):
                    raw_response["response_metadata"] = response_metadata
                if isinstance(additional_kwargs, dict):
                    raw_response["additional_kwargs"] = additional_kwargs

            logger.debug(
                "complete() OK provider=%s model=%s tokens=%s tool_calls=%d",
                self.provider_code,
                request.model_key,
                usage.total_tokens if usage else "n/a",
                len(tool_calls),
            )

            return ChatCompletionResult(
                content=self._extract_text(getattr(response, "content", None)),
                finish_reason=(
                    response_metadata.get("finish_reason")
                    if isinstance(response_metadata, dict)
                    else None
                ),
                usage=usage,
                provider_request_id=(
                    response_metadata.get("request_id")
                    if isinstance(response_metadata, dict)
                    else None
                ),
                raw_response=raw_response,
                tool_calls=tool_calls,
            )
        except AppError:
            raise
        except Exception as exc:
            logger.error(
                "complete() ERROR provider=%s model=%s exc=%s: %s",
                self.provider_code,
                request.model_key,
                exc.__class__.__name__,
                exc,
            )
            raise self._normalize_exception(exc) from exc

    # ------------------------------------------------------------------
    # stream() — streaming con soporte completo de tool calling
    # ------------------------------------------------------------------

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatStreamEvent, None]:
        """
        Streaming con soporte completo de tool calling.

        Protocolo de eventos emitidos:
          START  → indica inicio de turno del asistente
          DELTA  → fragmento de texto (puede haber 0 DELTAs si el turno es puro tool_use)
          TOOL_USE → el modelo solicitó tools; tool_calls contiene los calls ensamblados
          END    → fin del turno (siempre el último evento)
          ERROR  → error de proveedor (el stream se detiene)

        El consumidor (ChatService.stream_message) maneja el evento
        TOOL_USE ejecutando las herramientas y relanzando el stream en un loop.

        Nota: _prepare_model y _prepare_messages se ejecutan ANTES del try/except
        para que errores de configuración (API key, binding de tools, roles
        inválidos) se propaguen como excepciones normales y no como eventos ERROR.
        """
        self._validate_request_against_catalog(request, streaming=True)

        model = self._prepare_model(request)
        messages = self._prepare_messages(request)

        logger.debug(
            "stream() START provider=%s model=%s mensajes=%d tools=%d",
            self.provider_code,
            request.model_key,
            len(messages),
            len(request.tools),
        )

        yield ChatStreamEvent(type=ChatStreamEventType.START)

        # Dict para acumular los tool_call_chunks fragmentados.
        # Clave: índice del tool_call (int)
        # Valor: {"id": str, "name": str, "args": str}
        accumulated_tool_calls: dict[int, dict[str, str]] = {}

        try:
            async for chunk in model.astream(messages):
                # 1. Acumular tool_call_chunks (si los hay)
                self._process_stream_chunk_for_tool_calls(chunk, accumulated_tool_calls)

                # 2. Emitir delta de texto si existe
                delta = self._extract_text(getattr(chunk, "content", None))
                if delta:
                    yield ChatStreamEvent(type=ChatStreamEventType.DELTA, delta=delta)

            # ---- Fin del stream crudo del provider ----

            if accumulated_tool_calls:
                # El modelo solicitó herramientas. Ensamblar y emitir TOOL_USE.
                tool_calls = self._assemble_tool_calls_from_chunks(accumulated_tool_calls)

                if tool_calls:
                    logger.debug(
                        "stream() TOOL_USE provider=%s model=%s calls=%d",
                        self.provider_code,
                        request.model_key,
                        len(tool_calls),
                    )
                    yield ChatStreamEvent(
                        type=ChatStreamEventType.TOOL_USE,
                        tool_calls=tool_calls,
                    )
                else:
                    # Los chunks llegaron pero no pudieron ensamblarse
                    logger.warning(
                        "stream() tool_call_chunks recibidos pero no ensamblados "
                        "provider=%s model=%s",
                        self.provider_code,
                        request.model_key,
                    )

            logger.debug(
                "stream() END provider=%s model=%s",
                self.provider_code,
                request.model_key,
            )
            yield ChatStreamEvent(type=ChatStreamEventType.END)

        except AppError as exc:
            logger.error(
                "stream() ERROR provider=%s model=%s error_code=%s",
                self.provider_code,
                request.model_key,
                exc.error_code,
            )
            yield ChatStreamEvent(
                type=ChatStreamEventType.ERROR,
                error_code=exc.error_code,
                error_message=exc.message,
            )
        except Exception as exc:
            normalized = self._normalize_exception(exc)
            logger.error(
                "stream() ERROR provider=%s model=%s exc=%s: %s",
                self.provider_code,
                request.model_key,
                exc.__class__.__name__,
                exc,
            )
            yield ChatStreamEvent(
                type=ChatStreamEventType.ERROR,
                error_code=getattr(normalized, "error_code", "provider_error"),
                error_message=str(normalized),
            )