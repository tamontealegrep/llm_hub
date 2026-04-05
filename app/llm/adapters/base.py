import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, ClassVar
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.exceptions import (
    AppError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.llm.contracts import (
    LLMCompletionResult,
    LLMRequest,
    LLMStreamEvent,
    StreamEventType,
    TokenUsage,
)
from app.llm.factory import LangChainChatModelFactory
from app.llm.interfaces import LLMProviderAdapter
from app.tools.contracts import ToolCall, ToolDefinition

logger = logging.getLogger(__name__)


class BaseProviderAdapter(LLMProviderAdapter, ABC):
    """
    Clase base reutilizable para todos los adapters.

    Centraliza:
    - conversión de NormalizedMessage a mensajes LangChain
      (incluyendo AIMessage con tool_calls y ToolMessage)
    - binding de herramientas al modelo
    - extracción de texto, uso de tokens y tool_calls
    - normalización de excepciones
    - logging estructurado de cada invocación
    - implementación genérica de complete() y stream()
    """

    provider_code: ClassVar[str]

    def __init__(self, factory: LangChainChatModelFactory) -> None:
        self._factory = factory

    @abstractmethod
    def _build_chat_model(self, request: LLMRequest) -> BaseChatModel:
        raise NotImplementedError

    def _prepare_model(self, request: LLMRequest) -> Any:
        model = self._build_chat_model(request)

        if not request.tools:
            return model

        bind_tools = getattr(model, "bind_tools", None)
        if bind_tools is None:
            raise ProviderError(
                f"El proveedor '{self.provider_code}' no soporta tool calling en este adapter"
            )

        tool_schemas = [self._to_langchain_tool_schema(tool) for tool in request.tools]

        try:
            if request.tool_choice is not None:
                return bind_tools(tool_schemas, tool_choice=request.tool_choice)
            return bind_tools(tool_schemas)
        except TypeError:
            if request.tool_choice is not None:
                return bind_tools(tool_schemas)
            raise
        except NotImplementedError as exc:
            raise ProviderError(
                f"El proveedor '{self.provider_code}' no soporta tool calling"
            ) from exc

    def _prepare_messages(self, request: LLMRequest) -> list:
        return self._to_langchain_messages(request.messages)

    def _to_langchain_tool_schema(self, tool: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    def _to_langchain_messages(self, messages: list) -> list:
        """
        Convierte NormalizedMessage a mensajes LangChain.
        Soporta roles: system, user, assistant, tool.
        Para assistant, también soporta tool_calls.
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
                                    "id": tool_call.id,
                                    "name": tool_call.name,
                                    "args": tool_call.arguments,
                                    "type": "tool_call",
                                }
                                for tool_call in message.tool_calls
                            ],
                        )
                    )
                else:
                    converted.append(AIMessage(content=message.content))

            elif message.role == "tool":
                if not message.tool_call_id:
                    raise ValueError(
                        "NormalizedMessage con role='tool' requiere tool_call_id. "
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

    def _normalize_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        if arguments is None:
            return {}

        if isinstance(arguments, dict):
            return arguments

        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict):
                    return parsed
                return {"value": parsed}
            except json.JSONDecodeError:
                return {"raw": arguments}

        return {"value": arguments}

    def _extract_tool_calls(self, response: Any) -> list[ToolCall]:
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
        """
        Extrae uso de tokens desde usage_metadata o response_metadata.
        Calcula total_tokens como fallback cuando el proveedor no lo devuelve.
        """
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

    def _normalize_exception(self, exc: Exception) -> Exception:
        """
        Convierte excepciones externas en excepciones de dominio.

        Estrategia en dos capas:
        1. isinstance() por tipo concreto de cada librería.
        2. fallback por string matching.
        """
        try:
            import openai

            if isinstance(exc, openai.APITimeoutError):
                return ProviderTimeoutError(f"Timeout invocando proveedor '{self.provider_code}'")
            if isinstance(exc, openai.RateLimitError):
                return ProviderRateLimitError(f"Rate limit en proveedor '{self.provider_code}'")
            if isinstance(exc, openai.AuthenticationError):
                return ProviderAuthenticationError(
                    f"Error de autenticación en proveedor '{self.provider_code}'"
                )
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
                return ProviderAuthenticationError(
                    f"Error de autenticación en proveedor '{self.provider_code}'"
                )
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
                return ProviderAuthenticationError(
                    f"Error de autenticación en proveedor '{self.provider_code}'"
                )
            if isinstance(exc, google_exc.ServiceUnavailable):
                return ProviderUnavailableError(f"Proveedor '{self.provider_code}' no disponible")
        except ImportError:
            pass

        name = exc.__class__.__name__.lower()
        message = str(exc).lower()

        if "timeout" in name or "timeout" in message:
            return ProviderTimeoutError(f"Timeout invocando proveedor '{self.provider_code}'")
        if "rate" in name or "429" in message:
            return ProviderRateLimitError(f"Rate limit en proveedor '{self.provider_code}'")
        if "auth" in name or "401" in message or "api key" in message or "permission" in message:
            return ProviderAuthenticationError(
                f"Error de autenticación en proveedor '{self.provider_code}'"
            )
        if "503" in message or "unavailable" in message:
            return ProviderUnavailableError(f"Proveedor '{self.provider_code}' no disponible")

        return ProviderError(f"Error invocando proveedor '{self.provider_code}'")

    async def complete(self, request: LLMRequest) -> LLMCompletionResult:
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

            return LLMCompletionResult(
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

    async def stream(self, request: LLMRequest) -> AsyncGenerator[LLMStreamEvent, None]:
        """
        _prepare_model y _prepare_messages se ejecutan ANTES del try/except
        para que errores de configuración, roles inválidos o tool binding
        se propaguen como excepciones normales y no como eventos ERROR del stream.

        En esta fase 1, stream() no implementa tool calling completo.
        """
        model = self._prepare_model(request)
        messages = self._prepare_messages(request)

        logger.debug(
            "stream() START provider=%s model=%s mensajes=%d tools=%d",
            self.provider_code,
            request.model_key,
            len(messages),
            len(request.tools),
        )

        yield LLMStreamEvent(type=StreamEventType.START)

        try:
            async for chunk in model.astream(messages):
                delta = self._extract_text(getattr(chunk, "content", None))
                if delta:
                    yield LLMStreamEvent(type=StreamEventType.DELTA, delta=delta)

            logger.debug(
                "stream() END provider=%s model=%s",
                self.provider_code,
                request.model_key,
            )
            yield LLMStreamEvent(type=StreamEventType.END)

        except AppError as exc:
            logger.error(
                "stream() ERROR provider=%s model=%s error_code=%s",
                self.provider_code,
                request.model_key,
                exc.error_code,
            )
            yield LLMStreamEvent(
                type=StreamEventType.ERROR,
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
            yield LLMStreamEvent(
                type=StreamEventType.ERROR,
                error_code=getattr(normalized, "error_code", "provider_error"),
                error_message=str(normalized),
            )