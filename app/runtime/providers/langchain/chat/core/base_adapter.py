# app/runtime/providers/langchain/chat/core/base_adapter.py
from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, ClassVar, TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.shared.exceptions import (
    AppError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderAuthenticationError,
    ProviderUnavailableError,
)
from app.capabilities.chat.contracts import (
    ChatCompletionResult,
    ChatRequest,
    ChatStreamEvent,
    ChatStreamEventType,
)
from app.capabilities.chat.contracts import ChatMessage
from app.files.entities import FileAttachmentRef
from app.runtime.providers.base import ChatProviderAdapter
from app.model_catalog.service import ModelCatalogService

from app.runtime.providers.langchain.chat.core.meta import ProviderAdapterMeta
from app.runtime.providers.langchain.chat.core.messages import to_langchain_messages
from app.runtime.providers.langchain.chat.core.tool_utils import (
    to_langchain_tool_schema,
    extract_tool_calls,
    process_stream_chunk_for_tool_calls,
    assemble_tool_calls_from_chunks,
)
from app.runtime.providers.langchain.chat.core.extraction import extract_text, extract_usage
from app.runtime.providers.langchain.chat.core.validation import validate_request_against_catalog

if TYPE_CHECKING:
    from app.runtime.providers.langchain.factory import LangChainProviderFactory

logger = logging.getLogger(__name__)


class BaseLangChainChatAdapter(ChatProviderAdapter, ABC):

    provider_code: ClassVar[str]
    adapter_meta: ClassVar[ProviderAdapterMeta]

    def __init__(
        self,
        factory: "LangChainProviderFactory",
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
    # Normalización de excepciones — fallback genérico por string matching.
    # Cada adapter concreto puede sobreescribir este método para añadir
    # el manejo específico de las excepciones de su SDK.
    # ------------------------------------------------------------------

    def _normalize_exception(self, exc: Exception) -> Exception:
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

        tool_schemas = [
            to_langchain_tool_schema(t, provider_code=self.provider_code)
            for t in request.tools
        ]

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

    def _prepare_messages(self, request: ChatRequest) -> list:
        """
        Prepara los mensajes para LangChain.
        Ahora soporta attachments de forma multimodal y lee 'vision_input'
        directamente desde el catálogo YAML (nada hardcodeado).
        """
        supports_vision = self._get_supports_vision(request)
        return to_langchain_messages(
            request.messages,
            supports_vision=supports_vision
        )

    def _get_supports_vision(self, request: ChatRequest) -> bool:
        """
        Lee dinámicamente desde el catálogo de modelos (catalog/models/*.yaml)
        si el modelo actual soporta vision_input.
        
        Esto hace que NO tengas que hardcodear nada en los adapters.
        Si mañana agregas un nuevo modelo con vision_input: true en el YAML,
        automáticamente lo usará.
        """
        # La mayoría de adapters tienen acceso al catálogo a través del factory
        if hasattr(self, "_model_catalog") and self._model_catalog is not None:
            return self._model_catalog.supports_capability(
                request.provider_code,
                request.model_key,
                "vision_input"
            )

        # Fallback seguro (por si en algún adapter no está inyectado aún)
        return False

    # ------------------------------------------------------------------
    # complete() — invocación sin streaming
    # ------------------------------------------------------------------

    async def complete(self, request: ChatRequest) -> ChatCompletionResult:
        validate_request_against_catalog(request, self._model_catalog, streaming=False)

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

            usage = extract_usage(response)
            tool_calls = extract_tool_calls(response, provider_code=self.provider_code)

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
                content=extract_text(getattr(response, "content", None)),
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
        validate_request_against_catalog(request, self._model_catalog, streaming=True)

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

        accumulated_tool_calls: dict[int, dict[str, str]] = {}

        try:
            async for chunk in model.astream(messages):
                process_stream_chunk_for_tool_calls(chunk, accumulated_tool_calls)

                delta = extract_text(getattr(chunk, "content", None))
                if delta:
                    yield ChatStreamEvent(type=ChatStreamEventType.DELTA, delta=delta)

            if accumulated_tool_calls:
                tool_calls = assemble_tool_calls_from_chunks(
                    accumulated_tool_calls,
                    provider_code=self.provider_code,
                )

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
    
    # ------------------------------------------------------------------
    # File related helpers
    # ------------------------------------------------------------------
    def _model_supports_vision(self) -> bool:
        """
        Indica si este adapter concreto soporta entradas de imagen de forma nativa.
        Los adapters de Anthropic, OpenAI y Google lo sobreescriben.
        """
        return False

    def _build_human_message_with_attachments(self, message: ChatMessage) -> HumanMessage:
        """
        Construye HumanMessage multimodal (content blocks) cuando hay attachments.
        Estrategia híbrida (exactamente como en el manual original):
          - Texto del usuario siempre va primero.
          - Imágenes → bloque nativo si el modelo lo soporta.
          - Cualquier archivo → texto extraído o placeholder.
        """
        supports_vision = self._model_supports_vision()
        content_blocks: list[dict[str, Any]] = []

        # Texto del usuario
        if message.content:
            content_blocks.append({"type": "text", "text": message.content})

        for attachment in message.attachments:
            is_image = attachment.mime_type.startswith("image/")

            if is_image and supports_vision:
                # En fase futura podrás cargar los bytes reales y usar base64.
                # Por ahora usamos el resolved_text (ya extraído en FileService).
                if attachment.resolved_text:
                    content_blocks.append({
                        "type": "text",
                        "text": f"\n\n[Imagen adjunta: {attachment.filename}]\n{attachment.resolved_text}",
                    })
                else:
                    content_blocks.append({
                        "type": "text",
                        "text": f"\n\n[Imagen adjunta: {attachment.filename} ({attachment.mime_type})]",
                    })

            elif attachment.resolved_text:
                # Archivo de texto/PDF/etc. → lo insertamos como bloque de texto
                content_blocks.append({
                    "type": "text",
                    "text": (
                        f"\n\n--- Contenido de {attachment.filename} ---\n"
                        f"{attachment.resolved_text}\n"
                        "--- Fin del archivo ---"
                    ),
                })
            else:
                # Sin texto extraído
                content_blocks.append({
                    "type": "text",
                    "text": (
                        f"\n\n[Archivo adjunto: {attachment.filename} "
                        f"({attachment.mime_type}) — sin contenido extraído]"
                    ),
                })

        # Si solo hay un bloque de texto, LangChain lo acepta como string simple (más compatible)
        if len(content_blocks) == 1 and content_blocks[0]["type"] == "text":
            return HumanMessage(content=content_blocks[0]["text"])

        return HumanMessage(content=content_blocks)