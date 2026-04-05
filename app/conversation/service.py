from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID

from app.conversation.interfaces import ContextBuilder, ConversationRepository
from app.conversation.models import ConversationSession
from app.core.exceptions import (
    InvalidProviderSelectionError,
    ToolLoopLimitExceededError,
)
from app.llm.contracts import (
    LLMCompletionResult,
    LLMRequest,
    LLMRequestConfig,
    LLMStreamEvent,
    StreamEventType,
)
from app.llm.orchestrator import LLMOrchestrator
from app.llm.registry import ProviderRegistry
from app.tools.contracts import ToolDefinition, ToolExecutionContext
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


class ConversationService:
    """
    Servicio conversacional multiproveedor.

    En esta fase 1:
    - send_message() soporta tool calling completo
    - stream_message() NO soporta tool calling todavía
    """

    def __init__(
        self,
        *,
        repository: ConversationRepository,
        context_builder: ContextBuilder,
        orchestrator: LLMOrchestrator,
        provider_registry: ProviderRegistry,
        tool_registry: ToolRegistry | None = None,
        tool_executor: ToolExecutor | None = None,
        default_config: LLMRequestConfig | None = None,
        max_tool_iterations: int = 8,
    ) -> None:
        resolved_tool_registry = tool_registry or ToolRegistry()
        resolved_tool_executor = tool_executor or ToolExecutor(resolved_tool_registry)

        self._repository = repository
        self._context_builder = context_builder
        self._orchestrator = orchestrator
        self._provider_registry = provider_registry
        self._tool_registry = resolved_tool_registry
        self._tool_executor = resolved_tool_executor
        self._default_config = default_config or LLMRequestConfig()
        self._max_tool_iterations = max_tool_iterations

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    async def create_conversation(
        self,
        *,
        provider_code: str,
        model_key: str,
        system_prompt: str | None = None,
    ) -> ConversationSession:
        if not self._provider_registry.has(provider_code):
            raise InvalidProviderSelectionError(
                f"No hay adapter registrado para provider '{provider_code}'"
            )

        return await self._repository.create(
            provider_code=provider_code,
            model_key=model_key,
            system_prompt=system_prompt,
        )

    async def get_conversation(self, conversation_id: UUID) -> ConversationSession:
        return await self._repository.get(conversation_id)

    def available_tool_names(self) -> list[str]:
        return self._tool_registry.available_tool_names()

    async def change_model(
        self,
        *,
        conversation_id: UUID,
        provider_code: str | None = None,
        model_key: str | None = None,
    ) -> ConversationSession:
        if provider_code is not None and model_key is None:
            raise InvalidProviderSelectionError(
                "Si cambias provider_code, debes indicar también model_key"
            )

        if provider_code is not None and not self._provider_registry.has(provider_code):
            raise InvalidProviderSelectionError(
                f"No hay adapter registrado para provider '{provider_code}'"
            )

        session = await self._repository.get(conversation_id)
        session.switch_model(provider_code=provider_code, model_key=model_key)
        await self._repository.save(session)
        return session

    async def send_message(
        self,
        *,
        conversation_id: UUID,
        user_text: str,
        provider_code: str | None = None,
        model_key: str | None = None,
        tool_names: list[str] | None = None,
    ) -> LLMCompletionResult:
        tool_definitions = self._resolve_tool_definitions(tool_names)

        session = await self._prepare_session(
            conversation_id=conversation_id,
            user_text=user_text,
            provider_code=provider_code,
            model_key=model_key,
        )

        for iteration in range(self._max_tool_iterations + 1):
            request = self._build_request(
                session=session,
                tool_definitions=tool_definitions,
            )

            result = await self._orchestrator.complete(request)

            session.add_assistant_message(
                result.content,
                provider_code=session.current_provider_code,
                model_key=session.current_model_key,
                tool_calls=result.tool_calls,
            )
            await self._repository.save(session)

            if not result.tool_calls:
                return result

            if iteration >= self._max_tool_iterations:
                raise ToolLoopLimitExceededError(self._max_tool_iterations)

            execution_context = self._build_tool_execution_context(session)

            for tool_call in result.tool_calls:
                tool_result = await self._tool_executor.execute_call(
                    tool_call,
                    execution_context,
                )
                session.add_tool_message(
                    tool_result.content,
                    tool_call_id=tool_result.tool_call_id,
                    name=tool_result.name,
                )

            await self._repository.save(session)

        raise ToolLoopLimitExceededError(self._max_tool_iterations)

    async def stream_message(
        self,
        *,
        conversation_id: UUID,
        user_text: str,
        provider_code: str | None = None,
        model_key: str | None = None,
        tool_names: list[str] | None = None,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        if tool_names:
            raise NotImplementedError(
                "Tool calling no está soportado en stream_message() durante la fase 1"
            )

        session = await self._prepare_session(
            conversation_id=conversation_id,
            user_text=user_text,
            provider_code=provider_code,
            model_key=model_key,
        )

        request = self._build_request(
            session=session,
            tool_definitions=[],
        )

        async def generator() -> AsyncGenerator[LLMStreamEvent, None]:
            chunks: list[str] = []

            async for event in self._orchestrator.stream(request):
                if event.type == StreamEventType.DELTA and event.delta:
                    chunks.append(event.delta)
                    yield event
                    continue

                if event.type == StreamEventType.ERROR:
                    yield event
                    return

                if event.type == StreamEventType.END:
                    final_text = "".join(chunks)
                    if final_text:
                        session.add_assistant_message(
                            final_text,
                            provider_code=session.current_provider_code,
                            model_key=session.current_model_key,
                        )
                        await self._repository.save(session)
                    yield event
                    return

                yield event

        return generator()

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    async def _prepare_session(
        self,
        *,
        conversation_id: UUID,
        user_text: str,
        provider_code: str | None,
        model_key: str | None,
    ) -> ConversationSession:
        if provider_code is not None and model_key is None:
            raise InvalidProviderSelectionError(
                "Si cambias provider_code, debes indicar también model_key"
            )

        session = await self._repository.get(conversation_id)

        if provider_code is not None:
            if not self._provider_registry.has(provider_code):
                raise InvalidProviderSelectionError(
                    f"No hay adapter registrado para provider '{provider_code}'"
                )
            session.switch_model(provider_code=provider_code, model_key=model_key)

        elif model_key is not None:
            session.switch_model(model_key=model_key)

        session.add_user_message(user_text)
        await self._repository.save(session)

        return session

    def _resolve_tool_definitions(
        self,
        tool_names: list[str] | None,
    ) -> list[ToolDefinition]:
        if tool_names is not None:
            return self._tool_registry.list_definitions(tool_names)

        return self._tool_registry.list_definitions()

    def _build_tool_execution_context(
        self,
        session: ConversationSession,
    ) -> ToolExecutionContext:
        return ToolExecutionContext(
            conversation_id=str(session.id),
            provider_code=session.current_provider_code,
            model_key=session.current_model_key,
            metadata={"conversation_id": str(session.id)},
        )

    def _build_request(
        self,
        *,
        session: ConversationSession,
        tool_definitions: list[ToolDefinition] | None = None,
    ) -> LLMRequest:
        resolved_tools = list(tool_definitions or [])

        metadata: dict[str, Any] = {
            "conversation_id": str(session.id),
        }

        if resolved_tools:
            metadata["tool_names"] = [tool.name for tool in resolved_tools]

        return LLMRequest(
            provider_code=session.current_provider_code,
            model_key=session.current_model_key,
            messages=self._context_builder.build(session),
            config=self._default_config,
            metadata=metadata,
            tools=resolved_tools,
        )