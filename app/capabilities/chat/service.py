from collections.abc import AsyncGenerator
from typing import Any
from uuid import UUID

from app.conversations.interfaces import ConversationContextBuilder, ConversationRepository
from app.conversations.entities import ConversationSession
from app.shared.exceptions import (
    InvalidProviderSelectionError,
    ToolLoopLimitExceededError,
)
from app.capabilities.chat.contracts import (
    ChatCompletionResult,
    ChatRequest,
    ChatRequestConfig,
    ChatStreamEvent,
    ChatStreamEventType,
)
from app.runtime.execution.chat_orchestrator import ChatOrchestrator
from app.runtime.providers.registry import ProviderRegistry
from app.tools.contracts import ToolDefinition, ToolExecutionContext
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.model_catalog.service import ModelCatalogService


class ChatService:

    def __init__(
        self,
        *,
        repository: ConversationRepository,
        context_builder: ConversationContextBuilder,
        orchestrator: ChatOrchestrator,
        provider_registry: ProviderRegistry,
        model_catalog: ModelCatalogService,
        tool_registry: ToolRegistry | None = None,
        tool_executor: ToolExecutor | None = None,
        default_config: ChatRequestConfig | None = None,
        max_tool_iterations: int = 8,
    ) -> None:
        resolved_tool_registry = tool_registry or ToolRegistry()
        resolved_tool_executor = tool_executor or ToolExecutor(resolved_tool_registry)

        self._repository = repository
        self._context_builder = context_builder
        self._orchestrator = orchestrator
        self._provider_registry = provider_registry
        self._model_catalog = model_catalog
        self._tool_registry = resolved_tool_registry
        self._tool_executor = resolved_tool_executor
        self._default_config = default_config or ChatRequestConfig()
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
        self._ensure_provider_available(provider_code)
        self._validate_chat_target(provider_code, model_key)

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
        session = await self._repository.get(conversation_id)

        target_provider_code, target_model_key = self._resolve_effective_target(
            session=session,
            provider_code=provider_code,
            model_key=model_key,
        )

        self._ensure_provider_available(target_provider_code)
        self._validate_chat_target(target_provider_code, target_model_key)

        session.switch_model(
            provider_code=target_provider_code,
            model_key=target_model_key,
        )
        await self._repository.save(session)
        return session

    # ------------------------------------------------------------------
    # send_message() — modo no-streaming con tool loop completo
    # ------------------------------------------------------------------

    async def send_message(
        self,
        *,
        conversation_id: UUID,
        user_text: str,
        provider_code: str | None = None,
        model_key: str | None = None,
        tool_names: list[str] | None = None,
    ) -> ChatCompletionResult:
        """
        Envía un mensaje y espera la respuesta completa.

        tool_names:
          - None    → usa todas las tools registradas (si hay alguna)
          - []      → no envía tools al modelo
          - [names] → solo esas tools
        """
        tool_definitions = self._resolve_tool_definitions(tool_names)

        session, target_provider_code, target_model_key = (
            await self._load_validated_session_target(
                conversation_id=conversation_id,
                provider_code=provider_code,
                model_key=model_key,
                tools_enabled=bool(tool_definitions),
                streaming=False,
            )
        )

        session = await self._prepare_session(
            session=session,
            user_text=user_text,
            target_provider_code=target_provider_code,
            target_model_key=target_model_key,
        )

        for iteration in range(self._max_tool_iterations + 1):
            request = self._build_request(session=session, tool_definitions=tool_definitions)
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
                tool_result = await self._tool_executor.execute_call(tool_call, execution_context)
                session.add_tool_message(
                    tool_result.content,
                    tool_call_id=tool_result.tool_call_id,
                    name=tool_result.name,
                )

            await self._repository.save(session)

        raise ToolLoopLimitExceededError(self._max_tool_iterations)

    # ------------------------------------------------------------------
    # stream_message() — modo streaming con tool loop completo
    # ------------------------------------------------------------------

    async def stream_message(
        self,
        *,
        conversation_id: UUID,
        user_text: str,
        provider_code: str | None = None,
        model_key: str | None = None,
        tool_names: list[str] | None = None,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        """
        Envía un mensaje en modo streaming con tool calling completo.

        Protocolo de eventos que el consumidor externo recibe:

          START        → inicio del primer turno del asistente
          DELTA        → fragmento de texto (0..N por turno)
          TOOL_NOTIFY  → notificación informativa de que el modelo llamó a tools
                         (evento especial emitido por el servicio, no por el adapter)
          START        → inicio de turno siguiente (si hubo tools y el modelo
                         continúa respondiendo)
          DELTA        → más texto
          END          → fin definitivo

        El consumidor NO necesita manejar el loop: el servicio lo gestiona
        internamente. Recibe un stream unificado independientemente de cuántas
        iteraciones de tool calling hayan ocurrido.

        tool_names:
          - None    → usa todas las tools registradas (si hay alguna)
          - []      → no envía tools al modelo
          - [names] → solo esas tools
        """
        tool_definitions = self._resolve_tool_definitions(tool_names)

        session, target_provider_code, target_model_key = (
            await self._load_validated_session_target(
                conversation_id=conversation_id,
                provider_code=provider_code,
                model_key=model_key,
                tools_enabled=bool(tool_definitions),
                streaming=True,
            )
        )

        session = await self._prepare_session(
            session=session,
            user_text=user_text,
            target_provider_code=target_provider_code,
            target_model_key=target_model_key,
        )

        return self._stream_generator(
            session=session,
            tool_definitions=tool_definitions,
        )

    # ------------------------------------------------------------------
    # Generador interno del stream (gestiona el loop de tool calling)
    # ------------------------------------------------------------------

    async def _stream_generator(
        self,
        *,
        session: ConversationSession,
        tool_definitions: list[ToolDefinition],
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        """
        Generador que maneja el loop completo de streaming + tool calling.

        Por cada iteración:
          1. Construye el request con el historial actualizado.
          2. Abre el stream del adapter.
          3. Propaga los DELTA al consumidor.
          4. Si recibe TOOL_USE:
             a. Guarda el mensaje assistant con los tool_calls en la sesión.
             b. Ejecuta las tools y guarda los resultados.
             c. Emite un evento de notificación al consumidor (TOOL_NOTIFY).
             d. Inicia la siguiente iteración (loop).
          5. Al recibir END sin tool_calls previos, guarda el mensaje assistant
             con el texto acumulado y termina.
        """
        for iteration in range(self._max_tool_iterations + 1):
            request = self._build_request(session=session, tool_definitions=tool_definitions)

            accumulated_text: list[str] = []
            tool_calls_this_turn: list = []
            got_tool_use = False
            got_error = False

            async for event in self._orchestrator.stream(request):

                if event.type == ChatStreamEventType.START:
                    # Propagar siempre para que el consumidor sepa que
                    # hay un nuevo turno del asistente (útil en UIs).
                    yield event

                elif event.type == ChatStreamEventType.DELTA:
                    if event.delta:
                        accumulated_text.append(event.delta)
                    yield event

                elif event.type == ChatStreamEventType.TOOL_USE:
                    # El modelo solicitó tools. Guardar los tool_calls
                    # para procesarlos al finalizar el turno.
                    tool_calls_this_turn = event.tool_calls
                    got_tool_use = True
                    # No re-emitimos TOOL_USE al consumidor externo;
                    # en su lugar emitiremos un TOOL_NOTIFY más informativo
                    # una vez que hayamos ejecutado las tools.

                elif event.type == ChatStreamEventType.END:
                    final_text = "".join(accumulated_text)

                    if got_tool_use and tool_calls_this_turn:
                        # ----- Turno con tool calling -----
                        # 1. Persistir el mensaje assistant con tool_calls
                        session.add_assistant_message(
                            final_text,
                            provider_code=session.current_provider_code,
                            model_key=session.current_model_key,
                            tool_calls=tool_calls_this_turn,
                        )
                        await self._repository.save(session)

                        # 2. Verificar límite de iteraciones
                        if iteration >= self._max_tool_iterations:
                            yield ChatStreamEvent(
                                type=ChatStreamEventType.ERROR,
                                error_code="tool_loop_limit_exceeded",
                                error_message=(
                                    f"Se alcanzó el máximo de iteraciones de "
                                    f"tool calling ({self._max_tool_iterations})"
                                ),
                            )
                            return

                        # 3. Ejecutar tools y persistir resultados
                        execution_context = self._build_tool_execution_context(session)
                        executed_tools: list[dict] = []

                        for tool_call in tool_calls_this_turn:
                            tool_result = await self._tool_executor.execute_call(
                                tool_call, execution_context
                            )
                            session.add_tool_message(
                                tool_result.content,
                                tool_call_id=tool_result.tool_call_id,
                                name=tool_result.name,
                            )
                            executed_tools.append(
                                {
                                    "tool_call_id": tool_result.tool_call_id,
                                    "name": tool_result.name,
                                    "content": tool_result.content,
                                    "is_error": tool_result.is_error,
                                }
                            )

                        await self._repository.save(session)

                        # 4. Notificar al consumidor sobre las tools ejecutadas
                        yield ChatStreamEvent(
                            type=ChatStreamEventType.TOOL_USE,
                            tool_calls=tool_calls_this_turn,
                            raw_event={"executed_tools": executed_tools},
                        )

                        # 5. Continuar el loop — el break sale del for interno
                        # y la iteración del while continúa
                        break

                    else:
                        # ----- Turno final (solo texto) -----
                        if final_text:
                            session.add_assistant_message(
                                final_text,
                                provider_code=session.current_provider_code,
                                model_key=session.current_model_key,
                            )
                            await self._repository.save(session)

                        yield ChatStreamEvent(type=ChatStreamEventType.END)
                        return

                elif event.type == ChatStreamEventType.ERROR:
                    got_error = True
                    yield event
                    return

            # Si salimos del for por break (hubo tool_use), la iteración continúa.
            # Si no hubo tool_use y no hubo error pero tampoco END, algo fue mal.
            if not got_tool_use and not got_error:
                # Salvaguarda: el stream terminó sin END ni TOOL_USE
                final_text = "".join(accumulated_text)
                if final_text:
                    session.add_assistant_message(
                        final_text,
                        provider_code=session.current_provider_code,
                        model_key=session.current_model_key,
                    )
                    await self._repository.save(session)
                yield ChatStreamEvent(type=ChatStreamEventType.END)
                return

        # Agotamos las iteraciones sin llegar a un END limpio
        raise ToolLoopLimitExceededError(self._max_tool_iterations)

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    async def _prepare_session(
        self,
        *,
        session: ConversationSession,
        user_text: str,
        target_provider_code: str,
        target_model_key: str,
    ) -> ConversationSession:
        if (
            session.current_provider_code != target_provider_code
            or session.current_model_key != target_model_key
        ):
            session.switch_model(
                provider_code=target_provider_code,
                model_key=target_model_key,
            )

        session.add_user_message(user_text)
        await self._repository.save(session)
        return session

    def _resolve_tool_definitions(
        self,
        tool_names: list[str] | None,
    ) -> list[ToolDefinition]:
        """
        Resuelve las tool definitions a enviar al modelo.

        - tool_names=None  → todas las tools registradas
        - tool_names=[]    → sin tools (lista vacía explícita)
        - tool_names=[...] → solo las tools indicadas
        """
        if tool_names is None:
            return self._tool_registry.list_definitions()

        # lista explícita (puede ser vacía)
        return self._tool_registry.list_definitions(tool_names)

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
    ) -> ChatRequest:
        resolved_tools = list(tool_definitions or [])

        metadata: dict[str, Any] = {"conversation_id": str(session.id)}
        if resolved_tools:
            metadata["tool_names"] = [t.name for t in resolved_tools]

        return ChatRequest(
            provider_code=session.current_provider_code,
            model_key=session.current_model_key,
            messages=self._context_builder.build(session),
            config=self._default_config,
            metadata=metadata,
            tools=resolved_tools,
        )
    
    def _ensure_provider_available(self, provider_code: str) -> None:
        if not self._provider_registry.has(provider_code):
            raise InvalidProviderSelectionError(
                f"No hay adapter registrado para provider '{provider_code}'"
            )
        
    def _validate_chat_target(self, provider_code: str, model_key: str) -> None:
        self._model_catalog.require_model(provider_code, model_key)
        self._model_catalog.require_capability(provider_code, model_key, "chat")

    def _validate_chat_operation_capabilities(
        self,
        provider_code: str,
        model_key: str,
        *,
        tools_enabled: bool,
        streaming: bool,
    ) -> None:
        self._validate_chat_target(provider_code, model_key)

        if tools_enabled:
            self._model_catalog.require_capability(provider_code, model_key, "tools")

        if streaming:
            self._model_catalog.require_capability(provider_code, model_key, "streaming")

    def _resolve_effective_target(
        self,
        *,
        session: ConversationSession,
        provider_code: str | None,
        model_key: str | None,
    ) -> tuple[str, str]:
        if provider_code is not None and model_key is None:
            raise InvalidProviderSelectionError(
                "Si cambias provider_code, debes indicar también model_key"
            )

        if provider_code is not None:
            resolved_model_key = model_key
            if resolved_model_key is None:
                raise InvalidProviderSelectionError(
                    "Si cambias provider_code, debes indicar también model_key"
                )
            return provider_code, resolved_model_key

        if model_key is not None:
            return session.current_provider_code, model_key

        return session.current_provider_code, session.current_model_key
    
    async def _load_validated_session_target(
        self,
        *,
        conversation_id: UUID,
        provider_code: str | None,
        model_key: str | None,
        tools_enabled: bool,
        streaming: bool,
    ) -> tuple[ConversationSession, str, str]:
        session = await self._repository.get(conversation_id)

        target_provider_code, target_model_key = self._resolve_effective_target(
            session=session,
            provider_code=provider_code,
            model_key=model_key,
        )

        self._ensure_provider_available(target_provider_code)
        self._validate_chat_operation_capabilities(
            target_provider_code,
            target_model_key,
            tools_enabled=tools_enabled,
            streaming=streaming,
        )

        return session, target_provider_code, target_model_key