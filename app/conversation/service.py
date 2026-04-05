from collections.abc import AsyncGenerator
from uuid import UUID

from app.conversation.interfaces import ContextBuilder, ConversationRepository
from app.conversation.models import ConversationSession
from app.core.exceptions import InvalidProviderSelectionError
from app.llm.contracts import (
    LLMCompletionResult,
    LLMRequest,
    LLMRequestConfig,
    LLMStreamEvent,
    StreamEventType,
)
from app.llm.orchestrator import LLMOrchestrator
from app.llm.registry import ProviderRegistry


class ConversationService:
    """
    Servicio conversacional multiproveedor.

    Depende de interfaces (ConversationRepository, ContextBuilder) y no de
    implementaciones concretas, siguiendo el principio de inversión de dependencias.
    Esto permite sustituir InMemoryConversationRepository por PostgreSQL u otro
    backend sin modificar este servicio.
    """

    def __init__(
        self,
        *,
        repository: ConversationRepository,
        context_builder: ContextBuilder,
        orchestrator: LLMOrchestrator,
        provider_registry: ProviderRegistry,
        default_config: LLMRequestConfig | None = None,
    ) -> None:
        self._repository = repository
        self._context_builder = context_builder
        self._orchestrator = orchestrator
        self._provider_registry = provider_registry
        self._default_config = default_config or LLMRequestConfig()

    # ------------------------------------------------------------------
    # API publica
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

    async def change_model(
        self,
        *,
        conversation_id: UUID,
        provider_code: str | None = None,
        model_key: str | None = None,
    ) -> ConversationSession:
        if provider_code is not None and model_key is None:
            raise InvalidProviderSelectionError(
                "Si cambias provider_code, debes indicar tambien model_key"
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
    ) -> LLMCompletionResult:
        session, request = await self._prepare_session(
            conversation_id=conversation_id,
            user_text=user_text,
            provider_code=provider_code,
            model_key=model_key,
        )

        result = await self._orchestrator.complete(request)

        session.add_assistant_message(
            result.content,
            provider_code=session.current_provider_code,
            model_key=session.current_model_key,
        )
        await self._repository.save(session)
        return result

    async def stream_message(
        self,
        *,
        conversation_id: UUID,
        user_text: str,
        provider_code: str | None = None,
        model_key: str | None = None,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        session, request = await self._prepare_session(
            conversation_id=conversation_id,
            user_text=user_text,
            provider_code=provider_code,
            model_key=model_key,
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
    ) -> tuple[ConversationSession, LLMRequest]:
        """
        Logica compartida entre send_message() y stream_message():
        - valida el cambio de proveedor/modelo si se solicita
        - recupera la sesion
        - aplica el switch si corresponde
        - agrega el mensaje del usuario
        - construye el LLMRequest

        Retorna (session, request) para que el llamador pueda persistir
        la respuesta una vez que la tenga.
        """
        if provider_code is not None and model_key is None:
            raise InvalidProviderSelectionError(
                "Si cambias provider_code, debes indicar tambien model_key"
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

        request = LLMRequest(
            provider_code=session.current_provider_code,
            model_key=session.current_model_key,
            messages=self._context_builder.build(session),
            config=self._default_config,
            metadata={"conversation_id": str(session.id)},
        )

        return session, request
