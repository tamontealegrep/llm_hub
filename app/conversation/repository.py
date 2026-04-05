from uuid import UUID

from app.conversation.interfaces import ConversationRepository
from app.conversation.models import ConversationSession
from app.core.exceptions import ConversationNotFoundError


class InMemoryConversationRepository(ConversationRepository):
    """
    Implementación de ConversationRepository en memoria.
    Ideal para validar el flujo conversacional antes de usar una base de datos.
    Para producción, implementar ConversationRepository con PostgreSQL u otro backend.
    """

    def __init__(self) -> None:
        self._store: dict[UUID, ConversationSession] = {}

    async def create(
        self,
        *,
        provider_code: str,
        model_key: str,
        system_prompt: str | None = None,
    ) -> ConversationSession:
        session = ConversationSession(
            current_provider_code=provider_code,
            current_model_key=model_key,
            system_prompt=system_prompt,
        )
        self._store[session.id] = session
        return session

    async def get(self, conversation_id: UUID) -> ConversationSession:
        try:
            return self._store[conversation_id]
        except KeyError as exc:
            raise ConversationNotFoundError() from exc

    async def save(self, session: ConversationSession) -> None:
        self._store[session.id] = session
