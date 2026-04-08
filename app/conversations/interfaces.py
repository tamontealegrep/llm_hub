from abc import ABC, abstractmethod
from uuid import UUID

from app.conversations.entities import ConversationSession
from app.capabilities.chat.contracts import ChatMessage


class ConversationRepository(ABC):
    """
    Contrato abstracto para el almacenamiento de conversaciones.
    Permite intercambiar InMemoryConversationRepository por cualquier
    implementación persistente (PostgreSQL, Redis, etc.) sin tocar el servicio.
    """

    @abstractmethod
    async def create(
        self,
        *,
        provider_code: str,
        model_key: str,
        system_prompt: str | None = None,
    ) -> ConversationSession: ...

    @abstractmethod
    async def get(self, conversation_id: UUID) -> ConversationSession: ...

    @abstractmethod
    async def save(self, session: ConversationSession) -> None: ...


class ContextBuilder(ABC):
    """
    Contrato abstracto para la construcción del contexto que se envía al modelo.
    Permite intercambiar SimpleContextBuilder por versiones con trimming por tokens,
    resumen automático, etc.
    """

    @abstractmethod
    def build(self, session: ConversationSession) -> list[ChatMessage]: ...
