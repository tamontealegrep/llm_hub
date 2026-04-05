from app.conversation.interfaces import ContextBuilder
from app.conversation.models import ConversationRole, ConversationSession
from app.llm.contracts import NormalizedMessage


class SimpleContextBuilder(ContextBuilder):
    """
    Implementación de ContextBuilder.

    Comportamiento:
    - incluye system prompt si existe
    - incluye historial conversacional en orden
    - opcionalmente recorta por cantidad de mensajes (sin trimming por tokens)

    Para trimming por tokens, implementar ContextBuilder con tiktoken u otro
    contador de tokens sin modificar ConversationService.
    """

    def __init__(self, max_messages: int | None = None) -> None:
        self._max_messages = max_messages

    def build(self, session: ConversationSession) -> list[NormalizedMessage]:
        messages: list[NormalizedMessage] = []

        if session.system_prompt:
            messages.append(
                NormalizedMessage(
                    role="system",
                    content=session.system_prompt,
                )
            )

        history = session.messages
        if self._max_messages is not None:
            history = history[-self._max_messages:]

        for message in history:
            if message.role == ConversationRole.USER:
                messages.append(NormalizedMessage(role="user", content=message.content))
            elif message.role == ConversationRole.ASSISTANT:
                messages.append(NormalizedMessage(role="assistant", content=message.content))

        return messages
