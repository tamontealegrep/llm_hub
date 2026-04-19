from app.conversations.interfaces import ConversationContextBuilder
from app.conversations.entities import ConversationRole, ConversationSession
from app.capabilities.chat.contracts import ChatMessage


class MessageWindowContextBuilder(ConversationContextBuilder):
    """
    Implementación de ConversationContextBuilder.

    Comportamiento:
    - incluye system prompt si existe
    - incluye historial conversacional en orden
    - opcionalmente recorta por cantidad de mensajes (sin trimming por tokens)

    Para trimming por tokens, implementar ConversationContextBuilder con tiktoken u otro
    contador de tokens sin modificar ChatService.
    """

    def __init__(self, max_messages: int | None = None) -> None:
        self._max_messages = max_messages

    def build(self, session: ConversationSession) -> list[ChatMessage]:
        messages: list[ChatMessage] = []

        if session.system_prompt:
            messages.append(
                ChatMessage(
                    role="system",
                    content=session.system_prompt,
                )
            )

        history = session.messages
        if self._max_messages is not None:
            history = history[-self._max_messages:]

        for message in history:
            if message.role == ConversationRole.USER:
                messages.append(
                    ChatMessage(
                        role="user",
                        content=message.content,
                        attachments=list(message.attachments),
                    )
                )

            elif message.role == ConversationRole.ASSISTANT:
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=message.content,
                        tool_calls=list(message.tool_calls),
                    )
                )

            elif message.role == ConversationRole.TOOL:
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=message.content,
                        tool_call_id=message.tool_call_id,
                        name=message.name,
                    )
                )

        return messages