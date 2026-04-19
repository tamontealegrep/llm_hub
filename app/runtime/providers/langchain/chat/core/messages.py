# app/runtime/providers/langchain/chat/core/messages.py
from __future__ import annotations
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def to_langchain_messages(messages: list) -> list:
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