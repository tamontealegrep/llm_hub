# app/runtime/providers/langchain/chat/core/messages.py
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.capabilities.chat.contracts import ChatMessage
from app.files.entities import FileAttachmentRef


def _build_human_message_with_attachments(
    message: ChatMessage, supports_vision: bool = False
) -> HumanMessage:
    """
    Construye HumanMessage multimodal cuando hay attachments.
    Estrategia híbrida (igual que en el manual original).
    """
    content_blocks: list[dict[str, Any]] = []

    # 1. Texto principal del usuario
    if message.content:
        content_blocks.append({"type": "text", "text": message.content})

    # 2. Procesar cada attachment
    for attachment in message.attachments:
        is_image = attachment.mime_type.startswith("image/")

        if is_image and supports_vision:
            # Modelos con vision nativa (Claude, GPT-4o, Gemini)
            text = (
                f"\n\n[Imagen adjunta: {attachment.filename}]\n{attachment.resolved_text}"
                if attachment.resolved_text
                else f"\n\n[Imagen adjunta: {attachment.filename} ({attachment.mime_type})]"
            )
            content_blocks.append({"type": "text", "text": text})

        elif attachment.resolved_text:
            # Archivos de texto / PDF / etc.
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

    # Si solo hay un bloque de texto → enviarlo como string simple (mejor compatibilidad)
    if len(content_blocks) == 1 and content_blocks[0]["type"] == "text":
        return HumanMessage(content=content_blocks[0]["text"])

    return HumanMessage(content=content_blocks)


def to_langchain_messages(
    messages: list[ChatMessage], supports_vision: bool = False
) -> list:
    """
    Convierte ChatMessage a mensajes LangChain.
    Soporta roles: system, user, assistant (con o sin tool_calls), tool.
    
    Parámetro nuevo: supports_vision (usado solo para mensajes USER con attachments)
    """
    converted = []

    for message in messages:
        if message.role == "system":
            converted.append(SystemMessage(content=message.content))

        elif message.role == "user":
            if not message.attachments:
                # Sin attachments → comportamiento original
                converted.append(HumanMessage(content=message.content))
            else:
                # Con attachments → ruta multimodal
                converted.append(
                    _build_human_message_with_attachments(
                        message, supports_vision=supports_vision
                    )
                )

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