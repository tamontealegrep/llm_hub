# app/files/contracts.py
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID


@dataclass(frozen=True, slots=True)
class UploadFileRequest:
    """
    Request para subir un archivo al hub.

    content: bytes del archivo.
    filename: nombre original del archivo.
    mime_type: tipo MIME declarado por el cliente.
    owner_id: identificador del dueño (None = sistema / anónimo).
    """
    content: bytes
    filename: str
    mime_type: str
    owner_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UploadFileResult:
    """Resultado después de subir y procesar un archivo."""
    file_id: UUID
    filename: str
    mime_type: str
    size_bytes: int
    storage_uri: str
    extracted_text: str | None = None


@dataclass(frozen=True, slots=True)
class AttachmentInput:
    """
    Input simplificado para adjuntar un archivo a un mensaje.
    Se usa cuando el archivo ya fue subido y solo se pasa la referencia.
    """
    file_id: UUID


# Tipos MIME soportados en la primera ola
SUPPORTED_MIME_TYPES_V1: frozenset[str] = frozenset({
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "text/plain",
    "text/markdown",
    "application/json",
    "text/csv",
})

# Extensiones de archivo que se mapean a MIME types conocidos
EXTENSION_TO_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".csv": "text/csv",
}