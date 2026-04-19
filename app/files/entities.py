# app/files/entities.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FileStatus(str, Enum):
    PENDING = "pending"        # recién subido, aún no procesado
    READY = "ready"            # listo para usar
    EXTRACTING = "extracting"  # extrayendo texto (para uso futuro async)
    FAILED = "failed"          # falló el procesamiento


class AttachmentHandlingMode(str, Enum):
    """Cómo se enviará este archivo al LLM."""
    NATIVE = "native"              # el provider lo acepta directamente
    EXTRACT_TEXT = "extract_text"  # se extrae el texto y se envía como texto
    HYBRID = "hybrid"              # nativo si el modelo lo soporta, extract_text si no


@dataclass(slots=True)
class UploadedFile:
    """
    Entidad central de dominio para un archivo subido al hub.

    El contenido binario NO vive aquí — solo los metadatos y
    opcionalmente el texto extraído.
    El archivo físico se referencia mediante storage_uri.
    """
    id: UUID
    filename: str
    mime_type: str
    size_bytes: int
    storage_uri: str           # path local, URL de S3, GCS URI, etc.
    status: FileStatus = FileStatus.PENDING
    extracted_text: str | None = None   # contenido extraído (si aplica)
    checksum: str | None = None         # sha256 del contenido
    owner_id: str | None = None         # para cuando haya usuarios
    created_at: datetime = field(default_factory=_utcnow)
    extra_metadata: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        filename: str,
        mime_type: str,
        size_bytes: int,
        storage_uri: str,
        checksum: str | None = None,
        owner_id: str | None = None,
    ) -> "UploadedFile":
        return cls(
            id=uuid4(),
            filename=filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            storage_uri=storage_uri,
            checksum=checksum,
            owner_id=owner_id,
        )


@dataclass(frozen=True, slots=True)
class FileAttachmentRef:
    """
    Referencia ligera a un archivo adjunto dentro de un mensaje.

    Los mensajes y las sesiones solo guardan estas referencias,
    nunca el contenido binario ni la entidad completa.
    """
    file_id: UUID
    filename: str
    mime_type: str
    size_bytes: int

    # Texto ya resuelto (se puede poblar en el context builder
    # para no tener que ir al storage en cada request).
    resolved_text: str | None = None

    @classmethod
    def from_uploaded_file(cls, f: UploadedFile) -> "FileAttachmentRef":
        return cls(
            file_id=f.id,
            filename=f.filename,
            mime_type=f.mime_type,
            size_bytes=f.size_bytes,
            resolved_text=f.extracted_text,
        )