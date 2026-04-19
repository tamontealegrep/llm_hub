# app/files/validators.py
from __future__ import annotations

import hashlib
from pathlib import Path

from app.files.contracts import EXTENSION_TO_MIME, SUPPORTED_MIME_TYPES_V1
from app.shared.exceptions import AppError


class FileValidationError(AppError):
    """Error al validar un archivo antes de subirlo."""

    def __init__(self, reason: str, details: dict | None = None) -> None:
        super().__init__(reason, details=details or {})


# Límites globales (ajustables en el futuro desde config)
MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024   # 20 MB
MAX_FILES_PER_REQUEST: int = 10


def validate_mime_type(mime_type: str) -> None:
    """Lanza FileValidationError si el MIME type no está permitido."""
    if mime_type not in SUPPORTED_MIME_TYPES_V1:
        raise FileValidationError(
            f"Tipo de archivo no soportado: '{mime_type}'",
            details={
                "mime_type": mime_type,
                "supported": sorted(SUPPORTED_MIME_TYPES_V1),
            },
        )


def validate_extension(filename: str) -> str:
    """
    Verifica que la extensión del archivo sea reconocida.
    Retorna el MIME type asociado a la extensión.
    """
    suffix = Path(filename).suffix.lower()
    mime = EXTENSION_TO_MIME.get(suffix)
    if mime is None:
        raise FileValidationError(
            f"Extensión de archivo no reconocida: '{suffix}'",
            details={
                "filename": filename,
                "extension": suffix,
                "supported_extensions": sorted(EXTENSION_TO_MIME.keys()),
            },
        )
    return mime


def validate_size(size_bytes: int, max_bytes: int = MAX_FILE_SIZE_BYTES) -> None:
    """Lanza FileValidationError si el archivo es demasiado grande."""
    if size_bytes > max_bytes:
        raise FileValidationError(
            f"El archivo excede el tamaño máximo permitido "
            f"({size_bytes} bytes > {max_bytes} bytes)",
            details={
                "size_bytes": size_bytes,
                "max_bytes": max_bytes,
                "max_mb": max_bytes // (1024 * 1024),
            },
        )


def validate_attachment_count(count: int, max_count: int = MAX_FILES_PER_REQUEST) -> None:
    """Lanza FileValidationError si se supera el número máximo de adjuntos."""
    if count > max_count:
        raise FileValidationError(
            f"Se superó el máximo de adjuntos por request "
            f"({count} > {max_count})",
            details={"count": count, "max_count": max_count},
        )


def compute_checksum(content: bytes) -> str:
    """Calcula el checksum SHA-256 del contenido."""
    return hashlib.sha256(content).hexdigest()


def validate_upload(
    *,
    filename: str,
    mime_type: str,
    content: bytes,
) -> str:
    """
    Valida un archivo completo antes de subirlo.
    Retorna el checksum calculado.
    Lanza FileValidationError si algo no es válido.
    """
    validate_extension(filename)
    validate_mime_type(mime_type)
    validate_size(len(content))
    return compute_checksum(content)
