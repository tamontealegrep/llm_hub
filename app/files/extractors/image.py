# app/files/extractors/image.py
from __future__ import annotations

from app.files.extractors.base import FileExtractor

_IMAGE_MIME_TYPES = frozenset({
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
})


class ImageFileExtractor(FileExtractor):
    """
    Extractor para imágenes.
    Retorna un placeholder — el adapter las enviará de forma nativa si el modelo lo soporta.
    """

    def can_extract(self, mime_type: str) -> bool:
        return mime_type in _IMAGE_MIME_TYPES

    def extract(self, content: bytes, mime_type: str, filename: str) -> str:
        # TODO (Fase 5): OCR opcional.
        return f"[Imagen adjunta: {filename} ({mime_type}, {len(content)} bytes)]"