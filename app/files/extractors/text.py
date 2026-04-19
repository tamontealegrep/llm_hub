# app/files/extractors/text.py
from __future__ import annotations

from app.files.extractors.base import FileExtractor

_TEXT_MIME_TYPES = frozenset({
    "text/plain",
    "text/markdown",
    "application/json",
    "text/csv",
})


class TextFileExtractor(FileExtractor):
    """
    Extractor para archivos de texto plano: TXT, MD, JSON, CSV.
    Simplemente decodifica los bytes como UTF-8.
    """

    def can_extract(self, mime_type: str) -> bool:
        return mime_type in _TEXT_MIME_TYPES

    def extract(self, content: bytes, mime_type: str, filename: str) -> str:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback a latin-1 para archivos mal codificados
            return content.decode("latin-1", errors="replace")