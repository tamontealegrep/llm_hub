# app/files/extractors/pdf.py
from __future__ import annotations

from app.files.extractors.base import FileExtractor


class PdfFileExtractor(FileExtractor):
    """
    Extractor para PDFs.
    Placeholder — en Fase 5 se integrará pypdf o pdfplumber para extracción real.
    """

    def can_extract(self, mime_type: str) -> bool:
        return mime_type == "application/pdf"

    def extract(self, content: bytes, mime_type: str, filename: str) -> str:
        # TODO (Fase 5): integrar pypdf o pdfplumber.
        # Los adapters que soporten PDF nativo lo enviarán directamente al modelo.
        return f"[Archivo PDF adjunto: {filename} ({len(content)} bytes)]"