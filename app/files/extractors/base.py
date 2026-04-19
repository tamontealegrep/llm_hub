# app/files/extractors/base.py
from __future__ import annotations

from abc import ABC, abstractmethod


class FileExtractor(ABC):
    """
    Contrato abstracto para extractores de contenido.
    Cada extractor sabe convertir un tipo de archivo a texto plano.
    """

    @abstractmethod
    def can_extract(self, mime_type: str) -> bool:
        """Retorna True si este extractor sabe manejar el MIME type."""
        raise NotImplementedError

    @abstractmethod
    def extract(self, content: bytes, mime_type: str, filename: str) -> str:
        """
        Extrae el contenido textual del archivo.
        Lanza ValueError si no puede procesar el contenido.
        """
        raise NotImplementedError