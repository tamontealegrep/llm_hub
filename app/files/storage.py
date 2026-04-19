# app/files/storage.py
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from uuid import UUID


class FileStorage(ABC):
    """
    Abstracción para el almacenamiento físico de archivos.
    Permite cambiar de local → S3 → GCS sin tocar el resto del sistema.
    """

    @abstractmethod
    def save(self, file_id: UUID, content: bytes, filename: str) -> str:
        """Guarda el contenido y retorna el storage_uri."""
        raise NotImplementedError

    @abstractmethod
    def load(self, storage_uri: str) -> bytes:
        """Carga el contenido desde el storage_uri."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, storage_uri: str) -> None:
        """Elimina el archivo del storage."""
        raise NotImplementedError


class LocalFileStorage(FileStorage):
    """
    Implementación de FileStorage que guarda archivos en el sistema de archivos local.
    Ideal para desarrollo y pruebas.
    Para producción, implementar S3FileStorage o GCSFileStorage.
    """

    def __init__(self, base_dir: str | Path = "/tmp/llm_hub_files") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, file_id: UUID, content: bytes, filename: str) -> str:
        # Usamos file_id como nombre de directorio para evitar colisiones
        file_dir = self._base_dir / str(file_id)
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / filename
        file_path.write_bytes(content)
        return str(file_path)

    def load(self, storage_uri: str) -> bytes:
        path = Path(storage_uri)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado en storage: {storage_uri}")
        return path.read_bytes()

    def delete(self, storage_uri: str) -> None:
        path = Path(storage_uri)
        if path.exists():
            os.remove(path)
            # Eliminar directorio padre si está vacío
            try:
                path.parent.rmdir()
            except OSError:
                pass  # no vacío, no pasa nada