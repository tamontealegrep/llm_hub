# app/files/repository.py
from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from app.files.entities import UploadedFile
from app.shared.exceptions import AppError


class FileNotFoundError(AppError):
    def __init__(self, file_id: UUID) -> None:
        super().__init__(
            f"Archivo no encontrado: {file_id}",
            details={"file_id": str(file_id)},
        )


class FileRepository(ABC):
    """Contrato abstracto para persistencia de metadatos de archivos."""

    @abstractmethod
    async def save(self, file: UploadedFile) -> None: ...

    @abstractmethod
    async def get(self, file_id: UUID) -> UploadedFile: ...

    @abstractmethod
    async def delete(self, file_id: UUID) -> None: ...


class InMemoryFileRepository(FileRepository):
    """
    Implementación en memoria.
    Para producción, implementar con PostgreSQL u otro backend.
    """

    def __init__(self) -> None:
        self._store: dict[UUID, UploadedFile] = {}

    async def save(self, file: UploadedFile) -> None:
        self._store[file.id] = file

    async def get(self, file_id: UUID) -> UploadedFile:
        try:
            return self._store[file_id]
        except KeyError as exc:
            raise FileNotFoundError(file_id) from exc

    async def delete(self, file_id: UUID) -> None:
        self._store.pop(file_id, None)