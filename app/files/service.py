# app/files/service.py
from __future__ import annotations

from uuid import UUID

from app.files.contracts import UploadFileRequest, UploadFileResult
from app.files.entities import FileAttachmentRef, FileStatus, UploadedFile
from app.files.extractors.base import FileExtractor
from app.files.repository import FileRepository
from app.files.storage import FileStorage
from app.files.validators import validate_upload


class FileService:
    """
    Casos de uso de archivos:
    - upload: sube, valida, extrae texto y guarda metadatos
    - get_ref: retorna una FileAttachmentRef liviana para adjuntar a mensajes
    - get_file: retorna la entidad completa
    - load_content: retorna los bytes crudos desde storage
    """

    def __init__(
        self,
        repository: FileRepository,
        storage: FileStorage,
        extractors: list[FileExtractor],
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._extractors = extractors

    async def upload(self, request: UploadFileRequest) -> UploadFileResult:
        """
        Sube un archivo al hub:
        1. Valida MIME, extensión y tamaño.
        2. Calcula el checksum.
        3. Guarda en storage.
        4. Extrae texto si hay un extractor disponible.
        5. Persiste los metadatos.
        6. Retorna el resultado.
        """
        checksum = validate_upload(
            filename=request.filename,
            mime_type=request.mime_type,
            content=request.content,
        )

        uploaded_file = UploadedFile.create(
            filename=request.filename,
            mime_type=request.mime_type,
            size_bytes=len(request.content),
            storage_uri="",  # se actualiza después de guardar en storage
            checksum=checksum,
            owner_id=request.owner_id,
        )

        storage_uri = self._storage.save(
            file_id=uploaded_file.id,
            content=request.content,
            filename=request.filename,
        )
        uploaded_file.storage_uri = storage_uri

        extracted_text = self._try_extract(request.content, request.mime_type, request.filename)
        uploaded_file.extracted_text = extracted_text
        uploaded_file.status = FileStatus.READY

        await self._repository.save(uploaded_file)

        return UploadFileResult(
            file_id=uploaded_file.id,
            filename=uploaded_file.filename,
            mime_type=uploaded_file.mime_type,
            size_bytes=uploaded_file.size_bytes,
            storage_uri=uploaded_file.storage_uri,
            extracted_text=uploaded_file.extracted_text,
        )

    async def get_ref(self, file_id: UUID) -> FileAttachmentRef:
        """Retorna una referencia liviana al archivo (para adjuntar a mensajes)."""
        file = await self._repository.get(file_id)
        return FileAttachmentRef.from_uploaded_file(file)

    async def get_file(self, file_id: UUID) -> UploadedFile:
        """Retorna la entidad completa del archivo."""
        return await self._repository.get(file_id)

    async def load_content(self, file_id: UUID) -> bytes:
        """Carga el contenido binario del archivo desde storage."""
        file = await self._repository.get(file_id)
        return self._storage.load(file.storage_uri)

    def _try_extract(self, content: bytes, mime_type: str, filename: str) -> str | None:
        """
        Busca un extractor adecuado y extrae el texto.
        Si no hay extractor disponible, retorna None.
        """
        for extractor in self._extractors:
            if extractor.can_extract(mime_type):
                try:
                    return extractor.extract(content, mime_type, filename)
                except Exception:
                    return None
        return None