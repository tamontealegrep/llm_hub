from app.capabilities.image_generation.contracts import (
    ImageGenerationRequest,
    ImageGenerationResult,
)
from app.runtime.execution.image_orchestrator import ImageOrchestrator


class ImageGenerationService:
    def __init__(self, orchestrator: ImageOrchestrator) -> None:
        self._orchestrator = orchestrator

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        return await self._orchestrator.generate(request)