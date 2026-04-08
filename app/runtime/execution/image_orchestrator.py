from app.capabilities.image_generation.contracts import (
    ImageGenerationRequest,
    ImageGenerationResult,
)


class ImageOrchestrator:
    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        raise NotImplementedError(
            "ImageOrchestrator aún no está implementado. "
            "Se añadirá cuando integres providers de image_generation."
        )