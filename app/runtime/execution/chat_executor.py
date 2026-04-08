from collections.abc import AsyncIterator

from app.capabilities.chat.contracts import LLMCompletionResult, LLMRequest, LLMStreamEvent
from app.runtime.providers.registry import ProviderRegistry


class LLMOrchestrator:
    """
    Orquestador multiproveedor.
    Selecciona el adapter correcto y delega la invocación.
    """

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    async def complete(self, request: LLMRequest) -> LLMCompletionResult:
        adapter = self._registry.get(request.provider_code)
        return await adapter.complete(request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        adapter = self._registry.get(request.provider_code)
        async for event in adapter.stream(request):
            yield event