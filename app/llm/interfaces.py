from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.llm.contracts import LLMCompletionResult, LLMRequest, LLMStreamEvent


class LLMProviderAdapter(ABC):
    """
    Interfaz común para cualquier proveedor LLM.
    """

    provider_code: str

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMCompletionResult:
        raise NotImplementedError

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        raise NotImplementedError