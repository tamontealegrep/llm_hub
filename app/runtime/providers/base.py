from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.capabilities.chat.contracts import ChatCompletionResult, ChatRequest, ChatStreamEvent


class ChatProviderAdapter(ABC):
    """
    Interfaz común para cualquier proveedor LLM.
    """

    provider_code: str

    @abstractmethod
    async def complete(self, request: ChatRequest) -> ChatCompletionResult:
        raise NotImplementedError

    @abstractmethod
    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        raise NotImplementedError