from langchain_core.language_models.chat_models import BaseChatModel

from app.runtime.providers.langchain.chat.base import BaseProviderAdapter
from app.capabilities.chat.contracts import LLMRequest


class AnthropicAdapter(BaseProviderAdapter):
    provider_code = "anthropic"

    def _build_chat_model(self, request: LLMRequest) -> BaseChatModel:
        return self._factory.build_anthropic(request.model_key, request.config)
