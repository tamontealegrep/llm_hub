from langchain_core.language_models.chat_models import BaseChatModel

from app.runtime.providers.langchain.chat.base import BaseProviderAdapter
from app.capabilities.chat.contracts import LLMRequest


class OpenAIAdapter(BaseProviderAdapter):
    provider_code = "openai"

    def _build_chat_model(self, request: LLMRequest) -> BaseChatModel:
        return self._factory.build_openai(request.model_key, request.config)
