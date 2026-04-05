from langchain_core.language_models.chat_models import BaseChatModel

from app.llm.adapters.base import BaseProviderAdapter
from app.llm.contracts import LLMRequest


class GrokAdapter(BaseProviderAdapter):
    provider_code = "xai"

    def _build_chat_model(self, request: LLMRequest) -> BaseChatModel:
        return self._factory.build_xai(request.model_key, request.config)
