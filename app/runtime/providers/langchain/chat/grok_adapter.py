from langchain_core.language_models.chat_models import BaseChatModel

from app.runtime.providers.langchain.chat.base import BaseLangChainChatAdapter
from app.capabilities.chat.contracts import ChatRequest


class GrokAdapter(BaseLangChainChatAdapter):
    provider_code = "xai"

    def _build_chat_model(self, request: ChatRequest) -> BaseChatModel:
        return self._factory.build_xai(request.model_key, request.config)
