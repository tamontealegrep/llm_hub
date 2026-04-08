from langchain_core.language_models.chat_models import BaseChatModel

from app.runtime.providers.langchain.chat.base import BaseLangChainChatAdapter
from app.capabilities.chat.contracts import ChatRequest


class AnthropicChatAdapter(BaseLangChainChatAdapter):
    provider_code = "anthropic"

    def _build_chat_model(self, request: ChatRequest) -> BaseChatModel:
        return self._factory.build_anthropic(request.model_key, request.config)
