from langchain_core.language_models.chat_models import BaseChatModel

from app.runtime.providers.langchain.chat.base import ProviderAdapterMeta, BaseLangChainChatAdapter
from app.capabilities.chat.contracts import ChatRequest


class AnthropicChatAdapter(BaseLangChainChatAdapter):
    provider_code = "anthropic"
    adapter_meta = ProviderAdapterMeta(
        provider_code="anthropic",
        env_key_name="anthropic_api_key",
        required_packages=["langchain_anthropic"],
    )
    def _build_chat_model(self, request: ChatRequest) -> BaseChatModel:
        return self._factory.build_chat_model(request.model_key, request.config, self.adapter_meta)