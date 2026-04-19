from langchain_core.language_models.chat_models import BaseChatModel

from app.runtime.providers.langchain.chat.base import ProviderAdapterMeta, BaseLangChainChatAdapter
from app.capabilities.chat.contracts import ChatRequest


class OpenAIChatAdapter(BaseLangChainChatAdapter):
    provider_code = "openai"
    adapter_meta = ProviderAdapterMeta(
        provider_code="openai",
        env_key_name="openai_api_key",
        required_packages=["langchain_openai"],
    )
    def _build_chat_model(self, request: ChatRequest) -> BaseChatModel:
        return self._factory.build_chat_model(request.model_key, request.config, self.adapter_meta)