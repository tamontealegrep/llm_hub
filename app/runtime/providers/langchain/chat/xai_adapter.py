from langchain_core.language_models.chat_models import BaseChatModel

from app.runtime.providers.langchain.chat.base import ProviderAdapterMeta, BaseLangChainChatAdapter
from app.capabilities.chat.contracts import ChatRequest


class XAIChatAdapter(BaseLangChainChatAdapter):
    provider_code = "xai"
    adapter_meta = ProviderAdapterMeta(
        provider_code="xai",
        env_key_name="xai_api_key",
        required_packages=["langchain_xai"],
    )
    def _build_chat_model(self, request: ChatRequest) -> BaseChatModel:
        return self._factory.build_chat_model(request.model_key, request.config, self.adapter_meta)