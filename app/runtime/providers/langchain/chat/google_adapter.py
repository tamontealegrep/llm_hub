from langchain_core.language_models.chat_models import BaseChatModel

from app.runtime.providers.langchain.chat.base import ProviderAdapterMeta, BaseLangChainChatAdapter
from app.capabilities.chat.contracts import ChatRequest


class GoogleChatAdapter(BaseLangChainChatAdapter):
    provider_code = "google"
    adapter_meta = ProviderAdapterMeta(
        provider_code="google",
        env_key_name="google_api_key",
        required_packages=["langchain_google_genai"],
    )
    def _build_chat_model(self, request: ChatRequest) -> BaseChatModel:
        return self._factory.build_chat_model(request.model_key, request.config, self.adapter_meta)
