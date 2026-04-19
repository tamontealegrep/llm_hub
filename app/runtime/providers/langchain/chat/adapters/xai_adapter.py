from langchain_core.language_models.chat_models import BaseChatModel

from app.runtime.providers.langchain.chat.core import ProviderAdapterMeta, BaseLangChainChatAdapter
from app.capabilities.chat.contracts import ChatRequest
from app.shared.exceptions import (
    ProviderTimeoutError,
    ProviderRateLimitError,
    ProviderAuthenticationError,
    ProviderUnavailableError,
    ProviderError,
)


class XAIChatAdapter(BaseLangChainChatAdapter):
    provider_code = "xai"
    adapter_meta = ProviderAdapterMeta(
        provider_code="xai",
        env_key_name="xai_api_key",
        required_packages=["langchain_xai"],
    )

    def _build_chat_model(self, request: ChatRequest) -> BaseChatModel:
        return self._factory.build_chat_model(request.model_key, request.config, self.adapter_meta)

    def _normalize_exception(self, exc: Exception) -> Exception:
        # xAI comparte la librería openai internamente (mismo protocolo)
        try:
            import openai
            if isinstance(exc, openai.APITimeoutError):
                return ProviderTimeoutError(f"Timeout invocando proveedor '{self.provider_code}'")
            if isinstance(exc, openai.RateLimitError):
                return ProviderRateLimitError(f"Rate limit en proveedor '{self.provider_code}'")
            if isinstance(exc, openai.AuthenticationError):
                return ProviderAuthenticationError(f"Auth error en proveedor '{self.provider_code}'")
            if isinstance(exc, openai.APIStatusError) and exc.status_code == 503:
                return ProviderUnavailableError(f"Proveedor '{self.provider_code}' no disponible")
        except ImportError:
            pass
        # Fallback al manejo genérico del base
        return super()._normalize_exception(exc)