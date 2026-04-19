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


class AnthropicChatAdapter(BaseLangChainChatAdapter):
    provider_code = "anthropic"
    adapter_meta = ProviderAdapterMeta(
        provider_code="anthropic",
        env_key_name="anthropic_api_key",
        required_packages=["langchain_anthropic"],
    )

    def _build_chat_model(self, request: ChatRequest) -> BaseChatModel:
        return self._factory.build_chat_model(request.model_key, request.config, self.adapter_meta)

    def _normalize_exception(self, exc: Exception) -> Exception:
        try:
            import anthropic
            if isinstance(exc, anthropic.APITimeoutError):
                return ProviderTimeoutError(f"Timeout invocando proveedor '{self.provider_code}'")
            if isinstance(exc, anthropic.RateLimitError):
                return ProviderRateLimitError(f"Rate limit en proveedor '{self.provider_code}'")
            if isinstance(exc, anthropic.AuthenticationError):
                return ProviderAuthenticationError(f"Auth error en proveedor '{self.provider_code}'")
            if isinstance(exc, anthropic.APIStatusError) and exc.status_code == 503:
                return ProviderUnavailableError(f"Proveedor '{self.provider_code}' no disponible")
        except ImportError:
            pass
        # Fallback al manejo genérico del base
        return super()._normalize_exception(exc)