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


class GoogleChatAdapter(BaseLangChainChatAdapter):
    provider_code = "google"
    adapter_meta = ProviderAdapterMeta(
        provider_code="google",
        env_key_name="google_api_key",
        required_packages=["langchain_google_genai"],
    )

    def _build_chat_model(self, request: ChatRequest) -> BaseChatModel:
        return self._factory.build_chat_model(request.model_key, request.config, self.adapter_meta)

    def _normalize_exception(self, exc: Exception) -> Exception:
        try:
            from google.api_core import exceptions as google_exc
            if isinstance(exc, google_exc.DeadlineExceeded):
                return ProviderTimeoutError(f"Timeout invocando proveedor '{self.provider_code}'")
            if isinstance(exc, google_exc.ResourceExhausted):
                return ProviderRateLimitError(f"Rate limit en proveedor '{self.provider_code}'")
            if isinstance(exc, google_exc.Unauthenticated):
                return ProviderAuthenticationError(f"Auth error en proveedor '{self.provider_code}'")
            if isinstance(exc, google_exc.ServiceUnavailable):
                return ProviderUnavailableError(f"Proveedor '{self.provider_code}' no disponible")
        except ImportError:
            pass
        # Fallback al manejo genérico del base
        return super()._normalize_exception(exc)