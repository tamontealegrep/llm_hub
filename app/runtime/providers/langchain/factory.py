from langchain_core.language_models.chat_models import BaseChatModel

from app.shared.settings.env import EnvSettings
from app.shared.exceptions import ProviderConfigurationError
from app.capabilities.chat.contracts import LLMRequestConfig


class LangChainChatModelFactory:
    """
    Fábrica central de clientes/modelos LangChain por proveedor.
    Retorna BaseChatModel en todos los métodos para permitir tipado correcto
    en adapters y habilitar el soporte del type checker.
    """

    def __init__(self, settings: EnvSettings) -> None:
        self._settings = settings

    def build_openai(self, model_key: str, config: LLMRequestConfig) -> BaseChatModel:
        if not self._settings.openai_api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY no está configurada")

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_key,
            api_key=self._settings.openai_api_key,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
            timeout=config.timeout_seconds,
            top_p=config.top_p,
            # max_retries=2 es el default de langchain-openai; se deja explícito
            # para documentar la decisión (antes estaba en 0, lo que eliminaba
            # la resiliencia ante errores transitorios de red).
            max_retries=2,
        )

    def build_anthropic(self, model_key: str, config: LLMRequestConfig) -> BaseChatModel:
        if not self._settings.anthropic_api_key:
            raise ProviderConfigurationError("ANTHROPIC_API_KEY no está configurada")

        from langchain_anthropic import ChatAnthropic

        top_p = config.top_p if config.top_p != 1.0 else None

        return ChatAnthropic(
            model=model_key,
            api_key=self._settings.anthropic_api_key,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
            top_p=top_p,
            timeout=config.timeout_seconds,
        )

    def build_gemini(self, model_key: str, config: LLMRequestConfig) -> BaseChatModel:
        if not self._settings.google_api_key:
            raise ProviderConfigurationError("GOOGLE_API_KEY no está configurada")

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_key,
            google_api_key=self._settings.google_api_key,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            top_p=config.top_p,
            # convert_system_message_to_human=False permite que SystemMessage
            # se envíe nativamente (soportado desde langchain-google-genai >= 1.x).
            # El workaround manual en GeminiAdapter ya no es necesario.
            convert_system_message_to_human=False,
        )

    def build_xai(self, model_key: str, config: LLMRequestConfig) -> BaseChatModel:
        """
        xAI (Grok) utiliza la librería oficial langchain-xai para interactuar
        de forma nativa con sus modelos.
        """
        if not self._settings.xai_api_key:
            raise ProviderConfigurationError("XAI_API_KEY no está configurada")

        from langchain_xai import ChatXAI

        return ChatXAI(
            model=model_key,
            xai_api_key=self._settings.xai_api_key,
            temperature=config.temperature,
            # ChatXAI soporta max_tokens nativamente
            max_tokens=config.max_output_tokens,
            timeout=config.timeout_seconds,
            # top_p es un parámetro de primer nivel en ChatXAI
            top_p=config.top_p,
        )