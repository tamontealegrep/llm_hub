from langchain_core.language_models.chat_models import BaseChatModel

from app.shared.settings.env import EnvSettings
from app.shared.exceptions import ProviderConfigurationError
from app.capabilities.chat.contracts import ChatRequestConfig
from app.runtime.providers.langchain.chat.base import ProviderAdapterMeta
from app.runtime.providers.langchain.model_builders import BUILDER_REGISTRY


class LangChainProviderFactory:
    """
    Factory genérica: dado un adapter_meta y un config, construye el BaseChatModel.
    No conoce providers individuales; delega en BUILDER_REGISTRY.
    """

    def __init__(self, settings: EnvSettings) -> None:
        self._settings = settings

    def build_chat_model(
        self,
        model_key: str,
        config: ChatRequestConfig,
        meta: ProviderAdapterMeta,
    ) -> BaseChatModel:
        # 1. Verificar que tenemos un builder para este provider
        builder = BUILDER_REGISTRY.get(meta.provider_code)
        if builder is None:
            raise ProviderConfigurationError(
                f"No existe un builder registrado para provider '{meta.provider_code}'. "
                f"Agrégalo en app/runtime/providers/langchain/model_builders.py"
            )

        # 2. Resolver la API key si aplica
        api_key: str | None = None
        if meta.env_key_name:
            api_key = getattr(self._settings, meta.env_key_name, None)
            if not api_key and not meta.is_local:
                raise ProviderConfigurationError(
                    f"La variable de entorno '{meta.env_key_name.upper()}' "
                    f"no está configurada para el provider '{meta.provider_code}'"
                )

        # 3. Verificar paquetes requeridos
        for package in meta.required_packages:
            try:
                __import__(package)
            except ImportError:
                raise ProviderConfigurationError(
                    f"El paquete '{package}' no está instalado. "
                    f"Instálalo con: pip install {package.replace('_', '-')}"
                )

        return builder(model_key, config, api_key)

    # Método de conveniencia para obtener la API key de un provider
    def get_api_key(self, env_key_name: str) -> str | None:
        return getattr(self._settings, env_key_name, None)