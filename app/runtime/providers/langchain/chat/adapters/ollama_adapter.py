"""
Adapter para modelos locales vía Ollama.

Requisito: tener Ollama corriendo localmente (https://ollama.ai)
Instalación del paquete: pip install langchain-ollama

Uso:
  1. Instala Ollama en tu máquina.
  2. Descarga un modelo: `ollama pull llama3.2` (o el que prefieras).
  3. Crea catalog/models/ollama.yaml con los modelos que tengas descargados.
  4. No se necesita API key en .env.

Ventajas: privacidad total, sin costos por token, funciona offline.
Limitaciones: requiere hardware adecuado, no todos los modelos soportan tool calling.
"""
from langchain_core.language_models.chat_models import BaseChatModel

from app.runtime.providers.langchain.chat.core import ProviderAdapterMeta, BaseLangChainChatAdapter
from app.capabilities.chat.contracts import ChatRequest


class OllamaChatAdapter(BaseLangChainChatAdapter):
    provider_code = "ollama"
    adapter_meta = ProviderAdapterMeta(
        provider_code="ollama",
        env_key_name=None,          # No necesita API key
        required_packages=["langchain_ollama"],
        is_local=True,
    )

    def _build_chat_model(self, request: ChatRequest) -> BaseChatModel:
        return self._factory.build_chat_model(request.model_key, request.config, self.adapter_meta)