"""
Adapter para modelos HuggingFace vía Inference Endpoints (API remota).

Diferencia con huggingface_pipeline:
  - huggingface_pipeline: el modelo corre en TU máquina (local, en memoria).
  - huggingface_endpoint: el modelo corre en los servidores de HuggingFace (cloud).

Requisito: pip install langchain-huggingface
           HUGGINGFACE_API_KEY en .env (o HUGGINGFACEHUB_API_TOKEN)

Uso:
  1. Obtén tu API key en https://huggingface.co/settings/tokens
  2. Agrégala al .env: HUGGINGFACE_API_KEY=hf_...
  3. Crea catalog/models/huggingface_endpoint.yaml.
"""
from langchain_core.language_models.chat_models import BaseChatModel

from app.runtime.providers.langchain.chat.base import BaseLangChainChatAdapter, ProviderAdapterMeta
from app.capabilities.chat.contracts import ChatRequest


class HuggingFaceEndpointChatAdapter(BaseLangChainChatAdapter):
    provider_code = "huggingface_endpoint"
    adapter_meta = ProviderAdapterMeta(
        provider_code="huggingface_endpoint",
        env_key_name="huggingface_api_key",  # Requiere key para el endpoint
        required_packages=["langchain_huggingface"],
        is_local=False,
    )

    def _build_chat_model(self, request: ChatRequest) -> BaseChatModel:
        return self._factory.build_chat_model(request.model_key, request.config, self.adapter_meta)