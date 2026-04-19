"""
Adapter para modelos locales cargados en memoria vía HuggingFace + transformers.

Requisito: pip install langchain-huggingface transformers torch
           (o torch-cpu si no tienes GPU)

Uso:
  1. Instala las dependencias.
  2. En catalog/models/huggingface_pipeline.yaml define los modelos.
     El `key` del modelo debe ser el repo_id de HuggingFace
     (ej: "meta-llama/Llama-3.2-1B" o "microsoft/phi-2").
  3. Los modelos se descargan automáticamente en ~/.cache/huggingface al primer uso.
  4. No se necesita API key (solo para modelos públicos).

Ventajas: máximo control, sin dependencias externas en runtime.
Limitaciones: requiere GPU/RAM suficiente, descarga inicial lenta.

Cuándo usar este en vez de Ollama:
  - Necesitas acceso a modelos muy nuevos antes de que lleguen a Ollama.
  - Necesitas hacer fine-tuning o modificar el modelo.
  - Necesitas cuantización personalizada.
"""
from langchain_core.language_models.chat_models import BaseChatModel

from app.runtime.providers.langchain.chat.base import BaseLangChainChatAdapter, ProviderAdapterMeta
from app.capabilities.chat.contracts import ChatRequest


class HuggingFacePipelineChatAdapter(BaseLangChainChatAdapter):
    provider_code = "huggingface_pipeline"
    adapter_meta = ProviderAdapterMeta(
        provider_code="huggingface_pipeline",
        env_key_name=None,   # No necesita key para modelos públicos
        required_packages=["langchain_huggingface", "transformers", "torch"],
        is_local=True,
    )

    def _build_chat_model(self, request: ChatRequest) -> BaseChatModel:
        return self._factory.build_chat_model(request.model_key, request.config, self.adapter_meta)