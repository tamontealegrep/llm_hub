from app.runtime.providers.langchain.chat.base import BaseLangChainChatAdapter
from app.capabilities.chat.contracts import ChatRequest


class GeminiAdapter(BaseLangChainChatAdapter):
    """
    Adapter para Google Gemini vía langchain-google-genai.

    El workaround que fusionaba SystemMessage dentro del primer HumanMessage
    fue eliminado. ChatGoogleGenerativeAI >= 1.x soporta SystemMessage
    nativamente cuando convert_system_message_to_human=False (configurado
    en LangChainChatModelFactory.build_gemini).

    Si necesitas compatibilidad con versiones antiguas de langchain-google-genai
    (< 1.0), restaura el método _prepare_messages de la versión anterior y
    documenta la versión mínima requerida en los comentarios.
    """

    provider_code = "google"

    def _build_chat_model(self, request: ChatRequest):
        return self._factory.build_gemini(request.model_key, request.config)
