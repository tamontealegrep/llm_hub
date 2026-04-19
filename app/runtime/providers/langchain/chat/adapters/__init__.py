# app/runtime/providers/langchain/chat/adapters/__init__.py
# Expone todos los adapters concretos para facilitar imports desde discovery.
from app.runtime.providers.langchain.chat.adapters.anthropic_adapter import AnthropicChatAdapter
from app.runtime.providers.langchain.chat.adapters.google_adapter import GoogleChatAdapter
from app.runtime.providers.langchain.chat.adapters.huggingface_endpoint_adapter import HuggingFaceEndpointChatAdapter
from app.runtime.providers.langchain.chat.adapters.huggingface_pipeline_adapter import HuggingFacePipelineChatAdapter
from app.runtime.providers.langchain.chat.adapters.ollama_adapter import OllamaChatAdapter
from app.runtime.providers.langchain.chat.adapters.openai_adapter import OpenAIChatAdapter
from app.runtime.providers.langchain.chat.adapters.xai_adapter import XAIChatAdapter

__all__ = [
    "AnthropicChatAdapter",
    "GoogleChatAdapter",
    "HuggingFaceEndpointChatAdapter",
    "HuggingFacePipelineChatAdapter",
    "OllamaChatAdapter",
    "OpenAIChatAdapter",
    "XAIChatAdapter",
]