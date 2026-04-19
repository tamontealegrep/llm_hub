from __future__ import annotations
from typing import Callable, Any
from langchain_core.language_models.chat_models import BaseChatModel
from app.capabilities.chat.contracts import ChatRequestConfig


ModelBuilderFn = Callable[[str, ChatRequestConfig, str | None], BaseChatModel]


def build_openai(model_key: str, config: ChatRequestConfig, api_key: str | None) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model_key,
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        timeout=config.timeout_seconds,
        top_p=config.top_p,
        max_retries=2,
    )


def build_anthropic(model_key: str, config: ChatRequestConfig, api_key: str | None) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic
    top_p = config.top_p if config.top_p != 1.0 else None
    return ChatAnthropic(
        model=model_key,
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        top_p=top_p,
        timeout=config.timeout_seconds,
    )


def build_google(model_key: str, config: ChatRequestConfig, api_key: str | None) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model_key,
        google_api_key=api_key,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        top_p=config.top_p,
        convert_system_message_to_human=False,
    )


def build_xai(model_key: str, config: ChatRequestConfig, api_key: str | None) -> BaseChatModel:
    from langchain_xai import ChatXAI
    return ChatXAI(
        model=model_key,
        xai_api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        timeout=config.timeout_seconds,
        top_p=config.top_p,
    )


def build_ollama(model_key: str, config: ChatRequestConfig, api_key: str | None) -> BaseChatModel:
    """Builder para modelos locales vía Ollama."""
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=model_key,
        temperature=config.temperature,
    )


def build_huggingface_pipeline(model_key: str, config: ChatRequestConfig, api_key: str | None) -> BaseChatModel:
    """
    Builder para modelos locales vía HuggingFace + transformers.
    Carga el modelo en memoria local usando un pipeline de HuggingFace.
    """
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_key)
    model = AutoModelForCausalLM.from_pretrained(
        model_key,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.max_output_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    return HuggingFacePipeline(pipeline=pipe)


def build_huggingface_endpoint(model_key: str, config: ChatRequestConfig, api_key: str | None) -> BaseChatModel:
    """
    Builder para modelos HuggingFace via Inference Endpoint (requiere HF_API_KEY).
    """
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    endpoint = HuggingFaceEndpoint(
        repo_id=model_key,
        huggingfacehub_api_token=api_key,
        max_new_tokens=config.max_output_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    return ChatHuggingFace(llm=endpoint)


BUILDER_REGISTRY: dict[str, ModelBuilderFn] = {
    "openai": build_openai,
    "anthropic": build_anthropic,
    "google": build_google,
    "xai": build_xai,
    "ollama": build_ollama,
    "huggingface_pipeline": build_huggingface_pipeline,
    "huggingface_endpoint": build_huggingface_endpoint,
}