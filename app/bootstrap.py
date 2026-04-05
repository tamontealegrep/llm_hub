from dataclasses import dataclass

from app.conversation.context_builder import SimpleContextBuilder
from app.conversation.repository import InMemoryConversationRepository
from app.conversation.service import ConversationService
from app.core.config import Settings, get_settings
from app.llm.adapters.anthropic_adapter import AnthropicAdapter
from app.llm.adapters.gemini_adapter import GeminiAdapter
from app.llm.adapters.grok_adapter import GrokAdapter
from app.llm.adapters.openai_adapter import OpenAIAdapter
from app.llm.contracts import LLMRequestConfig
from app.llm.factory import LangChainChatModelFactory
from app.llm.orchestrator import LLMOrchestrator
from app.llm.registry import ProviderRegistry


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: Settings
    registry: ProviderRegistry
    service: ConversationService


def build_registry(settings: Settings) -> ProviderRegistry:
    factory = LangChainChatModelFactory(settings)
    registry = ProviderRegistry()

    if settings.openai_api_key:
        registry.register(OpenAIAdapter(factory))

    if settings.anthropic_api_key:
        registry.register(AnthropicAdapter(factory))

    if settings.google_api_key:
        registry.register(GeminiAdapter(factory))

    if settings.xai_api_key:
        registry.register(GrokAdapter(factory))

    return registry


def pick_default_provider_and_model(
    settings: Settings,
    registry: ProviderRegistry,
) -> tuple[str, str]:
    if registry.has("openai"):
        return "openai", settings.default_openai_model

    if registry.has("anthropic"):
        return "anthropic", settings.default_anthropic_model

    if registry.has("google"):
        return "google", settings.default_google_model

    if registry.has("xai"):
        return "xai", settings.default_xai_model

    raise RuntimeError("No hay proveedores configurados")


def build_container(
    *,
    settings: Settings | None = None,
    max_context_messages: int | None = None,
    default_config: LLMRequestConfig | None = None,
) -> AppContainer:
    resolved_settings = settings or get_settings()
    registry = build_registry(resolved_settings)

    if not registry.available_provider_codes():
        raise RuntimeError(
            "No hay proveedores configurados. Revisa tu .env y las API keys."
        )

    resolved_default_config = default_config
    if resolved_default_config is None:
        resolved_default_config = LLMRequestConfig(
            temperature=resolved_settings.default_temperature,
            top_p=resolved_settings.default_top_p,
            max_output_tokens=resolved_settings.default_max_output_tokens,
            timeout_seconds=resolved_settings.default_timeout_seconds,
        )

    repository = InMemoryConversationRepository()
    context_builder = SimpleContextBuilder(max_messages=max_context_messages)
    orchestrator = LLMOrchestrator(registry)

    service = ConversationService(
        repository=repository,
        context_builder=context_builder,
        orchestrator=orchestrator,
        provider_registry=registry,
        default_config=resolved_default_config,
    )

    return AppContainer(
        settings=resolved_settings,
        registry=registry,
        service=service,
    )