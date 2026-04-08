# app/bootstrap.py
from dataclasses import dataclass

from app.conversations.context.message_window import MessageWindowContextBuilder
from app.conversations.repository import InMemoryConversationRepository
from app.capabilities.chat.service import ConversationService
from app.capabilities.chat.contracts import ChatRequestConfig
from app.runtime.providers.langchain.chat.anthropic_adapter import AnthropicAdapter
from app.runtime.providers.langchain.chat.gemini_adapter import GeminiAdapter
from app.runtime.providers.langchain.chat.grok_adapter import GrokAdapter
from app.runtime.providers.langchain.chat.openai_adapter import OpenAIAdapter
from app.runtime.providers.langchain.factory import LangChainChatModelFactory
from app.runtime.providers.registry import ProviderRegistry
from app.runtime.execution.chat_executor import LLMOrchestrator
from app.shared.settings.env import EnvSettings, get_env_settings
from app.tools.builtin import register_builtin_tools
from app.tools.executor import ToolExecutor
from app.tools.loader import load_custom_tools
from app.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: EnvSettings
    registry: ProviderRegistry
    tool_registry: ToolRegistry
    service: ConversationService


def build_registry(settings: EnvSettings) -> ProviderRegistry:
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
    settings: EnvSettings,
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
    settings: EnvSettings | None = None,
    max_context_messages: int | None = None,
    default_config: ChatRequestConfig | None = None,
    with_builtin_tools: bool = False,
    with_custom_tools: bool = True,              # ← NUEVO (activo por defecto)
) -> AppContainer:
    """
    Construye el contenedor principal de la aplicación.

    Parámetros de tools:
        with_builtin_tools:
            Registra las tools integradas (get_current_utc_time, sum_numbers).
            Default: False (para no romper comportamiento anterior).

        with_custom_tools:
            Descubre y registra automáticamente todas las tools en
            app/tools/custom/. Default: True.
            Pasa False en tests unitarios donde no quieras cargar tools externas.
    """
    resolved_settings = settings or get_env_settings()
    registry = build_registry(resolved_settings)

    if not registry.available_provider_codes():
        raise RuntimeError(
            "No hay proveedores configurados. Revisa tu .env y las API keys."
        )

    resolved_default_config = default_config
    if resolved_default_config is None:
        resolved_default_config = ChatRequestConfig(
            temperature=resolved_settings.default_temperature,
            top_p=resolved_settings.default_top_p,
            max_output_tokens=resolved_settings.default_max_output_tokens,
            timeout_seconds=resolved_settings.default_timeout_seconds,
        )

    repository = InMemoryConversationRepository()
    context_builder = MessageWindowContextBuilder(max_messages=max_context_messages)
    orchestrator = LLMOrchestrator(registry)

    tool_registry = ToolRegistry()

    # Orden de registro: builtin primero, custom después.
    # Si una tool builtin y una custom tienen el mismo nombre,
    # la custom NO sobreescribe (el registry lanza ValueError en duplicados).
    if with_builtin_tools:
        register_builtin_tools(tool_registry)

    if with_custom_tools:
        load_custom_tools(tool_registry)

    tool_executor = ToolExecutor(tool_registry)

    service = ConversationService(
        repository=repository,
        context_builder=context_builder,
        orchestrator=orchestrator,
        provider_registry=registry,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        default_config=resolved_default_config,
        max_tool_iterations=resolved_settings.default_max_tool_iterations,
    )

    return AppContainer(
        settings=resolved_settings,
        registry=registry,
        tool_registry=tool_registry,
        service=service,
    )