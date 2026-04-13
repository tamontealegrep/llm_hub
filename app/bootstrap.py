# app/bootstrap.py
from dataclasses import dataclass

from app.conversations.context.message_window import MessageWindowContextBuilder
from app.conversations.repository import InMemoryConversationRepository
from app.capabilities.chat.service import ChatService
from app.capabilities.chat.contracts import ChatRequestConfig
from app.model_catalog.loader import load_model_catalog
from app.model_catalog.repository import InMemoryModelCatalogRepository
from app.model_catalog.service import ModelCatalogService
from app.runtime.providers.langchain.chat.anthropic_adapter import AnthropicChatAdapter
from app.runtime.providers.langchain.chat.google_adapter import GoogleChatAdapter
from app.runtime.providers.langchain.chat.xai_adapter import XAIChatAdapter
from app.runtime.providers.langchain.chat.openai_adapter import OpenAIChatAdapter
from app.runtime.providers.langchain.factory import LangChainProviderFactory
from app.runtime.providers.registry import ProviderRegistry
from app.runtime.execution.chat_orchestrator import ChatOrchestrator
from app.shared.settings.loader import RuntimeSettings, get_settings
from app.shared.settings.env import EnvSettings
from app.tools.builtin import register_builtin_tools
from app.tools.executor import ToolExecutor
from app.tools.loader import load_custom_tools
from app.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: RuntimeSettings
    model_catalog: ModelCatalogService
    provider_registry: ProviderRegistry
    tool_registry: ToolRegistry
    chat_service: ChatService
    default_chat_provider_code: str
    default_chat_model_key: str


def build_model_catalog(settings: RuntimeSettings) -> ModelCatalogService:
    bundle = load_model_catalog(settings.app.paths.model_catalog_dir)
    repository = InMemoryModelCatalogRepository(bundle)
    return ModelCatalogService(repository)


def build_chat_provider_registry(
    env_settings: EnvSettings,
    model_catalog: ModelCatalogService,
) -> ProviderRegistry:
    factory = LangChainProviderFactory(env_settings)
    provider_registry = ProviderRegistry()

    if env_settings.openai_api_key:
        provider_registry.register(OpenAIChatAdapter(factory, model_catalog))

    if env_settings.anthropic_api_key:
        provider_registry.register(AnthropicChatAdapter(factory, model_catalog))

    if env_settings.google_api_key:
        provider_registry.register(GoogleChatAdapter(factory, model_catalog))

    if env_settings.xai_api_key:
        provider_registry.register(XAIChatAdapter(factory, model_catalog))

    return provider_registry


def validate_provider_registry_against_catalog(
    provider_registry: ProviderRegistry,
    model_catalog: ModelCatalogService,
) -> None:
    for provider_code in provider_registry.available_provider_codes():
        if not model_catalog.provider_exists(provider_code):
            raise RuntimeError(
                f"El provider '{provider_code}' tiene credenciales configuradas, "
                f"pero no existe en catalog/models."
            )


def pick_default_chat_provider_and_model(
    settings: RuntimeSettings,
    model_catalog: ModelCatalogService,
    provider_registry: ProviderRegistry,
) -> tuple[str, str]:
    return model_catalog.pick_first_available_default_chat_model(
        available_provider_codes=provider_registry.available_provider_codes(),
        provider_priority=settings.app.routing.default_chat_provider_priority,
    )


def build_container(
    *,
    settings: RuntimeSettings | None = None,
    max_context_messages: int | None = None,
    default_chat_config: ChatRequestConfig | None = None,
    with_builtin_tools: bool = False,
    with_custom_tools: bool = True,
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
    resolved_settings = settings or get_settings()

    if not resolved_settings.app.features.chat_enabled:
        raise RuntimeError("La capability de chat está deshabilitada en configuración")
    
    model_catalog = build_model_catalog(resolved_settings)
    provider_registry = build_chat_provider_registry(
        resolved_settings.env,
        model_catalog,
    )

    if not provider_registry.available_provider_codes():
        raise RuntimeError(
            "No hay proveedores configurados. Revisa tu .env y las API keys."
        )
    
    validate_provider_registry_against_catalog(provider_registry, model_catalog)

    default_provider_code, default_model_key = pick_default_chat_provider_and_model(
        resolved_settings,
        model_catalog,
        provider_registry,
    )

    resolved_default_chat_config = default_chat_config
    if resolved_default_chat_config is None:
        resolved_default_chat_config = ChatRequestConfig(
            temperature=resolved_settings.app.defaults.temperature,
            top_p=resolved_settings.app.defaults.top_p,
            max_output_tokens=resolved_settings.app.defaults.max_output_tokens,
            timeout_seconds=resolved_settings.app.defaults.timeout_seconds,
        )

    repository = InMemoryConversationRepository()
    context_builder = MessageWindowContextBuilder(
        max_messages=max_context_messages
    )
    chat_orchestrator = ChatOrchestrator(provider_registry)

    tool_registry = ToolRegistry()

    # Orden de registro: builtin primero, custom después.
    # Si una tool builtin y una custom tienen el mismo nombre,
    # la custom NO sobreescribe (el registry lanza ValueError en duplicados).
    if with_builtin_tools:
        register_builtin_tools(tool_registry)

    if with_custom_tools:
        load_custom_tools(tool_registry)

    tool_executor = ToolExecutor(tool_registry)

    chat_service = ChatService(
        repository=repository,
        context_builder=context_builder,
        orchestrator=chat_orchestrator,
        provider_registry=provider_registry,
        model_catalog=model_catalog,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        default_config=resolved_default_chat_config,
        max_tool_iterations=resolved_settings.app.defaults.max_tool_iterations,
    )

    return AppContainer(
        settings=resolved_settings,
        model_catalog=model_catalog,
        provider_registry=provider_registry,
        tool_registry=tool_registry,
        chat_service=chat_service,
        default_chat_provider_code=default_provider_code,
        default_chat_model_key=default_model_key,
    )