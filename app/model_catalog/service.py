from __future__ import annotations

from typing import Any

from app.model_catalog.repository import InMemoryModelCatalogRepository
from app.model_catalog.schemas import (
    AliasTargetSchema,
    CanonicalParameterSchema,
    ModelDefinitionSchema,
    ProviderCatalogSchema,
)
from app.shared.exceptions import (
    InvalidModelParameterValueError,
    ModelCatalogError,
    ModelNotInCatalogError,
    UnsupportedModelCapabilityError,
    UnsupportedModelParameterError,
)


class ModelCatalogService:
    def __init__(self, repository: InMemoryModelCatalogRepository) -> None:
        self._repository = repository

    def list_provider_codes(self) -> list[str]:
        return self._repository.list_provider_codes()

    def list_providers(self) -> list[ProviderCatalogSchema]:
        return self._repository.list_providers()

    def provider_exists(self, provider_code: str) -> bool:
        return self._repository.has_provider(provider_code)

    def model_exists(self, provider_code: str, model_key: str) -> bool:
        return self._repository.has_model(provider_code, model_key)

    def alias_exists(self, alias_name: str) -> bool:
        return self._repository.has_alias(alias_name)

    def get_provider(self, provider_code: str) -> ProviderCatalogSchema:
        return self._repository.get_provider(provider_code)

    def list_models(self, provider_code: str) -> list[ModelDefinitionSchema]:
        return self._repository.list_models(provider_code)

    def get_model(self, provider_code: str, model_key: str) -> ModelDefinitionSchema:
        return self._repository.get_model(provider_code, model_key)
    
    def require_model(
        self,
        provider_code: str,
        model_key: str,
    ) -> ModelDefinitionSchema:
        if not self.provider_exists(provider_code) or not self.model_exists(
            provider_code,
            model_key,
        ):
            raise ModelNotInCatalogError(provider_code, model_key)

        return self._repository.get_model(provider_code, model_key)

    def supports_capability(
        self,
        provider_code: str,
        model_key: str,
        capability: str,
    ) -> bool:
        model = self.get_model(provider_code, model_key)
        return model.capabilities.has(capability)
    
    def require_capability(
        self,
        provider_code: str,
        model_key: str,
        capability: str,
    ) -> None:
        model = self.require_model(provider_code, model_key)

        if not model.capabilities.has(capability):
            raise UnsupportedModelCapabilityError(
                provider_code=provider_code,
                model_key=model_key,
                capability=capability,
            )

    def get_default_chat_model(self, provider_code: str) -> str:
        default_model = self._repository.get_default_chat_model(provider_code)
        if not default_model:
            raise ModelCatalogError(
                f"El provider '{provider_code}' no tiene default_chat_model definido",
                details={"provider_code": provider_code},
            )
        return default_model

    def get_model_defaults(self, provider_code: str, model_key: str) -> dict[str, Any]:
        model = self.get_model(provider_code, model_key)
        return model.defaults.model_dump(exclude_none=True)

    def get_supported_param(
        self,
        provider_code: str,
        model_key: str,
        param_name: str,
    ) -> CanonicalParameterSchema | None:
        model = self.require_model(provider_code, model_key)
        return model.supported_params.get(param_name)
    
    def get_supported_params(
        self,
        provider_code: str,
        model_key: str,
    ) -> dict[str, CanonicalParameterSchema]:
        model = self.get_model(provider_code, model_key)
        return dict(model.supported_params)

    def supports_param(
        self,
        provider_code: str,
        model_key: str,
        param_name: str,
    ) -> bool:
        param = self.get_supported_param(provider_code, model_key, param_name)
        return bool(param and param.supported)
    
    def require_supported_param(
        self,
        provider_code: str,
        model_key: str,
        param_name: str,
    ) -> CanonicalParameterSchema:
        param = self.get_supported_param(provider_code, model_key, param_name)

        if param is None or not param.supported:
            raise UnsupportedModelParameterError(
                provider_code=provider_code,
                model_key=model_key,
                parameter_name=param_name,
            )

        return param
    
    def validate_param_value(
        self,
        provider_code: str,
        model_key: str,
        param_name: str,
        value: object,
    ) -> None:
        if value is None:
            return

        param = self.require_supported_param(provider_code, model_key, param_name)

        if param.allowed_values:
            if value not in param.allowed_values:
                raise InvalidModelParameterValueError(
                    provider_code=provider_code,
                    model_key=model_key,
                    parameter_name=param_name,
                    value=value,
                    reason=f"Debe ser uno de: {param.allowed_values!r}",
                )
            return

        has_numeric_bounds = param.min is not None or param.max is not None
        is_numeric = isinstance(value, (int, float)) and not isinstance(value, bool)

        if has_numeric_bounds and not is_numeric:
            raise InvalidModelParameterValueError(
                provider_code=provider_code,
                model_key=model_key,
                parameter_name=param_name,
                value=value,
                reason="Debe ser un valor numérico",
            )

        if param.min is not None and is_numeric and value < param.min:
            raise InvalidModelParameterValueError(
                provider_code=provider_code,
                model_key=model_key,
                parameter_name=param_name,
                value=value,
                reason=f"Debe ser >= {param.min}",
            )

        if param.max is not None and is_numeric and value > param.max:
            raise InvalidModelParameterValueError(
                provider_code=provider_code,
                model_key=model_key,
                parameter_name=param_name,
                value=value,
                reason=f"Debe ser <= {param.max}",
            )

    def resolve_alias(self, alias_name: str) -> AliasTargetSchema:
        return self._repository.get_alias(alias_name)

    def resolve_provider_and_model(
        self,
        provider_or_alias: str,
        model_key: str | None = None,
        *,
        capability: str = "chat",
    ) -> tuple[str, str]:
        # Caso 1: provider + model explícito
        if model_key is not None:
            if not self.model_exists(provider_or_alias, model_key):
                raise ModelCatalogError(
                    "El provider/modelo solicitado no existe en el catálogo",
                    details={
                        "provider_code": provider_or_alias,
                        "model_key": model_key,
                    },
                )

            if capability and not self.supports_capability(
                provider_or_alias,
                model_key,
                capability,
            ):
                raise ModelCatalogError(
                    "El modelo existe, pero no soporta la capability solicitada",
                    details={
                        "provider_code": provider_or_alias,
                        "model_key": model_key,
                        "capability": capability,
                    },
                )

            return provider_or_alias, model_key

        # Caso 2: provider sin model_key -> usa default chat model
        if self.provider_exists(provider_or_alias):
            if capability != "chat":
                raise ModelCatalogError(
                    "Resolver por provider sin model_key solo está soportado para capability='chat' por ahora",
                    details={
                        "provider_code": provider_or_alias,
                        "capability": capability,
                    },
                )

            return provider_or_alias, self.get_default_chat_model(provider_or_alias)

        # Caso 3: alias
        alias = self.resolve_alias(provider_or_alias)

        if capability and alias.capability != capability:
            raise ModelCatalogError(
                "El alias existe, pero no corresponde a la capability solicitada",
                details={
                    "alias": provider_or_alias,
                    "expected_capability": capability,
                    "alias_capability": alias.capability,
                },
            )

        return alias.provider_code, alias.model_key

    def pick_first_available_default_chat_model(
        self,
        *,
        available_provider_codes: list[str],
        provider_priority: list[str],
    ) -> tuple[str, str]:
        if not available_provider_codes:
            raise ModelCatalogError(
                "No hay providers disponibles para seleccionar un default"
            )

        ordered_codes: list[str] = []

        for provider_code in provider_priority:
            if provider_code in available_provider_codes and provider_code not in ordered_codes:
                ordered_codes.append(provider_code)

        for provider_code in available_provider_codes:
            if provider_code not in ordered_codes:
                ordered_codes.append(provider_code)

        for provider_code in ordered_codes:
            if not self.provider_exists(provider_code):
                raise ModelCatalogError(
                    f"El provider '{provider_code}' está disponible en runtime pero no existe en el catálogo",
                    details={"provider_code": provider_code},
                )

            default_model = self._repository.get_default_chat_model(provider_code)
            if default_model and self.supports_capability(
                provider_code,
                default_model,
                "chat",
            ):
                return provider_code, default_model

        raise ModelCatalogError(
            "Ningún provider disponible tiene un default_chat_model válido en el catálogo",
            details={
                "available_provider_codes": available_provider_codes,
                "provider_priority": provider_priority,
            },
        )