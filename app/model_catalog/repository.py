from __future__ import annotations

from app.model_catalog.schemas import (
    AliasTargetSchema,
    ModelCatalogBundle,
    ModelDefinitionSchema,
    ProviderCatalogSchema,
)
from app.shared.exceptions import ModelCatalogError


class InMemoryModelCatalogRepository:
    def __init__(self, bundle: ModelCatalogBundle) -> None:
        self._bundle = bundle
        self._providers = {provider.provider_code: provider for provider in bundle.providers}
        self._models = {
            provider.provider_code: {model.key: model for model in provider.models}
            for provider in bundle.providers
        }
        self._aliases = dict(bundle.aliases)

    def list_provider_codes(self) -> list[str]:
        return sorted(self._providers.keys())

    def list_providers(self) -> list[ProviderCatalogSchema]:
        return [self._providers[code] for code in self.list_provider_codes()]

    def has_provider(self, provider_code: str) -> bool:
        return provider_code in self._providers

    def get_provider(self, provider_code: str) -> ProviderCatalogSchema:
        try:
            return self._providers[provider_code]
        except KeyError as exc:
            raise ModelCatalogError(
                f"Provider no encontrado en catálogo: '{provider_code}'",
                details={"provider_code": provider_code},
            ) from exc

    def list_models(self, provider_code: str) -> list[ModelDefinitionSchema]:
        provider = self.get_provider(provider_code)
        return list(provider.models)

    def has_model(self, provider_code: str, model_key: str) -> bool:
        return (
            provider_code in self._models
            and model_key in self._models[provider_code]
        )

    def get_model(self, provider_code: str, model_key: str) -> ModelDefinitionSchema:
        self.get_provider(provider_code)

        try:
            return self._models[provider_code][model_key]
        except KeyError as exc:
            raise ModelCatalogError(
                f"Modelo no encontrado en catálogo: '{provider_code}/{model_key}'",
                details={
                    "provider_code": provider_code,
                    "model_key": model_key,
                },
            ) from exc

    def get_default_chat_model(self, provider_code: str) -> str | None:
        provider = self.get_provider(provider_code)
        return provider.default_chat_model

    def has_alias(self, alias_name: str) -> bool:
        return alias_name in self._aliases

    def list_aliases(self) -> dict[str, AliasTargetSchema]:
        return dict(self._aliases)

    def get_alias(self, alias_name: str) -> AliasTargetSchema:
        try:
            return self._aliases[alias_name]
        except KeyError as exc:
            raise ModelCatalogError(
                f"Alias de modelo no encontrado: '{alias_name}'",
                details={"alias": alias_name},
            ) from exc