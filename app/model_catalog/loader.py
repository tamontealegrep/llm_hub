from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from app.model_catalog.schemas import (
    AliasCatalogSchema,
    ModelCatalogBundle,
    ProviderCatalogSchema,
)
from app.shared.exceptions import ModelCatalogError


def _iter_yaml_files(directory: Path) -> list[Path]:
    files = list(directory.glob("*.yaml")) + list(directory.glob("*.yml"))
    unique_files = {file.resolve(): file for file in files}
    return sorted(unique_files.values())


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ModelCatalogError(
            f"YAML inválido en catálogo: {path}",
            details={"path": str(path)},
        ) from exc

    if raw is None:
        return {}

    if not isinstance(raw, dict):
        raise ModelCatalogError(
            f"El archivo {path} debe contener un mapping YAML en la raíz",
            details={"path": str(path)},
        )

    return raw


def load_model_catalog(directory: str | Path) -> ModelCatalogBundle:
    catalog_dir = Path(directory)

    if not catalog_dir.exists() or not catalog_dir.is_dir():
        raise ModelCatalogError(
            "El directorio del catálogo de modelos no existe",
            details={"path": str(catalog_dir)},
        )

    yaml_files = _iter_yaml_files(catalog_dir)
    if not yaml_files:
        raise ModelCatalogError(
            "No se encontraron archivos YAML en el catálogo de modelos",
            details={"path": str(catalog_dir)},
        )

    providers: list[ProviderCatalogSchema] = []
    aliases: dict[str, Any] = {}
    seen_provider_codes: set[str] = set()

    for file_path in yaml_files:
        raw_data = _read_yaml_mapping(file_path)

        if file_path.name in {"aliases.yaml", "aliases.yml"}:
            try:
                alias_catalog = AliasCatalogSchema.model_validate(raw_data)
            except ValidationError as exc:
                raise ModelCatalogError(
                    "El archivo de aliases es inválido",
                    details={
                        "path": str(file_path),
                        "errors": exc.errors(),
                    },
                ) from exc

            aliases = alias_catalog.aliases
            continue

        try:
            provider_catalog = ProviderCatalogSchema.model_validate(raw_data)
        except ValidationError as exc:
            raise ModelCatalogError(
                "Archivo de catálogo de provider inválido",
                details={
                    "path": str(file_path),
                    "errors": exc.errors(),
                },
            ) from exc

        if provider_catalog.provider_code in seen_provider_codes:
            raise ModelCatalogError(
                f"provider_code duplicado en catálogo: '{provider_catalog.provider_code}'",
                details={
                    "path": str(file_path),
                    "provider_code": provider_catalog.provider_code,
                },
            )

        seen_provider_codes.add(provider_catalog.provider_code)
        providers.append(provider_catalog)

    bundle = ModelCatalogBundle(
        providers=providers,
        aliases=aliases,
    )

    _validate_alias_targets(bundle)
    return bundle


def _validate_alias_targets(bundle: ModelCatalogBundle) -> None:
    provider_map = {provider.provider_code: provider for provider in bundle.providers}
    model_map = {
        provider.provider_code: {model.key: model for model in provider.models}
        for provider in bundle.providers
    }

    for alias_name, target in bundle.aliases.items():
        if target.provider_code not in provider_map:
            raise ModelCatalogError(
                f"El alias '{alias_name}' apunta a un provider inexistente",
                details={
                    "alias": alias_name,
                    "provider_code": target.provider_code,
                },
            )

        if target.model_key not in model_map[target.provider_code]:
            raise ModelCatalogError(
                f"El alias '{alias_name}' apunta a un modelo inexistente",
                details={
                    "alias": alias_name,
                    "provider_code": target.provider_code,
                    "model_key": target.model_key,
                },
            )

        if not model_map[target.provider_code][target.model_key].capabilities.has(
            target.capability
        ):
            raise ModelCatalogError(
                f"El alias '{alias_name}' apunta a un modelo sin la capability requerida",
                details={
                    "alias": alias_name,
                    "provider_code": target.provider_code,
                    "model_key": target.model_key,
                    "capability": target.capability,
                },
            )
