from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from app.shared.exceptions import ConfigurationError
from app.shared.settings.env import EnvSettings, get_env_settings
from app.shared.settings.models import AppSettings, RuntimeSettings


def _read_yaml_file(path: Path, *, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ConfigurationError(
                f"No se encontró el archivo de configuración: {path}",
                details={"path": str(path)},
            )
        return {}

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"YAML inválido en {path}",
            details={"path": str(path)},
        ) from exc

    if raw is None:
        return {}

    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"El archivo {path} debe contener un mapping YAML en la raíz",
            details={"path": str(path)},
        )

    return raw


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def _resolve_override_path(env_settings: EnvSettings) -> Path:
    if env_settings.app_config_file:
        return Path(env_settings.app_config_file)

    return Path("config") / f"{env_settings.app_env}.yaml"


def load_settings(env_settings: EnvSettings | None = None) -> RuntimeSettings:
    resolved_env = env_settings or get_env_settings()

    base_path = Path("config/base.yaml")
    override_path = _resolve_override_path(resolved_env)

    override_required = resolved_env.app_config_file is not None

    base_data = _read_yaml_file(base_path, required=True)
    override_data = _read_yaml_file(override_path, required=override_required)

    merged_data = _deep_merge(base_data, override_data)

    try:
        app_settings = AppSettings.model_validate(merged_data)
    except ValidationError as exc:
        raise ConfigurationError(
            "La configuración YAML no es válida",
            details={"errors": exc.errors()},
        ) from exc

    return RuntimeSettings(
        env=resolved_env,
        app=app_settings,
    )


@lru_cache
def get_settings() -> RuntimeSettings:
    return load_settings()


def clear_settings_cache() -> None:
    get_env_settings.cache_clear()
    get_settings.cache_clear()