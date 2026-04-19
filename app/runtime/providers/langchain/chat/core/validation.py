# app/runtime/providers/langchain/chat/core/validation.py
from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.capabilities.chat.contracts import ChatRequest
    from app.model_catalog.service import ModelCatalogService


def iter_request_params(request: "ChatRequest") -> dict[str, Any]:
    params = dict(request.config.to_param_dict())
    if request.tool_choice is not None:
        params["tool_choice"] = request.tool_choice
    return params


def validate_request_config_against_catalog(
    request: "ChatRequest",
    model_catalog: "ModelCatalogService",
) -> None:
    for param_name, value in iter_request_params(request).items():
        model_catalog.validate_param_value(
            request.provider_code,
            request.model_key,
            param_name,
            value,
        )


def validate_request_against_catalog(
    request: "ChatRequest",
    model_catalog: "ModelCatalogService",
    *,
    streaming: bool,
) -> None:
    model_catalog.require_model(request.provider_code, request.model_key)
    model_catalog.require_capability(
        request.provider_code,
        request.model_key,
        "chat",
    )

    if request.tools:
        model_catalog.require_capability(
            request.provider_code,
            request.model_key,
            "tools",
        )

    if streaming:
        model_catalog.require_capability(
            request.provider_code,
            request.model_key,
            "streaming",
        )

    validate_request_config_against_catalog(request, model_catalog)