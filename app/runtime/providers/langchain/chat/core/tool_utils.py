# app/runtime/providers/langchain/chat/core/tool_utils.py
from __future__ import annotations
import copy
import json
import logging
from typing import Any
from uuid import uuid4

from app.tools.contracts import ToolCall, ToolDefinition

logger = logging.getLogger(__name__)


def clean_schema_for_gemini(schema: Any) -> None:
    """
    Elimina recursivamente 'additionalProperties' del esquema,
    ya que Gemini no lo soporta.
    """
    if isinstance(schema, dict):
        schema.pop("additionalProperties", None)
        for value in schema.values():
            clean_schema_for_gemini(value)
    elif isinstance(schema, list):
        for item in schema:
            clean_schema_for_gemini(item)


def to_langchain_tool_schema(tool: ToolDefinition, *, provider_code: str) -> dict[str, Any]:
    parameters = copy.deepcopy(tool.input_schema)

    if provider_code == "google":
        clean_schema_for_gemini(parameters)

    if "type" not in parameters:
        parameters["type"] = "object"

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


def normalize_tool_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"raw": arguments}
    return {"value": arguments}


def extract_tool_calls(response: Any, *, provider_code: str) -> list[ToolCall]:
    """Extrae tool_calls de una respuesta AIMessage completa."""
    raw_tool_calls = getattr(response, "tool_calls", None)

    if not raw_tool_calls:
        additional_kwargs = getattr(response, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict):
            raw_tool_calls = additional_kwargs.get("tool_calls")

    if not raw_tool_calls:
        return []

    parsed: list[ToolCall] = []

    for raw_call in raw_tool_calls:
        call_id: Any = None
        name: Any = None
        arguments: Any = None

        if isinstance(raw_call, dict):
            function_payload = raw_call.get("function")
            call_id = raw_call.get("id")
            name = raw_call.get("name")
            if not name and isinstance(function_payload, dict):
                name = function_payload.get("name")
            arguments = raw_call.get("args")
            if arguments is None and isinstance(function_payload, dict):
                arguments = function_payload.get("arguments")
        else:
            call_id = getattr(raw_call, "id", None)
            name = getattr(raw_call, "name", None)
            arguments = getattr(raw_call, "args", None)

        if not name:
            logger.warning(
                "Se ignoró un tool_call sin nombre provider=%s raw=%r",
                provider_code,
                raw_call,
            )
            continue

        try:
            parsed.append(
                ToolCall(
                    id=str(call_id or f"toolcall_{uuid4().hex}"),
                    name=str(name),
                    arguments=normalize_tool_arguments(arguments),
                )
            )
        except ValueError as exc:
            logger.warning(
                "Tool call inválido provider=%s raw=%r error=%s",
                provider_code,
                raw_call,
                exc,
            )

    return parsed


def process_stream_chunk_for_tool_calls(
    chunk: Any,
    accumulated: dict[int, dict[str, str]],
) -> None:
    """
    Acumula tool_call_chunks de un chunk de streaming en `accumulated`.
    """
    tool_call_chunks = getattr(chunk, "tool_call_chunks", None) or []

    for tc_chunk in tool_call_chunks:
        if isinstance(tc_chunk, dict):
            idx = tc_chunk.get("index") or 0
            call_id = tc_chunk.get("id") or ""
            name = tc_chunk.get("name") or ""
            args = tc_chunk.get("args") or ""
        else:
            idx = getattr(tc_chunk, "index", 0) or 0
            call_id = getattr(tc_chunk, "id", "") or ""
            name = getattr(tc_chunk, "name", "") or ""
            args = getattr(tc_chunk, "args", "") or ""

        if idx not in accumulated:
            accumulated[idx] = {"id": "", "name": "", "args": ""}

        accumulated[idx]["id"] += call_id
        accumulated[idx]["name"] += name
        accumulated[idx]["args"] += args

    if tool_call_chunks:
        return  # ya procesado

    # Ruta de fallback: additional_kwargs.tool_calls (OpenAI legacy)
    additional_kwargs = getattr(chunk, "additional_kwargs", None)
    if not isinstance(additional_kwargs, dict):
        return

    raw_calls = additional_kwargs.get("tool_calls") or []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            continue

        idx = raw_call.get("index") or 0
        call_id = raw_call.get("id") or ""

        function_payload = raw_call.get("function") or {}
        name = function_payload.get("name") or ""
        args = function_payload.get("arguments") or ""

        if idx not in accumulated:
            accumulated[idx] = {"id": "", "name": "", "args": ""}

        accumulated[idx]["id"] += call_id
        accumulated[idx]["name"] += name
        accumulated[idx]["args"] += args


def assemble_tool_calls_from_chunks(
    accumulated: dict[int, dict[str, str]],
    *,
    provider_code: str,
) -> list[ToolCall]:
    """
    Convierte el dict acumulado de chunks de tool_calls en ToolCall completos.
    """
    result: list[ToolCall] = []

    for idx in sorted(accumulated.keys()):
        entry = accumulated[idx]
        name = entry.get("name", "").strip()
        if not name:
            logger.warning(
                "Se ignoró tool_call_chunk sin nombre provider=%s idx=%d",
                provider_code,
                idx,
            )
            continue

        raw_args = entry.get("args", "").strip()
        arguments = normalize_tool_arguments(raw_args or "{}")
        call_id = entry.get("id", "").strip() or f"toolcall_{uuid4().hex}"

        try:
            result.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=arguments,
                )
            )
        except ValueError as exc:
            logger.warning(
                "Tool call inválido tras ensamblado provider=%s idx=%d error=%s",
                provider_code,
                idx,
                exc,
            )

    return result