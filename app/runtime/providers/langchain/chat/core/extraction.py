# app/runtime/providers/langchain/chat/core/extraction.py
from __future__ import annotations
from typing import Any

from app.capabilities.common.models import TokenUsage


def extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def extract_usage(response: Any) -> TokenUsage | None:
    prompt: int | None = None
    completion: int | None = None
    total: int | None = None

    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        prompt = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens")
        completion = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens")
        total = usage_metadata.get("total_tokens")
    else:
        response_metadata = getattr(response, "response_metadata", None)
        if isinstance(response_metadata, dict):
            token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
            if isinstance(token_usage, dict):
                prompt = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
                completion = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
                total = token_usage.get("total_tokens")

    if prompt is None and completion is None:
        return None

    if total is None:
        total = (prompt or 0) + (completion or 0)

    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )