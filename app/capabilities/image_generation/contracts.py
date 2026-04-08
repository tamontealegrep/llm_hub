from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None
    media_type: str | None = None


@dataclass(frozen=True, slots=True)
class ImageGenerationRequestConfig:
    size: str = "1024x1024"
    quality: str | None = None
    n: int = 1
    timeout_seconds: int = 120


@dataclass(frozen=True, slots=True)
class ImageGenerationRequest:
    provider_code: str
    model_key: str
    prompt: str
    config: ImageGenerationRequestConfig = field(
        default_factory=ImageGenerationRequestConfig
    )
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImageGenerationResult:
    images: list[GeneratedImage]
    provider_request_id: str | None = None
    raw_response: dict[str, Any] | None = None