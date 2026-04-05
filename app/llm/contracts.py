from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class NormalizedMessage:
    """
    Mensaje normalizado que representa un turno en la conversación.

    Roles soportados:
    - "system"    → instrucciones del sistema
    - "user"      → mensaje del usuario
    - "assistant" → respuesta del modelo
    - "tool"      → resultado de una herramienta (requiere tool_call_id)

    Validación temprana: si role="tool" y no hay tool_call_id, falla en construcción
    en lugar de propagar el error hasta _to_langchain_messages.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None  # requerido cuando role == "tool"
    name: str | None = None          # nombre de la herramienta (para trazabilidad)

    def __post_init__(self) -> None:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError(
                "NormalizedMessage con role='tool' requiere tool_call_id. "
                f"Contenido recibido: {self.content!r}"
            )


@dataclass(frozen=True, slots=True)
class LLMRequestConfig:
    temperature: float = 0.2
    top_p: float = 1.0
    max_output_tokens: int = 2048
    timeout_seconds: int = 60


@dataclass(frozen=True, slots=True)
class LLMRequest:
    provider_code: str
    model_key: str
    messages: list[NormalizedMessage]
    config: LLMRequestConfig
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class LLMCompletionResult:
    content: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    provider_request_id: str | None = None
    raw_response: dict[str, Any] | None = None


class StreamEventType(str, Enum):
    START = "start"
    DELTA = "delta"
    END = "end"
    ERROR = "error"


@dataclass(slots=True)
class LLMStreamEvent:
    type: StreamEventType
    delta: str | None = None
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    provider_request_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    raw_event: dict[str, Any] | None = None
