from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal
 
from app.tools.contracts import ToolCall, ToolDefinition
from app.capabilities.common.models import TokenUsage
 
 
@dataclass(frozen=True, slots=True)
class ChatMessage:
    """
    Mensaje normalizado que representa un turno en la conversación.
 
    Roles soportados:
    - "system"    → instrucciones del sistema
    - "user"      → mensaje del usuario
    - "assistant" → respuesta del modelo
    - "tool"      → resultado de una herramienta (requiere tool_call_id)
 
    Además, un mensaje "assistant" puede incluir tool_calls cuando el modelo
    solicita la ejecución de una o varias herramientas.
    """
 
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
 
    def __post_init__(self) -> None:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError(
                "ChatMessage con role='tool' requiere tool_call_id. "
                f"Contenido recibido: {self.content!r}"
            )
 
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError(
                "Solo los mensajes con role='tool' pueden incluir tool_call_id. "
                f"Role recibido: {self.role!r}"
            )
 
        if self.tool_calls and self.role != "assistant":
            raise ValueError(
                "Solo los mensajes con role='assistant' pueden incluir tool_calls. "
                f"Role recibido: {self.role!r}"
            )
 
        if any(not isinstance(tc, ToolCall) for tc in self.tool_calls):
            raise ValueError("tool_calls debe contener instancias de ToolCall")
 
 
@dataclass(frozen=True, slots=True)
class ChatRequestConfig:
    temperature: float = 0.2
    top_p: float = 1.0
    max_output_tokens: int = 2048
    timeout_seconds: int = 60

    def to_param_dict(self) -> dict[str, object]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
        }
    
 
@dataclass(frozen=True, slots=True)
class ChatRequest:
    provider_code: str
    model_key: str
    messages: list[ChatMessage]
    config: ChatRequestConfig
    metadata: dict[str, Any] = field(default_factory=dict)
    tools: list[ToolDefinition] = field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
 
 
@dataclass(frozen=True, slots=True)
class ChatCompletionResult:
    content: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    provider_request_id: str | None = None
    raw_response: dict[str, Any] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
 
    def __post_init__(self) -> None:
        if any(not isinstance(tc, ToolCall) for tc in self.tool_calls):
            raise ValueError("tool_calls debe contener instancias de ToolCall")
 
 
class ChatStreamEventType(str, Enum):
    START = "start"
    DELTA = "delta"
    TOOL_USE = "tool_use"
    END = "end"
    ERROR = "error"
 
 
@dataclass(slots=True)
class ChatStreamEvent:
    type: ChatStreamEventType
    delta: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    provider_request_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    raw_event: dict[str, Any] | None = None