from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ToolDefinition.name no puede estar vacío")

        if not self.description.strip():
            raise ValueError(f"La tool '{self.name}' debe tener una descripción")

        if not isinstance(self.input_schema, dict):
            raise ValueError(
                f"La tool '{self.name}' requiere input_schema como dict JSON Schema"
            )

        schema_type = self.input_schema.get("type")
        if schema_type is not None and schema_type != "object":
            raise ValueError(
                f"La tool '{self.name}' requiere input_schema con type='object'"
            )


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("ToolCall.id no puede estar vacío")

        if not self.name.strip():
            raise ValueError("ToolCall.name no puede estar vacío")

        if not isinstance(self.arguments, dict):
            raise ValueError("ToolCall.arguments debe ser un dict")


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    conversation_id: str
    provider_code: str
    model_key: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False