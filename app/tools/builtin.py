from datetime import datetime, timezone
from typing import Any

from app.tools.contracts import ToolDefinition, ToolExecutionContext
from app.tools.interfaces import ToolHandler
from app.tools.registry import ToolRegistry


class GetCurrentUtcTimeTool(ToolHandler):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_current_utc_time",
            description="Obtiene la fecha y hora actual en formato UTC ISO 8601.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )

    def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> Any:
        _ = arguments
        _ = context

        return {
            "utc_iso": datetime.now(timezone.utc).isoformat(),
        }


class SumNumbersTool(ToolHandler):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="sum_numbers",
            description="Suma una lista de números y devuelve el total.",
            input_schema={
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "description": "Lista de números a sumar",
                        "items": {"type": "number"},
                    }
                },
                "required": ["numbers"],
                "additionalProperties": False,
            },
        )

    def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> Any:
        _ = context

        numbers = arguments.get("numbers")
        if not isinstance(numbers, list):
            raise ValueError("Debes enviar 'numbers' como un array de números")

        normalized: list[int | float] = []
        for item in numbers:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("Todos los elementos de 'numbers' deben ser numéricos")
            normalized.append(item)

        return {
            "result": sum(normalized),
        }


def register_builtin_tools(registry: ToolRegistry) -> None:
    if not registry.has("get_current_utc_time"):
        registry.register(GetCurrentUtcTimeTool())

    if not registry.has("sum_numbers"):
        registry.register(SumNumbersTool())