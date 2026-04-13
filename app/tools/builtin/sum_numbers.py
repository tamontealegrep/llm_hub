from typing import Any

from app.tools.contracts import ToolDefinition, ToolExecutionContext
from app.tools.interfaces import ToolHandler


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


def register(registry) -> None:
    if not registry.has("sum_numbers"):
        registry.register(SumNumbersTool())