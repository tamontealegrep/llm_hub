from datetime import datetime, timezone
from typing import Any

from app.tools.contracts import ToolDefinition, ToolExecutionContext
from app.tools.interfaces import ToolHandler


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


def register(registry) -> None:
    if not registry.has("get_current_utc_time"):
        registry.register(GetCurrentUtcTimeTool())