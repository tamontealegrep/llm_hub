import inspect
import json
import logging
from typing import Any

from app.shared.exceptions import ToolNotFoundError
from app.tools.contracts import ToolCall, ToolExecutionContext, ToolExecutionResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Ejecuta una ToolCall usando el ToolRegistry.
    Serializa automáticamente la salida a string para poder enviarla
    al modelo como mensaje role='tool'.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute_call(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        try:
            handler = self._registry.get(call.name)
        except ToolNotFoundError:
            logger.warning("Tool no registrada: %s", call.name)
            return ToolExecutionResult(
                tool_call_id=call.id,
                name=call.name,
                content=self._serialize_output(
                    {"error": f"La tool '{call.name}' no está registrada"}
                ),
                is_error=True,
            )

        try:
            raw_output = handler.execute(call.arguments, context)
            if inspect.isawaitable(raw_output):
                raw_output = await raw_output

            if isinstance(raw_output, ToolExecutionResult):
                return raw_output

            return ToolExecutionResult(
                tool_call_id=call.id,
                name=call.name,
                content=self._serialize_output(raw_output),
                is_error=False,
            )
        except Exception as exc:
            logger.exception("Error ejecutando tool '%s'", call.name)
            return ToolExecutionResult(
                tool_call_id=call.id,
                name=call.name,
                content=self._serialize_output(
                    {
                        "error": str(exc),
                        "tool": call.name,
                        "arguments": call.arguments,
                    }
                ),
                is_error=True,
            )

    def _serialize_output(self, value: Any) -> str:
        if value is None:
            return "null"

        if isinstance(value, str):
            return value

        return json.dumps(value, ensure_ascii=False, default=str)