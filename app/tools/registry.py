from app.shared.exceptions import ToolNotFoundError
from app.tools.contracts import ToolDefinition
from app.tools.interfaces import ToolHandler


class ToolRegistry:
    """
    Registro de herramientas disponibles para tool calling.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, handler: ToolHandler) -> None:
        tool_name = handler.definition.name
        if tool_name in self._handlers:
            raise ValueError(f"La tool '{tool_name}' ya está registrada")
        self._handlers[tool_name] = handler

    def get(self, tool_name: str) -> ToolHandler:
        try:
            return self._handlers[tool_name]
        except KeyError as exc:
            raise ToolNotFoundError(tool_name) from exc

    def has(self, tool_name: str) -> bool:
        return tool_name in self._handlers

    def available_tool_names(self) -> list[str]:
        return sorted(self._handlers.keys())

    def list_definitions(self, tool_names: list[str] | None = None) -> list[ToolDefinition]:
        if tool_names is None:
            return [
                self._handlers[name].definition
                for name in self.available_tool_names()
            ]

        if not tool_names:
            return []

        definitions: list[ToolDefinition] = []
        seen: set[str] = set()

        for tool_name in tool_names:
            if tool_name in seen:
                continue
            definitions.append(self.get(tool_name).definition)
            seen.add(tool_name)

        return definitions