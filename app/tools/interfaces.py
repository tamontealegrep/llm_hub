from abc import ABC, abstractmethod
from typing import Any, Awaitable

from app.tools.contracts import ToolDefinition, ToolExecutionContext


class ToolHandler(ABC):
    """
    Contrato base para una herramienta.

    execute() puede ser síncrono o async.
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition: ...

    @abstractmethod
    def execute(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> Any | Awaitable[Any]: ...