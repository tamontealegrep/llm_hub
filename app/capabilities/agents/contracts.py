from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class AgentRunRequest:
    agent_key: str
    input_text: str
    conversation_id: str | None = None
    provider_code: str | None = None
    model_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentStepResult:
    name: str
    status: Literal["started", "completed", "failed"]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    output_text: str
    final_state: dict[str, Any] = field(default_factory=dict)
    steps: list[AgentStepResult] = field(default_factory=list)