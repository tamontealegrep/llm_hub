from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from app.tools.contracts import ToolCall


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConversationRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class ConversationMessage:
    role: ConversationRole
    content: str
    provider_code: str | None = None
    model_key: str | None = None
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if self.role == ConversationRole.TOOL and not self.tool_call_id:
            raise ValueError(
                "ConversationMessage con role=TOOL requiere tool_call_id"
            )

        if self.role != ConversationRole.TOOL and self.tool_call_id is not None:
            raise ValueError(
                "Solo los mensajes TOOL pueden incluir tool_call_id"
            )

        if self.tool_calls and self.role != ConversationRole.ASSISTANT:
            raise ValueError(
                "Solo los mensajes ASSISTANT pueden incluir tool_calls"
            )

        if any(not isinstance(tool_call, ToolCall) for tool_call in self.tool_calls):
            raise ValueError("tool_calls debe contener instancias de ToolCall")


@dataclass(slots=True)
class ConversationSession:
    current_provider_code: str
    current_model_key: str
    system_prompt: str | None = None

    id: UUID = field(default_factory=uuid4)
    messages: list[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def touch(self) -> None:
        self.updated_at = utcnow()

    def switch_model(
        self,
        *,
        provider_code: str | None = None,
        model_key: str | None = None,
    ) -> None:
        if provider_code is not None:
            self.current_provider_code = provider_code
        if model_key is not None:
            self.current_model_key = model_key
        self.touch()

    def add_user_message(self, content: str) -> ConversationMessage:
        message = ConversationMessage(
            role=ConversationRole.USER,
            content=content,
        )
        self.messages.append(message)
        self.touch()
        return message

    def add_assistant_message(
        self,
        content: str,
        *,
        provider_code: str | None = None,
        model_key: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> ConversationMessage:
        message = ConversationMessage(
            role=ConversationRole.ASSISTANT,
            content=content,
            provider_code=provider_code or self.current_provider_code,
            model_key=model_key or self.current_model_key,
            tool_calls=list(tool_calls or []),
        )
        self.messages.append(message)
        self.touch()
        return message

    def add_tool_message(
        self,
        content: str,
        *,
        tool_call_id: str,
        name: str | None = None,
    ) -> ConversationMessage:
        message = ConversationMessage(
            role=ConversationRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )
        self.messages.append(message)
        self.touch()
        return message