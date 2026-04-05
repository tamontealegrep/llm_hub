from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4


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
    created_at: datetime = field(default_factory=utcnow)


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
    ) -> ConversationMessage:
        message = ConversationMessage(
            role=ConversationRole.ASSISTANT,
            content=content,
            provider_code=provider_code or self.current_provider_code,
            model_key=model_key or self.current_model_key,
        )
        self.messages.append(message)
        self.touch()
        return message