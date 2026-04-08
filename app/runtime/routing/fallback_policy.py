from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FallbackPolicy:
    enabled: bool = False
    provider_codes: list[str] = field(default_factory=list)