from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.shared.settings.env import EnvSettings

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Temperature = Annotated[float, Field(ge=0.0, le=2.0)]
TopP = Annotated[float, Field(gt=0.0, le=1.0)]


class AppDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: Temperature = 0.2
    top_p: TopP = 1.0
    max_output_tokens: PositiveInt = 2048
    timeout_seconds: PositiveInt = 60
    max_tool_iterations: NonNegativeInt = 8


class FeatureFlags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chat_enabled: bool = True
    image_generation_enabled: bool = False
    agents_enabled: bool = False
    embeddings_enabled: bool = False


class PathSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_catalog_dir: str = "catalog/models"
    pricing_catalog_dir: str = "catalog/pricing"
    prompts_catalog_dir: str = "catalog/prompts"


class RoutingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_chat_provider_priority: list[str] = Field(
        default_factory=lambda: ["openai", "anthropic", "google", "xai"]
    )

    @model_validator(mode="after")
    def validate_priority(self) -> "RoutingSettings":
        if not self.default_chat_provider_priority:
            raise ValueError("default_chat_provider_priority no puede estar vacío")

        if len(self.default_chat_provider_priority) != len(
            set(self.default_chat_provider_priority)
        ):
            raise ValueError(
                "default_chat_provider_priority no puede contener duplicados"
            )

        return self


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defaults: AppDefaults = Field(default_factory=AppDefaults)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    paths: PathSettings = Field(default_factory=PathSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env: EnvSettings
    app: AppSettings