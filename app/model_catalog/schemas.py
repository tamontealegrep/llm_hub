from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CanonicalParameterSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supported: bool = True
    default: Any | None = None
    min: int | float | None = None
    max: int | float | None = None
    allowed_values: list[Any] = Field(default_factory=list)
    description: str | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "CanonicalParameterSchema":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min no puede ser mayor que max")
        return self


class CapabilityFlagsSchema(BaseModel):
    """
    Flags de capability del modelo.

    Se deja extra='allow' para poder añadir más flags en el futuro
    sin romper la validación de YAMLs existentes.
    """

    model_config = ConfigDict(extra="allow")

    chat: bool = False
    image_generation: bool = False
    agents: bool = False
    embeddings: bool = False
    tools: bool = False
    streaming: bool = False
    structured_output: bool = False
    vision_input: bool = False

    def has(self, capability: str) -> bool:
        if hasattr(self, capability):
            return bool(getattr(self, capability))

        extra = getattr(self, "__pydantic_extra__", None) or {}
        return bool(extra.get(capability, False))


class ModelDefaultsSchema(BaseModel):
    """
    Defaults efectivos del modelo dentro de tu plataforma.

    extra='allow' para soportar defaults nuevos en el futuro
    sin rehacer el schema en cada iteración.
    """

    model_config = ConfigDict(extra="allow")

    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    timeout_seconds: int | None = None


class ModelDefinitionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    display_name: str
    description: str | None = None
    capabilities: CapabilityFlagsSchema = Field(default_factory=CapabilityFlagsSchema)
    defaults: ModelDefaultsSchema = Field(default_factory=ModelDefaultsSchema)
    supported_params: dict[str, CanonicalParameterSchema] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_definition(self) -> "ModelDefinitionSchema":
        if not self.key.strip():
            raise ValueError("model.key no puede estar vacío")

        if not self.display_name.strip():
            raise ValueError("model.display_name no puede estar vacío")

        if len(self.tags) != len(set(self.tags)):
            raise ValueError(f"El modelo '{self.key}' no puede tener tags duplicados")

        return self


class ProviderCatalogSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_code: str
    display_name: str
    default_chat_model: str | None = None
    models: list[ModelDefinitionSchema] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_catalog(self) -> "ProviderCatalogSchema":
        if not self.provider_code.strip():
            raise ValueError("provider_code no puede estar vacío")

        if not self.display_name.strip():
            raise ValueError("display_name no puede estar vacío")

        if not self.models:
            raise ValueError(
                f"El provider '{self.provider_code}' debe definir al menos un modelo"
            )

        model_keys = [model.key for model in self.models]
        if len(model_keys) != len(set(model_keys)):
            raise ValueError(
                f"El provider '{self.provider_code}' tiene model keys duplicadas"
            )

        if self.default_chat_model is not None:
            model_map = {model.key: model for model in self.models}

            if self.default_chat_model not in model_map:
                raise ValueError(
                    f"default_chat_model='{self.default_chat_model}' no existe "
                    f"en el provider '{self.provider_code}'"
                )

            if not model_map[self.default_chat_model].capabilities.has("chat"):
                raise ValueError(
                    f"default_chat_model='{self.default_chat_model}' debe tener "
                    "capabilities.chat=true"
                )

        return self


class AliasTargetSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_code: str
    model_key: str
    capability: str = "chat"

    @model_validator(mode="after")
    def validate_alias_target(self) -> "AliasTargetSchema":
        if not self.provider_code.strip():
            raise ValueError("provider_code no puede estar vacío")

        if not self.model_key.strip():
            raise ValueError("model_key no puede estar vacío")

        if not self.capability.strip():
            raise ValueError("capability no puede estar vacío")

        return self


class AliasCatalogSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aliases: dict[str, AliasTargetSchema] = Field(default_factory=dict)


class ModelCatalogBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: list[ProviderCatalogSchema] = Field(default_factory=list)
    aliases: dict[str, AliasTargetSchema] = Field(default_factory=dict)