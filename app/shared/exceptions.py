class AppError(Exception):
    def __init__(
        self,
        message: str,
        *,
        error_code: str = "app_error",
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ProviderError(AppError):
    def __init__(
        self,
        message: str = "Error del proveedor",
        *,
        details: dict | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code="provider_error",
            details=details
        )


class ProviderConfigurationError(ProviderError):
    def __init__(
        self,
        message: str = "Configuración del proveedor incompleta"
    ) -> None:
        super().__init__(
            message,
            details={"type": "configuration"}
        )


class ProviderAuthenticationError(ProviderError):
    def __init__(
        self,
        message: str = "Error de autenticación con el proveedor"
    ) -> None:
        super().__init__(
            message,
            details={"type": "authentication"}
        )


class ProviderTimeoutError(ProviderError):
    def __init__(
        self,
        message: str = "Timeout del proveedor"
    ) -> None:
        super().__init__(
            message,
            details={"type": "timeout"}
        )


class ProviderRateLimitError(ProviderError):
    def __init__(
        self,
        message: str = "Rate limit del proveedor"
    ) -> None:
        super().__init__(
            message,
            details={"type": "rate_limit"}
        )


class ProviderUnavailableError(ProviderError):
    def __init__(
        self,
        message: str = "Proveedor no disponible"
    ) -> None:
        super().__init__(
            message,
            details={"type": "unavailable"}
        )


class ConversationNotFoundError(AppError):
    def __init__(
        self,
        message: str = "Conversación no encontrada"
    ) -> None:
        super().__init__(
            message,
            error_code="conversation_not_found"
        )


class InvalidProviderSelectionError(AppError):
    def __init__(
        self,
        message: str = "Selección de proveedor/modelo inválida"
    ) -> None:
        super().__init__(
            message,
            error_code="invalid_provider_selection"
        )


class ToolNotFoundError(AppError):
    def __init__(
        self,
        tool_name: str
    ) -> None:
        super().__init__(
            f"Herramienta no encontrada: '{tool_name}'",
            error_code="tool_not_found",
            details={"tool_name": tool_name},
        )


class ToolLoopLimitExceededError(AppError):
    def __init__(
        self,
        max_iterations: int
    ) -> None:
        super().__init__(
            f"Se alcanzó el máximo de iteraciones de tool calling ({max_iterations})",
            error_code="tool_loop_limit_exceeded",
            details={"max_iterations": max_iterations},
        )


class ConfigurationError(AppError):
    def __init__(
        self,
        message: str = "Configuración inválida",
        *,
        details: dict | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code="configuration_error",
            details=details,
        )


class InvalidModelParameterValueError(AppError):
    def __init__(
        self,
        provider_code: str,
        model_key: str,
        parameter_name: str,
        value: object,
        reason: str,
    ) -> None:
        super().__init__(
            (
                f"Valor inválido para '{parameter_name}' en "
                f"'{provider_code}/{model_key}': {value!r}. {reason}"
            ),
            error_code="invalid_model_parameter_value",
            details={
                "provider_code": provider_code,
                "model_key": model_key,
                "parameter_name": parameter_name,
                "value": value,
                "reason": reason,
            },
        )

        
class ModelCatalogError(AppError):
    def __init__(
        self,
        message: str = "Error en el catálogo de modelos",
        *,
        details: dict | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code="model_catalog_error",
            details=details,
        )


class ModelNotInCatalogError(AppError):
    def __init__(self, provider_code: str, model_key: str) -> None:
        super().__init__(
            f"El modelo '{provider_code}/{model_key}' no existe en el catálogo",
            error_code="model_not_in_catalog",
            details={
                "provider_code": provider_code,
                "model_key": model_key,
            },
        )


class UnsupportedModelCapabilityError(AppError):
    def __init__(
        self,
        provider_code: str,
        model_key: str,
        capability: str,
    ) -> None:
        super().__init__(
            (
                f"El modelo '{provider_code}/{model_key}' no soporta "
                f"la capability '{capability}'"
            ),
            error_code="unsupported_model_capability",
            details={
                "provider_code": provider_code,
                "model_key": model_key,
                "capability": capability,
            },
        )


class UnsupportedModelParameterError(AppError):
    def __init__(
        self,
        provider_code: str,
        model_key: str,
        parameter_name: str,
    ) -> None:
        super().__init__(
            (
                f"El modelo '{provider_code}/{model_key}' no soporta "
                f"el parámetro '{parameter_name}'"
            ),
            error_code="unsupported_model_parameter",
            details={
                "provider_code": provider_code,
                "model_key": model_key,
                "parameter_name": parameter_name,
            },
        )