from app.runtime.providers.base import ChatProviderAdapter


class ProviderRegistry:
    """
    Registro de adapters por provider_code.
    """

    def __init__(self) -> None:
        self._providers: dict[str, ChatProviderAdapter] = {}

    def register(self, adapter: ChatProviderAdapter) -> None:
        if adapter.provider_code in self._providers:
            raise ValueError(f"El provider '{adapter.provider_code}' ya está registrado")
        self._providers[adapter.provider_code] = adapter

    def get(self, provider_code: str) -> ChatProviderAdapter:
        try:
            return self._providers[provider_code]
        except KeyError as exc:
            raise ValueError(f"No existe adapter para provider '{provider_code}'") from exc

    def has(self, provider_code: str) -> bool:
        return provider_code in self._providers

    def available_provider_codes(self) -> list[str]:
        return sorted(self._providers.keys())