from app.runtime.routing.fallback_policy import FallbackPolicy
from app.runtime.routing.retry_policy import RetryPolicy


class Router:
    def __init__(
        self,
        *,
        retry_policy: RetryPolicy | None = None,
        fallback_policy: FallbackPolicy | None = None,
    ) -> None:
        self._retry_policy = retry_policy or RetryPolicy()
        self._fallback_policy = fallback_policy or FallbackPolicy()

    @property
    def retry_policy(self) -> RetryPolicy:
        return self._retry_policy

    @property
    def fallback_policy(self) -> FallbackPolicy:
        return self._fallback_policy

    def get_candidate_provider_codes(self, primary_provider_code: str) -> list[str]:
        candidates = [primary_provider_code]

        if self._fallback_policy.enabled:
            for provider_code in self._fallback_policy.provider_codes:
                if provider_code not in candidates:
                    candidates.append(provider_code)

        return candidates