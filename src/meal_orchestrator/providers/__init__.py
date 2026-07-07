from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from meal_orchestrator.domain import ProviderMenuRequest, ProviderResult


class MenuUnavailableError(RuntimeError):
    """Raised when the provider menu isn't published yet.

    May carry raw_response when the failure occurs after raw data was
    fetched (e.g. a purchased size missing from an otherwise valid
    response), so callers can persist it for debugging.
    """

    def __init__(self, message: str, raw_response: Any = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class ProviderNormalizationError(Exception):
    """Raised when raw provider data cannot be normalized into the canonical schema.

    Carries raw_response so callers can persist it for debugging even when
    normalization fails before a ProviderResult can be returned.
    """

    def __init__(self, message: str, raw_response: Any = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class ProviderAdapter(ABC):
    provider_id: str

    def expected_variants_per_meal(self, meal_type: str) -> int | None:
        """Expected dish-variant count for a meal type, or None to skip the check.

        Providers whose feed contractually offers a fixed number of dish
        variants (one per diet plan) per meal type should override this.
        Defaults to no check, for providers that make no such guarantee.
        """
        return None

    @abstractmethod
    def get_canonical_week_menu(
        self,
        request: ProviderMenuRequest,
    ) -> ProviderResult: ...


def build_provider_adapter(provider_id: str) -> ProviderAdapter:
    if provider_id == "example_provider":
        from meal_orchestrator.providers.example_provider import ExampleProviderAdapter
        return ExampleProviderAdapter()
    if provider_id == "ntfy":
        from meal_orchestrator.providers.ntfy import NtfyProviderAdapter
        return NtfyProviderAdapter()
    raise ValueError(f"unsupported provider: {provider_id}")
