from __future__ import annotations

from typing import Any, Protocol

from meal_orchestrator.domain import ProviderMenuRequest, ProviderResult


class MenuUnavailableError(RuntimeError):
    pass


class ProviderNormalizationError(Exception):
    """Raised when raw provider data cannot be normalized into the canonical schema.

    Carries raw_response so callers can persist it for debugging even when
    normalization fails before a ProviderResult can be returned.
    """

    def __init__(self, message: str, raw_response: Any = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class ProviderAdapter(Protocol):
    provider_id: str

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
