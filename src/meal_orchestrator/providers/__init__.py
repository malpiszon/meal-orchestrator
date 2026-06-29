from __future__ import annotations

from typing import Protocol

from meal_orchestrator.domain import ProviderMenuRequest, ProviderResult


class MenuUnavailableError(RuntimeError):
    pass


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
