from __future__ import annotations

from meal_orchestrator.providers.base import ProviderAdapter
from meal_orchestrator.providers.example_provider.client import ExampleProviderAdapter


def build_provider_adapter(provider_id: str) -> ProviderAdapter:
    if provider_id == ExampleProviderAdapter.provider_id:
        return ExampleProviderAdapter()
    raise ValueError(f"unsupported provider: {provider_id}")
