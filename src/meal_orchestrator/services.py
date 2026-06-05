from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from meal_orchestrator.delivery import (
    DiscordClient,
    EmailClient,
    StubDiscordClient,
    StubEmailClient,
)
from meal_orchestrator.llm import OpenRouterClient
from meal_orchestrator.providers import ProviderAdapter, build_provider_adapter


@dataclass(frozen=True)
class AppServices:
    provider_factory: Callable[[str], ProviderAdapter]
    llm_client: OpenRouterClient
    email_client: EmailClient
    discord_client: DiscordClient


def build_stub_services() -> AppServices:
    return AppServices(
        provider_factory=build_provider_adapter,
        llm_client=OpenRouterClient(dry_run=True),
        email_client=StubEmailClient(dry_run=True),
        discord_client=StubDiscordClient(dry_run=True),
    )
