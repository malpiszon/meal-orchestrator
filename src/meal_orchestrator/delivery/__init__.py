from __future__ import annotations

from typing import Any, Protocol

from meal_orchestrator.delivery.discord import DiscordWebhookClient
from meal_orchestrator.domain import DiscordMessage, EmailMessage


class EmailClient(Protocol):
    def send(self, message: EmailMessage, idempotency_key: str) -> None: ...


class DiscordClient(Protocol):
    def notify(self, message: DiscordMessage) -> None: ...


class MoWebClient(Protocol):
    dashboard_url: str

    def send(self, payload: dict[str, Any]) -> None: ...


def build_discord_client() -> DiscordWebhookClient:
    return DiscordWebhookClient()
