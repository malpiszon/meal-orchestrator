from __future__ import annotations

from meal_orchestrator.delivery.email import ResendEmailClient
from meal_orchestrator.domain import DiscordMessage

EmailClient = ResendEmailClient


class DiscordClient:
    def notify(self, message: DiscordMessage) -> None:
        raise NotImplementedError("real Discord transport is intentionally not implemented yet")
