from __future__ import annotations

from typing import Protocol

from meal_orchestrator.domain import DiscordMessage, EmailMessage


class EmailClient(Protocol):
    def send(self, message: EmailMessage, idempotency_key: str) -> None: ...


class DiscordClient:
    def notify(self, message: DiscordMessage) -> None:
        raise NotImplementedError("real Discord transport is intentionally not implemented yet")
