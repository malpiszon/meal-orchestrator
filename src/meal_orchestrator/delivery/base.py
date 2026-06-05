from __future__ import annotations

from typing import Protocol

from meal_orchestrator.domain import DeliveryResult, DiscordMessage, EmailMessage


class EmailClient(Protocol):
    def send(self, message: EmailMessage, idempotency_key: str) -> DeliveryResult: ...


class DiscordClient(Protocol):
    def notify(self, message: DiscordMessage) -> DeliveryResult: ...
