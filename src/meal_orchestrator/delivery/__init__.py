from __future__ import annotations

from meal_orchestrator.domain import DeliveryResult, DiscordMessage, EmailMessage


class DiscordClient:
    def notify(self, message: DiscordMessage) -> DeliveryResult:
        raise NotImplementedError("real Discord transport is intentionally not implemented yet")


class EmailClient:
    def send(self, message: EmailMessage, idempotency_key: str) -> DeliveryResult:
        raise NotImplementedError("real Resend transport is intentionally not implemented yet")
