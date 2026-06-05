from __future__ import annotations

import logging

from meal_orchestrator.domain import DeliveryResult, DiscordMessage, EmailMessage

logger = logging.getLogger(__name__)


class DiscordClient:
    def __init__(self, *, dry_run: bool = True) -> None:
        self.dry_run = dry_run

    def notify(self, message: DiscordMessage) -> DeliveryResult:
        if self.dry_run:
            logger.info("discord notification skipped in dry-run: %s", message.content)
            return DeliveryResult(
                success=True,
                provider_message_id=f"dry-run-discord:{message.webhook_env}",
                detail="discord message skipped in dry-run",
            )
        raise NotImplementedError("real Discord transport is intentionally not implemented yet")


class EmailClient:
    def __init__(self, *, dry_run: bool = True) -> None:
        self.dry_run = dry_run

    def send(self, message: EmailMessage, idempotency_key: str) -> DeliveryResult:
        if self.dry_run:
            logger.info("email delivery skipped in dry-run to %s", message.to)
            return DeliveryResult(
                success=True,
                provider_message_id=f"dry-run-email:{idempotency_key}",
                detail=f"email to {message.to} skipped in dry-run",
            )
        raise NotImplementedError("real Resend transport is intentionally not implemented yet")
