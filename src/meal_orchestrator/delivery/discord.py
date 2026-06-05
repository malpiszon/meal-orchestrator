from __future__ import annotations

from meal_orchestrator.domain import DeliveryResult, DiscordMessage


class StubDiscordClient:
    def __init__(self, *, dry_run: bool = True) -> None:
        self.dry_run = dry_run

    def notify(self, message: DiscordMessage) -> DeliveryResult:
        if self.dry_run:
            return DeliveryResult(
                success=True,
                provider_message_id=f"dry-run-discord:{message.webhook_env}",
                detail="discord message skipped in dry-run",
            )
        raise NotImplementedError("real Discord transport is intentionally not implemented yet")
