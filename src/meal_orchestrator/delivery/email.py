from __future__ import annotations

from meal_orchestrator.domain import DeliveryResult, EmailMessage


class StubEmailClient:
    def __init__(self, *, dry_run: bool = True) -> None:
        self.dry_run = dry_run

    def send(self, message: EmailMessage, idempotency_key: str) -> DeliveryResult:
        if self.dry_run:
            return DeliveryResult(
                success=True,
                provider_message_id=f"dry-run-email:{idempotency_key}",
                detail=f"email to {message.to} skipped in dry-run",
            )
        raise NotImplementedError("real Resend transport is intentionally not implemented yet")
