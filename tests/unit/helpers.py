from __future__ import annotations

from datetime import date
from pathlib import Path

from meal_orchestrator.config import AppConfig, UserConfig
from meal_orchestrator.config.models import DeliveryConfig, LlmConfig, RuntimeConfig
from meal_orchestrator.domain import (
    CanonicalDay,
    CanonicalMeal,
    CanonicalMenu,
    DeliveryResult,
    DiscordMessage,
    EmailMessage,
    MealVariant,
    PurchasedMeal,
)


def app_config() -> AppConfig:
    return AppConfig(
        runtime=RuntimeConfig(timezone="Europe/Warsaw"),
        llm=LlmConfig(
            provider="openrouter",
            model="test-model",
            timeout_seconds=30,
            max_retries=1,
        ),
        default_provider="example_provider",
        delivery=DeliveryConfig(
            email_from="Meal Orchestrator <meals@example.com>",
            operational_discord_webhook_env="DISCORD_OPS_WEBHOOK_URL",
        ),
    )


def user_config(prompt_file: Path = Path("prompt.md")) -> UserConfig:
    return UserConfig(
        id="alan",
        enabled=True,
        provider="example_provider",
        provider_offering_id=123,
        email="alan@example.com",
        discord_user_id="123",
        discord_webhook_env="DISCORD_ALAN_WEBHOOK_URL",
        prompt_file=prompt_file,
        purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
    )


def canonical_menu(*, complete: bool = True) -> CanonicalMenu:
    days = [
        CanonicalDay(
            date=date(2026, 6, day),
            meals=[
                CanonicalMeal(
                    type="breakfast",
                    variants=[MealVariant(name="Meal", composition="Ingredients")],
                )
            ],
        )
        for day in range(1, 6 if complete else 2)
    ]
    return CanonicalMenu(
        provider="example_provider",
        week_start=date(2026, 6, 1),
        week_end=date(2026, 6, 5),
        user_id="alan",
        days=days,
    )


class FakeEmailClient:
    def __init__(self) -> None:
        self.messages: list[EmailMessage] = []
        self.idempotency_keys: list[str] = []

    def send(self, message: EmailMessage, idempotency_key: str) -> DeliveryResult:
        self.messages.append(message)
        self.idempotency_keys.append(idempotency_key)
        return DeliveryResult(success=True)


class FakeDiscordClient:
    def __init__(self) -> None:
        self.messages: list[DiscordMessage] = []

    def notify(self, message: DiscordMessage) -> DeliveryResult:
        self.messages.append(message)
        return DeliveryResult(success=True)
