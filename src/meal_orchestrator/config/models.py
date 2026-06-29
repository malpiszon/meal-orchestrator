from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from meal_orchestrator.domain import PurchasedMeal


@dataclass(frozen=True)
class RuntimeConfig:
    timezone: str


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    model: str
    timeout_seconds: int
    max_retries: int


@dataclass(frozen=True)
class DeliveryConfig:
    email_from: str
    operational_discord_webhook_env: str


@dataclass(frozen=True)
class ArtifactConfig:
    path: Path
    retention_days: int
    max_runs_per_user: int


@dataclass(frozen=True)
class AppConfig:
    runtime: RuntimeConfig
    llm: LlmConfig
    default_provider: str
    delivery: DeliveryConfig
    artifacts: ArtifactConfig | None = None


@dataclass(frozen=True)
class UserConfig:
    id: str
    enabled: bool
    provider: str
    provider_offering_id: int | str
    email: str
    discord_user_id: str
    discord_webhook_env: str
    prompt_file: Path
    purchased_meals: list[PurchasedMeal]
