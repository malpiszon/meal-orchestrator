from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from meal_orchestrator.domain import PurchasedMeal


@dataclass(frozen=True)
class RuntimeConfig:
    timezone: str
    max_concurrent_users: int = 5


@dataclass(frozen=True)
class BatchConfig:
    enabled: bool = False
    state_dir: Path | None = None
    initial_poll_interval_seconds: int = 120
    max_poll_interval_seconds: int = 3600
    max_wait_hours: int = 26


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    model: str
    timeout_seconds: int
    max_retries: int
    dry_run_model: str | None = None
    fallback_models: list[str] = field(default_factory=list)
    batch: BatchConfig = field(default_factory=BatchConfig)


@dataclass(frozen=True)
class DeliveryConfig:
    email_from: str
    operational_discord_webhook_env: str | None


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
    discord_user_id: str | None
    discord_webhook_env: str | None
    prompt_file: Path
    purchased_meals: list[PurchasedMeal]
