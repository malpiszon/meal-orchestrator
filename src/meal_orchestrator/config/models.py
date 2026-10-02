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
    initial_check_delay_seconds: int = 15
    initial_poll_interval_seconds: int = 120
    max_poll_interval_seconds: int = 3600
    max_wait_hours: int = 26


# OpenRouter's Auto Router slug. It picks a concrete model per request, so it
# has no `:batch` endpoint and can't be used with batch mode.
AUTO_ROUTER_MODEL = "openrouter/auto"
AUTO_ROUTER_COST_TIERS = ("low", "medium", "high", "xhigh", "max")
REASONING_EFFORTS = ("minimal", "low", "medium", "high")


@dataclass(frozen=True)
class AutoRouterConfig:
    """Optional `auto-router` plugin settings, sent only for AUTO_ROUTER_MODEL."""

    cost_tier: str | None = None
    allowed_models: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    model: str
    timeout_seconds: int
    max_retries: int
    dry_run_model: str | None = None
    fallback_models: list[str] = field(default_factory=list)
    batch: BatchConfig = field(default_factory=BatchConfig)
    auto_router: AutoRouterConfig = field(default_factory=AutoRouterConfig)
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class MoWebDeliveryConfig:
    url: str
    token_env: str
    timeout_seconds: int = 10


@dataclass(frozen=True)
class DeliveryConfig:
    email_from: str
    operational_discord_webhook_env: str | None
    mo_web: MoWebDeliveryConfig | None = None


@dataclass(frozen=True)
class ArtifactConfig:
    path: Path
    retention_days: int
    max_runs: int


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
