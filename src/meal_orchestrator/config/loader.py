from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from meal_orchestrator.config.models import (
    AppConfig,
    ArtifactConfig,
    BatchConfig,
    DeliveryConfig,
    LlmConfig,
    RuntimeConfig,
    UserConfig,
)
from meal_orchestrator.domain import PurchasedMeal


class ConfigError(ValueError):
    pass


def load_app_config(path: Path) -> AppConfig:
    data = _load_yaml(path)
    llm_model = _required(data, "llm", "model")
    return AppConfig(
        runtime=RuntimeConfig(
            timezone=_required(data, "runtime", "timezone"),
            max_concurrent_users=_parse_max_concurrent_users(
                _optional(data, "runtime", "max_concurrent_users")
            ),
        ),
        llm=LlmConfig(
            provider=_required(data, "llm", "provider"),
            model=llm_model,
            timeout_seconds=int(_required(data, "llm", "timeout_seconds")),
            max_retries=int(_required(data, "llm", "max_retries")),
            dry_run_model=_optional(data, "llm", "dry_run_model"),
            fallback_models=_parse_fallback_models(
                _optional(data, "llm", "fallback_models"), llm_model
            ),
            batch=_parse_batch(data),
        ),
        default_provider=_required(data, "providers", "default"),
        delivery=DeliveryConfig(
            email_from=_required(data, "delivery", "email_from"),
            operational_discord_webhook_env=_optional(
                data, "delivery", "operational_discord_webhook_env"
            ),
        ),
        artifacts=_parse_artifacts(data),
    )


def _parse_fallback_models(raw: Any, model: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ConfigError("llm.fallback_models must be a list of strings")
    if model in raw:
        raise ConfigError(f"llm.fallback_models must not include the primary model: {model}")
    return raw


def _parse_max_concurrent_users(raw: Any) -> int:
    if raw is None:
        return RuntimeConfig.max_concurrent_users
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ConfigError("runtime.max_concurrent_users must be a positive integer")
    return raw


def _parse_batch(data: dict[str, Any]) -> BatchConfig:
    raw = data.get("llm", {}).get("batch") if isinstance(data.get("llm"), dict) else None
    if raw is None:
        return BatchConfig()
    if not isinstance(raw, dict):
        raise ConfigError("llm.batch must be a mapping")
    defaults = BatchConfig()
    enabled = raw.get("enabled", defaults.enabled)
    if not isinstance(enabled, bool):
        raise ConfigError("llm.batch.enabled must be a boolean")
    state_dir_raw = raw.get("state_dir")
    if enabled and state_dir_raw is None:
        raise ConfigError("llm.batch.state_dir is required when llm.batch.enabled is true")
    state_dir = Path(str(state_dir_raw)) if state_dir_raw is not None else None
    initial_check_delay_seconds = _non_negative_int(
        raw.get("initial_check_delay_seconds", defaults.initial_check_delay_seconds),
        "llm.batch.initial_check_delay_seconds",
    )
    initial_poll_interval_seconds = _positive_int(
        raw.get("initial_poll_interval_seconds", defaults.initial_poll_interval_seconds),
        "llm.batch.initial_poll_interval_seconds",
    )
    max_poll_interval_seconds = _positive_int(
        raw.get("max_poll_interval_seconds", defaults.max_poll_interval_seconds),
        "llm.batch.max_poll_interval_seconds",
    )
    if initial_poll_interval_seconds > max_poll_interval_seconds:
        raise ConfigError(
            "llm.batch.initial_poll_interval_seconds must not exceed "
            "llm.batch.max_poll_interval_seconds"
        )
    return BatchConfig(
        enabled=enabled,
        state_dir=state_dir,
        initial_check_delay_seconds=initial_check_delay_seconds,
        initial_poll_interval_seconds=initial_poll_interval_seconds,
        max_poll_interval_seconds=max_poll_interval_seconds,
        max_wait_hours=_positive_int(
            raw.get("max_wait_hours", defaults.max_wait_hours), "llm.batch.max_wait_hours"
        ),
    )


def _positive_int(value: Any, field_name: str) -> int:
    return _int_at_least(value, field_name, minimum=1)


def _non_negative_int(value: Any, field_name: str) -> int:
    return _int_at_least(value, field_name, minimum=0)


def _int_at_least(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        kind = "positive" if minimum == 1 else "non-negative"
        raise ConfigError(f"{field_name} must be a {kind} integer")
    return value


def _parse_artifacts(data: dict[str, Any]) -> ArtifactConfig | None:
    raw = data.get("artifacts")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("artifacts must be a mapping")
    enabled = raw.get("enabled")
    if not isinstance(enabled, bool):
        raise ConfigError("artifacts.enabled must be a boolean")
    if not enabled:
        return None
    path_raw = raw.get("path")
    if path_raw is None:
        raise ConfigError("artifacts.path is required when artifacts are enabled")
    retention_days = raw.get("retention_days")
    if not isinstance(retention_days, int) or retention_days < 1:
        raise ConfigError("artifacts.retention_days must be a positive integer")
    max_runs = raw.get("max_runs_per_user")
    if not isinstance(max_runs, int) or max_runs < 1:
        raise ConfigError("artifacts.max_runs_per_user must be a positive integer")
    return ArtifactConfig(
        path=Path(str(path_raw)),
        retention_days=retention_days,
        max_runs_per_user=max_runs,
    )


def load_users_config(path: Path) -> list[UserConfig]:
    data = _load_yaml(path)
    users = data.get("users")
    if not isinstance(users, list):
        raise ConfigError("users.yaml must contain a users list")

    parsed_users = [_parse_user(raw_user) for raw_user in users]
    user_ids = [user.id for user in parsed_users]
    duplicate_ids = {user_id for user_id in user_ids if user_ids.count(user_id) > 1}
    if duplicate_ids:
        raise ConfigError(f"duplicate user ids: {', '.join(sorted(duplicate_ids))}")
    return parsed_users


def _parse_user(raw_user: dict[str, Any]) -> UserConfig:
    purchased_meals = raw_user.get("purchased_meals")
    if not isinstance(purchased_meals, list) or not purchased_meals:
        raise ConfigError(f"user {raw_user.get('id', '<unknown>')} must define purchased_meals")

    return UserConfig(
        id=str(_field(raw_user, "id")),
        enabled=_parse_bool(_field(raw_user, "enabled"), "enabled"),
        provider=str(_field(raw_user, "provider")),
        provider_offering_id=_parse_offering_id(_field(raw_user, "provider_offering_id")),
        email=str(_field(raw_user, "email")),
        discord_user_id=(
            str(raw_user["discord_user_id"])
            if raw_user.get("discord_user_id") is not None
            else None
        ),
        discord_webhook_env=(
            str(raw_user["discord_webhook_env"])
            if raw_user.get("discord_webhook_env") is not None
            else None
        ),
        prompt_file=Path(_field(raw_user, "prompt_file")),
        purchased_meals=[
            PurchasedMeal(type=str(_field(meal, "type")), size=str(_field(meal, "size")))
            for meal in purchased_meals
        ],
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {path}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"configuration file must contain a mapping: {path}")
    return data


def _required(data: dict[str, Any], section: str, key: str) -> Any:
    section_data = data.get(section)
    value = section_data.get(key) if isinstance(section_data, dict) else None
    if value is None:
        raise ConfigError(f"missing required configuration value: {section}.{key}")
    return value


def _optional(data: dict[str, Any], section: str, key: str) -> Any:
    section_data = data.get(section)
    if not isinstance(section_data, dict):
        return None
    return section_data.get(key)


def _field(data: dict[str, Any], key: str) -> Any:
    value = data.get(key)
    if value is None:
        raise ConfigError(f"missing required user field: {key}")
    return value


def _parse_offering_id(value: Any) -> int | str:
    if isinstance(value, bool):
        raise ConfigError("provider_offering_id must be an integer or string")
    if isinstance(value, int | str):
        return value
    raise ConfigError("provider_offering_id must be an integer or string")


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{field_name} must be a boolean")
