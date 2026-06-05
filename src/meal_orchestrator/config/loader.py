from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from meal_orchestrator.config.models import (
    AppConfig,
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
    return AppConfig(
        runtime=RuntimeConfig(timezone=_required(data, "runtime", "timezone")),
        llm=LlmConfig(
            provider=_required(data, "llm", "provider"),
            model=_required(data, "llm", "model"),
            timeout_seconds=int(_required(data, "llm", "timeout_seconds")),
            max_retries=int(_required(data, "llm", "max_retries")),
        ),
        default_provider=_required(data, "providers", "default"),
        delivery=DeliveryConfig(
            email_from=_required(data, "delivery", "email_from"),
            operational_discord_webhook_env=_required(
                data, "delivery", "operational_discord_webhook_env"
            ),
        ),
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
        discord_user_id=str(_field(raw_user, "discord_user_id")),
        discord_webhook_env=str(_field(raw_user, "discord_webhook_env")),
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
    value = data.get(section, {}).get(key)
    if value is None:
        raise ConfigError(f"missing required configuration value: {section}.{key}")
    return value


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
