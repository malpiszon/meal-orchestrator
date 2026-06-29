from pathlib import Path

import pytest

from meal_orchestrator.config import ConfigError, load_app_config, load_users_config


def test_load_example_config_files() -> None:
    app = load_app_config(Path("config/app.example.yaml"))
    users = load_users_config(Path("config/users.example.yaml"))

    assert app.runtime.timezone == "Europe/Warsaw"
    assert app.llm.model == "openai/gpt-4.1-mini"
    assert app.artifacts is not None
    assert app.artifacts.retention_days == 14
    assert app.artifacts.max_runs_per_user == 10
    assert users[0].id == "example"
    assert users[0].purchased_meals[0].type == "breakfast"


def test_artifacts_config_disabled_returns_none(tmp_path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        """
runtime:
  timezone: Europe/Warsaw
llm:
  provider: openrouter
  model: test
  timeout_seconds: 30
  max_retries: 1
providers:
  default: example_provider
delivery:
  email_from: test@example.com
  operational_discord_webhook_env: DISCORD_OPS_WEBHOOK_URL
artifacts:
  enabled: false
""",
        encoding="utf-8",
    )
    app = load_app_config(path)
    assert app.artifacts is None


def test_artifacts_config_missing_path_raises(tmp_path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        """
runtime:
  timezone: Europe/Warsaw
llm:
  provider: openrouter
  model: test
  timeout_seconds: 30
  max_retries: 1
providers:
  default: example_provider
delivery:
  email_from: test@example.com
  operational_discord_webhook_env: DISCORD_OPS_WEBHOOK_URL
artifacts:
  enabled: true
  retention_days: 14
  max_runs_per_user: 10
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="artifacts.path"):
        load_app_config(path)


def test_artifacts_config_invalid_retention_days_raises(tmp_path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        """
runtime:
  timezone: Europe/Warsaw
llm:
  provider: openrouter
  model: test
  timeout_seconds: 30
  max_retries: 1
providers:
  default: example_provider
delivery:
  email_from: test@example.com
  operational_discord_webhook_env: DISCORD_OPS_WEBHOOK_URL
artifacts:
  enabled: true
  path: /data/artifacts
  retention_days: 0
  max_runs_per_user: 10
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="retention_days"):
        load_app_config(path)


def test_users_config_requires_users_list(tmp_path) -> None:
    path = tmp_path / "users.yaml"
    path.write_text("not_users: []", encoding="utf-8")

    with pytest.raises(ConfigError, match="users list"):
        load_users_config(path)


def test_users_config_rejects_empty_purchased_meals(tmp_path) -> None:
    path = tmp_path / "users.yaml"
    path.write_text(
        """
users:
  - id: alan
    enabled: true
    provider: example_provider
    provider_offering_id: 123
    email: alan@example.com
    discord_user_id: "123"
    discord_webhook_env: DISCORD_ALAN_WEBHOOK_URL
    prompt_file: prompts/alan.md
    purchased_meals: []
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="purchased_meals"):
        load_users_config(path)


def test_users_config_rejects_duplicate_user_ids(tmp_path) -> None:
    path = tmp_path / "users.yaml"
    path.write_text(
        """
users:
  - id: alan
    enabled: true
    provider: example_provider
    provider_offering_id: 123
    email: alan@example.com
    discord_user_id: "123"
    discord_webhook_env: DISCORD_ALAN_WEBHOOK_URL
    prompt_file: prompts/alan.md
    purchased_meals:
      - type: breakfast
        size: M
  - id: alan
    enabled: true
    provider: example_provider
    provider_offering_id: 456
    email: alan2@example.com
    discord_user_id: "456"
    discord_webhook_env: DISCORD_ALAN2_WEBHOOK_URL
    prompt_file: prompts/alan2.md
    purchased_meals:
      - type: lunch
        size: XL
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="duplicate user ids"):
        load_users_config(path)


def test_users_config_rejects_invalid_enabled_value(tmp_path) -> None:
    path = tmp_path / "users.yaml"
    path.write_text(
        """
users:
  - id: alan
    enabled: "yes"
    provider: example_provider
    provider_offering_id: 123
    email: alan@example.com
    discord_user_id: "123"
    discord_webhook_env: DISCORD_ALAN_WEBHOOK_URL
    prompt_file: prompts/alan.md
    purchased_meals:
      - type: breakfast
        size: M
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="enabled must be a boolean"):
        load_users_config(path)


def test_users_config_rejects_invalid_provider_offering_id(tmp_path) -> None:
    path = tmp_path / "users.yaml"
    path.write_text(
        """
users:
  - id: alan
    enabled: true
    provider: example_provider
    provider_offering_id:
      nested: value
    email: alan@example.com
    discord_user_id: "123"
    discord_webhook_env: DISCORD_ALAN_WEBHOOK_URL
    prompt_file: prompts/alan.md
    purchased_meals:
      - type: breakfast
        size: M
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="provider_offering_id"):
        load_users_config(path)
