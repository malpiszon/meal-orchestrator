from pathlib import Path

import pytest

from meal_orchestrator.config import ConfigError, load_app_config, load_users_config


def test_load_example_config_files() -> None:
    app = load_app_config(Path("config/app.example.yaml"))
    users = load_users_config(Path("config/users.example.yaml"))

    assert app.runtime.timezone == "Europe/Warsaw"
    assert app.runtime.max_concurrent_users == 5
    assert app.llm.model == "openai/gpt-5-mini"
    assert app.llm.dry_run_model == "google/gemini-2.5-flash-lite"
    assert app.artifacts is not None
    assert app.artifacts.retention_days == 14
    assert app.artifacts.max_runs_per_user == 10
    assert users[0].id == "example"
    assert users[0].purchased_meals[0].type == "breakfast"


def test_max_concurrent_users_defaults_when_absent(tmp_path) -> None:
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
""",
        encoding="utf-8",
    )
    app = load_app_config(path)
    assert app.runtime.max_concurrent_users == 5


def test_max_concurrent_users_loaded_when_present(tmp_path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        """
runtime:
  timezone: Europe/Warsaw
  max_concurrent_users: 3
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
""",
        encoding="utf-8",
    )
    app = load_app_config(path)
    assert app.runtime.max_concurrent_users == 3


def test_max_concurrent_users_rejects_non_positive_value(tmp_path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        """
runtime:
  timezone: Europe/Warsaw
  max_concurrent_users: 0
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
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="max_concurrent_users must be a positive integer"):
        load_app_config(path)


def test_max_concurrent_users_rejects_non_integer_value(tmp_path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        """
runtime:
  timezone: Europe/Warsaw
  max_concurrent_users: "five"
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
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="max_concurrent_users must be a positive integer"):
        load_app_config(path)


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
    assert app.llm.dry_run_model is None


def test_dry_run_model_loaded_when_present(tmp_path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        """
runtime:
  timezone: Europe/Warsaw
llm:
  provider: openrouter
  model: test
  dry_run_model: test-cheap-model
  timeout_seconds: 30
  max_retries: 1
providers:
  default: example_provider
delivery:
  email_from: test@example.com
  operational_discord_webhook_env: DISCORD_OPS_WEBHOOK_URL
""",
        encoding="utf-8",
    )
    app = load_app_config(path)
    assert app.llm.dry_run_model == "test-cheap-model"


def test_fallback_models_defaults_to_empty_list(tmp_path) -> None:
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
""",
        encoding="utf-8",
    )
    app = load_app_config(path)
    assert app.llm.fallback_models == []


def test_fallback_models_loaded_when_present(tmp_path) -> None:
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
  fallback_models:
    - openai/gpt-4.1-mini
    - anthropic/claude-haiku-4-5
providers:
  default: example_provider
delivery:
  email_from: test@example.com
  operational_discord_webhook_env: DISCORD_OPS_WEBHOOK_URL
""",
        encoding="utf-8",
    )
    app = load_app_config(path)
    assert app.llm.fallback_models == ["openai/gpt-4.1-mini", "anthropic/claude-haiku-4-5"]


def test_fallback_models_rejects_non_list(tmp_path) -> None:
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
  fallback_models: "openai/gpt-4.1-mini"
providers:
  default: example_provider
delivery:
  email_from: test@example.com
  operational_discord_webhook_env: DISCORD_OPS_WEBHOOK_URL
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="fallback_models must be a list of strings"):
        load_app_config(path)


def test_fallback_models_rejects_non_string_items(tmp_path) -> None:
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
  fallback_models:
    - openai/gpt-4.1-mini
    - 123
providers:
  default: example_provider
delivery:
  email_from: test@example.com
  operational_discord_webhook_env: DISCORD_OPS_WEBHOOK_URL
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="fallback_models must be a list of strings"):
        load_app_config(path)


def test_fallback_models_rejects_primary_model_as_its_own_fallback(tmp_path) -> None:
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
  fallback_models:
    - openai/gpt-4.1-mini
    - test
providers:
  default: example_provider
delivery:
  email_from: test@example.com
  operational_discord_webhook_env: DISCORD_OPS_WEBHOOK_URL
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="must not include the primary model"):
        load_app_config(path)


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
  - id: user
    enabled: true
    provider: example_provider
    provider_offering_id: 123
    email: user@example.com
    discord_user_id: "123"
    discord_webhook_env: DISCORD_USER_WEBHOOK_URL
    prompt_file: prompts/user.md
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
  - id: user
    enabled: true
    provider: example_provider
    provider_offering_id: 123
    email: user@example.com
    discord_user_id: "123"
    discord_webhook_env: DISCORD_USER_WEBHOOK_URL
    prompt_file: prompts/user.md
    purchased_meals:
      - type: breakfast
        size: M
  - id: user
    enabled: true
    provider: example_provider
    provider_offering_id: 456
    email: user2@example.com
    discord_user_id: "456"
    discord_webhook_env: DISCORD_USER2_WEBHOOK_URL
    prompt_file: prompts/user2.md
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
  - id: user
    enabled: "yes"
    provider: example_provider
    provider_offering_id: 123
    email: user@example.com
    discord_user_id: "123"
    discord_webhook_env: DISCORD_USER_WEBHOOK_URL
    prompt_file: prompts/user.md
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
  - id: user
    enabled: true
    provider: example_provider
    provider_offering_id:
      nested: value
    email: user@example.com
    discord_user_id: "123"
    discord_webhook_env: DISCORD_USER_WEBHOOK_URL
    prompt_file: prompts/user.md
    purchased_meals:
      - type: breakfast
        size: M
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="provider_offering_id"):
        load_users_config(path)


def _base_app_yaml(extra_llm: str = "") -> str:
    return f"""
runtime:
  timezone: Europe/Warsaw
llm:
  provider: openrouter
  model: test
  timeout_seconds: 30
  max_retries: 1
{extra_llm}
providers:
  default: example_provider
delivery:
  email_from: test@example.com
  operational_discord_webhook_env: DISCORD_OPS_WEBHOOK_URL
"""


def test_batch_config_defaults_when_absent(tmp_path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(_base_app_yaml(), encoding="utf-8")

    app = load_app_config(path)

    assert app.llm.batch.enabled is False
    assert app.llm.batch.state_dir is None
    assert app.llm.batch.initial_check_delay_seconds == 15
    assert app.llm.batch.initial_poll_interval_seconds == 120
    assert app.llm.batch.max_poll_interval_seconds == 3600
    assert app.llm.batch.max_wait_hours == 26


def test_batch_config_loaded_when_present(tmp_path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        _base_app_yaml(
            """  batch:
    enabled: true
    state_dir: /data/batch_state
    initial_check_delay_seconds: 5
    initial_poll_interval_seconds: 60
    max_poll_interval_seconds: 1800
    max_wait_hours: 10
"""
        ),
        encoding="utf-8",
    )

    app = load_app_config(path)

    assert app.llm.batch.enabled is True
    assert app.llm.batch.state_dir == Path("/data/batch_state")
    assert app.llm.batch.initial_check_delay_seconds == 5
    assert app.llm.batch.initial_poll_interval_seconds == 60
    assert app.llm.batch.max_poll_interval_seconds == 1800
    assert app.llm.batch.max_wait_hours == 10


def test_batch_config_allows_zero_initial_check_delay(tmp_path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        _base_app_yaml(
            """  batch:
    enabled: true
    state_dir: /data/batch_state
    initial_check_delay_seconds: 0
"""
        ),
        encoding="utf-8",
    )

    app = load_app_config(path)

    assert app.llm.batch.initial_check_delay_seconds == 0


def test_batch_config_rejects_negative_initial_check_delay(tmp_path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        _base_app_yaml(
            """  batch:
    enabled: true
    state_dir: /data/batch_state
    initial_check_delay_seconds: -1
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="initial_check_delay_seconds"):
        load_app_config(path)


def test_batch_config_requires_state_dir_when_enabled(tmp_path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        _base_app_yaml(
            """  batch:
    enabled: true
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="state_dir"):
        load_app_config(path)


def test_batch_config_rejects_non_positive_max_wait_hours(tmp_path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        _base_app_yaml(
            """  batch:
    enabled: true
    state_dir: /data/batch_state
    max_wait_hours: 0
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="max_wait_hours"):
        load_app_config(path)


def test_batch_config_rejects_initial_poll_interval_above_max(tmp_path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        _base_app_yaml(
            """  batch:
    enabled: true
    state_dir: /data/batch_state
    initial_poll_interval_seconds: 7200
    max_poll_interval_seconds: 3600
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="initial_poll_interval_seconds"):
        load_app_config(path)
