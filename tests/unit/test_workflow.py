from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

from meal_orchestrator import __version__
from meal_orchestrator.artifacts import ArtifactStore
from meal_orchestrator.config.models import ArtifactConfig
from meal_orchestrator.domain import (
    CanonicalMeal,
    LlmResult,
    ProviderMenuRequest,
    ProviderResult,
    RunContext,
    WorkflowStatus,
)
from meal_orchestrator.llm import EmptyLlmResponseError, LlmFailureDetails
from meal_orchestrator.providers import ProviderNormalizationError
from meal_orchestrator.rendering.html import render_html
from meal_orchestrator.rendering.labels import SUBJECT_EMOJI
from meal_orchestrator.rendering.plain_text import render_plain_text
from meal_orchestrator.retries import RetryError
from meal_orchestrator.workflow import UserWorkflowExecutor
from tests.unit.helpers import (
    FakeDiscordClient,
    FakeEmailClient,
    app_config,
    canonical_menu,
    user_config,
    week_assessment,
)


class FakeProvider:
    provider_id = "example_provider"

    def __init__(self, *, complete: bool = True) -> None:
        self.complete = complete
        self.requests: list[ProviderMenuRequest] = []

    def get_canonical_week_menu(self, request: ProviderMenuRequest):
        self.requests.append(request)
        return ProviderResult(
            menu=canonical_menu(complete=self.complete), raw_response={"raw": True}
        )


class FakeProviderWithNormalizationError:
    provider_id = "example_provider"

    def get_canonical_week_menu(self, request: ProviderMenuRequest):
        raise ProviderNormalizationError(
            "dish 'X' has no size 'XL'; available: ['L']",
            raw_response={"raw": "data from api"},
        )


class FakeLlmClient:
    def __init__(self, *, attempt: int = 1) -> None:
        self.requests = []
        self.attempt = attempt

    def generate(self, request, *, on_attempt=None):
        self.requests.append(request)
        if on_attempt is not None:
            on_attempt(self.attempt, None, {"id": "gen-example"}, {"accepted": True})
        return LlmResult(
            structured=week_assessment(request.payload.menu),
            model=request.model,
            response_metadata={"generation_id": "gen-example", "attempt": self.attempt},
        )


class FailingLlmClient:
    def generate(self, request, **_kwargs):
        raise RetryError(
            "openrouter failed after 3 attempt(s)",
            EmptyLlmResponseError(
                LlmFailureDetails(
                    reason="empty_message_content",
                    attempt=3,
                    response={
                        "id": "gen-example",
                        "model": request.model,
                        "choices": [{"message": {"content": None}}],
                    },
                )
            ),
        )


def _rejected_response(model: str, *, prompt_tokens: int, completion_tokens: int, cost: float):
    return (
        {
            "model": model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
            },
        },
        {"accepted": False, "reason": "incomplete_assessment", "model": model},
    )


class FallbackLlmClient:
    """Reports several rejected attempts across candidate models before accepting."""

    def generate(self, request, *, on_attempt=None):
        if on_attempt is not None:
            response, outcome = _rejected_response(
                "openai/gpt-5-mini", prompt_tokens=100, completion_tokens=50, cost=0.01
            )
            on_attempt(1, None, response, outcome)
            response, outcome = _rejected_response(
                "openai/gpt-5-mini", prompt_tokens=110, completion_tokens=60, cost=0.012
            )
            on_attempt(2, "retry feedback", response, outcome)
            on_attempt(
                3,
                None,
                None,
                {
                    "accepted": False,
                    "reason": "network_error",
                    "error": "timeout",
                    "model": "openai/gpt-5-mini",
                },
            )
            on_attempt(
                4,
                None,
                {
                    "model": "google/gemini-3.1-flash-lite",
                    "usage": {"prompt_tokens": 90, "completion_tokens": 40, "cost": 0.008},
                },
                {"accepted": True},
            )
        return LlmResult(
            structured=week_assessment(request.payload.menu),
            model="google/gemini-3.1-flash-lite",
            response_metadata={"generation_id": "gen-example", "attempt": 4},
        )


class FailingAfterAttemptsLlmClient:
    """Reports a couple of failed attempts before exhausting all candidates."""

    def generate(self, request, *, on_attempt=None):
        if on_attempt is not None:
            response, outcome = _rejected_response(
                "openai/gpt-5-mini", prompt_tokens=100, completion_tokens=50, cost=0.01
            )
            on_attempt(1, None, response, outcome)
            response, outcome = _rejected_response(
                "openai/gpt-5-mini", prompt_tokens=105, completion_tokens=55, cost=0.011
            )
            on_attempt(2, None, response, outcome)
        raise RetryError(
            "openrouter failed after 2 attempt(s)",
            EmptyLlmResponseError(
                LlmFailureDetails(
                    reason="incomplete_assessment",
                    attempt=2,
                    response={"id": "gen-example", "model": "openai/gpt-5-mini"},
                )
            ),
        )


def test_dry_run_calls_llm_with_dry_run_model_but_skips_delivery(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    provider = FakeProvider()
    llm = FakeLlmClient()
    email = FakeEmailClient()
    discord = FakeDiscordClient()

    result = _executor(tmp_path, provider, llm, email, discord).execute(
        user_config(PathLikePrompt(prompt_file, tmp_path)),
        _context(dry_run=True),
    )

    assert result.status == WorkflowStatus.COMPLETED
    assert provider.requests
    assert len(llm.requests) == 1
    assert llm.requests[0].model == "test-dry-run-model"
    assert email.messages == []
    assert discord.messages == []


def test_non_dry_run_calls_llm_and_email(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    llm = FakeLlmClient()
    email = FakeEmailClient()
    discord = FakeDiscordClient()

    result = _executor(tmp_path, FakeProvider(), llm, email, discord).execute(
        user_config(PathLikePrompt(prompt_file, tmp_path)),
        _context(dry_run=False),
    )

    assert result.status == WorkflowStatus.COMPLETED
    assert len(llm.requests) == 1
    assert email.messages[0].body == render_plain_text(
        week_assessment(canonical_menu()), canonical_menu(), "run-1"
    )
    assert email.messages[0].html_body == render_html(
        week_assessment(canonical_menu()), canonical_menu(), "run-1"
    )
    assert email.messages[0].subject == f"{SUBJECT_EMOJI} Meal plan for 2026-06-01 – 2026-06-05"
    assert email.idempotency_keys == ["run-1:alan:email"]


def test_llm_request_carries_configured_fallback_models(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    llm = FakeLlmClient()
    config = app_config(fallback_models=["openai/gpt-4.1-mini", "anthropic/claude-haiku-4-5"])

    _executor(
        tmp_path, FakeProvider(), llm, FakeEmailClient(), FakeDiscordClient(), config=config
    ).execute(
        user_config(PathLikePrompt(prompt_file, tmp_path)),
        _context(dry_run=False),
    )

    assert llm.requests[0].fallback_models == ["openai/gpt-4.1-mini", "anthropic/claude-haiku-4-5"]


def test_dry_run_omits_configured_fallback_models(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    llm = FakeLlmClient()
    config = app_config(fallback_models=["openai/gpt-4.1-mini"])

    _executor(
        tmp_path, FakeProvider(), llm, FakeEmailClient(), FakeDiscordClient(), config=config
    ).execute(
        user_config(PathLikePrompt(prompt_file, tmp_path)),
        _context(dry_run=True),
    )

    assert llm.requests[0].fallback_models == []


def test_no_email_client_skips_email(tmp_path, monkeypatch) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    monkeypatch.setenv("DISCORD_USER_WEBHOOK_URL", "https://example.com/user")
    llm = FakeLlmClient()
    discord = FakeDiscordClient()

    _executor(tmp_path, FakeProvider(), llm, None, discord).execute(
        user_config(PathLikePrompt(prompt_file, tmp_path)),
        _context(dry_run=False),
    )

    assert len(llm.requests) == 1
    assert len(discord.messages) == 1


def test_user_discord_notification_never_mentions_retries(tmp_path, monkeypatch) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    monkeypatch.setenv("DISCORD_USER_WEBHOOK_URL", "https://example.com/user")
    llm = FakeLlmClient(attempt=3)
    discord = FakeDiscordClient()

    _executor(tmp_path, FakeProvider(), llm, FakeEmailClient(), discord).execute(
        user_config(PathLikePrompt(prompt_file, tmp_path)),
        _context(dry_run=False),
    )

    assert len(discord.messages) == 1
    assert "retr" not in discord.messages[0].description


def test_workflow_result_reports_retry_count_on_success(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    llm = FakeLlmClient(attempt=3)

    result = _executor(
        tmp_path, FakeProvider(), llm, FakeEmailClient(), FakeDiscordClient()
    ).execute(
        user_config(PathLikePrompt(prompt_file, tmp_path)),
        _context(dry_run=False),
    )

    assert result.status == WorkflowStatus.COMPLETED
    assert result.retry_count == 2


def test_workflow_result_reports_zero_retries_when_first_attempt_succeeds(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    llm = FakeLlmClient(attempt=1)

    result = _executor(
        tmp_path, FakeProvider(), llm, FakeEmailClient(), FakeDiscordClient()
    ).execute(
        user_config(PathLikePrompt(prompt_file, tmp_path)),
        _context(dry_run=False),
    )

    assert result.status == WorkflowStatus.COMPLETED
    assert result.retry_count == 0


def test_incomplete_menu_skips_llm_and_email(tmp_path, monkeypatch) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    monkeypatch.setenv("DISCORD_USER_WEBHOOK_URL", "https://example.com/user")
    llm = FakeLlmClient()
    email = FakeEmailClient()
    discord = FakeDiscordClient()

    result = _executor(tmp_path, FakeProvider(complete=False), llm, email, discord).execute(
        user_config(PathLikePrompt(prompt_file, tmp_path)),
        _context(dry_run=False),
    )

    assert result.status == WorkflowStatus.MENU_UNAVAILABLE
    assert "2026-06-02" in result.detail
    assert llm.requests == []
    assert email.messages == []
    assert len(discord.messages) == 1


def test_purchased_meal_with_no_variants_is_treated_as_menu_unavailable(
    tmp_path, monkeypatch
) -> None:
    """A meal type present but with zero variants must fail fast here.

    Otherwise it reaches the LLM and, if the model echoes back zero variants
    for it too (which passes completeness validation, since there's nothing to
    require), the renderer crashes with IndexError deep in email delivery.
    """
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    monkeypatch.setenv("DISCORD_USER_WEBHOOK_URL", "https://example.com/user")
    menu = canonical_menu()
    empty_day = menu.days[0]
    menu = replace(
        menu,
        days=[
            replace(empty_day, meals=[CanonicalMeal(type="breakfast", variants=[])]),
            *menu.days[1:],
        ],
    )

    class EmptyVariantsProvider:
        provider_id = "example_provider"

        def get_canonical_week_menu(self, request: ProviderMenuRequest):
            return ProviderResult(menu=menu)

    llm = FakeLlmClient()
    email = FakeEmailClient()
    discord = FakeDiscordClient()

    result = _executor(tmp_path, EmptyVariantsProvider(), llm, email, discord).execute(
        user_config(PathLikePrompt(prompt_file, tmp_path)),
        _context(dry_run=False),
    )

    assert result.status == WorkflowStatus.MENU_UNAVAILABLE
    assert "no variants" in result.detail
    assert llm.requests == []
    assert email.messages == []


def test_artifacts_written_on_successful_run(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    store = ArtifactStore(
        ArtifactConfig(path=artifacts_dir, retention_days=14, max_runs_per_user=10)
    )

    executor = _executor(
        tmp_path, FakeProvider(), FakeLlmClient(), FakeEmailClient(), FakeDiscordClient(), store
    )
    executor.execute(user_config(prompt_file.relative_to(tmp_path)), _context(dry_run=False))

    run_dir = artifacts_dir / "alan" / "run-1"
    assert (run_dir / "provider_raw.json").exists()
    assert (run_dir / "canonical_menu.json").exists()
    assert (run_dir / "llm_request.json").exists()
    assert (run_dir / "llm_response.json").exists()
    attempt = json.loads((run_dir / "llm_attempts" / "attempt_01.json").read_text())
    assert attempt["accepted"] is True
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["user_id"] == "alan"
    assert metadata["app_version"] == __version__
    assert metadata["llm_response"] == {"generation_id": "gen-example", "attempt": 1}
    assert metadata["llm_attempts_summary"] == {
        "total_attempts": 1,
        "models_tried": [],
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
    }


def test_llm_artifacts_saved_on_dry_run(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    store = ArtifactStore(
        ArtifactConfig(path=artifacts_dir, retention_days=14, max_runs_per_user=10)
    )

    executor = _executor(
        tmp_path, FakeProvider(), FakeLlmClient(), FakeEmailClient(), FakeDiscordClient(), store
    )
    executor.execute(user_config(prompt_file.relative_to(tmp_path)), _context(dry_run=True))

    run_dir = artifacts_dir / "alan" / "run-1"
    assert (run_dir / "llm_request.json").exists()
    assert (run_dir / "llm_response.json").exists()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "completed"


def test_llm_attempts_summary_aggregates_fallback_attempts(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    store = ArtifactStore(
        ArtifactConfig(path=artifacts_dir, retention_days=14, max_runs_per_user=10)
    )

    executor = _executor(
        tmp_path,
        FakeProvider(),
        FallbackLlmClient(),
        FakeEmailClient(),
        FakeDiscordClient(),
        store,
    )
    executor.execute(user_config(prompt_file.relative_to(tmp_path)), _context(dry_run=False))

    metadata = json.loads((artifacts_dir / "alan" / "run-1" / "metadata.json").read_text())
    summary = metadata["llm_attempts_summary"]
    assert summary["total_attempts"] == 4
    assert summary["models_tried"] == ["openai/gpt-5-mini", "google/gemini-3.1-flash-lite"]
    assert summary["total_cost"] == 0.03
    assert summary["total_prompt_tokens"] == 300
    assert summary["total_completion_tokens"] == 150


def test_llm_attempts_summary_present_when_all_candidates_exhausted(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    store = ArtifactStore(
        ArtifactConfig(path=artifacts_dir, retention_days=14, max_runs_per_user=10)
    )

    executor = _executor(
        tmp_path,
        FakeProvider(),
        FailingAfterAttemptsLlmClient(),
        FakeEmailClient(),
        FakeDiscordClient(),
        store,
    )
    result = executor.execute(
        user_config(prompt_file.relative_to(tmp_path)), _context(dry_run=False)
    )

    assert result.status == WorkflowStatus.FAILED
    metadata = json.loads((artifacts_dir / "alan" / "run-1" / "metadata.json").read_text())
    summary = metadata["llm_attempts_summary"]
    assert summary["total_attempts"] == 2
    assert summary["models_tried"] == ["openai/gpt-5-mini"]
    assert round(summary["total_cost"], 3) == 0.021
    assert summary["total_prompt_tokens"] == 205
    assert summary["total_completion_tokens"] == 105


def test_metadata_written_on_failed_run(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    store = ArtifactStore(
        ArtifactConfig(path=artifacts_dir, retention_days=14, max_runs_per_user=10)
    )

    executor = _executor(
        tmp_path,
        FakeProvider(complete=False),
        FakeLlmClient(),
        FakeEmailClient(),
        FakeDiscordClient(),
        store,
    )
    executor.execute(user_config(prompt_file.relative_to(tmp_path)), _context(dry_run=False))

    run_dir = artifacts_dir / "alan" / "run-1"
    assert (run_dir / "canonical_menu.json").exists()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "menu_unavailable"


def test_llm_failure_skips_delivery_and_records_failed_step(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    store = ArtifactStore(
        ArtifactConfig(path=artifacts_dir, retention_days=14, max_runs_per_user=10)
    )
    email = FakeEmailClient()

    user = replace(user_config(prompt_file.relative_to(tmp_path)), id="example")
    result = _executor(
        tmp_path, FakeProvider(), FailingLlmClient(), email, FakeDiscordClient(), store
    ).execute(user, _context(dry_run=False))

    assert result.status == WorkflowStatus.FAILED
    assert result.failed_step == "llm"
    assert result.retry_count == 2
    assert email.messages == []
    metadata = json.loads((artifacts_dir / "example" / "run-1" / "metadata.json").read_text())
    assert metadata["failed_step"] == "llm"
    assert metadata["llm_failure"]["generation_id"] == "gen-example"
    assert not (artifacts_dir / "example" / "run-1" / "llm_failure.json").exists()


def test_discord_skipped_when_discord_user_id_is_none(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    discord = FakeDiscordClient()
    user = user_config(PathLikePrompt(prompt_file, tmp_path))
    user_without_id = _user_without_discord_user_id(user)

    executor = _executor(tmp_path, FakeProvider(), FakeLlmClient(), FakeEmailClient(), discord)
    result = executor.execute(user_without_id, _context(dry_run=False))

    assert result.status == WorkflowStatus.COMPLETED
    assert discord.messages == []


def test_discord_skipped_on_menu_unavailable_when_discord_user_id_is_none(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    discord = FakeDiscordClient()
    user = user_config(PathLikePrompt(prompt_file, tmp_path))
    user_without_id = _user_without_discord_user_id(user)

    executor = _executor(
        tmp_path, FakeProvider(complete=False), FakeLlmClient(), FakeEmailClient(), discord
    )
    result = executor.execute(user_without_id, _context(dry_run=False))

    assert result.status == WorkflowStatus.MENU_UNAVAILABLE
    assert discord.messages == []


def test_discord_skipped_when_webhook_env_var_not_set(tmp_path, monkeypatch) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    monkeypatch.delenv("DISCORD_USER_WEBHOOK_URL", raising=False)
    discord = FakeDiscordClient()

    executor = _executor(tmp_path, FakeProvider(), FakeLlmClient(), FakeEmailClient(), discord)
    result = executor.execute(
        user_config(PathLikePrompt(prompt_file, tmp_path)), _context(dry_run=False)
    )

    assert result.status == WorkflowStatus.COMPLETED
    assert discord.messages == []


def test_normalization_error_returns_failed_without_discord(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    llm = FakeLlmClient()
    email = FakeEmailClient()
    discord = FakeDiscordClient()

    result = _executor(
        tmp_path, FakeProviderWithNormalizationError(), llm, email, discord
    ).execute(
        user_config(PathLikePrompt(prompt_file, tmp_path)),
        _context(dry_run=False),
    )

    assert result.status == WorkflowStatus.FAILED
    assert "has no size" in result.detail
    assert llm.requests == []
    assert email.messages == []
    assert discord.messages == []


def test_normalization_error_saves_provider_raw_artifact(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"
    store = ArtifactStore(
        ArtifactConfig(path=artifacts_dir, retention_days=14, max_runs_per_user=10)
    )

    executor = _executor(
        tmp_path,
        FakeProviderWithNormalizationError(),
        FakeLlmClient(),
        FakeEmailClient(),
        FakeDiscordClient(),
        store,
    )
    executor.execute(user_config(PathLikePrompt(prompt_file, tmp_path)), _context(dry_run=False))

    run_dir = artifacts_dir / "alan" / "run-1"
    assert (run_dir / "provider_raw.json").exists()
    raw = json.loads((run_dir / "provider_raw.json").read_text())
    assert raw == {"raw": "data from api"}
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "failed"


def _executor(
    tmp_path, provider, llm, email, discord, artifact_store=None, config=None
) -> UserWorkflowExecutor:
    return UserWorkflowExecutor(
        app_config=config or app_config(),
        provider=provider,
        llm_client=llm,
        email_client=email,
        discord_client=discord,
        project_root=tmp_path,
        artifact_store=artifact_store,
    )


def _context(*, dry_run: bool) -> RunContext:
    return RunContext(
        run_id="run-1",
        week_start=date(2026, 6, 1),
        week_end=date(2026, 6, 5),
        dry_run=dry_run,
        provider_id="example_provider",
    )


def PathLikePrompt(prompt_file, project_root):
    return prompt_file.relative_to(project_root)


def _user_without_discord_user_id(user):
    from dataclasses import replace
    return replace(user, discord_user_id=None)
