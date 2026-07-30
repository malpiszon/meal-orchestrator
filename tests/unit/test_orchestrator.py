from __future__ import annotations

from dataclasses import replace
from datetime import date

from meal_orchestrator.domain import LlmResult, ProviderMenuRequest, ProviderResult, WorkflowStatus
from meal_orchestrator.llm import EmptyLlmResponseError, LlmFailureDetails
from meal_orchestrator.orchestrator import RunOptions, RunOrchestrator
from meal_orchestrator.retries import RetryError
from tests.unit.helpers import (
    FakeDiscordClient,
    FakeEmailClient,
    app_config,
    canonical_menu,
    user_config,
    week_assessment,
)


class RecordingProvider:
    provider_id = "example_provider"

    def __init__(self) -> None:
        self.requests: list[ProviderMenuRequest] = []

    def get_canonical_week_menu(self, request: ProviderMenuRequest):
        self.requests.append(request)
        return ProviderResult(menu=canonical_menu())


class FailingProvider:
    provider_id = "example_provider"

    def get_canonical_week_menu(self, request: ProviderMenuRequest):
        raise RuntimeError("provider exploded")


class FakeLlmClient:
    def __init__(self, *, attempt: int = 1) -> None:
        self.attempt = attempt

    def generate(self, request, **_kwargs):
        return LlmResult(
            structured=week_assessment(request.payload.menu),
            model=request.model,
            response_metadata={"attempt": self.attempt},
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


def _no_capability_check(model: str) -> None:
    pass


def test_orchestrator_uses_provider_override(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    provider = RecordingProvider()
    discord = FakeDiscordClient()

    orchestrator = RunOrchestrator(
        app_config=app_config(),
        users=[user_config(prompt_file.relative_to(tmp_path))],
        project_root=tmp_path,
        provider_factory=lambda provider_id: provider,
        llm_client=FakeLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=discord,
        capability_check=_no_capability_check,
    )

    result = orchestrator.run(
        RunOptions(
            provider_override="override_provider",
            week_start=date(2026, 6, 1),
            dry_run=True,
        )
    )

    assert result[0].status == WorkflowStatus.COMPLETED
    assert provider.requests[0].provider_offering_id == 123


def test_orchestrator_sends_operational_notification_on_completed(tmp_path, monkeypatch) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    monkeypatch.setenv("DISCORD_OPS_WEBHOOK_URL", "https://example.com/ops")
    monkeypatch.setenv("DISCORD_USER_WEBHOOK_URL", "https://example.com/user")
    discord = FakeDiscordClient()

    orchestrator = RunOrchestrator(
        app_config=app_config(),
        users=[user_config(prompt_file.relative_to(tmp_path))],
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=FakeLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=discord,
        capability_check=_no_capability_check,
    )

    result = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert result[0].status == WorkflowStatus.COMPLETED
    ops_msg = discord.messages[-1]
    assert ops_msg.webhook_env == "DISCORD_OPS_WEBHOOK_URL"
    assert "completed" in ops_msg.description


def test_operational_notification_includes_retry_count(tmp_path, monkeypatch) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    monkeypatch.setenv("DISCORD_OPS_WEBHOOK_URL", "https://example.com/ops")
    monkeypatch.setenv("DISCORD_USER_WEBHOOK_URL", "https://example.com/user")
    discord = FakeDiscordClient()

    orchestrator = RunOrchestrator(
        app_config=app_config(),
        users=[user_config(prompt_file.relative_to(tmp_path))],
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=FakeLlmClient(attempt=3),
        email_client=FakeEmailClient(),
        discord_client=discord,
        capability_check=_no_capability_check,
    )

    result = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert result[0].status == WorkflowStatus.COMPLETED
    ops_msg = discord.messages[-1]
    assert "(2 retries)" in ops_msg.description
    user_msg = next(m for m in discord.messages if m.webhook_env != "DISCORD_OPS_WEBHOOK_URL")
    assert "retr" not in user_msg.description


def test_orchestrator_sends_operational_notification_on_failure(tmp_path, monkeypatch) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    monkeypatch.setenv("DISCORD_OPS_WEBHOOK_URL", "https://example.com/ops")
    monkeypatch.setenv("DISCORD_USER_WEBHOOK_URL", "https://example.com/user")
    discord = FakeDiscordClient()

    orchestrator = RunOrchestrator(
        app_config=app_config(),
        users=[user_config(prompt_file.relative_to(tmp_path))],
        project_root=tmp_path,
        provider_factory=lambda provider_id: FailingProvider(),
        llm_client=FakeLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=discord,
        capability_check=_no_capability_check,
    )

    result = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert result[0].status == WorkflowStatus.FAILED
    assert discord.messages[0].webhook_env == "DISCORD_OPS_WEBHOOK_URL"
    assert "step provider" in discord.messages[0].description
    assert "provider exploded" in discord.messages[0].description


def test_operational_notification_includes_retry_count_on_llm_failure(
    tmp_path, monkeypatch
) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    monkeypatch.setenv("DISCORD_OPS_WEBHOOK_URL", "https://example.com/ops")
    monkeypatch.setenv("DISCORD_USER_WEBHOOK_URL", "https://example.com/user")
    discord = FakeDiscordClient()

    orchestrator = RunOrchestrator(
        app_config=app_config(),
        users=[user_config(prompt_file.relative_to(tmp_path))],
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=FailingLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=discord,
        capability_check=_no_capability_check,
    )

    result = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert result[0].status == WorkflowStatus.FAILED
    ops_msg = discord.messages[0]
    assert ops_msg.webhook_env == "DISCORD_OPS_WEBHOOK_URL"
    assert "(2 retries)" in ops_msg.description


def test_orchestrator_skips_operational_notification_when_env_var_not_set(
    tmp_path, monkeypatch
) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    monkeypatch.delenv("DISCORD_OPS_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("DISCORD_USER_WEBHOOK_URL", "https://example.com/user")
    discord = FakeDiscordClient()

    orchestrator = RunOrchestrator(
        app_config=app_config(),
        users=[user_config(prompt_file.relative_to(tmp_path))],
        project_root=tmp_path,
        provider_factory=lambda provider_id: FailingProvider(),
        llm_client=FakeLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=discord,
        capability_check=_no_capability_check,
    )

    result = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert result[0].status == WorkflowStatus.FAILED
    assert discord.messages == []


def test_orchestrator_wires_configured_max_retries_into_llm_client(monkeypatch, tmp_path) -> None:
    """OpenRouterClient must be constructed with app_config.llm.max_retries.

    Only exercised when no llm_client override is supplied, since every other
    orchestrator test bypasses this construction path.
    """
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    captured_kwargs: dict = {}

    class SpyOpenRouterClient:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

        def generate(self, request, **_kwargs):
            return LlmResult(structured=week_assessment(request.payload.menu), model=request.model)

    monkeypatch.setattr("meal_orchestrator.orchestrator.OpenRouterClient", SpyOpenRouterClient)

    orchestrator = RunOrchestrator(
        app_config=app_config(),
        users=[user_config(prompt_file.relative_to(tmp_path))],
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=True))

    assert captured_kwargs == {"max_retries": app_config().llm.max_retries}


def test_orchestrator_dry_run_suppresses_ops_notification(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    discord = FakeDiscordClient()

    orchestrator = RunOrchestrator(
        app_config=app_config(),
        users=[user_config(prompt_file.relative_to(tmp_path))],
        project_root=tmp_path,
        provider_factory=lambda provider_id: FailingProvider(),
        llm_client=FakeLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=discord,
        capability_check=_no_capability_check,
    )

    result = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=True))

    assert result[0].status == WorkflowStatus.FAILED
    assert discord.messages == []


def test_orchestrator_runs_capability_check_with_resolved_model(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    captured_models: list[str] = []

    orchestrator = RunOrchestrator(
        app_config=app_config(),
        users=[user_config(prompt_file.relative_to(tmp_path))],
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=FakeLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=captured_models.append,
    )

    orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))
    orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=True))
    orchestrator.run(
        RunOptions(week_start=date(2026, 6, 1), dry_run=False, llm_model="override-model")
    )

    assert captured_models == ["test-model", "test-dry-run-model", "override-model"]


def test_orchestrator_fails_all_users_gracefully_when_capability_check_fails(tmp_path) -> None:
    """A capability-check failure must degrade like every other failure mode.

    Not propagate as a raw uncaught exception that crashes the whole process
    with no per-user WorkflowResult and no ops notification.
    """
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")

    def _failing_capability_check(model: str) -> None:
        raise RuntimeError("model does not support structured outputs")

    other_user_prompt = tmp_path / "other_prompt.md"
    other_user_prompt.write_text("Choose meals.", encoding="utf-8")
    users = [
        user_config(prompt_file.relative_to(tmp_path)),
        replace(
            user_config(other_user_prompt.relative_to(tmp_path)),
            id="other",
            email="other@example.com",
        ),
    ]

    orchestrator = RunOrchestrator(
        app_config=app_config(),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=FakeLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_failing_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert len(results) == 2
    for result in results:
        assert result.status == WorkflowStatus.FAILED
        assert result.failed_step == "capability_check"
        assert "does not support structured outputs" in result.detail


def test_capability_check_failure_sends_ops_notification(tmp_path, monkeypatch) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    monkeypatch.setenv("DISCORD_OPS_WEBHOOK_URL", "https://example.com/ops")
    discord = FakeDiscordClient()

    def _failing_capability_check(model: str) -> None:
        raise RuntimeError("model does not support structured outputs")

    orchestrator = RunOrchestrator(
        app_config=app_config(),
        users=[user_config(prompt_file.relative_to(tmp_path))],
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=FakeLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=discord,
        capability_check=_failing_capability_check,
    )

    orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert len(discord.messages) == 1
    assert discord.messages[0].webhook_env == "DISCORD_OPS_WEBHOOK_URL"
    assert "Capability check failed" in discord.messages[0].description


def test_capability_check_failure_suppresses_ops_notification_on_dry_run(
    tmp_path, monkeypatch
) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    monkeypatch.setenv("DISCORD_OPS_WEBHOOK_URL", "https://example.com/ops")
    discord = FakeDiscordClient()

    def _failing_capability_check(model: str) -> None:
        raise RuntimeError("model does not support structured outputs")

    orchestrator = RunOrchestrator(
        app_config=app_config(),
        users=[user_config(prompt_file.relative_to(tmp_path))],
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=FakeLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=discord,
        capability_check=_failing_capability_check,
    )

    orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=True))

    assert discord.messages == []
