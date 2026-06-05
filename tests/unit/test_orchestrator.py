from __future__ import annotations

from datetime import date

from meal_orchestrator.domain import LlmResult, ProviderMenuRequest, WorkflowStatus
from meal_orchestrator.orchestrator import RunOptions, RunOrchestrator
from tests.unit.helpers import (
    FakeDiscordClient,
    FakeEmailClient,
    app_config,
    canonical_menu,
    user_config,
)


class RecordingProvider:
    provider_id = "example_provider"

    def __init__(self) -> None:
        self.requests: list[ProviderMenuRequest] = []

    def get_canonical_week_menu(self, request: ProviderMenuRequest):
        self.requests.append(request)
        return canonical_menu()


class FailingProvider:
    provider_id = "example_provider"

    def get_canonical_week_menu(self, request: ProviderMenuRequest):
        raise RuntimeError("provider exploded")


class FakeLlmClient:
    def generate(self, request):
        return LlmResult(text="Generated", model=request.model)


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
    )

    result = orchestrator.run(
        RunOptions(
            provider_override="override_provider",
            week_start=date(2026, 6, 1),
            dry_run=True,
            skip_discord=True,
        )
    )

    assert result[0].status == WorkflowStatus.COMPLETED
    assert provider.requests[0].provider_offering_id == 123


def test_orchestrator_sends_operational_notification_on_unhandled_failure(tmp_path) -> None:
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
    )

    result = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=True))

    assert result[0].status == WorkflowStatus.FAILED
    assert discord.messages[0].webhook_env == "DISCORD_OPS_WEBHOOK_URL"
    assert "provider exploded" in discord.messages[0].content
