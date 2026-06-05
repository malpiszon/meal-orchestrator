from __future__ import annotations

from datetime import date

from meal_orchestrator.domain import LlmResult, ProviderMenuRequest, RunContext, WorkflowStatus
from meal_orchestrator.workflow import UserWorkflowExecutor
from tests.unit.helpers import (
    FakeDiscordClient,
    FakeEmailClient,
    app_config,
    canonical_menu,
    user_config,
)


class FakeProvider:
    provider_id = "example_provider"

    def __init__(self, *, complete: bool = True) -> None:
        self.complete = complete
        self.requests: list[ProviderMenuRequest] = []

    def get_canonical_week_menu(self, request: ProviderMenuRequest):
        self.requests.append(request)
        return canonical_menu(complete=self.complete)


class FakeLlmClient:
    def __init__(self) -> None:
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return LlmResult(text="Generated meal plan", model=request.model)


def test_dry_run_builds_prompt_without_llm_or_email(tmp_path) -> None:
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
    assert llm.requests == []
    assert email.messages == []
    assert len(discord.messages) == 1


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
    assert email.messages[0].body == "Generated meal plan"
    assert email.idempotency_keys == ["run-1:alan:email"]


def test_skip_email_only_suppresses_email(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    llm = FakeLlmClient()
    email = FakeEmailClient()
    discord = FakeDiscordClient()

    _executor(tmp_path, FakeProvider(), llm, email, discord).execute(
        user_config(PathLikePrompt(prompt_file, tmp_path)),
        _context(dry_run=False, skip_email=True),
    )

    assert len(llm.requests) == 1
    assert email.messages == []
    assert len(discord.messages) == 1


def test_incomplete_menu_skips_llm_and_email(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
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


def _executor(tmp_path, provider, llm, email, discord) -> UserWorkflowExecutor:
    return UserWorkflowExecutor(
        app_config=app_config(),
        provider=provider,
        llm_client=llm,
        email_client=email,
        discord_client=discord,
        project_root=tmp_path,
    )


def _context(*, dry_run: bool, skip_email: bool = False) -> RunContext:
    return RunContext(
        run_id="run-1",
        week_start=date(2026, 6, 1),
        week_end=date(2026, 6, 5),
        dry_run=dry_run,
        skip_email=skip_email,
        skip_discord=False,
        provider_id="example_provider",
    )


def PathLikePrompt(prompt_file, project_root):
    return prompt_file.relative_to(project_root)
