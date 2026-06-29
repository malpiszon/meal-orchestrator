from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from meal_orchestrator.artifacts import ArtifactStore
from meal_orchestrator.config.models import ArtifactConfig
from meal_orchestrator.domain import (
    LlmResult,
    ProviderMenuRequest,
    ProviderResult,
    RunContext,
    WorkflowStatus,
)
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
        return ProviderResult(
            menu=canonical_menu(complete=self.complete), raw_response={"raw": True}
        )


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
    assert (run_dir / "llm_response.txt").exists()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["user_id"] == "alan"


def test_llm_request_artifact_saved_on_dry_run(tmp_path: Path) -> None:
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
    assert not (run_dir / "llm_response.txt").exists()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "completed"


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


def _executor(tmp_path, provider, llm, email, discord, artifact_store=None) -> UserWorkflowExecutor:
    return UserWorkflowExecutor(
        app_config=app_config(),
        provider=provider,
        llm_client=llm,
        email_client=email,
        discord_client=discord,
        project_root=tmp_path,
        artifact_store=artifact_store,
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
