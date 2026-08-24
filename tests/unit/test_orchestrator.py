from __future__ import annotations

import threading
import time
from dataclasses import replace
from datetime import date

from meal_orchestrator.delivery.discord import COLOR_WARNING
from meal_orchestrator.domain import LlmResult, ProviderMenuRequest, ProviderResult, WorkflowStatus
from meal_orchestrator.llm import EmptyLlmResponseError, LlmFailureDetails
from meal_orchestrator.orchestrator import RunOptions, RunOrchestrator
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


class _ConcurrencyTracker:
    """Tracks the peak number of overlapping calls to `track()`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = 0
        self.peak = 0

    def track(self, work) -> None:
        with self._lock:
            self._current += 1
            self.peak = max(self.peak, self._current)
        try:
            work()
        finally:
            with self._lock:
                self._current -= 1


def _user(tmp_path, user_id: str, *, provider: str = "example_provider"):
    prompt_file = tmp_path / f"{user_id}.md"
    prompt_file.write_text(f"Choose meals for {user_id}.", encoding="utf-8")
    return replace(
        user_config(prompt_file.relative_to(tmp_path)),
        id=user_id,
        email=f"{user_id}@example.com",
        provider=provider,
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
    def __init__(self, *, attempt: int = 1, served_model: str | None = None) -> None:
        self.attempt = attempt
        self.served_model = served_model

    def generate(self, request, **_kwargs):
        return LlmResult(
            structured=week_assessment(request.payload.menu),
            model=self.served_model or request.model,
            attempt=self.attempt,
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


class TrackingLlmClient:
    """Records peak concurrent `generate()` calls via `tracker`, holding each call
    open for `delay` seconds so overlapping calls have a chance to be observed."""

    def __init__(self, tracker: _ConcurrencyTracker, *, delay: float = 0.03) -> None:
        self._tracker = tracker
        self._delay = delay

    def generate(self, request, **_kwargs):
        self._tracker.track(lambda: time.sleep(self._delay))
        return LlmResult(
            structured=week_assessment(request.payload.menu), model=request.model, attempt=1
        )


def _no_capability_check(model: str, **_kwargs) -> None:
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


def test_orchestrator_writes_run_level_metadata_for_sync_run(tmp_path) -> None:
    """A plain (non-batch) run must also leave a run-level metadata.json —
    built via the same `build_run_metadata` shared with BatchCoordinator, so
    the schema (batch_id/batch_status/aggregate_usage present as null rather
    than absent) can't drift between the two modes.
    """
    import json

    from meal_orchestrator.config.models import ArtifactConfig

    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")

    orchestrator = RunOrchestrator(
        app_config=app_config(
            artifacts=ArtifactConfig(path=tmp_path / "artifacts", retention_days=14, max_runs=10)
        ),
        users=[user_config(prompt_file.relative_to(tmp_path))],
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=FakeLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    result = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert result[0].status == WorkflowStatus.COMPLETED
    run_dirs = [d for d in (tmp_path / "artifacts").iterdir() if d.is_dir()]
    assert len(run_dirs) == 1
    metadata = json.loads((run_dirs[0] / "metadata.json").read_text())
    assert metadata["mode"] == "sync"
    assert metadata["batch_id"] is None
    assert metadata["batch_status"] is None
    assert metadata["aggregate_usage"] is None
    assert metadata["users"] == [user_config(prompt_file.relative_to(tmp_path)).id]


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
    # Run id and retry count share a single parenthetical, e.g. "(run <id>, 2 retries)" —
    # not two separate ones back to back like "(run <id>). (2 retries)".
    assert ", 2 retries)" in ops_msg.description
    assert "). (2 retries)" not in ops_msg.description
    user_msg = next(m for m in discord.messages if m.webhook_env != "DISCORD_OPS_WEBHOOK_URL")
    assert "retr" not in user_msg.description


def test_operational_notification_flags_fallback_model_on_completion(
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
        llm_client=FakeLlmClient(served_model="fallback-model"),
        email_client=FakeEmailClient(),
        discord_client=discord,
        capability_check=_no_capability_check,
    )

    orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    ops_msg = discord.messages[-1]
    assert "fallback model fallback-model" in ops_msg.description
    assert "configured primary: test-model" in ops_msg.description
    assert ops_msg.color == COLOR_WARNING


def test_operational_notification_includes_cost_tokens_and_time(tmp_path, monkeypatch) -> None:
    class UsageTrackingLlmClient:
        def generate(self, request, on_attempt=None, **_kwargs):
            if on_attempt is not None:
                on_attempt(
                    1,
                    None,
                    {
                        "model": request.model,
                        "usage": {"cost": 0.0025, "prompt_tokens": 120, "completion_tokens": 340},
                    },
                    {"model": request.model},
                )
            return LlmResult(
                structured=week_assessment(request.payload.menu), model=request.model, attempt=1
            )

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
        llm_client=UsageTrackingLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=discord,
        capability_check=_no_capability_check,
    )

    result = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert result[0].status == WorkflowStatus.COMPLETED
    assert result[0].cost == 0.0025
    assert result[0].prompt_tokens == 120
    assert result[0].completion_tokens == 340
    assert result[0].duration_seconds is not None
    assert result[0].duration_seconds >= 0
    ops_msg = discord.messages[-1]
    assert "Cost: $0.002500" in ops_msg.description
    assert "tokens: 120 in / 340 out" in ops_msg.description
    assert "time:" in ops_msg.description


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


def test_operational_notification_omits_redundant_retry_note_on_llm_failure(
    tmp_path, monkeypatch
) -> None:
    """The underlying error text already states the attempt count.

    ("openrouter failed after 3 attempt(s)"), so the failure ops message must not
    also append a separately-worded "(N retries)" note that states the same thing
    with a different (off-by-one) number and reads like a stray trailing fragment.
    """
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
    assert result[0].retry_count == 2
    ops_msg = discord.messages[0]
    assert ops_msg.webhook_env == "DISCORD_OPS_WEBHOOK_URL"
    assert "failed after 3 attempt(s)" in ops_msg.description
    assert "retries)" not in ops_msg.description


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
            return LlmResult(
                structured=week_assessment(request.payload.menu), model=request.model, attempt=1
            )

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

    def _capturing_capability_check(model: str, **_kwargs) -> None:
        captured_models.append(model)

    orchestrator = RunOrchestrator(
        app_config=app_config(),
        users=[user_config(prompt_file.relative_to(tmp_path))],
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=FakeLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_capturing_capability_check,
    )

    orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))
    orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=True))
    orchestrator.run(
        RunOptions(week_start=date(2026, 6, 1), dry_run=False, llm_model="override-model")
    )

    assert captured_models == ["test-model", "test-dry-run-model", "override-model"]


def test_orchestrator_runs_capability_check_with_configured_fallback_models(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")
    captured_fallback_models: list[list[str]] = []

    def _capturing_capability_check(model: str, *, fallback_models=None, **_kwargs) -> None:
        captured_fallback_models.append(fallback_models)

    orchestrator = RunOrchestrator(
        app_config=app_config(fallback_models=["openai/gpt-4.1-mini"]),
        users=[user_config(prompt_file.relative_to(tmp_path))],
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=FakeLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_capturing_capability_check,
    )

    orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert captured_fallback_models == [["openai/gpt-4.1-mini"]]


def test_orchestrator_fails_all_users_gracefully_when_capability_check_fails(tmp_path) -> None:
    """A capability-check failure must degrade like every other failure mode.

    Not propagate as a raw uncaught exception that crashes the whole process
    with no per-user WorkflowResult and no ops notification.
    """
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Choose meals.", encoding="utf-8")

    def _failing_capability_check(model: str, **_kwargs) -> None:
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

    def _failing_capability_check(model: str, **_kwargs) -> None:
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

    def _failing_capability_check(model: str, **_kwargs) -> None:
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


def test_menu_fetch_never_overlaps_even_with_a_high_concurrency_limit(tmp_path) -> None:
    tracker = _ConcurrencyTracker()

    class TrackedProvider:
        provider_id = "example_provider"

        def get_canonical_week_menu(self, request: ProviderMenuRequest):
            tracker.track(lambda: time.sleep(0.03))
            return ProviderResult(menu=canonical_menu())

    users = [_user(tmp_path, f"user{i}") for i in range(4)]

    orchestrator = RunOrchestrator(
        app_config=app_config(max_concurrent_users=4),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: TrackedProvider(),
        llm_client=FakeLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert all(r.status == WorkflowStatus.COMPLETED for r in results)
    assert tracker.peak == 1


def test_post_menu_steps_genuinely_overlap_across_users(tmp_path) -> None:
    users = [_user(tmp_path, f"user{i}") for i in range(3)]
    barrier = threading.Barrier(len(users), timeout=5)

    class BarrierLlmClient:
        def generate(self, request, **_kwargs):
            barrier.wait()
            return LlmResult(
                structured=week_assessment(request.payload.menu), model=request.model, attempt=1
            )

    orchestrator = RunOrchestrator(
        app_config=app_config(max_concurrent_users=len(users)),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=BarrierLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    # Every user's LLM call must have reached the barrier concurrently for all
    # of them to pass it — a sequential loop would deadlock/timeout instead.
    assert all(r.status == WorkflowStatus.COMPLETED for r in results)


def test_phase_a_isolates_menu_outcomes_from_other_users(tmp_path) -> None:
    users = [
        _user(tmp_path, "good", provider="provider_ok"),
        _user(tmp_path, "unavailable", provider="provider_unavailable"),
        _user(tmp_path, "broken", provider="provider_broken"),
    ]

    class UnavailableProvider:
        provider_id = "provider_unavailable"

        def get_canonical_week_menu(self, request: ProviderMenuRequest):
            return ProviderResult(menu=canonical_menu(complete=False))

    def factory(provider_id: str):
        if provider_id == "provider_ok":
            return RecordingProvider()
        if provider_id == "provider_unavailable":
            return UnavailableProvider()
        return FailingProvider()

    orchestrator = RunOrchestrator(
        app_config=app_config(),
        users=users,
        project_root=tmp_path,
        provider_factory=factory,
        llm_client=FakeLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    by_id = {r.user_id: r for r in results}
    assert by_id["good"].status == WorkflowStatus.COMPLETED
    assert by_id["unavailable"].status == WorkflowStatus.MENU_UNAVAILABLE
    assert by_id["broken"].status == WorkflowStatus.FAILED
    assert by_id["broken"].failed_step == "provider"


def test_phase_b_respects_configured_concurrency_limit(tmp_path) -> None:
    tracker = _ConcurrencyTracker()

    users = [_user(tmp_path, f"user{i}") for i in range(6)]

    orchestrator = RunOrchestrator(
        app_config=app_config(max_concurrent_users=2),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=TrackingLlmClient(tracker),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert all(r.status == WorkflowStatus.COMPLETED for r in results)
    assert tracker.peak == 2


def test_results_preserve_selected_users_order_regardless_of_completion_order(tmp_path) -> None:
    users = [_user(tmp_path, "first"), _user(tmp_path, "second"), _user(tmp_path, "third")]

    class DelayedByPromptLlmClient:
        _delays = {"first": 0.15, "second": 0.0, "third": 0.0}

        def generate(self, request, **_kwargs):
            for marker, delay in self._delays.items():
                if marker in request.payload.user_prompt:
                    time.sleep(delay)
                    break
            return LlmResult(
                structured=week_assessment(request.payload.menu), model=request.model, attempt=1
            )

    orchestrator = RunOrchestrator(
        app_config=app_config(max_concurrent_users=3),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=DelayedByPromptLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert [r.user_id for r in results] == ["first", "second", "third"]
    assert all(r.status == WorkflowStatus.COMPLETED for r in results)


def test_run_options_max_concurrent_users_overrides_config_default(tmp_path) -> None:
    tracker = _ConcurrencyTracker()

    users = [_user(tmp_path, f"user{i}") for i in range(4)]

    orchestrator = RunOrchestrator(
        app_config=app_config(max_concurrent_users=4),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=TrackingLlmClient(tracker),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(
        RunOptions(week_start=date(2026, 6, 1), dry_run=False, max_concurrent_users=1)
    )

    assert all(r.status == WorkflowStatus.COMPLETED for r in results)
    assert tracker.peak == 1


def test_non_positive_max_concurrent_users_override_falls_back_to_config(tmp_path) -> None:
    """A malformed override (0, negative) must not crash ThreadPoolExecutor construction.

    It should be treated as "no override" and fall back to the configured value,
    rather than propagating an unhandled ValueError out of run().
    """
    tracker = _ConcurrencyTracker()
    users = [_user(tmp_path, f"user{i}") for i in range(6)]

    for invalid_override in (0, -1):
        tracker.peak = 0
        orchestrator = RunOrchestrator(
            app_config=app_config(max_concurrent_users=2),
            users=users,
            project_root=tmp_path,
            provider_factory=lambda provider_id: RecordingProvider(),
            llm_client=TrackingLlmClient(tracker),
            email_client=FakeEmailClient(),
            discord_client=FakeDiscordClient(),
            capability_check=_no_capability_check,
        )

        results = orchestrator.run(
            RunOptions(
                week_start=date(2026, 6, 1), dry_run=False, max_concurrent_users=invalid_override
            )
        )

        assert all(r.status == WorkflowStatus.COMPLETED for r in results)
        assert tracker.peak == 2


def test_phase_b_worker_exception_outside_own_handling_is_isolated(tmp_path, monkeypatch) -> None:
    """A bug that raises out of execute_from_menu itself — bypassing its own internal
    try/except — must still degrade to a per-user FAILED result labeled failed_step
    "worker" (distinct from the Phase A "setup" failure label), not crash the run.
    """

    def _broken_execute_from_menu(self, *args, **kwargs):
        raise RuntimeError("worker exploded outside its own try/except")

    monkeypatch.setattr(UserWorkflowExecutor, "execute_from_menu", _broken_execute_from_menu)

    users = [_user(tmp_path, "alice"), _user(tmp_path, "bob")]

    orchestrator = RunOrchestrator(
        app_config=app_config(),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=FakeLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert len(results) == 2
    for result in results:
        assert result.status == WorkflowStatus.FAILED
        assert result.failed_step == "worker"
        assert "worker exploded" in result.detail
