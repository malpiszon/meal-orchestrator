from __future__ import annotations

from dataclasses import replace
from datetime import date

from meal_orchestrator.batch_runner import PendingBatchState, PendingBatchUser, save_state
from meal_orchestrator.config import AppConfig
from meal_orchestrator.config.models import BatchConfig
from meal_orchestrator.domain import LlmResult, ProviderMenuRequest, ProviderResult, WorkflowStatus
from meal_orchestrator.orchestrator import RunOptions, RunOrchestrator
from tests.unit.helpers import (
    FakeDiscordClient,
    FakeEmailClient,
    app_config,
    canonical_menu,
    week_assessment,
)


def _user(tmp_path, user_id: str):
    from tests.unit.helpers import user_config

    prompt_file = tmp_path / f"{user_id}.md"
    prompt_file.write_text(f"Choose meals for {user_id}.", encoding="utf-8")
    return replace(
        user_config(prompt_file.relative_to(tmp_path)),
        id=user_id,
        email=f"{user_id}@example.com",
    )


class RecordingProvider:
    provider_id = "example_provider"

    def get_canonical_week_menu(self, request: ProviderMenuRequest):
        return ProviderResult(menu=canonical_menu())


def _no_capability_check(model: str, **_kwargs) -> None:
    pass


def _batch_app_config(**batch_kwargs) -> AppConfig:
    return app_config(batch=BatchConfig(enabled=True, **batch_kwargs))


def test_batch_mode_submits_polls_and_delivers(tmp_path, monkeypatch) -> None:
    users = [_user(tmp_path, "alan"), _user(tmp_path, "bob")]
    submitted_rows = []

    def fake_submit_batch(rows, **_kwargs):
        submitted_rows.extend(rows)
        return "batch-1"

    def fake_get_batch(batch_id, **_kwargs):
        return {
            "status": "completed",
            "results": [
                {
                    "custom_id": row.custom_id,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "model": row.model,
                            "choices": [
                                {
                                    "message": {
                                        "content": week_assessment(
                                            row.payload.menu
                                        ).model_dump_json()
                                    }
                                }
                            ],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                        },
                    },
                    "error": None,
                }
                for row in submitted_rows
            ],
        }

    monkeypatch.setattr("meal_orchestrator.orchestrator.submit_batch", fake_submit_batch)
    monkeypatch.setattr("meal_orchestrator.orchestrator.get_batch", fake_get_batch)

    class UnusedLlmClient:
        def generate(self, request, **_kwargs):
            raise AssertionError("synchronous LLM client must not be used when batch succeeds")

    email_client = FakeEmailClient()
    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=UnusedLlmClient(),
        email_client=email_client,
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert all(r.status == WorkflowStatus.COMPLETED for r in results)
    assert len(email_client.messages) == 2
    assert not (tmp_path / ".batch_state" / "pending_batch.json").exists()
    assert not (tmp_path / ".batch_state" / "run.lock").exists()


def test_batch_mode_retries_row_failures_synchronously_with_fallback_models(
    tmp_path, monkeypatch
) -> None:
    """A failed/missing batch row must get the same resilience a sync-mode call
    would — retried synchronously, including trying configured fallback_models —
    rather than being marked permanently FAILED with no second chance.
    """
    users = [_user(tmp_path, "alan"), _user(tmp_path, "bob")]

    def fake_submit_batch(rows, **_kwargs):
        return "batch-1"

    def fake_get_batch(batch_id, **_kwargs):
        return {"status": "completed", "results": []}  # every row missing -> per-row error

    monkeypatch.setattr("meal_orchestrator.orchestrator.submit_batch", fake_submit_batch)
    monkeypatch.setattr("meal_orchestrator.orchestrator.get_batch", fake_get_batch)

    class FallbackOnlyLlmClient:
        """Only succeeds when called with the configured fallback model — proves
        the synchronous retry path (which tries fallback_models) is actually
        used for a batch row failure, not a bare permanent-FAILED synthesis."""

        def generate(self, request, **_kwargs):
            candidates = [request.model, *request.fallback_models]
            if "fallback-model" not in candidates:
                raise AssertionError("fallback_models were dropped for the batch retry path")
            return LlmResult(
                structured=week_assessment(request.payload.menu),
                model="fallback-model",
                attempt=2,
            )

    email_client = FakeEmailClient()
    orchestrator = RunOrchestrator(
        app_config=app_config(
            fallback_models=["fallback-model"], batch=BatchConfig(enabled=True)
        ),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=FallbackOnlyLlmClient(),
        email_client=email_client,
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert all(r.status == WorkflowStatus.COMPLETED for r in results)
    assert all(r.model == "fallback-model" for r in results)
    assert len(email_client.messages) == 2


def test_batch_mode_falls_back_to_sync_when_batch_times_out(tmp_path, monkeypatch) -> None:
    users = [_user(tmp_path, "alan")]

    def fake_submit_batch(rows, **_kwargs):
        return "batch-1"

    monkeypatch.setattr("meal_orchestrator.orchestrator.submit_batch", fake_submit_batch)
    monkeypatch.setattr(
        "meal_orchestrator.orchestrator.poll_until_terminal", lambda *a, **k: None
    )

    class SyncFallbackLlmClient:
        def generate(self, request, **_kwargs):
            return LlmResult(
                structured=week_assessment(request.payload.menu), model=request.model, attempt=1
            )

    monkeypatch.setenv("DISCORD_OPS_WEBHOOK_URL", "https://example.com/ops")
    discord = FakeDiscordClient()

    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=SyncFallbackLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=discord,
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert results[0].status == WorkflowStatus.COMPLETED
    assert not (tmp_path / ".batch_state" / "pending_batch.json").exists()
    fallback_msg = next(
        m for m in discord.messages if "falling back to synchronous" in m.description
    )
    assert fallback_msg.color is not None


def test_batch_resume_deadline_counts_from_original_submission_not_resume_time(
    tmp_path, monkeypatch
) -> None:
    """max_wait_hours must bound total wait from the batch's original submission,
    not reset to a fresh window every time a crashed process resumes polling —
    otherwise a batch that keeps crashing right before completing could poll
    forever, well past the configured cap.
    """
    from datetime import UTC, datetime, timedelta

    long_expired_submission = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
    save_state(
        tmp_path,
        PendingBatchState(
            run_id="run-resume",
            batch_id="batch-existing",
            submitted_at=long_expired_submission,
            week_start="2026-06-01",
            week_end="2026-06-05",
            model="test-model",
            users=[PendingBatchUser(user_id="alan", custom_id="run-resume:alan")],
        ),
    )

    get_batch_calls = []

    def fake_get_batch(batch_id, **_kwargs):
        get_batch_calls.append(batch_id)
        return {"status": "in_progress"}  # still pending, forever

    monkeypatch.setattr("meal_orchestrator.orchestrator.get_batch", fake_get_batch)

    class SyncFallbackLlmClient:
        def generate(self, request, **_kwargs):
            return LlmResult(
                structured=week_assessment(request.payload.menu), model=request.model, attempt=1
            )

    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(max_wait_hours=1),
        users=[_user(tmp_path, "alan")],
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=SyncFallbackLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    # Submission was already 5h old against a 1h cap, so the deadline must be
    # treated as already passed — a single status check, then immediate
    # fallback to sync — not a fresh 1h wait counted from "now".
    assert len(get_batch_calls) == 1
    assert results[0].status == WorkflowStatus.COMPLETED


def test_batch_mode_resumes_pending_state_instead_of_resubmitting(tmp_path, monkeypatch) -> None:
    users = [_user(tmp_path, "alan")]
    save_state(
        tmp_path,
        PendingBatchState(
            run_id="run-resume",
            batch_id="batch-existing",
            submitted_at="2026-06-01T00:00:00+00:00",
            week_start="2026-06-01",
            week_end="2026-06-05",
            model="test-model",
            users=[PendingBatchUser(user_id="alan", custom_id="run-resume:alan")],
        ),
    )

    submit_calls = []
    monkeypatch.setattr(
        "meal_orchestrator.orchestrator.submit_batch",
        lambda rows, **k: submit_calls.append(rows) or "should-not-be-called",
    )

    def fake_get_batch(batch_id, **_kwargs):
        assert batch_id == "batch-existing"
        return {
            "status": "completed",
            "results": [
                {
                    "custom_id": "run-resume:alan",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "model": "test-model",
                            "choices": [
                                {
                                    "message": {
                                        "content": week_assessment(
                                            canonical_menu()
                                        ).model_dump_json()
                                    }
                                }
                            ],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                        },
                    },
                    "error": None,
                }
            ],
        }

    monkeypatch.setattr("meal_orchestrator.orchestrator.get_batch", fake_get_batch)

    class UnusedLlmClient:
        def generate(self, request, **_kwargs):
            raise AssertionError("synchronous LLM client must not be used on batch resume")

    email_client = FakeEmailClient()
    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=UnusedLlmClient(),
        email_client=email_client,
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert submit_calls == []
    assert results[0].status == WorkflowStatus.COMPLETED
    assert len(email_client.messages) == 1
    assert not (tmp_path / ".batch_state" / "pending_batch.json").exists()


def test_batch_mode_skipped_for_dry_run(tmp_path, monkeypatch) -> None:
    users = [_user(tmp_path, "alan")]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("batch client must not be used for dry runs")

    monkeypatch.setattr("meal_orchestrator.orchestrator.submit_batch", fail_if_called)

    class SyncLlmClient:
        def generate(self, request, **_kwargs):
            return LlmResult(
                structured=week_assessment(request.payload.menu), model=request.model, attempt=1
            )

    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=SyncLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=True))

    assert results[0].status == WorkflowStatus.COMPLETED


def test_batch_submission_falls_back_to_sync_when_lock_already_held(tmp_path) -> None:
    from meal_orchestrator.batch_runner import acquire_lock

    users = [_user(tmp_path, "alan")]
    assert acquire_lock(tmp_path) is True  # simulate another invocation already running

    class SyncLlmClient:
        def generate(self, request, **_kwargs):
            return LlmResult(
                structured=week_assessment(request.payload.menu), model=request.model, attempt=1
            )

    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=SyncLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert results[0].status == WorkflowStatus.COMPLETED


def test_batch_resume_reports_failure_not_empty_success_when_lock_held(tmp_path) -> None:
    """A resume that can't acquire the lock must surface as a FAILED result
    (and a non-zero-looking outcome), not an empty list that a caller like
    cli.py's `any(status == FAILED for r in results)` would read as success.
    """
    from meal_orchestrator.batch_runner import acquire_lock

    save_state(
        tmp_path,
        PendingBatchState(
            run_id="run-locked",
            batch_id="batch-locked",
            submitted_at="2026-06-01T00:00:00+00:00",
            week_start="2026-06-01",
            week_end="2026-06-05",
            model="test-model",
            users=[PendingBatchUser(user_id="alan", custom_id="run-locked:alan")],
        ),
    )
    assert acquire_lock(tmp_path) is True  # simulate another invocation already running

    class UnusedLlmClient:
        def generate(self, request, **_kwargs):
            raise AssertionError("must not process anything while locked")

    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(),
        users=[_user(tmp_path, "alan")],
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=UnusedLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert len(results) == 1
    assert results[0].status == WorkflowStatus.FAILED
    assert "lock" in results[0].detail
    assert any(r.status == WorkflowStatus.FAILED for r in results)


def test_batch_submission_degrades_to_sync_when_state_dir_unwritable(tmp_path) -> None:
    """If .batch_state can't be created (e.g. read-only project root), the run
    must fall back to synchronous processing instead of crashing outright.
    """
    users = [_user(tmp_path, "alan")]
    # Create a plain file where the state directory would go, so mkdir(parents=True)
    # fails with OSError (FileExistsError) instead of succeeding.
    (tmp_path / ".batch_state").write_text("not a directory", encoding="utf-8")

    class SyncLlmClient:
        def generate(self, request, **_kwargs):
            return LlmResult(
                structured=week_assessment(request.payload.menu), model=request.model, attempt=1
            )

    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=SyncLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert results[0].status == WorkflowStatus.COMPLETED
