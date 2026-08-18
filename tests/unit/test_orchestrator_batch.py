from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

from meal_orchestrator.batch_coordinator import BatchCoordinator
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


def _batch_app_config(tmp_path, **batch_kwargs) -> AppConfig:
    # 0 so `poll_until_terminal`'s pre-first-check delay/backoff (real
    # time.sleep, not injectable through the orchestrator) doesn't actually
    # slow tests down.
    batch_kwargs.setdefault("initial_check_delay_seconds", 0)
    batch_kwargs.setdefault("initial_poll_interval_seconds", 0)
    return app_config(
        batch=BatchConfig(enabled=True, state_dir=tmp_path / "batch_state", **batch_kwargs)
    )


def test_batch_state_persists_through_delivery_not_cleared_before_it(tmp_path, monkeypatch) -> None:
    """The pending-batch state file must stay in place until delivery is
    actually finished, not right after polling reports COMPLETED — otherwise
    a crash during delivery (email/Discord for many users) loses the ability
    to resume, forcing a full duplicate (billable) batch resubmission.
    """
    users = [_user(tmp_path, "alan")]
    state_path = tmp_path / "batch_state" / "pending_batch.json"
    captured_custom_ids: list[str] = []

    def capturing_submit_batch(rows, **_kwargs):
        captured_custom_ids.extend(row.custom_id for row in rows)
        return "batch-1"

    def fake_get_batch(batch_id, **_kwargs):
        return {
            "status": "completed",
            "results": [
                {
                    "custom_id": custom_id,
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
                for custom_id in captured_custom_ids
            ],
        }

    monkeypatch.setattr("meal_orchestrator.batch_coordinator.submit_batch", capturing_submit_batch)
    monkeypatch.setattr("meal_orchestrator.batch_coordinator.get_batch", fake_get_batch)

    state_existed_during_delivery = []
    original_deliver = BatchCoordinator._deliver

    def spy_deliver(self, *args, **kwargs):
        state_existed_during_delivery.append(state_path.exists())
        return original_deliver(self, *args, **kwargs)

    monkeypatch.setattr(BatchCoordinator, "_deliver", spy_deliver)

    class UnusedLlmClient:
        def generate(self, request, **_kwargs):
            raise AssertionError("synchronous LLM client must not be used when batch succeeds")

    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(tmp_path),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=UnusedLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert results[0].status == WorkflowStatus.COMPLETED
    assert state_existed_during_delivery == [True]
    assert not state_path.exists()


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

    monkeypatch.setattr("meal_orchestrator.batch_coordinator.submit_batch", fake_submit_batch)
    monkeypatch.setattr("meal_orchestrator.batch_coordinator.get_batch", fake_get_batch)

    class UnusedLlmClient:
        def generate(self, request, **_kwargs):
            raise AssertionError("synchronous LLM client must not be used when batch succeeds")

    email_client = FakeEmailClient()
    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(tmp_path),
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
    assert not (tmp_path / "batch_state" / "pending_batch.json").exists()
    assert not (tmp_path / "batch_state" / "run.lock").exists()


def test_batch_mode_saves_raw_batch_result_as_artifact_not_just_a_log_line(
    tmp_path, monkeypatch
) -> None:
    """The raw OpenRouter batch response (aggregate cost/usage, request
    counts, every row's full LLM output) must be durably saved — logging it
    on every poll would dump an arbitrarily large payload to stdout, so it's
    only ever written to a dedicated artifact file, once per run.
    """
    from meal_orchestrator.config.models import ArtifactConfig

    users = [_user(tmp_path, "alan")]
    submitted_rows = []

    def fake_submit_batch(rows, **_kwargs):
        submitted_rows.extend(rows)
        return "batch-1"

    def fake_get_batch(batch_id, **_kwargs):
        return {
            "status": "completed",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.001234},
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

    monkeypatch.setattr("meal_orchestrator.batch_coordinator.submit_batch", fake_submit_batch)
    monkeypatch.setattr("meal_orchestrator.batch_coordinator.get_batch", fake_get_batch)

    class UnusedLlmClient:
        def generate(self, request, **_kwargs):
            raise AssertionError("synchronous LLM client must not be used when batch succeeds")

    orchestrator = RunOrchestrator(
        app_config=app_config(
            batch=BatchConfig(
                enabled=True,
                state_dir=tmp_path / "batch_state",
                initial_check_delay_seconds=0,
                initial_poll_interval_seconds=0,
            ),
            artifacts=ArtifactConfig(
                path=tmp_path / "artifacts", retention_days=14, max_runs_per_user=10
            ),
        ),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=UnusedLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert results[0].status == WorkflowStatus.COMPLETED
    batch_files = list((tmp_path / "artifacts" / "batches").iterdir())
    assert len(batch_files) == 1  # one file per run
    saved = json.loads(batch_files[0].read_text())
    assert saved["usage"]["cost"] == 0.001234
    assert saved["results"][0]["custom_id"].endswith(":alan")


def test_batch_mode_status_check_log_is_small_not_the_full_response(
    tmp_path, monkeypatch, caplog
) -> None:
    """The per-check status log must stay a small, fixed-size line — the
    full response (potentially every row's complete LLM output) belongs
    only in the batch-result artifact, not in every poll's log record.
    """
    import logging

    users = [_user(tmp_path, "alan")]
    submitted_rows = []
    huge_content = week_assessment(canonical_menu()).model_dump_json()

    def fake_submit_batch(rows, **_kwargs):
        submitted_rows.extend(rows)
        return "batch-1"

    def fake_get_batch(batch_id, **_kwargs):
        return {
            "status": "completed",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.001234},
            "request_counts": {"total": 1, "completed": 1, "failed": 0},
            "results": [
                {
                    "custom_id": row.custom_id,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "model": row.model,
                            "choices": [{"message": {"content": huge_content}}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                        },
                    },
                    "error": None,
                }
                for row in submitted_rows
            ],
        }

    monkeypatch.setattr("meal_orchestrator.batch_coordinator.submit_batch", fake_submit_batch)
    monkeypatch.setattr("meal_orchestrator.batch_coordinator.get_batch", fake_get_batch)

    class UnusedLlmClient:
        def generate(self, request, **_kwargs):
            raise AssertionError("synchronous LLM client must not be used when batch succeeds")

    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(tmp_path),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=UnusedLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    with caplog.at_level(logging.INFO):
        results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert results[0].status == WorkflowStatus.COMPLETED
    check_logs = [r for r in caplog.records if "batch status check" in r.message]
    assert len(check_logs) == 1
    assert check_logs[0].status == "completed"
    assert check_logs[0].request_counts == {"total": 1, "completed": 1, "failed": 0}
    assert not hasattr(check_logs[0], "batch_data")
    assert huge_content not in check_logs[0].getMessage()


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

    monkeypatch.setattr("meal_orchestrator.batch_coordinator.submit_batch", fake_submit_batch)
    monkeypatch.setattr("meal_orchestrator.batch_coordinator.get_batch", fake_get_batch)

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

    monkeypatch.setenv("DISCORD_OPS_WEBHOOK_URL", "https://example.com/ops")
    email_client = FakeEmailClient()
    discord = FakeDiscordClient()
    orchestrator = RunOrchestrator(
        app_config=app_config(
            fallback_models=["fallback-model"],
            batch=BatchConfig(
                enabled=True,
                state_dir=tmp_path / "batch_state",
                initial_check_delay_seconds=0,
                initial_poll_interval_seconds=0,
            ),
        ),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=FallbackOnlyLlmClient(),
        email_client=email_client,
        discord_client=discord,
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert all(r.status == WorkflowStatus.COMPLETED for r in results)
    assert all(r.model == "fallback-model" for r in results)
    assert len(email_client.messages) == 2
    # Both rows needed synchronous retry — a systemic-looking failure — so a
    # single aggregate ops alert must fire alongside the per-user ones, not
    # just N individual "completed" notifications with no pattern visible.
    summary_msg = next(
        m for m in discord.messages if "batch rows required synchronous fallback" in m.description
    )
    assert "2/2" in summary_msg.description
    assert "alan" in summary_msg.description
    assert "bob" in summary_msg.description


def test_batch_mode_single_row_failure_still_sends_aggregate_alert(tmp_path, monkeypatch) -> None:
    """Even a single failed row (not just a mass/systemic failure) gets the
    summary alert — it's cheap to send and there's no safe threshold below
    which a paid-for batch row silently needing a full-price retry should
    stay invisible at the aggregate level.
    """
    users = [_user(tmp_path, "alan"), _user(tmp_path, "bob")]
    monkeypatch.setenv("DISCORD_OPS_WEBHOOK_URL", "https://example.com/ops")

    submitted_ids: list[str] = []

    def capturing_submit_batch(rows, **_kwargs):
        submitted_ids.extend(row.custom_id for row in rows)
        return "batch-1"

    def fake_get_batch(batch_id, **_kwargs):
        # Only alan's row comes back completed — bob's is simply absent from
        # the results, which parse_batch_results reports as "missing_from_batch".
        completed_ids = [row_id for row_id in submitted_ids if "alan" in row_id]
        return {
            "status": "completed",
            "results": [
                {
                    "custom_id": row_id,
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
                for row_id in completed_ids
            ],
        }

    monkeypatch.setattr("meal_orchestrator.batch_coordinator.submit_batch", capturing_submit_batch)
    monkeypatch.setattr("meal_orchestrator.batch_coordinator.get_batch", fake_get_batch)

    class SyncFallbackLlmClient:
        def generate(self, request, **_kwargs):
            return LlmResult(
                structured=week_assessment(request.payload.menu), model=request.model, attempt=1
            )

    discord = FakeDiscordClient()
    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(tmp_path),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=SyncFallbackLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=discord,
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert all(r.status == WorkflowStatus.COMPLETED for r in results)
    summary_msg = next(
        m for m in discord.messages if "batch rows required synchronous fallback" in m.description
    )
    assert "1/2" in summary_msg.description
    assert "bob" in summary_msg.description


def test_batch_mode_no_aggregate_alert_when_every_row_succeeds(tmp_path, monkeypatch) -> None:
    users = [_user(tmp_path, "alan")]
    monkeypatch.setenv("DISCORD_OPS_WEBHOOK_URL", "https://example.com/ops")
    submitted_ids: list[str] = []

    def capturing_submit_batch(rows, **_kwargs):
        submitted_ids.extend(row.custom_id for row in rows)
        return "batch-1"

    def fake_get_batch(batch_id, **_kwargs):
        return {
            "status": "completed",
            "results": [
                {
                    "custom_id": custom_id,
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
                for custom_id in submitted_ids
            ],
        }

    monkeypatch.setattr("meal_orchestrator.batch_coordinator.submit_batch", capturing_submit_batch)
    monkeypatch.setattr("meal_orchestrator.batch_coordinator.get_batch", fake_get_batch)

    class UnusedLlmClient:
        def generate(self, request, **_kwargs):
            raise AssertionError("synchronous LLM client must not be used when batch succeeds")

    discord = FakeDiscordClient()
    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(tmp_path),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=UnusedLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=discord,
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert all(r.status == WorkflowStatus.COMPLETED for r in results)
    assert not any(
        "batch rows required synchronous fallback" in m.description for m in discord.messages
    )


def test_batch_mode_falls_back_to_sync_when_batch_times_out(tmp_path, monkeypatch) -> None:
    users = [_user(tmp_path, "alan")]

    def fake_submit_batch(rows, **_kwargs):
        return "batch-1"

    monkeypatch.setattr("meal_orchestrator.batch_coordinator.submit_batch", fake_submit_batch)
    monkeypatch.setattr(
        "meal_orchestrator.batch_coordinator.poll_until_terminal", lambda *a, **k: None
    )

    class SyncFallbackLlmClient:
        def generate(self, request, **_kwargs):
            return LlmResult(
                structured=week_assessment(request.payload.menu), model=request.model, attempt=1
            )

    monkeypatch.setenv("DISCORD_OPS_WEBHOOK_URL", "https://example.com/ops")
    discord = FakeDiscordClient()

    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(tmp_path),
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
    assert not (tmp_path / "batch_state" / "pending_batch.json").exists()
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
        tmp_path / "batch_state",
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

    monkeypatch.setattr("meal_orchestrator.batch_coordinator.get_batch", fake_get_batch)

    class SyncFallbackLlmClient:
        def generate(self, request, **_kwargs):
            return LlmResult(
                structured=week_assessment(request.payload.menu), model=request.model, attempt=1
            )

    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(tmp_path, max_wait_hours=1),
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
        tmp_path / "batch_state",
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
        "meal_orchestrator.batch_coordinator.submit_batch",
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

    monkeypatch.setattr("meal_orchestrator.batch_coordinator.get_batch", fake_get_batch)

    class UnusedLlmClient:
        def generate(self, request, **_kwargs):
            raise AssertionError("synchronous LLM client must not be used on batch resume")

    email_client = FakeEmailClient()
    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(tmp_path),
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
    assert not (tmp_path / "batch_state" / "pending_batch.json").exists()


def test_batch_resume_saves_raw_batch_result_as_artifact(tmp_path, monkeypatch) -> None:
    """The artifact_store=... wiring added to submit_and_process must also
    reach the resume path — resume() calls the same _await_and_deliver, so a
    resumed batch's raw response should be saved exactly like a fresh one.
    """
    from meal_orchestrator.config.models import ArtifactConfig

    users = [_user(tmp_path, "alan")]
    save_state(
        tmp_path / "batch_state",
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

    monkeypatch.setattr(
        "meal_orchestrator.batch_coordinator.submit_batch",
        lambda rows, **k: "should-not-be-called",
    )

    def fake_get_batch(batch_id, **_kwargs):
        return {
            "status": "completed",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.000567},
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

    monkeypatch.setattr("meal_orchestrator.batch_coordinator.get_batch", fake_get_batch)

    class UnusedLlmClient:
        def generate(self, request, **_kwargs):
            raise AssertionError("synchronous LLM client must not be used on batch resume")

    orchestrator = RunOrchestrator(
        app_config=app_config(
            batch=BatchConfig(enabled=True, state_dir=tmp_path / "batch_state"),
            artifacts=ArtifactConfig(
                path=tmp_path / "artifacts", retention_days=14, max_runs_per_user=10
            ),
        ),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=UnusedLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert results[0].status == WorkflowStatus.COMPLETED
    saved = json.loads((tmp_path / "artifacts" / "batches" / "run-resume.json").read_text())
    assert saved["usage"]["cost"] == 0.000567


def test_batch_resume_warns_when_a_pending_user_was_removed_from_config(
    tmp_path, monkeypatch, caplog
) -> None:
    """A user present in the saved batch state but no longer in the current
    user config must not be silently dropped — their share of the batch was
    already submitted and billed, so losing it needs to be visible in logs.
    """
    import logging

    users = [_user(tmp_path, "alan")]  # "bob" was removed from config
    save_state(
        tmp_path / "batch_state",
        PendingBatchState(
            run_id="run-resume",
            batch_id="batch-existing",
            submitted_at="2026-06-01T00:00:00+00:00",
            week_start="2026-06-01",
            week_end="2026-06-05",
            model="test-model",
            users=[
                PendingBatchUser(user_id="alan", custom_id="run-resume:alan"),
                PendingBatchUser(user_id="bob", custom_id="run-resume:bob"),
            ],
        ),
    )

    monkeypatch.setattr(
        "meal_orchestrator.batch_coordinator.submit_batch", lambda rows, **k: "should-not-be-called"
    )

    def fake_get_batch(batch_id, **_kwargs):
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

    monkeypatch.setattr("meal_orchestrator.batch_coordinator.get_batch", fake_get_batch)

    class UnusedLlmClient:
        def generate(self, request, **_kwargs):
            raise AssertionError("synchronous LLM client must not be used on batch resume")

    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(tmp_path),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=UnusedLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=FakeDiscordClient(),
        capability_check=_no_capability_check,
    )

    with caplog.at_level(logging.WARNING):
        results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert results[0].status == WorkflowStatus.COMPLETED
    assert any(
        "bob" in record.message and "no longer in the user config" in record.message
        for record in caplog.records
    )


def test_batch_mode_skipped_for_dry_run(tmp_path, monkeypatch) -> None:
    users = [_user(tmp_path, "alan")]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("batch client must not be used for dry runs")

    monkeypatch.setattr("meal_orchestrator.batch_coordinator.submit_batch", fail_if_called)

    class SyncLlmClient:
        def generate(self, request, **_kwargs):
            return LlmResult(
                structured=week_assessment(request.payload.menu), model=request.model, attempt=1
            )

    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(tmp_path),
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


def test_batch_submission_falls_back_to_sync_when_lock_already_held(tmp_path, monkeypatch) -> None:
    from meal_orchestrator.batch_runner import acquire_lock

    monkeypatch.setenv("DISCORD_OPS_WEBHOOK_URL", "https://example.com/ops")
    users = [_user(tmp_path, "alan")]
    assert acquire_lock(tmp_path / "batch_state") is True  # simulate another invocation running

    class SyncLlmClient:
        def generate(self, request, **_kwargs):
            return LlmResult(
                structured=week_assessment(request.payload.menu), model=request.model, attempt=1
            )

    discord = FakeDiscordClient()
    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(tmp_path),
        users=users,
        project_root=tmp_path,
        provider_factory=lambda provider_id: RecordingProvider(),
        llm_client=SyncLlmClient(),
        email_client=FakeEmailClient(),
        discord_client=discord,
        capability_check=_no_capability_check,
    )

    results = orchestrator.run(RunOptions(week_start=date(2026, 6, 1), dry_run=False))

    assert results[0].status == WorkflowStatus.COMPLETED
    # The submit-side lock-busy fallback must alert ops the same way the
    # timeout/failure fallback does — silently falling back to full-price
    # synchronous calls with only a log line would go unnoticed.
    fallback_msg = next(
        m for m in discord.messages if "falling back to synchronous" in m.description
    )
    assert "run lock already held" in fallback_msg.description


def test_batch_resume_reports_failure_not_empty_success_when_lock_held(tmp_path) -> None:
    """A resume that can't acquire the lock must surface as a FAILED result
    (and a non-zero-looking outcome), not an empty list that a caller like
    cli.py's `any(status == FAILED for r in results)` would read as success.
    """
    from meal_orchestrator.batch_runner import acquire_lock

    save_state(
        tmp_path / "batch_state",
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
    assert acquire_lock(tmp_path / "batch_state") is True  # simulate another invocation running

    class UnusedLlmClient:
        def generate(self, request, **_kwargs):
            raise AssertionError("must not process anything while locked")

    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(tmp_path),
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
    """If llm.batch.state_dir can't be created (e.g. read-only/unmounted), the
    run must fall back to synchronous processing instead of crashing outright.
    """
    users = [_user(tmp_path, "alan")]
    # Create a plain file where the state directory would go, so mkdir(parents=True)
    # fails with OSError (FileExistsError) instead of succeeding.
    (tmp_path / "batch_state").write_text("not a directory", encoding="utf-8")

    class SyncLlmClient:
        def generate(self, request, **_kwargs):
            return LlmResult(
                structured=week_assessment(request.payload.menu), model=request.model, attempt=1
            )

    orchestrator = RunOrchestrator(
        app_config=_batch_app_config(tmp_path),
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
