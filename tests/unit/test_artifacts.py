from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from meal_orchestrator.artifacts import ArtifactStore
from meal_orchestrator.config.models import ArtifactConfig
from meal_orchestrator.domain import LlmRequest, LlmResult, PromptPayload
from tests.unit.helpers import canonical_menu, week_assessment


def _config(
    tmp_path: Path, *, retention_days: int = 14, max_runs_per_user: int = 10
) -> ArtifactConfig:
    return ArtifactConfig(
        path=tmp_path / "artifacts",
        retention_days=retention_days,
        max_runs_per_user=max_runs_per_user,
    )


def _llm_request() -> LlmRequest:
    return LlmRequest(
        model="test-model",
        payload=PromptPayload(
            app_prompt="Score every variant.", user_prompt="Choose meals.", menu=canonical_menu()
        ),
        timeout_seconds=30,
    )


def _llm_result() -> LlmResult:
    return LlmResult(structured=week_assessment(canonical_menu()), model="test-model", attempt=1)


def test_saves_all_artifacts(tmp_path: Path) -> None:
    store = ArtifactStore(_config(tmp_path))
    run = store.for_run("run-1", "alan")

    run.save_provider_raw({"raw": "data"})
    run.save_canonical_menu(canonical_menu())
    run.save_llm_request(_llm_request())
    run.save_llm_response(_llm_result())
    run.save_metadata({"run_id": "run-1", "status": "completed"})

    run_dir = tmp_path / "artifacts" / "alan" / "run-1"
    assert (run_dir / "provider_raw.json").exists()
    assert (run_dir / "canonical_menu.json").exists()
    assert (run_dir / "llm_request.json").exists()
    assert (run_dir / "llm_response.json").exists()
    assert (run_dir / "metadata.json").exists()


def test_artifacts_content(tmp_path: Path) -> None:
    store = ArtifactStore(_config(tmp_path))
    run = store.for_run("run-1", "alan")

    run.save_provider_raw({"key": "value"})
    raw = json.loads((tmp_path / "artifacts" / "alan" / "run-1" / "provider_raw.json").read_text())
    assert raw == {"key": "value"}

    run.save_llm_response(_llm_result())
    response = json.loads(
        (tmp_path / "artifacts" / "alan" / "run-1" / "llm_response.json").read_text()
    )
    assert response == week_assessment(canonical_menu()).model_dump(mode="json")

    run.save_llm_request(_llm_request())
    req = json.loads((tmp_path / "artifacts" / "alan" / "run-1" / "llm_request.json").read_text())
    assert req["model"] == "test-model"
    assert req["app_prompt"] == "Score every variant."
    assert req["user_prompt"] == "Choose meals."
    assert "days" in req["menu"]


def test_skips_llm_response_artifact_when_structured_is_invalid(tmp_path: Path) -> None:
    store = ArtifactStore(_config(tmp_path))
    run = store.for_run("run-1", "example")

    run.save_llm_response(
        LlmResult(structured=None, model="test-model", attempt=1)  # type: ignore[arg-type]
    )

    assert not (tmp_path / "artifacts" / "example" / "run-1" / "llm_response.json").exists()


def test_noop_when_no_config(tmp_path: Path) -> None:
    store = ArtifactStore(None)
    run = store.for_run("run-1", "alan")
    run.save_metadata({"status": "completed"})
    assert not (tmp_path / "artifacts").exists()


def test_noop_when_mkdir_fails(tmp_path: Path) -> None:
    unwritable = tmp_path / "readonly"
    unwritable.mkdir()
    unwritable.chmod(0o555)
    try:
        config = ArtifactConfig(path=unwritable, retention_days=14, max_runs_per_user=10)
        store = ArtifactStore(config)
        run = store.for_run("run-1", "alan")
        run.save_metadata({"status": "completed"})
        assert not (unwritable / "alan").exists()
    finally:
        unwritable.chmod(0o755)


def test_cleanup_removes_old_runs(tmp_path: Path) -> None:
    store = ArtifactStore(_config(tmp_path, retention_days=7))
    artifacts_dir = tmp_path / "artifacts"

    old_dir = artifacts_dir / "alan" / "old-run"
    new_dir = artifacts_dir / "alan" / "new-run"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)

    old_time = (datetime.now(UTC) - timedelta(days=8)).timestamp()
    os.utime(old_dir, (old_time, old_time))

    store.cleanup()

    assert not old_dir.exists()
    assert new_dir.exists()


def test_cleanup_respects_max_runs_per_user(tmp_path: Path) -> None:
    store = ArtifactStore(_config(tmp_path, max_runs_per_user=2))
    artifacts_dir = tmp_path / "artifacts"

    for i in range(4):
        run_dir = artifacts_dir / "alan" / f"run-{i}"
        run_dir.mkdir(parents=True)
        t = time.time() - (4 - i)
        os.utime(run_dir, (t, t))

    store.cleanup()

    remaining = sorted(d.name for d in (artifacts_dir / "alan").iterdir())
    assert remaining == ["run-2", "run-3"]


def test_cleanup_noop_when_path_missing(tmp_path: Path) -> None:
    store = ArtifactStore(_config(tmp_path))
    store.cleanup()


def test_save_batch_result_writes_one_file_per_run(tmp_path: Path) -> None:
    store = ArtifactStore(_config(tmp_path))

    store.save_batch_result("run-1", {"id": "batch-1", "status": "completed"})

    path = tmp_path / "artifacts" / "batches" / "run-1.json"
    assert json.loads(path.read_text()) == {"id": "batch-1", "status": "completed"}


def test_save_batch_result_noop_when_no_config(tmp_path: Path) -> None:
    store = ArtifactStore(None)
    store.save_batch_result("run-1", {"id": "batch-1", "status": "completed"})
    assert not (tmp_path / "artifacts").exists()


def test_save_batch_result_returns_true_on_success(tmp_path: Path) -> None:
    store = ArtifactStore(_config(tmp_path))
    assert store.save_batch_result("run-1", {"id": "batch-1"}) is True


def test_save_batch_result_returns_false_and_does_not_raise_on_unserializable_data(
    tmp_path: Path,
) -> None:
    """The write must degrade gracefully (like every other artifact writer)
    for any write failure, not just OSError — e.g. a value json.dumps can't
    serialize at all (a real-world case: a lone/unpaired surrogate codepoint
    in LLM-generated text raises UnicodeEncodeError, not OSError).
    """
    store = ArtifactStore(_config(tmp_path))

    result = store.save_batch_result("run-1", {"bad": object()})  # not JSON-serializable

    assert result is False
    assert not (tmp_path / "artifacts" / "batches" / "run-1.json").exists()


def test_for_run_disabled_when_user_id_collides_with_reserved_batches_name(
    tmp_path: Path,
) -> None:
    """A user configured with id "batches" would otherwise collide with the
    reserved directory batch results live in and silently escape per-user
    retention — artifacts must be disabled for that user instead.
    """
    store = ArtifactStore(_config(tmp_path))

    run = store.for_run("run-1", "batches")
    run.save_metadata({"status": "completed"})

    assert not (tmp_path / "artifacts" / "batches" / "run-1").exists()


def test_cleanup_removes_expired_batch_results(tmp_path: Path) -> None:
    """batches/ holds one flat file per run, not per-user run directories, so
    it gets its own time-based cleanup here rather than being walked by
    `_cleanup_user_dir`'s per-user (`max_runs_per_user`) logic.
    """
    store = ArtifactStore(_config(tmp_path, retention_days=7))
    store.save_batch_result("old-run", {"id": "batch-1", "status": "completed"})
    store.save_batch_result("new-run", {"id": "batch-2", "status": "completed"})

    old_file = tmp_path / "artifacts" / "batches" / "old-run.json"
    new_file = tmp_path / "artifacts" / "batches" / "new-run.json"
    old_time = (datetime.now(UTC) - timedelta(days=8)).timestamp()
    os.utime(old_file, (old_time, old_time))

    store.cleanup()

    assert not old_file.exists()
    assert new_file.exists()
