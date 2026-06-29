from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from meal_orchestrator.artifacts import ArtifactStore
from meal_orchestrator.config.models import ArtifactConfig
from meal_orchestrator.domain import LlmRequest, LlmResult, PromptPayload
from tests.unit.helpers import canonical_menu


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
        payload=PromptPayload(user_prompt="Choose meals.", menu=canonical_menu()),
        timeout_seconds=30,
    )


def test_saves_all_artifacts(tmp_path: Path) -> None:
    store = ArtifactStore(_config(tmp_path))
    run = store.for_run("run-1", "alan")

    run.save_provider_raw({"raw": "data"})
    run.save_canonical_menu(canonical_menu())
    run.save_llm_request(_llm_request())
    run.save_llm_response(LlmResult(text="response text", model="test-model"))
    run.save_metadata({"run_id": "run-1", "status": "completed"})

    run_dir = tmp_path / "artifacts" / "alan" / "run-1"
    assert (run_dir / "provider_raw.json").exists()
    assert (run_dir / "canonical_menu.json").exists()
    assert (run_dir / "llm_request.json").exists()
    assert (run_dir / "llm_response.txt").exists()
    assert (run_dir / "metadata.json").exists()


def test_artifacts_content(tmp_path: Path) -> None:
    store = ArtifactStore(_config(tmp_path))
    run = store.for_run("run-1", "alan")

    run.save_provider_raw({"key": "value"})
    raw = json.loads((tmp_path / "artifacts" / "alan" / "run-1" / "provider_raw.json").read_text())
    assert raw == {"key": "value"}

    run.save_llm_response(LlmResult(text="meal suggestion", model="test-model"))
    text = (tmp_path / "artifacts" / "alan" / "run-1" / "llm_response.txt").read_text()
    assert text == "meal suggestion"

    run.save_llm_request(_llm_request())
    req = json.loads((tmp_path / "artifacts" / "alan" / "run-1" / "llm_request.json").read_text())
    assert req["model"] == "test-model"
    assert req["user_prompt"] == "Choose meals."
    assert "days" in req["menu"]


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
