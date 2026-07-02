from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from meal_orchestrator.config.models import ArtifactConfig
from meal_orchestrator.domain import CanonicalMenu, LlmRequest, LlmResult

logger = logging.getLogger(__name__)


class RunArtifacts:
    """No-op artifact writer — used when artifacts are disabled."""

    def save_provider_raw(self, raw: Any) -> None:
        pass

    def save_canonical_menu(self, menu: CanonicalMenu) -> None:
        pass

    def save_llm_request(self, request: LlmRequest) -> None:
        pass

    def save_llm_response(self, result: LlmResult) -> None:
        pass

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        pass


class _FilesystemRunArtifacts(RunArtifacts):
    def __init__(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir

    def save_provider_raw(self, raw: Any) -> None:
        self._write_safe(
            "provider_raw.json",
            lambda: _write_json(self._run_dir / "provider_raw.json", raw),
        )

    def save_canonical_menu(self, menu: CanonicalMenu) -> None:
        self._write_safe(
            "canonical_menu.json",
            lambda: _write_json(self._run_dir / "canonical_menu.json", menu.to_compact_dict()),
        )

    def save_llm_request(self, request: LlmRequest) -> None:
        self._write_safe(
            "llm_request.json",
            lambda: _write_json(
                self._run_dir / "llm_request.json",
                {
                    "model": request.model,
                    "timeout_seconds": request.timeout_seconds,
                    "user_prompt": request.payload.user_prompt,
                    "menu": request.payload.menu.to_compact_dict(),
                },
            ),
        )

    def save_llm_response(self, result: LlmResult) -> None:
        self._write_safe(
            "llm_response.txt",
            lambda: (self._run_dir / "llm_response.txt").write_text(result.text, encoding="utf-8"),
        )

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        self._write_safe(
            "metadata.json",
            lambda: _write_json(self._run_dir / "metadata.json", metadata),
        )

    def _write_safe(self, name: str, write: Any) -> None:
        try:
            write()
        except Exception:
            logger.warning("artifact write failed: %s", self._run_dir / name, exc_info=True)


class ArtifactStore:
    """Manages artifact persistence and cleanup for workflow runs."""

    def __init__(self, config: ArtifactConfig | None = None) -> None:
        self._config = config

    def for_run(self, run_id: str, user_id: str) -> RunArtifacts:
        if not self._config:
            return RunArtifacts()
        run_dir = self._config.path / user_id / run_id
        try:
            return _FilesystemRunArtifacts(run_dir)
        except OSError:
            logger.warning(
                "artifact store: failed to create run directory %s — artifacts disabled for "
                "this run",
                run_dir,
                exc_info=True,
                extra={"run_id": run_id, "user_id": user_id, "step": "artifacts"},
            )
            return RunArtifacts()

    def cleanup(self) -> None:
        if not self._config:
            return
        base = self._config.path
        if not base.exists():
            return
        cutoff = datetime.now(UTC) - timedelta(days=self._config.retention_days)
        for user_dir in base.iterdir():
            if not user_dir.is_dir():
                continue
            _cleanup_user_dir(user_dir, cutoff, self._config.max_runs_per_user)


def _cleanup_user_dir(user_dir: Path, cutoff: datetime, max_runs: int) -> None:
    run_dirs = sorted(
        [d for d in user_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs[max_runs:]:
        logger.debug("artifact cleanup: removing excess run %s", run_dir)
        shutil.rmtree(run_dir, ignore_errors=True)
    for run_dir in run_dirs[:max_runs]:
        mtime = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=UTC)
        if mtime < cutoff:
            logger.debug("artifact cleanup: removing expired run %s", run_dir)
            shutil.rmtree(run_dir, ignore_errors=True)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
