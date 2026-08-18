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

# Basenames of the two run-level files written directly under a run
# directory (see ArtifactStore.save_run_metadata / save_batch_result). A
# user_id is an unvalidated free-form string (config/loader.py) that becomes
# a directory name nested inside that same run directory — a user literally
# configured with one of these ids would otherwise collide with a run-level
# file of the identical name.
_RESERVED_RUN_FILENAMES = {"metadata.json", "batch_result.json"}


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

    def save_llm_attempt(self, attempt: int, data: dict[str, Any]) -> None:
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
                    "app_prompt": request.payload.app_prompt,
                    "user_prompt": request.payload.user_prompt,
                    "menu": request.payload.menu.to_compact_dict(),
                },
            ),
        )

    def save_llm_response(self, result: LlmResult) -> None:
        self._write_safe(
            "llm_response.json",
            lambda: _write_json(
                self._run_dir / "llm_response.json", result.structured.model_dump(mode="json")
            ),
        )

    def save_llm_attempt(self, attempt: int, data: dict[str, Any]) -> None:
        name = f"llm_attempts/attempt_{attempt:02d}.json"
        self._write_safe(
            name,
            lambda: _write_json(self._run_dir / name, data),
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
    """Manages artifact persistence and cleanup for workflow runs.

    Layout is run-first: everything belonging to one run — the run-level
    metadata, the raw batch response when batch mode is used, and every
    participating user's own per-user detail — lives under one
    `<path>/<run_id>/` directory, so a run's full footprint is a single
    directory to inspect or prune.
    """

    def __init__(self, config: ArtifactConfig | None = None) -> None:
        self._config = config

    def for_run(self, run_id: str, user_id: str) -> RunArtifacts:
        if not self._config:
            return RunArtifacts()
        if user_id in _RESERVED_RUN_FILENAMES:
            logger.warning(
                "artifact store: user id %r collides with a reserved run-level filename "
                "— artifacts disabled for this user",
                user_id,
                extra={"run_id": run_id, "user_id": user_id, "step": "artifacts"},
            )
            return RunArtifacts()
        run_dir = self._config.path / run_id / user_id
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

    def save_run_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        """Persist run-level metadata (mode, batch id/status, aggregate batch
        usage when applicable, week, participating users) once per run —
        facts that belong to the run as a whole, not to any single user, so
        they don't belong inside a per-user metadata.json.
        """
        if not self._config:
            return
        path = self._config.path / run_id / "metadata.json"
        try:
            _write_json(path, metadata)
        except Exception:
            logger.warning(
                "artifact store: failed to save run metadata %s",
                path,
                exc_info=True,
                extra={"run_id": run_id, "step": "artifacts"},
            )

    def save_batch_result(self, run_id: str, data: dict[str, Any]) -> bool:
        """Persist the raw OpenRouter batch response (status, request_counts,
        aggregate usage/cost, per-row results) once per run — this is the
        only place that data is durably captured; logging it would mean
        dumping an arbitrarily large payload (every row's full LLM output)
        to stdout on every run.

        Returns whether the save actually succeeded, so a caller about to
        tell an operator "see the saved artifact for detail" doesn't say so
        when there's nothing there to see.
        """
        if not self._config:
            return False
        path = self._config.path / run_id / "batch_result.json"
        try:
            _write_json(path, data)
            return True
        except Exception:
            # Broad on purpose, matching _FilesystemRunArtifacts._write_safe: a
            # write can fail for reasons beyond OSError (e.g. an LLM-generated
            # string with a lone surrogate codepoint failing UTF-8 encoding),
            # and an artifact write failing must never crash the batch delivery
            # it's incidental to.
            logger.warning(
                "artifact store: failed to save batch result %s",
                path,
                exc_info=True,
                extra={"run_id": run_id, "step": "artifacts"},
            )
            return False

    def cleanup(self) -> None:
        if not self._config:
            return
        base = self._config.path
        if not base.exists():
            return
        cutoff = datetime.now(UTC) - timedelta(days=self._config.retention_days)
        run_dirs = sorted(
            (d for d in base.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for run_dir in run_dirs[self._config.max_runs :]:
            logger.debug("artifact cleanup: removing excess run %s", run_dir)
            shutil.rmtree(run_dir, ignore_errors=True)
        for run_dir in run_dirs[: self._config.max_runs]:
            mtime = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=UTC)
            if mtime < cutoff:
                logger.debug("artifact cleanup: removing expired run %s", run_dir)
                shutil.rmtree(run_dir, ignore_errors=True)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
