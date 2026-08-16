from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from meal_orchestrator.config.models import BatchConfig

logger = logging.getLogger(__name__)

_STATE_DIR_NAME = ".batch_state"
_STATE_FILE_NAME = "pending_batch.json"
_LOCK_FILE_NAME = "run.lock"


@dataclass(frozen=True)
class PendingBatchUser:
    user_id: str
    custom_id: str


@dataclass(frozen=True)
class PendingBatchState:
    """Durable record of an in-flight batch, written before the wait loop starts.

    Lets a crashed/restarted process resume polling the same batch instead of
    resubmitting duplicate (billable) work.
    """

    run_id: str
    batch_id: str
    submitted_at: str
    week_start: str
    week_end: str
    model: str
    users: list[PendingBatchUser]


def _state_dir(project_root: Path) -> Path:
    return project_root / _STATE_DIR_NAME


def state_file_path(project_root: Path) -> Path:
    return _state_dir(project_root) / _STATE_FILE_NAME


def save_state(project_root: Path, state: PendingBatchState) -> None:
    """Persist `state`, best-effort.

    A failure here (e.g. a read-only project root) must not crash the run —
    it only means a crash during the wait loop won't be resumable, which is
    strictly worse than not persisting but still far better than the whole
    run failing outright.
    """
    path = state_file_path(project_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    except OSError:
        logger.warning(
            "pending batch state could not be persisted at %s — a crash during this "
            "batch's wait would not be resumable",
            path,
            exc_info=True,
        )


def load_state(project_root: Path) -> PendingBatchState | None:
    path = state_file_path(project_root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return PendingBatchState(
            run_id=data["run_id"],
            batch_id=data["batch_id"],
            submitted_at=data["submitted_at"],
            week_start=data["week_start"],
            week_end=data["week_end"],
            model=data["model"],
            users=[PendingBatchUser(**user) for user in data["users"]],
        )
    except Exception:
        logger.warning(
            "pending batch state file unreadable, ignoring: %s", path, exc_info=True
        )
        return None


def clear_state(project_root: Path) -> None:
    state_file_path(project_root).unlink(missing_ok=True)


def acquire_lock(project_root: Path) -> bool:
    """Best-effort guard against two invocations both running a blocking batch
    wait at once. Returns False if another live process already holds the
    lock, or if the lock can't be acquired at all (e.g. a read-only project
    root) — either way, callers treat False as "can't do the durable/
    coordinated thing right now" and fall back accordingly.

    Uses an atomic create-then-link for the actual acquisition (see
    `_try_create_lock_file`) so two processes racing to acquire at the same
    instant can't both succeed — a plain exists()-then-write() check has a
    window where both would see "no lock" and both proceed, defeating the
    whole point of the lock.
    """
    lock_path = _state_dir(project_root) / _LOCK_FILE_NAME
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if _try_create_lock_file(lock_path):
            return True
        if _lock_holder_is_alive(lock_path):
            return False
        # Stale lock from a dead process — clear it and make one retry attempt.
        # If we lose a race here too, be conservative and report unavailable
        # rather than looping; the caller falls back gracefully either way.
        lock_path.unlink(missing_ok=True)
        return _try_create_lock_file(lock_path)
    except OSError:
        logger.warning(
            "batch run lock could not be acquired at %s due to a filesystem error — "
            "treating as unavailable",
            lock_path,
            exc_info=True,
        )
        return False


def _try_create_lock_file(lock_path: Path) -> bool:
    """Atomically create `lock_path` already containing our PID.

    Writing content via `os.open(O_CREAT|O_EXCL)` then a separate write()
    leaves a window where the file exists but is still empty — a concurrent
    caller can observe that empty file, fail to parse a PID from it, and
    wrongly conclude the lock is stale/abandoned and safe to steal. Instead,
    write the full content to a uniquely-named temp file first, then
    `os.link` it into place: `link()` atomically fails with FileExistsError
    if the target already exists, and by construction there's no moment
    where the target exists without its final content.
    """
    tmp_path = lock_path.with_name(f"{lock_path.name}.{uuid4().hex}.tmp")
    tmp_path.write_text(str(os.getpid()), encoding="utf-8")
    try:
        os.link(tmp_path, lock_path)
        return True
    except FileExistsError:
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


def _lock_holder_is_alive(lock_path: Path) -> bool:
    try:
        pid = int(lock_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        return False
    return True


def release_lock(project_root: Path) -> None:
    (_state_dir(project_root) / _LOCK_FILE_NAME).unlink(missing_ok=True)


def poll_until_terminal(
    batch_id: str,
    config: BatchConfig,
    *,
    get_batch: Any,
    is_pending: Any,
    sleep: Any = time.sleep,
    started_at: datetime | None = None,
) -> dict[str, Any] | None:
    """Poll `get_batch(batch_id)` with exponential backoff until it leaves the
    pending states, or `max_wait_hours` elapses (returns None on timeout).

    Backoff starts at `initial_poll_interval_seconds`, doubles each check, and
    is capped at `max_poll_interval_seconds` — matched to typical batch
    turnaround (seconds to minutes) without hammering the API if a batch runs
    long.

    A single status check can fail (network blip, transient HTTP error,
    malformed response body) without ending the wait — over a loop that can
    span many hours, one bad check must not crash the whole run. It's logged
    and treated as an inconclusive/pending check instead; persistent failures
    are still bounded by `max_wait_hours`, which lets the caller's existing
    timeout-fallback path take over.
    """
    deadline = (started_at or datetime.now(UTC)).timestamp() + config.max_wait_hours * 3600
    # Clamped defensively in case initial > max (loader.py validates this too, but
    # a caller could construct BatchConfig directly without going through it).
    interval = min(config.initial_poll_interval_seconds, config.max_poll_interval_seconds)
    while True:
        try:
            data = get_batch(batch_id)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning(
                "batch status check failed for batch_id=%s (will retry): %s",
                batch_id,
                exc,
                exc_info=True,
            )
        else:
            if not is_pending(data):
                return data
        if time.time() >= deadline:
            return None
        sleep(min(interval, max(0.0, deadline - time.time())))
        interval = min(interval * 2, config.max_poll_interval_seconds)
