from __future__ import annotations

from meal_orchestrator.batch_runner import (
    PendingBatchState,
    PendingBatchUser,
    acquire_lock,
    clear_state,
    load_state,
    poll_until_terminal,
    release_lock,
    save_state,
)
from meal_orchestrator.config.models import BatchConfig


def _state() -> PendingBatchState:
    return PendingBatchState(
        run_id="run-1",
        batch_id="batch-1",
        submitted_at="2026-06-01T00:00:00+00:00",
        week_start="2026-06-01",
        week_end="2026-06-05",
        model="openai/gpt-4o-mini",
        users=[PendingBatchUser(user_id="alan", custom_id="run-1:alan")],
    )


def test_save_load_clear_state_roundtrip(tmp_path) -> None:
    assert load_state(tmp_path) is None

    save_state(tmp_path, _state())
    loaded = load_state(tmp_path)

    assert loaded == _state()

    clear_state(tmp_path)
    assert load_state(tmp_path) is None


def test_load_state_ignores_corrupt_file(tmp_path) -> None:
    from meal_orchestrator.batch_runner import state_file_path

    path = state_file_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")

    assert load_state(tmp_path) is None


def test_save_state_does_not_raise_when_state_dir_unwritable(tmp_path) -> None:
    # A plain file where the state directory would go makes mkdir(parents=True) fail.
    (tmp_path / ".batch_state").write_text("not a directory", encoding="utf-8")

    save_state(tmp_path, _state())  # must not raise

    assert load_state(tmp_path) is None


def test_acquire_lock_returns_false_when_state_dir_unwritable(tmp_path) -> None:
    (tmp_path / ".batch_state").write_text("not a directory", encoding="utf-8")

    assert acquire_lock(tmp_path) is False  # must not raise


def test_acquire_lock_blocks_second_caller_then_releases(tmp_path) -> None:
    assert acquire_lock(tmp_path) is True
    assert acquire_lock(tmp_path) is False

    release_lock(tmp_path)

    assert acquire_lock(tmp_path) is True
    release_lock(tmp_path)


def test_poll_until_terminal_returns_immediately_when_already_done() -> None:
    config = BatchConfig(initial_poll_interval_seconds=1, max_poll_interval_seconds=10)
    sleeps: list[float] = []

    result = poll_until_terminal(
        "batch-1",
        config,
        get_batch=lambda batch_id: {"status": "completed"},
        is_pending=lambda data: data["status"] != "completed",
        sleep=sleeps.append,
    )

    assert result == {"status": "completed"}
    assert sleeps == []


def test_poll_until_terminal_backs_off_then_completes() -> None:
    config = BatchConfig(initial_poll_interval_seconds=1, max_poll_interval_seconds=10)
    statuses = iter(["in_progress", "in_progress", "completed"])
    sleeps: list[float] = []

    result = poll_until_terminal(
        "batch-1",
        config,
        get_batch=lambda batch_id: {"status": next(statuses)},
        is_pending=lambda data: data["status"] != "completed",
        sleep=sleeps.append,
    )

    assert result == {"status": "completed"}
    assert sleeps == [1, 2]


def test_poll_until_terminal_clamps_initial_interval_to_max() -> None:
    """Defense in depth: even if a BatchConfig is constructed directly with
    initial_poll_interval_seconds > max_poll_interval_seconds (bypassing the
    loader's cross-validation), the first sleep must still respect the max.
    """
    config = BatchConfig(initial_poll_interval_seconds=100, max_poll_interval_seconds=10)
    statuses = iter(["in_progress", "completed"])
    sleeps: list[float] = []

    result = poll_until_terminal(
        "batch-1",
        config,
        get_batch=lambda batch_id: {"status": next(statuses)},
        is_pending=lambda data: data["status"] != "completed",
        sleep=sleeps.append,
    )

    assert result == {"status": "completed"}
    assert sleeps == [10]


def test_poll_until_terminal_returns_none_on_timeout() -> None:
    import datetime as dt

    config = BatchConfig(
        initial_poll_interval_seconds=1, max_poll_interval_seconds=10, max_wait_hours=1
    )
    started_at = dt.datetime.now(dt.UTC) - dt.timedelta(hours=2)

    result = poll_until_terminal(
        "batch-1",
        config,
        get_batch=lambda batch_id: {"status": "in_progress"},
        is_pending=lambda data: True,
        sleep=lambda seconds: None,
        started_at=started_at,
    )

    assert result is None
