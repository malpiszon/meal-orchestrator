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
    path.write_text("not json", encoding="utf-8")

    assert load_state(tmp_path) is None


def test_save_state_does_not_raise_when_state_dir_unwritable(tmp_path) -> None:
    # A plain file where the state directory would go makes mkdir(parents=True) fail.
    state_dir = tmp_path / "state"
    state_dir.write_text("not a directory", encoding="utf-8")

    save_state(state_dir, _state())  # must not raise

    assert load_state(state_dir) is None


def test_acquire_lock_returns_false_when_state_dir_unwritable(tmp_path) -> None:
    state_dir = tmp_path / "state"
    state_dir.write_text("not a directory", encoding="utf-8")

    assert acquire_lock(state_dir) is False  # must not raise


def test_acquire_lock_blocks_second_caller_then_releases(tmp_path) -> None:
    assert acquire_lock(tmp_path) is True
    assert acquire_lock(tmp_path) is False

    release_lock(tmp_path)

    assert acquire_lock(tmp_path) is True
    release_lock(tmp_path)


def test_acquire_lock_exactly_one_winner_under_concurrent_contention(tmp_path) -> None:
    """Regression test for a check-then-write TOCTOU race: many threads racing
    to acquire the lock at once must yield exactly one True, not several —
    the whole point of the lock is to prevent double batch submission.
    """
    import threading

    outcomes: list[bool] = []
    outcomes_lock = threading.Lock()
    start_barrier = threading.Barrier(20)

    def attempt() -> None:
        start_barrier.wait()  # line everyone up to maximize actual overlap
        got = acquire_lock(tmp_path)
        with outcomes_lock:
            outcomes.append(got)

    threads = [threading.Thread(target=attempt) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 19


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


def test_poll_until_terminal_waits_before_first_check_when_requested() -> None:
    """Right after a fresh submission, OpenRouter's batch API isn't reliably
    queryable yet (observed: an immediate GET 404s on a batch that was just
    successfully created) — a positive initial_check_delay_seconds skips
    that guaranteed-to-fail immediate check by waiting first.
    """
    config = BatchConfig(initial_poll_interval_seconds=1, max_poll_interval_seconds=10)
    sleeps: list[float] = []
    calls: list[str] = []

    def get_batch(batch_id):
        calls.append(batch_id)
        return {"status": "completed"}

    result = poll_until_terminal(
        "batch-1",
        config,
        get_batch=get_batch,
        is_pending=lambda data: data["status"] != "completed",
        sleep=sleeps.append,
        initial_check_delay_seconds=15,
    )

    assert result == {"status": "completed"}
    assert sleeps == [15]  # waited once before the (only, successful) check
    assert calls == ["batch-1"]


def test_poll_until_terminal_skips_wait_when_initial_check_delay_is_zero() -> None:
    """0 (the default, used when resuming a possibly-already-finished batch)
    must behave exactly like omitting the argument — check immediately.
    """
    config = BatchConfig(initial_poll_interval_seconds=1, max_poll_interval_seconds=10)
    sleeps: list[float] = []

    result = poll_until_terminal(
        "batch-1",
        config,
        get_batch=lambda batch_id: {"status": "completed"},
        is_pending=lambda data: data["status"] != "completed",
        sleep=sleeps.append,
        initial_check_delay_seconds=0,
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


def test_poll_until_terminal_survives_a_transient_error_and_keeps_polling() -> None:
    """A single network blip / transient HTTP error fetching batch status must
    not crash the whole (potentially many-hours-long) wait — it should be
    logged and treated as inconclusive, with polling continuing normally.
    """
    import urllib.error

    config = BatchConfig(initial_poll_interval_seconds=1, max_poll_interval_seconds=10)
    sleeps: list[float] = []
    calls = {"n": 0}

    def flaky_get_batch(batch_id):
        calls["n"] += 1
        if calls["n"] == 2:
            raise urllib.error.URLError("connection reset")
        if calls["n"] < 4:
            return {"status": "in_progress"}
        return {"status": "completed"}

    result = poll_until_terminal(
        "batch-1",
        config,
        get_batch=flaky_get_batch,
        is_pending=lambda data: data["status"] != "completed",
        sleep=sleeps.append,
    )

    assert result == {"status": "completed"}
    assert calls["n"] == 4


def test_poll_until_terminal_eventually_times_out_if_every_check_errors() -> None:
    import datetime as dt
    import urllib.error

    config = BatchConfig(
        initial_poll_interval_seconds=1, max_poll_interval_seconds=10, max_wait_hours=1
    )
    started_at = dt.datetime.now(dt.UTC) - dt.timedelta(hours=2)

    def always_fails(batch_id):
        raise urllib.error.URLError("connection reset")

    result = poll_until_terminal(
        "batch-1",
        config,
        get_batch=always_fails,
        is_pending=lambda data: True,
        sleep=lambda seconds: None,
        started_at=started_at,
    )

    assert result is None


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


def test_poll_until_terminal_clamps_initial_check_delay_to_deadline() -> None:
    """A large initial_check_delay_seconds must not overshoot an already-
    (near-)expired deadline — clamped to 0 here, then the check still
    happens (matching the no-delay case's "always check at least once").
    """
    import datetime as dt

    config = BatchConfig(
        initial_poll_interval_seconds=1, max_poll_interval_seconds=10, max_wait_hours=1
    )
    started_at = dt.datetime.now(dt.UTC) - dt.timedelta(hours=2)
    sleeps: list[float] = []
    calls: list[str] = []

    def get_batch(batch_id):
        calls.append(batch_id)
        return {"status": "in_progress"}

    result = poll_until_terminal(
        "batch-1",
        config,
        get_batch=get_batch,
        is_pending=lambda data: True,
        sleep=sleeps.append,
        started_at=started_at,
        initial_check_delay_seconds=120,
    )

    assert result is None
    assert sleeps == [0.0]
    assert calls == ["batch-1"]


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
