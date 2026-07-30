from __future__ import annotations

from unittest.mock import patch

import pytest

from meal_orchestrator.retries import RetryError, with_retries


def _always_retryable(exc: Exception) -> bool:
    return True


def _never_retryable(exc: Exception) -> bool:
    return False


def test_succeeds_on_first_attempt() -> None:
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    result = with_retries(fn, retryable=_always_retryable, operation_name="test")
    assert result == "ok"
    assert len(calls) == 1


def test_retries_and_succeeds() -> None:
    attempts = []

    def fn():
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("transient")
        return "ok"

    result = with_retries(
        fn,
        max_attempts=3,
        base_delay_seconds=0,
        retryable=_always_retryable,
        operation_name="test",
    )
    assert result == "ok"
    assert len(attempts) == 3


def test_raises_retry_error_after_exhaustion() -> None:
    def fn():
        raise OSError("always fails")

    with pytest.raises(RetryError) as exc_info:
        with_retries(
            fn,
            max_attempts=3,
            base_delay_seconds=0,
            retryable=_always_retryable,
            operation_name="test",
        )

    assert "3 attempt(s)" in str(exc_info.value)
    assert isinstance(exc_info.value.last_exception, OSError)


def test_non_retryable_exception_raised_immediately() -> None:
    attempts = []

    def fn():
        attempts.append(1)
        raise ValueError("not retryable")

    with pytest.raises(ValueError, match="not retryable"):
        with_retries(
            fn,
            max_attempts=5,
            base_delay_seconds=0,
            retryable=_never_retryable,
            operation_name="test",
        )

    assert len(attempts) == 1


def test_retryable_predicate_called_with_exception() -> None:
    seen: list[Exception] = []

    def retryable(exc: Exception) -> bool:
        seen.append(exc)
        return True

    exc = OSError("boom")

    def fn():
        raise exc

    with pytest.raises(RetryError):
        with_retries(fn, max_attempts=2, base_delay_seconds=0, retryable=retryable)

    assert seen == [exc, exc]


def test_on_retry_called_once_before_each_retry_with_the_exception() -> None:
    seen: list[Exception] = []
    attempts = []

    def fn():
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError(f"transient {len(attempts)}")
        return "ok"

    result = with_retries(
        fn,
        max_attempts=3,
        base_delay_seconds=0,
        retryable=_always_retryable,
        on_retry=seen.append,
    )

    assert result == "ok"
    assert [str(exc) for exc in seen] == ["transient 1", "transient 2"]


def test_on_retry_not_called_on_first_attempt_success() -> None:
    calls = []

    def fn():
        return "ok"

    with_retries(fn, retryable=_always_retryable, on_retry=calls.append)

    assert calls == []


def test_on_retry_not_called_on_final_exhausted_attempt() -> None:
    calls = []

    def fn():
        raise OSError("always fails")

    with pytest.raises(RetryError):
        with_retries(
            fn,
            max_attempts=2,
            base_delay_seconds=0,
            retryable=_always_retryable,
            on_retry=calls.append,
        )

    # Two attempts total, but only one retry happens (before attempt 2) — the
    # final failing attempt exhausts retries and must not invoke on_retry.
    assert len(calls) == 1


def test_delay_seconds_override_controls_sleep_duration() -> None:
    sleeps: list[float] = []

    def fn():
        raise OSError("always fails")

    with patch("time.sleep", side_effect=sleeps.append):
        with pytest.raises(RetryError):
            with_retries(
                fn,
                max_attempts=3,
                base_delay_seconds=1,
                retryable=_always_retryable,
                delay_seconds=lambda exc, attempt: attempt * 10.0,
            )

    assert sleeps == [10.0, 20.0]


def test_on_retry_not_called_for_non_retryable_exception() -> None:
    calls = []

    def fn():
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        with_retries(
            fn,
            max_attempts=3,
            base_delay_seconds=0,
            retryable=_never_retryable,
            on_retry=calls.append,
        )

    assert calls == []
