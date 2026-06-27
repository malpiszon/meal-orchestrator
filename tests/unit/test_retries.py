from __future__ import annotations

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
