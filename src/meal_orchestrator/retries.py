from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryError(RuntimeError):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Exception) -> None:
        super().__init__(message)
        self.last_exception = last_exception


def with_retries(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    retryable: Callable[[Exception], bool],
    operation_name: str = "operation",
) -> T:
    """Execute fn with exponential backoff retries.

    Args:
        fn: Zero-argument callable to attempt.
        max_attempts: Total number of attempts (1 = no retries).
        base_delay_seconds: Delay before the second attempt.
        backoff_factor: Multiplier applied to delay on each subsequent attempt.
        retryable: Predicate that returns True for exceptions that should be retried.
        operation_name: Used in log messages.

    Returns:
        The return value of fn on success.

    Raises:
        RetryError: When all attempts are exhausted.
        Exception: Any non-retryable exception from fn is re-raised immediately.
    """
    last_exc: Exception | None = None
    delay = base_delay_seconds

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if not retryable(exc):
                raise
            last_exc = exc
            if attempt < max_attempts:
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                    operation_name,
                    attempt,
                    max_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(
                    "%s failed (attempt %d/%d): %s — no more retries",
                    operation_name,
                    attempt,
                    max_attempts,
                    exc,
                )

    raise RetryError(
        f"{operation_name} failed after {max_attempts} attempt(s)",
        last_exception=last_exc,  # type: ignore[arg-type]
    )
