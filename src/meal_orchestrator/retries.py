from __future__ import annotations

import logging
import time
import urllib.error
from collections.abc import Callable

logger = logging.getLogger(__name__)


def is_transient_http_error(exc: Exception) -> bool:
    """Return True for HTTP errors that are safe to retry."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (429, 500, 502, 503, 504)
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return True
    return False


class RetryError(RuntimeError):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Exception) -> None:
        super().__init__(message)
        self.last_exception = last_exception


def with_retries[T](
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    retryable: Callable[[Exception], bool],
    operation_name: str = "operation",
) -> T:  # noqa: UP047 — `from __future__ import annotations` prevents PEP 695 syntax
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
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    delay = base_delay_seconds

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if not retryable(exc):
                raise
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
                    f"{operation_name} failed after {max_attempts} attempt(s): {exc}",
                    last_exception=exc,
                ) from exc

    raise AssertionError("unreachable")
