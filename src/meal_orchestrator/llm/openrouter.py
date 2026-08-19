from __future__ import annotations

import json
import logging
import os
import urllib.error
from collections.abc import Callable
from typing import Any

from meal_orchestrator.domain import LlmRequest, LlmResult
from meal_orchestrator.http import post_json
from meal_orchestrator.llm.openrouter_shared import (
    EmptyLlmResponseError,
    IncompleteAssessmentError,
    LlmFailureDetails,
    OpenRouterHttpError,
    OpenRouterResponseError,
    StructuredOutputError,
    build_request_body,
    build_request_headers,
    build_result,
    error_type_from_error,
    response_structured,
)
from meal_orchestrator.retries import RetryError, is_transient_http_error, with_retries

__all__ = [
    "EmptyLlmResponseError",
    "IncompleteAssessmentError",
    "LlmFailureDetails",
    "OpenRouterClient",
    "OpenRouterHttpError",
    "OpenRouterResponseError",
    "StructuredOutputError",
]

logger = logging.getLogger(__name__)

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_BASE_DELAY = 1.0
_BACKOFF_FACTOR = 2.0

# These clear on their own but rarely within a couple of seconds, so they get a much
# longer backoff than a generic transient error (e.g. a dropped connection).
_SLOW_CLEARING_ERROR_TYPES = frozenset({"rate_limit_exceeded", "provider_overloaded"})
_SLOW_CLEARING_BASE_DELAY = 15.0


class OpenRouterClient:
    def __init__(self, *, api_key: str | None = None, max_retries: int = 3) -> None:
        self._api_key = api_key if api_key is not None else os.environ["OPENROUTER_API_KEY"]
        self._max_retries = max_retries

    @property
    def api_key(self) -> str:
        """The key this client sends requests with.

        Exposed so callers that talk to other OpenRouter endpoints outside
        this class (e.g. the batch client) can reuse the same configured
        credential instead of independently re-reading OPENROUTER_API_KEY.
        """
        return self._api_key

    def generate(
        self,
        request: LlmRequest,
        *,
        on_attempt: Callable[[int, str | None, Any, dict[str, Any]], None] | None = None,
    ) -> LlmResult:
        headers = build_request_headers(self._api_key)
        attempt = 0
        feedback: str | None = None

        def _call(model: str) -> tuple[dict[str, Any], Any]:
            nonlocal attempt
            attempt += 1
            body = json.dumps(build_request_body(model, request.payload, feedback)).encode(
                "utf-8"
            )
            try:
                raw = post_json(
                    _API_URL, headers=headers, body=body, timeout_seconds=request.timeout_seconds
                )
            except urllib.error.HTTPError as exc:
                error = _openrouter_http_error(exc, attempt)
                if on_attempt is not None:
                    on_attempt(attempt, feedback, error.details.response, _rejected_outcome(error))
                raise error from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if on_attempt is not None:
                    on_attempt(attempt, feedback, None, _network_error_outcome(exc, model))
                raise
            response = json.loads(raw.decode("utf-8"))
            try:
                assessment = response_structured(response, request.payload.menu, attempt)
            except OpenRouterResponseError as exc:
                if on_attempt is not None:
                    on_attempt(attempt, feedback, response, _rejected_outcome(exc))
                raise
            if on_attempt is not None:
                on_attempt(attempt, feedback, response, {"accepted": True})
            return response, assessment

        def _on_retry(exc: Exception) -> None:
            nonlocal feedback
            if isinstance(exc, OpenRouterResponseError):
                new_feedback = _feedback_for(exc)
                if new_feedback is not None:
                    feedback = new_feedback

        # OpenRouter's own `models` array is documented to fail over mid-request on
        # error, but in practice it does not trigger for the embedded-error-in-a-200
        # -response shape (e.g. rate_limit_exceeded) — observed keeping the same
        # rate-limited model selected across every retry of an unchanged request. So
        # fallback is driven here instead: each candidate model gets its own full
        # max_retries budget (with the existing backoff/slow-clearing delays), and only
        # once that budget is exhausted do we move to the next configured model.
        candidates = [request.model, *request.fallback_models]
        last_retry_error: RetryError | None = None
        for index, model in enumerate(candidates):
            try:
                response, assessment = with_retries(
                    lambda model=model: _call(model),
                    max_attempts=self._max_retries,
                    base_delay_seconds=_BASE_DELAY,
                    backoff_factor=_BACKOFF_FACTOR,
                    retryable=lambda exc: is_transient_http_error(exc)
                    or (isinstance(exc, OpenRouterResponseError) and exc.retryable),
                    operation_name=f"openrouter generate model={model}",
                    on_retry=_on_retry,
                    delay_seconds=_retry_delay_seconds,
                )
            except RetryError as exc:
                last_retry_error = exc
                # `with_retries` doesn't call `on_retry` for the final attempt of a
                # model's own budget (it's about to raise, not retry) — but that
                # final attempt's problems are exactly what the next candidate model
                # needs to hear, so recompute feedback from it here.
                if isinstance(exc.last_exception, OpenRouterResponseError):
                    new_feedback = _feedback_for(exc.last_exception, cross_model=True)
                    if new_feedback is not None:
                        feedback = new_feedback
                if index < len(candidates) - 1:
                    logger.warning(
                        "openrouter generate model=%s exhausted %d attempt(s), "
                        "falling back to model=%s",
                        model,
                        self._max_retries,
                        candidates[index + 1],
                    )
                continue
            return build_result(response, assessment, attempt, model)
        assert last_retry_error is not None  # candidates is never empty
        if len(candidates) > 1:
            # Naming only the last candidate here would hide that other models were
            # tried first and failed for unrelated reasons (e.g. the primary was
            # rate-limited while the fallback separately produced a bad completion) —
            # callers (ops notifications, logs) only see this final message.
            tried = ", ".join(candidates)
            raise RetryError(
                f"openrouter generate exhausted all {len(candidates)} candidate model(s) "
                f"({tried}): {last_retry_error.last_exception}",
                last_exception=last_retry_error.last_exception,
            ) from last_retry_error
        raise last_retry_error


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    if _is_slow_clearing(exc):
        return _SLOW_CLEARING_BASE_DELAY * (_BACKOFF_FACTOR ** (attempt - 1))
    return _BASE_DELAY * (_BACKOFF_FACTOR ** (attempt - 1))


def _is_slow_clearing(exc: Exception) -> bool:
    details = getattr(exc, "details", None)
    if isinstance(details, LlmFailureDetails) and details.error_type in _SLOW_CLEARING_ERROR_TYPES:
        return True
    # A genuine HTTP 429 is the canonical shape for rate limiting — treat it as
    # slow-clearing even when the response body carries no error_type metadata,
    # rather than relying solely on the embedded-error-in-a-200 shape above.
    return isinstance(exc, urllib.error.HTTPError) and exc.code == 429


def _rejected_outcome(exc: OpenRouterResponseError | OpenRouterHttpError) -> dict[str, Any]:
    # "attempt" is dropped here — the caller already knows the attempt number and is the
    # single source of truth for it, so we don't let this metadata silently override it.
    metadata = {key: value for key, value in exc.details.to_metadata().items() if key != "attempt"}
    outcome: dict[str, Any] = {"accepted": False, **metadata}
    if isinstance(exc, StructuredOutputError):
        outcome["parse_error"] = exc.parse_error
    if isinstance(exc, IncompleteAssessmentError):
        outcome["problems"] = exc.problems
    return outcome


def _network_error_outcome(exc: Exception, model: str) -> dict[str, Any]:
    return {"accepted": False, "reason": "network_error", "error": str(exc), "model": model}


def _feedback_for(exc: OpenRouterResponseError, *, cross_model: bool = False) -> str | None:
    # `cross_model` distinguishes "you, just now" from "a prior attempt by a
    # different model" — this feedback is only ever true from the first-person
    # perspective when it's addressed back to the same model that produced it.
    subject = "A previous attempt by a different model" if cross_model else "Your previous response"
    if isinstance(exc, StructuredOutputError):
        return (
            f"{subject} did not match the required JSON schema "
            f"({exc.parse_error}). Return a response that strictly matches the schema."
        )
    if isinstance(exc, IncompleteAssessmentError):
        joined = "; ".join(exc.problems)
        return (
            f"{subject} was incomplete: {joined}. Assess every meal and "
            "every variant listed in the menu JSON — do not skip any of them."
        )
    return None


def _openrouter_http_error(
    error: urllib.error.HTTPError, attempt: int
) -> OpenRouterHttpError:
    response_body = getattr(error, "response_body", "")
    try:
        response = json.loads(response_body)
    except json.JSONDecodeError:
        response = {"error": {"code": error.code, "message": response_body or error.reason}}
    error_type = (
        error_type_from_error(response.get("error")) if isinstance(response, dict) else None
    )
    return OpenRouterHttpError(
        error,
        LlmFailureDetails(
            reason="http_error",
            attempt=attempt,
            response=response,
            http_status=error.code,
            error_type=error_type,
        ),
    )
