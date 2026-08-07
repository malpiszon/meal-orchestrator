from __future__ import annotations

import json
import logging
import os
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from meal_orchestrator import APP_NAME
from meal_orchestrator.domain import CanonicalMenu, LlmRequest, LlmResult, PromptPayload
from meal_orchestrator.domain.llm_output import (
    WeekAssessment,
    validate_completeness,
    week_assessment_json_schema,
)
from meal_orchestrator.http import post_json
from meal_orchestrator.retries import RetryError, is_transient_http_error, with_retries

logger = logging.getLogger(__name__)

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_BASE_DELAY = 1.0
_BACKOFF_FACTOR = 2.0

# OpenRouter's documented error_type taxonomy (openrouter.ai/docs/api-reference/errors)
# for errors embedded in choices[].error when finish_reason == "error". Types outside
# this set (including none at all, e.g. a bare provider timeout) are treated as
# transient and retried, matching prior behavior.
_NON_RETRYABLE_FINISH_ERROR_TYPES = frozenset(
    {
        "authentication",
        "payment_required",
        "permission_denied",
        "context_length_exceeded",
        "content_policy_violation",
    }
)

# These clear on their own but rarely within a couple of seconds, so they get a much
# longer backoff than a generic transient error (e.g. a dropped connection).
_SLOW_CLEARING_ERROR_TYPES = frozenset({"rate_limit_exceeded", "provider_overloaded"})
_SLOW_CLEARING_BASE_DELAY = 15.0

@dataclass(frozen=True)
class LlmFailureDetails:
    """Diagnostic data returned with an unusable OpenRouter completion."""

    reason: str
    attempt: int
    response: Any
    http_status: int | None = None
    error_type: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "http_status": self.http_status,
            "error_type": self.error_type,
            **_response_metadata(self.response, self.attempt),
        }


class OpenRouterResponseError(RuntimeError):
    """Raised when OpenRouter returns an unusable completion response."""

    def __init__(self, details: LlmFailureDetails, *, retryable: bool) -> None:
        super().__init__(_failure_message(details))
        self.details = details
        self.retryable = retryable


class EmptyLlmResponseError(OpenRouterResponseError):
    """Raised when OpenRouter returns a completion without usable text."""

    def __init__(self, details: LlmFailureDetails) -> None:
        super().__init__(details, retryable=True)


class StructuredOutputError(OpenRouterResponseError):
    """Raised when the completion's content isn't valid JSON matching WeekAssessment."""

    def __init__(self, details: LlmFailureDetails, *, parse_error: str) -> None:
        super().__init__(details, retryable=True)
        self.parse_error = parse_error


class IncompleteAssessmentError(OpenRouterResponseError):
    """Raised when a schema-valid assessment is missing meals/variants from the menu."""

    def __init__(self, details: LlmFailureDetails, *, problems: list[str]) -> None:
        super().__init__(details, retryable=True)
        self.problems = problems


class OpenRouterHttpError(urllib.error.HTTPError):
    """HTTP error carrying the parsed OpenRouter failure details."""

    def __init__(self, error: urllib.error.HTTPError, details: LlmFailureDetails) -> None:
        super().__init__(error.url, error.code, error.reason, error.headers, None)
        self.details = details


def _build_message_content(
    payload: PromptPayload, feedback: str | None
) -> list[dict[str, str]]:
    # Separate blocks keep instructions and data structurally distinct for large JSON payloads.
    menu_json = json.dumps(
        payload.menu.to_compact_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    blocks = [
        {"type": "text", "text": f"App instructions:\n{payload.app_prompt}"},
        {"type": "text", "text": f"User instructions:\n{payload.user_prompt}"},
        {"type": "text", "text": f"Canonical menu JSON:\n{menu_json}"},
    ]
    if feedback:
        blocks.append({"type": "text", "text": feedback})
    return blocks


# The schema shape is static for the process lifetime, so it's built once here
# rather than on every request/retry attempt.
_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "week_assessment",
        "strict": True,
        "schema": week_assessment_json_schema(),
    },
}


class OpenRouterClient:
    def __init__(self, *, api_key: str | None = None, max_retries: int = 3) -> None:
        self._api_key = api_key if api_key is not None else os.environ["OPENROUTER_API_KEY"]
        self._max_retries = max_retries

    def generate(
        self,
        request: LlmRequest,
        *,
        on_attempt: Callable[[int, str | None, Any, dict[str, Any]], None] | None = None,
    ) -> LlmResult:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/malpiszon/meal-orchestrator",
            "X-OpenRouter-Title": APP_NAME,
            "X-OpenRouter-Experimental-Metadata": "enabled",
        }
        attempt = 0
        feedback: str | None = None

        def _call(model: str) -> tuple[dict[str, Any], WeekAssessment]:
            nonlocal attempt
            attempt += 1
            body = json.dumps(
                {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": _build_message_content(request.payload, feedback),
                        }
                    ],
                    "response_format": _RESPONSE_FORMAT,
                    # Skip any endpoint that would silently ignore response_format
                    # rather than risk a non-schema-conforming completion from it.
                    "provider": {"require_parameters": True},
                }
            ).encode("utf-8")
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
                assessment = _response_structured(response, request.payload.menu, attempt)
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
            return _build_result(response, assessment, attempt, model)
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


def _build_result(
    response: dict[str, Any], assessment: WeekAssessment, attempt: int, requested_model: str
) -> LlmResult:
    usage = response.get("usage")
    token_usage = None
    if usage:
        token_usage = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
    model = response.get("model", requested_model)
    logger.info("openrouter: model=%s tokens=%s", model, token_usage)
    return LlmResult(
        structured=assessment,
        model=model,
        token_usage=token_usage,
        response_metadata=_response_metadata(response, attempt, compact_routing_metadata=True),
    )


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


def _response_structured(response: Any, menu: CanonicalMenu, attempt: int) -> WeekAssessment:
    text = _response_text(response, attempt)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _structured_output_error(f"invalid JSON: {exc}", attempt, response) from exc
    try:
        assessment = WeekAssessment.model_validate(parsed)
    except ValidationError as exc:
        raise _structured_output_error(str(exc), attempt, response) from exc
    problems = validate_completeness(assessment, menu)
    if problems:
        raise _incomplete_assessment_error(problems, attempt, response)
    return assessment


def _response_text(response: Any, attempt: int) -> str:
    try:
        text = response["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError) as exc:
        raise _empty_response_error("missing_message_content", attempt, response) from exc
    if not isinstance(text, str) or not text.strip():
        raise _empty_response_error("empty_message_content", attempt, response)
    choice = _first_choice(response) if isinstance(response, dict) else {}
    finish_reason = choice.get("finish_reason")
    if finish_reason in {"content_filter", "error", "length"}:
        error_type = _choice_error_type(choice) if finish_reason == "error" else None
        reason = f"finish_reason_{finish_reason}"
        if error_type is not None:
            reason = f"{reason}_{error_type}"
        raise OpenRouterResponseError(
            LlmFailureDetails(
                reason=reason,
                attempt=attempt,
                response=response,
                error_type=error_type,
            ),
            retryable=(
                finish_reason == "error" and error_type not in _NON_RETRYABLE_FINISH_ERROR_TYPES
            ),
        )
    return text


def _choice_error_type(choice: dict[str, Any]) -> str | None:
    return _error_type_from_error(choice.get("error"))


def _error_type_from_error(error: Any) -> str | None:
    if not isinstance(error, dict):
        return None
    metadata = error.get("metadata")
    if not isinstance(metadata, dict):
        return None
    error_type = metadata.get("error_type")
    return error_type if isinstance(error_type, str) else None


def _empty_response_error(reason: str, attempt: int, response: Any) -> EmptyLlmResponseError:
    return EmptyLlmResponseError(
        LlmFailureDetails(reason=reason, attempt=attempt, response=response)
    )


def _structured_output_error(
    parse_error: str, attempt: int, response: Any
) -> StructuredOutputError:
    return StructuredOutputError(
        LlmFailureDetails(reason="invalid_structured_output", attempt=attempt, response=response),
        parse_error=parse_error,
    )


def _incomplete_assessment_error(
    problems: list[str], attempt: int, response: Any
) -> IncompleteAssessmentError:
    return IncompleteAssessmentError(
        LlmFailureDetails(reason="incomplete_assessment", attempt=attempt, response=response),
        problems=problems,
    )


def _openrouter_http_error(
    error: urllib.error.HTTPError, attempt: int
) -> OpenRouterHttpError:
    response_body = getattr(error, "response_body", "")
    try:
        response = json.loads(response_body)
    except json.JSONDecodeError:
        response = {"error": {"code": error.code, "message": response_body or error.reason}}
    error_type = (
        _error_type_from_error(response.get("error")) if isinstance(response, dict) else None
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


def _first_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    choice = choices[0]
    return choice if isinstance(choice, dict) else {}


def _response_metadata(
    response: Any, attempt: int, *, compact_routing_metadata: bool = False
) -> dict[str, Any]:
    response = response if isinstance(response, dict) else {}
    choice = _first_choice(response)
    routing_metadata = response.get("openrouter_metadata")
    if compact_routing_metadata:
        routing_metadata = _compact_routing_metadata(routing_metadata)
    return {
        "attempt": attempt,
        "generation_id": response.get("id"),
        "model": response.get("model"),
        "provider": response.get("provider"),
        "finish_reason": choice.get("finish_reason"),
        "native_finish_reason": choice.get("native_finish_reason"),
        "response_error": response.get("error"),
        "choice_error": choice.get("error"),
        "usage": response.get("usage"),
        "openrouter_metadata": routing_metadata,
    }


def _compact_routing_metadata(metadata: Any) -> Any:
    if not isinstance(metadata, dict):
        return metadata

    compact = {
        key: metadata[key]
        for key in ("requested", "strategy", "region", "summary", "attempt", "is_byok")
        if key in metadata
    }
    endpoints = metadata.get("endpoints")
    if not isinstance(endpoints, dict):
        return compact
    available = endpoints.get("available")
    if not isinstance(available, list):
        return compact
    selected_endpoints = [
        {key: endpoint[key] for key in ("provider", "model") if key in endpoint}
        for endpoint in available
        if isinstance(endpoint, dict) and endpoint.get("selected")
    ]
    if selected_endpoints:
        compact["selected_endpoints"] = selected_endpoints
    return compact


_UPSTREAM_MESSAGE_MAX_CHARS = 300


def _failure_message(details: LlmFailureDetails) -> str:
    metadata = details.to_metadata()
    # Omit finish_reason here when `reason` already spells it out (e.g.
    # "finish_reason_error_rate_limit_exceeded") to avoid saying the same thing twice.
    # This relies on `reason` being built as f"finish_reason_{finish_reason}[...]" at
    # each raise site in _response_text — keep that format in sync with this check.
    context_fields = [("native_finish_reason", metadata["native_finish_reason"])]
    if not details.reason.startswith("finish_reason_"):
        context_fields.insert(0, ("finish_reason", metadata["finish_reason"]))
    context_fields.append(("generation_id", metadata["generation_id"]))
    context = ", ".join(f"{key}={value}" for key, value in context_fields if value is not None)
    suffix = f" ({context})" if context else ""
    upstream_message = _upstream_error_message(metadata)
    detail = f": {upstream_message}" if upstream_message else ""
    return f"OpenRouter response has {details.reason.replace('_', ' ')}{suffix}{detail}"


def _upstream_error_message(metadata: dict[str, Any]) -> str | None:
    for key in ("choice_error", "response_error"):
        error = metadata.get(key)
        if isinstance(error, dict) and isinstance(error.get("message"), str) and error["message"]:
            message: str = error["message"]
            if len(message) > _UPSTREAM_MESSAGE_MAX_CHARS:
                message = message[:_UPSTREAM_MESSAGE_MAX_CHARS].rstrip() + "…"
            return message
    return None
