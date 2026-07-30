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
from meal_orchestrator.retries import is_transient_http_error, with_retries

logger = logging.getLogger(__name__)

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_BASE_DELAY = 1.0
_BACKOFF_FACTOR = 2.0

_SCAFFOLDING_INSTRUCTION = (
    "Assess every meal variant present in the menu JSON below: give each one a score "
    "and up to two short justification points. Do not select a single winner and skip "
    "the rest — every variant of every meal of every day must be assessed."
)


@dataclass(frozen=True)
class LlmFailureDetails:
    """Diagnostic data returned with an unusable OpenRouter completion."""

    reason: str
    attempt: int
    response: Any
    http_status: int | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "http_status": self.http_status,
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
        {"type": "text", "text": f"User instructions:\n{payload.user_prompt}"},
        {"type": "text", "text": f"Canonical menu JSON:\n{menu_json}"},
        {"type": "text", "text": _SCAFFOLDING_INSTRUCTION},
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

        def _call() -> tuple[dict[str, Any], WeekAssessment]:
            nonlocal attempt
            attempt += 1
            body = json.dumps(
                {
                    "model": request.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": _build_message_content(request.payload, feedback),
                        }
                    ],
                    "response_format": _RESPONSE_FORMAT,
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
                    on_attempt(attempt, feedback, None, _network_error_outcome(exc))
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

        response, assessment = with_retries(
            _call,
            max_attempts=self._max_retries,
            base_delay_seconds=_BASE_DELAY,
            backoff_factor=_BACKOFF_FACTOR,
            retryable=lambda exc: is_transient_http_error(exc)
            or (isinstance(exc, OpenRouterResponseError) and exc.retryable),
            operation_name=f"openrouter generate model={request.model}",
            on_retry=_on_retry,
        )
        return _build_result(response, assessment, attempt, request.model)


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


def _network_error_outcome(exc: Exception) -> dict[str, Any]:
    return {"accepted": False, "reason": "network_error", "error": str(exc)}


def _feedback_for(exc: OpenRouterResponseError) -> str | None:
    if isinstance(exc, StructuredOutputError):
        return (
            "Your previous response did not match the required JSON schema "
            f"({exc.parse_error}). Return a response that strictly matches the schema."
        )
    if isinstance(exc, IncompleteAssessmentError):
        joined = "; ".join(exc.problems)
        return (
            f"Your previous response was incomplete: {joined}. Assess every meal and "
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
        raise OpenRouterResponseError(
            LlmFailureDetails(
                reason=f"finish_reason_{finish_reason}",
                attempt=attempt,
                response=response,
            ),
            retryable=finish_reason == "error",
        )
    return text


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
    return OpenRouterHttpError(
        error,
        LlmFailureDetails(
            reason="http_error",
            attempt=attempt,
            response=response,
            http_status=error.code,
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


def _failure_message(details: LlmFailureDetails) -> str:
    metadata = details.to_metadata()
    context = ", ".join(
        f"{key}={value}"
        for key, value in (
            ("finish_reason", metadata["finish_reason"]),
            ("native_finish_reason", metadata["native_finish_reason"]),
            ("generation_id", metadata["generation_id"]),
        )
        if value is not None
    )
    suffix = f" ({context})" if context else ""
    return f"OpenRouter response has {details.reason.replace('_', ' ')}{suffix}"
