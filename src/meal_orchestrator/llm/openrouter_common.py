from __future__ import annotations

import json
import logging
import urllib.error
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from meal_orchestrator import APP_NAME
from meal_orchestrator.domain import CanonicalMenu, LlmResult, PromptPayload
from meal_orchestrator.domain.llm_output import (
    WeekAssessment,
    validate_completeness,
    week_assessment_json_schema,
)

logger = logging.getLogger(__name__)

_HTTP_REFERER = "https://github.com/malpiszon/meal-orchestrator"

# OpenRouter's documented error_type taxonomy (openrouter.ai/docs/api-reference/errors)
# for errors embedded in choices[].error when finish_reason == "error". Types outside
# this set (including none at all, e.g. a bare provider timeout) are treated as
# transient and retried, matching prior behavior.
NON_RETRYABLE_FINISH_ERROR_TYPES = frozenset(
    {
        "authentication",
        "payment_required",
        "permission_denied",
        "context_length_exceeded",
        "content_policy_violation",
    }
)


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


def build_request_headers(api_key: str) -> dict[str, str]:
    """Headers common to every OpenRouter request: auth, app attribution, and
    the experimental-metadata opt-in that populates `openrouter_metadata` on
    responses (surfaced via `_response_metadata` into failure diagnostics for
    both the synchronous and batch clients).

    Shared by the synchronous client (`openrouter.py`) and the batch client
    (`openrouter_batch.py`) so both request paths authenticate, are
    attributed, and get the same diagnostic data identically.
    """
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": _HTTP_REFERER,
        "X-OpenRouter-Title": APP_NAME,
        "X-OpenRouter-Experimental-Metadata": "enabled",
    }


def build_request_body(
    model: str, payload: PromptPayload, feedback: str | None = None
) -> dict[str, Any]:
    """Build the chat-completion request body for one model call.

    Shared by the synchronous client (`openrouter.py`) and the batch client
    (`openrouter_batch.py`) so both send an identical request shape.
    """
    return {
        "model": model,
        "messages": [{"role": "user", "content": _build_message_content(payload, feedback)}],
        "response_format": _RESPONSE_FORMAT,
        # Skip any endpoint that would silently ignore response_format rather
        # than risk a non-schema-conforming completion from it.
        "provider": {"require_parameters": True},
    }


def parse_batch_completion(
    response: dict[str, Any], menu: CanonicalMenu, requested_model: str
) -> LlmResult:
    """Parse a single completed batch row's response body into an LlmResult.

    Reuses the same schema validation/completeness checks as the synchronous
    client, but performs no retries — a batch row that fails validation raises
    directly (OpenRouterResponseError) for the caller to handle as a per-row
    failure, since the batch has already run and there's nothing left to retry.
    """
    assessment = response_structured(response, menu, attempt=1)
    return build_result(response, assessment, attempt=1, requested_model=requested_model)


def build_result(
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
        attempt=attempt,
        token_usage=token_usage,
    )


def response_structured(response: Any, menu: CanonicalMenu, attempt: int) -> WeekAssessment:
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
                finish_reason == "error" and error_type not in NON_RETRYABLE_FINISH_ERROR_TYPES
            ),
        )
    return text


def _choice_error_type(choice: dict[str, Any]) -> str | None:
    return error_type_from_error(choice.get("error"))


def error_type_from_error(error: Any) -> str | None:
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


def _first_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    choice = choices[0]
    return choice if isinstance(choice, dict) else {}


def _response_metadata(response: Any, attempt: int) -> dict[str, Any]:
    response = response if isinstance(response, dict) else {}
    choice = _first_choice(response)
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
        "openrouter_metadata": response.get("openrouter_metadata"),
    }


_UPSTREAM_MESSAGE_MAX_CHARS = 300


def _failure_message(details: LlmFailureDetails) -> str:
    metadata = details.to_metadata()
    # Omit finish_reason here when `reason` already spells it out (e.g.
    # "finish_reason_error_rate_limit_exceeded") to avoid saying the same thing twice.
    # This relies on `reason` being built as f"finish_reason_{finish_reason}[...]" at
    # each raise site in `_response_text` — keep that format in sync with this check.
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
