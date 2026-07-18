from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from meal_orchestrator import APP_NAME
from meal_orchestrator.domain import LlmRequest, LlmResult, PromptPayload
from meal_orchestrator.http import post_json
from meal_orchestrator.retries import is_transient_http_error, with_retries

logger = logging.getLogger(__name__)

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_BASE_DELAY = 1.0
_BACKOFF_FACTOR = 2.0


@dataclass(frozen=True)
class LlmFailureDetails:
    """Diagnostic data returned with an unusable OpenRouter completion."""

    reason: str
    attempt: int
    response: Any

    def to_metadata(self) -> dict[str, Any]:
        return {"reason": self.reason, **_response_metadata(self.response, self.attempt)}


class EmptyLlmResponseError(RuntimeError):
    """Raised when OpenRouter returns a completion without usable text."""

    def __init__(self, details: LlmFailureDetails) -> None:
        super().__init__(_failure_message(details))
        self.details = details


def _build_message_content(payload: PromptPayload) -> list[dict[str, str]]:
    # Separate blocks keep instructions and data structurally distinct for large JSON payloads.
    menu_json = json.dumps(
        payload.menu.to_compact_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {"type": "text", "text": f"User instructions:\n{payload.user_prompt}"},
        {"type": "text", "text": f"Canonical menu JSON:\n{menu_json}"},
        {"type": "text", "text": "Return plain text only."},
    ]


class OpenRouterClient:
    def __init__(self, *, api_key: str | None = None, max_retries: int = 3) -> None:
        self._api_key = api_key if api_key is not None else os.environ["OPENROUTER_API_KEY"]
        self._max_retries = max_retries

    def generate(self, request: LlmRequest) -> LlmResult:
        body = json.dumps(
            {
                "model": request.model,
                "messages": [
                    {"role": "user", "content": _build_message_content(request.payload)}
                ],
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/malpiszon/meal-orchestrator",
            "X-OpenRouter-Title": APP_NAME,
            "X-OpenRouter-Experimental-Metadata": "enabled",
        }
        attempt = 0

        def _call() -> tuple[dict[str, Any], str]:
            nonlocal attempt
            attempt += 1
            raw = post_json(
                _API_URL, headers=headers, body=body, timeout_seconds=request.timeout_seconds
            )
            response = json.loads(raw.decode("utf-8"))
            return response, _response_text(response, attempt)

        response, text = with_retries(
            _call,
            max_attempts=self._max_retries,
            base_delay_seconds=_BASE_DELAY,
            backoff_factor=_BACKOFF_FACTOR,
            retryable=lambda exc: is_transient_http_error(exc)
            or isinstance(exc, EmptyLlmResponseError),
            operation_name=f"openrouter generate model={request.model}",
        )

        usage = response.get("usage")
        token_usage = None
        if usage:
            token_usage = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        model = response.get("model", request.model)

        logger.info("openrouter: model=%s tokens=%s", model, token_usage)
        return LlmResult(
            text=text,
            model=model,
            token_usage=token_usage,
            response_metadata=_response_metadata(response, attempt),
        )


def _response_text(response: Any, attempt: int) -> str:
    try:
        text = response["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError) as exc:
        raise _empty_response_error("missing_message_content", attempt, response) from exc
    if not isinstance(text, str) or not text.strip():
        raise _empty_response_error("empty_message_content", attempt, response)
    return text


def _empty_response_error(reason: str, attempt: int, response: Any) -> EmptyLlmResponseError:
    return EmptyLlmResponseError(
        LlmFailureDetails(reason=reason, attempt=attempt, response=response)
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
