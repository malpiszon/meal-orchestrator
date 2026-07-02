from __future__ import annotations

import json
import logging
import os
from typing import Any

from meal_orchestrator import APP_NAME
from meal_orchestrator.domain import LlmRequest, LlmResult, PromptPayload
from meal_orchestrator.http import post_json
from meal_orchestrator.retries import is_transient_http_error, with_retries

logger = logging.getLogger(__name__)

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_BASE_DELAY = 1.0
_BACKOFF_FACTOR = 2.0


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
        }

        def _call() -> dict[str, Any]:
            raw = post_json(
                _API_URL, headers=headers, body=body, timeout_seconds=request.timeout_seconds
            )
            return json.loads(raw.decode("utf-8"))

        response = with_retries(
            _call,
            max_attempts=self._max_retries,
            base_delay_seconds=_BASE_DELAY,
            backoff_factor=_BACKOFF_FACTOR,
            retryable=is_transient_http_error,
            operation_name=f"openrouter generate model={request.model}",
        )

        text = response["choices"][0]["message"]["content"]
        usage = response.get("usage")
        token_usage = None
        if usage:
            token_usage = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        model = response.get("model", request.model)

        logger.info("openrouter: model=%s tokens=%s", model, token_usage)
        return LlmResult(text=text, model=model, token_usage=token_usage)
