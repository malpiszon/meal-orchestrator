from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

from meal_orchestrator import USER_AGENT
from meal_orchestrator.domain import DiscordMessage
from meal_orchestrator.retries import is_transient_http_error, with_retries

logger = logging.getLogger(__name__)

_BASE_DELAY = 1.0
_BACKOFF_FACTOR = 2.0


class DiscordWebhookClient:
    def __init__(
        self,
        *,
        max_retries: int = 3,
        timeout_seconds: int = 10,
    ) -> None:
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds

    def notify(self, message: DiscordMessage) -> None:
        webhook_url = os.environ[message.webhook_env]
        body = json.dumps({"content": message.content}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

        def _call() -> None:
            req = urllib.request.Request(webhook_url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                    resp.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise urllib.error.HTTPError(
                    exc.url, exc.code, f"{exc.reason} — {detail}", exc.headers, None
                ) from exc

        with_retries(
            _call,
            max_attempts=self._max_retries,
            base_delay_seconds=_BASE_DELAY,
            backoff_factor=_BACKOFF_FACTOR,
            retryable=is_transient_http_error,
            operation_name=f"discord webhook env={message.webhook_env}",
        )
        logger.info("discord notification sent: webhook_env=%s", message.webhook_env)
