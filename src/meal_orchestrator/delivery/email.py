from __future__ import annotations

import json
import logging
import os

from meal_orchestrator import USER_AGENT
from meal_orchestrator.domain import EmailMessage
from meal_orchestrator.http import post_json
from meal_orchestrator.retries import is_transient_http_error, with_retries

logger = logging.getLogger(__name__)

_API_URL = "https://api.resend.com/emails"
_BASE_DELAY = 1.0
_BACKOFF_FACTOR = 2.0


class ResendEmailClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout_seconds: int = 30,
    ) -> None:
        self._api_key = api_key
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds

    def send(self, message: EmailMessage, idempotency_key: str) -> None:
        api_key = self._api_key if self._api_key is not None else os.environ["RESEND_API_KEY"]
        body = json.dumps(
            {
                "from": message.from_address,
                "to": [message.to],
                "subject": message.subject,
                "text": message.body,
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key,
            "User-Agent": USER_AGENT,
        }

        def _call() -> None:
            post_json(_API_URL, headers=headers, body=body, timeout_seconds=self._timeout_seconds)

        with_retries(
            _call,
            max_attempts=self._max_retries,
            base_delay_seconds=_BASE_DELAY,
            backoff_factor=_BACKOFF_FACTOR,
            retryable=is_transient_http_error,
            operation_name=f"resend email to={message.to}",
        )
        logger.info("email sent: to=%s subject=%s", message.to, message.subject)
