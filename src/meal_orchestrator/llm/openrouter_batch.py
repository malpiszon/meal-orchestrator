from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from meal_orchestrator.domain import LlmResult, PromptPayload
from meal_orchestrator.http import get_json, post_json
from meal_orchestrator.llm.openrouter import (
    OpenRouterResponseError,
    build_request_body,
    build_request_headers,
    parse_batch_completion,
)

logger = logging.getLogger(__name__)

_BATCH_API_URL = "https://openrouter.ai/api/beta/batches"


class BatchStatus(StrEnum):
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


PENDING_STATUSES = frozenset(
    {BatchStatus.VALIDATING, BatchStatus.IN_PROGRESS, BatchStatus.FINALIZING}
)


@dataclass(frozen=True)
class BatchRequestRow:
    """One user's LLM request, to be submitted as a row in a single batch."""

    custom_id: str
    model: str
    payload: PromptPayload


@dataclass(frozen=True)
class BatchRowError:
    """A single batch row that didn't produce a usable result."""

    custom_id: str
    reason: str
    detail: str | None = None


def submit_batch(
    rows: list[BatchRequestRow], *, api_key: str | None = None, timeout_seconds: int = 60
) -> str:
    """Submit one batch containing every row and return the OpenRouter batch id."""
    if not rows:
        raise ValueError("submit_batch requires at least one row")
    body = json.dumps(
        {
            "endpoint": "/v1/chat/completions",
            "model": rows[0].model,
            "requests": [
                {"custom_id": row.custom_id, "body": build_request_body(row.model, row.payload)}
                for row in rows
            ],
        }
    ).encode("utf-8")
    raw = post_json(
        _BATCH_API_URL, headers=_headers(api_key), body=body, timeout_seconds=timeout_seconds
    )
    data = json.loads(raw.decode("utf-8"))
    batch_id = data["id"]
    logger.info("openrouter batch submitted: batch_id=%s rows=%d", batch_id, len(rows))
    return batch_id


def get_batch(
    batch_id: str, *, api_key: str | None = None, timeout_seconds: int = 60
) -> dict[str, Any]:
    """Fetch the current state of a batch (status, and results once completed)."""
    raw = get_json(
        f"{_BATCH_API_URL}/{batch_id}",
        headers=_headers(api_key),
        timeout_seconds=timeout_seconds,
    )
    return json.loads(raw.decode("utf-8"))


def batch_status(batch_data: dict[str, Any]) -> BatchStatus:
    return BatchStatus(batch_data["status"])


def parse_batch_results(
    rows: list[BatchRequestRow], batch_data: dict[str, Any]
) -> tuple[dict[str, LlmResult], dict[str, BatchRowError]]:
    """Match a completed batch's results back to rows by custom_id.

    A row missing from the results, or one whose response didn't parse, is
    reported as a `BatchRowError` rather than raising — a partial batch
    failure must not take down every other user's already-successful result.
    """
    rows_by_custom_id = {row.custom_id: row for row in rows}
    results: dict[str, LlmResult] = {}
    errors: dict[str, BatchRowError] = {}

    returned_ids = set()
    for entry in batch_data.get("results") or []:
        custom_id = entry.get("custom_id")
        row = rows_by_custom_id.get(custom_id)
        if row is None:
            continue
        returned_ids.add(custom_id)
        response_wrapper = entry.get("response")
        if entry.get("error") or not response_wrapper or response_wrapper.get("status_code") != 200:
            errors[custom_id] = BatchRowError(
                custom_id=custom_id,
                reason="batch_row_error",
                detail=str(entry.get("error") or response_wrapper),
            )
            continue
        body = response_wrapper.get("body") or {}
        try:
            results[custom_id] = parse_batch_completion(body, row.payload.menu, row.model)
        except OpenRouterResponseError as exc:
            errors[custom_id] = BatchRowError(
                custom_id=custom_id, reason=exc.details.reason, detail=str(exc)
            )

    for custom_id in rows_by_custom_id:
        if custom_id not in returned_ids:
            errors[custom_id] = BatchRowError(custom_id=custom_id, reason="missing_from_batch")

    return results, errors


def _headers(api_key: str | None) -> dict[str, str]:
    key = api_key if api_key is not None else os.environ["OPENROUTER_API_KEY"]
    return build_request_headers(key)
