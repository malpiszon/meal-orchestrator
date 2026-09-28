from __future__ import annotations

import json
import time
import urllib.error
from email.message import Message
from unittest.mock import patch

import pytest

from meal_orchestrator.domain import PromptPayload
from meal_orchestrator.llm.openrouter_batch import (
    BatchRequestRow,
    BatchStatus,
    batch_status,
    get_batch,
    parse_batch_results,
    submit_batch,
)
from meal_orchestrator.retries import RetryError
from tests.unit.helpers import canonical_menu, week_assessment


def _row(custom_id: str = "run-1:alan", model: str = "openai/gpt-4o-mini") -> BatchRequestRow:
    return BatchRequestRow(
        custom_id=custom_id,
        model=model,
        payload=PromptPayload(
            app_prompt="Assess every meal variant.",
            user_prompt="Choose the best meals.",
            menu=canonical_menu(),
        ),
    )


def _completion_body(custom_id: str, model: str = "openai/gpt-4o-mini") -> dict:
    content = week_assessment(canonical_menu()).model_dump_json()
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "model": model,
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        },
        "error": None,
    }


def test_submit_batch_posts_one_row_per_request() -> None:
    with patch("meal_orchestrator.llm.openrouter_batch.post_json") as mock_post:
        mock_post.return_value = json.dumps({"id": "batch-123"}).encode("utf-8")
        batch_id = submit_batch([_row("run-1:alan"), _row("run-1:bob")], api_key="key")

    assert batch_id == "batch-123"
    _, kwargs = mock_post.call_args
    body = json.loads(kwargs["body"].decode("utf-8"))
    assert body["endpoint"] == "/v1/chat/completions"
    assert [r["custom_id"] for r in body["requests"]] == ["run-1:alan", "run-1:bob"]
    assert body["requests"][0]["body"]["response_format"]["type"] == "json_schema"


def _http_error(code: int, reset_in: float | None = None) -> urllib.error.HTTPError:
    error = urllib.error.HTTPError("https://example", code, "err", None, None)  # type: ignore[arg-type]
    if reset_in is not None:
        reset_ms = int((time.time() + reset_in) * 1000)
        error.response_body = json.dumps(  # type: ignore[attr-defined]
            {"error": {"metadata": {"headers": {"X-RateLimit-Reset": str(reset_ms)}}}}
        )
    return error


def test_submit_batch_retries_rate_limit_then_succeeds() -> None:
    with (
        patch("meal_orchestrator.llm.openrouter_batch.post_json") as mock_post,
        patch("meal_orchestrator.retries.time.sleep") as mock_sleep,
    ):
        mock_post.side_effect = [_http_error(429), json.dumps({"id": "batch-123"}).encode("utf-8")]
        batch_id = submit_batch([_row("run-1:alan")], api_key="key")

    assert batch_id == "batch-123"
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once_with(30.0)


def test_submit_batch_waits_until_rate_limit_reset() -> None:
    with (
        patch("meal_orchestrator.llm.openrouter_batch.post_json") as mock_post,
        patch("meal_orchestrator.retries.time.sleep") as mock_sleep,
    ):
        mock_post.side_effect = [
            _http_error(429, reset_in=45),
            json.dumps({"id": "batch-123"}).encode("utf-8"),
        ]
        submit_batch([_row("run-1:alan")], api_key="key")

    (delay,) = mock_sleep.call_args.args
    assert 40 < delay <= 45


@pytest.mark.parametrize("reset_in", [-10, 600])
def test_submit_batch_ignores_unusable_reset(reset_in: float) -> None:
    with (
        patch("meal_orchestrator.llm.openrouter_batch.post_json") as mock_post,
        patch("meal_orchestrator.retries.time.sleep") as mock_sleep,
    ):
        mock_post.side_effect = [
            _http_error(429, reset_in=reset_in),
            json.dumps({"id": "batch-123"}).encode("utf-8"),
        ]
        submit_batch([_row("run-1:alan")], api_key="key")

    mock_sleep.assert_called_once_with(30.0)


def test_submit_batch_reads_reset_from_body_when_headers_are_empty() -> None:
    error = _http_error(429, reset_in=45)
    error.hdrs = Message()  # an empty Message is falsy but not None
    with (
        patch("meal_orchestrator.llm.openrouter_batch.post_json") as mock_post,
        patch("meal_orchestrator.retries.time.sleep") as mock_sleep,
    ):
        mock_post.side_effect = [error, json.dumps({"id": "batch-123"}).encode("utf-8")]
        submit_batch([_row("run-1:alan")], api_key="key")

    (delay,) = mock_sleep.call_args.args
    assert 40 < delay <= 45


def test_submit_batch_gives_up_after_repeated_rate_limits() -> None:
    with (
        patch("meal_orchestrator.llm.openrouter_batch.post_json") as mock_post,
        patch("meal_orchestrator.retries.time.sleep") as mock_sleep,
    ):
        mock_post.side_effect = _http_error(429)
        with pytest.raises(RetryError, match="3 attempt"):
            submit_batch([_row("run-1:alan")], api_key="key")

    assert mock_post.call_count == 3
    assert [c.args[0] for c in mock_sleep.call_args_list] == [30.0, 60.0]


def test_submit_batch_does_not_retry_other_errors() -> None:
    with (
        patch("meal_orchestrator.llm.openrouter_batch.post_json") as mock_post,
        patch("meal_orchestrator.retries.time.sleep") as mock_sleep,
    ):
        mock_post.side_effect = _http_error(400)
        with pytest.raises(urllib.error.HTTPError):
            submit_batch([_row("run-1:alan")], api_key="key")

    assert mock_post.call_count == 1
    mock_sleep.assert_not_called()


def test_submit_batch_requires_rows() -> None:
    with pytest.raises(ValueError):
        submit_batch([], api_key="key")


def test_get_batch_and_status() -> None:
    with patch("meal_orchestrator.llm.openrouter_batch.get_json") as mock_get:
        mock_get.return_value = json.dumps({"id": "batch-123", "status": "in_progress"}).encode(
            "utf-8"
        )
        data = get_batch("batch-123", api_key="key")

    assert batch_status(data) == BatchStatus.IN_PROGRESS


def test_parse_batch_results_matches_by_custom_id() -> None:
    rows = [_row("run-1:alan"), _row("run-1:bob")]
    batch_data = {"results": [_completion_body("run-1:alan"), _completion_body("run-1:bob")]}

    results, errors = parse_batch_results(rows, batch_data)

    assert set(results) == {"run-1:alan", "run-1:bob"}
    assert errors == {}
    assert results["run-1:alan"].model == "openai/gpt-4o-mini"


def test_parse_batch_results_reports_missing_row_as_error() -> None:
    rows = [_row("run-1:alan"), _row("run-1:bob")]
    batch_data = {"results": [_completion_body("run-1:alan")]}

    results, errors = parse_batch_results(rows, batch_data)

    assert set(results) == {"run-1:alan"}
    assert errors["run-1:bob"].reason == "missing_from_batch"


def test_parse_batch_results_reports_row_level_error() -> None:
    rows = [_row("run-1:alan")]
    batch_data = {
        "results": [
            {
                "custom_id": "run-1:alan",
                "response": None,
                "error": {"message": "internal error"},
            }
        ]
    }

    results, errors = parse_batch_results(rows, batch_data)

    assert results == {}
    assert errors["run-1:alan"].reason == "batch_row_error"


def test_parse_batch_results_reports_schema_validation_failure() -> None:
    rows = [_row("run-1:alan")]
    bad_body = _completion_body("run-1:alan")
    bad_body["response"]["body"]["choices"][0]["message"]["content"] = "not json"
    batch_data = {"results": [bad_body]}

    results, errors = parse_batch_results(rows, batch_data)

    assert results == {}
    assert errors["run-1:alan"].reason == "invalid_structured_output"
