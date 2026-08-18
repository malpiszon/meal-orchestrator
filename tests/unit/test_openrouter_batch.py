from __future__ import annotations

import json
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
