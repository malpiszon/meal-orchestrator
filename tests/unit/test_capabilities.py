from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from meal_orchestrator.llm.capabilities import (
    UnsupportedModelError,
    assert_structured_output_supported,
)


def _models_response(entries: list[dict]) -> bytes:
    return json.dumps({"data": entries}).encode("utf-8")


class TestAssertStructuredOutputSupported:
    def test_passes_when_model_supports_response_format(self) -> None:
        body = _models_response(
            [{"id": "openai/gpt-4o-mini", "supported_parameters": ["response_format", "tools"]}]
        )
        with patch("meal_orchestrator.llm.capabilities.get_json", return_value=body):
            assert_structured_output_supported("openai/gpt-4o-mini", api_key="test-key")

    def test_passes_when_model_supports_structured_outputs(self) -> None:
        body = _models_response(
            [{"id": "openai/gpt-4o-mini", "supported_parameters": ["structured_outputs"]}]
        )
        with patch("meal_orchestrator.llm.capabilities.get_json", return_value=body):
            assert_structured_output_supported("openai/gpt-4o-mini", api_key="test-key")

    def test_raises_when_model_lacks_structured_output_support(self) -> None:
        body = _models_response(
            [{"id": "some/basic-model", "supported_parameters": ["temperature"]}]
        )
        with patch("meal_orchestrator.llm.capabilities.get_json", return_value=body):
            with pytest.raises(UnsupportedModelError, match="does not support structured outputs"):
                assert_structured_output_supported("some/basic-model", api_key="test-key")

    def test_raises_when_model_not_found(self) -> None:
        body = _models_response(
            [{"id": "other/model", "supported_parameters": ["response_format"]}]
        )
        with patch("meal_orchestrator.llm.capabilities.get_json", return_value=body):
            with pytest.raises(UnsupportedModelError, match="not found"):
                assert_structured_output_supported("missing/model", api_key="test-key")

    def test_sends_bearer_token(self) -> None:
        captured = {}

        def fake_get_json(url, *, headers, timeout_seconds):
            captured["headers"] = headers
            return _models_response(
                [{"id": "openai/gpt-4o-mini", "supported_parameters": ["response_format"]}]
            )

        with patch("meal_orchestrator.llm.capabilities.get_json", side_effect=fake_get_json):
            assert_structured_output_supported("openai/gpt-4o-mini", api_key="secret-key")

        assert captured["headers"]["Authorization"] == "Bearer secret-key"
