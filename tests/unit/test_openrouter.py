from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from meal_orchestrator.domain import LlmRequest, PromptPayload
from meal_orchestrator.llm.openrouter import (
    EmptyLlmResponseError,
    OpenRouterClient,
    OpenRouterHttpError,
    OpenRouterResponseError,
)
from meal_orchestrator.retries import RetryError
from tests.unit.helpers import canonical_menu


def _make_request(model: str = "openai/gpt-4o-mini") -> LlmRequest:
    return LlmRequest(
        model=model,
        payload=PromptPayload(user_prompt="Choose the best meals.", menu=canonical_menu()),
        timeout_seconds=30,
    )


def _mock_response(text: str | None, model: str = "openai/gpt-4o-mini") -> bytes:
    return json.dumps(
        {
            "model": model,
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
    ).encode("utf-8")


def _mock_empty_response() -> bytes:
    return json.dumps(
        {
            "id": "gen-example",
            "model": "openai/gpt-4o-mini",
            "provider": "Example Provider",
            "choices": [
                {
                    "message": {"content": None},
                    "finish_reason": "error",
                    "native_finish_reason": "provider_timeout",
                    "error": {"code": 504, "message": "Provider timed out"},
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "openrouter_metadata": {"attempt": 1},
        }
    ).encode("utf-8")


def _mock_partial_response(finish_reason: str) -> bytes:
    return json.dumps(
        {
            "id": "gen-example",
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "message": {"content": "Partial meal plan"},
                    "finish_reason": finish_reason,
                }
            ],
        }
    ).encode("utf-8")


def _mock_urlopen(response_body: bytes):
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestOpenRouterClientGenerate:
    def test_returns_llm_result_with_text(self) -> None:
        with patch(
            "urllib.request.urlopen", return_value=_mock_urlopen(_mock_response("Eat salad."))
        ):
            client = OpenRouterClient(api_key="test-key")
            result = client.generate(_make_request())

        assert result.text == "Eat salad."

    def test_returns_model_from_response(self) -> None:
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(_mock_response("ok", model="openai/gpt-4o-mini")),
        ):
            client = OpenRouterClient(api_key="test-key")
            result = client.generate(_make_request())

        assert result.model == "openai/gpt-4o-mini"

    def test_returns_token_usage(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_mock_response("ok"))):
            client = OpenRouterClient(api_key="test-key")
            result = client.generate(_make_request())

        assert result.token_usage == {"prompt_tokens": 100, "completion_tokens": 50}

    def test_returns_response_metadata(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_mock_response("ok"))):
            client = OpenRouterClient(api_key="test-key")
            result = client.generate(_make_request())

        assert result.response_metadata == {
            "attempt": 1,
            "generation_id": None,
            "model": "openai/gpt-4o-mini",
            "provider": None,
            "finish_reason": None,
            "native_finish_reason": None,
            "response_error": None,
            "choice_error": None,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "openrouter_metadata": None,
        }

    def test_sends_bearer_token(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["auth"] = req.get_header("Authorization")
            return _mock_urlopen(_mock_response("ok"))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = OpenRouterClient(api_key="secret-key")
            client.generate(_make_request())

        assert captured["auth"] == "Bearer secret-key"

    def test_requests_openrouter_routing_metadata(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["metadata"] = req.get_header("X-openrouter-experimental-metadata")
            return _mock_urlopen(_mock_response("ok"))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = OpenRouterClient(api_key="test-key")
            client.generate(_make_request())

        assert captured["metadata"] == "enabled"

    def test_sends_model_in_request_body(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _mock_urlopen(_mock_response("ok"))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = OpenRouterClient(api_key="test-key")
            client.generate(_make_request(model="anthropic/claude-haiku-4-5"))

        assert captured["body"]["model"] == "anthropic/claude-haiku-4-5"

    def test_message_content_is_array_with_separate_json_block(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _mock_urlopen(_mock_response("ok"))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = OpenRouterClient(api_key="test-key")
            client.generate(_make_request())

        content = captured["body"]["messages"][0]["content"]
        assert isinstance(content, list)
        texts = [block["text"] for block in content]
        assert any("Choose the best meals." in t for t in texts)
        assert any("Canonical menu JSON:" in t for t in texts)
        assert any("Return plain text only." in t for t in texts)

    def test_json_block_is_separate_from_instructions(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _mock_urlopen(_mock_response("ok"))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = OpenRouterClient(api_key="test-key")
            client.generate(_make_request())

        content = captured["body"]["messages"][0]["content"]
        instruction_block = next(b for b in content if "Choose the best meals." in b["text"])
        json_block = next(b for b in content if "Canonical menu JSON:" in b["text"])
        assert instruction_block is not json_block

    def test_retries_on_500_and_eventually_succeeds(self) -> None:
        http_500 = urllib.error.HTTPError(
            url="https://openrouter.ai", code=500, msg="Server Error", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise http_500
            return _mock_urlopen(_mock_response("ok"))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                result = client.generate(_make_request())

        assert call_count == 3
        assert result.text == "ok"

    def test_raises_retry_error_after_exhausted_retries(self) -> None:
        http_500 = urllib.error.HTTPError(
            url="https://openrouter.ai", code=500, msg="Server Error", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        with patch("urllib.request.urlopen", side_effect=http_500):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                with pytest.raises(RetryError):
                    client.generate(_make_request())

    def test_retries_when_response_has_no_text_and_eventually_succeeds(self) -> None:
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_urlopen(_mock_response(None))
            return _mock_urlopen(_mock_response("ok"))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                result = client.generate(_make_request())

        assert call_count == 2
        assert result.text == "ok"

    def test_raises_retry_error_when_response_has_no_text_after_all_attempts(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_mock_empty_response())):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                with pytest.raises(RetryError) as exc_info:
                    client.generate(_make_request())

        assert isinstance(exc_info.value.last_exception, EmptyLlmResponseError)
        details = exc_info.value.last_exception.details
        assert details.to_metadata() == {
            "reason": "empty_message_content",
            "http_status": None,
            "attempt": 3,
            "generation_id": "gen-example",
            "model": "openai/gpt-4o-mini",
            "provider": "Example Provider",
            "finish_reason": "error",
            "native_finish_reason": "provider_timeout",
            "response_error": None,
            "choice_error": {"code": 504, "message": "Provider timed out"},
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "openrouter_metadata": {"attempt": 1},
        }

    def test_retries_when_response_ends_with_error_after_partial_content(self) -> None:
        with patch(
            "urllib.request.urlopen", return_value=_mock_urlopen(_mock_partial_response("error"))
        ) as urlopen:
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                with pytest.raises(RetryError) as exc_info:
                    client.generate(_make_request())

        assert urlopen.call_count == 3
        assert isinstance(exc_info.value.last_exception, OpenRouterResponseError)
        assert exc_info.value.last_exception.details.reason == "finish_reason_error"

    @pytest.mark.parametrize("finish_reason", ["content_filter", "length"])
    def test_rejects_partial_response_with_non_retryable_finish_reason(
        self, finish_reason: str
    ) -> None:
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(_mock_partial_response(finish_reason)),
        ) as urlopen:
            client = OpenRouterClient(api_key="test-key", max_retries=3)
            with pytest.raises(OpenRouterResponseError) as exc_info:
                client.generate(_make_request())

        assert urlopen.call_count == 1
        assert exc_info.value.details.reason == f"finish_reason_{finish_reason}"

    def test_preserves_structured_http_error_details(self) -> None:
        http_error = urllib.error.HTTPError(
            url="https://openrouter.ai", code=429, msg="Too Many Requests", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        http_error.response_body = json.dumps(
            {"error": {"code": 429, "message": "Rate limit exceeded"}}
        )

        with patch("meal_orchestrator.llm.openrouter.post_json", side_effect=http_error):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=2)
                with pytest.raises(RetryError) as exc_info:
                    client.generate(_make_request())

        assert isinstance(exc_info.value.last_exception, OpenRouterHttpError)
        assert exc_info.value.last_exception.details.to_metadata()["response_error"] == {
            "code": 429,
            "message": "Rate limit exceeded",
        }

    def test_non_retryable_error_raised_immediately(self) -> None:
        http_401 = urllib.error.HTTPError(
            url="https://openrouter.ai", code=401, msg="Unauthorized", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise http_401

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = OpenRouterClient(api_key="bad-key", max_retries=3)
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                client.generate(_make_request())

        assert exc_info.value.code == 401
        assert call_count == 1
