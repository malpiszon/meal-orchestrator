from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from meal_orchestrator.domain import LlmRequest, PromptPayload
from meal_orchestrator.llm.openrouter import OpenRouterClient
from meal_orchestrator.retries import RetryError
from tests.unit.helpers import canonical_menu


def _make_request(model: str = "openai/gpt-4o-mini") -> LlmRequest:
    return LlmRequest(
        model=model,
        payload=PromptPayload(user_prompt="Choose the best meals.", menu=canonical_menu()),
        timeout_seconds=30,
    )


def _mock_response(text: str, model: str = "openai/gpt-4o-mini") -> bytes:
    return json.dumps(
        {
            "model": model,
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
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

    def test_sends_bearer_token(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["auth"] = req.get_header("Authorization")
            return _mock_urlopen(_mock_response("ok"))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = OpenRouterClient(api_key="secret-key")
            client.generate(_make_request())

        assert captured["auth"] == "Bearer secret-key"

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
