from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from meal_orchestrator.domain import LlmRequest, PromptPayload
from meal_orchestrator.llm.openrouter import (
    EmptyLlmResponseError,
    IncompleteAssessmentError,
    OpenRouterClient,
    OpenRouterHttpError,
    OpenRouterResponseError,
    StructuredOutputError,
)
from meal_orchestrator.retries import RetryError
from tests.unit.helpers import canonical_menu, week_assessment


def _make_request(
    model: str = "openai/gpt-4o-mini", fallback_models: list[str] | None = None
) -> LlmRequest:
    return LlmRequest(
        model=model,
        payload=PromptPayload(user_prompt="Choose the best meals.", menu=canonical_menu()),
        timeout_seconds=30,
        fallback_models=fallback_models or [],
    )


def _assessment_json(**kwargs) -> str:
    return week_assessment(canonical_menu(), **kwargs).model_dump_json()


def _mock_response(
    content: str | None,
    model: str = "openai/gpt-4o-mini",
    openrouter_metadata: dict | None = None,
) -> bytes:
    response = {
        "model": model,
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }
    if openrouter_metadata is not None:
        response["openrouter_metadata"] = openrouter_metadata
    return json.dumps(response).encode("utf-8")


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


def _mock_error_response(error_type: str) -> bytes:
    return json.dumps(
        {
            "id": "gen-example",
            "model": "openai/gpt-4o-mini",
            "choices": [
                {
                    "message": {"content": "Partial meal plan"},
                    "finish_reason": "error",
                    "error": {
                        "code": 429,
                        "message": "rate limited",
                        "metadata": {"error_type": error_type},
                    },
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
    def test_returns_llm_result_with_structured_assessment(self) -> None:
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(_mock_response(_assessment_json())),
        ):
            client = OpenRouterClient(api_key="test-key")
            result = client.generate(_make_request())

        assert result.structured == week_assessment(canonical_menu())

    def test_returns_model_from_response(self) -> None:
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(
                _mock_response(_assessment_json(), model="openai/gpt-4o-mini")
            ),
        ):
            client = OpenRouterClient(api_key="test-key")
            result = client.generate(_make_request())

        assert result.model == "openai/gpt-4o-mini"

    def test_returns_token_usage(self) -> None:
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(_mock_response(_assessment_json())),
        ):
            client = OpenRouterClient(api_key="test-key")
            result = client.generate(_make_request())

        assert result.token_usage == {"prompt_tokens": 100, "completion_tokens": 50}

    def test_returns_response_metadata(self) -> None:
        routing_metadata = {
            "requested": "openai/gpt-4o-mini",
            "strategy": "direct",
            "region": "WAW",
            "summary": "available=2, selected=OpenAI",
            "attempt": 1,
            "is_byok": False,
            "endpoints": {
                "total": 3,
                "available": [
                    {
                        "provider": "OpenAI",
                        "model": "openai/gpt-4o-mini-2025-08-07",
                        "selected": True,
                    },
                    {
                        "provider": "Azure",
                        "model": "openai/gpt-4o-mini-2025-08-07",
                        "selected": False,
                    },
                ],
            },
        }
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(
                _mock_response(_assessment_json(), openrouter_metadata=routing_metadata)
            ),
        ):
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
            "openrouter_metadata": {
                "requested": "openai/gpt-4o-mini",
                "strategy": "direct",
                "region": "WAW",
                "summary": "available=2, selected=OpenAI",
                "attempt": 1,
                "is_byok": False,
                "selected_endpoints": [
                    {"provider": "OpenAI", "model": "openai/gpt-4o-mini-2025-08-07"}
                ],
            },
        }

    def test_sends_bearer_token(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["auth"] = req.get_header("Authorization")
            return _mock_urlopen(_mock_response(_assessment_json()))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = OpenRouterClient(api_key="secret-key")
            client.generate(_make_request())

        assert captured["auth"] == "Bearer secret-key"

    def test_requests_openrouter_routing_metadata(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["metadata"] = req.get_header("X-openrouter-experimental-metadata")
            return _mock_urlopen(_mock_response(_assessment_json()))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = OpenRouterClient(api_key="test-key")
            client.generate(_make_request())

        assert captured["metadata"] == "enabled"

    def test_sends_model_in_request_body(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _mock_urlopen(_mock_response(_assessment_json()))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = OpenRouterClient(api_key="test-key")
            client.generate(_make_request(model="anthropic/claude-haiku-4-5"))

        assert captured["body"]["model"] == "anthropic/claude-haiku-4-5"

    def test_sends_only_model_when_no_fallback_models_configured(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _mock_urlopen(_mock_response(_assessment_json()))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = OpenRouterClient(api_key="test-key")
            client.generate(_make_request(model="openai/gpt-4o-mini"))

        assert captured["body"]["model"] == "openai/gpt-4o-mini"
        assert "models" not in captured["body"]

    def test_sends_models_array_when_fallback_models_configured(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _mock_urlopen(_mock_response(_assessment_json()))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = OpenRouterClient(api_key="test-key")
            client.generate(
                _make_request(
                    model="openai/gpt-4o-mini",
                    fallback_models=["openai/gpt-4.1-mini", "anthropic/claude-haiku-4-5"],
                )
            )

        assert "model" not in captured["body"]
        assert captured["body"]["models"] == [
            "openai/gpt-4o-mini",
            "openai/gpt-4.1-mini",
            "anthropic/claude-haiku-4-5",
        ]

    def test_sends_require_parameters_provider_preference(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _mock_urlopen(_mock_response(_assessment_json()))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = OpenRouterClient(api_key="test-key")
            client.generate(_make_request())

        assert captured["body"]["provider"] == {"require_parameters": True}

    def test_sends_strict_json_schema_response_format(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _mock_urlopen(_mock_response(_assessment_json()))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = OpenRouterClient(api_key="test-key")
            client.generate(_make_request())

        response_format = captured["body"]["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["schema"]["properties"]["days"]

    def test_message_content_is_array_with_separate_json_block(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _mock_urlopen(_mock_response(_assessment_json()))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = OpenRouterClient(api_key="test-key")
            client.generate(_make_request())

        content = captured["body"]["messages"][0]["content"]
        assert isinstance(content, list)
        texts = [block["text"] for block in content]
        assert any("Choose the best meals." in t for t in texts)
        assert any("Canonical menu JSON:" in t for t in texts)
        assert any("Assess every meal variant" in t for t in texts)

    def test_json_block_is_separate_from_instructions(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _mock_urlopen(_mock_response(_assessment_json()))

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
            return _mock_urlopen(_mock_response(_assessment_json()))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                result = client.generate(_make_request())

        assert call_count == 3
        assert result.structured == week_assessment(canonical_menu())

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
            return _mock_urlopen(_mock_response(_assessment_json()))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                result = client.generate(_make_request())

        assert call_count == 2
        assert result.structured == week_assessment(canonical_menu())

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
            "error_type": None,
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

    def test_retries_finish_reason_error_with_rate_limit_error_type(self) -> None:
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(_mock_error_response("rate_limit_exceeded")),
        ) as urlopen:
            with patch("time.sleep") as sleep:
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                with pytest.raises(RetryError) as exc_info:
                    client.generate(_make_request())

        assert urlopen.call_count == 3
        assert (
            exc_info.value.last_exception.details.reason
            == "finish_reason_error_rate_limit_exceeded"
        )
        # Rate limits get a much longer backoff than the default transient-error delay.
        assert [call.args[0] for call in sleep.call_args_list] == [15.0, 30.0]

    def test_retries_finish_reason_error_with_unspecified_error_type_at_default_delay(
        self,
    ) -> None:
        with patch(
            "urllib.request.urlopen", return_value=_mock_urlopen(_mock_partial_response("error"))
        ) as urlopen:
            with patch("time.sleep") as sleep:
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                with pytest.raises(RetryError):
                    client.generate(_make_request())

        assert urlopen.call_count == 3
        assert [call.args[0] for call in sleep.call_args_list] == [1.0, 2.0]

    def test_includes_upstream_error_message_in_failure_detail(self) -> None:
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(_mock_error_response("rate_limit_exceeded")),
        ):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=1)
                with pytest.raises(RetryError) as exc_info:
                    client.generate(_make_request())

        assert "rate limited" in str(exc_info.value)

    @pytest.mark.parametrize(
        "error_type",
        [
            "authentication",
            "payment_required",
            "permission_denied",
            "context_length_exceeded",
            "content_policy_violation",
        ],
    )
    def test_rejects_finish_reason_error_with_permanent_error_type_immediately(
        self, error_type: str
    ) -> None:
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(_mock_error_response(error_type)),
        ) as urlopen:
            client = OpenRouterClient(api_key="test-key", max_retries=3)
            with pytest.raises(OpenRouterResponseError) as exc_info:
                client.generate(_make_request())

        assert urlopen.call_count == 1
        assert exc_info.value.details.reason == f"finish_reason_error_{error_type}"

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

    def test_retries_genuine_http_429_with_slow_clearing_delay(self) -> None:
        http_error = urllib.error.HTTPError(
            url="https://openrouter.ai", code=429, msg="Too Many Requests", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        http_error.response_body = json.dumps(
            {"error": {"code": 429, "message": "Rate limit exceeded"}}
        )

        with patch("meal_orchestrator.llm.openrouter.post_json", side_effect=http_error):
            with patch("time.sleep") as sleep:
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                with pytest.raises(RetryError):
                    client.generate(_make_request())

        # A genuine HTTP 429 must get the same slow-clearing backoff as the
        # embedded-error-in-a-200 shape, even though this response body carries no
        # error_type metadata — the HTTP status alone is enough to treat it as such.
        assert [call.args[0] for call in sleep.call_args_list] == [15.0, 30.0]

    def test_includes_truncated_upstream_error_message_in_failure_detail(self) -> None:
        long_message = "x" * 400

        def _mock_long_error_response() -> bytes:
            return json.dumps(
                {
                    "id": "gen-example",
                    "model": "openai/gpt-4o-mini",
                    "choices": [
                        {
                            "message": {"content": "Partial meal plan"},
                            "finish_reason": "error",
                            "error": {
                                "code": 429,
                                "message": long_message,
                                "metadata": {"error_type": "rate_limit_exceeded"},
                            },
                        }
                    ],
                }
            ).encode("utf-8")

        with patch(
            "urllib.request.urlopen", return_value=_mock_urlopen(_mock_long_error_response())
        ):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=1)
                with pytest.raises(RetryError) as exc_info:
                    client.generate(_make_request())

        message = str(exc_info.value)
        assert long_message not in message
        assert "x" * 300 + "…" in message

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

    def test_retries_on_invalid_json_content_and_eventually_succeeds(self) -> None:
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_urlopen(_mock_response("not json"))
            return _mock_urlopen(_mock_response(_assessment_json()))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                result = client.generate(_make_request())

        assert call_count == 2
        assert result.structured == week_assessment(canonical_menu())

    def test_on_attempt_called_for_each_attempt_with_outcome(self) -> None:
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_urlopen(_mock_response(None))
            return _mock_urlopen(_mock_response(_assessment_json()))

        attempts: list[tuple] = []

        def on_attempt(attempt, feedback, response, outcome):
            attempts.append((attempt, feedback, response, outcome))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                result = client.generate(_make_request(), on_attempt=on_attempt)

        assert result.structured == week_assessment(canonical_menu())
        assert len(attempts) == 2
        first_attempt, first_feedback, first_response, first_outcome = attempts[0]
        assert first_attempt == 1
        assert first_feedback is None
        assert first_response is not None
        assert first_outcome["accepted"] is False
        assert first_outcome["reason"] == "empty_message_content"

        second_attempt, second_feedback, second_response, second_outcome = attempts[1]
        assert second_attempt == 2
        assert second_feedback is None
        assert second_response is not None
        assert second_outcome == {"accepted": True}

    def test_on_attempt_called_for_network_error_and_eventually_succeeds(self) -> None:
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise urllib.error.URLError("connection refused")
            return _mock_urlopen(_mock_response(_assessment_json()))

        attempts: list[tuple] = []

        def on_attempt(attempt, feedback, response, outcome):
            attempts.append((attempt, feedback, response, outcome))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                result = client.generate(_make_request(), on_attempt=on_attempt)

        assert result.structured == week_assessment(canonical_menu())
        assert len(attempts) == 2
        first_attempt, first_feedback, first_response, first_outcome = attempts[0]
        assert first_attempt == 1
        assert first_feedback is None
        assert first_response is None
        assert first_outcome["accepted"] is False
        assert first_outcome["reason"] == "network_error"
        assert "connection refused" in first_outcome["error"]

    def test_raises_retry_error_when_content_never_matches_schema(self) -> None:
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(_mock_response(json.dumps({"days": "not-a-list"}))),
        ):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=2)
                with pytest.raises(RetryError) as exc_info:
                    client.generate(_make_request())

        assert isinstance(exc_info.value.last_exception, StructuredOutputError)
        assert exc_info.value.last_exception.details.reason == "invalid_structured_output"

    def test_feeds_back_schema_error_into_next_attempt(self) -> None:
        bodies = []

        def side_effect(req, timeout=None):
            bodies.append(json.loads(req.data.decode("utf-8")))
            if len(bodies) == 1:
                return _mock_urlopen(_mock_response("not json"))
            return _mock_urlopen(_mock_response(_assessment_json()))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                client.generate(_make_request())

        second_attempt_texts = [block["text"] for block in bodies[1]["messages"][0]["content"]]
        assert any("did not match the required JSON schema" in t for t in second_attempt_texts)

    def test_raises_retry_error_when_assessment_is_incomplete(self) -> None:
        incomplete_menu = canonical_menu()
        incomplete_assessment = week_assessment(incomplete_menu)
        # Drop the only day, leaving no meals assessed at all.
        incomplete_assessment = incomplete_assessment.model_copy(update={"days": []})

        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(
                _mock_response(incomplete_assessment.model_dump_json())
            ),
        ):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=2)
                with pytest.raises(RetryError) as exc_info:
                    client.generate(_make_request())

        assert isinstance(exc_info.value.last_exception, IncompleteAssessmentError)
        assert exc_info.value.last_exception.details.reason == "incomplete_assessment"
        assert exc_info.value.last_exception.problems

    def test_retries_on_url_error_and_eventually_succeeds(self) -> None:
        """Plain network errors (not HTTPError) must still be retried.

        A prior hand-rolled retry loop only caught urllib.error.HTTPError around
        the request, silently dropping retry coverage for connection-level
        failures — regression-test that URLError goes through the same
        transient-retry path as a 5xx.
        """
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise urllib.error.URLError("connection reset")
            return _mock_urlopen(_mock_response(_assessment_json()))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                result = client.generate(_make_request())

        assert call_count == 2
        assert result.structured == week_assessment(canonical_menu())

    def test_retries_on_timeout_error_and_eventually_succeeds(self) -> None:
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timed out")
            return _mock_urlopen(_mock_response(_assessment_json()))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                result = client.generate(_make_request())

        assert call_count == 2
        assert result.structured == week_assessment(canonical_menu())

    def test_raises_value_error_for_non_positive_max_retries(self) -> None:
        client = OpenRouterClient(api_key="test-key", max_retries=0)

        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            client.generate(_make_request())

    def test_preserves_earlier_feedback_when_a_later_error_has_no_specific_feedback(
        self,
    ) -> None:
        """Corrective feedback from an earlier attempt must not be discarded.

        Attempt 1 fails with a schema error (which has specific feedback text).
        Attempt 2 fails with an empty response (which _feedback_for doesn't
        recognize, so it returns None) — that must not erase attempt 1's
        feedback before attempt 3 is sent.
        """
        bodies = []

        def side_effect(req, timeout=None):
            bodies.append(json.loads(req.data.decode("utf-8")))
            if len(bodies) == 1:
                return _mock_urlopen(_mock_response("not json"))
            if len(bodies) == 2:
                return _mock_urlopen(_mock_response(None))
            return _mock_urlopen(_mock_response(_assessment_json()))

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):
                client = OpenRouterClient(api_key="test-key", max_retries=3)
                client.generate(_make_request())

        assert len(bodies) == 3
        third_attempt_texts = [block["text"] for block in bodies[2]["messages"][0]["content"]]
        assert any("did not match the required JSON schema" in t for t in third_attempt_texts)
