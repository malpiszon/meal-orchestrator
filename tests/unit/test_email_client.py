from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from meal_orchestrator.delivery.email import ResendEmailClient
from meal_orchestrator.domain import EmailMessage
from meal_orchestrator.retries import RetryError


def _make_message() -> EmailMessage:
    return EmailMessage(
        to="user@example.com",
        from_address="Meal Orchestrator <meals@example.com>",
        subject="Meal plan for 2026-06-02",
        body="Eat well this week.",
    )


def _mock_urlopen():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'{"id": "abc123"}'
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestResendEmailClientSend:
    def test_sends_without_error(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen()):
            client = ResendEmailClient(api_key="test-key")
            client.send(_make_message(), idempotency_key="run123:alan:email")

    def test_sends_bearer_auth_header(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["auth"] = req.get_header("Authorization")
            return _mock_urlopen()

        with patch("urllib.request.urlopen", side_effect=side_effect):
            ResendEmailClient(api_key="secret-key").send(
                _make_message(), idempotency_key="key1"
            )

        assert captured["auth"] == "Bearer secret-key"

    def test_sends_idempotency_key_header(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["idem"] = req.get_header("Idempotency-key")
            return _mock_urlopen()

        with patch("urllib.request.urlopen", side_effect=side_effect):
            ResendEmailClient(api_key="test-key").send(
                _make_message(), idempotency_key="run123:alan:email"
            )

        assert captured["idem"] == "run123:alan:email"

    def test_sends_correct_body_fields(self) -> None:
        captured = {}

        def side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _mock_urlopen()

        msg = _make_message()
        with patch("urllib.request.urlopen", side_effect=side_effect):
            ResendEmailClient(api_key="test-key").send(msg, idempotency_key="key1")

        body = captured["body"]
        assert body["from"] == msg.from_address
        assert body["to"] == [msg.to]
        assert body["subject"] == msg.subject
        assert body["text"] == msg.body

    def test_retries_on_429_and_succeeds(self) -> None:
        http_429 = urllib.error.HTTPError(
            url="https://api.resend.com", code=429, msg="Too Many Requests", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise http_429
            return _mock_urlopen()

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):
                ResendEmailClient(api_key="test-key", max_retries=3).send(
                    _make_message(), idempotency_key="key1"
                )

        assert call_count == 3

    def test_retries_on_500_and_succeeds(self) -> None:
        http_500 = urllib.error.HTTPError(
            url="https://api.resend.com", code=500, msg="Server Error", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise http_500
            return _mock_urlopen()

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):
                ResendEmailClient(api_key="test-key", max_retries=3).send(
                    _make_message(), idempotency_key="key1"
                )

        assert call_count == 2

    def test_raises_retry_error_after_exhausted_retries(self) -> None:
        http_500 = urllib.error.HTTPError(
            url="https://api.resend.com", code=500, msg="Server Error", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        with patch("urllib.request.urlopen", side_effect=http_500):
            with patch("time.sleep"):
                with pytest.raises(RetryError):
                    ResendEmailClient(api_key="test-key", max_retries=3).send(
                        _make_message(), idempotency_key="key1"
                    )

    def test_non_retryable_error_raised_immediately(self) -> None:
        http_403 = urllib.error.HTTPError(
            url="https://api.resend.com", code=403, msg="Forbidden", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise http_403

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                ResendEmailClient(api_key="test-key", max_retries=3).send(
                    _make_message(), idempotency_key="key1"
                )

        assert exc_info.value.code == 403
        assert call_count == 1

    def test_reads_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESEND_API_KEY", "env-key")
        captured = {}

        def side_effect(req, timeout=None):
            captured["auth"] = req.get_header("Authorization")
            return _mock_urlopen()

        with patch("urllib.request.urlopen", side_effect=side_effect):
            ResendEmailClient().send(_make_message(), idempotency_key="key1")

        assert captured["auth"] == "Bearer env-key"
