from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from meal_orchestrator.delivery.discord import DiscordWebhookClient
from meal_orchestrator.domain import DiscordMessage
from meal_orchestrator.retries import RetryError

_WEBHOOK_ENV = "DISCORD_TEST_WEBHOOK_URL"
_WEBHOOK_URL = "https://discord.com/api/webhooks/123/abc"


def _make_message() -> DiscordMessage:
    return DiscordMessage(
        webhook_env=_WEBHOOK_ENV,
        title="Meal plan ready",
        description="Hey <@123>, your meal plan for 2026-07-07–2026-07-11 is ready.",
        color=0x2ECC71,
    )


def _mock_urlopen():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b""
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestDiscordWebhookClientNotify:
    def test_sends_without_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_WEBHOOK_ENV, _WEBHOOK_URL)
        with patch("urllib.request.urlopen", return_value=_mock_urlopen()):
            DiscordWebhookClient().notify(_make_message())

    def test_resolves_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_WEBHOOK_ENV, _WEBHOOK_URL)
        captured = {}

        def side_effect(req, timeout=None):
            captured["url"] = req.full_url
            return _mock_urlopen()

        with patch("urllib.request.urlopen", side_effect=side_effect):
            DiscordWebhookClient().notify(_make_message())

        assert captured["url"] == _WEBHOOK_URL

    def test_sends_embed_in_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_WEBHOOK_ENV, _WEBHOOK_URL)
        captured = {}

        def side_effect(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _mock_urlopen()

        msg = _make_message()
        with patch("urllib.request.urlopen", side_effect=side_effect):
            DiscordWebhookClient().notify(msg)

        embed = captured["body"]["embeds"][0]
        assert embed["title"] == msg.title
        assert embed["description"] == msg.description
        assert embed["color"] == msg.color

    def test_raises_key_error_when_env_not_set(self) -> None:
        client = DiscordWebhookClient()
        msg = DiscordMessage(
            webhook_env="MISSING_DISCORD_ENV", title="T", description="D", color=0
        )
        with pytest.raises(KeyError):
            client.notify(msg)

    def test_retries_on_429_and_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_WEBHOOK_ENV, _WEBHOOK_URL)
        http_429 = urllib.error.HTTPError(
            url=_WEBHOOK_URL, code=429, msg="Too Many Requests", hdrs={}, fp=None  # type: ignore[arg-type]
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
                DiscordWebhookClient(max_retries=3).notify(_make_message())

        assert call_count == 3

    def test_raises_retry_error_after_exhausted_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_WEBHOOK_ENV, _WEBHOOK_URL)
        http_500 = urllib.error.HTTPError(
            url=_WEBHOOK_URL, code=500, msg="Server Error", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        with patch("urllib.request.urlopen", side_effect=http_500):
            with patch("time.sleep"):
                with pytest.raises(RetryError):
                    DiscordWebhookClient(max_retries=3).notify(_make_message())

    def test_non_retryable_error_raised_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_WEBHOOK_ENV, _WEBHOOK_URL)
        http_403 = urllib.error.HTTPError(
            url=_WEBHOOK_URL, code=403, msg="Forbidden", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise http_403

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                DiscordWebhookClient(max_retries=3).notify(_make_message())

        assert exc_info.value.code == 403
        assert call_count == 1
