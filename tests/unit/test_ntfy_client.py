from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from meal_orchestrator.providers import MenuUnavailableError
from meal_orchestrator.providers.ntfy.client import NtfyClient
from meal_orchestrator.retries import RetryError

# Minimal valid response that the client accepts.
_FIXTURE_DIR = Path("tests/fixtures/ntfy")

_VALID_RESPONSE = json.dumps(
    {
        "results": [{"diet_variant_meal_type_id": 1, "simple_product_id": 10}],
        "includes": {},
    }
).encode()

_EMPTY_RESPONSE = json.dumps({"results": [], "includes": {}}).encode()


def _mock_urlopen(response_body: bytes, status: int = 200):
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestNtfyClientFetchWeekRaw:
    def test_returns_one_entry_per_day(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_VALID_RESPONSE)):
            client = NtfyClient()
            result = client.fetch_week_raw(date(2026, 6, 29), date(2026, 6, 29), offer_id=6)

        assert len(result) == 1
        assert result[0]["date"] == "2026-06-29"
        assert result[0]["offer_id"] == 6

    def test_returns_five_entries_for_full_week(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_VALID_RESPONSE)):
            client = NtfyClient()
            result = client.fetch_week_raw(date(2026, 6, 29), date(2026, 7, 3), offer_id=6)

        assert len(result) == 5
        dates = [r["date"] for r in result]
        assert dates == ["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02", "2026-07-03"]

    def test_results_preserved_in_raw_output(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_VALID_RESPONSE)):
            client = NtfyClient()
            result = client.fetch_week_raw(date(2026, 6, 29), date(2026, 6, 29), offer_id=6)

        assert result[0]["results"] == [{"diet_variant_meal_type_id": 1, "simple_product_id": 10}]

    def test_raises_menu_unavailable_on_empty_results(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(_EMPTY_RESPONSE)):
            client = NtfyClient()
            with pytest.raises(MenuUnavailableError):
                client.fetch_week_raw(date(2026, 6, 29), date(2026, 6, 29), offer_id=6)

    def test_raises_menu_unavailable_on_404(self) -> None:
        http_error = urllib.error.HTTPError(
            url="https://example.com", code=404, msg="Not Found", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        with patch("urllib.request.urlopen", side_effect=http_error):
            client = NtfyClient()
            with pytest.raises(MenuUnavailableError):
                client.fetch_week_raw(date(2026, 6, 29), date(2026, 6, 29), offer_id=6)

    def test_retries_on_500_and_eventually_succeeds(self) -> None:
        http_500 = urllib.error.HTTPError(
            url="https://example.com", code=500, msg="Server Error", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise http_500
            return _mock_urlopen(_VALID_RESPONSE)

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):  # skip actual delays
                client = NtfyClient()
                result = client.fetch_week_raw(date(2026, 6, 29), date(2026, 6, 29), offer_id=6)

        assert call_count == 3
        assert len(result) == 1

    def test_raises_retry_error_after_exhausted_retries(self) -> None:
        http_500 = urllib.error.HTTPError(
            url="https://example.com", code=500, msg="Server Error", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        with patch("urllib.request.urlopen", side_effect=http_500):
            with patch("time.sleep"):
                client = NtfyClient()
                with pytest.raises(RetryError):
                    client.fetch_week_raw(date(2026, 6, 29), date(2026, 6, 29), offer_id=6)

    def test_non_retryable_http_error_raised_immediately(self) -> None:
        http_403 = urllib.error.HTTPError(
            url="https://example.com", code=403, msg="Forbidden", hdrs={}, fp=None  # type: ignore[arg-type]
        )
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise http_403

        with patch("urllib.request.urlopen", side_effect=side_effect):
            client = NtfyClient()
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                client.fetch_week_raw(date(2026, 6, 29), date(2026, 6, 29), offer_id=6)

        assert exc_info.value.code == 403
        assert call_count == 1  # no retries

    def test_raises_value_error_on_malformed_json(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(b"not json")):
            client = NtfyClient()
            with pytest.raises(ValueError, match="malformed JSON"):
                client.fetch_week_raw(date(2026, 6, 29), date(2026, 6, 29), offer_id=6)

    def test_unwraps_data_envelope(self) -> None:
        wrapped = json.dumps(
            {
                "data": {
                    "results": [{"diet_variant_meal_type_id": 5, "simple_product_id": 99}],
                    "includes": {},
                }
            }
        ).encode()
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(wrapped)):
            client = NtfyClient()
            result = client.fetch_week_raw(date(2026, 6, 29), date(2026, 6, 29), offer_id=6)

        assert result[0]["results"] == [{"diet_variant_meal_type_id": 5, "simple_product_id": 99}]


class TestNtfyClientWithFixtures:
    """Smoke tests using real captured fixture payloads."""

    @pytest.mark.parametrize(
        "fixture_date",
        ["2026-06-29", "2026-06-30"],
    )
    def test_fixture_response_is_accepted(self, fixture_date: str) -> None:
        fixture = (_FIXTURE_DIR / f"raw_offer6_{fixture_date}.json").read_bytes()
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(fixture)):
            client = NtfyClient()
            result = client.fetch_week_raw(
                date.fromisoformat(fixture_date),
                date.fromisoformat(fixture_date),
                offer_id=6,
            )

        assert len(result) == 1
        assert result[0]["date"] == fixture_date
        assert len(result[0]["results"]) == 106
