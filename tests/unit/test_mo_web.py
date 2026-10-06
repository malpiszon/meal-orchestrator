from __future__ import annotations

import json
import urllib.error
from datetime import date
from unittest.mock import patch

import pytest

from meal_orchestrator.delivery.mo_web import MoWebHttpClient, build_mo_web_payload
from meal_orchestrator.domain import (
    CanonicalDay,
    CanonicalMeal,
    CanonicalMenu,
    DayAssessment,
    Justification,
    MealAssessment,
    MealVariant,
    Nutrition,
    VariantAssessment,
    WeekAssessment,
)
from meal_orchestrator.ops_notifications import mo_web_failure_description
from meal_orchestrator.retries import RetryError

_URL = "https://mo-web.example.com/api/mo/deliveries"
_TOKEN = "0123456789abcdef"


def _menu() -> CanonicalMenu:
    return CanonicalMenu(
        provider="ntfy",
        week_start=date(2026, 10, 5),
        week_end=date(2026, 10, 9),
        user_id="example",
        days=[
            CanonicalDay(
                date=date(2026, 10, 5),
                meals=[
                    CanonicalMeal(
                        type="breakfast",
                        variants=[
                            MealVariant(
                                name="Kofty",
                                composition="lamb, mint",
                                provider_meal_id="496",
                                nutrition=Nutrition(protein_g=23.8, salt_g=2.0),
                            ),
                            MealVariant(name="Owsianka", composition="", provider_meal_id="12"),
                            MealVariant(name="Omlet", composition="eggs", provider_meal_id="7"),
                        ],
                    )
                ],
            )
        ],
    )


def _variant(index: int, score: int, text: str) -> VariantAssessment:
    return VariantAssessment(
        variant_index=index,
        name=f"variant {index}",
        score=score,
        justifications=[Justification(icon="🥗", text=text)],
    )


def _assessment() -> WeekAssessment:
    # Deliberately out of menu order and with the best score last in the menu.
    return WeekAssessment(
        days=[
            DayAssessment(
                date=date(2026, 10, 5),
                meals=[
                    MealAssessment(
                        meal_type="breakfast",
                        variants=[
                            _variant(2, 9, "omlet"),
                            _variant(0, 6, "kofty"),
                            _variant(1, 3, "owsianka"),
                        ],
                    )
                ],
            )
        ]
    )


def test_payload_keeps_menu_order_and_joins_scores() -> None:
    payload = build_mo_web_payload(_menu(), _assessment(), "user@example.com", "run-1")

    assert payload == {
        "schema_version": 1,
        "run_id": "run-1",
        "provider": "ntfy",
        "week_start": "2026-10-05",
        "week_end": "2026-10-09",
        "user": {"email": "user@example.com"},
        "days": [
            {
                "date": "2026-10-05",
                "meals": [
                    {
                        "type": "breakfast",
                        "variants": [
                            {
                                "provider_meal_id": "496",
                                "name": "Kofty",
                                "composition": "lamb, mint",
                                "nutrition": {"protein_g": 23.8, "salt_g": 2.0},
                                "score": 6,
                                "justifications": [{"icon": "🥗", "text": "kofty"}],
                            },
                            {
                                "provider_meal_id": "12",
                                "name": "Owsianka",
                                "composition": "",
                                "score": 3,
                                "justifications": [{"icon": "🥗", "text": "owsianka"}],
                            },
                            {
                                "provider_meal_id": "7",
                                "name": "Omlet",
                                "composition": "eggs",
                                "score": 9,
                                "justifications": [{"icon": "🥗", "text": "omlet"}],
                            },
                        ],
                    }
                ],
            }
        ],
    }


def _http_error(code: int, body: dict | None = None) -> urllib.error.HTTPError:
    error = urllib.error.HTTPError(_URL, code, "error", {}, None)
    error.response_body = json.dumps(body) if body is not None else ""
    return error


class TestMoWebHttpClient:
    def test_dashboard_url_is_on_the_api_origin(self) -> None:
        client = MoWebHttpClient(url="https://mo.example.com/api/mo/deliveries", token=_TOKEN)

        assert client.dashboard_url == "https://mo.example.com/dashboard"

    def test_sends_payload_with_bearer_token(self) -> None:
        with patch(
            "meal_orchestrator.delivery.mo_web.post_json",
            return_value=b'{"plan_id": "p1", "week_start": "2026-10-05", "account_created": true}',
        ) as post:
            MoWebHttpClient(url=_URL, token=_TOKEN, timeout_seconds=7).send({"a": "ż"})

        args, kwargs = post.call_args
        assert args == (_URL,)
        assert kwargs["headers"]["Authorization"] == f"Bearer {_TOKEN}"
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert json.loads(kwargs["body"]) == {"a": "ż"}
        assert kwargs["timeout_seconds"] == 7

    def test_tolerates_non_json_success_body(self) -> None:
        with patch("meal_orchestrator.delivery.mo_web.post_json", return_value=b"OK"):
            MoWebHttpClient(url=_URL, token=_TOKEN).send({})

    def test_retries_transient_errors(self) -> None:
        with (
            patch(
                "meal_orchestrator.delivery.mo_web.post_json",
                side_effect=[_http_error(503, {"error": "not_configured"}), b"{}"],
            ) as post,
            patch("meal_orchestrator.retries.time.sleep"),
        ):
            MoWebHttpClient(url=_URL, token=_TOKEN).send({})

        assert post.call_count == 2

    def test_does_not_retry_contract_violation(self) -> None:
        with (
            patch(
                "meal_orchestrator.delivery.mo_web.post_json",
                side_effect=_http_error(400, {"error": "invalid_payload", "issues": []}),
            ) as post,
            pytest.raises(urllib.error.HTTPError),
        ):
            MoWebHttpClient(url=_URL, token=_TOKEN).send({})

        assert post.call_count == 1

    def test_does_not_retry_conflict(self) -> None:
        with (
            patch(
                "meal_orchestrator.delivery.mo_web.post_json",
                side_effect=_http_error(409, {"error": "conflict"}),
            ) as post,
            pytest.raises(urllib.error.HTTPError),
        ):
            MoWebHttpClient(url=_URL, token=_TOKEN).send({})

        assert post.call_count == 1


class TestMoWebFailureDescription:
    def test_includes_status_error_and_issues(self) -> None:
        error = _http_error(
            400, {"error": "invalid_payload", "issues": [{"path": ["week_start"], "message": "x"}]}
        )

        description = mo_web_failure_description(
            run_id="run-1", user_id="example", week_start=date(2026, 10, 5), error=error
        )

        assert "user example" in description
        assert "2026-10-05" in description
        assert "HTTP 400" in description
        assert "invalid_payload" in description
        assert '"week_start"' in description

    def test_unwraps_retry_error(self) -> None:
        error = RetryError("failed after 3 attempt(s)", _http_error(503, {"error": "x"}))

        description = mo_web_failure_description(
            run_id="run-1", user_id="example", week_start=date(2026, 10, 5), error=error
        )

        assert "HTTP 503" in description

    def test_network_error_uses_exception_text(self) -> None:
        description = mo_web_failure_description(
            run_id="run-1",
            user_id="example",
            week_start=date(2026, 10, 5),
            error=urllib.error.URLError("connection refused"),
        )

        assert "connection refused" in description

    def test_long_issue_list_is_truncated_under_discord_limit(self) -> None:
        issues = [{"path": ["days", i], "message": "x" * 1000} for i in range(50)]
        error = _http_error(400, {"error": "invalid_payload", "issues": issues})

        description = mo_web_failure_description(
            run_id="run-1", user_id="example", week_start=date(2026, 10, 5), error=error
        )

        assert len(description) <= 4096
        assert "… and 45 more" in description
