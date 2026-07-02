from __future__ import annotations

from datetime import date

import pytest

from meal_orchestrator.domain import ProviderMenuRequest, PurchasedMeal
from meal_orchestrator.providers import MenuUnavailableError, ProviderNormalizationError
from meal_orchestrator.providers.ntfy import NtfyProviderAdapter
from meal_orchestrator.providers.ntfy.client import NtfyClient

_WEEK_START = date(2026, 6, 29)
_WEEK_END = date(2026, 7, 3)

_MEAL_TYPE_BREAKFAST = {
    "id": 1, "diet_id": 1, "meal_name": {"key": "BREAKFAST", "value": "Śniadanie"}
}
_PRODUCT_BREAKFAST_M = {
    "id": 10,
    "name": "Owsianka",
    "size_tag": {"value": "M"},
    "composition": "płatki owsiane, mleko",
    "protein": 10.0,
}

_INCLUDES = {
    "diet_variant_meal_types": [_MEAL_TYPE_BREAKFAST],
    "simple_products": [_PRODUCT_BREAKFAST_M],
}
_RESULTS = [{"diet_variant_meal_type_id": 1, "simple_product_id": 10, "diet_variant_id": 1}]


def _make_raw_day(day_date: str) -> dict:
    return {"date": day_date, "offer_id": 6, "results": _RESULTS, "includes": _INCLUDES}


def _request(*, size: str) -> ProviderMenuRequest:
    return ProviderMenuRequest(
        week_start=_WEEK_START,
        week_end=_WEEK_END,
        provider_offering_id=6,
        user_id="alan",
        purchased_meals=[PurchasedMeal(type="breakfast", size=size)],
    )


def test_adapter_propagates_menu_unavailable_uncaught(monkeypatch) -> None:
    """A purchased size that isn't published yet must surface as MenuUnavailableError,

    not get wrapped into ProviderNormalizationError, so the workflow treats it as
    expected menu unavailability rather than a hard failure.
    """
    monkeypatch.setattr(
        NtfyClient,
        "fetch_week_raw",
        lambda self, week_start, week_end, offer_id: [_make_raw_day("2026-06-29")],
    )

    with pytest.raises(MenuUnavailableError):
        NtfyProviderAdapter().get_canonical_week_menu(_request(size="XXL"))


def test_adapter_wraps_malformed_data_as_normalization_error(monkeypatch) -> None:
    """Genuinely malformed provider data (unresolvable product reference) should

    still fail fast as ProviderNormalizationError, distinct from menu unavailability.
    """
    broken_results = [{"diet_variant_meal_type_id": 1, "simple_product_id": 999}]
    monkeypatch.setattr(
        NtfyClient,
        "fetch_week_raw",
        lambda self, week_start, week_end, offer_id: [
            {"date": "2026-06-29", "offer_id": 6, "results": broken_results, "includes": _INCLUDES}
        ],
    )

    with pytest.raises(ProviderNormalizationError):
        NtfyProviderAdapter().get_canonical_week_menu(_request(size="M"))
