from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from meal_orchestrator.domain import PurchasedMeal
from meal_orchestrator.providers.ntfy.normalizer import normalize_ntfy_week

_FIXTURE_DIR = Path("tests/fixtures/ntfy")

_WEEK_START = date(2026, 6, 29)
_WEEK_END = date(2026, 7, 3)

# Minimal inline raw_day payloads used for targeted unit tests.
_MEAL_TYPE_BREAKFAST = {
    "id": 1, "diet_id": 1, "meal_name": {"key": "BREAKFAST", "value": "Śniadanie"}
}
_MEAL_TYPE_SECOND_BREAKFAST = {
    "id": 2, "diet_id": 1, "meal_name": {"key": "SECOND-BREAKFAST", "value": "Drugie śniadanie"}
}
_MEAL_TYPE_LUNCH = {"id": 3, "diet_id": 1, "meal_name": {"key": "LUNCH", "value": "Obiad"}}
_MEAL_TYPE_TEA = {"id": 4, "diet_id": 1, "meal_name": {"key": "TEA", "value": "Podwieczorek"}}
_MEAL_TYPE_DINNER = {"id": 5, "diet_id": 1, "meal_name": {"key": "DINNER", "value": "Kolacja"}}
_MEAL_TYPE_SNACK = {"id": 6, "diet_id": 1, "meal_name": {"key": "SNACK", "value": "Przekąska"}}

_PRODUCT_BREAKFAST_M = {
    "id": 10,
    "name": "Owsianka",
    "size_tag": {"value": "M"},
    "composition": "płatki owsiane, mleko",
    "protein": 10.0,
    "fat": 5.0,
    "saturated_fat": 1.5,
    "carb": 40.0,
    "sugar": 8.0,
    "fiber": 3.0,
    "salt": 0.1,
}
_PRODUCT_BREAKFAST_L = {**_PRODUCT_BREAKFAST_M, "id": 11, "size_tag": {"value": "L"}}

_PRODUCT_LUNCH_XL = {
    "id": 20,
    "name": "Kurczak z ryżem",
    "size_tag": {"value": "XL"},
    "composition": "kurczak, ryż, warzywa",
    "protein": 50.0,
    "fat": 20.0,
    "saturated_fat": 3.0,
    "carb": 60.0,
    "sugar": 5.0,
    "fiber": 4.0,
    "salt": 2.0,
}
_PRODUCT_SECOND_BREAKFAST_S = {
    **_PRODUCT_BREAKFAST_M, "id": 30, "name": "Jabłko", "size_tag": {"value": "S"}
}
_PRODUCT_TEA_S = {
    **_PRODUCT_BREAKFAST_M, "id": 40, "name": "Herbata z mlekiem", "size_tag": {"value": "S"}
}
_PRODUCT_DINNER_M = {
    **_PRODUCT_BREAKFAST_M, "id": 50, "name": "Sałatka wieczorna", "size_tag": {"value": "M"}
}
_PRODUCT_SNACK_S = {
    **_PRODUCT_BREAKFAST_M, "id": 60, "name": "Orzechy", "size_tag": {"value": "S"}
}

_INCLUDES_FULL = {
    "diet_variant_meal_types": [_MEAL_TYPE_BREAKFAST, _MEAL_TYPE_LUNCH],
    "simple_products": [_PRODUCT_BREAKFAST_M, _PRODUCT_BREAKFAST_L, _PRODUCT_LUNCH_XL],
}

_RESULTS_FULL = [
    {"diet_variant_meal_type_id": 1, "simple_product_id": 10, "diet_variant_id": 1},
    {"diet_variant_meal_type_id": 1, "simple_product_id": 11, "diet_variant_id": 2},
    {"diet_variant_meal_type_id": 3, "simple_product_id": 20, "diet_variant_id": 1},
]


def _make_raw_day(
    day_date: str = "2026-06-29",
    results: list | None = None,
    includes: dict | None = None,
) -> dict:
    return {
        "date": day_date,
        "offer_id": 6,
        "results": results if results is not None else _RESULTS_FULL,
        "includes": includes if includes is not None else _INCLUDES_FULL,
    }


def _load_fixture_raw_day(fixture_date: str) -> dict:
    data = json.loads((_FIXTURE_DIR / f"raw_offer6_{fixture_date}.json").read_bytes())
    root = data.get("data", data)
    return {
        "date": fixture_date,
        "offer_id": 6,
        "results": root.get("results", []),
        "includes": root.get("includes", {}),
    }


class TestNormalizeMealTypeFiltering:
    def test_filters_to_purchased_meal_types(self) -> None:
        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day()],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
        )

        payload = menu.to_compact_dict()
        assert len(payload["days"]) == 1
        meals = payload["days"][0]["meals"]
        assert len(meals) == 1
        assert meals[0]["type"] == "breakfast"

    def test_includes_all_purchased_meal_types(self) -> None:
        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day()],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[
                PurchasedMeal(type="breakfast", size="M"),
                PurchasedMeal(type="lunch", size="XL"),
            ],
        )

        types = {m["type"] for m in menu.to_compact_dict()["days"][0]["meals"]}
        assert types == {"breakfast", "lunch"}

    def test_all_six_ntfy_meal_types_map_to_canonical_names(self) -> None:
        includes = {
            "diet_variant_meal_types": [
                _MEAL_TYPE_BREAKFAST,
                _MEAL_TYPE_SECOND_BREAKFAST,
                _MEAL_TYPE_LUNCH,
                _MEAL_TYPE_TEA,
                _MEAL_TYPE_DINNER,
                _MEAL_TYPE_SNACK,
            ],
            "simple_products": [
                _PRODUCT_BREAKFAST_M,
                _PRODUCT_SECOND_BREAKFAST_S,
                _PRODUCT_LUNCH_XL,
                _PRODUCT_TEA_S,
                _PRODUCT_DINNER_M,
                _PRODUCT_SNACK_S,
            ],
        }
        results = [
            {
                "diet_variant_meal_type_id": mt["id"],
                "simple_product_id": p["id"],
                "diet_variant_id": 1,
            }
            for mt, p in [
                (_MEAL_TYPE_BREAKFAST, _PRODUCT_BREAKFAST_M),
                (_MEAL_TYPE_SECOND_BREAKFAST, _PRODUCT_SECOND_BREAKFAST_S),
                (_MEAL_TYPE_LUNCH, _PRODUCT_LUNCH_XL),
                (_MEAL_TYPE_TEA, _PRODUCT_TEA_S),
                (_MEAL_TYPE_DINNER, _PRODUCT_DINNER_M),
                (_MEAL_TYPE_SNACK, _PRODUCT_SNACK_S),
            ]
        ]

        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day(results=results, includes=includes)],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[
                PurchasedMeal(type="breakfast", size="M"),
                PurchasedMeal(type="second_breakfast", size="S"),
                PurchasedMeal(type="lunch", size="XL"),
                PurchasedMeal(type="tea", size="S"),
                PurchasedMeal(type="dinner", size="M"),
                PurchasedMeal(type="snack", size="S"),
            ],
        )

        meal_types = {m["type"] for m in menu.to_compact_dict()["days"][0]["meals"]}
        assert meal_types == {"breakfast", "second_breakfast", "lunch", "tea", "dinner", "snack"}

    def test_unpurchased_meal_types_excluded(self) -> None:
        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day()],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="lunch", size="XL")],
        )

        types = {m["type"] for m in menu.to_compact_dict()["days"][0]["meals"]}
        assert "breakfast" not in types


class TestNormalizeSizeFiltering:
    def test_keeps_only_purchased_size(self) -> None:
        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day()],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
        )

        variants = menu.to_compact_dict()["days"][0]["meals"][0]["variants"]
        assert len(variants) == 1
        assert variants[0]["name"] == "Owsianka"

    def test_raises_on_unavailable_size(self) -> None:
        with pytest.raises(ValueError, match="no size"):
            normalize_ntfy_week(
                raw_days=[_make_raw_day()],
                provider_id="ntfy",
                week_start=_WEEK_START,
                week_end=_WEEK_END,
                user_id="alan",
                purchased_meals=[PurchasedMeal(type="breakfast", size="XXL")],
            )


class TestNormalizeVariants:
    def test_all_dish_variants_for_meal_type_included(self) -> None:
        results = [
            {"diet_variant_meal_type_id": 1, "simple_product_id": 10, "diet_variant_id": 1},
            {"diet_variant_meal_type_id": 1, "simple_product_id": 30, "diet_variant_id": 2},
        ]
        product_b = {**_PRODUCT_BREAKFAST_M, "id": 30, "name": "Granola"}
        includes = {
            "diet_variant_meal_types": [_MEAL_TYPE_BREAKFAST],
            "simple_products": [_PRODUCT_BREAKFAST_M, product_b],
        }

        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day(results=results, includes=includes)],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
        )

        variants = menu.to_compact_dict()["days"][0]["meals"][0]["variants"]
        names = {v["name"] for v in variants}
        assert names == {"Owsianka", "Granola"}

    def test_nutrition_fields_mapped_correctly(self) -> None:
        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day()],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
        )

        nutrition = menu.to_compact_dict()["days"][0]["meals"][0]["variants"][0]["nutrition"]
        assert nutrition["protein_g"] == 10.0
        assert nutrition["fat_g"] == 5.0
        assert nutrition["saturated_fat_g"] == 1.5
        assert nutrition["carbs_g"] == 40.0
        assert nutrition["sugar_g"] == 8.0
        assert nutrition["fiber_g"] == 3.0
        assert nutrition["salt_g"] == 0.1

    def test_composition_whitespace_normalized(self) -> None:
        product = {**_PRODUCT_BREAKFAST_M, "composition": "  płatki  owsiane,\tmleko  "}
        includes = {
            "diet_variant_meal_types": [_MEAL_TYPE_BREAKFAST],
            "simple_products": [product],
        }
        results = [{"diet_variant_meal_type_id": 1, "simple_product_id": 10, "diet_variant_id": 1}]

        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day(results=results, includes=includes)],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
        )

        composition = menu.to_compact_dict()["days"][0]["meals"][0]["variants"][0]["composition"]
        assert composition == "płatki owsiane, mleko"

    def test_missing_nutrition_fields_omitted(self) -> None:
        product = {
            "id": 10,
            "name": "Owsianka",
            "size_tag": {"value": "M"},
            "composition": "owsianka",
            "protein": 10.0,
        }
        includes = {
            "diet_variant_meal_types": [_MEAL_TYPE_BREAKFAST],
            "simple_products": [product],
        }
        results = [{"diet_variant_meal_type_id": 1, "simple_product_id": 10, "diet_variant_id": 1}]

        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day(results=results, includes=includes)],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
        )

        nutrition = menu.to_compact_dict()["days"][0]["meals"][0]["variants"][0]["nutrition"]
        assert "protein_g" in nutrition
        assert "fat_g" not in nutrition


class TestNormalizeDayFiltering:
    def test_days_outside_week_range_excluded(self) -> None:
        raw_days = [
            _make_raw_day("2026-06-28"),  # before week_start
            _make_raw_day("2026-06-29"),  # in range
        ]

        menu = normalize_ntfy_week(
            raw_days=raw_days,
            provider_id="ntfy",
            week_start=date(2026, 6, 29),
            week_end=date(2026, 7, 3),
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
        )

        dates = [d["date"] for d in menu.to_compact_dict()["days"]]
        assert dates == ["2026-06-29"]

    def test_empty_days_list_when_no_matching_meals(self) -> None:
        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day()],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="snack", size="S")],
        )

        assert menu.to_compact_dict()["days"] == []


class TestNormalizeErrorCases:
    def test_raises_on_missing_product_reference(self) -> None:
        includes = {
            "diet_variant_meal_types": [_MEAL_TYPE_BREAKFAST],
            "simple_products": [],  # product 10 missing
        }
        results = [{"diet_variant_meal_type_id": 1, "simple_product_id": 10, "diet_variant_id": 1}]

        with pytest.raises(ValueError, match="simple_product_id=10"):
            normalize_ntfy_week(
                raw_days=[_make_raw_day(results=results, includes=includes)],
                provider_id="ntfy",
                week_start=_WEEK_START,
                week_end=_WEEK_END,
                user_id="alan",
                purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
            )

    def test_unknown_meal_type_key_skipped(self) -> None:
        unknown_mt = {"id": 99, "diet_id": 1, "meal_name": {"key": "UNKNOWN", "value": "???"}}
        includes = {
            "diet_variant_meal_types": [unknown_mt],
            "simple_products": [_PRODUCT_BREAKFAST_M],
        }
        results = [{"diet_variant_meal_type_id": 99, "simple_product_id": 10, "diet_variant_id": 1}]

        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day(results=results, includes=includes)],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
        )

        assert menu.to_compact_dict()["days"] == []


class TestNormalizeCanonicalShape:
    def test_canonical_metadata_fields(self) -> None:
        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day()],
            provider_id="ntfy",
            week_start=date(2026, 6, 29),
            week_end=date(2026, 7, 3),
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
        )

        payload = menu.to_compact_dict()
        assert payload["provider"] == "ntfy"
        assert payload["week_start"] == "2026-06-29"
        assert payload["week_end"] == "2026-07-03"
        assert payload["user"] == {"id": "alan"}


class TestNormalizeWithRealFixtures:
    """Smoke tests against captured ntfy fixture payloads."""

    def test_fixture_2026_06_29_matches_canonical(self) -> None:
        raw_days = [_load_fixture_raw_day("2026-06-29")]
        canonical_fixture = json.loads(
            (_FIXTURE_DIR / "canonical_offer6_week_2026-06-29.json").read_bytes()
        )

        menu = normalize_ntfy_week(
            raw_days=raw_days,
            provider_id="ntfy",
            week_start=date(2026, 6, 29),
            week_end=date(2026, 7, 3),
            user_id="alan",
            purchased_meals=[
                PurchasedMeal(type="breakfast", size="M"),
                PurchasedMeal(type="lunch", size="XL"),
            ],
        )

        payload = menu.to_compact_dict()
        # Compare only the first day since only one raw fixture is loaded here.
        assert payload["days"][0] == canonical_fixture["days"][0]

    def test_fixture_full_week_matches_canonical(self) -> None:
        raw_days = [
            _load_fixture_raw_day("2026-06-29"),
            _load_fixture_raw_day("2026-06-30"),
            _load_fixture_raw_day("2026-07-01"),
            _load_fixture_raw_day("2026-07-02"),
            _load_fixture_raw_day("2026-07-03"),
        ]
        canonical_fixture = json.loads(
            (_FIXTURE_DIR / "canonical_offer6_week_2026-06-29.json").read_bytes()
        )

        menu = normalize_ntfy_week(
            raw_days=raw_days,
            provider_id="ntfy",
            week_start=date(2026, 6, 29),
            week_end=date(2026, 7, 3),
            user_id="alan",
            purchased_meals=[
                PurchasedMeal(type="breakfast", size="M"),
                PurchasedMeal(type="lunch", size="XL"),
            ],
        )

        assert menu.to_compact_dict() == canonical_fixture

    @pytest.mark.parametrize(
        "fixture_date",
        ["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02", "2026-07-03"],
    )
    def test_fixture_produces_two_meals_per_day(self, fixture_date: str) -> None:
        menu = normalize_ntfy_week(
            raw_days=[_load_fixture_raw_day(fixture_date)],
            provider_id="ntfy",
            week_start=date(2026, 6, 29),
            week_end=date(2026, 7, 3),
            user_id="alan",
            purchased_meals=[
                PurchasedMeal(type="breakfast", size="M"),
                PurchasedMeal(type="lunch", size="XL"),
            ],
        )

        day = menu.to_compact_dict()["days"][0]
        assert len(day["meals"]) == 2
        meal_types = {m["type"] for m in day["meals"]}
        assert meal_types == {"breakfast", "lunch"}

    @pytest.mark.parametrize(
        "fixture_date",
        ["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02", "2026-07-03"],
    )
    def test_fixture_each_meal_has_three_variants(self, fixture_date: str) -> None:
        menu = normalize_ntfy_week(
            raw_days=[_load_fixture_raw_day(fixture_date)],
            provider_id="ntfy",
            week_start=date(2026, 6, 29),
            week_end=date(2026, 7, 3),
            user_id="alan",
            purchased_meals=[
                PurchasedMeal(type="breakfast", size="M"),
                PurchasedMeal(type="lunch", size="XL"),
            ],
        )

        for meal in menu.to_compact_dict()["days"][0]["meals"]:
            assert len(meal["variants"]) == 3, f"{meal['type']} should have 3 variants"
