from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from meal_orchestrator.domain import PurchasedMeal
from meal_orchestrator.providers import MenuUnavailableError
from meal_orchestrator.providers.ntfy.normalizer import normalize_ntfy_week

_FIXTURE_DIR = Path("tests/fixtures/ntfy")

_WEEK_START = date(2026, 6, 29)
_WEEK_END = date(2026, 7, 3)

# ntfy always offers exactly 3 dish options (one per diet plan) per meal type.
_VARIANTS_PER_MEAL = 3

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


def _filler_dishes(
    mt_id: int, size: str, count: int, id_start: int
) -> tuple[list[dict], list[dict]]:
    """Build `count` extra single-size dishes for meal type `mt_id`.

    Used to pad hand-crafted fixtures up to the 3-dishes-per-meal-type
    invariant real ntfy data always satisfies, without the test needing to
    care about the filler dishes themselves.
    """
    products = []
    results = []
    for i in range(count):
        pid = id_start + i
        products.append(
            {
                "id": pid,
                "name": f"Filler dish {pid}",
                "size_tag": {"value": size},
                "composition": "filler",
                "protein": 1.0,
            }
        )
        results.append(
            {
                "diet_variant_meal_type_id": mt_id,
                "simple_product_id": pid,
                "diet_variant_id": pid,
                "configurable_product_id": pid,
            }
        )
    return products, results


_FILLER_BREAKFAST_PRODUCTS, _FILLER_BREAKFAST_RESULTS = _filler_dishes(
    mt_id=1, size="M", count=2, id_start=901
)
_FILLER_LUNCH_PRODUCTS, _FILLER_LUNCH_RESULTS = _filler_dishes(
    mt_id=3, size="XL", count=2, id_start=911
)

_INCLUDES_FULL = {
    "diet_variant_meal_types": [_MEAL_TYPE_BREAKFAST, _MEAL_TYPE_LUNCH],
    "simple_products": [
        _PRODUCT_BREAKFAST_M,
        _PRODUCT_BREAKFAST_L,
        _PRODUCT_LUNCH_XL,
        *_FILLER_BREAKFAST_PRODUCTS,
        *_FILLER_LUNCH_PRODUCTS,
    ],
}

_RESULTS_FULL = [
    {
        "diet_variant_meal_type_id": 1,
        "simple_product_id": 10,
        "diet_variant_id": 1,
        "configurable_product_id": 100,
    },
    {
        "diet_variant_meal_type_id": 1,
        "simple_product_id": 11,
        "diet_variant_id": 2,
        "configurable_product_id": 100,
    },
    {
        "diet_variant_meal_type_id": 3,
        "simple_product_id": 20,
        "diet_variant_id": 1,
        "configurable_product_id": 200,
    },
    *_FILLER_BREAKFAST_RESULTS,
    *_FILLER_LUNCH_RESULTS,
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
        meal_type_specs = [
            (_MEAL_TYPE_BREAKFAST, _PRODUCT_BREAKFAST_M, "M"),
            (_MEAL_TYPE_SECOND_BREAKFAST, _PRODUCT_SECOND_BREAKFAST_S, "S"),
            (_MEAL_TYPE_LUNCH, _PRODUCT_LUNCH_XL, "XL"),
            (_MEAL_TYPE_TEA, _PRODUCT_TEA_S, "S"),
            (_MEAL_TYPE_DINNER, _PRODUCT_DINNER_M, "M"),
            (_MEAL_TYPE_SNACK, _PRODUCT_SNACK_S, "S"),
        ]

        products = [_PRODUCT_BREAKFAST_M, _PRODUCT_SECOND_BREAKFAST_S, _PRODUCT_LUNCH_XL,
                    _PRODUCT_TEA_S, _PRODUCT_DINNER_M, _PRODUCT_SNACK_S]
        results = [
            {
                "diet_variant_meal_type_id": mt["id"],
                "simple_product_id": p["id"],
                "diet_variant_id": 1,
                "configurable_product_id": p["id"],
            }
            for mt, p, _size in meal_type_specs
        ]
        for mt, _p, size in meal_type_specs:
            filler_products, filler_results = _filler_dishes(
                mt_id=mt["id"], size=size, count=2, id_start=1000 + mt["id"] * 10
            )
            products.extend(filler_products)
            results.extend(filler_results)

        includes = {
            "diet_variant_meal_types": [mt for mt, _p, _s in meal_type_specs],
            "simple_products": products,
        }

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
        assert len(variants) == _VARIANTS_PER_MEAL
        names = {v["name"] for v in variants}
        assert "Owsianka" in names

    def test_raises_on_unavailable_size(self) -> None:
        with pytest.raises(MenuUnavailableError, match="no size"):
            normalize_ntfy_week(
                raw_days=[_make_raw_day()],
                provider_id="ntfy",
                week_start=_WEEK_START,
                week_end=_WEEK_END,
                user_id="alan",
                purchased_meals=[PurchasedMeal(type="breakfast", size="XXL")],
            )

    def test_mismatched_name_variant_merged_via_configurable_product_id(self) -> None:
        # Simulates the provider data-quality issue where the same dish appears with
        # a differently spelled (or even different-language) name for one size. Since
        # both rows share the same configurable_product_id, they must be merged into a
        # single group so that all size variants are considered together and the
        # correct size is found without duplication or failure.
        product_xl = {**_PRODUCT_LUNCH_XL}  # name "Kurczak z ryżem", size XL
        product_l_mismatched = {
            **_PRODUCT_LUNCH_XL, "id": 21, "name": "Kurczak z ryzem", "size_tag": {"value": "L"}
        }
        filler_products, filler_results = _filler_dishes(
            mt_id=3, size="XL", count=2, id_start=921
        )
        includes = {
            "diet_variant_meal_types": [_MEAL_TYPE_LUNCH],
            "simple_products": [product_xl, product_l_mismatched, *filler_products],
        }
        results = [
            {
                "diet_variant_meal_type_id": 3,
                "simple_product_id": 20,
                "diet_variant_id": 1,
                "configurable_product_id": 200,
            },
            {
                "diet_variant_meal_type_id": 3,
                "simple_product_id": 21,
                "diet_variant_id": 2,
                "configurable_product_id": 200,
            },
            *filler_results,
        ]

        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day(results=results, includes=includes)],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="lunch", size="XL")],
        )

        variants = menu.to_compact_dict()["days"][0]["meals"][0]["variants"]
        assert len(variants) == _VARIANTS_PER_MEAL
        names = {v["name"] for v in variants}
        assert "Kurczak z ryżem" in names


class TestNormalizeVariants:
    def test_all_dish_variants_for_meal_type_included(self) -> None:
        product_b = {**_PRODUCT_BREAKFAST_M, "id": 30, "name": "Granola"}
        filler_products, filler_results = _filler_dishes(
            mt_id=1, size="M", count=1, id_start=931
        )
        results = [
            {
                "diet_variant_meal_type_id": 1,
                "simple_product_id": 10,
                "diet_variant_id": 1,
                "configurable_product_id": 100,
            },
            {
                "diet_variant_meal_type_id": 1,
                "simple_product_id": 30,
                "diet_variant_id": 2,
                "configurable_product_id": 300,
            },
            *filler_results,
        ]
        includes = {
            "diet_variant_meal_types": [_MEAL_TYPE_BREAKFAST],
            "simple_products": [_PRODUCT_BREAKFAST_M, product_b, *filler_products],
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
        assert {"Owsianka", "Granola"} <= names

    def test_nutrition_fields_mapped_correctly(self) -> None:
        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day()],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
        )

        variants = menu.to_compact_dict()["days"][0]["meals"][0]["variants"]
        nutrition = next(v for v in variants if v["name"] == "Owsianka")["nutrition"]
        assert nutrition["protein_g"] == 10.0
        assert nutrition["fat_g"] == 5.0
        assert nutrition["saturated_fat_g"] == 1.5
        assert nutrition["carbs_g"] == 40.0
        assert nutrition["sugar_g"] == 8.0
        assert nutrition["fiber_g"] == 3.0
        assert nutrition["salt_g"] == 0.1

    def test_composition_whitespace_normalized(self) -> None:
        product = {**_PRODUCT_BREAKFAST_M, "composition": "  płatki  owsiane,\tmleko  "}
        filler_products, filler_results = _filler_dishes(
            mt_id=1, size="M", count=2, id_start=941
        )
        includes = {
            "diet_variant_meal_types": [_MEAL_TYPE_BREAKFAST],
            "simple_products": [product, *filler_products],
        }
        results = [
            {
                "diet_variant_meal_type_id": 1,
                "simple_product_id": 10,
                "diet_variant_id": 1,
                "configurable_product_id": 100,
            },
            *filler_results,
        ]

        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day(results=results, includes=includes)],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
        )

        variants = menu.to_compact_dict()["days"][0]["meals"][0]["variants"]
        composition = next(v for v in variants if v["name"] == "Owsianka")["composition"]
        assert composition == "płatki owsiane, mleko"

    def test_missing_nutrition_fields_omitted(self) -> None:
        product = {
            "id": 10,
            "name": "Owsianka",
            "size_tag": {"value": "M"},
            "composition": "owsianka",
            "protein": 10.0,
        }
        filler_products, filler_results = _filler_dishes(
            mt_id=1, size="M", count=2, id_start=951
        )
        includes = {
            "diet_variant_meal_types": [_MEAL_TYPE_BREAKFAST],
            "simple_products": [product, *filler_products],
        }
        results = [
            {
                "diet_variant_meal_type_id": 1,
                "simple_product_id": 10,
                "diet_variant_id": 1,
                "configurable_product_id": 100,
            },
            *filler_results,
        ]

        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day(results=results, includes=includes)],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
        )

        variants = menu.to_compact_dict()["days"][0]["meals"][0]["variants"]
        nutrition = next(v for v in variants if v["name"] == "Owsianka")["nutrition"]
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
        results = [
            {
                "diet_variant_meal_type_id": 1,
                "simple_product_id": 10,
                "diet_variant_id": 1,
                "configurable_product_id": 100,
            }
        ]

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
        results = [
            {
                "diet_variant_meal_type_id": 99,
                "simple_product_id": 10,
                "diet_variant_id": 1,
                "configurable_product_id": 100,
            }
        ]

        menu = normalize_ntfy_week(
            raw_days=[_make_raw_day(results=results, includes=includes)],
            provider_id="ntfy",
            week_start=_WEEK_START,
            week_end=_WEEK_END,
            user_id="alan",
            purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
        )

        assert menu.to_compact_dict()["days"] == []

    def test_raises_when_dish_count_not_exactly_three(self) -> None:
        # ntfy contractually offers exactly 3 dish options per meal type. A
        # day with only 1 dish (e.g. due to a dropped/misgrouped row) must
        # fail loudly rather than silently deliver an incomplete menu.
        includes = {
            "diet_variant_meal_types": [_MEAL_TYPE_BREAKFAST],
            "simple_products": [_PRODUCT_BREAKFAST_M],
        }
        results = [
            {
                "diet_variant_meal_type_id": 1,
                "simple_product_id": 10,
                "diet_variant_id": 1,
                "configurable_product_id": 100,
            }
        ]

        with pytest.raises(ValueError, match="expected 3 dishes"):
            normalize_ntfy_week(
                raw_days=[_make_raw_day(results=results, includes=includes)],
                provider_id="ntfy",
                week_start=_WEEK_START,
                week_end=_WEEK_END,
                user_id="alan",
                purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
            )


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

    @pytest.mark.parametrize("size", ["S", "M", "L", "XL", "XXL"])
    def test_fixture_2026_07_16_mismatched_size_name_still_matches(self, size: str) -> None:
        # Regression test for a real prod incident: on this date, the ntfy API
        # returned one size (M) of a lunch dish under a different-language
        # name than its other sizes (S/L/XL/XXL), even though all five rows
        # share the same configurable_product_id — i.e. they are the same
        # logical dish. Grouping by accent-folded name (as opposed to
        # configurable_product_id) splits this into two groups — one with
        # only M — so requesting any size other than M incorrectly raised
        # MenuUnavailableError even though that size was clearly present in
        # the payload for the same dish.
        def _load(fixture_date: str) -> dict:
            data = json.loads(
                (_FIXTURE_DIR / f"raw_offer8_{fixture_date}.json").read_bytes()
            )
            root = data.get("data", data)
            return {
                "date": fixture_date,
                "offer_id": 8,
                "results": root.get("results", []),
                "includes": root.get("includes", {}),
            }

        menu = normalize_ntfy_week(
            raw_days=[_load("2026-07-16")],
            provider_id="ntfy",
            week_start=date(2026, 7, 13),
            week_end=date(2026, 7, 17),
            user_id="example",
            purchased_meals=[PurchasedMeal(type="lunch", size=size)],
        )

        lunch = next(
            m for m in menu.to_compact_dict()["days"][0]["meals"] if m["type"] == "lunch"
        )
        assert len(lunch["variants"]) == 3

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
