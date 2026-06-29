from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from meal_orchestrator.domain import (
    CanonicalDay,
    CanonicalMeal,
    CanonicalMenu,
    MealVariant,
    Nutrition,
    ProviderMenuRequest,
    ProviderResult,
    PurchasedMeal,
)

NUTRITION_FIELDS = {
    "protein_g",
    "fat_g",
    "saturated_fat_g",
    "carbs_g",
    "sugar_g",
    "fiber_g",
    "salt_g",
}


class ExampleProviderAdapter:
    provider_id = "example_provider"

    def get_canonical_week_menu(
        self,
        request: ProviderMenuRequest,
    ) -> ProviderResult:
        raw_menu = _placeholder_raw_menu(
            request.week_start,
            request.week_end,
            request.provider_offering_id,
        )
        menu = normalize_example_provider_menu(
            raw_menu=raw_menu,
            provider_id=self.provider_id,
            week_start=request.week_start,
            week_end=request.week_end,
            user_id=request.user_id,
            purchased_meals=request.purchased_meals,
        )
        return ProviderResult(menu=menu, raw_response=raw_menu)


def normalize_example_provider_menu(
    raw_menu: dict[str, Any],
    provider_id: str,
    week_start: date,
    week_end: date,
    user_id: str,
    purchased_meals: list[PurchasedMeal],
) -> CanonicalMenu:
    raw_days = raw_menu.get("days")
    if not isinstance(raw_days, list):
        raise ValueError("example provider payload must contain a days list")

    requested = {(meal.type, meal.size) for meal in purchased_meals}
    days: list[CanonicalDay] = []

    for raw_day in raw_days:
        day_date = date.fromisoformat(_required(raw_day, "date"))
        if day_date < week_start or day_date > week_end:
            continue

        canonical_meals = _normalize_meals(raw_day, requested)
        if canonical_meals:
            days.append(CanonicalDay(date=day_date, meals=canonical_meals))

    return CanonicalMenu(
        provider=provider_id,
        week_start=week_start,
        week_end=week_end,
        user_id=user_id,
        days=days,
    )


def _normalize_meals(
    raw_day: dict[str, Any], requested: set[tuple[str, str]]
) -> list[CanonicalMeal]:
    raw_meals = raw_day.get("meals")
    if not isinstance(raw_meals, list):
        raise ValueError("example provider day payload must contain a meals list")

    meals: list[CanonicalMeal] = []
    for raw_meal in raw_meals:
        meal_type = _required(raw_meal, "type")
        raw_sizes = raw_meal.get("sizes")
        if not isinstance(raw_sizes, dict):
            raise ValueError("example provider meal payload must contain sizes")

        for requested_type, requested_size in requested:
            if meal_type != requested_type or requested_size not in raw_sizes:
                continue
            variants = [_normalize_variant(variant) for variant in raw_sizes[requested_size]]
            meals.append(CanonicalMeal(type=meal_type, variants=variants))

    return meals


def _normalize_variant(raw_variant: dict[str, Any]) -> MealVariant:
    nutrition = raw_variant.get("nutrition") or {}
    if not isinstance(nutrition, dict):
        raise ValueError("example provider variant nutrition must be a mapping")

    return MealVariant(
        name=_required(raw_variant, "name"),
        composition=_normalize_whitespace(_required(raw_variant, "composition")),
        nutrition=Nutrition(
            **{field: nutrition[field] for field in NUTRITION_FIELDS if field in nutrition}
        ),
    )


def _required(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None:
        raise ValueError(f"missing required provider field: {key}")
    return str(value)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _placeholder_raw_menu(week_start: date, week_end: date, offering_id: int | str) -> dict:
    return {
        "offering_id": offering_id,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "days": [_placeholder_day(week_start + timedelta(days=offset)) for offset in range(5)],
    }


def _placeholder_day(day: date) -> dict:
    return {
        "date": day.isoformat(),
        "meals": [
            {
                "type": "breakfast",
                "sizes": {
                    "M": [
                        {
                            "name": "Tortilla",
                            "composition": "Chicken, vegetables, tortilla, yogurt sauce",
                            "nutrition": {
                                "protein_g": 28,
                                "fat_g": 18,
                                "saturated_fat_g": 6,
                                "carbs_g": 62,
                                "sugar_g": 8,
                                "fiber_g": 9,
                                "salt_g": 2.1,
                            },
                        }
                    ]
                },
            },
            {
                "type": "lunch",
                "sizes": {
                    "XL": [
                        {
                            "name": "Rice bowl",
                            "composition": "Rice, turkey, broccoli, sesame sauce",
                            "nutrition": {
                                "protein_g": 42,
                                "fat_g": 20,
                                "carbs_g": 88,
                                "fiber_g": 11,
                                "salt_g": 2.8,
                            },
                        }
                    ]
                },
            },
        ],
    }
