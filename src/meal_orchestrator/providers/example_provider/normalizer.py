from __future__ import annotations

import re
from datetime import date
from typing import Any

from meal_orchestrator.domain import (
    CanonicalDay,
    CanonicalMeal,
    CanonicalMenu,
    MealVariant,
    Nutrition,
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
