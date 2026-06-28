from __future__ import annotations

import re
from collections import defaultdict
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

# Maps ntfy meal_name.key values to canonical meal type strings used in config.
_MEAL_TYPE_KEY_MAP: dict[str, str] = {
    "BREAKFAST": "breakfast",
    "SECOND-BREAKFAST": "second_breakfast",
    "LUNCH": "lunch",
    "TEA": "tea",
    "DINNER": "dinner",
    "SNACK": "snack",
}


def normalize_ntfy_week(
    raw_days: list[dict[str, Any]],
    provider_id: str,
    week_start: date,
    week_end: date,
    user_id: str,
    purchased_meals: list[PurchasedMeal],
) -> CanonicalMenu:
    """Transform a list of raw ntfy daily payloads into a CanonicalMenu.

    Each entry in raw_days is the output of NtfyClient.fetch_week_raw() for one
    date: {"date": ..., "offer_id": ..., "results": [...], "includes": {...}}.
    """
    days: list[CanonicalDay] = []

    for raw_day in raw_days:
        day_date = date.fromisoformat(raw_day["date"])
        if day_date < week_start or day_date > week_end:
            continue

        canonical_meals = _normalize_day(raw_day, purchased_meals)
        if canonical_meals:
            days.append(CanonicalDay(date=day_date, meals=canonical_meals))

    return CanonicalMenu(
        provider=provider_id,
        week_start=week_start,
        week_end=week_end,
        user_id=user_id,
        days=days,
    )


def _normalize_day(
    raw_day: dict[str, Any],
    purchased_meals: list[PurchasedMeal],
) -> list[CanonicalMeal]:
    includes = raw_day.get("includes") or {}
    results = raw_day.get("results") or []

    if not isinstance(results, list):
        raise ValueError("ntfy: day payload 'results' must be a list")

    meal_type_map = _build_meal_type_map(includes)
    product_map = _build_product_map(includes)

    # Group products by (canonical_type, dish_name).
    # Each entry holds all size variants of a dish for one meal slot.
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for row in results:
        mt_id = row.get("diet_variant_meal_type_id")
        prod_id = row.get("simple_product_id")
        if mt_id is None or prod_id is None:
            continue

        canonical_type = meal_type_map.get(mt_id)
        if canonical_type is None:
            continue

        product = product_map.get(prod_id)
        if product is None:
            raise ValueError(
                f"ntfy: simple_product_id={prod_id} referenced in results but not in includes"
            )

        name = product.get("name")
        if not name:
            raise ValueError(
                f"ntfy: simple_product_id={prod_id} has no name"
            )
        groups[canonical_type][name].append(product)

    # For each purchased meal, pick the matching size variant of every dish.
    canonical_meals: list[CanonicalMeal] = []
    for pm in purchased_meals:
        dishes = groups.get(pm.type)
        if not dishes:
            continue

        variants: list[MealVariant] = []
        for dish_name, size_variants in dishes.items():
            product = _pick_size(dish_name, size_variants, pm.size)
            variants.append(_to_meal_variant(product))

        if variants:
            canonical_meals.append(CanonicalMeal(type=pm.type, variants=variants))

    return canonical_meals


def _build_meal_type_map(includes: dict[str, Any]) -> dict[int, str]:
    """Return {meal_type_id: canonical_type} for known ntfy meal type keys."""
    result: dict[int, str] = {}
    for mt in includes.get("diet_variant_meal_types") or []:
        mt_id = mt.get("id")
        key = (mt.get("meal_name") or {}).get("key")
        if mt_id is None or key is None:
            continue
        canonical = _MEAL_TYPE_KEY_MAP.get(key)
        if canonical is not None:
            result[mt_id] = canonical
    return result


def _build_product_map(includes: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Return {product_id: product_dict}."""
    return {p["id"]: p for p in (includes.get("simple_products") or []) if "id" in p}


def _pick_size(
    dish_name: str,
    size_variants: list[dict[str, Any]],
    target_size: str,
) -> dict[str, Any]:
    """Return the variant matching target_size exactly.

    Raises ValueError if the purchased size is not available for this dish.
    """
    available: list[str] = []
    for variant in size_variants:
        size_tag = variant.get("size_tag")
        size = size_tag.get("value") if size_tag else None
        if size == target_size:
            return variant
        available.append(size or "?")
    raise ValueError(
        f"ntfy: dish {dish_name!r} has no size {target_size!r}; "
        f"available: {sorted(set(available))}"
    )


def _to_meal_variant(product: dict[str, Any]) -> MealVariant:
    name = product.get("name")
    if not name:
        raise ValueError(f"ntfy: product id={product.get('id')} has no name")

    composition = _normalize_whitespace(product.get("composition") or "")

    return MealVariant(
        name=name,
        composition=composition,
        nutrition=Nutrition(
            protein_g=_float_or_none(product, "protein"),
            fat_g=_float_or_none(product, "fat"),
            saturated_fat_g=_float_or_none(product, "saturated_fat"),
            carbs_g=_float_or_none(product, "carb"),
            sugar_g=_float_or_none(product, "sugar"),
            fiber_g=_float_or_none(product, "fiber"),
            salt_g=_float_or_none(product, "salt"),
        ),
    )


def _float_or_none(product: dict[str, Any], key: str) -> float | None:
    value = product.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"ntfy: expected numeric nutrition value, got "
            f"{type(value).__name__}: {value!r}"
        )
    return float(value)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
