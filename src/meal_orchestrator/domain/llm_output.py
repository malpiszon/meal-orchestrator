from __future__ import annotations

import datetime
from typing import Any

from pydantic import BaseModel, Field

from meal_orchestrator.domain.models import CanonicalMenu


class Justification(BaseModel):
    icon: str = Field(
        description=(
            "A single emoji that visually represents this justification point "
            "(e.g. \U0001F4AA for high protein, ⚠️ for a nutritional concern). "
            "Choose an emoji that fits this specific point's content."
        )
    )
    text: str = Field(
        description=(
            "One short sentence stating the point. State only the single most salient "
            "factor, not an exhaustive list. Do not restate raw numeric values from the "
            "input (grams of protein, salt, fiber, etc.) unless that exact number is the "
            "key reason for the point. Do not make two points that say the same thing in "
            "different words."
        )
    )


class VariantAssessment(BaseModel):
    variant_index: int = Field(
        description=(
            "The zero-based index of this variant within this exact meal on this exact "
            "date, matching the position of the 'variants' array in the input menu for "
            "that meal. Never attribute a variant to a different meal or day than where "
            "it appears in the input."
        )
    )
    name: str = Field(description="The variant's name, copied from the input, for readability.")
    score: int = Field(
        ge=1,
        le=10,
        description=(
            "Overall quality score for this specific meal variant, from 1 (poor) to 10 "
            "(excellent)."
        ),
    )
    justifications: list[Justification] = Field(
        min_length=1,
        max_length=2,
        description="One or two short justification points supporting the score.",
    )


class MealAssessment(BaseModel):
    meal_type: str = Field(
        description=(
            "The meal type exactly as given in the input for this day (e.g. breakfast, "
            "second_breakfast, lunch, tea, dinner, snack)."
        )
    )
    variants: list[VariantAssessment] = Field(
        description=(
            "An assessment for every variant of this meal present in the input — never "
            "omit a variant and never select only a single winner."
        )
    )


class DayAssessment(BaseModel):
    date: datetime.date = Field(
        description="The calendar date of this day, exactly as given in the input."
    )
    meals: list[MealAssessment] = Field(
        description="An assessment for every meal present in the input for this date."
    )


class WeekAssessment(BaseModel):
    days: list[DayAssessment] = Field(
        description=(
            "An assessment for every day, every meal, and every meal variant present in "
            "the input menu. Do not select a single winning meal — score and justify "
            "every option so the reader can compare them."
        )
    )


def validate_completeness(assessment: WeekAssessment, menu: CanonicalMenu) -> list[str]:
    """Return a list of human-readable problems, empty if assessment fully covers menu.

    Only checks against meals/variants actually present in the menu — a user who
    purchased a subset of the six canonical meal types is expected to have that
    same subset assessed, nothing more.
    """
    problems: list[str] = []
    assessed_days = {day.date: day for day in assessment.days}
    for canonical_day in menu.days:
        day = assessed_days.get(canonical_day.date)
        if day is None:
            problems.append(f"missing assessment for date {canonical_day.date.isoformat()}")
            continue
        assessed_meals = {meal.meal_type: meal for meal in day.meals}
        for canonical_meal in canonical_day.meals:
            meal = assessed_meals.get(canonical_meal.type)
            if meal is None:
                problems.append(
                    f"missing assessment for meal '{canonical_meal.type}' "
                    f"on {canonical_day.date.isoformat()}"
                )
                continue
            expected_indices = set(range(len(canonical_meal.variants)))
            actual_indices = {variant.variant_index for variant in meal.variants}
            # Set equality alone would miss a duplicated index (e.g. [0, 1, 1] for a
            # 2-variant meal collapses to {0, 1}), so also require matching counts.
            has_duplicate = len(meal.variants) != len(canonical_meal.variants)
            if actual_indices != expected_indices or has_duplicate:
                problems.append(
                    f"variant index mismatch for meal '{canonical_meal.type}' "
                    f"on {canonical_day.date.isoformat()}: "
                    f"expected {sorted(expected_indices)}, got "
                    f"{sorted(variant.variant_index for variant in meal.variants)}"
                )
    return problems


def week_assessment_json_schema() -> dict[str, Any]:
    """JSON schema for WeekAssessment, tightened for OpenRouter/OpenAI strict mode.

    Strict structured-output enforcement generally requires every object to set
    additionalProperties: false and list every property as required.
    """
    return _make_strict(WeekAssessment.model_json_schema())


def _make_strict(schema: dict[str, Any]) -> dict[str, Any]:
    schema = dict(schema)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        schema["properties"] = {key: _make_strict(value) for key, value in properties.items()}
        schema["required"] = list(properties.keys())
        schema["additionalProperties"] = False
    items = schema.get("items")
    if isinstance(items, dict):
        schema["items"] = _make_strict(items)
    defs = schema.get("$defs")
    if isinstance(defs, dict):
        schema["$defs"] = {key: _make_strict(value) for key, value in defs.items()}
    return schema
