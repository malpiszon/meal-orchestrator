from __future__ import annotations

from collections.abc import Iterator

from meal_orchestrator.domain import (
    CanonicalDay,
    CanonicalMeal,
    CanonicalMenu,
    DayAssessment,
    MealAssessment,
    VariantAssessment,
    WeekAssessment,
)


def iter_days(
    assessment: WeekAssessment, menu: CanonicalMenu
) -> Iterator[tuple[CanonicalDay, DayAssessment]]:
    """Pair each canonical day with its assessment, in `menu`'s authoritative order.

    Day/meal order and identity are driven by `menu`, not by the order of
    arrays in `assessment`: structured-output array order isn't a guarantee
    worth depending on.
    """
    assessed_days = {day.date: day for day in assessment.days}
    for canonical_day in menu.days:
        yield canonical_day, assessed_days[canonical_day.date]


def iter_meals(
    canonical_day: CanonicalDay, day: DayAssessment
) -> Iterator[tuple[CanonicalMeal, MealAssessment]]:
    """Pair each canonical meal with its assessment, in `canonical_day`'s order."""
    assessed_meals = {meal.meal_type: meal for meal in day.meals}
    for canonical_meal in canonical_day.meals:
        yield canonical_meal, assessed_meals[canonical_meal.type]


def sorted_variants_by_score(meal: MealAssessment) -> tuple[list[VariantAssessment], int]:
    """Return this meal's variants sorted by score descending, plus the top score.

    Sorting and identifying the top score(s) are done here, deterministically,
    rather than trusted to the model.
    """
    sorted_variants = sorted(meal.variants, key=lambda variant: variant.score, reverse=True)
    return sorted_variants, sorted_variants[0].score
