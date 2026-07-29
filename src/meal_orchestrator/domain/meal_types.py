from __future__ import annotations

# The single source of truth for the six canonical meal-type keys used across the
# app (CanonicalMeal.type, PurchasedMeal.type, MealAssessment.meal_type, ...). Any
# module that needs to enumerate or validate meal types should derive from this
# rather than hardcoding its own copy of the list.
CANONICAL_MEAL_TYPES: tuple[str, ...] = (
    "breakfast",
    "second_breakfast",
    "lunch",
    "tea",
    "dinner",
    "snack",
)
