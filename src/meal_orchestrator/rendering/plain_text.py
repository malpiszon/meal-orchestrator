from __future__ import annotations

from meal_orchestrator.domain import CanonicalMeal, CanonicalMenu, MealAssessment, WeekAssessment
from meal_orchestrator.rendering.labels import DAY_EMOJI, meal_label, weekday_name


def render_plain_text(assessment: WeekAssessment, menu: CanonicalMenu) -> str:
    """Render a WeekAssessment as plain text (channel-agnostic — usable for email today).

    Day/meal order and identity are driven by `menu`, the authoritative source, not
    by the order of arrays in `assessment`: structured-output array order isn't a
    guarantee worth depending on. Sorting variants by score and marking ties for the
    top score are done here, deterministically, rather than trusted to the model.
    """
    assessed_days = {day.date: day for day in assessment.days}
    lines: list[str] = []
    for canonical_day in menu.days:
        day = assessed_days[canonical_day.date]
        assessed_meals = {meal.meal_type: meal for meal in day.meals}
        lines.append(f"{DAY_EMOJI} {weekday_name(canonical_day.date)}")
        lines.append("")
        for canonical_meal in canonical_day.meals:
            emoji, label = meal_label(canonical_meal.type)
            lines.append(f"{emoji} {label}")
            lines.append("")
            lines.extend(_render_meal(assessed_meals[canonical_meal.type], canonical_meal))
    return "\n".join(lines).rstrip("\n") + "\n"


def _render_meal(meal: MealAssessment, canonical_meal: CanonicalMeal) -> list[str]:
    sorted_variants = sorted(meal.variants, key=lambda variant: variant.score, reverse=True)
    max_score = sorted_variants[0].score
    lines: list[str] = []
    for variant in sorted_variants:
        canonical_variant = canonical_meal.variants[variant.variant_index]
        star = "⭐ " if variant.score == max_score else ""
        lines.append(f"{star}{variant.score}/10  {canonical_variant.name}")
        lines.extend(
            f"• {justification.icon} {justification.text}"
            for justification in variant.justifications
        )
        lines.append("")
    return lines
