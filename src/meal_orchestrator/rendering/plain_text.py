from __future__ import annotations

from meal_orchestrator.domain import CanonicalMeal, CanonicalMenu, MealAssessment, WeekAssessment
from meal_orchestrator.rendering.join import iter_days, iter_meals, sorted_variants_by_score
from meal_orchestrator.rendering.labels import DAY_EMOJI, meal_label, weekday_name


def render_plain_text(assessment: WeekAssessment, menu: CanonicalMenu, run_id: str) -> str:
    """Render a WeekAssessment as plain text (channel-agnostic — usable for email today)."""
    lines: list[str] = []
    for canonical_day, day in iter_days(assessment, menu):
        lines.append(f"{DAY_EMOJI} {weekday_name(canonical_day.date)}")
        lines.append("")
        for canonical_meal, meal in iter_meals(canonical_day, day):
            emoji, label = meal_label(canonical_meal.type)
            lines.append(f"{emoji} {label}")
            lines.append("")
            lines.extend(_render_meal(meal, canonical_meal))
    lines.append(f"Run ID: {run_id}")
    return "\n".join(lines).rstrip("\n") + "\n"


def _render_meal(meal: MealAssessment, canonical_meal: CanonicalMeal) -> list[str]:
    sorted_variants, max_score = sorted_variants_by_score(meal)
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
