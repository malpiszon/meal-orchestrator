from __future__ import annotations

from datetime import date
from html import escape

from meal_orchestrator.domain import (
    CanonicalMeal,
    CanonicalMenu,
    MealAssessment,
    VariantAssessment,
    WeekAssessment,
)
from meal_orchestrator.rendering.join import iter_days, iter_meals, sorted_variants_by_score
from meal_orchestrator.rendering.labels import DAY_EMOJI, meal_label, weekday_name

_FONT_STACK = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"

# Styling lives in a single <style> block, applied via classes, rather than
# repeated inline on every element or wrapped in nested tables: for a full week
# (6 meal types x ~3 variants x 7 days) that repetition/nesting alone pushed
# messages past Gmail's ~102KB clip threshold, triggering "[Wiadomość
# skrócona] Pokaż całą wiadomość". Plain divs (no table layout) are safe here
# because this email only targets Gmail, not Outlook/legacy clients that need
# table-based layout for consistent rendering. Gmail (web and app) supports
# <style> + class selectors — it only strips !important and some
# pseudo-selectors, not class rules.
_STYLE = f"""\
body{{margin:0;padding:16px 8px;background:#f5f6f7;font-family:{_FONT_STACK}}}
.card{{max-width:600px;margin:0 auto;background:#fff;border-radius:8px;overflow:hidden}}
.day-hdr{{background:#2f6f4f;padding:14px 20px;font-size:18px;font-weight:bold;color:#fff}}
.day-body{{padding:4px 20px 20px}}
.meal{{margin-top:16px}}
.meal-hdr{{font-size:14px;font-weight:bold;color:#1f2933;padding-bottom:8px}}
.variant{{margin-bottom:10px;padding:10px 12px;border:1px solid #e2e6ea;border-radius:6px}}
.variant.top{{border-color:#2f6f4f;background:#f0f8f2}}
.badge{{display:inline-block;background:#8a97a3;color:#fff;font-size:12px;font-weight:bold;
border-radius:10px;padding:2px 8px}}
.variant.top .badge{{background:#2f6f4f}}
.variant-name{{display:inline-block;padding-left:8px;font-size:14px;font-weight:bold;
color:#1f2933}}
.justification{{margin-top:6px;font-size:13px;color:#5a6b7a}}
"""


def render_html(assessment: WeekAssessment, menu: CanonicalMenu) -> str:
    """Render a WeekAssessment as a Gmail-safe HTML email.

    Div-based layout with a <style> block of classes (see `_STYLE` for why not
    inline styles or tables), no images/webfonts. Day/meal order and identity
    are driven by `menu`, the authoritative source (see `rendering.join`).
    """
    day_sections: list[str] = []
    for canonical_day, day in iter_days(assessment, menu):
        meal_sections = [
            _render_meal(assessed_meal, canonical_meal)
            for canonical_meal, assessed_meal in iter_meals(canonical_day, day)
        ]
        day_sections.append(_render_day(canonical_day.date, meal_sections))

    body = "".join(day_sections)
    return f"""\
<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plan posiłków</title>
<style>
{_STYLE}
</style>
</head>
<body>
<div class="card">
{body}
</div>
</body>
</html>
"""


def _render_day(day_date: date, meal_sections: list[str]) -> str:
    meals_html = "".join(meal_sections)
    return f"""\
<div class="day-hdr">{DAY_EMOJI} {escape(weekday_name(day_date))}</div>
<div class="day-body">{meals_html}</div>
"""


def _render_meal(meal: MealAssessment, canonical_meal: CanonicalMeal) -> str:
    emoji, label = meal_label(canonical_meal.type)
    sorted_variants, max_score = sorted_variants_by_score(meal)
    variant_cards = "".join(
        _render_variant(variant, canonical_meal, is_top=variant.score == max_score)
        for variant in sorted_variants
    )
    return f"""\
<div class="meal">
<div class="meal-hdr">{emoji} {escape(label)}</div>
{variant_cards}
</div>
"""


def _render_variant(
    variant: VariantAssessment, canonical_meal: CanonicalMeal, *, is_top: bool
) -> str:
    canonical_variant = canonical_meal.variants[variant.variant_index]
    variant_class = "variant top" if is_top else "variant"
    star = "⭐ " if is_top else ""
    justification_items = "".join(
        f'<div class="justification">{escape(justification.icon)} '
        f"{escape(justification.text)}</div>"
        for justification in variant.justifications
    )
    return f"""\
<div class="{variant_class}">
<span class="badge">{star}{variant.score}/10</span>\
<span class="variant-name">{escape(canonical_variant.name)}</span>
{justification_items}
</div>
"""
