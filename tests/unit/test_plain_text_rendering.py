from __future__ import annotations

from datetime import date

from meal_orchestrator.domain import (
    CanonicalDay,
    CanonicalMeal,
    CanonicalMenu,
    DayAssessment,
    Justification,
    MealAssessment,
    MealVariant,
    VariantAssessment,
    WeekAssessment,
)
from meal_orchestrator.rendering.labels import meal_label, weekday_name
from meal_orchestrator.rendering.plain_text import render_plain_text


def _menu_with_variants(*names: str) -> CanonicalMenu:
    return CanonicalMenu(
        provider="example_provider",
        week_start=date(2026, 6, 1),
        week_end=date(2026, 6, 1),
        user_id="alan",
        days=[
            CanonicalDay(
                date=date(2026, 6, 1),
                meals=[
                    CanonicalMeal(
                        type="breakfast",
                        variants=[
                            MealVariant(name=name, composition="Ingredients") for name in names
                        ],
                    )
                ],
            )
        ],
    )


def _assessment(menu: CanonicalMenu, scores: list[int]) -> WeekAssessment:
    variants = [
        VariantAssessment(
            variant_index=index,
            name=variant.name,
            score=score,
            justifications=[Justification(icon="💪", text="Reason.")],
        )
        for index, (variant, score) in enumerate(
            zip(menu.days[0].meals[0].variants, scores, strict=True)
        )
    ]
    return WeekAssessment(
        days=[
            DayAssessment(
                date=menu.days[0].date,
                meals=[MealAssessment(meal_type="breakfast", variants=variants)],
            )
        ]
    )


def test_sorts_variants_by_score_descending() -> None:
    menu = _menu_with_variants("Low", "High", "Mid")
    assessment = _assessment(menu, [4, 9, 6])

    rendered = render_plain_text(assessment, menu)

    assert rendered.index("High") < rendered.index("Mid") < rendered.index("Low")


def test_marks_all_variants_tied_for_top_score_with_star() -> None:
    menu = _menu_with_variants("A", "B", "C")
    assessment = _assessment(menu, [8, 8, 5])

    rendered = render_plain_text(assessment, menu)

    assert rendered.count("⭐") == 2
    a_line = next(line for line in rendered.splitlines() if line.endswith("A"))
    b_line = next(line for line in rendered.splitlines() if line.endswith("B"))
    c_line = next(line for line in rendered.splitlines() if line.endswith("C"))
    assert a_line.startswith("⭐")
    assert b_line.startswith("⭐")
    assert not c_line.startswith("⭐")


def test_includes_day_and_meal_headers() -> None:
    menu = _menu_with_variants("Only")
    assessment = _assessment(menu, [8])

    rendered = render_plain_text(assessment, menu)

    emoji, label = meal_label("breakfast")
    assert f"{weekday_name(date(2026, 6, 1))}" in rendered
    assert f"{emoji} {label}" in rendered


def test_renders_bullets_with_icon_and_text() -> None:
    menu = _menu_with_variants("Only")
    assessment = _assessment(menu, [8])

    rendered = render_plain_text(assessment, menu)

    assert "• 💪 Reason." in rendered


def test_uses_canonical_menu_name_not_model_echoed_name() -> None:
    menu = _menu_with_variants("Canonical Name")
    assessment = WeekAssessment(
        days=[
            DayAssessment(
                date=menu.days[0].date,
                meals=[
                    MealAssessment(
                        meal_type="breakfast",
                        variants=[
                            VariantAssessment(
                                variant_index=0,
                                name="Paraphrased Name",
                                score=8,
                                justifications=[Justification(icon="💪", text="Reason.")],
                            )
                        ],
                    )
                ],
            )
        ]
    )

    rendered = render_plain_text(assessment, menu)

    assert "Canonical Name" in rendered
    assert "Paraphrased Name" not in rendered
