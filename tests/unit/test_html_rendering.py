from __future__ import annotations

from datetime import date, timedelta

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
from meal_orchestrator.domain.meal_types import CANONICAL_MEAL_TYPES
from meal_orchestrator.rendering.html import render_html
from meal_orchestrator.rendering.labels import meal_label, weekday_name

# Gmail clips messages whose HTML part exceeds roughly 102KB, showing
# "[Wiadomość skrócona] Pokaż całą wiadomość" instead of the full email.
_GMAIL_CLIP_THRESHOLD_BYTES = 102_000


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

    rendered = render_html(assessment, menu, "test-run-id")

    assert rendered.index("High") < rendered.index("Mid") < rendered.index("Low")


def test_marks_all_variants_tied_for_top_score_with_star() -> None:
    menu = _menu_with_variants("A", "B", "C")
    assessment = _assessment(menu, [8, 8, 5])

    rendered = render_html(assessment, menu, "test-run-id")

    assert rendered.count("⭐") == 2


def test_includes_day_and_meal_headers() -> None:
    menu = _menu_with_variants("Only")
    assessment = _assessment(menu, [8])

    rendered = render_html(assessment, menu, "test-run-id")

    emoji, label = meal_label("breakfast")
    assert weekday_name(date(2026, 6, 1)) in rendered
    assert f"{emoji} {label}" in rendered


def test_renders_justification_icon_and_text() -> None:
    menu = _menu_with_variants("Only")
    assessment = _assessment(menu, [8])

    rendered = render_html(assessment, menu, "test-run-id")

    assert "💪" in rendered
    assert "Reason." in rendered


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

    rendered = render_html(assessment, menu, "test-run-id")

    assert "Canonical Name" in rendered
    assert "Paraphrased Name" not in rendered


def test_escapes_html_special_characters_in_llm_text() -> None:
    menu = _menu_with_variants("<script>alert(1)</script>")
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
                                name="<script>alert(1)</script>",
                                score=8,
                                justifications=[
                                    Justification(icon="💪", text="A & B <c>")
                                ],
                            )
                        ],
                    )
                ],
            )
        ]
    )

    rendered = render_html(assessment, menu, "test-run-id")

    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "A &amp; B &lt;c&gt;" in rendered


def test_produces_well_formed_html_document() -> None:
    menu = _menu_with_variants("Only")
    assessment = _assessment(menu, [8])

    rendered = render_html(assessment, menu, "test-run-id")

    assert rendered.startswith("<!doctype html>")
    assert "<html" in rendered
    assert "</html>" in rendered


def test_greeting_reports_average_and_high_score_count() -> None:
    menu = _menu_with_variants("A", "B", "C")
    assessment = _assessment(menu, [9, 9, 3])

    rendered = render_html(assessment, menu, "test-run-id")

    assert "Hi Alan! This week's top picks average 9.0/10, with 1 dish at 9+." in rendered


def test_greeting_pluralises_multiple_high_scores() -> None:
    menu = CanonicalMenu(
        provider="example_provider",
        week_start=date(2026, 6, 1),
        week_end=date(2026, 6, 2),
        user_id="example",
        days=[
            CanonicalDay(
                date=date(2026, 6, 1),
                meals=[
                    CanonicalMeal(
                        type="breakfast", variants=[MealVariant(name="A", composition="x")]
                    )
                ],
            ),
            CanonicalDay(
                date=date(2026, 6, 2),
                meals=[
                    CanonicalMeal(
                        type="breakfast", variants=[MealVariant(name="B", composition="x")]
                    )
                ],
            ),
        ],
    )
    assessment = WeekAssessment(
        days=[
            DayAssessment(
                date=date(2026, 6, 1),
                meals=[
                    MealAssessment(
                        meal_type="breakfast",
                        variants=[
                            VariantAssessment(
                                variant_index=0,
                                name="A",
                                score=9,
                                justifications=[Justification(icon="💪", text="Reason.")],
                            )
                        ],
                    )
                ],
            ),
            DayAssessment(
                date=date(2026, 6, 2),
                meals=[
                    MealAssessment(
                        meal_type="breakfast",
                        variants=[
                            VariantAssessment(
                                variant_index=0,
                                name="B",
                                score=10,
                                justifications=[Justification(icon="💪", text="Reason.")],
                            )
                        ],
                    )
                ],
            ),
        ]
    )

    rendered = render_html(assessment, menu, "test-run-id")

    assert "with 2 dishes at 9+." in rendered


def test_greeting_omits_high_score_clause_when_none_qualify() -> None:
    menu = _menu_with_variants("A", "B")
    assessment = _assessment(menu, [6, 7])

    rendered = render_html(assessment, menu, "test-run-id")

    assert "This week's top picks average 7.0/10." in rendered
    assert "9+" not in rendered
    assert "dish" not in rendered


def test_does_not_render_static_sign_off() -> None:
    menu = _menu_with_variants("Only")
    assessment = _assessment(menu, [8])

    rendered = render_html(assessment, menu, "test-run-id")

    assert "Enjoy your week" not in rendered
    assert "sign-off" not in rendered


def test_full_week_stays_under_gmail_clip_threshold() -> None:
    start = date(2026, 6, 1)
    canonical_days = []
    assessed_days = []
    for offset in range(7):
        day_date = start + timedelta(days=offset)
        meals = []
        meal_assessments = []
        for meal_type in CANONICAL_MEAL_TYPES:
            variants = [
                MealVariant(
                    name=f"Danie {meal_type} wariant {i} z dłuższym opisem składników i sosem",
                    composition="Skladniki, skladniki, skladniki.",
                )
                for i in range(3)
            ]
            meals.append(CanonicalMeal(type=meal_type, variants=variants))
            meal_assessments.append(
                MealAssessment(
                    meal_type=meal_type,
                    variants=[
                        VariantAssessment(
                            variant_index=i,
                            name="x",
                            score=5 + i,
                            justifications=[
                                Justification(
                                    icon="💪",
                                    text="Wysoka zawartość białka i błonnika, "
                                    "umiarkowana ilość cukru dodanego.",
                                ),
                                Justification(
                                    icon="🌰",
                                    text="Dobre źródło zdrowych tłuszczów "
                                    "roślinnych i witamin.",
                                ),
                            ],
                        )
                        for i in range(3)
                    ],
                )
            )
        canonical_days.append(CanonicalDay(date=day_date, meals=meals))
        assessed_days.append(DayAssessment(date=day_date, meals=meal_assessments))

    menu = CanonicalMenu(
        provider="example_provider",
        week_start=start,
        week_end=start + timedelta(days=6),
        user_id="alan",
        days=canonical_days,
    )
    assessment = WeekAssessment(days=assessed_days)

    rendered = render_html(assessment, menu, "test-run-id")

    assert len(rendered.encode("utf-8")) < _GMAIL_CLIP_THRESHOLD_BYTES
