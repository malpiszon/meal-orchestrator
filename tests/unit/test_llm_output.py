from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from meal_orchestrator.domain import CanonicalDay, CanonicalMeal, CanonicalMenu, MealVariant
from meal_orchestrator.domain.llm_output import (
    DayAssessment,
    Justification,
    MealAssessment,
    VariantAssessment,
    WeekAssessment,
    validate_completeness,
    week_assessment_json_schema,
)
from tests.unit.helpers import canonical_menu, week_assessment


def _variant(*, index: int = 0, score: int = 8) -> VariantAssessment:
    return VariantAssessment(
        variant_index=index,
        name="Meal",
        score=score,
        justifications=[Justification(icon="💪", text="Good protein.")],
    )


class TestVariantAssessmentValidation:
    def test_rejects_score_below_one(self) -> None:
        with pytest.raises(ValidationError):
            _variant(score=0)

    def test_rejects_score_above_ten(self) -> None:
        with pytest.raises(ValidationError):
            _variant(score=11)

    def test_rejects_more_than_two_justifications(self) -> None:
        with pytest.raises(ValidationError):
            VariantAssessment(
                variant_index=0,
                name="Meal",
                score=8,
                justifications=[
                    Justification(icon="💪", text="One."),
                    Justification(icon="🌱", text="Two."),
                    Justification(icon="⚠️", text="Three."),
                ],
            )

    def test_rejects_zero_justifications(self) -> None:
        with pytest.raises(ValidationError):
            VariantAssessment(variant_index=0, name="Meal", score=8, justifications=[])

    def test_accepts_one_or_two_justifications(self) -> None:
        assert _variant().justifications


class TestWeekAssessmentJsonSchema:
    def test_is_strict_with_no_additional_properties(self) -> None:
        schema = week_assessment_json_schema()

        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"].keys())
        for definition in schema["$defs"].values():
            if "properties" in definition:
                assert definition["additionalProperties"] is False
                assert set(definition["required"]) == set(definition["properties"].keys())


class TestValidateCompleteness:
    def test_no_problems_when_assessment_matches_menu(self) -> None:
        menu = canonical_menu()
        assert validate_completeness(week_assessment(menu), menu) == []

    def test_reports_missing_date(self) -> None:
        menu = canonical_menu()
        assessment = week_assessment(menu).model_copy(update={"days": []})

        problems = validate_completeness(assessment, menu)

        assert any("missing assessment for date" in p for p in problems)

    def test_reports_missing_meal_type(self) -> None:
        menu = canonical_menu()
        assessment = week_assessment(menu)
        days = list(assessment.days)
        days[0] = days[0].model_copy(update={"meals": []})
        assessment = assessment.model_copy(update={"days": days})

        problems = validate_completeness(assessment, menu)

        assert any("missing assessment for meal 'breakfast'" in p for p in problems)

    def test_reports_variant_index_mismatch(self) -> None:
        menu = canonical_menu()
        assessment = week_assessment(menu)
        days = list(assessment.days)
        meals = list(days[0].meals)
        meals[0] = MealAssessment(meal_type=meals[0].meal_type, variants=[_variant(index=5)])
        days[0] = days[0].model_copy(update={"meals": meals})
        assessment = assessment.model_copy(update={"days": days})

        problems = validate_completeness(assessment, menu)

        assert any("variant index mismatch" in p for p in problems)

    def test_reports_duplicate_variant_index_even_when_set_covers_expected_range(self) -> None:
        """A duplicate index alongside a missing one must still be flagged.

        {0, 1, 1} as a *set* collapses to {0, 1}, matching a 2-variant meal's
        expected indices exactly — so the check must also compare counts, not
        just set membership, or a duplicated (and therefore incomplete)
        assessment slips through as "complete".
        """
        menu = CanonicalMenu(
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
                                MealVariant(name="A", composition="..."),
                                MealVariant(name="B", composition="..."),
                            ],
                        )
                    ],
                )
            ],
        )
        assessment = WeekAssessment(
            days=[
                DayAssessment(
                    date=date(2026, 6, 1),
                    meals=[
                        MealAssessment(
                            meal_type="breakfast",
                            variants=[_variant(index=0), _variant(index=1), _variant(index=1)],
                        )
                    ],
                )
            ]
        )

        problems = validate_completeness(assessment, menu)

        assert any("variant index mismatch" in p for p in problems)

    def test_ignores_meal_types_not_purchased(self) -> None:
        """A user who purchased only a subset of the six meal types is fully covered.

        validate_completeness only checks meals/variants present in the menu — it
        must not require the other canonical meal types just because they exist
        in the broader vocabulary.
        """
        menu = canonical_menu()
        assert len({meal.type for day in menu.days for meal in day.meals}) == 1

        problems = validate_completeness(week_assessment(menu), menu)

        assert problems == []


def test_day_assessment_requires_meals_field() -> None:
    with pytest.raises(ValidationError):
        DayAssessment(date="2026-06-01")  # type: ignore[call-arg]


def test_week_assessment_round_trips_through_json() -> None:
    menu = canonical_menu()
    assessment = week_assessment(menu)

    restored = WeekAssessment.model_validate_json(assessment.model_dump_json())

    assert restored == assessment
