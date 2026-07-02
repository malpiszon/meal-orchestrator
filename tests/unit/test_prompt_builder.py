from datetime import date

from meal_orchestrator.domain import CanonicalDay, CanonicalMeal, CanonicalMenu, MealVariant
from meal_orchestrator.prompt_builder import build_prompt_payload


def test_prompt_builder_combines_user_prompt_and_compact_menu(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Prefer high protein.", encoding="utf-8")
    menu = CanonicalMenu(
        provider="example_provider",
        week_start=date(2026, 6, 1),
        week_end=date(2026, 6, 5),
        user_id="alan",
        days=[
            CanonicalDay(
                date=date(2026, 6, 1),
                meals=[
                    CanonicalMeal(
                        type="breakfast",
                        variants=[MealVariant(name="Tortilla", composition="Chicken")],
                    )
                ],
            )
        ],
    )

    payload = build_prompt_payload(prompt_file, menu)

    assert payload.user_prompt == "Prefer high protein."
    assert payload.menu is menu
