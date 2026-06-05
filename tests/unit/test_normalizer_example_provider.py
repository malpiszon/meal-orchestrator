import json
from datetime import date
from pathlib import Path

from meal_orchestrator.domain import PurchasedMeal
from meal_orchestrator.providers.example_provider.normalizer import normalize_example_provider_menu


def test_normalizer_filters_to_purchased_meals_and_size() -> None:
    raw_menu = json.loads(Path("tests/fixtures/provider_menu_raw.json").read_text())

    menu = normalize_example_provider_menu(
        raw_menu=raw_menu,
        provider_id="example_provider",
        week_start=date(2026, 6, 1),
        week_end=date(2026, 6, 1),
        user_id="alan",
        purchased_meals=[PurchasedMeal(type="breakfast", size="M")],
    )

    payload = menu.to_compact_dict()
    assert payload["provider"] == "example_provider"
    assert payload["user"] == {"id": "alan"}
    assert payload["days"][0]["meals"] == [
        {
            "type": "breakfast",
            "variants": [
                {
                    "name": "Tortilla",
                    "composition": "Chicken, vegetables, tortilla",
                    "nutrition": {
                        "protein_g": 28,
                        "fat_g": 18,
                        "saturated_fat_g": 6,
                        "carbs_g": 62,
                        "sugar_g": 8,
                        "fiber_g": 9,
                        "salt_g": 2.1,
                    },
                }
            ],
        }
    ]
