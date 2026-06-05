from __future__ import annotations

from datetime import date, timedelta

from meal_orchestrator.domain import CanonicalMenu, ProviderMenuRequest
from meal_orchestrator.providers.example_provider.normalizer import normalize_example_provider_menu


class ExampleProviderAdapter:
    provider_id = "example_provider"

    def get_canonical_week_menu(
        self,
        request: ProviderMenuRequest,
    ) -> CanonicalMenu:
        raw_menu = _placeholder_raw_menu(
            request.week_start,
            request.week_end,
            request.provider_offering_id,
        )
        return normalize_example_provider_menu(
            raw_menu=raw_menu,
            provider_id=self.provider_id,
            week_start=request.week_start,
            week_end=request.week_end,
            user_id=request.user_id,
            purchased_meals=request.purchased_meals,
        )


def _placeholder_raw_menu(week_start: date, week_end: date, offering_id: int | str) -> dict:
    return {
        "offering_id": offering_id,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "days": [_placeholder_day(week_start + timedelta(days=offset)) for offset in range(5)],
    }


def _placeholder_day(day: date) -> dict:
    return {
        "date": day.isoformat(),
        "meals": [
            {
                "type": "breakfast",
                "sizes": {
                    "M": [
                        {
                            "name": "Tortilla",
                            "composition": "Chicken, vegetables, tortilla, yogurt sauce",
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
                    ]
                },
            },
            {
                "type": "lunch",
                "sizes": {
                    "XL": [
                        {
                            "name": "Rice bowl",
                            "composition": "Rice, turkey, broccoli, sesame sauce",
                            "nutrition": {
                                "protein_g": 42,
                                "fat_g": 20,
                                "carbs_g": 88,
                                "fiber_g": 11,
                                "salt_g": 2.8,
                            },
                        }
                    ]
                },
            },
        ],
    }
