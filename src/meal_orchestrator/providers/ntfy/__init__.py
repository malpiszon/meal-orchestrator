from __future__ import annotations

from meal_orchestrator.domain import ProviderMenuRequest, ProviderResult
from meal_orchestrator.providers import (
    MenuUnavailableError,
    ProviderAdapter,
    ProviderNormalizationError,
)

from .client import NtfyClient
from .normalizer import normalize_ntfy_week

# ntfy contractually offers a fixed number of dish variants (one per diet
# plan) for each meal type, every day. Anything else means data is missing or
# misgrouped upstream and must not be delivered to the user silently.
# SNACK only ever offers 2 diet-plan variants; every other meal type offers 3.
_DEFAULT_VARIANTS_PER_MEAL: int | None = 3
_VARIANTS_PER_MEAL_OVERRIDES: dict[str, int | None] = {
    "snack": 2,
}


class NtfyProviderAdapter(ProviderAdapter):
    provider_id = "ntfy"

    def __init__(self) -> None:
        self._client = NtfyClient()

    def expected_variants_per_meal(self, meal_type: str) -> int | None:
        return _VARIANTS_PER_MEAL_OVERRIDES.get(meal_type, _DEFAULT_VARIANTS_PER_MEAL)

    def get_canonical_week_menu(self, request: ProviderMenuRequest) -> ProviderResult:
        raw_days = self._client.fetch_week_raw(
            week_start=request.week_start,
            week_end=request.week_end,
            offer_id=request.provider_offering_id,
        )
        try:
            menu = normalize_ntfy_week(
                raw_days=raw_days,
                provider_id=self.provider_id,
                week_start=request.week_start,
                week_end=request.week_end,
                user_id=request.user_id,
                purchased_meals=request.purchased_meals,
                expected_variants_per_meal=self.expected_variants_per_meal,
            )
        except ValueError as exc:
            raise ProviderNormalizationError(str(exc), raw_response=raw_days) from exc
        except MenuUnavailableError as exc:
            exc.raw_response = raw_days
            raise
        return ProviderResult(menu=menu, raw_response=raw_days)
