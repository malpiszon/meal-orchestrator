from __future__ import annotations

from meal_orchestrator.domain import CanonicalMenu, ProviderMenuRequest

from .client import NtfyClient
from .normalizer import normalize_ntfy_week


class NtfyProviderAdapter:
    provider_id = "ntfy"

    def __init__(self) -> None:
        self._client = NtfyClient()

    def get_canonical_week_menu(self, request: ProviderMenuRequest) -> CanonicalMenu:
        raw_days = self._client.fetch_week_raw(
            week_start=request.week_start,
            week_end=request.week_end,
            offer_id=request.provider_offering_id,
        )
        return normalize_ntfy_week(
            raw_days=raw_days,
            provider_id=self.provider_id,
            week_start=request.week_start,
            week_end=request.week_end,
            user_id=request.user_id,
            purchased_meals=request.purchased_meals,
        )
