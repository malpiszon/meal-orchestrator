from __future__ import annotations

from typing import Protocol

from meal_orchestrator.domain import CanonicalMenu, ProviderMenuRequest


class MenuUnavailableError(RuntimeError):
    pass


class ProviderAdapter(Protocol):
    provider_id: str

    def get_canonical_week_menu(
        self,
        request: ProviderMenuRequest,
    ) -> CanonicalMenu: ...
