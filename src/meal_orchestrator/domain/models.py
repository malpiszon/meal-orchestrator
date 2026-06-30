from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any


@dataclass(frozen=True)
class PurchasedMeal:
    type: str
    size: str


@dataclass(frozen=True)
class RunContext:
    run_id: str
    week_start: date
    week_end: date
    dry_run: bool
    provider_id: str
    llm_model: str | None = None


@dataclass(frozen=True)
class ProviderMenuRequest:
    week_start: date
    week_end: date
    provider_offering_id: int | str
    user_id: str
    purchased_meals: list[PurchasedMeal]


@dataclass(frozen=True)
class Nutrition:
    protein_g: float | None = None
    fat_g: float | None = None
    saturated_fat_g: float | None = None
    carbs_g: float | None = None
    sugar_g: float | None = None
    fiber_g: float | None = None
    salt_g: float | None = None

    def to_compact_dict(self) -> dict[str, float]:
        return {key: value for key, value in self.__dict__.items() if value is not None}


@dataclass(frozen=True)
class MealVariant:
    name: str
    composition: str
    nutrition: Nutrition = field(default_factory=Nutrition)

    def to_compact_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "composition": self.composition,
        }
        nutrition = self.nutrition.to_compact_dict()
        if nutrition:
            payload["nutrition"] = nutrition
        return payload


@dataclass(frozen=True)
class CanonicalMeal:
    type: str
    variants: list[MealVariant]

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "variants": [variant.to_compact_dict() for variant in self.variants],
        }


@dataclass(frozen=True)
class CanonicalDay:
    date: date
    meals: list[CanonicalMeal]

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "meals": [meal.to_compact_dict() for meal in self.meals],
        }


@dataclass(frozen=True)
class CanonicalMenu:
    provider: str
    week_start: date
    week_end: date
    user_id: str
    days: list[CanonicalDay]

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "week_start": self.week_start.isoformat(),
            "week_end": self.week_end.isoformat(),
            "user": {"id": self.user_id},
            "days": [day.to_compact_dict() for day in self.days],
        }


@dataclass(frozen=True)
class ProviderResult:
    menu: CanonicalMenu
    raw_response: Any = None


@dataclass(frozen=True)
class PromptPayload:
    user_prompt: str
    menu: CanonicalMenu


@dataclass(frozen=True)
class LlmRequest:
    model: str
    payload: PromptPayload
    timeout_seconds: int


@dataclass(frozen=True)
class LlmResult:
    text: str
    model: str
    token_usage: dict[str, int] | None = None


@dataclass(frozen=True)
class EmailMessage:
    to: str
    from_address: str
    subject: str
    body: str


@dataclass(frozen=True)
class DiscordMessage:
    webhook_env: str
    content: str


class WorkflowStatus(StrEnum):
    COMPLETED = "completed"
    MENU_UNAVAILABLE = "menu_unavailable"
    FAILED = "failed"


@dataclass(frozen=True)
class WorkflowResult:
    user_id: str
    status: WorkflowStatus
    detail: str | None = None
