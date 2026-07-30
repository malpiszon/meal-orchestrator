from __future__ import annotations

from datetime import date

from meal_orchestrator.domain.meal_types import CANONICAL_MEAL_TYPES

DAY_EMOJI = "\U0001F37D️"  # 🍽️
SUBJECT_EMOJI = "\U0001F9D1‍\U0001F373"  # 🧑‍🍳

# Covers every canonical meal type even though any given user typically purchases
# only a subset — shared across renderers so a future format (HTML, webpage, ...)
# doesn't have to redefine these.
_MEAL_LABELS: dict[str, tuple[str, str]] = {
    "breakfast": ("\U0001F373", "Śniadanie"),
    "second_breakfast": ("\U0001F370", "II Śniadanie"),
    "lunch": ("\U0001F372", "Obiad"),
    "tea": ("☕", "Podwieczorek"),
    "dinner": ("\U0001F957", "Kolacja"),
    "snack": ("\U0001F36A", "Przekąska"),
}

if set(_MEAL_LABELS) != set(CANONICAL_MEAL_TYPES):
    # A plain `raise`, not `assert` — this must still fire under `python -O`,
    # where assert statements are stripped.
    raise RuntimeError("_MEAL_LABELS has drifted from CANONICAL_MEAL_TYPES — update both together")

_WEEKDAY_NAMES_PL = (
    "Poniedziałek",
    "Wtorek",
    "Środa",
    "Czwartek",
    "Piątek",
    "Sobota",
    "Niedziela",
)


class UnknownMealTypeError(ValueError):
    """Raised when a meal type has no rendering label configured."""


def meal_label(meal_type: str) -> tuple[str, str]:
    """Return (emoji, Polish label) for a canonical meal type."""
    try:
        return _MEAL_LABELS[meal_type]
    except KeyError as exc:
        raise UnknownMealTypeError(
            f"no rendering label configured for meal type: {meal_type}"
        ) from exc


def weekday_name(value: date) -> str:
    """Return the Polish weekday name for a date."""
    return _WEEKDAY_NAMES_PL[value.weekday()]
