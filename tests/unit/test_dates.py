from datetime import date

from meal_orchestrator.domain import nearest_upcoming_monday, week_end_for


def test_nearest_upcoming_monday_returns_today_when_today_is_monday() -> None:
    assert nearest_upcoming_monday(date(2026, 6, 1)) == date(2026, 6, 1)


def test_nearest_upcoming_monday_returns_next_monday_for_friday() -> None:
    assert nearest_upcoming_monday(date(2026, 6, 5)) == date(2026, 6, 8)


def test_week_end_is_friday() -> None:
    assert week_end_for(date(2026, 6, 1)) == date(2026, 6, 5)
