from __future__ import annotations

from datetime import date, timedelta


def nearest_upcoming_monday(today: date) -> date:
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        return today
    return today + timedelta(days=days_until_monday)


def week_end_for(week_start: date) -> date:
    return week_start + timedelta(days=4)
