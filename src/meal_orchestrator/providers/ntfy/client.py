from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import date, timedelta
from typing import Any

from meal_orchestrator.providers import MenuUnavailableError
from meal_orchestrator.retries import with_retries

logger = logging.getLogger(__name__)

_BASE_URL = "https://orion-api.ntfy.pl/api/v2.0/diet-menu/{date}"
_EXPANSIONS = "diet_variant_meal_types,simple_products"

# Timeout for a single HTTP request in seconds.
_REQUEST_TIMEOUT = 30

# Default retry policy for transient network/server errors.
_MAX_ATTEMPTS = 3
_BASE_DELAY = 1.0
_BACKOFF_FACTOR = 2.0


def _build_headers() -> dict[str, str]:
    return {
        "Api-Language": "pl",
        "Trace-Id": str(uuid.uuid4()),
        "Origin": "https://ntfy.pl",
        "Referer": "https://ntfy.pl/",
        "User-Agent": "meal-orchestrator/0.1",
        "Content-Type": "application/json",
    }


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (429, 500, 502, 503, 504)
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return True
    return False


def _fetch_day_raw(date_str: str, offer_id: int | str) -> dict[str, Any]:
    """Fetch the raw API response for a single date and offer."""
    params = urllib.parse.urlencode(
        {"diet_offer_id": offer_id, "expansions__in": _EXPANSIONS}
    )
    url = f"{_BASE_URL.format(date=date_str)}?{params}"
    req = urllib.request.Request(url, headers=_build_headers())

    logger.debug("ntfy fetch day=%s offer_id=%s", date_str, offer_id)

    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise MenuUnavailableError(
                f"ntfy: menu not found for date={date_str} offer_id={offer_id}"
            ) from exc
        raise

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"ntfy: malformed JSON response for date={date_str}: {exc}"
        ) from exc

    root = data.get("data", data)
    results = root.get("results", [])
    includes = root.get("includes", {})

    if not results:
        raise MenuUnavailableError(
            f"ntfy: empty results for date={date_str} offer_id={offer_id}"
        )

    return {"date": date_str, "offer_id": offer_id, "results": results, "includes": includes}


def _fetch_day_with_retries(date_str: str, offer_id: int | str) -> dict[str, Any]:
    return with_retries(
        lambda: _fetch_day_raw(date_str, offer_id),
        max_attempts=_MAX_ATTEMPTS,
        base_delay_seconds=_BASE_DELAY,
        backoff_factor=_BACKOFF_FACTOR,
        retryable=_is_transient,
        operation_name=f"ntfy fetch date={date_str}",
    )


class NtfyClient:
    """HTTP client for the Nice To Fit You (ntfy) meal provider API.

    Fetches raw daily menu data for a week. Each day is fetched independently
    because the API is date-scoped. Raw responses are returned as-is for the
    normalizer to transform into the canonical schema.
    """

    def fetch_week_raw(
        self,
        week_start: date,
        week_end: date,
        offer_id: int | str,
    ) -> list[dict[str, Any]]:
        """Fetch raw daily menus for each weekday in [week_start, week_end].

        Returns a list of per-day raw payloads, one entry per date.

        Raises:
            MenuUnavailableError: If any day's menu is missing from the provider.
            RetryError: If a transient failure persists across all retry attempts.
            ValueError: If the provider returns malformed data.
        """
        days: list[dict[str, Any]] = []
        current = week_start

        while current <= week_end:
            date_str = current.isoformat()
            try:
                raw = _fetch_day_with_retries(date_str, offer_id)
            except MenuUnavailableError:
                logger.info("ntfy: menu unavailable for date=%s", date_str)
                raise

            logger.info(
                "ntfy: fetched date=%s offer_id=%s results=%d",
                date_str,
                offer_id,
                len(raw["results"]),
            )
            days.append(raw)
            current += timedelta(days=1)

        return days
