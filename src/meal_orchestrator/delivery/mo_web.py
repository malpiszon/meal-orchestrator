from __future__ import annotations

import json
import logging
from typing import Any

from meal_orchestrator import USER_AGENT
from meal_orchestrator.domain import CanonicalMenu, WeekAssessment
from meal_orchestrator.http import post_json
from meal_orchestrator.rendering.join import iter_days, iter_meals
from meal_orchestrator.retries import is_transient_http_error, with_retries

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_BASE_DELAY = 1.0
_BACKOFF_FACTOR = 2.0


def build_mo_web_payload(
    menu: CanonicalMenu, assessment: WeekAssessment, email: str, run_id: str
) -> dict[str, Any]:
    """Build the mo-web delivery request body: the full menu with every variant's score.

    Variants stay in the menu's original order (not sorted by score): mo-web's
    tie rule picks the first-listed option.
    """
    days = []
    for canonical_day, assessed_day in iter_days(assessment, menu):
        meals = []
        for canonical_meal, assessed_meal in iter_meals(canonical_day, assessed_day):
            assessed_variants = {v.variant_index: v for v in assessed_meal.variants}
            variants = []
            for index, variant in enumerate(canonical_meal.variants):
                variant_assessment = assessed_variants[index]
                payload_variant: dict[str, Any] = {
                    "provider_meal_id": variant.provider_meal_id,
                    "name": variant.name,
                    "composition": variant.composition,
                }
                nutrition = variant.nutrition.to_compact_dict()
                if nutrition:
                    payload_variant["nutrition"] = nutrition
                payload_variant["score"] = variant_assessment.score
                payload_variant["justifications"] = [
                    justification.model_dump()
                    for justification in variant_assessment.justifications
                ]
                variants.append(payload_variant)
            meals.append({"type": canonical_meal.type, "variants": variants})
        days.append({"date": canonical_day.date.isoformat(), "meals": meals})
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "provider": menu.provider,
        "week_start": menu.week_start.isoformat(),
        "week_end": menu.week_end.isoformat(),
        "user": {"email": email},
        "days": days,
    }


class MoWebHttpClient:
    def __init__(
        self,
        *,
        url: str,
        token: str,
        timeout_seconds: int = 10,
        max_retries: int = 3,
    ) -> None:
        self._url = url
        self._token = token
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

    def send(self, payload: dict[str, Any]) -> None:
        logger.info("mo-web delivery started: url=%s", self._url)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }

        def _call() -> bytes:
            return post_json(
                self._url, headers=headers, body=body, timeout_seconds=self._timeout_seconds
            )

        response = with_retries(
            _call,
            max_attempts=self._max_retries,
            base_delay_seconds=_BASE_DELAY,
            backoff_factor=_BACKOFF_FACTOR,
            retryable=is_transient_http_error,
            operation_name=f"mo-web delivery url={self._url}",
        )
        result = _parse_json_object(response)
        logger.info(
            "mo-web delivery stored: plan_id=%s account_created=%s",
            result.get("plan_id"),
            result.get("account_created"),
        )


def _parse_json_object(raw: bytes) -> dict[str, Any]:
    """Parse a response body leniently — a stored delivery shouldn't look failed
    just because the body isn't what we expected.
    """
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
