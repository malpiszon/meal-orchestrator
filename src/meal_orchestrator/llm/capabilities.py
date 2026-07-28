from __future__ import annotations

import json
import logging
import os
from typing import Any

from meal_orchestrator.http import get_json

logger = logging.getLogger(__name__)

_MODELS_API_URL = "https://openrouter.ai/api/v1/models"
_STRUCTURED_OUTPUT_PARAMETERS = {"response_format", "structured_outputs"}


class UnsupportedModelError(RuntimeError):
    """Raised when a configured model doesn't support structured JSON outputs on OpenRouter."""


def assert_structured_output_supported(
    model: str, *, api_key: str | None = None, timeout_seconds: int = 30
) -> None:
    """Fail fast if `model` doesn't advertise structured-output support on OpenRouter.

    This only catches models that are flatly incapable; OpenRouter's routing can
    still select a provider that silently ignores response_format for a given
    request, which is why generate() also validates every actual response.
    """
    api_key = api_key if api_key is not None else os.environ["OPENROUTER_API_KEY"]
    raw = get_json(
        _MODELS_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout_seconds=timeout_seconds,
    )
    data = json.loads(raw.decode("utf-8"))
    entry = _find_model(data, model)
    if entry is None:
        raise UnsupportedModelError(f"model not found in OpenRouter model list: {model}")
    supported = set(entry.get("supported_parameters") or [])
    if not (_STRUCTURED_OUTPUT_PARAMETERS & supported):
        raise UnsupportedModelError(
            f"model does not support structured outputs on OpenRouter: {model} "
            f"(supported_parameters={sorted(supported)})"
        )
    logger.info("openrouter capability check passed: model=%s", model)


def _find_model(data: Any, model: str) -> dict[str, Any] | None:
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None
    for entry in models:
        if isinstance(entry, dict) and entry.get("id") == model:
            return entry
    return None
