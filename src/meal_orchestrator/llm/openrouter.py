from __future__ import annotations

from meal_orchestrator.domain import LlmRequest, LlmResult


class OpenRouterClient:
    def generate(self, request: LlmRequest) -> LlmResult:
        raise NotImplementedError("real OpenRouter transport is intentionally not implemented yet")
