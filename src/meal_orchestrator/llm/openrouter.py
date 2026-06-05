from __future__ import annotations

from meal_orchestrator.domain import LlmRequest, LlmResult
from meal_orchestrator.prompt_builder import render_llm_request_text


class OpenRouterClient:
    def __init__(self, *, dry_run: bool = True) -> None:
        self.dry_run = dry_run

    def generate(self, request: LlmRequest) -> LlmResult:
        rendered = render_llm_request_text(request.payload)
        if self.dry_run:
            return LlmResult(
                text=(
                    "Dry-run recommendation placeholder.\n\n"
                    f"Model: {request.model}\n"
                    f"Prompt characters: {len(rendered)}"
                ),
                model=request.model,
                token_usage={"prompt_tokens": 0, "completion_tokens": 0},
            )
        raise NotImplementedError("real OpenRouter transport is intentionally not implemented yet")
