from __future__ import annotations

from pathlib import Path

from meal_orchestrator.domain import CanonicalMenu, PromptPayload


def build_prompt_payload(
    prompt_file: Path,
    menu: CanonicalMenu,
) -> PromptPayload:
    user_prompt = prompt_file.read_text(encoding="utf-8").strip()
    return PromptPayload(
        user_prompt=user_prompt,
        menu=menu,
    )
