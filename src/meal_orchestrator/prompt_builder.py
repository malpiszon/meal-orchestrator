from __future__ import annotations

from pathlib import Path

from meal_orchestrator.domain import CanonicalMenu, PromptPayload


def build_prompt_payload(
    app_prompt_file: Path,
    user_prompt_file: Path,
    menu: CanonicalMenu,
) -> PromptPayload:
    app_prompt = app_prompt_file.read_text(encoding="utf-8").strip()
    user_prompt = user_prompt_file.read_text(encoding="utf-8").strip()
    return PromptPayload(
        app_prompt=app_prompt,
        user_prompt=user_prompt,
        menu=menu,
    )
