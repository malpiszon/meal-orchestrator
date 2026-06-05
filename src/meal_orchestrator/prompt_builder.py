from __future__ import annotations

import json
from pathlib import Path

from meal_orchestrator.domain import CanonicalMenu, PromptPayload


def build_prompt_payload(
    prompt_file: Path,
    menu: CanonicalMenu,
    provider: str,
) -> PromptPayload:
    user_prompt = prompt_file.read_text(encoding="utf-8").strip()
    return PromptPayload(
        user_prompt=user_prompt,
        menu=menu,
        metadata={
            "provider": provider,
            "week_start": menu.week_start.isoformat(),
            "week_end": menu.week_end.isoformat(),
        },
    )


def render_llm_request_text(payload: PromptPayload) -> str:
    menu_json = json.dumps(
        payload.menu.to_compact_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "\n\n".join(
        [
            "User instructions:",
            payload.user_prompt,
            "Canonical menu JSON:",
            menu_json,
            "Return plain text only.",
        ]
    )
