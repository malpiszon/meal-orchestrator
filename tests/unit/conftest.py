from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _app_prompt_file(tmp_path: Path) -> None:
    """Every test that treats tmp_path as project_root needs prompts/app.md to exist.

    build_prompt_payload always reads it from a fixed, non-configurable location
    (see workflow.py's _build_llm_request), so tests can't opt out of this.
    """
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    (prompts_dir / "app.md").write_text("Score every variant.", encoding="utf-8")
