# meal-orchestrator

Scheduled automation service that fetches weekly meal menus from diet catering
providers, filters them to purchased meals, sends the payload to an LLM through
OpenRouter, then delivers the recommendation by email and Discord.

The ntfy provider integration and OpenRouter LLM client are implemented. Email
and Discord delivery are still stubs; use `--dry-run` to exercise the full
workflow without real external calls.

## Local usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
meal-orchestrator --config config/app.example.yaml --users config/users.example.yaml --dry-run
```

Run with real LLM output but skip delivery (email and Discord are still stubs):

```bash
meal-orchestrator --config config/app.example.yaml --users config/users.example.yaml --skip-email --skip-discord
meal-orchestrator --user example --week-start 2026-06-01 --skip-email --skip-discord
meal-orchestrator --llm-model openai/gpt-4.1-nano --skip-email --skip-discord
```

Skip all external calls for local testing:

```bash
meal-orchestrator --config config/app.example.yaml --users config/users.example.yaml --dry-run
```

## Docker

```bash
docker compose run --rm meal-orchestrator --dry-run
```

## Tests

```bash
ruff check .
pytest
```
