# meal-orchestrator

Scheduled automation service that fetches weekly meal menus from diet catering
providers, filters them to purchased meals, sends the payload to an LLM through
OpenRouter, then delivers the recommendation by email and Discord.

The ntfy provider integration, OpenRouter LLM client, and Resend email delivery
are implemented. Discord delivery is a stub. Email is sent automatically when
`RESEND_API_KEY` is set; Discord notifications are sent when the relevant webhook
env vars are configured. Use `--dry-run` to suppress all external delivery.

## Local usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
meal-orchestrator --config config/app.example.yaml --users config/users.example.yaml --dry-run
```

Run with real LLM output and delivery (email sent if `RESEND_API_KEY` is set):

```bash
meal-orchestrator --config config/app.example.yaml --users config/users.example.yaml
meal-orchestrator --config config/app.example.yaml --users config/users.example.yaml --user example --week-start 2026-06-01
meal-orchestrator --config config/app.example.yaml --users config/users.example.yaml --llm-model openai/gpt-4.1-nano
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
