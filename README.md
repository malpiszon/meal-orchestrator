# meal-orchestrator

Scheduled automation service that fetches weekly meal menus from diet catering
providers, filters them to purchased meals, sends the payload to an LLM through
OpenRouter, then delivers the recommendation by email and Discord.

The `ntfy` provider integration, OpenRouter LLM client, Resend email delivery,
and Discord webhook notifications are all implemented. Email is sent
automatically when `RESEND_API_KEY` is set; Discord notifications are sent when
the relevant webhook env vars are configured. Use `--dry-run` to suppress all
external delivery.

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

CLI flags: `--config`, `--users`, `--user`, `--provider`, `--week-start`,
`--dry-run`, `--llm-model`, `--log-level`.

## Configuration

Runtime settings live in `config/app.example.yaml` (copy and edit as
`app.local.yaml` or similar), per-user settings in `config/users.example.yaml`.
Secrets are read from environment variables, never from the YAML files:

- `OPENROUTER_API_KEY` — required.
- `RESEND_API_KEY` — optional; email delivery is skipped when absent.
- `DISCORD_OPS_WEBHOOK_URL` — optional; operational notifications are skipped when absent.
- Per-user `discord_webhook_env` values referenced in `users.yaml` — optional; that user's Discord notification is skipped when absent.

Debug artifacts (raw provider response, canonical menu, LLM request/response,
run metadata) are written per-run under `artifacts.path` when
`artifacts.enabled: true`, with cleanup governed by `retention_days` and
`max_runs_per_user`.

## Docker

```bash
docker compose run --rm meal-orchestrator --dry-run
```

## Tests

```bash
ruff check .
pytest
```

## Design notes

- **Single sequential service.** One container, one CLI entrypoint, users
  processed one after another. Keeps deployment and failure handling simple
  on a home-hosted Raspberry Pi; parallelism can be added later around the
  per-user workflow boundary if needed.
- **Provider-specific normalizers.** Each provider (e.g. `ntfy`) owns its own
  raw-response parsing and its own transformation into the canonical menu
  shape, rather than sharing a generic parser. Provider APIs are inconsistent
  enough that a shared abstraction would be premature.
- **Direct OpenRouter boundary.** `OpenRouterClient` is called directly from
  the workflow instead of behind a generic `LlmClient` interface. OpenRouter
  already abstracts over multiple model providers, so an extra layer isn't
  worth the indirection until batch/async execution is needed.
- **Menu unavailability is an expected outcome, not an error.** Providers
  publish menus a limited number of days ahead. A `MenuUnavailableError`
  (distinct from parse/normalization failures) short-circuits a user's
  workflow: LLM/email are skipped and a status notification is sent instead
  of failing the run.
- **Failure handling by step:** provider fetch and OpenRouter calls retry
  transient errors with exponential backoff and a timeout; normalization and
  config loading fail fast (bad data/config, not worth retrying); email
  delivery retries and blocks the workflow on exhaustion; Discord
  notifications (user and operational) are best-effort and never block or
  fail the run.
- **English-only messaging for now.** Logs and both Discord message types are
  in English to avoid taking on i18n before it's needed.
- **File-based configuration.** YAML plus environment variables for secrets;
  no database. Sufficient for a small number of users on a fixed weekly
  schedule.
