# meal-orchestrator

[![GitHubPackage][GitHubPackageBadge]][GitHubPackageLink]
[![DockerPublishing][DockerPublishingBadge]][DockerLink]
[![DockerSize][DockerSizeBadge]][DockerLink]
[![DockerPulls][DockerPullsBadge]][DockerLink]

A scheduled automation service that fetches a diet catering provider's weekly
menu, filters it down to what a user actually purchased, asks an LLM (via
OpenRouter) to score and justify every meal variant, and delivers the result
by email and Discord.

## Workflow

Before processing any user, the run verifies the configured LLM model (and
any configured `llm.fallback_models`) support structured JSON outputs on
OpenRouter, failing the whole run early if any of them don't.

For each configured user, per run:

1. Fetch the provider's menu for the target week (Monday–Friday).
2. Normalize the raw provider response into a compact canonical menu
   (purchased meal types and sizes only).
3. Build a prompt from the app-level rules (`prompts/app.md`), the user's own
   instructions, and the canonical menu.
4. Send the prompt to an LLM through OpenRouter, requesting a structured
   assessment (score + justifications for every meal variant), retrying with
   feedback if the response is malformed or incomplete.
5. Render the assessment as plain text and email it, and post a Discord
   notification.

A run processes users sequentially and also sends an operational Discord
notification summarizing success/failure. If a user's menu isn't published
yet for the target week, that user's LLM/email/delivery steps are skipped and
a status notification is sent instead of treating it as an error.

## Features

- Multi-user configuration, run all users or a single one (`--user`).
- Pluggable provider adapters; `ntfy` is the working integration, plus a
  minimal `example_provider` used for tests and as a template.
- OpenRouter LLM client requesting structured JSON output (per-variant score
  and justifications), with configurable model, timeout, and retries; empty,
  provider-error, malformed, or incomplete completions retry (with feedback
  telling the model what to fix), while filtered or truncated completions
  stop the user's workflow without delivery. A capability check rejects
  models without structured-output support before any user is processed. An
  optional `llm.dry_run_model` is used instead of `llm.model` during
  `--dry-run` runs, so validation runs can use a cheaper model. An optional
  `llm.fallback_models` list gives each configured model its own full retry
  budget in order: once the primary model's retries are exhausted, the next
  model is tried the same way, and so on, instead of failing the whole run.
  (OpenRouter's own `models` array looked like a simpler way to do this, but
  it doesn't fail over for the embedded-error-in-a-200-response shape rate
  limits actually use, so the client drives the switch itself.) Fallback is
  omitted during `--dry-run` runs so a rate-limited dry run can't escalate
  past `dry_run_model`'s cost tier.
- Email delivery via Resend, Discord notifications via webhooks (per-user and
  operational), both optional and independently configurable.
- `--dry-run` mode that runs the full pipeline (including the LLM call)
  without sending email or Discord messages.
- Per-run debug artifacts (raw provider response, canonical menu, LLM
  request/response, per-attempt LLM trail including rejected attempts, run
  metadata including LLM diagnostics and the failed step) with
  retention-based cleanup.
- Structured JSON logging to stdout.

## Project structure

```
src/meal_orchestrator/
  cli.py            entrypoint: argument parsing, env var checks
  orchestrator.py    run-level orchestration: user selection, target week, operational notifications
  workflow.py         per-user workflow: fetch -> normalize -> prompt -> LLM -> email -> Discord
  config/              YAML loading and config dataclasses
  domain/              shared dataclasses (canonical menu, requests/results, workflow status)
  providers/           provider adapters, one package per provider
  llm/                 OpenRouter client, model capability check
  rendering/           renders a structured LLM assessment to plain text
  delivery/            Resend email client, Discord webhook client
  observability/       structured logging setup
  artifacts.py         per-run debug artifact persistence
  retries.py           shared retry/backoff helper
  http.py              shared HTTP request helper
tests/
  unit/                unit tests, mocked HTTP, fakes for delivery clients
  fixtures/            captured raw/canonical provider payloads
config/
  app.example.yaml     runtime settings
  users.example.yaml   per-user settings
prompts/
  app.md                common rules sent to the LLM for every user
  example.md            template for a per-user prompt file
  {user}.local.md        per-user prompt files (gitignored), referenced from users.yaml
```

## Configuration

Copy the example files and edit them:

- `config/app.example.yaml` — timezone, LLM model/timeout/retries/fallback
  models, default provider, delivery settings, artifact retention.
- `config/users.example.yaml` — one entry per user: provider, provider
  offering id, email, Discord ids, prompt file, and purchased meals (type +
  size).
- `prompts/example.md` — template for a user's own prompt file (their
  personal context and preferences, in any language); copy it to
  `prompts/<user>.local.md` and point that user's `prompt_file` at it.

Secrets are read from environment variables, never from YAML:

- `OPENROUTER_API_KEY` — required; the process exits before doing anything
  else if it's missing.
- `RESEND_API_KEY` — optional; email delivery is skipped when absent.
- `DISCORD_OPS_WEBHOOK_URL` — optional; operational notifications are
  skipped when absent.
- Each user's `discord_webhook_env` (referenced by name from
  `users.yaml`) — optional; that user's Discord notification is skipped when
  absent.

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the application

```bash
meal-orchestrator --config config/app.example.yaml --users config/users.example.yaml --dry-run
```

Flags:

| Flag | Purpose |
| --- | --- |
| `--config` | Path to the app config YAML (default `config/app.example.yaml`) |
| `--users` | Path to the users config YAML (default `config/users.example.yaml`) |
| `--user` | Run a single user by id instead of all enabled users |
| `--provider` | Override the configured provider for this run |
| `--week-start` | Run against a specific week (`YYYY-MM-DD`), instead of the nearest upcoming Monday |
| `--dry-run` | Run the full pipeline including the LLM call, but skip email/Discord delivery |
| `--llm-model` | Override the configured OpenRouter model |
| `--log-level` | Log level (default `INFO`) |

Exit code is `0` if every user's workflow completed (including expected
menu-unavailable outcomes), `1` if any user's workflow failed.

## Testing

```bash
ruff check .
pytest
```

Provider normalizers are tested against captured fixture payloads; delivery
and LLM clients are tested against mocked HTTP responses, not real network
calls.

## Docker

```bash
docker compose run --rm meal-orchestrator --dry-run
```

CI builds the image on every push/PR to validate it, but only publishes it
when a version tag is released. Released images are published to both GHCR
and Docker Hub as `ghcr.io/malpiszon/meal-orchestrator` and
`malpiszon/meal-orchestrator`, tagged `<major>.<minor>.<patch>`,
`<major>.<minor>`, `<major>`, and `latest`:

```bash
docker run --rm ghcr.io/malpiszon/meal-orchestrator:latest --help
```

## Versioning

The application version is derived entirely from Git tags via
`setuptools_scm` — there is no version string to bump in the codebase.
Pushing a tag matching `vX.Y.Z` triggers the release workflow, which builds
and publishes the Docker image and creates the GitHub Release with
auto-generated notes.

## Adding a new provider

1. Add a `providers/<name>/` package with a client (raw HTTP fetch, with
   retry for transient failures) and a normalizer (raw response -> canonical
   menu).
2. Implement a class extending the `ProviderAdapter` base class in
   `providers/__init__.py`: a `provider_id` attribute and
   `get_canonical_week_menu(request) -> ProviderResult`. Optionally override
   `expected_variants_per_meal(meal_type)` if the provider guarantees a fixed
   dish-variant count per meal type (default: no check).
3. Raise `MenuUnavailableError` for expected non-availability (e.g. the
   provider hasn't published a given week/size yet) and
   `ProviderNormalizationError` for malformed/unexpected data — these are
   handled differently by the workflow (status notification vs. failure).
4. Register the provider id in `build_provider_adapter()` in
   `providers/__init__.py`.
5. Add normalizer tests against fixture payloads, and point a user's
   `provider` field at the new id.

## Design principles

- **Single sequential service.** One process, one CLI entrypoint, users
  processed one after another. Simple to run and reason about; parallelism
  can be added later around the per-user workflow boundary if it's ever
  needed.
- **Provider-specific normalizers.** Each provider owns its own raw-response
  parsing and canonical transformation rather than sharing a generic parser
  — provider APIs are inconsistent enough that a shared abstraction would be
  premature.
- **Direct OpenRouter boundary.** No generic `LlmClient` interface — the
  workflow calls `OpenRouterClient` directly. OpenRouter already abstracts
  over multiple model providers, so the extra layer isn't worth it until
  batch/async execution is needed.
- **Menu unavailability is an expected outcome, not an error.** It short-
  circuits a user's workflow (skip LLM/email, send a status notification)
  rather than failing the run.
- **Retry vs. fail-fast vs. best-effort, chosen per step.** Provider fetch
  and OpenRouter calls retry transient errors with backoff; config loading
  and normalization fail fast; email delivery retries and blocks the
  workflow on exhaustion; Discord notifications are best-effort and never
  block or fail a run.
- **File-based configuration.** YAML plus environment variables for secrets;
  no database — adequate for a small, fixed set of users on a weekly
  schedule.
- **English-only messaging for now.** Logs and Discord messages are English
  to avoid taking on i18n before it's needed.

## Known limitations

- No web UI or database; everything is driven by CLI + YAML config.
- Only one LLM request per user per run (no batching).
- If any purchased meal is missing for any day in the target week, that
  user's entire run is treated as menu-unavailable — there's no
  partial-week handling.
- No exactly-once delivery guarantee for email/Discord.
- No internal scheduler — a weekly run must be triggered externally (cron,
  systemd timer, CI schedule, etc.).

[GitHubPackageBadge]: https://img.shields.io/github/actions/workflow/status/malpiszon/meal-orchestrator/release.yml?label=GHCR
[GitHubPackageLink]: https://github.com/malpiszon/meal-orchestrator/pkgs/container/meal-orchestrator
[DockerPublishingBadge]: https://img.shields.io/github/actions/workflow/status/malpiszon/meal-orchestrator/release.yml?label=Docker%20Hub
[DockerPullsBadge]: https://badgen.net/docker/pulls/malpiszon/meal-orchestrator?icon=docker&label=Docker+Pulls&labelColor=black&color=green
[DockerSizeBadge]: https://badgen.net/docker/size/malpiszon/meal-orchestrator?icon=docker&label=Docker+Size&labelColor=black&color=green
[DockerLink]: https://hub.docker.com/r/malpiszon/meal-orchestrator
