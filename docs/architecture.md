# meal-orchestrator Architecture Proposal

Status: Draft

## Summary

`meal-orchestrator` is a small scheduled automation service that fetches weekly meal menus from diet catering providers, filters them to user-specific purchased meals and sizes, sends the compact canonical payload to an LLM through OpenRouter, then delivers the generated recommendation by email and Discord.

The initial system should be a single containerized Python service with a CLI entrypoint, started by an external weekly scheduler. It should keep the workflow sequential per user, avoid infrastructure dependencies beyond Docker, and expose clear interfaces for provider integrations, LLM execution, and delivery channels.

## Goals

- Run manually or on a weekly schedule.
- Support multiple users from configuration.
- Fetch provider menus for the nearest upcoming Monday through Friday.
- Normalize provider responses into a compact canonical JSON schema.
- Include only purchased meals, purchased sizes, available variants, and relevant nutrition fields.
- Generate plain-text LLM output through OpenRouter.
- Send email through Resend.
- Notify users and operational channels through Discord webhooks.
- Run on a Raspberry Pi in Docker Swarm.
- Keep the architecture extensible for future providers and batch LLM execution.

## Non-Goals

- No web UI initially.
- No distributed task queue.
- No Redis, Kafka, Airflow, Temporal, Celery.
- No exactly-once delivery guarantee.
- No shared LLM batches across users initially.
- No database requirement for the first version.
- The application must not require Kubernetes.

## High-Level Architecture

```text
Manual CLI / External Scheduler
        |
        v
Run Orchestrator
        |
        v
User Workflow Executor --------------+
        |                             |
        v                             v
Provider Adapter -> Prompt Builder
                                      |
                                      v
                              OpenRouter Client
                                      |
                                      v
                           Email + Discord Delivery
                                      |
                                      v
                           Operational Notifications
```

The service is organized around a per-user workflow. A run may process one user or many users, but each user's workflow executes sequentially.

Future parallelism may process different users concurrently once rate limits, logging, and delivery idempotency are understood.

## Components

### Run Orchestrator

Responsible for:

- Loading application configuration.
- Resolving run parameters.
- Selecting users to process.
- Calculating the target week.
- Executing each user workflow.
- Capturing unhandled workflow failures.
- Sending operational error notifications.

The orchestrator should not know provider response details, email payload details, or LLM transport details.

### User Workflow Executor

Executes one user's workflow in order:

1. Load resolved user configuration.
2. Fetch weekly provider data.
3. Normalize and filter data.
4. Build LLM prompt payload.
5. Execute LLM request.
6. Send email.
7. Send user Discord notification.

This component owns step-level logging and maps failures to retry behavior.

### Provider Facade

Provider boundary for meal provider integrations.

Responsibilities:

- Hide provider-specific API clients and response formats.
- Fetch raw menu data for a requested week.
- Select the provider menu offering configured for the user.
- Normalize provider responses into the canonical menu shape.
- Return canonical menu data to the workflow.

Provider integrations should be replaceable behind a common interface.

### Provider Adapter

Provider-specific HTTP/API integration plus transformation logic.

Responsibilities:

- Build provider API requests.
- Map the user's configured offering to the provider-specific API representation, such as a URL argument.
- Execute HTTP calls with timeout and retry policy.
- Return raw provider payloads.
- Preserve raw data enough for parser diagnostics.
- Transform raw provider data into the canonical schema.
- Keep provider-specific parsing knowledge close to the provider API integration.

Existing provider API integration code should be moved or wrapped here.

### Normalizer

Provider-specific transformation from raw provider data to canonical internal data.

The normalizer should live inside the provider adapter package. Provider APIs use different response shapes, so transformation knowledge should stay close to the provider integration rather than becoming a shared generic component too early.

Responsibilities:

- Remove irrelevant provider metadata.
- Keep only meals purchased by the user.
- Keep only the purchased size for each selected meal.
- Preserve all available variants for the selected meal and size.
- Produce canonical JSON.
- Fail fast on unknown, malformed, or incompatible provider data.

Normalization is the most important cost-control boundary because it determines the token payload sent to the LLM.

### Prompt Builder

Responsible for combining:

- User-defined prompt.
- Canonical menu JSON.
- Execution metadata, such as target week and provider name.
- Optional formatting instructions.

Initial output should be plain text from the LLM, but the prompt builder should keep the message construction isolated so structured output can be added later.

### OpenRouter Client

OpenRouter transport boundary.

Responsibilities:

- Execute synchronous chat/completion requests initially.
- Apply model selection from configuration.
- Enforce timeouts.
- Retry transient failures.
- Return generated plain text.

The first implementation may call an `OpenRouterClient` directly from the workflow because OpenRouter already abstracts over multiple model providers. If keeping a separate `LlmClient` protocol makes the implementation noisier, defer it until async/batch execution is introduced. The code should still keep OpenRouter-specific request construction isolated from provider and delivery code.

### Delivery Clients

Email and Discord delivery should be separate from workflow logic.

Email client:

- Sends generated content through Resend.
- Supports retry.
- Accepts an idempotency key if Resend supports it for the selected API path or records one in logs.

Discord client:

- May be one client with different message strategies for user notifications and operational notifications if that keeps the code simpler.
- Sends a short user delivery status notification, initially in Polish, for example: `Hej <@user_id>, Twoja dieta została zaplanowana.`
- Sends system errors to a separate operational webhook/channel.
- User notifications are best effort.
- Operational notifications retry briefly but must never block indefinitely.

## Execution Flow

### Weekly Run

1. An external scheduler starts a one-shot container at the configured time.
2. Orchestrator loads configuration.
3. Orchestrator calculates the nearest upcoming Monday.
4. Orchestrator resolves target date range as Monday through Friday.
5. For each selected enabled user:
   1. Resolve the provider offering configured for the user.
   2. Fetch provider data for target week and offering.
   3. Detect whether the menu for the requested dates is available.
   4. Normalize and filter menu for purchased meals and sizes.
   5. Build LLM payload.
   6. Call OpenRouter.
   7. Send email through Resend.
   8. Send Discord user notification.
6. Any workflow error is logged and sent to the operational Discord webhook.
7. The container exits after the run finishes.

The preferred initial deployment model is a short-lived task container started by external scheduling managed outside the application, for example by an Ansible-provisioned cron/systemd timer on the host. This avoids keeping an idle service process alive only to wait for the weekly run.

An internal scheduler mode can be added later if Docker Swarm service constraints make external scheduling inconvenient, but it should not be the default assumption.

### Manual Run

Manual runs use the same workflow with optional overrides:

- `--user <id>` to run a single user.
- `--provider <id>` to override provider where valid.
- `--week-start YYYY-MM-DD` to test a specific week.
- `--skip-email` to avoid sending email.
- `--skip-discord` to avoid Discord user notifications.
- `--dry-run` to fetch, normalize, and build prompts without external delivery.
- `--llm-model <model>` to override the configured model.

### Unavailable Menus

Providers may publish menus only a fixed number of days in advance. A menu for the target week or for specific dates within the week may not be available when the workflow runs.

The provider adapter should distinguish between:

- transient provider/API failures, which are retried;
- expected menu unavailability, which should produce a clear workflow result and user/operational status;
- malformed provider data, which should fail fast.

Initial behavior should be conservative: if any purchased meal for the requested Monday-Friday range is missing because the provider has not published it yet, skip LLM/email delivery for that user and send a short status notification. A later version may support partial-week processing if that becomes useful.

## Canonical Data Model

The canonical JSON should be intentionally compact and stable.

Initial shape:

```json
{
  "provider": "provider_id",
  "week_start": "2026-06-01",
  "week_end": "2026-06-05",
  "user": {
    "id": "user_id"
  },
  "days": [
    {
      "date": "2026-06-01",
      "meals": [
        {
          "type": "breakfast",
          "variants": [
            {
              "name": "Tortilla",
              "composition": "Chicken, vegetables, tortilla, yogurt sauce",
              "nutrition": {
                "protein_g": 28,
                "fat_g": 18,
                "saturated_fat_g": 6,
                "carbs_g": 62,
                "sugar_g": 8,
                "fiber_g": 9,
                "salt_g": 2.1
              }
            }
          ]
        }
      ]
    }
  ]
}
```

Notes:

- `nutrition` should use the most comprehensive currently known provider field set: protein, fat, saturated fat, carbohydrates, sugar, fiber, and salt.
- `composition` should be whitespace-normalized and included because ingredients are useful for choosing the best meal according to the user's prompt.
- `purchased_size` is used during filtering but omitted from the LLM payload because every included variant already corresponds to the purchased size.
- Size and calories are omitted because they are not important for the selection strategy.
- Provider rating is omitted because less healthy meals can have higher ratings, which may contradict goals defined in the user's prompt.
- Provider-specific variant IDs, descriptions, and tags are omitted initially because they are not expected to help the LLM choose the best meal according to the user's prompt.
- Missing optional fields should be omitted instead of included as `null`.
- Raw provider responses should not be sent to the LLM.

## Core Interfaces

The following interface boundaries should guide implementation. Exact syntax can evolve with the chosen language conventions.

```python
class ProviderAdapter(Protocol):
    def get_canonical_week_menu(
        self,
        week_start: date,
        week_end: date,
        provider_offering_id: int,
        user_meals: list[PurchasedMeal],
    ) -> CanonicalMenu: ...


class OpenRouterClient:
    def generate(self, request: LlmRequest) -> LlmResult: ...


class EmailClient(Protocol):
    def send(self, message: EmailMessage, idempotency_key: str) -> DeliveryResult: ...


class DiscordClient(Protocol):
    def notify(self, message: DiscordMessage) -> DeliveryResult: ...
```

Recommended domain objects:

- `UserConfig`
- `PurchasedMeal`
- `CanonicalMenu`
- `PromptPayload`
- `LlmRequest`
- `LlmResult`
- `DeliveryResult`

`RunConfig` is not needed initially as a persistent configuration model. Runtime options can be represented as parsed CLI arguments or a small internal `RunOptions` value if that makes the orchestrator cleaner. `ProviderConfig` should be added only when provider-level settings outgrow environment variables and user-level provider selection.

## Configuration Model

Configuration should be file-based, with secrets injected through environment variables.

Recommended files:

- `config/app.yaml` for global runtime settings.
- `config/users.yaml` for user-specific configuration.
- `.env` for local development secrets only.

Example `app.yaml`:

```yaml
runtime:
  timezone: "Europe/Warsaw"

llm:
  provider: "openrouter"
  model: "openai/gpt-4.1-mini"
  timeout_seconds: 120
  max_retries: 3

providers:
  default: "example_provider"

delivery:
  email_from: "Meal Orchestrator <meals@example.com>"
  operational_discord_webhook_env: "DISCORD_OPS_WEBHOOK_URL"

artifacts:
  enabled: true
  path: "/data/artifacts"
  retention_days: 14
  max_runs_per_user: 10
```

Example `users.yaml`:

```yaml
users:
  - id: "alan"
    enabled: true
    provider: "example_provider"
    provider_offering_id: 123
    email: "alan@example.com"
    discord_user_id: "123456789012345678"
    discord_webhook_env: "DISCORD_ALAN_WEBHOOK_URL"
    prompt_file: "prompts/alan.md"
    purchased_meals:
      - type: "breakfast"
        size: "M"
      - type: "lunch"
        size: "XL"
```

Required environment variables:

- `OPENROUTER_API_KEY`
- `RESEND_API_KEY`
- `DISCORD_OPS_WEBHOOK_URL`
- User Discord webhook variables referenced by config.
- Provider API credentials, if required by a provider.

The user Discord webhook may initially be the same webhook for all users. Keeping it configurable per user is still useful because it allows future separation without changing the config model. Each user should also have `discord_user_id` or an equivalent mention identifier so user notifications can mention the recipient directly.

Each user must define which provider offering should be selected, for example `Sport` or `Less gluten`. For the initial provider this is a simple integer passed as a URL argument, so `provider_offering_id` should stay as a plain field in `users.yaml`. If a future provider needs a different shape, that provider can introduce a more specific config field when needed.

Future support should account for offering changes during the week. This may involve additional details, such as different purchased meal sizes after the offering change, so the exact config shape should be designed later when implementing that capability.

## Failure Handling

| Step | Strategy |
| --- | --- |
| Config load | Fail fast before user workflows start. |
| Provider fetch | Retry with exponential backoff and request timeout. |
| Menu unavailable | Treat as an expected non-success result; skip LLM/email delivery and send short status notification. |
| Normalization/parsing | Fail fast; provider data shape likely changed or parser is wrong. |
| Prompt building | Fail fast; invalid local configuration or canonical schema. |
| OpenRouter sync request | Retry transient failures with timeout. |
| OpenRouter batch polling | Future bounded retry with max polling duration. |
| Email sending | Retry transient failures; include idempotency key in logs/request if supported. |
| User Discord notification | Best effort; failure should not fail the completed workflow. |
| Operational Discord notification | Retry briefly; never block indefinitely. |

Every workflow run should produce a run ID. Logs should include:

- `run_id`
- `user_id`
- `provider`
- `week_start`
- `step`
- `attempt`

## Observability

Initial observability should be structured logs to stdout, collected by Docker.

Logs, operational Discord notifications, exception messages, and technical diagnostics should be written in English. User-facing Discord delivery notifications are the only initially planned Polish messages.

Minimum useful events:

- Run started/completed/failed.
- User workflow started/completed/failed.
- Provider fetch duration and attempt count.
- Normalized payload size.
- LLM model, duration, and token usage when available.
- Email delivery status.
- Discord notification status.

Avoid logging secrets, full provider payloads, or full LLM prompts by default.

## Debug Artifacts

One-shot containers should write larger debugging artifacts to a bind-mounted host directory or Docker volume, not to the container filesystem. This allows the container to exit after each run while preserving selected files for inspection.

Recommended artifact path:

```text
/data/artifacts/
  <user_id>/
    <run_id>/
      provider_raw.json
      canonical_menu.json
      llm_request.json
      llm_response.txt
      metadata.json
```

Artifact rules:

- Store artifacts only when enabled in configuration.
- Redact secrets, credentials, webhook URLs, and provider authentication fields.
- Prefer storing raw provider responses and canonical JSON as files rather than logs.
- Include `run_id`, `user_id`, provider, week start, model, timestamps, and step statuses in `metadata.json`.
- Run cleanup at the beginning or end of each container execution.
- Delete artifacts older than `retention_days`.
- Also cap retained runs per user with `max_runs_per_user` so frequent manual testing does not grow storage indefinitely.

This gives enough state to debug provider parsing and prompt construction while keeping retention bounded for a home-hosted Raspberry Pi deployment.

## LLM Execution Strategy

Initial mode: synchronous per user.

Future mode: batch per user.

The first implementation can use an `OpenRouterClient` directly as the LLM boundary. OpenRouter already provides access to different model providers, so an additional abstraction should only be introduced if it improves readability or is needed for async/batch support.

A future `BatchOpenRouterClient` or `LlmClient` protocol can submit one user's request, poll with bounded retries, and return the same `LlmResult` object.

Shared batches between users should be avoided initially because they make cancellation, partial failures, privacy boundaries, and per-user retries more complex.

## Deployment

### Local Development

- Run through CLI.
- Load `.env` locally.
- Use cheap LLM model by default.
- Use `--dry-run` and `--skip-email` for parser and prompt testing.

### Raspberry Pi / Docker Swarm

Recommended deployment:

- One Docker image.
- One-shot container execution started by host-level scheduling.
- Scheduling managed outside the application, for example by an Ansible-provisioned cron job or systemd timer.
- Secrets are provided as environment variables or Docker secrets.
- Logs go to stdout/stderr.
- Debug artifacts are written to a mounted host path or Docker volume with bounded retention.
- Container exits with a success, expected non-success, or failure status after processing the selected users.

Manual runs should use the same image and command path, with CLI flags for user selection, dry runs, and delivery skipping.

An always-running container with an internal scheduler is a fallback option, not the preferred initial model.

### GitHub Actions

Pipeline stages:

1. Install dependencies.
2. Run formatting/linting.
3. Run unit tests.
4. Build Docker image.
5. Publish image to GitHub Container Registry and Docker Hub.

Deployment to the Raspberry Pi can remain manual initially.

## Initial Project Structure

```text
meal-orchestrator/
  README.md
  pyproject.toml
  Dockerfile
  docker-compose.yml
  .github/
    workflows/
      ci.yml
  config/
    app.example.yaml
    users.example.yaml
  docs/
    architecture.md
  prompts/
    example.md
  src/
    meal_orchestrator/
      __init__.py
      cli.py
      orchestrator.py
      workflow.py
      config/
        __init__.py
        loader.py
        models.py
      domain/
        __init__.py
        models.py
        dates.py
      providers/
        __init__.py
        base.py
        facade.py
        example_provider/
          __init__.py
          client.py
          normalizer.py
      llm/
        __init__.py
        openrouter.py
      delivery/
        __init__.py
        email.py
        discord.py
      observability/
        __init__.py
        logging.py
      retries.py
  tests/
    unit/
      test_dates.py
      test_normalizer_example_provider.py
      test_prompt_builder.py
    fixtures/
      provider_menu_raw.json
```

## Testing Strategy

Initial tests should focus on deterministic logic:

- Week calculation.
- Config validation.
- Provider normalization/filtering.
- Prompt payload construction.
- Retry policy behavior using fake clients.

Provider parser tests should use captured fixture payloads with sensitive fields removed.

External API clients should be tested with mocked HTTP responses, not real network calls in CI.

## Design Decisions

### Single service first

A single service keeps deployment simple on Raspberry Pi and avoids operational dependencies. Component boundaries still make future extraction possible if needed.

### File configuration first

YAML configuration is adequate for home-hosted scheduled automation. A database can be added later if users or workflow history become dynamic.

### Sequential user workflow

Sequential execution avoids subtle ordering and retry issues. Parallel processing between users can be added later around the existing workflow boundary.

### Provider-specific normalizers

Provider APIs are likely inconsistent. Keeping normalizers provider-specific avoids forcing a generic parser abstraction too early while still returning a stable canonical schema.

### External scheduling first

The initial service should run as a one-shot command started by host-level scheduling. This keeps the application simpler than a constantly running scheduler process and fits the expected weekly execution pattern.

### One Discord client is acceptable

Discord delivery can be implemented as one client with user and operational message strategies. Splitting it into multiple classes is only useful if the message behavior diverges enough to justify it.

### Direct OpenRouter boundary

OpenRouter is the initial LLM boundary. A separate generic LLM interface can be introduced later when batch execution or non-OpenRouter transports require it.

### Plain-text LLM output first

Plain text is directly useful for email. The LLM request/response boundary remains isolated so structured output can be introduced without changing provider or delivery code.

### Short Polish user notifications first

User Discord notifications should initially contain only a short delivery status in Polish, for example `Hej <@user_id>, Twoja dieta została zaplanowana.` Operational notifications can remain technical, in English, and do not need to use the same wording.
