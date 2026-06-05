# meal-orchestrator

Minimal executable skeleton for a scheduled meal recommendation workflow.

This first version intentionally uses stub integrations only. It loads YAML
configuration, runs a per-user placeholder workflow, writes structured logs, and
supports dry-run execution.

## Local usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
meal-orchestrator --config config/app.example.yaml --users config/users.example.yaml --dry-run
```

Useful options (must be combined with `--dry-run` in this skeleton):

```bash
meal-orchestrator --user example --week-start 2026-06-01 --dry-run
meal-orchestrator --skip-email --skip-discord --dry-run
meal-orchestrator --llm-model openai/gpt-4.1-mini --dry-run
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
