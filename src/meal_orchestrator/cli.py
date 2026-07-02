from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from meal_orchestrator.config import load_app_config, load_users_config
from meal_orchestrator.domain import WorkflowStatus
from meal_orchestrator.observability import configure_logging
from meal_orchestrator.orchestrator import RunOptions, RunOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meal-orchestrator")
    parser.add_argument("--config", type=Path, default=Path("config/app.example.yaml"))
    parser.add_argument("--users", type=Path, default=Path("config/users.example.yaml"))
    parser.add_argument("--user", dest="user_id")
    parser.add_argument("--provider", dest="provider_override")
    parser.add_argument("--week-start", type=date.fromisoformat)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--llm-model")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.print_usage(sys.stderr)
        print("error: OPENROUTER_API_KEY environment variable is required", file=sys.stderr)
        return 2
    configure_logging(args.log_level)
    app_config = load_app_config(args.config)
    users = load_users_config(args.users)
    orchestrator = RunOrchestrator(
        app_config=app_config,
        users=users,
        project_root=Path.cwd(),
    )
    results = orchestrator.run(
        RunOptions(
            user_id=args.user_id,
            provider_override=args.provider_override,
            week_start=args.week_start,
            dry_run=args.dry_run,
            llm_model=args.llm_model,
        )
    )
    return 1 if any(result.status == WorkflowStatus.FAILED for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
