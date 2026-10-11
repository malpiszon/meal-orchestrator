from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import sentry_sdk
from sentry_sdk.crons import capture_checkin
from sentry_sdk.crons.consts import MonitorStatus

from meal_orchestrator.config import load_app_config, load_users_config
from meal_orchestrator.domain import WorkflowStatus
from meal_orchestrator.observability import configure_logging
from meal_orchestrator.orchestrator import RunOptions, RunOrchestrator

MONITOR_SLUG = "mo-weekly-run"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meal-orchestrator")
    parser.add_argument("--config", type=Path, default=Path("config/app.example.yaml"))
    parser.add_argument("--users", type=Path, default=Path("config/users.example.yaml"))
    parser.add_argument("--user", dest="user_id")
    parser.add_argument("--provider", dest="provider_override")
    parser.add_argument("--week-start", type=date.fromisoformat)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--llm-model")
    parser.add_argument("--max-concurrent-users", type=int)
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
    sentry_sdk.init(dsn=os.environ.get("SENTRY_DSN"))
    check_in_id = capture_checkin(
        monitor_slug=MONITOR_SLUG,
        status=MonitorStatus.IN_PROGRESS,
    )
    app_config = load_app_config(args.config)
    users = load_users_config(args.users)
    orchestrator = RunOrchestrator(
        app_config=app_config,
        users=users,
        project_root=Path.cwd(),
    )
    try:
        results = orchestrator.run(
            RunOptions(
                user_id=args.user_id,
                provider_override=args.provider_override,
                week_start=args.week_start,
                dry_run=args.dry_run,
                llm_model=args.llm_model,
                max_concurrent_users=args.max_concurrent_users,
            )
        )
    except Exception:
        capture_checkin(
            monitor_slug=MONITOR_SLUG, check_in_id=check_in_id, status=MonitorStatus.ERROR
        )
        raise
    failed = any(result.status == WorkflowStatus.FAILED for result in results)
    capture_checkin(
        monitor_slug=MONITOR_SLUG,
        check_in_id=check_in_id,
        status=MonitorStatus.ERROR if failed else MonitorStatus.OK,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
