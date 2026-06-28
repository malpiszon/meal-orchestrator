from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from meal_orchestrator.config import AppConfig, UserConfig
from meal_orchestrator.delivery import DiscordClient, EmailClient
from meal_orchestrator.domain import (
    DiscordMessage,
    RunContext,
    WorkflowResult,
    WorkflowStatus,
    nearest_upcoming_monday,
    week_end_for,
)
from meal_orchestrator.llm import OpenRouterClient
from meal_orchestrator.providers import ProviderAdapter, build_provider_adapter
from meal_orchestrator.workflow import UserWorkflowExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunOptions:
    user_id: str | None = None
    provider_override: str | None = None
    week_start: date | None = None
    dry_run: bool = False
    skip_email: bool = False
    skip_discord: bool = False
    llm_model: str | None = None


class RunOrchestrator:
    def __init__(
        self,
        *,
        app_config: AppConfig,
        users: list[UserConfig],
        project_root: Path,
        provider_factory: Callable[[str], ProviderAdapter] | None = None,
        llm_client: OpenRouterClient | None = None,
        email_client: EmailClient | None = None,
        discord_client: DiscordClient | None = None,
    ) -> None:
        self.app_config = app_config
        self.users = users
        self.project_root = project_root
        self.provider_factory_override = provider_factory
        self.llm_client_override = llm_client
        self.email_client_override = email_client
        self.discord_client_override = discord_client

    def run(self, options: RunOptions) -> list[WorkflowResult]:
        run_id = uuid4().hex

        tz = ZoneInfo(self.app_config.runtime.timezone)
        today = datetime.now(tz).date()
        week_start = options.week_start or nearest_upcoming_monday(today)
        week_end = week_end_for(week_start)
        selected_users = self._select_users(options.user_id)

        logger.info(
            "run started",
            extra={"run_id": run_id, "week_start": week_start.isoformat(), "step": "start"},
        )

        discord_client = self.discord_client_override or DiscordClient(dry_run=options.dry_run)
        email_client = self.email_client_override or EmailClient(dry_run=options.dry_run)
        llm_client = self.llm_client_override or OpenRouterClient(dry_run=options.dry_run)
        provider_factory = self.provider_factory_override or build_provider_adapter

        results: list[WorkflowResult] = []
        for user in selected_users:
            provider_id = (
                options.provider_override or user.provider or self.app_config.default_provider
            )
            try:
                provider = provider_factory(provider_id)
                executor = UserWorkflowExecutor(
                    app_config=self.app_config,
                    provider=provider,
                    llm_client=llm_client,
                    email_client=email_client,
                    discord_client=discord_client,
                    project_root=self.project_root,
                )
                result = executor.execute(
                    user,
                    RunContext(
                        run_id=run_id,
                        week_start=week_start,
                        week_end=week_end,
                        dry_run=options.dry_run,
                        skip_email=options.skip_email,
                        skip_discord=options.skip_discord,
                        provider_id=provider_id,
                        llm_model=options.llm_model,
                    ),
                )
            except Exception as exc:
                logger.exception(
                    "user workflow setup failed",
                    extra={
                        "run_id": run_id,
                        "user_id": user.id,
                        "provider": provider_id,
                        "week_start": week_start.isoformat(),
                        "step": "failed",
                    },
                )
                result = WorkflowResult(
                    user_id=user.id, status=WorkflowStatus.FAILED, detail=str(exc)
                )

            if result.status == WorkflowStatus.FAILED:
                _send_operational_notification(
                    discord_client=discord_client,
                    webhook_env=self.app_config.delivery.operational_discord_webhook_env,
                    user_id=user.id,
                    run_id=run_id,
                    detail=result.detail or "unknown error",
                )

            results.append(result)

        logger.info(
            "run completed",
            extra={"run_id": run_id, "week_start": week_start.isoformat(), "step": "complete"},
        )
        return results

    def _select_users(self, user_id: str | None) -> list[UserConfig]:
        enabled_users = [user for user in self.users if user.enabled]
        if user_id is None:
            return enabled_users
        selected = [user for user in enabled_users if user.id == user_id]
        if not selected:
            raise ValueError(f"enabled user not found: {user_id}")
        return selected


def _send_operational_notification(
    *,
    discord_client: DiscordClient,
    webhook_env: str,
    user_id: str,
    run_id: str,
    detail: str,
) -> None:
    try:
        discord_client.notify(
            DiscordMessage(
                webhook_env=webhook_env,
                content=(
                    f"Meal orchestrator workflow failed for user {user_id} "
                    f"during run {run_id}: {detail}"
                ),
            )
        )
    except Exception:
        logger.warning(
            "operational discord notification failed",
            exc_info=True,
            extra={"run_id": run_id, "user_id": user_id, "step": "ops_notify"},
        )
