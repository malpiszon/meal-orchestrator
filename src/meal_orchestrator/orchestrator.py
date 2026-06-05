from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from uuid import uuid4

from meal_orchestrator.config import AppConfig, UserConfig
from meal_orchestrator.domain import (
    DiscordMessage,
    RunContext,
    WorkflowResult,
    WorkflowStatus,
    nearest_upcoming_monday,
    week_end_for,
)
from meal_orchestrator.services import AppServices, build_stub_services
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
        services: AppServices | None = None,
    ) -> None:
        self.app_config = app_config
        self.users = users
        self.project_root = project_root
        self.services = services or build_stub_services()

    def run(self, options: RunOptions) -> list[WorkflowResult]:
        run_id = uuid4().hex
        week_start = options.week_start or nearest_upcoming_monday(date.today())
        week_end = week_end_for(week_start)
        selected_users = self._select_users(options.user_id)

        logger.info(
            "run started",
            extra={"run_id": run_id, "week_start": week_start.isoformat(), "step": "start"},
        )

        results: list[WorkflowResult] = []
        for user in selected_users:
            provider_id = (
                options.provider_override or user.provider or self.app_config.default_provider
            )
            provider = self.services.provider_factory(provider_id)
            executor = UserWorkflowExecutor(
                app_config=self.app_config,
                provider=provider,
                llm_client=self.services.llm_client,
                email_client=self.services.email_client,
                discord_client=self.services.discord_client,
                project_root=self.project_root,
            )
            try:
                results.append(
                    executor.execute(
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
                )
            except Exception as exc:
                logger.exception(
                    "user workflow failed",
                    extra={
                        "run_id": run_id,
                        "user_id": user.id,
                        "provider": provider_id,
                        "week_start": week_start.isoformat(),
                        "step": "failed",
                    },
                )
                self.services.discord_client.notify(
                    DiscordMessage(
                        webhook_env=self.app_config.delivery.operational_discord_webhook_env,
                        content=(
                            f"Meal orchestrator workflow failed for user {user.id} "
                            f"during run {run_id}: {exc}"
                        ),
                    )
                )
                results.append(
                    WorkflowResult(user_id=user.id, status=WorkflowStatus.FAILED, detail=str(exc))
                )

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
