from __future__ import annotations

import logging
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo

from meal_orchestrator.artifacts import ArtifactStore
from meal_orchestrator.config import AppConfig, UserConfig
from meal_orchestrator.delivery import DiscordClient, EmailClient, build_discord_client
from meal_orchestrator.delivery.discord import COLOR_ERROR, COLOR_SUCCESS, COLOR_WARNING
from meal_orchestrator.delivery.email import ResendEmailClient
from meal_orchestrator.domain import (
    DiscordMessage,
    RunContext,
    WorkflowResult,
    WorkflowStatus,
    nearest_upcoming_monday,
    week_end_for,
)
from meal_orchestrator.llm import OpenRouterClient, assert_structured_output_supported
from meal_orchestrator.providers import ProviderAdapter, build_provider_adapter
from meal_orchestrator.workflow import UserWorkflowExecutor

logger = logging.getLogger(__name__)


class CapabilityCheck(Protocol):
    def __call__(self, model: str, *, fallback_models: list[str] | None = None) -> None: ...


@dataclass(frozen=True)
class RunOptions:
    user_id: str | None = None
    provider_override: str | None = None
    week_start: date | None = None
    dry_run: bool = False
    llm_model: str | None = None
    max_concurrent_users: int | None = None


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
        capability_check: CapabilityCheck | None = None,
    ) -> None:
        self.app_config = app_config
        self.users = users
        self.project_root = project_root
        self.provider_factory_override = provider_factory
        self.llm_client_override = llm_client
        self.email_client_override = email_client
        self.discord_client_override = discord_client
        self.capability_check_override = capability_check

    def run(self, options: RunOptions) -> list[WorkflowResult]:
        run_id = uuid4().hex

        tz = ZoneInfo(self.app_config.runtime.timezone)
        today = datetime.now(tz).date()
        week_start = options.week_start or nearest_upcoming_monday(today)
        week_end = week_end_for(week_start)
        selected_users = self._select_users(options.user_id)
        max_concurrent_users = self._resolve_max_concurrent_users(options)

        logger.info(
            "run started",
            extra={
                "run_id": run_id,
                "week_start": week_start.isoformat(),
                "step": "start",
                "max_concurrent_users": max_concurrent_users,
            },
        )

        discord_client = self.discord_client_override or build_discord_client()

        capability_check = self.capability_check_override or assert_structured_output_supported
        model = self._resolve_model(options)
        try:
            capability_check(model, fallback_models=self.app_config.llm.fallback_models)
        except Exception as exc:
            logger.error(
                "capability check failed: model=%s error=%s",
                model,
                exc,
                exc_info=True,
                extra={"run_id": run_id, "step": "capability_check"},
            )
            _notify_capability_check_failed(
                discord_client=discord_client,
                app_config=self.app_config,
                dry_run=options.dry_run,
                run_id=run_id,
                model=model,
                error=exc,
            )
            return [
                WorkflowResult(
                    user_id=user.id,
                    status=WorkflowStatus.FAILED,
                    detail=f"capability check failed for model {model}: {exc}",
                    failed_step="capability_check",
                )
                for user in selected_users
            ]

        if self.email_client_override is not None:
            email_client = self.email_client_override
        else:
            email_client = ResendEmailClient() if os.environ.get("RESEND_API_KEY") else None
        llm_client = self.llm_client_override or OpenRouterClient(
            max_retries=self.app_config.llm.max_retries
        )
        provider_factory = self.provider_factory_override or build_provider_adapter
        artifact_store = ArtifactStore(self.app_config.artifacts)
        try:
            artifact_store.cleanup()
        except Exception:
            logger.warning("artifact cleanup failed", exc_info=True, extra={"run_id": run_id})

        ops_webhook = self.app_config.delivery.operational_discord_webhook_env

        def _send_ops_notification_for(user_id: str, result: WorkflowResult) -> None:
            if options.dry_run or not ops_webhook:
                return
            if os.environ.get(ops_webhook):
                _send_operational_notification(
                    discord_client=discord_client,
                    webhook_env=ops_webhook,
                    user_id=user_id,
                    run_id=run_id,
                    result=result,
                    expected_model=model,
                )
            else:
                logger.info(
                    "operational discord notification skipped: env var not set",
                    extra={
                        "run_id": run_id,
                        "user_id": user_id,
                        "step": "ops_notify",
                        "webhook_env": ops_webhook,
                    },
                )

        results_by_user_id: dict[str, WorkflowResult] = {}
        # Outcome element is a workflow._MenuFetchOutcome (private, not imported here).
        pending: dict[str, tuple[UserWorkflowExecutor, UserConfig, RunContext, object]] = {}

        # Phase A: sequential menu fetch for every user. Never parallelized —
        # concurrent requests to the menu provider risk looking like a
        # burst/DOS as the number of users grows.
        for user in selected_users:
            provider_id = (
                options.provider_override or user.provider or self.app_config.default_provider
            )
            run_context = RunContext(
                run_id=run_id,
                week_start=week_start,
                week_end=week_end,
                dry_run=options.dry_run,
                provider_id=provider_id,
                llm_model=options.llm_model,
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
                    artifact_store=artifact_store,
                )
                outcome = executor.fetch_menu(user, run_context)
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
                    user_id=user.id,
                    status=WorkflowStatus.FAILED,
                    detail=str(exc),
                    failed_step="setup",
                )
                results_by_user_id[user.id] = result
                _send_ops_notification_for(user.id, result)
                continue

            if outcome.result is not None:
                results_by_user_id[user.id] = outcome.result
                _send_ops_notification_for(user.id, outcome.result)
            else:
                pending[user.id] = (executor, user, run_context, outcome)

        # Phase B: bounded-parallel remainder (prompt -> LLM -> email -> Discord)
        # for every user whose menu was fetched successfully.
        if pending:
            max_workers = min(len(pending), max_concurrent_users)
            with ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="user-worker"
            ) as pool:
                future_to_user_id: dict[Future[WorkflowResult], str] = {
                    pool.submit(
                        pending_executor.execute_from_menu,
                        pending_user,
                        pending_run_context,
                        pending_outcome.menu,
                        pending_outcome.artifacts,
                        pending_outcome.state,
                        pending_outcome.log_context,
                    ): user_id
                    for user_id, (
                        pending_executor,
                        pending_user,
                        pending_run_context,
                        pending_outcome,
                    ) in pending.items()
                }
                for future in as_completed(future_to_user_id):
                    user_id = future_to_user_id[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        logger.exception(
                            "user workflow worker failed unexpectedly",
                            extra={"run_id": run_id, "user_id": user_id, "step": "failed"},
                        )
                        result = WorkflowResult(
                            user_id=user_id,
                            status=WorkflowStatus.FAILED,
                            detail=str(exc),
                            failed_step="worker",
                        )
                    results_by_user_id[user_id] = result
                    _send_ops_notification_for(user_id, result)

        results = [results_by_user_id[user.id] for user in selected_users]

        logger.info(
            "run completed",
            extra={"run_id": run_id, "week_start": week_start.isoformat(), "step": "complete"},
        )
        return results

    def _resolve_model(self, options: RunOptions) -> str:
        default_model = self.app_config.llm.model
        if options.dry_run:
            default_model = self.app_config.llm.dry_run_model or default_model
        return options.llm_model or default_model

    def _resolve_max_concurrent_users(self, options: RunOptions) -> int:
        if options.max_concurrent_users is not None and options.max_concurrent_users >= 1:
            return options.max_concurrent_users
        return self.app_config.runtime.max_concurrent_users

    def _select_users(self, user_id: str | None) -> list[UserConfig]:
        enabled_users = [user for user in self.users if user.enabled]
        if user_id is None:
            return enabled_users
        selected = [user for user in enabled_users if user.id == user_id]
        if not selected:
            raise ValueError(f"enabled user not found: {user_id}")
        return selected


def _build_ops_message(
    webhook_env: str, user_id: str, run_id: str, result: WorkflowResult, expected_model: str
) -> DiscordMessage:
    if result.status == WorkflowStatus.COMPLETED:
        run_note = _run_note(run_id, result.retry_count)
        description = f"Workflow completed for user {user_id} {run_note}."
        color = COLOR_SUCCESS
        if result.model and result.model != expected_model:
            description += (
                f" Served by fallback model {result.model} (configured primary: {expected_model})."
            )
            color = COLOR_WARNING
        return DiscordMessage(
            webhook_env=webhook_env,
            title="Workflow completed",
            description=description,
            color=color,
        )
    if result.status == WorkflowStatus.MENU_UNAVAILABLE:
        detail = result.detail or "unknown reason"
        return DiscordMessage(
            webhook_env=webhook_env,
            title="Menu unavailable",
            description=f"Menu unavailable for user {user_id} (run {run_id}): {detail}",
            color=COLOR_WARNING,
        )
    failed_step = result.failed_step or "unknown"
    return DiscordMessage(
        webhook_env=webhook_env,
        title="Workflow failed",
        description=(
            f"Workflow failed for user {user_id} (run {run_id}) at step {failed_step}: "
            # The underlying error message already states the attempt count
            # (e.g. "failed after 3 attempt(s)"), so no separate retry note here.
            f"{result.detail or 'unknown error'}"
        ),
        color=COLOR_ERROR,
    )


def _run_note(run_id: str, retry_count: int | None) -> str:
    if not retry_count:
        return f"(run {run_id})"
    return f"(run {run_id}, {retry_count} retr{'y' if retry_count == 1 else 'ies'})"


def _notify_capability_check_failed(
    *,
    discord_client: DiscordClient,
    app_config: AppConfig,
    dry_run: bool,
    run_id: str,
    model: str,
    error: Exception,
) -> None:
    webhook_env = app_config.delivery.operational_discord_webhook_env
    if dry_run or not webhook_env or not os.environ.get(webhook_env):
        return
    try:
        discord_client.notify(
            DiscordMessage(
                webhook_env=webhook_env,
                title="Workflow aborted",
                description=(
                    f"Capability check failed for model {model} (run {run_id}): {error}"
                ),
                color=COLOR_ERROR,
            )
        )
    except Exception:
        logger.warning(
            "operational discord notification failed",
            exc_info=True,
            extra={"run_id": run_id, "step": "capability_check"},
        )


def _send_operational_notification(
    *,
    discord_client: DiscordClient,
    webhook_env: str,
    user_id: str,
    run_id: str,
    result: WorkflowResult,
    expected_model: str,
) -> None:
    try:
        discord_client.notify(
            _build_ops_message(webhook_env, user_id, run_id, result, expected_model)
        )
    except Exception:
        logger.warning(
            "operational discord notification failed",
            exc_info=True,
            extra={"run_id": run_id, "user_id": user_id, "step": "ops_notify"},
        )
