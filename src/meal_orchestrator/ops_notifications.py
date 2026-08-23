from __future__ import annotations

import logging
import os

from meal_orchestrator.config import AppConfig
from meal_orchestrator.delivery import DiscordClient
from meal_orchestrator.delivery.discord import COLOR_ERROR, COLOR_SUCCESS, COLOR_WARNING
from meal_orchestrator.domain import DiscordMessage, WorkflowResult, WorkflowStatus
from meal_orchestrator.worker_pool import NotifyOps

logger = logging.getLogger(__name__)


def notify_safely(
    discord_client: DiscordClient,
    message: DiscordMessage,
    *,
    run_id: str,
    step: str,
    user_id: str | None = None,
) -> None:
    """Send `message`, logging (not raising) on failure — an operational
    notification must never fail the run it's reporting on.
    """
    try:
        discord_client.notify(message)
    except Exception:
        extra = {"run_id": run_id, "step": step}
        if user_id is not None:
            extra["user_id"] = user_id
        logger.warning("operational discord notification failed", exc_info=True, extra=extra)


def ops_webhook_env(app_config: AppConfig) -> str | None:
    """The operational Discord webhook's env var name if it's configured and
    actually set, else None — the one place that answers "can an ops-level
    notification be sent right now", shared by every ops notifier (per-user,
    batch summary, batch fallback) so the check can't drift between them.
    """
    webhook_env = app_config.delivery.operational_discord_webhook_env
    return webhook_env if webhook_env and os.environ.get(webhook_env) else None


def build_ops_notifier(
    app_config: AppConfig,
    discord_client: DiscordClient,
    model: str,
    run_id: str,
    *,
    dry_run: bool,
) -> NotifyOps:
    """Build a per-user callback that sends the operational Discord notification
    summarizing that user's workflow outcome — a no-op if dry-run or no
    operational webhook is configured/set.
    """
    configured_webhook_env = app_config.delivery.operational_discord_webhook_env

    def _notify(user_id: str, result: WorkflowResult) -> None:
        if dry_run:
            return
        webhook_env = ops_webhook_env(app_config)
        if webhook_env is None:
            if configured_webhook_env:
                logger.info(
                    "operational discord notification skipped: env var not set",
                    extra={
                        "run_id": run_id,
                        "user_id": user_id,
                        "step": "ops_notify",
                        "webhook_env": configured_webhook_env,
                    },
                )
            return
        notify_safely(
            discord_client,
            _build_message(webhook_env, user_id, run_id, result, model),
            run_id=run_id,
            step="ops_notify",
            user_id=user_id,
        )

    return _notify


def notify_capability_check_failed(
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
    notify_safely(
        discord_client,
        DiscordMessage(
            webhook_env=webhook_env,
            title="Workflow aborted",
            description=f"Capability check failed for model {model} (run {run_id}): {error}",
            color=COLOR_ERROR,
        ),
        run_id=run_id,
        step="capability_check",
    )


def _build_message(
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
