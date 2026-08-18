from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import NamedTuple, Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo

from meal_orchestrator.artifacts import ArtifactStore
from meal_orchestrator.batch_coordinator import BatchCoordinator
from meal_orchestrator.batch_runner import PendingBatchState
from meal_orchestrator.config import AppConfig, UserConfig
from meal_orchestrator.delivery import DiscordClient, EmailClient, build_discord_client
from meal_orchestrator.delivery.email import ResendEmailClient
from meal_orchestrator.domain import (
    RunContext,
    WorkflowResult,
    WorkflowStatus,
    nearest_upcoming_monday,
    week_end_for,
)
from meal_orchestrator.llm import OpenRouterClient, assert_structured_output_supported
from meal_orchestrator.ops_notifications import build_ops_notifier, notify_capability_check_failed
from meal_orchestrator.providers import ProviderAdapter, build_provider_adapter
from meal_orchestrator.worker_pool import NotifyOps
from meal_orchestrator.workflow import (
    PendingUser,
    UserWorkflowExecutor,
    process_pending_synchronously,
)

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


class _RunClients(NamedTuple):
    """Clients/collaborators shared across every user processed in a run."""

    email_client: EmailClient | None
    llm_client: OpenRouterClient
    discord_client: DiscordClient
    provider_factory: Callable[[str], ProviderAdapter]
    artifact_store: ArtifactStore


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
        self.batch_coordinator = BatchCoordinator(app_config=app_config, project_root=project_root)

    def run(self, options: RunOptions) -> list[WorkflowResult]:
        batch_enabled = self.app_config.llm.batch.enabled and not options.dry_run
        if batch_enabled:
            resumed = self._resume_pending_batch_if_any(options)
            if resumed is not None:
                return resumed

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
        model = self._resolve_model(options)

        capability_failures = self._run_capability_check(
            model, discord_client, run_id, options, selected_users
        )
        if capability_failures is not None:
            return capability_failures

        clients = self._build_shared_clients(discord_client, run_id)
        notify_ops = build_ops_notifier(
            self.app_config, discord_client, model, run_id, dry_run=options.dry_run
        )

        results_by_user_id, pending = self._fetch_menus_sequentially(
            selected_users, options, run_id, week_start, week_end, clients, notify_ops
        )
        if batch_enabled:
            results_by_user_id.update(
                self.batch_coordinator.submit_and_process(
                    pending,
                    run_id=run_id,
                    week_start=week_start,
                    week_end=week_end,
                    model=model,
                    max_concurrent_users=max_concurrent_users,
                    notify_ops=notify_ops,
                    discord_client=discord_client,
                    api_key=self._llm_api_key(clients),
                )
            )
        else:
            results_by_user_id.update(
                process_pending_synchronously(pending, max_concurrent_users, run_id, notify_ops)
            )
        results = [results_by_user_id[user.id] for user in selected_users]

        logger.info(
            "run completed",
            extra={"run_id": run_id, "week_start": week_start.isoformat(), "step": "complete"},
        )
        return results

    def _run_capability_check(
        self,
        model: str,
        discord_client: DiscordClient,
        run_id: str,
        options: RunOptions,
        selected_users: list[UserConfig],
    ) -> list[WorkflowResult] | None:
        """Verify the model supports structured outputs before any user is processed.

        Returns a synthetic FAILED result per user (aborting the whole run) if the
        check fails, or None if it passed and processing should continue.
        """
        capability_check = self.capability_check_override or assert_structured_output_supported
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
            notify_capability_check_failed(
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
        return None

    def _build_shared_clients(self, discord_client: DiscordClient, run_id: str) -> _RunClients:
        """Construct (or reuse overridden) clients shared across every user this run."""
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
        return _RunClients(
            email_client=email_client,
            llm_client=llm_client,
            discord_client=discord_client,
            provider_factory=provider_factory,
            artifact_store=artifact_store,
        )

    def _fetch_menus_sequentially(
        self,
        selected_users: list[UserConfig],
        options: RunOptions,
        run_id: str,
        week_start: date,
        week_end: date,
        clients: _RunClients,
        notify_ops: NotifyOps,
    ) -> tuple[dict[str, WorkflowResult], dict[str, PendingUser]]:
        """Fetch every user's menu one at a time.

        Never parallelized — concurrent requests to the menu provider risk
        looking like a burst/DOS as the number of users grows.
        """
        results_by_user_id: dict[str, WorkflowResult] = {}
        pending: dict[str, PendingUser] = {}

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
                provider = clients.provider_factory(provider_id)
                executor = UserWorkflowExecutor(
                    app_config=self.app_config,
                    provider=provider,
                    llm_client=clients.llm_client,
                    email_client=clients.email_client,
                    discord_client=clients.discord_client,
                    project_root=self.project_root,
                    artifact_store=clients.artifact_store,
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
                notify_ops(user.id, result)
                continue

            if outcome.result is not None:
                results_by_user_id[user.id] = outcome.result
                notify_ops(user.id, outcome.result)
            else:
                pending[user.id] = PendingUser(executor, user, run_context, outcome)

        return results_by_user_id, pending

    def _resume_pending_batch_if_any(self, options: RunOptions) -> list[WorkflowResult] | None:
        """Check for a batch submitted by a prior, crashed/restarted invocation.

        Returns the resumed run's results if one was found, or None if there
        was nothing to resume (the caller should start a normal run).
        """
        state = self.batch_coordinator.find_pending_state()
        if state is None:
            return None
        logger.info(
            "resuming pending openrouter batch",
            extra={"run_id": state.run_id, "batch_id": state.batch_id, "step": "batch_resume"},
        )

        discord_client = self.discord_client_override or build_discord_client()
        notify_ops = build_ops_notifier(
            self.app_config, discord_client, state.model, state.run_id, dry_run=options.dry_run
        )

        if not self.batch_coordinator.try_acquire_lock():
            logger.warning(
                "batch resume skipped: another invocation already holds the run lock",
                extra={"run_id": state.run_id, "step": "batch_resume"},
            )
            # Reported as FAILED (not silently dropped/empty) so the exit code
            # and ops notification both reflect that these users got nothing
            # done this invocation, rather than looking like a clean no-op run.
            results = []
            for pending_user in state.users:
                result = WorkflowResult(
                    user_id=pending_user.user_id,
                    status=WorkflowStatus.FAILED,
                    detail="batch resume skipped: another invocation already holds the run lock",
                    failed_step="batch_resume",
                )
                notify_ops(pending_user.user_id, result)
                results.append(result)
            return results

        clients = self._build_shared_clients(discord_client, state.run_id)
        week_start = date.fromisoformat(state.week_start)
        week_end = date.fromisoformat(state.week_end)

        users_by_id = {user.id: user for user in self.users}
        selected_users = [
            users_by_id[pending_user.user_id]
            for pending_user in state.users
            if pending_user.user_id in users_by_id
        ]
        resume_options = RunOptions(dry_run=False, llm_model=state.model)
        results_by_user_id, pending = self._fetch_menus_sequentially(
            selected_users, resume_options, state.run_id, week_start, week_end, clients, notify_ops
        )
        self._warn_about_unresumable_users(state, users_by_id, pending)
        try:
            results_by_user_id.update(
                self.batch_coordinator.resume(
                    state,
                    pending,
                    max_concurrent_users=self._resolve_max_concurrent_users(options),
                    notify_ops=notify_ops,
                    discord_client=discord_client,
                    api_key=self._llm_api_key(clients),
                )
            )
        finally:
            self.batch_coordinator.release_lock()
        return [
            results_by_user_id[user.id]
            for user in selected_users
            if user.id in results_by_user_id
        ]

    @staticmethod
    def _warn_about_unresumable_users(
        state: PendingBatchState,
        users_by_id: dict[str, UserConfig],
        pending: dict[str, PendingUser],
    ) -> None:
        for pending_user in state.users:
            if pending_user.user_id not in users_by_id:
                logger.warning(
                    "batch resume: user %s from the pending batch is no longer in the "
                    "user config — any already-completed (and billed) batch result for "
                    "them is being discarded, since they can no longer be delivered to",
                    pending_user.user_id,
                    extra={
                        "run_id": state.run_id,
                        "user_id": pending_user.user_id,
                        "step": "batch_resume",
                    },
                )
            elif pending_user.user_id not in pending:
                logger.warning(
                    "batch resume: menu re-fetch did not succeed for user %s before batch "
                    "results could be matched — any already-completed (and billed) batch "
                    "result for them is being discarded, since delivery needs the menu",
                    pending_user.user_id,
                    extra={
                        "run_id": state.run_id,
                        "user_id": pending_user.user_id,
                        "step": "batch_resume",
                    },
                )

    @staticmethod
    def _llm_api_key(clients: _RunClients) -> str | None:
        """api_key is only guaranteed on the real OpenRouterClient — duck-typed
        test/custom llm_client implementations fall back to None, which the
        batch client resolves from OPENROUTER_API_KEY itself (unchanged
        behavior for anything that doesn't expose a configured key).
        """
        return getattr(clients.llm_client, "api_key", None)

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
