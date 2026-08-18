from __future__ import annotations

import functools
import logging
import os
from datetime import UTC, date, datetime
from pathlib import Path

from meal_orchestrator.batch_runner import (
    PendingBatchState,
    PendingBatchUser,
    acquire_lock,
    clear_state,
    load_state,
    poll_until_terminal,
    release_lock,
    save_state,
)
from meal_orchestrator.config import AppConfig
from meal_orchestrator.delivery import DiscordClient
from meal_orchestrator.delivery.discord import COLOR_WARNING
from meal_orchestrator.domain import DiscordMessage, LlmResult, WorkflowResult
from meal_orchestrator.llm import (
    PENDING_STATUSES,
    BatchRequestRow,
    BatchRowError,
    BatchStatus,
    batch_status,
    get_batch,
    parse_batch_results,
    submit_batch,
)
from meal_orchestrator.ops_notifications import notify_safely
from meal_orchestrator.worker_pool import NotifyOps, run_pool
from meal_orchestrator.workflow import PendingUser, process_pending_synchronously

logger = logging.getLogger(__name__)


class BatchCoordinator:
    """Owns the OpenRouter batch subsystem: durable state, the cross-process
    run lock, submission, resumption after a crash, polling with a
    timeout-fallback, and per-row delivery with synchronous fallback.

    `RunOrchestrator` still fetches menus and builds the shared clients
    (identical to the non-batch path) — this class only takes over once a
    `pending` set of already-fetched users is handed to it, which keeps menu
    fetching (single-flight, provider-facing) out of a class that's about
    OpenRouter's batch API.
    """

    def __init__(self, *, app_config: AppConfig, project_root: Path) -> None:
        self.app_config = app_config
        self.project_root = project_root

    def find_pending_state(self) -> PendingBatchState | None:
        return load_state(self.project_root)

    def try_acquire_lock(self) -> bool:
        return acquire_lock(self.project_root)

    def release_lock(self) -> None:
        release_lock(self.project_root)

    def submit_and_process(
        self,
        pending: dict[str, PendingUser],
        *,
        run_id: str,
        week_start: date,
        week_end: date,
        model: str,
        max_concurrent_users: int,
        notify_ops: NotifyOps,
        discord_client: DiscordClient,
        api_key: str | None,
    ) -> dict[str, WorkflowResult]:
        """Submit every pending user's request as one OpenRouter batch, then
        block (with exponential-backoff polling) until it completes or times
        out. No second scheduled job is involved — this runs to completion
        within the current invocation.
        """
        if not pending:
            return {}
        if not self.try_acquire_lock():
            logger.warning(
                "batch submission skipped: another invocation already holds the run lock; "
                "falling back to synchronous processing",
                extra={"run_id": run_id, "step": "batch_submit"},
            )
            self._notify_fallback(
                discord_client, run_id, "batch submission skipped: run lock already held"
            )
            return process_pending_synchronously(pending, max_concurrent_users, run_id, notify_ops)
        try:
            rows = self._build_rows(pending, run_id)
            batch_id = submit_batch(rows, api_key=api_key)
            submitted_at = datetime.now(UTC)
            save_state(
                self.project_root,
                PendingBatchState(
                    run_id=run_id,
                    batch_id=batch_id,
                    submitted_at=submitted_at.isoformat(),
                    week_start=week_start.isoformat(),
                    week_end=week_end.isoformat(),
                    model=model,
                    users=[
                        PendingBatchUser(
                            user_id=user_id, custom_id=self._custom_id(run_id, user_id)
                        )
                        for user_id in pending
                    ],
                ),
            )
            return self._await_and_deliver(
                batch_id,
                rows,
                pending,
                max_concurrent_users=max_concurrent_users,
                run_id=run_id,
                notify_ops=notify_ops,
                discord_client=discord_client,
                api_key=api_key,
                started_at=submitted_at,
            )
        finally:
            self.release_lock()

    def resume(
        self,
        state: PendingBatchState,
        pending: dict[str, PendingUser],
        *,
        max_concurrent_users: int,
        notify_ops: NotifyOps,
        discord_client: DiscordClient,
        api_key: str | None,
    ) -> dict[str, WorkflowResult]:
        """Continue polling/delivering a batch submitted by a prior, crashed/
        restarted invocation. Caller (`RunOrchestrator`) is responsible for
        holding the run lock across its own menu re-fetch and this call.
        """
        rows = self._build_rows(pending, state.run_id)
        return self._await_and_deliver(
            state.batch_id,
            rows,
            pending,
            max_concurrent_users=max_concurrent_users,
            run_id=state.run_id,
            notify_ops=notify_ops,
            discord_client=discord_client,
            api_key=api_key,
            started_at=datetime.fromisoformat(state.submitted_at),
        )

    def _build_rows(
        self, pending: dict[str, PendingUser], run_id: str
    ) -> list[BatchRequestRow]:
        rows = []
        for user_id, pending_user in pending.items():
            llm_request = pending_user.executor.build_llm_request(
                pending_user.user,
                pending_user.run_context,
                pending_user.outcome.menu,
                pending_user.outcome.log_context,
            )
            pending_user.outcome.artifacts.save_llm_request(llm_request)
            rows.append(
                BatchRequestRow(
                    custom_id=self._custom_id(run_id, user_id),
                    model=llm_request.model,
                    payload=llm_request.payload,
                )
            )
        return rows

    @staticmethod
    def _custom_id(run_id: str, user_id: str) -> str:
        """The batch row id used both when submitting (as `custom_id` in the
        request) and when matching results back (`parse_batch_results`,
        `_deliver`) — must stay identical across both, hence one definition.
        """
        return f"{run_id}:{user_id}"

    def _await_and_deliver(
        self,
        batch_id: str,
        rows: list[BatchRequestRow],
        pending: dict[str, PendingUser],
        *,
        max_concurrent_users: int,
        run_id: str,
        notify_ops: NotifyOps,
        discord_client: DiscordClient,
        api_key: str | None,
        started_at: datetime,
    ) -> dict[str, WorkflowResult]:
        """`started_at` must be the batch's original submission time (not "now"),
        so max_wait_hours bounds the total wait from submission — otherwise a
        resumed batch (a fresh process, calling this again after a crash) would
        silently reset the deadline to a full new max_wait_hours window instead
        of continuing to count down from when the batch was actually submitted.
        """
        batch_config = self.app_config.llm.batch
        data = poll_until_terminal(
            batch_id,
            batch_config,
            get_batch=lambda bid: get_batch(bid, api_key=api_key),
            is_pending=lambda d: batch_status(d) in PENDING_STATUSES,
            started_at=started_at,
        )

        # State is kept alive through delivery (not cleared right after polling
        # ends) so a crash during delivery can still resume — get_batch is a
        # free re-check against an already-finished batch, and per-user email
        # delivery is idempotent (same idempotency_key across a resumed run_id),
        # so re-running delivery after a crash is safe, not just re-billable.
        if data is None or batch_status(data) != BatchStatus.COMPLETED:
            logger.warning(
                "openrouter batch did not complete usably; falling back to synchronous "
                "processing",
                extra={
                    "run_id": run_id,
                    "batch_id": batch_id,
                    "status": batch_status(data).value if data is not None else "timed_out",
                    "step": "batch_fallback",
                },
            )
            self._notify_fallback(
                discord_client, run_id, f"batch {batch_id} did not complete in time or failed"
            )
            results = process_pending_synchronously(
                pending, max_concurrent_users, run_id, notify_ops
            )
            clear_state(self.project_root)
            return results

        results, errors = parse_batch_results(rows, data)
        delivered = self._deliver(
            results, errors, pending, max_concurrent_users, run_id, notify_ops, discord_client
        )
        clear_state(self.project_root)
        return delivered

    def _deliver(
        self,
        results: dict[str, LlmResult],
        errors: dict[str, BatchRowError],
        pending: dict[str, PendingUser],
        max_concurrent_users: int,
        run_id: str,
        notify_ops: NotifyOps,
        discord_client: DiscordClient,
    ) -> dict[str, WorkflowResult]:
        """Deliver every completed row, and retry every failed/missing row
        synchronously (full retry + fallback_models + artifacts trail via
        execute_from_menu) rather than marking it permanently FAILED — a
        batch row failure gets the same resilience a sync-mode call would.
        """
        if not pending:
            return {}

        fallback_user_ids: list[str] = []
        work_items = {}
        for user_id, pending_user in pending.items():
            custom_id = self._custom_id(run_id, user_id)
            if custom_id in results:
                work_items[user_id] = functools.partial(
                    pending_user.executor.execute_from_llm_result,
                    pending_user.user,
                    pending_user.run_context,
                    pending_user.outcome.menu,
                    results[custom_id],
                    pending_user.outcome.artifacts,
                    pending_user.outcome.state,
                    pending_user.outcome.log_context,
                )
            else:
                fallback_user_ids.append(user_id)
                row_error = errors.get(custom_id)
                logger.warning(
                    "batch row unusable for user=%s reason=%s; retrying synchronously",
                    user_id,
                    row_error.reason if row_error else "missing_from_batch",
                    extra={"run_id": run_id, "user_id": user_id, "step": "batch_row_retry"},
                )
                work_items[user_id] = functools.partial(
                    pending_user.executor.execute_from_menu,
                    pending_user.user,
                    pending_user.run_context,
                    pending_user.outcome.menu,
                    pending_user.outcome.artifacts,
                    pending_user.outcome.state,
                    pending_user.outcome.log_context,
                )

        results_by_user_id = run_pool(
            work_items,
            max_concurrent_users,
            run_id=run_id,
            notify_ops=notify_ops,
            worker_label="batch delivery worker",
        )

        if fallback_user_ids:
            # A handful of individual row failures is normal and already visible
            # via the per-user ops notifications above — but a systemic problem
            # with the batch (e.g. most/all rows failing) is easy to miss as a
            # pattern buried in N separate messages, so it also gets one summary
            # alert naming exactly who needed the synchronous fallback.
            self._notify_fallback(
                discord_client,
                run_id,
                f"{len(fallback_user_ids)}/{len(pending)} batch rows required synchronous "
                f"fallback (users: {', '.join(sorted(fallback_user_ids))})",
            )

        return results_by_user_id

    def _notify_fallback(self, discord_client: DiscordClient, run_id: str, reason: str) -> None:
        webhook_env = self.app_config.delivery.operational_discord_webhook_env
        if not webhook_env or not os.environ.get(webhook_env):
            return
        notify_safely(
            discord_client,
            DiscordMessage(
                webhook_env=webhook_env,
                title="Batch processing fell back to synchronous",
                description=(
                    f"Run {run_id}: {reason}; falling back to synchronous per-user calls."
                ),
                color=COLOR_WARNING,
            ),
            run_id=run_id,
            step="batch_fallback",
        )
