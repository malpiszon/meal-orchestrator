from __future__ import annotations

import functools
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from meal_orchestrator import __version__
from meal_orchestrator.artifacts import ArtifactStore
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
from meal_orchestrator.delivery.discord import COLOR_ERROR, COLOR_SUCCESS, COLOR_WARNING
from meal_orchestrator.domain import DiscordMessage, LlmResult, WorkflowResult, WorkflowStatus
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
from meal_orchestrator.ops_notifications import notify_safely, ops_webhook_env
from meal_orchestrator.worker_pool import NotifyOps, run_pool
from meal_orchestrator.workflow import PendingUser, process_pending_synchronously

logger = logging.getLogger(__name__)


def build_run_metadata(
    *,
    run_id: str,
    week_start: date,
    week_end: date,
    model: str,
    users: list[str],
    mode: str,
    started_at: datetime,
    batch_id: str | None = None,
    batch_status: str | None = None,
    aggregate_usage: dict[str, Any] | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Build the run-level metadata.json payload — shared by every mode a run
    can end up in (plain sync, batch, and every batch-fallback variant) so
    the schema can't drift between the sync path (`RunOrchestrator`) and the
    batch path (`BatchCoordinator`).
    """
    metadata: dict[str, Any] = {
        "app_version": __version__,
        "run_id": run_id,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "model": model,
        "mode": mode,
        "batch_id": batch_id,
        "batch_status": batch_status,
        "aggregate_usage": aggregate_usage,
        "users": sorted(users),
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(UTC).isoformat(),
    }
    if fallback_reason is not None:
        metadata["fallback_reason"] = fallback_reason
    return metadata


def _build_batch_summary_message(
    webhook_env: str, run_id: str, results: dict[str, WorkflowResult]
) -> DiscordMessage:
    # execute_from_llm_result/execute_from_menu (the only work _deliver
    # dispatches) never return MENU_UNAVAILABLE — the menu is already
    # fetched by the time either runs — so only these two statuses are
    # possible here.
    completed = sorted(uid for uid, r in results.items() if r.status == WorkflowStatus.COMPLETED)
    failed = sorted(uid for uid, r in results.items() if r.status == WorkflowStatus.FAILED)
    color = COLOR_ERROR if failed else COLOR_SUCCESS
    description = (
        f"Run {run_id}: {len(completed)} completed, {len(failed)} failed "
        f"(of {len(results)} batch-delivered)."
    )
    if failed:
        description += f" Failed: {', '.join(failed)} (already alerted individually)."
    return DiscordMessage(
        webhook_env=webhook_env,
        title="Batch run summary",
        description=description,
        color=color,
    )


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

    def __init__(self, *, app_config: AppConfig) -> None:
        self.app_config = app_config

    @property
    def _state_dir(self) -> Path:
        state_dir = self.app_config.llm.batch.state_dir
        assert state_dir is not None, (
            "BatchCoordinator used without llm.batch.state_dir configured — the config "
            "loader requires it whenever llm.batch.enabled is true, and callers must not "
            "reach here otherwise"
        )
        return state_dir

    def find_pending_state(self) -> PendingBatchState | None:
        return load_state(self._state_dir)

    def try_acquire_lock(self) -> bool:
        return acquire_lock(self._state_dir)

    def release_lock(self) -> None:
        release_lock(self._state_dir)

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
        artifact_store: ArtifactStore,
    ) -> dict[str, WorkflowResult]:
        """Submit every pending user's request as one OpenRouter batch, then
        block (with exponential-backoff polling) until it completes or times
        out. No second scheduled job is involved — this runs to completion
        within the current invocation.
        """
        if not pending:
            self._save_run_metadata(
                artifact_store,
                run_id=run_id,
                week_start=week_start,
                week_end=week_end,
                model=model,
                users=[],
                mode="empty",
                started_at=datetime.now(UTC),
            )
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
            fallback_started_at = datetime.now(UTC)
            results = process_pending_synchronously(
                pending, max_concurrent_users, run_id, notify_ops
            )
            self._save_run_metadata(
                artifact_store,
                run_id=run_id,
                week_start=week_start,
                week_end=week_end,
                model=model,
                users=list(pending),
                mode="sync_fallback",
                started_at=fallback_started_at,
                fallback_reason="run lock already held",
            )
            return results
        try:
            rows = self._build_rows(pending, run_id)
            batch_id = submit_batch(rows, api_key=api_key)
            submitted_at = datetime.now(UTC)
            save_state(
                self._state_dir,
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
                week_start=week_start,
                week_end=week_end,
                model=model,
                notify_ops=notify_ops,
                discord_client=discord_client,
                api_key=api_key,
                started_at=submitted_at,
                artifact_store=artifact_store,
                initial_check_delay_seconds=self.app_config.llm.batch.initial_check_delay_seconds,
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
        artifact_store: ArtifactStore,
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
            week_start=date.fromisoformat(state.week_start),
            week_end=date.fromisoformat(state.week_end),
            model=state.model,
            notify_ops=notify_ops,
            discord_client=discord_client,
            api_key=api_key,
            started_at=datetime.fromisoformat(state.submitted_at),
            artifact_store=artifact_store,
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
        week_start: date,
        week_end: date,
        model: str,
        notify_ops: NotifyOps,
        discord_client: DiscordClient,
        api_key: str | None,
        started_at: datetime,
        artifact_store: ArtifactStore,
        initial_check_delay_seconds: float = 0.0,
    ) -> dict[str, WorkflowResult]:
        """`started_at` must be the batch's original submission time (not "now"),
        so max_wait_hours bounds the total wait from submission — otherwise a
        resumed batch (a fresh process, calling this again after a crash) would
        silently reset the deadline to a full new max_wait_hours window instead
        of continuing to count down from when the batch was actually submitted.
        """

        def _get_batch_and_log(bid: str) -> dict[str, Any]:
            # A completed batch's `data` can be huge (every row's full LLM
            # output) — log a small, fixed-size progress line here instead of
            # the raw response; the raw response itself is saved as an
            # artifact below once polling ends, not logged to stdout.
            data = get_batch(bid, api_key=api_key)
            logger.info(
                "batch status check: batch_id=%s status=%s request_counts=%s",
                bid,
                data.get("status"),
                data.get("request_counts"),
                extra={
                    "run_id": run_id,
                    "batch_id": bid,
                    "status": data.get("status"),
                    "request_counts": data.get("request_counts"),
                    "step": "batch_check",
                },
            )
            return data

        batch_config = self.app_config.llm.batch
        data = poll_until_terminal(
            batch_id,
            batch_config,
            get_batch=_get_batch_and_log,
            is_pending=lambda d: batch_status(d) in PENDING_STATUSES,
            started_at=started_at,
            initial_check_delay_seconds=initial_check_delay_seconds,
        )
        batch_result_saved = data is not None and artifact_store.save_batch_result(run_id, data)

        # State is kept alive through delivery (not cleared right after polling
        # ends) so a crash during delivery can still resume — get_batch is a
        # free re-check against an already-finished batch, and per-user email
        # delivery is idempotent (same idempotency_key across a resumed run_id),
        # so re-running delivery after a crash is safe, not just re-billable.
        if data is None or batch_status(data) != BatchStatus.COMPLETED:
            logger.warning(
                "openrouter batch did not complete usably; falling back to synchronous "
                "processing%s",
                " (see the saved batch artifact for full detail, e.g. an error/reason "
                "field OpenRouter may have included)"
                if batch_result_saved
                else "",
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
            self._save_run_metadata(
                artifact_store,
                run_id=run_id,
                week_start=week_start,
                week_end=week_end,
                model=model,
                users=list(pending),
                mode="batch_fallback",
                batch_id=batch_id,
                batch_status=batch_status(data).value if data is not None else "timed_out",
                aggregate_usage=data.get("usage") if data is not None else None,
                started_at=started_at,
                fallback_reason="batch did not complete in time or failed",
            )
            clear_state(self._state_dir)
            return results

        results, errors = parse_batch_results(rows, data)
        delivered = self._deliver(
            results, errors, pending, max_concurrent_users, run_id, notify_ops, discord_client
        )
        self._save_run_metadata(
            artifact_store,
            run_id=run_id,
            week_start=week_start,
            week_end=week_end,
            model=model,
            users=list(pending),
            mode="batch",
            batch_id=batch_id,
            batch_status=batch_status(data).value,
            aggregate_usage=data.get("usage"),
            started_at=started_at,
        )
        clear_state(self._state_dir)
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

        Notification is split by row origin: a row that fell back to
        synchronous retry is, at that point, indistinguishable from plain
        sync-mode work and gets the same real-time per-user notification for
        every status. A row actually delivered from the batch only pages
        immediately on FAILED (still urgent); COMPLETED rows are folded into
        one summary sent after delivery finishes, since the batch resolved
        them all together rather than at genuinely different times.
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

        fallback_set = set(fallback_user_ids)

        def _dispatch_notify(user_id: str, result: WorkflowResult) -> None:
            if user_id in fallback_set or result.status == WorkflowStatus.FAILED:
                notify_ops(user_id, result)

        results_by_user_id = run_pool(
            work_items,
            max_concurrent_users,
            run_id=run_id,
            notify_ops=_dispatch_notify,
            worker_label="batch delivery worker",
        )

        batch_results = {
            uid: r for uid, r in results_by_user_id.items() if uid not in fallback_set
        }
        if batch_results:
            self._notify_batch_summary(discord_client, run_id, batch_results)

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

    @staticmethod
    def _save_run_metadata(
        artifact_store: ArtifactStore,
        *,
        run_id: str,
        week_start: date,
        week_end: date,
        model: str,
        users: list[str],
        mode: str,
        started_at: datetime,
        batch_id: str | None = None,
        batch_status: str | None = None,
        aggregate_usage: dict[str, Any] | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        """Write the run-level metadata.json for a run that went through the
        batch subsystem — covering every exit point (no pending users,
        lock-skip fallback, not-completed-usably fallback, and a clean batch
        delivery) so a run's `<run_id>/metadata.json` always exists once
        batch mode has touched it, carrying the one thing OpenRouter only
        ever reports in aggregate: the batch's actual cost (`aggregate_usage`).
        """
        artifact_store.save_run_metadata(
            run_id,
            build_run_metadata(
                run_id=run_id,
                week_start=week_start,
                week_end=week_end,
                model=model,
                users=users,
                mode=mode,
                started_at=started_at,
                batch_id=batch_id,
                batch_status=batch_status,
                aggregate_usage=aggregate_usage,
                fallback_reason=fallback_reason,
            ),
        )

    def _notify_batch_summary(
        self, discord_client: DiscordClient, run_id: str, results: dict[str, WorkflowResult]
    ) -> None:
        """One message covering every row actually delivered from the batch
        (excludes rows that fell back to synchronous retry — those already
        got their own real-time notification, see `_deliver`) — replaces N
        near-simultaneous per-user messages with one, since the batch
        resolved all of them together rather than at genuinely different
        times.
        """
        webhook_env = ops_webhook_env(self.app_config)
        if webhook_env is None:
            return
        notify_safely(
            discord_client,
            _build_batch_summary_message(webhook_env, run_id, results),
            run_id=run_id,
            step="batch_summary",
        )

    def _notify_fallback(self, discord_client: DiscordClient, run_id: str, reason: str) -> None:
        webhook_env = ops_webhook_env(self.app_config)
        if webhook_env is None:
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
