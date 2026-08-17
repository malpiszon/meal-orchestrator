from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from meal_orchestrator.domain import WorkflowResult, WorkflowStatus

logger = logging.getLogger(__name__)


def run_pool(
    work_items: dict[str, Callable[[], WorkflowResult]],
    max_workers: int,
    *,
    thread_name_prefix: str,
    run_id: str,
    notify_ops: Callable[[str, WorkflowResult], None],
    worker_label: str,
) -> dict[str, WorkflowResult]:
    """Run every `work_items[user_id]` callable on a thread pool, collecting one
    WorkflowResult per user_id and reporting it through `notify_ops` as it
    completes.

    Shared by the plain concurrent-user path and the batch-delivery path —
    both need the same "bounded pool, one result per user, an unexpected
    worker exception becomes a FAILED result rather than crashing the run"
    behavior, differing only in what each unit of work actually does.
    """
    results_by_user_id: dict[str, WorkflowResult] = {}
    if not work_items:
        return results_by_user_id

    workers = min(len(work_items), max_workers)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=thread_name_prefix) as pool:
        future_to_user_id: dict[Future[WorkflowResult], str] = {}
        for user_id, work in work_items.items():
            future = pool.submit(work)
            future_to_user_id[future] = user_id

        for future in as_completed(future_to_user_id):
            user_id = future_to_user_id[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.exception(
                    "%s failed unexpectedly",
                    worker_label,
                    extra={"run_id": run_id, "user_id": user_id, "step": "failed"},
                )
                result = WorkflowResult(
                    user_id=user_id,
                    status=WorkflowStatus.FAILED,
                    detail=str(exc),
                    failed_step="worker",
                )
            results_by_user_id[user_id] = result
            notify_ops(user_id, result)

    return results_by_user_id
