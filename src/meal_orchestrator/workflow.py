from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from meal_orchestrator import __version__
from meal_orchestrator.artifacts import ArtifactStore, RunArtifacts
from meal_orchestrator.config import AppConfig, UserConfig
from meal_orchestrator.delivery import DiscordClient, EmailClient
from meal_orchestrator.delivery.discord import COLOR_SUCCESS, COLOR_WARNING
from meal_orchestrator.domain import (
    CanonicalMenu,
    DiscordMessage,
    EmailMessage,
    LlmRequest,
    LlmResult,
    ProviderMenuRequest,
    RunContext,
    WorkflowResult,
    WorkflowStatus,
)
from meal_orchestrator.llm import (
    LlmFailureDetails,
    OpenRouterClient,
    OpenRouterHttpError,
    OpenRouterResponseError,
)
from meal_orchestrator.prompt_builder import build_prompt_payload
from meal_orchestrator.providers import (
    MenuUnavailableError,
    ProviderAdapter,
    ProviderNormalizationError,
)
from meal_orchestrator.rendering.html import render_html
from meal_orchestrator.rendering.labels import SUBJECT_EMOJI
from meal_orchestrator.rendering.plain_text import render_plain_text

logger = logging.getLogger(__name__)


@dataclass
class _LlmAttemptsSummary:
    """Running tally of every LLM attempt (across retries and model fallback)."""

    total_attempts: int = 0
    models_tried: list[str] = field(default_factory=list)
    total_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def add_attempt(self, response: object, outcome: dict) -> None:
        self.total_attempts += 1
        model = response.get("model") if isinstance(response, dict) else None
        if model is None:
            model = outcome.get("model")
        if model and model not in self.models_tried:
            self.models_tried.append(model)
        usage = response.get("usage") if isinstance(response, dict) else None
        if isinstance(usage, dict):
            cost = usage.get("cost")
            if isinstance(cost, (int, float)):
                self.total_cost += cost
            prompt_tokens = usage.get("prompt_tokens")
            if isinstance(prompt_tokens, int):
                self.total_prompt_tokens += prompt_tokens
            completion_tokens = usage.get("completion_tokens")
            if isinstance(completion_tokens, int):
                self.total_completion_tokens += completion_tokens

    def to_metadata(self) -> dict:
        return {
            "total_attempts": self.total_attempts,
            "models_tried": self.models_tried,
            "total_cost": round(self.total_cost, 8),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }


@dataclass
class _WorkflowState:
    started_at: datetime
    status: WorkflowStatus = WorkflowStatus.FAILED
    model: str | None = None
    token_usage: dict | None = None
    error: str | None = None
    failed_step: str = "provider"
    llm_failure: LlmFailureDetails | None = None
    llm_response: dict | None = None
    llm_attempts_summary: _LlmAttemptsSummary | None = None


class UserWorkflowExecutor:
    def __init__(
        self,
        *,
        app_config: AppConfig,
        provider: ProviderAdapter,
        llm_client: OpenRouterClient,
        email_client: EmailClient | None,
        discord_client: DiscordClient,
        project_root: Path,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.app_config = app_config
        self.provider = provider
        self.llm_client = llm_client
        self.email_client = email_client
        self.discord_client = discord_client
        self.project_root = project_root
        self.artifact_store = artifact_store or ArtifactStore()

    def execute(self, user: UserConfig, run_context: RunContext) -> WorkflowResult:
        log_context = {
            "run_id": run_context.run_id,
            "user_id": user.id,
            "provider": run_context.provider_id,
            "week_start": run_context.week_start.isoformat(),
        }
        artifacts = self.artifact_store.for_run(run_context.run_id, user.id)
        state = _WorkflowState(started_at=datetime.now(UTC))

        logger.info("user workflow started", extra={**log_context, "step": "start"})
        try:
            menu = self._fetch_menu(user, run_context, artifacts, log_context)

            state.failed_step = "prompt"
            llm_request = self._build_llm_request(user, run_context, menu, log_context)
            artifacts.save_llm_request(llm_request)

            state.failed_step = "llm"
            llm_result = self._generate_plan(llm_request, artifacts, state, log_context)
            state.model = llm_result.model
            state.token_usage = llm_result.token_usage
            state.llm_response = llm_result.response_metadata

            state.failed_step = "email"
            self._deliver_email(user, run_context, menu, llm_result, log_context)

            state.failed_step = "discord"
            self._notify_plan_ready(user, run_context, log_context)

            logger.info("user workflow completed", extra={**log_context, "step": "complete"})
            state.status = WorkflowStatus.COMPLETED
            retry_count = (llm_result.response_metadata or {}).get("attempt", 1) - 1
            return WorkflowResult(
                user_id=user.id,
                status=WorkflowStatus.COMPLETED,
                retry_count=retry_count,
                model=llm_result.model,
            )
        except ProviderNormalizationError as exc:
            if exc.raw_response is not None:
                artifacts.save_provider_raw(exc.raw_response)
            state.error = str(exc)
            logger.error(
                "provider normalization failed",
                exc_info=True,
                extra={**log_context, "step": "provider", "error": state.error},
            )
            return WorkflowResult(
                user_id=user.id,
                status=WorkflowStatus.FAILED,
                detail=state.error,
                failed_step=state.failed_step,
            )
        except MenuUnavailableError as exc:
            if exc.raw_response is not None:
                artifacts.save_provider_raw(exc.raw_response)
            state.error = str(exc)
            logger.info("menu unavailable", extra={**log_context, "step": "provider"})
            state.status = WorkflowStatus.MENU_UNAVAILABLE
            self._notify_menu_unavailable(user, run_context, log_context)
            return WorkflowResult(
                user_id=user.id,
                status=WorkflowStatus.MENU_UNAVAILABLE,
                detail=state.error,
            )
        except Exception as exc:
            state.error = str(exc)
            state.llm_failure = _llm_failure_details(exc)
            logger.error(
                "user workflow failed",
                exc_info=True,
                extra={**log_context, "step": "failed", "error": state.error},
            )
            retry_count = state.llm_failure.attempt - 1 if state.llm_failure is not None else None
            return WorkflowResult(
                user_id=user.id,
                status=WorkflowStatus.FAILED,
                detail=state.error,
                failed_step=state.failed_step,
                retry_count=retry_count,
            )
        finally:
            self._save_metadata(artifacts, user, run_context, state)

    def _fetch_menu(
        self,
        user: UserConfig,
        run_context: RunContext,
        artifacts: RunArtifacts,
        log_context: dict,
    ) -> CanonicalMenu:
        provider_result = self.provider.get_canonical_week_menu(
            ProviderMenuRequest(
                week_start=run_context.week_start,
                week_end=run_context.week_end,
                provider_offering_id=user.provider_offering_id,
                user_id=user.id,
                purchased_meals=user.purchased_meals,
            )
        )
        menu = provider_result.menu
        if provider_result.raw_response is not None:
            artifacts.save_provider_raw(provider_result.raw_response)
        artifacts.save_canonical_menu(menu)
        _ensure_complete_requested_menu(menu, user)
        logger.info(
            "provider menu normalized",
            extra={
                **log_context,
                "step": "provider",
                "days": len(menu.days),
                "payload_bytes": _json_size(menu),
            },
        )
        return menu

    def _build_llm_request(
        self,
        user: UserConfig,
        run_context: RunContext,
        menu: CanonicalMenu,
        log_context: dict,
    ) -> LlmRequest:
        prompt_payload = build_prompt_payload(
            prompt_file=self.project_root / user.prompt_file,
            menu=menu,
        )
        logger.info("prompt payload built", extra={**log_context, "step": "prompt"})
        default_model = self.app_config.llm.model
        if run_context.dry_run:
            default_model = self.app_config.llm.dry_run_model or default_model
        return LlmRequest(
            model=run_context.llm_model or default_model,
            payload=prompt_payload,
            timeout_seconds=self.app_config.llm.timeout_seconds,
            # Fallback models are sized/priced for the prod model; carrying them into a
            # dry run would undercut dry_run_model's whole purpose of keeping cost down.
            fallback_models=[] if run_context.dry_run else self.app_config.llm.fallback_models,
        )

    def _generate_plan(
        self,
        request: LlmRequest,
        artifacts: RunArtifacts,
        state: _WorkflowState,
        log_context: dict,
    ) -> LlmResult:
        state.llm_attempts_summary = _LlmAttemptsSummary()

        def _on_attempt(
            attempt: int, feedback: str | None, response: object, outcome: dict
        ) -> None:
            artifacts.save_llm_attempt(
                attempt,
                {"attempt": attempt, "feedback_sent": feedback, "response": response, **outcome},
            )
            state.llm_attempts_summary.add_attempt(response, outcome)

        result = self.llm_client.generate(request, on_attempt=_on_attempt)
        artifacts.save_llm_response(result)
        logger.info("llm result generated", extra={**log_context, "step": "llm"})
        return result

    def _deliver_email(
        self,
        user: UserConfig,
        run_context: RunContext,
        menu: CanonicalMenu,
        llm_result: LlmResult,
        log_context: dict,
    ) -> None:
        if run_context.dry_run or self.email_client is None:
            logger.info("email delivery skipped", extra={**log_context, "step": "email"})
            return
        try:
            self.email_client.send(
                EmailMessage(
                    to=user.email,
                    from_address=self.app_config.delivery.email_from,
                    subject=(
                        f"{SUBJECT_EMOJI} Meal plan for {run_context.week_start.isoformat()}"
                        f" – {run_context.week_end.isoformat()}"
                    ),
                    body=render_plain_text(llm_result.structured, menu, run_context.run_id),
                    html_body=render_html(llm_result.structured, menu, run_context.run_id),
                ),
                idempotency_key=f"{run_context.run_id}:{user.id}:email",
            )
        except Exception as exc:
            logger.error(
                "email delivery failed",
                exc_info=True,
                extra={**log_context, "step": "email", "error": str(exc)},
            )
            raise

    def _notify_plan_ready(
        self, user: UserConfig, run_context: RunContext, log_context: dict
    ) -> None:
        if not _discord_enabled(run_context, user):
            logger.info(
                "discord notification skipped",
                extra={
                    **log_context,
                    "step": "discord",
                    "reason": _discord_disabled_reason(run_context, user),
                },
            )
            return
        try:
            self.discord_client.notify(
                DiscordMessage(
                    webhook_env=user.discord_webhook_env,
                    title="Meal plan ready",
                    description=(
                        f"Hey <@{user.discord_user_id}>, your meal plan for "
                        f"{run_context.week_start.isoformat()}–{run_context.week_end.isoformat()} "
                        "is ready."
                    ),
                    color=COLOR_SUCCESS,
                )
            )
            logger.info("discord notification processed", extra={**log_context, "step": "discord"})
        except Exception as exc:
            logger.warning(
                "discord user notification failed (best effort)",
                exc_info=True,
                extra={**log_context, "step": "discord", "error": str(exc)},
            )

    def _notify_menu_unavailable(
        self, user: UserConfig, run_context: RunContext, log_context: dict
    ) -> None:
        if not _discord_enabled(run_context, user):
            return
        try:
            self.discord_client.notify(
                DiscordMessage(
                    webhook_env=user.discord_webhook_env,
                    title="Menu not available yet",
                    description=(
                        f"Hey <@{user.discord_user_id}>, the menu for "
                        f"{run_context.week_start.isoformat()}–{run_context.week_end.isoformat()} "
                        "is not available yet."
                    ),
                    color=COLOR_WARNING,
                )
            )
        except Exception as exc:
            logger.warning(
                "discord user notification failed for menu unavailable (best effort)",
                exc_info=True,
                extra={**log_context, "step": "discord", "error": str(exc)},
            )

    def _save_metadata(
        self,
        artifacts: RunArtifacts,
        user: UserConfig,
        run_context: RunContext,
        state: _WorkflowState,
    ) -> None:
        metadata: dict = {
            "app_version": __version__,
            "run_id": run_context.run_id,
            "user_id": user.id,
            "provider": run_context.provider_id,
            "week_start": run_context.week_start.isoformat(),
            "week_end": run_context.week_end.isoformat(),
            "model": state.model,
            "token_usage": state.token_usage,
            "started_at": state.started_at.isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
            "status": str(state.status),
        }
        if state.error is not None:
            metadata["error"] = state.error
        if state.status == WorkflowStatus.FAILED:
            metadata["failed_step"] = state.failed_step
        if state.llm_response is not None:
            metadata["llm_response"] = state.llm_response
        if state.llm_attempts_summary is not None:
            metadata["llm_attempts_summary"] = state.llm_attempts_summary.to_metadata()
        if state.llm_failure is not None:
            metadata["llm_failure"] = state.llm_failure.to_metadata()
        artifacts.save_metadata(metadata)


def _discord_enabled(run_context: RunContext, user: UserConfig) -> bool:
    return _discord_disabled_reason(run_context, user) is None


def _discord_disabled_reason(run_context: RunContext, user: UserConfig) -> str | None:
    if run_context.dry_run:
        return "dry run"
    if not user.discord_webhook_env or not user.discord_user_id:
        return "not configured"
    if not os.environ.get(user.discord_webhook_env):
        return "env var not set"
    return None


def _json_size(menu) -> int:
    return len(json.dumps(menu.to_compact_dict(), ensure_ascii=False).encode("utf-8"))


def _ensure_complete_requested_menu(menu: CanonicalMenu, user: UserConfig) -> None:
    days_by_date = {day.date: day for day in menu.days}
    current = menu.week_start
    while current <= menu.week_end:
        day = days_by_date.get(current)
        if day is None:
            raise MenuUnavailableError(
                f"missing purchased meals for requested date: {current.isoformat()}"
            )
        meals_by_type = {meal.type: meal for meal in day.meals}
        for purchased_meal in user.purchased_meals:
            meal = meals_by_type.get(purchased_meal.type)
            if meal is None:
                raise MenuUnavailableError(
                    "missing purchased meal "
                    f"{purchased_meal.type} for requested date: {current.isoformat()}"
                )
            if not meal.variants:
                raise MenuUnavailableError(
                    f"purchased meal {purchased_meal.type} for requested date "
                    f"{current.isoformat()} has no variants"
                )
        current += timedelta(days=1)


def _llm_failure_details(exc: Exception) -> LlmFailureDetails | None:
    if isinstance(exc, (OpenRouterHttpError, OpenRouterResponseError)):
        return exc.details
    last_exception = getattr(exc, "last_exception", None)
    if isinstance(last_exception, (OpenRouterHttpError, OpenRouterResponseError)):
        return last_exception.details
    return None
