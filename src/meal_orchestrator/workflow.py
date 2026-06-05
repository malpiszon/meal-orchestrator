from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path

from meal_orchestrator.config import AppConfig, UserConfig
from meal_orchestrator.delivery import DiscordClient, EmailClient
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
from meal_orchestrator.llm import OpenRouterClient
from meal_orchestrator.prompt_builder import build_prompt_payload
from meal_orchestrator.providers import MenuUnavailableError, ProviderAdapter

logger = logging.getLogger(__name__)


class UserWorkflowExecutor:
    def __init__(
        self,
        *,
        app_config: AppConfig,
        provider: ProviderAdapter,
        llm_client: OpenRouterClient,
        email_client: EmailClient,
        discord_client: DiscordClient,
        project_root: Path,
    ) -> None:
        self.app_config = app_config
        self.provider = provider
        self.llm_client = llm_client
        self.email_client = email_client
        self.discord_client = discord_client
        self.project_root = project_root

    def execute(self, user: UserConfig, run_context: RunContext) -> WorkflowResult:
        log_context = {
            "run_id": run_context.run_id,
            "user_id": user.id,
            "provider": run_context.provider_id,
            "week_start": run_context.week_start.isoformat(),
        }
        logger.info("user workflow started", extra={**log_context, "step": "start"})
        try:
            menu = self.provider.get_canonical_week_menu(
                ProviderMenuRequest(
                    week_start=run_context.week_start,
                    week_end=run_context.week_end,
                    provider_offering_id=user.provider_offering_id,
                    user_id=user.id,
                    purchased_meals=user.purchased_meals,
                )
            )
            _ensure_complete_requested_menu(menu, user)
            logger.info(
                "provider menu normalized",
                extra={
                    **log_context,
                    "step": "provider",
                    "payload_bytes": _json_size(menu),
                },
            )

            prompt_payload = build_prompt_payload(
                prompt_file=self.project_root / user.prompt_file,
                menu=menu,
                provider=run_context.provider_id,
            )
            logger.info("prompt payload built", extra={**log_context, "step": "prompt"})

            if run_context.dry_run:
                llm_result = LlmResult(
                    text="Dry-run recommendation placeholder.",
                    model=run_context.llm_model or self.app_config.llm.model,
                    token_usage={"prompt_tokens": 0, "completion_tokens": 0},
                )
                logger.info("llm generation skipped", extra={**log_context, "step": "llm"})
            else:
                llm_result = self.llm_client.generate(
                    LlmRequest(
                        model=run_context.llm_model or self.app_config.llm.model,
                        payload=prompt_payload,
                        timeout_seconds=self.app_config.llm.timeout_seconds,
                    )
                )
                logger.info("llm result generated", extra={**log_context, "step": "llm"})

            if not run_context.skip_email and not run_context.dry_run:
                self.email_client.send(
                    EmailMessage(
                        to=user.email,
                        from_address=self.app_config.delivery.email_from,
                        subject=f"Meal plan for {run_context.week_start.isoformat()}",
                        body=llm_result.text,
                    ),
                    idempotency_key=f"{run_context.run_id}:{user.id}:email",
                )
            else:
                logger.info("email delivery skipped", extra={**log_context, "step": "email"})

            if not run_context.skip_discord:
                try:
                    self.discord_client.notify(
                        DiscordMessage(
                            webhook_env=user.discord_webhook_env,
                            content=(
                                f"Hej <@{user.discord_user_id}>, "
                                "Twoja dieta została zaplanowana."
                            ),
                        )
                    )
                    logger.info(
                        "discord notification processed",
                        extra={**log_context, "step": "discord"},
                    )
                except Exception as exc:
                    logger.warning(
                        "discord user notification failed (best effort)",
                        exc_info=True,
                        extra={**log_context, "step": "discord", "error": str(exc)},
                    )

            logger.info("user workflow completed", extra={**log_context, "step": "complete"})
            return WorkflowResult(user_id=user.id, status=WorkflowStatus.COMPLETED)
        except MenuUnavailableError as exc:
            logger.info("menu unavailable", extra={**log_context, "step": "provider"})
            if not run_context.skip_discord:
                try:
                    self.discord_client.notify(
                        DiscordMessage(
                            webhook_env=user.discord_webhook_env,
                            content=(
                                f"Hej <@{user.discord_user_id}>, "
                                "menu na wybrany tydzień nie jest jeszcze dostępne."
                            ),
                        )
                    )
                except Exception as exc_discord:
                    logger.warning(
                        "discord user notification failed for menu unavailable (best effort)",
                        exc_info=True,
                        extra={**log_context, "step": "discord", "error": str(exc_discord)},
                    )
            return WorkflowResult(
                user_id=user.id,
                status=WorkflowStatus.MENU_UNAVAILABLE,
                detail=str(exc),
            )


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
        meal_types = {meal.type for meal in day.meals}
        for purchased_meal in user.purchased_meals:
            if purchased_meal.type not in meal_types:
                raise MenuUnavailableError(
                    "missing purchased meal "
                    f"{purchased_meal.type} for requested date: {current.isoformat()}"
                )
        current += timedelta(days=1)
