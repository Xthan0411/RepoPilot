"""Deterministic budget accounting and terminal-limit checks."""

from __future__ import annotations

from datetime import datetime

from repopilot.enums import TerminationReason
from repopilot.providers.base import ProviderUsageDelta
from repopilot.schemas import BudgetState


class BudgetController:
    """Apply one Provider decision to immutable budget state."""

    @staticmethod
    def before_decision(
        budget: BudgetState,
        *,
        now: datetime,
    ) -> TerminationReason | None:
        """Block a new Provider call when any existing hard limit is reached."""

        BudgetController._validate_now(now)
        if now >= budget.deadline_at:
            return TerminationReason.TIMEOUT
        if budget.steps_used >= budget.max_steps:
            return TerminationReason.MAX_STEPS
        if budget.tokens_used >= budget.max_tokens:
            return TerminationReason.TOKEN_LIMIT
        if budget.cost_used_usd >= budget.max_cost_usd:
            return TerminationReason.COST_LIMIT
        return None

    @staticmethod
    def charge_decision(
        budget: BudgetState,
        usage: ProviderUsageDelta,
    ) -> BudgetState:
        """Record the exact decision usage, including a possible overshoot."""

        return BudgetState(
            steps_used=budget.steps_used + 1,
            max_steps=budget.max_steps,
            tokens_used=budget.tokens_used + usage.total_tokens,
            max_tokens=budget.max_tokens,
            cost_used_usd=budget.cost_used_usd + usage.estimated_cost_usd,
            max_cost_usd=budget.max_cost_usd,
            started_at=budget.started_at,
            deadline_at=budget.deadline_at,
        )

    @staticmethod
    def after_decision(
        budget: BudgetState,
        *,
        now: datetime,
    ) -> TerminationReason | None:
        """Stop before acting when a completed Provider call overshot a limit."""

        BudgetController._validate_now(now)
        if now >= budget.deadline_at:
            return TerminationReason.TIMEOUT
        if budget.tokens_used > budget.max_tokens:
            return TerminationReason.TOKEN_LIMIT
        if budget.cost_used_usd > budget.max_cost_usd:
            return TerminationReason.COST_LIMIT
        return None

    @staticmethod
    def _validate_now(now: datetime) -> None:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("budget clock must return a timezone-aware datetime")
