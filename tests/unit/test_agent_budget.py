"""Budget accounting precedence and boundary tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from repopilot.agent import BudgetController
from repopilot.enums import TerminationReason
from repopilot.providers import ProviderUsageDelta
from repopilot.schemas import BudgetState

STARTED_AT = datetime(2026, 8, 5, 8, tzinfo=UTC)


def make_budget(**updates: object) -> BudgetState:
    values: dict[str, object] = {
        "steps_used": 0,
        "max_steps": 4,
        "tokens_used": 0,
        "max_tokens": 100,
        "cost_used_usd": Decimal("0"),
        "max_cost_usd": Decimal("1"),
        "started_at": STARTED_AT,
        "deadline_at": STARTED_AT + timedelta(minutes=10),
    }
    values.update(updates)
    return BudgetState.model_validate(values)


def make_usage(*, tokens: int = 12, cost: str = "0.1") -> ProviderUsageDelta:
    return ProviderUsageDelta(
        input_tokens=tokens,
        output_tokens=0,
        total_tokens=tokens,
        estimated_cost_usd=Decimal(cost),
        latency_ms=5,
    )


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (
            make_budget(
                steps_used=4,
                tokens_used=100,
                cost_used_usd=Decimal("1"),
            ),
            TerminationReason.MAX_STEPS,
        ),
        (
            make_budget(tokens_used=100, cost_used_usd=Decimal("1")),
            TerminationReason.TOKEN_LIMIT,
        ),
        (
            make_budget(cost_used_usd=Decimal("1")),
            TerminationReason.COST_LIMIT,
        ),
    ],
)
def test_before_decision_has_stable_non_time_limit_precedence(
    budget: BudgetState,
    expected: TerminationReason,
) -> None:
    assert (
        BudgetController.before_decision(
            budget,
            now=STARTED_AT,
        )
        is expected
    )


def test_timeout_has_highest_pre_decision_precedence() -> None:
    budget = make_budget(
        steps_used=4,
        tokens_used=100,
        cost_used_usd=Decimal("1"),
    )

    assert (
        BudgetController.before_decision(
            budget,
            now=budget.deadline_at,
        )
        is TerminationReason.TIMEOUT
    )


def test_exact_post_decision_token_and_cost_limits_are_allowed_until_next_call() -> None:
    charged = BudgetController.charge_decision(
        make_budget(max_tokens=12, max_cost_usd=Decimal("0.1")),
        make_usage(),
    )

    assert BudgetController.after_decision(charged, now=STARTED_AT) is None
    assert (
        BudgetController.before_decision(charged, now=STARTED_AT) is TerminationReason.TOKEN_LIMIT
    )


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (make_usage(tokens=101, cost="0.1"), TerminationReason.TOKEN_LIMIT),
        (make_usage(tokens=12, cost="1.1"), TerminationReason.COST_LIMIT),
    ],
)
def test_post_decision_records_overshoot_and_stops_before_action(
    usage: ProviderUsageDelta,
    expected: TerminationReason,
) -> None:
    charged = BudgetController.charge_decision(make_budget(), usage)

    assert BudgetController.after_decision(charged, now=STARTED_AT) is expected
    assert charged.steps_used == 1
    assert charged.tokens_used == usage.total_tokens
    assert charged.cost_used_usd == usage.estimated_cost_usd


def test_budget_clock_must_be_timezone_aware() -> None:
    naive = datetime(2026, 8, 5, 8)

    with pytest.raises(ValueError, match="timezone-aware"):
        BudgetController.before_decision(make_budget(), now=naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        BudgetController.after_decision(make_budget(), now=naive)
