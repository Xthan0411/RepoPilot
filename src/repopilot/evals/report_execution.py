"""Pure execution metrics over already verified, retained evaluation evidence.

This module neither loads artifacts nor executes models/tests. Policy refusals are
observations about calls, not adversarial interception or host-safety guarantees.
All rates retain their denominator; unavailable measurements remain ``None``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from typing import Literal

from pydantic import Field

from repopilot.enums import EvaluationAttemptStatus, TraceNode
from repopilot.evals.runner import EvaluationAttemptEvent
from repopilot.evals.single_shot import SingleShotTrace
from repopilot.schemas import FinalResult, StrictModel
from repopilot.trace import AgentTrace

AttemptStatus = Literal["completed", "technical_failure", "missing", "interrupted"]
_BUDGET_TERMINATIONS = frozenset({"max_steps", "token_limit", "cost_limit", "timeout"})
_PATCH_POLICY_CODES = frozenset({"path_denied", "protected_test_path"})
_REPOSITORY_POLICY_CODES = frozenset(
    {"invalid_path", "outside_root", "symlink_forbidden", "denied_path"}
)


class RateMetric(StrictModel):
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    rate: Decimal | None


class DistributionSummary(StrictModel):
    sample_count: int = Field(ge=0)
    mean: Decimal | None
    p50: Decimal | None
    p95: Decimal | None
    percentile_method: Literal["linear_r7_h=(n-1)*q"] = "linear_r7_h=(n-1)*q"


class SlotExecutionMetrics(StrictModel):
    arm_id: str
    task_id: str
    attempt_status: AttemptStatus
    task_success: bool | None
    issue_reproduced: bool | None
    patch_applied: bool | None
    tests_passed: bool | None
    termination_reason: str | None
    budget_terminated: bool | None
    # Model-directed Agent tools only. Single-shot evaluator actions are separate.
    tool_steps: int | None
    evaluator_steps: int | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    known_cost_usd: Decimal | None
    reserved_unknown_cost_usd: Decimal
    provider_latency_ms: int | None
    event_wall_ms: Decimal | None
    wall_interval_count: int
    provider_decision_calls: int | None
    provider_attempts_reported: int | None
    provider_retry_attempts_reported: int | None
    provider_failed_decisions: int | None
    tool_calls: int | None
    tool_errors: int | None
    tool_errors_by_code: dict[str, int]
    patch_attempts: int | None
    patch_rejections: int | None
    patch_rejections_by_code: dict[str, int]
    policy_rejected_tool_calls: int | None
    observed_policy_decisions: int | None
    sandbox_policy_denials: int | None
    replans: int | None
    validation_repair_count: int | None
    validation_repair_rescued: bool | None
    # The ordinary repair set is not an adversarial challenge or host audit.
    adversarial_challenge_count: None = None
    unsafe_host_execution_count: None = None
    secret_exposure_count: None = None
    safety_evaluation: Literal["not_measured_by_this_dataset"] = "not_measured_by_this_dataset"


class ExecutionAggregateMetrics(StrictModel):
    planned_slots: int
    completed_slots: int
    technical_failure_slots: int
    missing_slots: int
    interrupted_slots: int
    successful_slots: int
    completed_unsuccessful_slots: int
    success_rate_all_slots: RateMetric
    success_rate_completed: RateMetric
    reproduction_rate_completed: RateMetric
    patch_apply_rate_completed: RateMetric
    test_pass_rate_completed: RateMetric
    technical_failure_rate_all_slots: RateMetric
    budget_termination_rate_completed: RateMetric
    timeout_rate_completed: RateMetric
    cost_limit_rate_completed: RateMetric
    sandbox_error_rate_completed: RateMetric
    termination_counts: dict[str, int]
    usage_observed_slots: int
    usage_missing_slots: int
    known_cost_usd: Decimal
    reserved_unknown_cost_usd: Decimal
    accounted_cost_usd: Decimal
    known_cost_per_success_usd: Decimal | None
    accounted_cost_per_success_usd: Decimal | None
    cost_usd: DistributionSummary
    total_tokens: DistributionSummary
    tool_steps: DistributionSummary
    evaluator_steps: DistributionSummary
    provider_latency_ms: DistributionSummary
    event_wall_ms: DistributionSummary
    tool_calls: int
    tool_errors: int
    tool_error_rate: RateMetric
    tool_errors_by_code: dict[str, int]
    patch_attempts: int
    patch_rejections: int
    patch_rejection_rate: RateMetric
    patch_rejections_by_code: dict[str, int]
    policy_rejected_tool_calls: int
    policy_rejection_rate_of_tool_calls: RateMetric
    observed_policy_decisions: int
    sandbox_policy_denials: int
    sandbox_denial_rate_of_observed_decisions: RateMetric
    trace_observed_slots: int
    provider_decision_calls: int
    provider_attempts_reported: int
    provider_retry_attempts_reported: int
    provider_failed_decisions: int
    validation_repair_slots: int
    validation_repair_rescued_slots: int
    validation_repair_rescue_rate: RateMetric
    adversarial_interception_rate: None = None
    unsafe_host_execution_count: None = None
    secret_exposure_count: None = None
    safety_evaluation: Literal["not_measured_by_this_dataset"] = "not_measured_by_this_dataset"


def percentile_linear(values: Sequence[int | Decimal], quantile: Decimal) -> Decimal | None:
    """R7 linear interpolation: sorted sample index h=(n-1)*q; empty is unknown."""
    if not isinstance(quantile, Decimal) or not quantile.is_finite() or not 0 <= quantile <= 1:
        raise ValueError("quantile must be a finite Decimal in [0, 1]")
    samples = sorted(Decimal(value) for value in values)
    if any(not value.is_finite() for value in samples):
        raise ValueError("non-finite metric sample")
    if not samples:
        return None
    position = Decimal(len(samples) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(samples) - 1)
    return samples[lower] + (samples[upper] - samples[lower]) * (position - lower)


def _distribution(values: Sequence[int | Decimal]) -> DistributionSummary:
    return DistributionSummary(
        sample_count=len(values),
        mean=sum((Decimal(value) for value in values), Decimal(0)) / len(values)
        if values
        else None,
        p50=percentile_linear(values, Decimal("0.5")),
        p95=percentile_linear(values, Decimal("0.95")),
    )


def _rate(numerator: int, denominator: int) -> RateMetric:
    if not 0 <= numerator <= denominator:
        raise ValueError("rate numerator exceeds its denominator")
    return RateMetric(
        numerator=numerator,
        denominator=denominator,
        rate=Decimal(numerator) / denominator if denominator else None,
    )


def _event_wall(events: Sequence[EvaluationAttemptEvent]) -> tuple[Decimal | None, int]:
    """Sum closed execution intervals, excluding downtime between attempts.

    This is executor wall time, including preparation/evaluation/cleanup and
    result persistence, not Provider latency or total experiment calendar time.
    An open interval makes total wall time unknown; closed count is still shown.
    """
    starts = {}
    total = Decimal(0)
    count = 0
    previous_sequence = -1
    for event in events:
        if event.sequence <= previous_sequence:
            raise ValueError("events must be ordered and unique")
        previous_sequence = event.sequence
        key = (event.attempt_id, event.execution_index, event.execution_run_id)
        if event.status is EvaluationAttemptStatus.STARTED:
            if key in starts:
                raise ValueError("duplicate execution start")
            starts[key] = event.recorded_at
        else:
            start = starts.pop(key, None)
            if start is None or event.recorded_at < start:
                raise ValueError("terminal event lacks a valid matching start")
            delta = event.recorded_at - start
            micros = (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
            total += Decimal(micros) / 1000
            count += 1
    return (total if count and not starts else None), count


def extract_execution_metrics(
    *,
    arm_id: str,
    task_id: str,
    attempt_status: AttemptStatus,
    result: FinalResult | None = None,
    trace: AgentTrace | SingleShotTrace | None = None,
    events: Sequence[EvaluationAttemptEvent] = (),
    reserved_unknown_cost_usd: Decimal = Decimal(0),
) -> SlotExecutionMetrics:
    """Extract metrics after caller verification of artifact/history integrity.

    No unknown technical-failure usage is invented from an absent result. The
    caller supplies any conservative reserve. Responses' retry counts exclude
    attempts whose failed Provider decision did not retain attempt accounting.
    """
    if (
        not isinstance(reserved_unknown_cost_usd, Decimal)
        or not reserved_unknown_cost_usd.is_finite()
        or reserved_unknown_cost_usd < 0
    ):
        raise ValueError("invalid unknown-cost reserve")
    if (attempt_status == "completed") != (result is not None and trace is not None):
        raise ValueError("only completed slots require result and trace")
    if attempt_status != "completed" and (result is not None or trace is not None):
        raise ValueError("non-completed usage cannot be inferred from partial evidence")
    if any(event.task_id != task_id for event in events):
        raise ValueError("event task identity mismatch")
    if result is not None and trace is not None:
        if result.task_id != task_id or trace.task_id != task_id or result.run_id != trace.run_id:
            raise ValueError("result/trace identity mismatch")
        if events and (
            events[-1].status is not EvaluationAttemptStatus.COMPLETED
            or events[-1].execution_run_id != result.run_id
        ):
            raise ValueError("completed execution event mismatch")
    wall, interval_count = _event_wall(events)
    tool_errors: Counter[str] = Counter()
    patch_rejections: Counter[str] = Counter()
    policy_rejections = set()
    observations: dict[str, str] = {}
    calls = patch_calls = decisions = attempts = retry_attempts = provider_errors = 0
    replans = validation_repairs = 0

    def observe(payload: object) -> bool:
        if not isinstance(payload, dict) or "policy_decision" not in payload:
            return False
        call_id, decision = str(payload["call_id"]), str(payload["policy_decision"])
        previous = observations.setdefault(call_id, decision)
        if previous != decision:
            raise ValueError("conflicting observation policy decision")
        return decision == "denied"

    if isinstance(trace, AgentTrace):
        replans = trace.final_state.replan_count
        for event in trace.events:
            decisions += int(event.provider_request is not None)
            provider_errors += int(event.provider_error_code is not None)
            if event.provider_response is not None:
                attempts += event.provider_response.attempts
                retry_attempts += event.provider_response.attempts - 1
            if (
                event.state_after.replan_count > event.state_before.replan_count
                and event.state_before.active_step_id == "validate"
                and event.state_after.active_step_id == "patch"
            ):
                validation_repairs += (
                    event.state_after.replan_count - event.state_before.replan_count
                )
            if event.node is not TraceNode.EXECUTE_TOOL:
                continue
            tool = event.tool_result
            if tool is None:
                raise ValueError("tool event missing result")
            calls += 1
            payload = tool.payload or {}
            denied = observe(payload.get("observation"))
            if tool.error_code is not None:
                tool_errors[str(tool.error_code)] += 1
            denied = denied or str(tool.repository_error_code) in _REPOSITORY_POLICY_CODES
            if tool.tool_name == "apply_patch":
                patch_calls += 1
                if payload.get("status") == "rejected":
                    code = str(payload.get("error_code") or "unspecified")
                    patch_rejections[code] += 1
                    denied = denied or code in _PATCH_POLICY_CODES
            if denied:
                policy_rejections.add(str(tool.call_id))
    elif isinstance(trace, SingleShotTrace):
        decisions = attempts = trace.provider_requests
        retry_attempts = trace.provider_retries
        provider_errors = int(trace.provider_failure is not None)
        for observation in trace.setup_observations:
            observe(observation.model_dump(mode="json"))
        for test in (trace.baseline, trace.validation):
            if test is not None:
                observe(test.observation.model_dump(mode="json"))
        if trace.patch is not None:
            patch_calls = 1  # Evaluator patch application, not an Agent tool call.
            if trace.patch.observation is not None:
                observe(trace.patch.observation.model_dump(mode="json"))
            if trace.patch.status == "rejected":
                patch_rejections[str(trace.patch.error_code or "unspecified")] += 1
    metrics = result.task_metrics if result is not None else None
    usage = result.usage if result is not None else None
    reason = str(result.termination_reason) if result is not None else None
    known = trace is not None
    return SlotExecutionMetrics(
        arm_id=arm_id,
        task_id=task_id,
        attempt_status=attempt_status,
        task_success=metrics.task_success if metrics else None,
        issue_reproduced=metrics.issue_reproduced if metrics else None,
        patch_applied=metrics.patch_applied if metrics else None,
        tests_passed=metrics.tests_passed if metrics else None,
        termination_reason=reason,
        budget_terminated=reason in _BUDGET_TERMINATIONS if reason else None,
        tool_steps=(0 if isinstance(trace, SingleShotTrace) else metrics.tool_steps)
        if metrics
        else None,
        evaluator_steps=metrics.tool_steps
        if metrics and isinstance(trace, SingleShotTrace)
        else None,
        input_tokens=usage.input_tokens if usage else None,
        output_tokens=usage.output_tokens if usage else None,
        total_tokens=usage.total_tokens if usage else None,
        known_cost_usd=usage.estimated_cost_usd if usage else None,
        reserved_unknown_cost_usd=reserved_unknown_cost_usd,
        provider_latency_ms=usage.latency_ms if usage else None,
        event_wall_ms=wall,
        wall_interval_count=interval_count,
        provider_decision_calls=decisions if known else None,
        provider_attempts_reported=attempts if known else None,
        provider_retry_attempts_reported=retry_attempts if known else None,
        provider_failed_decisions=provider_errors if known else None,
        tool_calls=calls if known else None,
        tool_errors=sum(tool_errors.values()) if known else None,
        tool_errors_by_code=dict(sorted(tool_errors.items())),
        patch_attempts=patch_calls if known else None,
        patch_rejections=sum(patch_rejections.values()) if known else None,
        patch_rejections_by_code=dict(sorted(patch_rejections.items())),
        policy_rejected_tool_calls=len(policy_rejections) if known else None,
        observed_policy_decisions=len(observations) if known else None,
        sandbox_policy_denials=sum(value == "denied" for value in observations.values())
        if known
        else None,
        replans=replans if known else None,
        validation_repair_count=validation_repairs if known else None,
        validation_repair_rescued=bool(validation_repairs and metrics and metrics.task_success)
        if known
        else None,
    )


def aggregate_execution_metrics(rows: Sequence[SlotExecutionMetrics]) -> ExecutionAggregateMetrics:
    """Aggregate supplied planned slots, retaining failures/missing rows in the total."""
    if len({(row.arm_id, row.task_id) for row in rows}) != len(rows):
        raise ValueError("duplicate arm/task slot")
    count = len(rows)
    completed = sum(row.attempt_status == "completed" for row in rows)
    technical = sum(row.attempt_status == "technical_failure" for row in rows)
    successes = sum(row.task_success is True for row in rows)
    reasons = Counter(row.termination_reason for row in rows if row.termination_reason is not None)
    tool_errors: Counter[str] = Counter()
    patch_rejections: Counter[str] = Counter()
    for row in rows:
        tool_errors.update(row.tool_errors_by_code)
        patch_rejections.update(row.patch_rejections_by_code)
    costs = [row.known_cost_usd for row in rows if row.known_cost_usd is not None]
    known_cost = sum(costs, Decimal(0))
    reserved = sum((row.reserved_unknown_cost_usd for row in rows), Decimal(0))
    calls = sum(row.tool_calls or 0 for row in rows)
    patch_calls = sum(row.patch_attempts or 0 for row in rows)
    policy_rejects = sum(row.policy_rejected_tool_calls or 0 for row in rows)
    policy_count = sum(row.observed_policy_decisions or 0 for row in rows)
    policy_denied = sum(row.sandbox_policy_denials or 0 for row in rows)
    repairs = sum(bool(row.validation_repair_count) for row in rows)
    rescued = sum(row.validation_repair_rescued is True for row in rows)
    return ExecutionAggregateMetrics(
        planned_slots=count,
        completed_slots=completed,
        technical_failure_slots=technical,
        missing_slots=sum(row.attempt_status == "missing" for row in rows),
        interrupted_slots=sum(row.attempt_status == "interrupted" for row in rows),
        successful_slots=successes,
        completed_unsuccessful_slots=completed - successes,
        success_rate_all_slots=_rate(successes, count),
        success_rate_completed=_rate(successes, completed),
        reproduction_rate_completed=_rate(
            sum(row.issue_reproduced is True for row in rows), completed
        ),
        patch_apply_rate_completed=_rate(sum(row.patch_applied is True for row in rows), completed),
        test_pass_rate_completed=_rate(sum(row.tests_passed is True for row in rows), completed),
        technical_failure_rate_all_slots=_rate(technical, count),
        budget_termination_rate_completed=_rate(
            sum(row.budget_terminated is True for row in rows), completed
        ),
        timeout_rate_completed=_rate(reasons["timeout"], completed),
        cost_limit_rate_completed=_rate(reasons["cost_limit"], completed),
        sandbox_error_rate_completed=_rate(reasons["sandbox_error"], completed),
        termination_counts=dict(sorted(reasons.items())),
        usage_observed_slots=len(costs),
        usage_missing_slots=count - len(costs),
        known_cost_usd=known_cost,
        reserved_unknown_cost_usd=reserved,
        accounted_cost_usd=known_cost + reserved,
        known_cost_per_success_usd=known_cost / successes if successes else None,
        accounted_cost_per_success_usd=(known_cost + reserved) / successes if successes else None,
        cost_usd=_distribution(costs),
        total_tokens=_distribution(
            [row.total_tokens for row in rows if row.total_tokens is not None]
        ),
        tool_steps=_distribution([row.tool_steps for row in rows if row.tool_steps is not None]),
        evaluator_steps=_distribution(
            [row.evaluator_steps for row in rows if row.evaluator_steps is not None]
        ),
        provider_latency_ms=_distribution(
            [row.provider_latency_ms for row in rows if row.provider_latency_ms is not None]
        ),
        event_wall_ms=_distribution(
            [row.event_wall_ms for row in rows if row.event_wall_ms is not None]
        ),
        tool_calls=calls,
        tool_errors=sum(tool_errors.values()),
        tool_error_rate=_rate(sum(tool_errors.values()), calls),
        tool_errors_by_code=dict(sorted(tool_errors.items())),
        patch_attempts=patch_calls,
        patch_rejections=sum(patch_rejections.values()),
        patch_rejection_rate=_rate(sum(patch_rejections.values()), patch_calls),
        patch_rejections_by_code=dict(sorted(patch_rejections.items())),
        policy_rejected_tool_calls=policy_rejects,
        policy_rejection_rate_of_tool_calls=_rate(policy_rejects, calls),
        observed_policy_decisions=policy_count,
        sandbox_policy_denials=policy_denied,
        sandbox_denial_rate_of_observed_decisions=_rate(policy_denied, policy_count),
        trace_observed_slots=sum(row.tool_calls is not None for row in rows),
        provider_decision_calls=sum(row.provider_decision_calls or 0 for row in rows),
        provider_attempts_reported=sum(row.provider_attempts_reported or 0 for row in rows),
        provider_retry_attempts_reported=sum(
            row.provider_retry_attempts_reported or 0 for row in rows
        ),
        provider_failed_decisions=sum(row.provider_failed_decisions or 0 for row in rows),
        validation_repair_slots=repairs,
        validation_repair_rescued_slots=rescued,
        validation_repair_rescue_rate=_rate(rescued, repairs),
    )
