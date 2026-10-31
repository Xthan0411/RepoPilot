"""Offline execution reporting: explicit denominators, latency units and safety limits."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid5

import pytest
from test_repair_agent import (
    RevisionApplyTool,
    ValidationSequenceTool,
    _run_script,
    _validation_repair_responses,
    registry,
    verified_source,
)

from repopilot.enums import (
    ErrorType,
    EvaluationAttemptStatus,
    EvaluationRunnerErrorCode,
    RepositoryErrorCode,
    RunStatus,
    TerminationReason,
    ToolErrorCode,
    ToolResultStatus,
)
from repopilot.evals.report_execution import (
    aggregate_execution_metrics,
    extract_execution_metrics,
    percentile_linear,
)
from repopilot.evals.runner import EvaluationAttemptEvent, _canonical_sha256
from repopilot.evals.single_shot import SingleShotTrace
from repopilot.schemas import ToolExecutionResult


@pytest.fixture
def evidence():
    _, trace, result, _ = verified_source()
    responses = [event.provider_response for event in trace.events if event.provider_response]
    usage = result.usage.model_copy(
        update={
            "input_tokens": sum(response.usage.input_tokens for response in responses),
            "output_tokens": sum(response.usage.output_tokens for response in responses),
            "total_tokens": sum(response.usage.total_tokens for response in responses),
            "estimated_cost_usd": Decimal("0.10"),
            "latency_ms": 1234,
        }
    )
    return trace, result.model_copy(update={"usage": usage})


def _event(result, sequence, status, at, *, execution_index=0):
    run_id = result.run_id if execution_index == 0 else uuid5(result.run_id, str(execution_index))
    refs = {
        key: {
            "relative_path": name,
            "sha256": "a" * 64,
            "size_bytes": 100,
            "media_type": "application/json",
        }
        for key, name in (
            ("result_ref", "result.json"),
            ("trace_ref", "trace.json"),
            ("artifact_manifest_ref", "artifacts.json"),
        )
    }
    payload = {
        "schema_version": 1,
        "sequence": sequence,
        "attempt_id": "attempt_fixture",
        "task_id": result.task_id,
        "repetition": 0,
        "execution_index": execution_index,
        "execution_run_id": run_id,
        "thread_id": "fixture",
        "output_path": f"executions/attempt_fixture/{execution_index:04d}",
        "status": status,
        "recorded_at": at,
        "evidence": {
            "task_id": result.task_id,
            "run_id": run_id,
            "status": result.status,
            "termination_reason": result.termination_reason,
            "task_success": result.task_metrics.task_success,
            "usage": result.usage,
            **refs,
        }
        if status == EvaluationAttemptStatus.COMPLETED
        else None,
        "failure_code": EvaluationRunnerErrorCode.EXECUTION_FAILURE
        if status == EvaluationAttemptStatus.FAILED
        else None,
    }
    return EvaluationAttemptEvent.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def _row(evidence, **changes):
    trace, result = evidence
    return extract_execution_metrics(
        **{
            "arm_id": "fixture-arm",
            "task_id": result.task_id,
            "attempt_status": "completed",
            "result": result,
            "trace": trace,
            **changes,
        }
    )


def test_provider_time_is_not_executor_wall_time(evidence):
    _, result = evidence
    start = datetime(2026, 1, 1, tzinfo=UTC)
    events = (
        _event(result, 0, EvaluationAttemptStatus.STARTED, start),
        _event(result, 1, EvaluationAttemptStatus.COMPLETED, start + timedelta(seconds=25)),
    )
    row = _row(evidence, events=events)
    assert row.provider_latency_ms == 1234
    assert row.event_wall_ms == Decimal(25000)
    assert row.wall_interval_count == 1
    assert row.tool_calls == row.tool_steps == 5
    assert row.patch_attempts == 1
    assert row.provider_decision_calls == 6
    assert row.provider_retry_attempts_reported == 0


def test_denominators_keep_technical_missing_and_interrupted_slots(evidence):
    success = _row(evidence)
    rows = [success]
    for status in ("technical_failure", "missing", "interrupted"):
        rows.append(
            extract_execution_metrics(
                arm_id="fixture-arm",
                task_id=f"fixture-{status}",
                attempt_status=status,
                reserved_unknown_cost_usd=Decimal(1)
                if status == "technical_failure"
                else Decimal(0),
            )
        )
    report = aggregate_execution_metrics(rows)
    assert report.planned_slots == 4
    assert report.completed_slots == report.successful_slots == 1
    assert report.technical_failure_slots == report.missing_slots == report.interrupted_slots == 1
    assert report.success_rate_all_slots.rate == Decimal("0.25")
    assert report.success_rate_completed.rate == Decimal(1)
    assert report.usage_observed_slots == 1 and report.usage_missing_slots == 3
    assert report.known_cost_usd == Decimal("0.10")
    assert report.accounted_cost_usd == Decimal("1.10")
    assert report.cost_usd.sample_count == 1
    assert report.event_wall_ms.sample_count == 0
    assert report.event_wall_ms.mean is None
    assert rows[1].known_cost_usd is None


def test_failed_completions_contribute_cost_latency_and_rate_denominators(evidence):
    first = _row(evidence)
    _, result = evidence
    failed = result.model_copy(
        update={
            "status": RunStatus.STOPPED,
            "termination_reason": TerminationReason.TOKEN_LIMIT,
            "task_metrics": result.task_metrics.model_copy(update={"task_success": False}),
            "usage": result.usage.model_copy(update={"estimated_cost_usd": Decimal("0.20")}),
        }
    )
    second = _row(evidence, arm_id="other-arm", result=failed)
    report = aggregate_execution_metrics([first, second])
    assert report.known_cost_usd == Decimal("0.30")
    assert report.known_cost_per_success_usd == Decimal("0.30")
    assert report.cost_usd.mean == Decimal("0.15")
    assert report.success_rate_completed.denominator == 2
    assert report.success_rate_completed.rate == Decimal("0.5")
    assert report.completed_unsuccessful_slots == 1
    assert report.budget_termination_rate_completed.rate == Decimal("0.5")
    assert report.termination_counts == {"success": 1, "token_limit": 1}


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [
        ([], "0.5", None),
        ([7], "0.95", "7"),
        ([30, 0, 20, 10], "0.5", "15"),
        ([30, 0, 20, 10], "0.95", "28.5"),
        ([2, 8], "0", "2"),
        ([2, 8], "1", "8"),
    ],
)
def test_percentiles_use_explicit_linear_r7(values, q, expected):
    assert percentile_linear(values, Decimal(q)) == (Decimal(expected) if expected else None)


@pytest.mark.parametrize("q", [Decimal("-0.1"), Decimal("1.1"), Decimal("NaN"), 0.5])
def test_invalid_quantiles_are_rejected(q):
    with pytest.raises(ValueError):
        percentile_linear([1, 2], q)


def test_zero_denominators_are_unknown_not_perfect_or_zero_performance():
    report = aggregate_execution_metrics([])
    assert report.success_rate_all_slots.denominator == 0
    assert report.success_rate_all_slots.rate is None
    assert report.known_cost_per_success_usd is None
    assert report.tool_error_rate.rate is None
    assert report.adversarial_interception_rate is None
    assert report.unsafe_host_execution_count is None
    assert report.secret_exposure_count is None


def test_event_wall_sums_closed_intervals_excluding_resume_downtime(evidence):
    _, result = evidence
    start = datetime(2026, 1, 1, tzinfo=UTC)
    events = (
        _event(result, 0, EvaluationAttemptStatus.STARTED, start),
        _event(result, 1, EvaluationAttemptStatus.INTERRUPTED, start + timedelta(seconds=2)),
        _event(
            result,
            2,
            EvaluationAttemptStatus.STARTED,
            start + timedelta(hours=1),
            execution_index=1,
        ),
        _event(
            result,
            3,
            EvaluationAttemptStatus.FAILED,
            start + timedelta(hours=1, seconds=3),
            execution_index=1,
        ),
    )
    row = extract_execution_metrics(
        arm_id="fixture-arm",
        task_id=result.task_id,
        attempt_status="technical_failure",
        events=events,
    )
    assert row.event_wall_ms == Decimal(5000)
    assert row.wall_interval_count == 2
    assert row.provider_latency_ms is None
    open_row = extract_execution_metrics(
        arm_id="fixture-arm",
        task_id=result.task_id,
        attempt_status="interrupted",
        events=events[:-1],
    )
    assert open_row.event_wall_ms is None
    assert open_row.wall_interval_count == 1


def test_event_order_negative_duration_and_identity_mismatches_fail_closed(evidence):
    _, result = evidence
    start = datetime(2026, 1, 1, tzinfo=UTC)
    first = _event(result, 0, EvaluationAttemptStatus.STARTED, start)
    final = _event(result, 1, EvaluationAttemptStatus.COMPLETED, start - timedelta(seconds=1))
    with pytest.raises(ValueError, match="valid matching start"):
        _row(evidence, events=(first, final))
    with pytest.raises(ValueError):
        _row(evidence, events=(final, first))
    with pytest.raises(ValueError, match="identity"):
        _row(evidence, task_id="different-task")


@pytest.mark.parametrize("reserve", [Decimal(-1), Decimal("NaN"), Decimal("Infinity"), 1.0])
def test_unknown_cost_requires_nonnegative_finite_decimal(reserve):
    with pytest.raises(ValueError, match="reserve"):
        extract_execution_metrics(
            arm_id="a",
            task_id="t",
            attempt_status="technical_failure",
            reserved_unknown_cost_usd=reserve,
        )


def test_unknown_results_are_not_reported_as_zero_usage(evidence):
    trace, result = evidence
    with pytest.raises(ValueError, match="require result and trace"):
        _row(evidence, trace=None)
    with pytest.raises(ValueError):
        extract_execution_metrics(
            arm_id="a",
            task_id=result.task_id,
            attempt_status="technical_failure",
            result=result,
            trace=trace,
        )
    row = _row(evidence)
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_execution_metrics([row, row])


@pytest.mark.parametrize(
    "code,expected_policy", [("apply_conflict", 0), ("protected_test_path", 1)]
)
def test_invalid_patch_is_not_automatically_a_safety_attack(evidence, code, expected_policy):
    trace, _ = evidence
    events = []
    for event in trace.events:
        tool = event.tool_result
        if tool is not None and tool.tool_name == "apply_patch":
            tool = tool.model_copy(update={"payload": {"status": "rejected", "error_code": code}})
            event = event.model_copy(update={"tool_result": tool})
        events.append(event)
    row = _row(evidence, trace=trace.model_copy(update={"events": tuple(events)}))
    assert row.patch_rejections_by_code == {code: 1}
    assert row.policy_rejected_tool_calls == expected_policy
    assert row.tool_errors == 0  # A structured patch refusal is not a Registry failure.
    report = aggregate_execution_metrics([row])
    assert report.policy_rejection_rate_of_tool_calls.denominator == row.tool_calls
    assert report.adversarial_interception_rate is None


def test_repository_refusal_counts_a_tool_error_and_policy_rejection_once(evidence):
    trace, _ = evidence
    events = []
    for event in trace.events:
        tool = event.tool_result
        if tool is not None and tool.tool_name == "read_file":
            denied = ToolExecutionResult(
                call_id=tool.call_id,
                tool_name=tool.tool_name,
                status=ToolResultStatus.ERROR,
                error_type=ErrorType.PATH_VIOLATION,
                error_code=ToolErrorCode.REPOSITORY_ERROR,
                repository_error_code=RepositoryErrorCode.OUTSIDE_ROOT,
                message="path rejected",
            )
            event = event.model_copy(update={"tool_result": denied})
        events.append(event)
    row = _row(evidence, trace=trace.model_copy(update={"events": tuple(events)}))
    assert row.tool_errors_by_code == {"repository_error": 1}
    assert row.policy_rejected_tool_calls == 1
    assert row.sandbox_policy_denials == 0  # Rejected before Sandbox invocation.


def test_validation_repair_is_not_confused_with_search_replanning(evidence):
    _, _, trace = _run_script(
        _validation_repair_responses(),
        tools=registry(test_tool=ValidationSequenceTool(), patch_tool=RevisionApplyTool()),
    )
    row = _row(evidence, trace=trace)
    assert row.replans == row.validation_repair_count == 1
    assert row.validation_repair_rescued is True
    assert row.patch_attempts == 2
    assert aggregate_execution_metrics([row]).validation_repair_rescue_rate.rate == Decimal(1)


def test_one_shot_evaluator_operations_are_not_agent_tools(evidence):
    _, result = evidence
    # Only extractor-read fields are relevant here; integrity validation belongs
    # to the orchestrator and has its own full one-shot executor tests.
    one_shot = SingleShotTrace.model_construct(
        task_id=result.task_id,
        run_id=result.run_id,
        provider_requests=1,
        provider_retries=0,
        provider_failure=None,
        setup_observations=(),
        baseline=None,
        validation=None,
        patch=None,
    )
    row = _row(evidence, trace=one_shot)
    assert row.provider_decision_calls == row.provider_attempts_reported == 1
    assert row.tool_calls == row.tool_steps == 0
    assert row.evaluator_steps == result.task_metrics.tool_steps
    aggregate = aggregate_execution_metrics([row])
    assert aggregate.tool_steps.mean == 0
    assert aggregate.evaluator_steps.mean == result.task_metrics.tool_steps
    assert row.validation_repair_count == 0
    assert row.safety_evaluation == "not_measured_by_this_dataset"


def test_provider_transport_retries_remain_separate_from_graph_replanning(evidence):
    trace, _ = evidence
    events = list(trace.events)
    index = next(i for i, event in enumerate(events) if event.provider_response)
    response = events[index].provider_response.model_copy(update={"attempts": 3})
    events[index] = events[index].model_copy(update={"provider_response": response})
    row = _row(evidence, trace=trace.model_copy(update={"events": tuple(events)}))
    assert row.provider_retry_attempts_reported == 2
    assert row.provider_attempts_reported == row.provider_decision_calls + 2
    assert row.replans == 0
