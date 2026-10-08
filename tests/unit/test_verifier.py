"""Rule matrix and anti-self-report tests for the deterministic Verifier."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from repopilot.enums import (
    ErrorCategory,
    ErrorType,
    PatchStatus,
    ReproductionStatus,
    RunStatus,
    TerminationReason,
    VerifierErrorCode,
)
from repopilot.enums import TestRunCategory as RepoTestRunCategory
from repopilot.enums import TestStatus as RepoTestStatus
from repopilot.schemas import (
    ArtifactRef,
    LocalizationMetrics,
    PatchState,
    ReproductionResult,
    TaskMetrics,
    UsageMetrics,
)
from repopilot.schemas import TestRunResult as RepoTestRunResult
from repopilot.verifier import (
    FinalResultContext,
    RuleBasedVerifier,
    VerificationEvidence,
    build_final_result,
)

BASELINE_CALL_ID = UUID("11111111-1111-4111-8111-111111111111")
VALIDATION_CALL_ID = UUID("22222222-2222-4222-8222-222222222222")
RUN_ID = UUID("33333333-3333-4333-8333-333333333333")
COMMAND_SHA = "a" * 64


def artifact(name: str, *, size_bytes: int = 20) -> ArtifactRef:
    return ArtifactRef(
        relative_path=f"artifacts/sandbox/{name}",
        sha256="b" * 64,
        size_bytes=size_bytes,
        media_type="text/plain; charset=utf-8",
    )


def completed_test_run(
    *,
    status: RepoTestStatus,
    category: RepoTestRunCategory,
    call_id: UUID = VALIDATION_CALL_ID,
    command_sha256: str = COMMAND_SHA,
    command_index: int = 0,
    with_artifact: bool = True,
    truncated: bool = False,
) -> RepoTestRunResult:
    values: dict[str, Any] = {
        "observation_call_id": call_id,
        "command_index": command_index,
        "command_sha256": command_sha256,
        "status": status,
        "category": category,
        "exit_code": 0,
        "duration_ms": 10,
        "tests_run": 2,
        "failures": 0,
        "errors": 0,
        "failing_tests": (),
        "stdout_ref": artifact(f"{call_id}.stdout.txt") if with_artifact else None,
        "stderr_ref": None,
        "truncated": truncated,
        "error_type": None,
        "summary": f"{category.value} test evidence",
    }
    if status is RepoTestStatus.FAILED:
        values.update(
            exit_code=1,
            failures=1 if category is RepoTestRunCategory.ASSERTION_FAILURE else 0,
            errors=1 if category is RepoTestRunCategory.TEST_ERROR else 0,
            failing_tests=("tests.test_bug.TestBug.test_regression",),
        )
    elif status is RepoTestStatus.ERROR:
        values.update(
            exit_code=2,
            tests_run=0,
            error_type={
                RepoTestRunCategory.TIMEOUT: ErrorType.TOOL_TIMEOUT,
                RepoTestRunCategory.SANDBOX_ERROR: ErrorType.SANDBOX_ERROR,
                RepoTestRunCategory.POLICY_DENIED: ErrorType.POLICY_DENIED,
                RepoTestRunCategory.IMPORT_ERROR: ErrorType.TEST_ERROR,
                RepoTestRunCategory.NO_TESTS_COLLECTED: ErrorType.TEST_ERROR,
                RepoTestRunCategory.OUTPUT_TRUNCATED: ErrorType.TEST_ERROR,
                RepoTestRunCategory.UNKNOWN_NONZERO: ErrorType.TEST_ERROR,
                RepoTestRunCategory.EXECUTION_ERROR: ErrorType.TOOL_EXECUTION_ERROR,
            }[category],
        )
        if category in {
            RepoTestRunCategory.TIMEOUT,
            RepoTestRunCategory.POLICY_DENIED,
        }:
            values["exit_code"] = None
        if category is RepoTestRunCategory.OUTPUT_TRUNCATED:
            values["truncated"] = True
    return RepoTestRunResult.model_validate(values)


def reproduced(
    *,
    with_artifact: bool = True,
) -> ReproductionResult:
    test_run = completed_test_run(
        status=RepoTestStatus.FAILED,
        category=RepoTestRunCategory.ASSERTION_FAILURE,
        call_id=BASELINE_CALL_ID,
        with_artifact=with_artifact,
    )
    return ReproductionResult(
        status=ReproductionStatus.REPRODUCED,
        observation_call_id=test_run.observation_call_id,
        failing_tests=test_run.failing_tests,
        summary=test_run.summary,
        test_run=test_run,
    )


def applied_patch(*, revision: int = 1, size_bytes: int = 80) -> PatchState:
    return PatchState(
        status=PatchStatus.APPLIED,
        diff_ref=artifact("candidate.patch.diff", size_bytes=size_bytes),
        files_changed=("pkg/module.py",),
        revision=revision,
    )


def passing_validation(**updates: Any) -> RepoTestRunResult:
    values = completed_test_run(
        status=RepoTestStatus.PASSED,
        category=RepoTestRunCategory.PASSED,
    ).model_dump()
    values.update(updates)
    return RepoTestRunResult.model_validate(values)


def evidence(
    *,
    reproduction_result: ReproductionResult | None = None,
    patch_state: PatchState | None = None,
    validation_test_run: RepoTestRunResult | None = None,
) -> VerificationEvidence:
    return VerificationEvidence(
        reproduction_result=reproduction_result or reproduced(),
        patch_state=patch_state or applied_patch(),
        validation_test_run=validation_test_run or passing_validation(),
        tool_steps=7,
    )


def context() -> FinalResultContext:
    return FinalResultContext(
        task_id="verifier-task",
        run_id=RUN_ID,
        localization_metrics=LocalizationMetrics(
            file_recall_at_1=1,
            file_recall_at_3=1,
            file_recall_at_5=1,
            mean_reciprocal_rank=1,
            first_relevant_file_rank=1,
            indexing_latency_ms=2,
            retrieval_latency_ms=1,
        ),
        usage=UsageMetrics(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            estimated_cost_usd=Decimal("0"),
            latency_ms=20,
        ),
        trace_path="runs/verifier-task/trace.json",
    )


def test_only_complete_independent_evidence_produces_success() -> None:
    decision = RuleBasedVerifier().verify(evidence())
    result = build_final_result(decision, context())

    assert decision.status is RunStatus.SUCCESS
    assert decision.termination_reason is TerminationReason.SUCCESS
    assert decision.error_category is None
    assert decision.verifier_error_code is None
    assert decision.validation_result.test_run == passing_validation()
    assert decision.task_metrics == TaskMetrics(
        issue_reproduced=True,
        patch_applied=True,
        tests_passed=True,
        task_success=True,
        tool_steps=7,
    )
    assert result.patch_path == "artifacts/sandbox/candidate.patch.diff"
    assert result.model_validate_json(result.model_dump_json()) == result


def test_model_self_report_is_neither_accepted_input_nor_success_authority() -> None:
    failed_validation = completed_test_run(
        status=RepoTestStatus.FAILED,
        category=RepoTestRunCategory.ASSERTION_FAILURE,
    )
    payload = evidence(validation_test_run=failed_validation).model_dump()
    payload["model_claimed_success"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        VerificationEvidence.model_validate(payload)

    decision = RuleBasedVerifier().verify(
        evidence(validation_test_run=failed_validation),
    )
    assert decision.status is RunStatus.FAILED
    assert decision.termination_reason is TerminationReason.TESTS_FAILED
    assert decision.verifier_error_code is VerifierErrorCode.TESTS_FAILED
    assert decision.task_metrics.task_success is False


@pytest.mark.parametrize(
    ("reproduction_result", "status", "termination", "code"),
    [
        (
            ReproductionResult(status=ReproductionStatus.NOT_ATTEMPTED),
            RunStatus.STOPPED,
            TerminationReason.TOOL_ERROR,
            VerifierErrorCode.REPRODUCTION_NOT_ATTEMPTED,
        ),
        (
            ReproductionResult(
                status=ReproductionStatus.NOT_REPRODUCED,
                observation_call_id=BASELINE_CALL_ID,
                test_run=completed_test_run(
                    status=RepoTestStatus.PASSED,
                    category=RepoTestRunCategory.PASSED,
                    call_id=BASELINE_CALL_ID,
                ),
            ),
            RunStatus.FAILED,
            TerminationReason.TESTS_FAILED,
            VerifierErrorCode.ISSUE_NOT_REPRODUCED,
        ),
        (
            ReproductionResult(
                status=ReproductionStatus.ERROR,
                observation_call_id=BASELINE_CALL_ID,
                test_run=completed_test_run(
                    status=RepoTestStatus.ERROR,
                    category=RepoTestRunCategory.TIMEOUT,
                    call_id=BASELINE_CALL_ID,
                ),
            ),
            RunStatus.STOPPED,
            TerminationReason.TIMEOUT,
            VerifierErrorCode.REPRODUCTION_ERROR,
        ),
        (
            reproduced(with_artifact=False),
            RunStatus.STOPPED,
            TerminationReason.TOOL_ERROR,
            VerifierErrorCode.REPRODUCTION_EVIDENCE_INVALID,
        ),
    ],
)
def test_baseline_gate_fails_closed(
    reproduction_result: ReproductionResult,
    status: RunStatus,
    termination: TerminationReason,
    code: VerifierErrorCode,
) -> None:
    decision = RuleBasedVerifier().verify(evidence(reproduction_result=reproduction_result))

    assert decision.status is status
    assert decision.termination_reason is termination
    assert decision.verifier_error_code is code
    assert decision.task_metrics.task_success is False


@pytest.mark.parametrize(
    ("patch_state", "code"),
    [
        (PatchState(), VerifierErrorCode.PATCH_NOT_APPLIED),
        (
            PatchState(
                status=PatchStatus.REJECTED,
                diff_ref=artifact("rejected.patch.diff"),
                files_changed=("pkg/module.py",),
            ),
            VerifierErrorCode.PATCH_NOT_APPLIED,
        ),
        (applied_patch(revision=0), VerifierErrorCode.PATCH_EVIDENCE_INVALID),
        (
            applied_patch(size_bytes=0),
            VerifierErrorCode.PATCH_EVIDENCE_INVALID,
        ),
    ],
)
def test_patch_gate_requires_nonempty_applied_artifact(
    patch_state: PatchState,
    code: VerifierErrorCode,
) -> None:
    decision = RuleBasedVerifier().verify(evidence(patch_state=patch_state))

    assert decision.status is RunStatus.FAILED
    assert decision.termination_reason is TerminationReason.INVALID_PATCH
    assert decision.error_category is ErrorCategory.INVALID_PATCH
    assert decision.verifier_error_code is code
    assert decision.task_metrics.task_success is False


def test_validation_must_run_same_command_in_a_new_observation() -> None:
    not_run = RuleBasedVerifier().verify(
        VerificationEvidence(
            reproduction_result=reproduced(),
            patch_state=applied_patch(),
            validation_test_run=None,
            tool_steps=2,
        )
    )
    mismatched = RuleBasedVerifier().verify(
        evidence(validation_test_run=passing_validation(command_sha256="c" * 64))
    )
    reused = RuleBasedVerifier().verify(
        evidence(validation_test_run=passing_validation(observation_call_id=BASELINE_CALL_ID))
    )

    assert not_run.verifier_error_code is VerifierErrorCode.VALIDATION_NOT_RUN
    assert mismatched.verifier_error_code is VerifierErrorCode.TEST_COMMAND_MISMATCH
    assert reused.verifier_error_code is VerifierErrorCode.OBSERVATION_REUSED
    assert {not_run.status, mismatched.status, reused.status} == {RunStatus.STOPPED}


def test_ambiguous_passing_test_cannot_authorize_success() -> None:
    no_artifact = RuleBasedVerifier().verify(
        evidence(validation_test_run=passing_validation(stdout_ref=None))
    )
    zero_tests = RuleBasedVerifier().verify(
        evidence(validation_test_run=passing_validation(tests_run=0))
    )

    for decision in (no_artifact, zero_tests):
        assert decision.status is RunStatus.STOPPED
        assert decision.termination_reason is TerminationReason.TOOL_ERROR
        assert decision.verifier_error_code is VerifierErrorCode.VALIDATION_EVIDENCE_INVALID
        assert decision.task_metrics.task_success is False


@pytest.mark.parametrize(
    ("category", "termination", "error_category"),
    [
        (
            RepoTestRunCategory.TIMEOUT,
            TerminationReason.TIMEOUT,
            ErrorCategory.BUDGET_EXHAUSTED,
        ),
        (
            RepoTestRunCategory.SANDBOX_ERROR,
            TerminationReason.SANDBOX_ERROR,
            ErrorCategory.SANDBOX_FAILURE,
        ),
        (
            RepoTestRunCategory.POLICY_DENIED,
            TerminationReason.TOOL_ERROR,
            ErrorCategory.TOOL_MISUSE,
        ),
        (
            RepoTestRunCategory.IMPORT_ERROR,
            TerminationReason.TOOL_ERROR,
            ErrorCategory.TEST_MISREAD,
        ),
        (
            RepoTestRunCategory.NO_TESTS_COLLECTED,
            TerminationReason.TOOL_ERROR,
            ErrorCategory.TEST_MISREAD,
        ),
        (
            RepoTestRunCategory.OUTPUT_TRUNCATED,
            TerminationReason.TOOL_ERROR,
            ErrorCategory.TEST_MISREAD,
        ),
        (
            RepoTestRunCategory.UNKNOWN_NONZERO,
            TerminationReason.TOOL_ERROR,
            ErrorCategory.TEST_MISREAD,
        ),
    ],
)
def test_validation_infrastructure_errors_have_deterministic_terminal_mapping(
    category: RepoTestRunCategory,
    termination: TerminationReason,
    error_category: ErrorCategory,
) -> None:
    test_run = completed_test_run(status=RepoTestStatus.ERROR, category=category)
    decision = RuleBasedVerifier().verify(evidence(validation_test_run=test_run))

    assert decision.status is RunStatus.STOPPED
    assert decision.termination_reason is termination
    assert decision.error_category is error_category
    assert decision.verifier_error_code is VerifierErrorCode.TEST_INFRASTRUCTURE_ERROR
    assert decision.validation_result.test_run == test_run


def test_success_contract_rejects_directly_forged_incomplete_evidence() -> None:
    with pytest.raises(ValidationError, match="reproduced issue"):
        build_final_result(
            RuleBasedVerifier()
            .verify(
                evidence(
                    reproduction_result=ReproductionResult(
                        status=ReproductionStatus.NOT_ATTEMPTED,
                    )
                )
            )
            .model_copy(
                update={
                    "status": RunStatus.SUCCESS,
                    "termination_reason": TerminationReason.SUCCESS,
                    "verifier_error_code": None,
                    "error_category": None,
                    "task_metrics": TaskMetrics(
                        issue_reproduced=False,
                        patch_applied=True,
                        tests_passed=True,
                        task_success=False,
                        tool_steps=7,
                    ),
                }
            ),
            context(),
        )
