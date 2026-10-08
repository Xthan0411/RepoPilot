"""Deterministic, model-independent repair verification.

The Verifier consumes only typed runtime evidence. It never calls a model, executes a
command, reads evaluator gold data, or mutates a repository.
"""

from __future__ import annotations

from typing import Self
from uuid import UUID

from pydantic import Field, model_validator

from repopilot.enums import (
    ErrorCategory,
    PatchStatus,
    ReproductionStatus,
    RunStatus,
    TerminationReason,
    TestRunCategory,
    TestStatus,
    ValidationStatus,
    VerifierErrorCode,
)
from repopilot.schemas import (
    FinalResult,
    LocalizationMetrics,
    PatchState,
    RelativePath,
    ReproductionResult,
    StrictModel,
    TaskId,
    TaskMetrics,
    TestRunResult,
    UsageMetrics,
    ValidationResult,
)

RULE_BASED_VERIFIER_VERSION = "rule-based-verifier-v1"


class VerificationEvidence(StrictModel):
    """Complete runtime evidence available to the rule-based Verifier."""

    reproduction_result: ReproductionResult
    patch_state: PatchState
    validation_test_run: TestRunResult | None = None
    tool_steps: int = Field(ge=0)


class VerificationDecision(StrictModel):
    """Verifier-owned outcome before reporting-only metrics are attached."""

    status: RunStatus
    termination_reason: TerminationReason
    patch_path: RelativePath | None = None
    reproduction_result: ReproductionResult
    validation_result: ValidationResult
    task_metrics: TaskMetrics
    error_category: ErrorCategory | None = None
    verifier_error_code: VerifierErrorCode | None = None
    summary: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.task_metrics.patch_applied and self.patch_path is None:
            raise ValueError("an applied patch decision requires patch_path")
        if self.status is RunStatus.SUCCESS:
            if (
                self.termination_reason is not TerminationReason.SUCCESS
                or not self.task_metrics.task_success
                or self.error_category is not None
                or self.verifier_error_code is not None
            ):
                raise ValueError("a successful decision cannot contain failure state")
            reproduction_test = self.reproduction_result.test_run
            validation_test = self.validation_result.test_run
            if (
                self.reproduction_result.status is not ReproductionStatus.REPRODUCED
                or self.validation_result.status is not ValidationStatus.PASSED
                or reproduction_test is None
                or validation_test is None
            ):
                raise ValueError("a successful decision requires completed verifier evidence")
            if (
                reproduction_test.command_index != validation_test.command_index
                or reproduction_test.command_sha256 != validation_test.command_sha256
                or reproduction_test.observation_call_id == validation_test.observation_call_id
            ):
                raise ValueError("a successful decision requires independent matching test runs")
        elif (
            self.termination_reason is TerminationReason.SUCCESS or self.verifier_error_code is None
        ):
            raise ValueError("an unsuccessful decision requires a verifier failure code")
        return self


class FinalResultContext(StrictModel):
    """Reporting inputs that cannot influence the Verifier's decision."""

    task_id: TaskId
    run_id: UUID
    localization_metrics: LocalizationMetrics
    usage: UsageMetrics
    trace_path: RelativePath


class RuleBasedVerifier:
    """Fail-closed repair-success authority backed only by structured evidence."""

    def verify(self, evidence: VerificationEvidence) -> VerificationDecision:
        """Apply the frozen evidence matrix in deterministic precedence order."""

        validation = _build_validation_result(evidence.validation_test_run)
        patch_path = _patch_path(evidence.patch_state)
        metrics = _task_metrics(
            reproduction=evidence.reproduction_result,
            patch_state=evidence.patch_state,
            validation=validation,
            tool_steps=evidence.tool_steps,
        )

        reproduction_failure = _check_reproduction(evidence.reproduction_result)
        if reproduction_failure is not None:
            return _failure_decision(
                evidence=evidence,
                validation=validation,
                metrics=metrics,
                patch_path=patch_path,
                failure=reproduction_failure,
            )

        patch_failure = _check_patch(evidence.patch_state)
        if patch_failure is not None:
            return _failure_decision(
                evidence=evidence,
                validation=validation,
                metrics=metrics,
                patch_path=patch_path,
                failure=patch_failure,
            )

        validation_failure = _check_validation(
            reproduction=evidence.reproduction_result,
            test_run=evidence.validation_test_run,
        )
        if validation_failure is not None:
            return _failure_decision(
                evidence=evidence,
                validation=validation,
                metrics=metrics,
                patch_path=patch_path,
                failure=validation_failure,
            )

        success_metrics = TaskMetrics(
            issue_reproduced=True,
            patch_applied=True,
            tests_passed=True,
            task_success=True,
            tool_steps=evidence.tool_steps,
        )
        return VerificationDecision(
            status=RunStatus.SUCCESS,
            termination_reason=TerminationReason.SUCCESS,
            patch_path=patch_path,
            reproduction_result=evidence.reproduction_result,
            validation_result=validation,
            task_metrics=success_metrics,
            summary=(
                "Verified repair: baseline failed, artifact-backed patch applied, tests passed."
            ),
        )


def build_final_result(
    decision: VerificationDecision,
    context: FinalResultContext,
) -> FinalResult:
    """Attach reporting metrics without recalculating or overriding verification."""

    return FinalResult(
        task_id=context.task_id,
        run_id=context.run_id,
        status=decision.status,
        termination_reason=decision.termination_reason,
        patch_path=decision.patch_path,
        reproduction_result=decision.reproduction_result,
        validation_result=decision.validation_result,
        localization_metrics=context.localization_metrics,
        task_metrics=decision.task_metrics,
        usage=context.usage,
        trace_path=context.trace_path,
        error_category=decision.error_category,
        verifier_error_code=decision.verifier_error_code,
    )


class _Failure(StrictModel):
    status: RunStatus
    termination_reason: TerminationReason
    error_category: ErrorCategory
    code: VerifierErrorCode
    summary: str = Field(min_length=1, max_length=500)


def _check_reproduction(reproduction: ReproductionResult) -> _Failure | None:
    if reproduction.status is ReproductionStatus.NOT_ATTEMPTED:
        return _Failure(
            status=RunStatus.STOPPED,
            termination_reason=TerminationReason.TOOL_ERROR,
            error_category=ErrorCategory.TEST_MISREAD,
            code=VerifierErrorCode.REPRODUCTION_NOT_ATTEMPTED,
            summary="Verification stopped: baseline reproduction was not attempted.",
        )
    if reproduction.status is ReproductionStatus.NOT_REPRODUCED:
        return _Failure(
            status=RunStatus.FAILED,
            termination_reason=TerminationReason.TESTS_FAILED,
            error_category=ErrorCategory.TEST_MISREAD,
            code=VerifierErrorCode.ISSUE_NOT_REPRODUCED,
            summary=(
                "Verification failed: the configured baseline test did not reproduce the issue."
            ),
        )
    if reproduction.status is ReproductionStatus.ERROR:
        status, termination, category = _infrastructure_outcome(reproduction.test_run)
        return _Failure(
            status=status,
            termination_reason=termination,
            error_category=category,
            code=VerifierErrorCode.REPRODUCTION_ERROR,
            summary="Verification stopped: baseline reproduction ended in an infrastructure error.",
        )

    test_run = reproduction.test_run
    if (
        test_run is None
        or test_run.status is not TestStatus.FAILED
        or test_run.category not in {TestRunCategory.ASSERTION_FAILURE, TestRunCategory.TEST_ERROR}
        or test_run.tests_run == 0
        or test_run.failures + test_run.errors == 0
        or not test_run.failing_tests
        or not _has_output_artifact(test_run)
        or test_run.truncated
    ):
        return _Failure(
            status=RunStatus.STOPPED,
            termination_reason=TerminationReason.TOOL_ERROR,
            error_category=ErrorCategory.TEST_MISREAD,
            code=VerifierErrorCode.REPRODUCTION_EVIDENCE_INVALID,
            summary=(
                "Verification stopped: reproduced baseline evidence is incomplete or ambiguous."
            ),
        )
    return None


def _check_patch(patch_state: PatchState) -> _Failure | None:
    if patch_state.status is not PatchStatus.APPLIED:
        return _Failure(
            status=RunStatus.FAILED,
            termination_reason=TerminationReason.INVALID_PATCH,
            error_category=ErrorCategory.INVALID_PATCH,
            code=VerifierErrorCode.PATCH_NOT_APPLIED,
            summary="Verification failed: no candidate patch reached the applied state.",
        )
    if (
        patch_state.revision < 1
        or patch_state.diff_ref is None
        or patch_state.diff_ref.size_bytes == 0
        or not patch_state.files_changed
    ):
        return _Failure(
            status=RunStatus.FAILED,
            termination_reason=TerminationReason.INVALID_PATCH,
            error_category=ErrorCategory.INVALID_PATCH,
            code=VerifierErrorCode.PATCH_EVIDENCE_INVALID,
            summary="Verification failed: applied patch metadata is incomplete or invalid.",
        )
    return None


def _check_validation(
    *,
    reproduction: ReproductionResult,
    test_run: TestRunResult | None,
) -> _Failure | None:
    if test_run is None:
        return _Failure(
            status=RunStatus.STOPPED,
            termination_reason=TerminationReason.TOOL_ERROR,
            error_category=ErrorCategory.TEST_MISREAD,
            code=VerifierErrorCode.VALIDATION_NOT_RUN,
            summary="Verification stopped: no post-patch validation test was run.",
        )

    baseline = reproduction.test_run
    if baseline is None:
        return _Failure(
            status=RunStatus.STOPPED,
            termination_reason=TerminationReason.TOOL_ERROR,
            error_category=ErrorCategory.TEST_MISREAD,
            code=VerifierErrorCode.REPRODUCTION_EVIDENCE_INVALID,
            summary="Verification stopped: baseline command identity is unavailable.",
        )
    if (
        baseline.command_index != test_run.command_index
        or baseline.command_sha256 != test_run.command_sha256
    ):
        return _Failure(
            status=RunStatus.STOPPED,
            termination_reason=TerminationReason.TOOL_ERROR,
            error_category=ErrorCategory.TEST_MISREAD,
            code=VerifierErrorCode.TEST_COMMAND_MISMATCH,
            summary="Verification stopped: baseline and validation used different test commands.",
        )
    if baseline.observation_call_id == test_run.observation_call_id:
        return _Failure(
            status=RunStatus.STOPPED,
            termination_reason=TerminationReason.TOOL_ERROR,
            error_category=ErrorCategory.TEST_MISREAD,
            code=VerifierErrorCode.OBSERVATION_REUSED,
            summary="Verification stopped: baseline output was reused as validation evidence.",
        )
    if test_run.status is TestStatus.ERROR:
        status, termination, category = _infrastructure_outcome(test_run)
        return _Failure(
            status=status,
            termination_reason=termination,
            error_category=category,
            code=VerifierErrorCode.TEST_INFRASTRUCTURE_ERROR,
            summary=f"Verification stopped: validation ended with {test_run.category.value}.",
        )
    if not _valid_completed_test_evidence(test_run):
        return _Failure(
            status=RunStatus.STOPPED,
            termination_reason=TerminationReason.TOOL_ERROR,
            error_category=ErrorCategory.TEST_MISREAD,
            code=VerifierErrorCode.VALIDATION_EVIDENCE_INVALID,
            summary="Verification stopped: post-patch test evidence is incomplete or ambiguous.",
        )
    if test_run.status is TestStatus.FAILED:
        return _Failure(
            status=RunStatus.FAILED,
            termination_reason=TerminationReason.TESTS_FAILED,
            error_category=ErrorCategory.INVALID_PATCH,
            code=VerifierErrorCode.TESTS_FAILED,
            summary="Verification failed: the applied patch did not make the test command pass.",
        )
    return None


def _valid_completed_test_evidence(test_run: TestRunResult) -> bool:
    if test_run.tests_run == 0 or test_run.truncated or not _has_output_artifact(test_run):
        return False
    if test_run.status is TestStatus.PASSED:
        return (
            test_run.category is TestRunCategory.PASSED
            and test_run.exit_code == 0
            and test_run.failures == 0
            and test_run.errors == 0
        )
    if test_run.status is TestStatus.FAILED:
        return (
            test_run.category in {TestRunCategory.ASSERTION_FAILURE, TestRunCategory.TEST_ERROR}
            and test_run.exit_code not in {None, 0}
            and test_run.failures + test_run.errors > 0
            and bool(test_run.failing_tests)
        )
    return False


def _build_validation_result(test_run: TestRunResult | None) -> ValidationResult:
    if test_run is None:
        return ValidationResult(
            status=ValidationStatus.NOT_RUN,
            summary="Post-patch validation was not run.",
        )
    failed = test_run.failures + test_run.errors
    return ValidationResult(
        status={
            TestStatus.PASSED: ValidationStatus.PASSED,
            TestStatus.FAILED: ValidationStatus.FAILED,
            TestStatus.ERROR: ValidationStatus.ERROR,
        }[test_run.status],
        observation_call_id=test_run.observation_call_id,
        tests_passed=max(test_run.tests_run - failed, 0),
        tests_failed=failed,
        summary=test_run.summary,
        test_run=test_run,
    )


def _task_metrics(
    *,
    reproduction: ReproductionResult,
    patch_state: PatchState,
    validation: ValidationResult,
    tool_steps: int,
) -> TaskMetrics:
    return TaskMetrics(
        issue_reproduced=reproduction.status is ReproductionStatus.REPRODUCED,
        patch_applied=patch_state.status is PatchStatus.APPLIED,
        tests_passed=validation.status is ValidationStatus.PASSED,
        task_success=False,
        tool_steps=tool_steps,
    )


def _failure_decision(
    *,
    evidence: VerificationEvidence,
    validation: ValidationResult,
    metrics: TaskMetrics,
    patch_path: RelativePath | None,
    failure: _Failure,
) -> VerificationDecision:
    return VerificationDecision(
        status=failure.status,
        termination_reason=failure.termination_reason,
        patch_path=patch_path,
        reproduction_result=evidence.reproduction_result,
        validation_result=validation,
        task_metrics=metrics,
        error_category=failure.error_category,
        verifier_error_code=failure.code,
        summary=failure.summary,
    )


def _patch_path(patch_state: PatchState) -> RelativePath | None:
    if patch_state.status is not PatchStatus.APPLIED or patch_state.diff_ref is None:
        return None
    return patch_state.diff_ref.relative_path


def _has_output_artifact(test_run: TestRunResult) -> bool:
    return test_run.stdout_ref is not None or test_run.stderr_ref is not None


def _infrastructure_outcome(
    test_run: TestRunResult | None,
) -> tuple[RunStatus, TerminationReason, ErrorCategory]:
    if test_run is not None and test_run.category is TestRunCategory.TIMEOUT:
        return (
            RunStatus.STOPPED,
            TerminationReason.TIMEOUT,
            ErrorCategory.BUDGET_EXHAUSTED,
        )
    if test_run is not None and test_run.category is TestRunCategory.SANDBOX_ERROR:
        return (
            RunStatus.STOPPED,
            TerminationReason.SANDBOX_ERROR,
            ErrorCategory.SANDBOX_FAILURE,
        )
    if test_run is not None and test_run.category is TestRunCategory.POLICY_DENIED:
        return (
            RunStatus.STOPPED,
            TerminationReason.TOOL_ERROR,
            ErrorCategory.TOOL_MISUSE,
        )
    return (
        RunStatus.STOPPED,
        TerminationReason.TOOL_ERROR,
        ErrorCategory.TEST_MISREAD,
    )


__all__ = [
    "RULE_BASED_VERIFIER_VERSION",
    "FinalResultContext",
    "RuleBasedVerifier",
    "VerificationDecision",
    "VerificationEvidence",
    "build_final_result",
]
