"""Fail-closed interpretation tests for unittest Sandbox Observations."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from repopilot.enums import (
    ErrorType,
    PolicyDecision,
    ReproductionStatus,
)
from repopilot.enums import TestRunCategory as RepoTestRunCategory
from repopilot.enums import TestStatus as RepoTestStatus
from repopilot.schemas import ArtifactRef, CommandSpec, Observation
from repopilot.schemas import TestRunResult as RepoTestRunResult
from repopilot.schemas import TestState as RepoTestState
from repopilot.tools import (
    advance_test_state,
    build_reproduction_result,
    interpret_test_observation,
)

CALL_ID = UUID("55555555-5555-4555-8555-555555555555")
STARTED_AT = datetime(2026, 8, 6, 8, tzinfo=UTC)
COMMAND = CommandSpec(
    argv=("python", "-m", "unittest", "discover", "-s", "checks", "-q"),
    timeout_seconds=30,
)
STDOUT_REF = ArtifactRef(
    relative_path="stdout.txt",
    sha256="a" * 64,
    size_bytes=100,
    media_type="text/plain",
)
STDERR_REF = ArtifactRef(
    relative_path="stderr.txt",
    sha256="b" * 64,
    size_bytes=200,
    media_type="text/plain",
)


def observation(
    *,
    exit_code: int | None = 0,
    stdout: str | None = None,
    stderr: str | None = None,
    truncated: bool = False,
    policy_decision: PolicyDecision = PolicyDecision.ALLOWED,
    error_type: ErrorType | None = None,
) -> Observation:
    return Observation(
        call_id=CALL_ID,
        started_at=STARTED_AT,
        duration_ms=19,
        exit_code=exit_code,
        stdout_ref=STDOUT_REF if stdout is not None else None,
        stderr_ref=STDERR_REF if stderr is not None or truncated else None,
        stdout_preview=stdout,
        stderr_preview=stderr,
        truncated=truncated,
        sandbox_id="sandbox-test-run",
        policy_decision=policy_decision,
        error_type=error_type,
    )


@pytest.mark.parametrize(
    ("captured", "category", "status", "failing_tests"),
    [
        (
            observation(
                stderr=(
                    "----------------------------------------------------------------------\n"
                    "Ran 2 tests in 0.001s\n\nOK\n"
                )
            ),
            RepoTestRunCategory.PASSED,
            RepoTestStatus.PASSED,
            (),
        ),
        (
            observation(
                exit_code=1,
                stderr=(
                    "FAIL: test_value (checks.ParserTests.test_value)\n"
                    "----------------------------------------------------------------------\n"
                    "Ran 2 tests in 0.001s\n\nFAILED (failures=1)\n"
                ),
            ),
            RepoTestRunCategory.ASSERTION_FAILURE,
            RepoTestStatus.FAILED,
            ("checks.ParserTests.test_value",),
        ),
        (
            observation(
                exit_code=1,
                stderr=(
                    "ERROR: test_blank (checks.ParserTests.test_blank)\n"
                    "----------------------------------------------------------------------\n"
                    "Ran 2 tests in 0.001s\n\nFAILED (errors=1)\n"
                ),
            ),
            RepoTestRunCategory.TEST_ERROR,
            RepoTestStatus.FAILED,
            ("checks.ParserTests.test_blank",),
        ),
    ],
)
def test_interprets_completed_unittest_outcomes(
    captured: Observation,
    category: RepoTestRunCategory,
    status: RepoTestStatus,
    failing_tests: tuple[str, ...],
) -> None:
    result = interpret_test_observation(captured, COMMAND, command_index=0)

    assert result.category is category
    assert result.status is status
    assert result.tests_run == 2
    assert result.failing_tests == failing_tests
    assert len(result.command_sha256) == 64


def test_completed_results_drive_reproduction_and_agent_test_state() -> None:
    captured = observation(
        exit_code=1,
        stderr=(
            "FAIL: test_value (checks.ParserTests.test_value)\n"
            "Ran 1 test in 0.001s\nFAILED (failures=1)\n"
        ),
    )
    result = interpret_test_observation(captured, COMMAND, command_index=0)

    reproduction = build_reproduction_result(result)
    state = advance_test_state(RepoTestState(), result)

    assert reproduction.status is ReproductionStatus.REPRODUCED
    assert reproduction.test_run == result
    assert state.status is RepoTestStatus.FAILED
    assert state.commands_run == 1
    assert state.last_result == result


@pytest.mark.parametrize(
    ("captured", "category", "error_type"),
    [
        (
            observation(
                exit_code=1,
                stderr=(
                    "ERROR: checks.test_missing "
                    "(unittest.loader._FailedTest.checks.test_missing)\n"
                    "Failed to import test module: checks.test_missing\n"
                    "Ran 1 test in 0.001s\nFAILED (errors=1)\n"
                ),
            ),
            RepoTestRunCategory.IMPORT_ERROR,
            ErrorType.TEST_ERROR,
        ),
        (
            observation(stderr="Ran 0 tests in 0.000s\n\nOK\n"),
            RepoTestRunCategory.NO_TESTS_COLLECTED,
            ErrorType.TEST_ERROR,
        ),
        (
            observation(exit_code=None, error_type=ErrorType.TOOL_TIMEOUT),
            RepoTestRunCategory.TIMEOUT,
            ErrorType.TOOL_TIMEOUT,
        ),
        (
            observation(
                exit_code=None,
                policy_decision=PolicyDecision.DENIED,
                error_type=ErrorType.POLICY_DENIED,
            ),
            RepoTestRunCategory.POLICY_DENIED,
            ErrorType.POLICY_DENIED,
        ),
        (
            observation(exit_code=None, error_type=ErrorType.SANDBOX_ERROR),
            RepoTestRunCategory.SANDBOX_ERROR,
            ErrorType.SANDBOX_ERROR,
        ),
        (
            observation(exit_code=None, error_type=ErrorType.TOOL_EXECUTION_ERROR),
            RepoTestRunCategory.EXECUTION_ERROR,
            ErrorType.TOOL_EXECUTION_ERROR,
        ),
        (
            observation(exit_code=0, stderr="Ran 2 tests", truncated=True),
            RepoTestRunCategory.OUTPUT_TRUNCATED,
            ErrorType.TEST_ERROR,
        ),
        (
            observation(exit_code=3),
            RepoTestRunCategory.UNKNOWN_NONZERO,
            ErrorType.TEST_ERROR,
        ),
    ],
)
def test_ambiguous_or_infrastructure_outcomes_fail_closed(
    captured: Observation,
    category: RepoTestRunCategory,
    error_type: ErrorType,
) -> None:
    result = interpret_test_observation(captured, COMMAND, command_index=0)

    assert result.category is category
    assert result.status is RepoTestStatus.ERROR
    assert result.error_type is error_type
    assert build_reproduction_result(result).status is ReproductionStatus.ERROR


def test_test_result_rejects_contradictory_pass_evidence() -> None:
    valid = interpret_test_observation(
        observation(stderr="Ran 1 test in 0.001s\n\nOK\n"),
        COMMAND,
        command_index=0,
    )

    with pytest.raises(ValidationError, match="contradictory"):
        RepoTestRunResult.model_validate(
            {
                **valid.model_dump(mode="python"),
                "failures": 1,
            }
        )
