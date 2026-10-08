"""Framework-aware, fail-closed interpretation of Sandbox test Observations."""

from __future__ import annotations

import hashlib
import json
import re

from repopilot.enums import (
    ErrorType,
    PolicyDecision,
    ReproductionStatus,
    TestRunCategory,
    TestStatus,
)
from repopilot.schemas import (
    CommandSpec,
    Observation,
    ReproductionResult,
    TestRunResult,
    TestState,
)

_RAN_TESTS = re.compile(r"\bRan\s+(\d+)\s+tests?\b")
_FAILURE_COUNT = re.compile(r"\bfailures=(\d+)\b")
_ERROR_COUNT = re.compile(r"\berrors=(\d+)\b")
_FAILING_TEST = re.compile(
    r"^(?:FAIL|ERROR):\s+([^\s(]+)(?:\s+\(([^)\n]+)\))?",
    re.MULTILINE,
)
_IMPORT_ERROR_MARKERS = (
    "Failed to import test module:",
    "unittest.loader._FailedTest",
)


def interpret_test_observation(
    observation: Observation,
    command: CommandSpec,
    *,
    command_index: int,
) -> TestRunResult:
    """Convert one bounded unittest Observation into a conservative typed result."""

    output = (observation.stdout_preview or "") + "\n" + (observation.stderr_preview or "")
    tests_run = _first_count(_RAN_TESTS, output)
    failures = _first_count(_FAILURE_COUNT, output)
    errors = _first_count(_ERROR_COUNT, output)
    failing_tests = _failing_test_names(output)
    category, status, error_type = _classify(
        observation,
        output=output,
        tests_run=tests_run,
        failures=failures,
        errors=errors,
    )
    if category is TestRunCategory.IMPORT_ERROR:
        failing_tests = ()
    exit_text = "none" if observation.exit_code is None else str(observation.exit_code)
    return TestRunResult(
        observation_call_id=observation.call_id,
        command_index=command_index,
        command_sha256=_command_sha256(command),
        status=status,
        category=category,
        exit_code=observation.exit_code,
        duration_ms=observation.duration_ms,
        tests_run=tests_run,
        failures=failures,
        errors=errors,
        failing_tests=failing_tests,
        stdout_ref=observation.stdout_ref,
        stderr_ref=observation.stderr_ref,
        truncated=observation.truncated,
        error_type=error_type,
        summary=(
            f"{category.value}; tests={tests_run}; failures={failures}; "
            f"errors={errors}; exit={exit_text}"
        ),
    )


def build_reproduction_result(test_run: TestRunResult) -> ReproductionResult:
    """Map test semantics to issue-reproduction semantics without evaluator gold."""

    status = {
        TestStatus.PASSED: ReproductionStatus.NOT_REPRODUCED,
        TestStatus.FAILED: ReproductionStatus.REPRODUCED,
        TestStatus.ERROR: ReproductionStatus.ERROR,
    }.get(test_run.status)
    if status is None:
        raise ValueError("only completed test runs can become reproduction evidence")
    return ReproductionResult(
        status=status,
        observation_call_id=test_run.observation_call_id,
        failing_tests=test_run.failing_tests,
        summary=test_run.summary,
        test_run=test_run,
    )


def advance_test_state(current: TestState, test_run: TestRunResult) -> TestState:
    """Record one completed command without discarding its structured evidence."""

    if test_run.status in {TestStatus.NOT_RUN, TestStatus.RUNNING}:
        raise ValueError("only completed test runs can update test state")
    return TestState(
        status=test_run.status,
        last_observation_call_id=test_run.observation_call_id,
        commands_run=current.commands_run + 1,
        last_result=test_run,
    )


def _classify(
    observation: Observation,
    *,
    output: str,
    tests_run: int,
    failures: int,
    errors: int,
) -> tuple[TestRunCategory, TestStatus, ErrorType | None]:
    if observation.policy_decision is PolicyDecision.DENIED:
        return (
            TestRunCategory.POLICY_DENIED,
            TestStatus.ERROR,
            ErrorType.POLICY_DENIED,
        )
    if observation.error_type is ErrorType.TOOL_TIMEOUT:
        return TestRunCategory.TIMEOUT, TestStatus.ERROR, ErrorType.TOOL_TIMEOUT
    if observation.error_type is ErrorType.SANDBOX_ERROR:
        return (
            TestRunCategory.SANDBOX_ERROR,
            TestStatus.ERROR,
            ErrorType.SANDBOX_ERROR,
        )
    if observation.error_type is not None:
        return (
            TestRunCategory.EXECUTION_ERROR,
            TestStatus.ERROR,
            observation.error_type,
        )
    if observation.truncated:
        return (
            TestRunCategory.OUTPUT_TRUNCATED,
            TestStatus.ERROR,
            ErrorType.TEST_ERROR,
        )
    if any(marker in output for marker in _IMPORT_ERROR_MARKERS):
        return TestRunCategory.IMPORT_ERROR, TestStatus.ERROR, ErrorType.TEST_ERROR
    if tests_run == 0 and _RAN_TESTS.search(output) is not None:
        return (
            TestRunCategory.NO_TESTS_COLLECTED,
            TestStatus.ERROR,
            ErrorType.TEST_ERROR,
        )
    if observation.exit_code == 0:
        return TestRunCategory.PASSED, TestStatus.PASSED, None
    if failures > 0:
        return TestRunCategory.ASSERTION_FAILURE, TestStatus.FAILED, None
    if errors > 0 and tests_run > 0:
        return TestRunCategory.TEST_ERROR, TestStatus.FAILED, None
    if observation.exit_code is not None:
        return (
            TestRunCategory.UNKNOWN_NONZERO,
            TestStatus.ERROR,
            ErrorType.TEST_ERROR,
        )
    return (
        TestRunCategory.EXECUTION_ERROR,
        TestStatus.ERROR,
        ErrorType.TEST_ERROR,
    )


def _first_count(pattern: re.Pattern[str], output: str) -> int:
    match = pattern.search(output)
    return 0 if match is None else int(match.group(1))


def _failing_test_names(output: str) -> tuple[str, ...]:
    names: list[str] = []
    for match in _FAILING_TEST.finditer(output):
        name = match.group(2) or match.group(1)
        if name not in names:
            names.append(name)
    return tuple(names)


def _command_sha256(command: CommandSpec) -> str:
    payload = json.dumps(
        command.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
