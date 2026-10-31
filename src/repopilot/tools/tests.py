"""Framework-aware, fail-closed interpretation of Sandbox test Observations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePath
from typing import Literal

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
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PYTEST_SUMMARY = re.compile(
    r"(?P<counts>no tests ran|[0-9][a-z0-9, ]*) in [0-9]+(?:\.[0-9]+)?s"
    r"(?: \([0-9:]+\))?"
)
_PYTEST_COUNT = re.compile(
    r"([0-9]{1,9}) (passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)"
)
_PYTEST_SHORT_SUMMARY = re.compile(r"=+ short test summary info =+")


@dataclass(frozen=True)
class _PytestResult:
    category: TestRunCategory
    tests_run: int = 0
    failures: int = 0
    errors: int = 0
    failing_tests: tuple[str, ...] = ()
    failure_rows: tuple[tuple[Literal["FAILED", "ERROR"], str], ...] = ()


def interpret_test_observation(
    observation: Observation,
    command: CommandSpec,
    *,
    command_index: int,
) -> TestRunResult:
    """Interpret bounded pytest/unittest output; never execute or trust output as code."""

    output = _observation_output(observation)
    pytest_result = (
        _parse_pytest(output, observation.exit_code) if is_pytest_command(command) else None
    )
    tests_run = (
        _first_count(_RAN_TESTS, output) if pytest_result is None else pytest_result.tests_run
    )
    failures = (
        _first_count(_FAILURE_COUNT, output) if pytest_result is None else pytest_result.failures
    )
    errors = _first_count(_ERROR_COUNT, output) if pytest_result is None else pytest_result.errors
    failing_tests = (
        _failing_test_names(output) if pytest_result is None else pytest_result.failing_tests
    )
    category, status, error_type = _classify(
        observation,
        output=output,
        tests_run=tests_run,
        failures=failures,
        errors=errors,
        pytest_result=pytest_result,
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
    pytest_result: _PytestResult | None = None,
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
    if pytest_result is not None:
        category = pytest_result.category
        if category is TestRunCategory.PASSED:
            return category, TestStatus.PASSED, None
        if category in {TestRunCategory.ASSERTION_FAILURE, TestRunCategory.TEST_ERROR}:
            return category, TestStatus.FAILED, None
        return category, TestStatus.ERROR, ErrorType.TEST_ERROR
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


def is_pytest_command(command: CommandSpec) -> bool:
    """Identify supported pytest entry points without inspecting untrusted test output."""

    executable = PurePath(command.argv[0]).name
    if executable in {"pytest", "py.test"}:
        return True
    if re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)?)?", executable) is None:
        return False
    arguments = iter(command.argv[1:])
    for argument in arguments:
        if argument in {"-B", "-E", "-I", "-O", "-OO", "-s", "-S", "-u"}:
            continue
        if argument in {"-W", "-X"}:
            next(arguments, None)
            continue
        if argument.startswith(("-W", "-X")):
            continue
        return argument == "-m" and next(arguments, None) == "pytest"
    return False


def pytest_failure_rows(
    observation: Observation, command: CommandSpec
) -> tuple[tuple[Literal["FAILED", "ERROR"], str], ...]:
    """Return validated final-summary rows for evaluator-internal output projection.

    IDs may contain private parameter values: callers must redact them before exposing
    these rows to an Agent. Preserve every row, including multiple outcomes for one ID.
    Infrastructure errors and incomplete/ambiguous summaries never produce rows.
    """

    if not is_pytest_command(command):
        return ()
    interpreted = interpret_test_observation(observation, command, command_index=0)
    if interpreted.status is not TestStatus.FAILED:
        return ()
    return _parse_pytest(_observation_output(observation), observation.exit_code).failure_rows


def pytest_failure_type_hints(
    observation: Observation, command: CommandSpec
) -> tuple[str | None, ...]:
    """Expose only fixed exception labels from validated final-summary rows.

    Never return exception messages, custom type names, source or parameter values.
    This is a diagnostic projection, not a claim that FAILED implies AssertionError.
    """
    rows = pytest_failure_rows(observation, command)
    if not rows:
        return ()
    allowed = frozenset(
        {
            "AssertionError",
            "TypeError",
            "ValueError",
            "AttributeError",
            "KeyError",
            "IndexError",
            "NameError",
            "RuntimeError",
            "NotImplementedError",
            "ZeroDivisionError",
            "OverflowError",
            "StopIteration",
        }
    )
    lines = [line.strip() for line in _observation_output(observation).splitlines() if line.strip()]
    header = max(i for i, line in enumerate(lines) if _PYTEST_SHORT_SUMMARY.fullmatch(line))
    hints: list[str | None] = []
    for line in lines[header + 1 : -1]:
        kind, _, detail = line.partition(" ")
        if kind not in {"FAILED", "ERROR"}:
            continue
        node = _pytest_node_id(detail)
        suffix = detail[len(node or "") :]
        label = suffix[3:].partition(":")[0] if suffix.startswith(" - ") else ""
        hints.append(label if label in allowed and len(hints) < 16 else None)
    return tuple(hints) if len(hints) == len(rows) else tuple(None for _ in rows)


def _observation_output(observation: Observation) -> str:
    output = (observation.stdout_preview or "") + "\n" + (observation.stderr_preview or "")
    return _ANSI_ESCAPE.sub("", output)


def _parse_pytest(output: str, exit_code: int | None) -> _PytestResult:
    """Require a terminal summary and exit-code agreement, not arbitrary printed counts.

    Skip/xfail/xpass outcomes are not evidence that a test passed or reproduced a bug.
    Counts describe completed pytest outcomes (a teardown error can accompany a pass).
    This bounded text parser is not an authenticity boundary against a compromised runner.
    """

    invalid = _PytestResult(
        TestRunCategory.EXECUTION_ERROR
        if exit_code in {None, 0}
        else TestRunCategory.UNKNOWN_NONZERO
    )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if exit_code == 2:
        category = (
            TestRunCategory.IMPORT_ERROR
            if any("ERROR collecting " in line for line in lines)
            else TestRunCategory.EXECUTION_ERROR
        )
        return _PytestResult(category)
    if exit_code == 5:
        return _PytestResult(TestRunCategory.NO_TESTS_COLLECTED)
    if exit_code not in {0, 1} or not lines:
        return invalid
    if any(line.startswith("[XPASS(strict)]") for line in lines):
        return invalid
    summary = _PYTEST_SUMMARY.fullmatch(lines[-1].strip("= "))
    if summary is None:
        return invalid
    counts: dict[str, int] = {}
    if summary["counts"] != "no tests ran":
        for part in summary["counts"].split(", "):
            count = _PYTEST_COUNT.fullmatch(part)
            if count is None:
                return invalid
            kind = {"error": "errors", "warning": "warnings"}.get(count[2], count[2])
            if kind in counts:
                return invalid
            counts[kind] = int(count[1])
    failures = counts.get("failed", 0)
    errors = counts.get("errors", 0)
    tests_run = counts.get("passed", 0) + failures + errors
    if exit_code == 0:
        if failures or errors:
            return invalid
        return _PytestResult(
            TestRunCategory.PASSED if tests_run else TestRunCategory.NO_TESTS_COLLECTED,
            tests_run=tests_run,
        )
    if failures + errors == 0:
        return invalid
    headers = [index for index, line in enumerate(lines) if _PYTEST_SHORT_SUMMARY.fullmatch(line)]
    if not headers:
        return invalid
    failed_ids: list[str] = []
    error_ids: list[str] = []
    rows: list[tuple[Literal["FAILED", "ERROR"], str]] = []
    for line in lines[headers[-1] + 1 : -1]:
        kind, _, detail = line.partition(" ")
        if kind not in {"FAILED", "ERROR"}:
            continue
        node_id = _pytest_node_id(detail)
        if node_id is None or "::" not in node_id or "[XPASS(strict)]" in detail:
            return invalid
        if kind == "FAILED":
            failed_ids.append(node_id)
            rows.append(("FAILED", node_id))
        else:
            error_ids.append(node_id)
            rows.append(("ERROR", node_id))
    if len(failed_ids) != failures or len(error_ids) != errors:
        return invalid
    return _PytestResult(
        TestRunCategory.ASSERTION_FAILURE if failures else TestRunCategory.TEST_ERROR,
        tests_run=tests_run,
        failures=failures,
        errors=errors,
        failing_tests=tuple(dict.fromkeys((*failed_ids, *error_ids))),
        failure_rows=tuple(rows),
    )


def _pytest_node_id(detail: str) -> str | None:
    # A parameter can contain spaces or " - "; only split outside its brackets.
    # Unbalanced IDs cannot be disambiguated from the error message safely.
    depth = 0
    for index, character in enumerate(detail):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0 and detail.startswith(" - ", index):
            return detail[:index]
    return detail if depth == 0 else None


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
