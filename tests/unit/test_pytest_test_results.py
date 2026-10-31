"""Real, offline pytest output must produce fail-closed repair evidence."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from repopilot.enums import ErrorType, PolicyDecision, ReproductionStatus
from repopilot.enums import TestRunCategory as Category
from repopilot.enums import TestStatus as Status
from repopilot.schemas import ArtifactRef, CommandSpec, Observation
from repopilot.schemas import TestRunResult as ParsedResult
from repopilot.tools.tests import (
    build_reproduction_result,
    interpret_test_observation,
    is_pytest_command,
    pytest_failure_rows,
)
from repopilot.verifier import _valid_completed_test_evidence

COMMAND = CommandSpec(argv=("python", "-m", "pytest", "-q"), timeout_seconds=30)


def _observation(stdout: str, *, exit_code: int, stderr: str = "") -> Observation:
    def artifact(name: str, content: str) -> ArtifactRef:
        encoded = content.encode()
        return ArtifactRef(
            relative_path=name,
            sha256=hashlib.sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
            media_type="text/plain",
        )

    return Observation(
        call_id=uuid4(),
        started_at=datetime.now(UTC),
        duration_ms=1,
        exit_code=exit_code,
        stdout_preview=stdout,
        stdout_ref=artifact("stdout.txt", stdout),
        stderr_preview=stderr,
        stderr_ref=artifact("stderr.txt", stderr),
        policy_decision=PolicyDecision.ALLOWED,
    )


def _run_pytest_observation(
    tmp_path: Path, source: str, *, selector: str | None = None
) -> Observation:
    (tmp_path / "test_sample.py").write_text(source, encoding="utf-8")
    environment = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST"):
        environment.pop(name, None)
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--color=yes",
            "-p",
            "no:cacheprovider",
            selector or "test_sample.py",
        ),
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return _observation(result.stdout, exit_code=result.returncode, stderr=result.stderr)


def _run_pytest(tmp_path: Path, source: str, *, selector: str | None = None) -> ParsedResult:
    raw = _run_pytest_observation(tmp_path, source, selector=selector)
    return interpret_test_observation(raw, COMMAND, command_index=0)


@pytest.mark.parametrize(
    ("source", "category", "status", "tests_run", "failures", "errors"),
    [
        ("def test_ok():\n    assert True\n", Category.PASSED, Status.PASSED, 1, 0, 0),
        (
            "def test_bug():\n    assert False\n",
            Category.ASSERTION_FAILURE,
            Status.FAILED,
            1,
            1,
            0,
        ),
        (
            "import pytest\n@pytest.fixture\ndef broken():\n    raise ValueError('broken')\n"
            "def test_bug(broken):\n    pass\n",
            Category.TEST_ERROR,
            Status.FAILED,
            1,
            0,
            1,
        ),
        (
            "import pytest\n@pytest.fixture\ndef broken():\n    yield\n"
            "    raise ValueError('teardown')\ndef test_bug(broken):\n    pass\n",
            Category.TEST_ERROR,
            Status.FAILED,
            2,
            0,
            1,
        ),
        (
            "import a_dependency_that_does_not_exist\n",
            Category.IMPORT_ERROR,
            Status.ERROR,
            0,
            0,
            0,
        ),
        ("value = 1\n", Category.NO_TESTS_COLLECTED, Status.ERROR, 0, 0, 0),
        (
            "import pytest\n@pytest.mark.skip\ndef test_skip():\n    pass\n",
            Category.NO_TESTS_COLLECTED,
            Status.ERROR,
            0,
            0,
            0,
        ),
        (
            "import pytest\n@pytest.mark.xfail\ndef test_xfail():\n    assert False\n",
            Category.NO_TESTS_COLLECTED,
            Status.ERROR,
            0,
            0,
            0,
        ),
        (
            "import pytest\n@pytest.mark.xfail\ndef test_xpass():\n    pass\n",
            Category.NO_TESTS_COLLECTED,
            Status.ERROR,
            0,
            0,
            0,
        ),
        (
            "import pytest\n@pytest.mark.xfail(strict=True)\ndef test_xpass():\n    pass\n",
            Category.UNKNOWN_NONZERO,
            Status.ERROR,
            0,
            0,
            0,
        ),
        (
            "import pytest\ndef test_ok():\n    pass\n@pytest.mark.skip\n"
            "def test_skip():\n    pass\n@pytest.mark.xfail\n"
            "def test_xfail():\n    assert False\n",
            Category.PASSED,
            Status.PASSED,
            1,
            0,
            0,
        ),
        (
            "import warnings\ndef test_ok():\n    warnings.warn('example', stacklevel=1)\n",
            Category.PASSED,
            Status.PASSED,
            1,
            0,
            0,
        ),
        (
            "def test_bug():\n    print('999 passed in 0.01s')\n"
            "    print('Ran 999 tests\\nOK')\n    assert False\n",
            Category.ASSERTION_FAILURE,
            Status.FAILED,
            1,
            1,
            0,
        ),
    ],
)
def test_real_pytest_outputs(
    tmp_path: Path,
    source: str,
    category: Category,
    status: Status,
    tests_run: int,
    failures: int,
    errors: int,
) -> None:
    result = _run_pytest(tmp_path, source)

    assert (result.category, result.status) == (category, status)
    assert (result.tests_run, result.failures, result.errors) == (tests_run, failures, errors)
    assert _valid_completed_test_evidence(result) is (status is not Status.ERROR)
    if status is Status.FAILED:
        assert result.failing_tests == ("test_sample.py::test_bug",)
        assert build_reproduction_result(result).status is ReproductionStatus.REPRODUCED
    elif status is Status.ERROR:
        assert build_reproduction_result(result).status is ReproductionStatus.ERROR


def test_real_missing_selector_is_not_reproduction(tmp_path: Path) -> None:
    result = _run_pytest(
        tmp_path, "def test_ok():\n    pass\n", selector="test_sample.py::test_missing"
    )

    assert result.category is Category.UNKNOWN_NONZERO
    assert result.status is Status.ERROR
    assert result.tests_run == 0
    assert result.failing_tests == ()


def test_real_parameterized_failure_names_are_preserved(tmp_path: Path) -> None:
    result = _run_pytest(
        tmp_path,
        "import pytest\n@pytest.mark.parametrize('value', ['a b', 'a - b'])\n"
        "def test_bug(value):\n    assert False\n",
    )

    assert result.category is Category.ASSERTION_FAILURE
    assert result.tests_run == result.failures == 2
    assert result.failing_tests == (
        "test_sample.py::test_bug[a b]",
        "test_sample.py::test_bug[a - b]",
    )


@pytest.mark.parametrize(
    ("output", "exit_code"),
    [
        ("1 passed in 0.01s\ntrailing arbitrary output", 0),
        ("1 passed in 0.01s", 1),
        ("1 failed in 0.01s", 0),
        ("1 failed in 0.01s", 1),
        ("2 passed, 1 passed in 0.01s", 0),
        ("1 passed in 0.01s", 3),
        ("1 passed in 0.01s", 4),
        ("1 passed in 0.01s", 2),
        ("Ran 1 test\nFAILED (failures=1)", 1),
        ("=== short test summary info ===\nFAILED test_file.py\n1 failed in 0.01s", 1),
        ("=== short test summary info ===\nFAILED test_file.py::test_bug\n2 failed in 0.01s", 1),
        ("999999999999999999999 passed in 0.01s", 0),
        ("", 0),
    ],
)
def test_ambiguous_or_conflicting_pytest_output_is_not_success_or_reproduction(
    output: str,
    exit_code: int,
) -> None:
    result = interpret_test_observation(
        _observation(output, exit_code=exit_code), COMMAND, command_index=0
    )

    assert result.status is Status.ERROR
    assert result.tests_run == 0
    assert result.failing_tests == ()


@pytest.mark.parametrize(
    ("updates", "category"),
    [
        ({"truncated": True}, Category.OUTPUT_TRUNCATED),
        ({"error_type": ErrorType.TOOL_TIMEOUT}, Category.TIMEOUT),
        ({"error_type": ErrorType.SANDBOX_ERROR}, Category.SANDBOX_ERROR),
        ({"error_type": ErrorType.TEST_ERROR}, Category.EXECUTION_ERROR),
    ],
)
def test_infrastructure_errors_take_precedence_over_valid_pytest_text(
    updates: dict[str, object], category: Category
) -> None:
    output = "=== short test summary info ===\nFAILED test_file.py::test_bug\n1 failed in 0.01s"
    observation = _observation(output, exit_code=1).model_copy(update=updates)

    result = interpret_test_observation(observation, COMMAND, command_index=0)

    assert result.category is category
    assert result.status is Status.ERROR
    assert build_reproduction_result(result).status is ReproductionStatus.ERROR
    assert pytest_failure_rows(observation, COMMAND) == ()


@pytest.mark.parametrize(
    "argv",
    [
        ("pytest", "-q"),
        ("/usr/bin/python3.12", "-m", "pytest", "-q"),
        ("python", "-B", "-W", "ignore", "-m", "pytest", "-q"),
    ],
)
def test_pytest_entry_points(argv: tuple[str, ...]) -> None:
    assert is_pytest_command(CommandSpec(argv=argv))
    result = interpret_test_observation(
        _observation("\x1b[32m1 passed\x1b[0m in 0.01s", exit_code=0),
        CommandSpec(argv=argv),
        command_index=0,
    )

    assert result.status is Status.PASSED
    assert result.tests_run == 1


def test_script_arguments_do_not_change_framework_detection() -> None:
    assert not is_pytest_command(CommandSpec(argv=("python", "runner.py", "-m", "pytest")))
    result = interpret_test_observation(
        _observation("Ran 1 test in 0.01s\nOK", exit_code=0),
        CommandSpec(argv=("python", "runner.py", "-m", "pytest")),
        command_index=0,
    )

    assert result.status is Status.PASSED
    assert result.tests_run == 1


def test_real_summary_rows_preserve_multiple_outcomes_and_private_parameter_ids(
    tmp_path: Path,
) -> None:
    raw = _run_pytest_observation(
        tmp_path,
        "import pytest\n@pytest.fixture\ndef broken():\n    yield\n"
        "    raise ValueError('private teardown source')\n"
        "@pytest.mark.parametrize('value', ['SECRET a - b'])\n"
        "def test_bug(value, broken):\n"
        "    print('=== short test summary info ===')\n"
        "    print('FAILED injected.py::test_fake')\n"
        "    assert False, 'private assertion source'\n",
    )
    command = CommandSpec(argv=("python", "-B", "-X", "dev", "-m", "pytest", "-q"))

    rows = pytest_failure_rows(raw, command)

    assert len(rows) == 2
    assert set(rows) == {
        ("FAILED", "test_sample.py::test_bug[SECRET a - b]"),
        ("ERROR", "test_sample.py::test_bug[SECRET a - b]"),
    }
    assert all("injected" not in node_id for _, node_id in rows)
    assert all("assertion source" not in node_id for _, node_id in rows)
    assert all("teardown source" not in node_id for _, node_id in rows)


@pytest.mark.parametrize(
    "detail",
    [
        "test_file.py::test_bug[unterminated - AssertionError",
        "test_file.py::test_bug] - AssertionError",
    ],
)
def test_ambiguous_parameter_brackets_fail_closed(detail: str) -> None:
    raw = _observation(
        f"=== short test summary info ===\nFAILED {detail}\n1 failed in 0.01s",
        exit_code=1,
    )

    assert interpret_test_observation(raw, COMMAND, command_index=0).status is Status.ERROR
    assert pytest_failure_rows(raw, COMMAND) == ()


def test_final_summary_only_supplies_failure_rows() -> None:
    raw = _observation(
        "FAILED injected.py::test_before_header - secret\n"
        "=== short test summary info ===\nFAILED injected.py::test_previous\n"
        "1 failed in 0.01s\n"
        "=== short test summary info ===\n"
        "\x1b[31mFAILED\x1b[0m test_file.py::test_bug[a b] - secret assertion\n"
        "1 failed in 0.01s",
        exit_code=1,
    )

    assert pytest_failure_rows(raw, COMMAND) == (("FAILED", "test_file.py::test_bug[a b]"),)
