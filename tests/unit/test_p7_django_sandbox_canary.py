"""P7-S6 Django Sandbox compatibility canary tests."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from repopilot.enums import ErrorType, PolicyDecision
from repopilot.evals.dataset import load_frozen_dataset
from repopilot.schemas import ArtifactRef, Observation, TaskSpec

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p7_django_sandbox_canary import (  # noqa: E402
    DATASET_MANIFEST_PATH,
    TASK_ID,
    CanaryError,
    _raise_command_failure,
    build_dockerfile,
    cap_command_timeouts,
)


def _task() -> TaskSpec:
    bundle = load_frozen_dataset(
        PROJECT_ROOT / DATASET_MANIFEST_PATH,
        repository_root=PROJECT_ROOT,
    )
    return next(
        item.task for item in bundle.external_evaluation_tasks if item.task.task_id == TASK_ID
    )


def test_canary_projection_caps_commands_without_changing_agent_prompt_counts() -> None:
    task = _task()
    projected = cap_command_timeouts(task)

    assert projected.task_id == task.task_id
    assert projected.issue_body == task.issue_body
    assert projected.base_commit == task.base_commit
    assert projected.timeout_seconds == task.timeout_seconds
    assert len(projected.setup_commands) == len(task.setup_commands)
    assert len(projected.test_commands) == len(task.test_commands)
    assert [item.timeout_seconds for item in task.setup_commands] == [600, 600]
    assert [item.timeout_seconds for item in projected.setup_commands] == [300, 300]
    assert [item.timeout_seconds for item in task.test_commands] == [600]
    assert [item.timeout_seconds for item in projected.test_commands] == [300]


def test_adapter_dockerfile_is_gold_free_offline_and_uses_testbed_python() -> None:
    image = f"swebench/task@sha256:{'a' * 64}"
    dockerfile = build_dockerfile(image)

    assert dockerfile.startswith(f"FROM {image}\n")
    assert "/opt/miniconda3/envs/testbed/bin" in dockerfile
    assert 'ENV HOME="/workspace"' in dockerfile
    assert 'ENV PYTHONUSERBASE="/workspace/.local"' in dockerfile
    assert 'ENV TMPDIR="/workspace"' in dockerfile
    assert "/workspace/.local/bin" in dockerfile
    assert "no-index = true" in dockerfile
    assert "user = true" in dockerfile
    assert "no-build-isolation = false" in dockerfile
    assert "\nbuild-isolation = false" not in dockerfile
    assert "65532" in dockerfile
    assert "test_patch" not in dockerfile
    assert "expected_patch" not in dockerfile
    assert "OPENAI" not in dockerfile


def test_adapter_dockerfile_uses_validated_project_distribution() -> None:
    image = f"swebench/task@sha256:{'a' * 64}"

    dockerfile = build_dockerfile(image, distribution_name="pytest")

    assert "python -m pip uninstall -y pytest" in dockerfile
    with pytest.raises(CanaryError, match="distribution_name_invalid"):
        build_dockerfile(image, distribution_name="pytest; curl example.invalid")


def test_direct_file_help_invocation_resolves_scripts_package() -> None:
    completed = subprocess.run(
        (sys.executable, str(PROJECT_ROOT / "scripts/p7_django_sandbox_canary.py"), "--help"),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Run a Gold-isolated Django canary" in completed.stdout


def test_setup_failure_emits_bounded_diagnostic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    observation = Observation(
        call_id=uuid4(),
        started_at=datetime.now(UTC),
        duration_ms=1,
        exit_code=1,
        stdout_preview="x" * 5_000,
        stderr_preview="permission denied",
        stdout_ref=ArtifactRef(
            relative_path="artifacts/sbx/call.stdout",
            sha256="a" * 64,
            size_bytes=5_000,
            media_type="text/plain",
        ),
        stderr_ref=ArtifactRef(
            relative_path="artifacts/sbx/call.stderr",
            sha256="b" * 64,
            size_bytes=17,
            media_type="text/plain",
        ),
        policy_decision=PolicyDecision.ALLOWED,
        error_type=ErrorType.TOOL_EXECUTION_ERROR,
    )

    with pytest.raises(CanaryError, match="setup_1_failed") as raised:
        _raise_command_failure("setup_1_failed", observation)

    assert raised.value.code == "setup_1_failed"
    line = capsys.readouterr().err.strip()
    diagnostic = json.loads(line.removeprefix("canary_command_failure="))
    assert diagnostic["code"] == "setup_1_failed"
    assert diagnostic["exit_code"] == 1
    assert diagnostic["stderr_tail"] == "permission denied"
    assert diagnostic["stdout_tail"] == "x" * 4_000
