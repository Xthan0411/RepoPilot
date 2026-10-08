"""Real-Docker proof that every P5 baseline fixture fails as declared."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from uuid import uuid4

import pytest

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    PolicyDecision,
    ReproductionStatus,
    SandboxNetworkMode,
)
from repopilot.enums import TestRunCategory as RepoTestRunCategory
from repopilot.enums import TestStatus as RepoTestStatus
from repopilot.evals import load_repair_fixture_manifest
from repopilot.sandbox import (
    DockerCliClient,
    DockerClientError,
    DockerSandbox,
    RunRequest,
    SandboxPolicy,
)
from repopilot.tools import build_reproduction_result, interpret_test_observation

pytestmark = pytest.mark.integration

FIXTURE_MANIFEST = Path(__file__).parents[1] / "fixtures" / "p5_repair_tasks" / "manifest.json"
EXPECTED_CATEGORIES = {
    "p5-config-parser-blank": RepoTestRunCategory.TEST_ERROR,
    "p5-parser-service-async": RepoTestRunCategory.ASSERTION_FAILURE,
}


def test_real_docker_reproduces_every_declared_base_failure(tmp_path: Path) -> None:
    image = os.environ.get("REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE")
    if image is None:
        pytest.skip("set REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE to a local digest-pinned Python image")

    manifest = load_repair_fixture_manifest(FIXTURE_MANIFEST)
    settings = RepoPilotSettings(
        sandbox_network=SandboxNetworkMode.NONE,
        sandbox_cpu_limit=0.25,
        sandbox_memory_limit_mb=128,
        sandbox_pids_limit=32,
        sandbox_tmpfs_limit_mb=16,
        sandbox_command_timeout_seconds=30,
        sandbox_max_output_bytes=16_384,
        sandbox_max_artifact_bytes=65_536,
        patch_max_bytes=65_536,
    )
    client = DockerCliClient()

    for case in manifest.cases:
        task = case.evaluation_task.task
        sandbox = DockerSandbox(
            policy=SandboxPolicy.from_settings(settings),
            image_reference=image,
            artifact_root=tmp_path / f"{task.task_id}-artifacts",
            client=client,
        )
        handle = sandbox.prepare(task)
        assert sandbox.preparation is not None
        container_id = sandbox.preparation.container_id

        try:
            observation = sandbox.run(
                handle,
                RunRequest(
                    call_id=uuid4(),
                    command=task.test_commands[0],
                ),
            )
            assert observation.policy_decision is PolicyDecision.ALLOWED
            assert observation.error_type is None
            assert observation.exit_code == case.expected_base_exit_code
            combined_preview = (observation.stdout_preview or "") + (
                observation.stderr_preview or ""
            )
            for marker in case.expected_failure_markers:
                assert marker in combined_preview
            assert "diff --git" not in combined_preview

            test_run = interpret_test_observation(
                observation,
                task.test_commands[0],
                command_index=0,
            )
            reproduction = build_reproduction_result(test_run)
            assert test_run.category is EXPECTED_CATEGORIES[task.task_id]
            assert test_run.status is RepoTestStatus.FAILED
            assert test_run.tests_run == 2
            assert test_run.failing_tests
            assert test_run.stdout_ref == observation.stdout_ref
            assert test_run.stderr_ref == observation.stderr_ref
            assert reproduction.status is ReproductionStatus.REPRODUCED
            assert reproduction.test_run == test_run
        finally:
            sandbox.destroy(handle)

        artifacts = sandbox.collect_artifacts(handle)
        assert len(artifacts) == 2
        artifact_directory = tmp_path / f"{task.task_id}-artifacts" / handle.sandbox_id
        for artifact in artifacts:
            content = (artifact_directory / Path(artifact.relative_path).name).read_bytes()
            assert len(content) == artifact.size_bytes
            assert hashlib.sha256(content).hexdigest() == artifact.sha256
            assert case.evaluation_task.expected_patch.encode() not in content
        with pytest.raises(DockerClientError):
            client.inspect_container(container_id)
