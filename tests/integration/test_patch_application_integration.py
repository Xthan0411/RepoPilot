"""Real-Docker proof for atomic patch application, recovery, and no-follow writes."""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from repopilot.config import RepoPilotSettings
from repopilot.enums import PatchErrorCode, PatchStatus, SandboxNetworkMode
from repopilot.enums import TestStatus as RepoTestStatus
from repopilot.evals import load_repair_fixture_manifest
from repopilot.sandbox import (
    DOCKER_WORKSPACE_ROOT,
    DockerCliClient,
    DockerExecSpec,
    DockerSandbox,
    ReadFileRequest,
    RunRequest,
    SandboxPolicy,
)
from repopilot.schemas import ToolCall
from repopilot.tools import (
    PatchToolSession,
    interpret_test_observation,
)

pytestmark = pytest.mark.integration

FIXTURE_MANIFEST = Path(__file__).parents[1] / "fixtures" / "p5_repair_tasks" / "manifest.json"
REQUESTED_AT = datetime(2026, 8, 6, 9, tzinfo=UTC)
PARTIAL_CONFLICT = (
    "diff --git a/config_parser/__init__.py b/config_parser/__init__.py\n"
    "--- a/config_parser/__init__.py\n"
    "+++ b/config_parser/__init__.py\n"
    "@@ -1 +1 @@\n"
    '-"""Tiny key/value parser repair fixture."""\n'
    '+"""This change must never survive the later conflict."""\n'
    "diff --git a/config_parser/parser.py b/config_parser/parser.py\n"
    "--- a/config_parser/parser.py\n"
    "+++ b/config_parser/parser.py\n"
    "@@ -1 +1 @@\n"
    "-THIS LINE DOES NOT EXIST\n"
    "+replacement\n"
)


def settings() -> RepoPilotSettings:
    return RepoPilotSettings(
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


def image() -> str:
    configured = os.environ.get("REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE")
    if configured is None:
        pytest.skip("set REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE to a local Python image digest")
    return configured


def call(diff: str) -> ToolCall:
    return ToolCall(
        call_id=uuid4(),
        tool_name="apply_patch",
        validated_arguments={"diff": diff},
        requested_at=REQUESTED_AT,
    )


def test_real_docker_applies_both_gold_isolated_fixture_patches(
    tmp_path: Path,
) -> None:
    manifest = load_repair_fixture_manifest(FIXTURE_MANIFEST)
    configuration = settings()
    client = DockerCliClient()

    for case in manifest.cases:
        task = case.evaluation_task.task
        artifact_root = tmp_path / f"{task.task_id}-artifacts"
        sandbox = DockerSandbox(
            policy=SandboxPolicy.from_settings(configuration),
            image_reference=image(),
            artifact_root=artifact_root,
            client=client,
        )
        handle = sandbox.prepare(task)
        session = PatchToolSession.from_settings(
            sandbox=sandbox,
            handle=handle,
            task=task,
            settings=configuration,
        )

        try:
            applied = session.apply(
                case.evaluation_task.expected_patch,
                call=call(case.evaluation_task.expected_patch),
            )
            inspected = session.inspect()
            test_observation = sandbox.run(
                handle,
                RunRequest(
                    call_id=uuid4(),
                    command=task.test_commands[0],
                ),
            )
            test_run = interpret_test_observation(
                test_observation,
                task.test_commands[0],
                command_index=0,
            )

            assert applied.status is PatchStatus.APPLIED
            assert applied.patch_state.revision == 1
            assert applied.patch_state.files_changed == case.evaluation_task.gold_files
            assert applied.patch_state.diff_ref is not None
            assert (
                applied.patch_state.diff_ref.sha256
                == hashlib.sha256(case.evaluation_task.expected_patch.encode()).hexdigest()
            )
            assert inspected.available is True
            assert inspected.patch_state == applied.patch_state
            assert test_run.status is RepoTestStatus.PASSED
            assert test_run.tests_run == 2
        finally:
            sandbox.destroy(handle)

        artifacts = sandbox.collect_artifacts(handle)
        assert len(artifacts) == 5
        directory = artifact_root / handle.sandbox_id
        for artifact in artifacts:
            content = (directory / Path(artifact.relative_path).name).read_bytes()
            assert len(content) == artifact.size_bytes
            assert hashlib.sha256(content).hexdigest() == artifact.sha256


def test_real_docker_conflict_leaves_all_files_unchanged_then_recovers(
    tmp_path: Path,
) -> None:
    case = load_repair_fixture_manifest(FIXTURE_MANIFEST).cases[0]
    task = case.evaluation_task.task
    configuration = settings()
    sandbox = DockerSandbox(
        policy=SandboxPolicy.from_settings(configuration),
        image_reference=image(),
        artifact_root=tmp_path / "conflict-artifacts",
        client=DockerCliClient(),
    )
    handle = sandbox.prepare(task)
    session = PatchToolSession.from_settings(
        sandbox=sandbox,
        handle=handle,
        task=task,
        settings=configuration,
    )

    try:
        rejected = session.apply(PARTIAL_CONFLICT, call=call(PARTIAL_CONFLICT))
        unchanged = sandbox.read_file(
            handle,
            ReadFileRequest(
                call_id=uuid4(),
                relative_path="config_parser/__init__.py",
            ),
        )
        recovered = session.apply(
            case.evaluation_task.expected_patch,
            call=call(case.evaluation_task.expected_patch),
        )

        assert rejected.status is PatchStatus.REJECTED
        assert rejected.error_code is PatchErrorCode.APPLY_CONFLICT
        assert rejected.patch_state.status is PatchStatus.NONE
        assert unchanged.exit_code == 0
        assert unchanged.stdout_preview == (
            '"""Tiny key/value parser repair fixture."""\n\n'
            "from config_parser.parser import parse_config\n\n"
            '__all__ = ["parse_config"]\n'
        )
        assert recovered.status is PatchStatus.APPLIED
        assert recovered.patch_state.revision == 1
    finally:
        sandbox.destroy(handle)


def test_real_docker_runtime_symlink_cannot_redirect_patch_write(
    tmp_path: Path,
) -> None:
    case = load_repair_fixture_manifest(FIXTURE_MANIFEST).cases[0]
    task = case.evaluation_task.task
    configuration = settings()
    client = DockerCliClient()
    sandbox = DockerSandbox(
        policy=SandboxPolicy.from_settings(configuration),
        image_reference=image(),
        artifact_root=tmp_path / "symlink-artifacts",
        client=client,
    )
    handle = sandbox.prepare(task)
    assert sandbox.preparation is not None
    session = PatchToolSession.from_settings(
        sandbox=sandbox,
        handle=handle,
        task=task,
        settings=configuration,
    )
    mutation = client.execute(
        sandbox.preparation.container_id,
        DockerExecSpec(
            argv=(
                "python",
                "-I",
                "-c",
                (
                    "import os;"
                    "os.unlink('/workspace/repository/config_parser/parser.py');"
                    "os.symlink('__init__.py','/workspace/repository/config_parser/parser.py')"
                ),
            ),
            workdir=DOCKER_WORKSPACE_ROOT,
            timeout_seconds=5,
            max_artifact_bytes=4_096,
        ),
    )
    assert mutation.exit_code == 0

    try:
        rejected = session.apply(
            case.evaluation_task.expected_patch,
            call=call(case.evaluation_task.expected_patch),
        )
        unaffected = sandbox.read_file(
            handle,
            ReadFileRequest(
                call_id=uuid4(),
                relative_path="config_parser/__init__.py",
            ),
        )

        assert rejected.status is PatchStatus.REJECTED
        assert rejected.error_code is PatchErrorCode.APPLY_CONFLICT
        assert unaffected.exit_code == 0
        assert "Tiny key/value parser repair fixture" in (unaffected.stdout_preview or "")
    finally:
        sandbox.destroy(handle)
