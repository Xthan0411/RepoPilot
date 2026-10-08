"""Lifecycle and no-side-effect tests for the Sandbox Protocol and FakeSandbox."""

from __future__ import annotations

import json
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from repopilot.enums import ErrorType, PolicyDecision, SandboxLifecycle
from repopilot.sandbox.base import (
    ApplyPatchRequest,
    ReadFileRequest,
    RunRequest,
    SandboxAdapter,
    SandboxHandle,
)
from repopilot.sandbox.fake import (
    FakeSandbox,
    FakeSandboxConfigurationError,
    FakeSandboxStep,
    SandboxLifecycleError,
    SandboxScriptExhaustedError,
    SandboxScriptMismatchError,
)
from repopilot.schemas import ArtifactRef, CommandSpec, Observation, TaskSpec

FIXTURES = Path(__file__).parents[1] / "fixtures"
RUN_CALL_ID = UUID("88888888-8888-4888-8888-888888888888")
READ_CALL_ID = UUID("66666666-6666-4666-8666-666666666666")
PATCH_CALL_ID = UUID("99999999-9999-4999-8999-999999999999")
STARTED_AT = datetime(2026, 7, 27, 8, 1, tzinfo=UTC)


def load_fixture(name: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((FIXTURES / name).read_text(encoding="utf-8")),
    )


def make_task() -> TaskSpec:
    return TaskSpec.model_validate(load_fixture("valid_task.json"))


def make_handle(
    *,
    sandbox_id: str = "sandbox-run-001",
    task_id: str = "fixture-001",
) -> SandboxHandle:
    return SandboxHandle(
        sandbox_id=sandbox_id,
        task_id=task_id,
        repository_commit="a" * 40,
        prepared_at=STARTED_AT,
    )


def make_artifact(
    relative_path: str = "runs/run-001/candidate.diff",
    *,
    content_hash: str = "c" * 64,
) -> ArtifactRef:
    return ArtifactRef(
        relative_path=relative_path,
        sha256=content_hash,
        size_bytes=96,
        media_type="text/x-diff",
    )


def make_observation(call_id: UUID) -> Observation:
    return Observation(
        call_id=call_id,
        started_at=STARTED_AT,
        duration_ms=2,
        exit_code=0,
        sandbox_id="sandbox-run-001",
        policy_decision=PolicyDecision.ALLOWED,
    )


def assert_lifecycle(sandbox: FakeSandbox, expected: SandboxLifecycle) -> None:
    assert sandbox.lifecycle is expected


def make_script() -> tuple[
    tuple[FakeSandboxStep, ...],
    RunRequest,
    ReadFileRequest,
    ApplyPatchRequest,
]:
    run_request = RunRequest(
        call_id=RUN_CALL_ID,
        command=CommandSpec(argv=("python", "-m", "pytest", "-q"), timeout_seconds=300),
    )
    read_request = ReadFileRequest(
        call_id=READ_CALL_ID,
        relative_path="parser.py",
        start_line=1,
        end_line=80,
    )
    patch_request = ApplyPatchRequest(
        call_id=PATCH_CALL_ID,
        patch_ref=make_artifact(),
    )
    steps = (
        FakeSandboxStep(
            request=run_request,
            observation=make_observation(RUN_CALL_ID),
        ),
        FakeSandboxStep(
            request=read_request,
            observation=make_observation(READ_CALL_ID),
        ),
        FakeSandboxStep(
            request=patch_request,
            observation=make_observation(PATCH_CALL_ID),
        ),
    )
    return steps, run_request, read_request, patch_request


def test_sandbox_contracts_and_fixture_round_trip() -> None:
    handle = make_handle()
    step = FakeSandboxStep.model_validate(load_fixture("valid_fake_sandbox_step.json"))

    assert SandboxHandle.model_validate_json(handle.model_dump_json()) == handle
    assert FakeSandboxStep.model_validate_json(step.model_dump_json()) == step
    assert isinstance(step.request, ReadFileRequest)


def test_fake_sandbox_runs_exact_script_and_idempotent_cleanup() -> None:
    steps, run_request, read_request, patch_request = make_script()
    artifact = make_artifact("runs/run-001/result.json", content_hash="e" * 64)
    sandbox = FakeSandbox(handle=make_handle(), steps=steps, artifacts=(artifact,))
    adapter: SandboxAdapter = sandbox

    assert isinstance(adapter, SandboxAdapter)
    assert_lifecycle(sandbox, SandboxLifecycle.NEW)

    handle = sandbox.prepare(make_task())
    assert_lifecycle(sandbox, SandboxLifecycle.PREPARED)
    assert sandbox.run(handle, run_request) == steps[0].observation
    assert sandbox.read_file(handle, read_request) == steps[1].observation
    assert sandbox.apply_patch(handle, patch_request) == steps[2].observation
    assert sandbox.collect_artifacts(handle) == (artifact,)
    assert sandbox.remaining_steps == 0

    sandbox.destroy(handle)
    sandbox.destroy(handle)
    assert_lifecycle(sandbox, SandboxLifecycle.DESTROYED)


def test_fake_sandbox_performs_no_host_file_network_or_process_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = FakeSandboxStep.model_validate(load_fixture("valid_fake_sandbox_step.json"))
    sandbox = FakeSandbox(handle=make_handle(), steps=(step,))
    handle = sandbox.prepare(make_task())

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("FakeSandbox attempted an external side effect")

    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    assert isinstance(step.request, ReadFileRequest)
    assert sandbox.read_file(handle, step.request) == step.observation


def test_fake_sandbox_rejects_operations_before_prepare_and_after_destroy() -> None:
    steps, run_request, _, _ = make_script()
    sandbox = FakeSandbox(handle=make_handle(), steps=steps)

    with pytest.raises(SandboxLifecycleError, match="prepared") as before:
        sandbox.run(make_handle(), run_request)
    assert before.value.error_type is ErrorType.SANDBOX_ERROR

    handle = sandbox.prepare(make_task())
    with pytest.raises(SandboxLifecycleError, match="already prepared"):
        sandbox.prepare(make_task())

    sandbox.destroy(handle)
    with pytest.raises(SandboxLifecycleError, match="destroyed"):
        sandbox.run(handle, run_request)


def test_fake_sandbox_rejects_destroy_before_prepare() -> None:
    sandbox = FakeSandbox(handle=make_handle(), steps=())

    with pytest.raises(SandboxLifecycleError, match="before prepare"):
        sandbox.destroy(make_handle())


def test_fake_sandbox_rejects_task_or_handle_mismatch() -> None:
    sandbox = FakeSandbox(handle=make_handle(), steps=())
    wrong_task_payload = load_fixture("valid_task.json")
    wrong_task_payload["task_id"] = "fixture-002"

    with pytest.raises(FakeSandboxConfigurationError, match="requested task"):
        sandbox.prepare(TaskSpec.model_validate(wrong_task_payload))

    handle = sandbox.prepare(make_task())
    with pytest.raises(SandboxLifecycleError, match="handle"):
        sandbox.collect_artifacts(make_handle(sandbox_id="sandbox-other"))
    sandbox.destroy(handle)


def test_fake_sandbox_request_mismatch_does_not_consume_step() -> None:
    steps, _, _, _ = make_script()
    sandbox = FakeSandbox(handle=make_handle(), steps=steps)
    handle = sandbox.prepare(make_task())
    unexpected = RunRequest(
        call_id=RUN_CALL_ID,
        command=CommandSpec(argv=("python", "-V")),
    )

    with pytest.raises(SandboxScriptMismatchError) as captured:
        sandbox.run(handle, unexpected)

    assert captured.value.error_type is ErrorType.INVALID_TOOL_ARGUMENTS
    assert sandbox.remaining_steps == 3


def test_fake_sandbox_reports_script_exhaustion() -> None:
    sandbox = FakeSandbox(handle=make_handle(), steps=())
    handle = sandbox.prepare(make_task())
    request = RunRequest(
        call_id=RUN_CALL_ID,
        command=CommandSpec(argv=("python", "-V")),
    )

    with pytest.raises(SandboxScriptExhaustedError) as captured:
        sandbox.run(handle, request)

    assert captured.value.error_type is ErrorType.SANDBOX_ERROR


def test_fake_sandbox_step_correlates_request_and_observation() -> None:
    request = ReadFileRequest(
        call_id=READ_CALL_ID,
        relative_path="parser.py",
    )

    with pytest.raises(ValidationError, match="call_id"):
        FakeSandboxStep(
            request=request,
            observation=make_observation(RUN_CALL_ID),
        )


def test_fake_sandbox_configuration_requires_unique_calls_and_artifacts() -> None:
    steps, _, _, _ = make_script()
    with pytest.raises(FakeSandboxConfigurationError, match="call_id"):
        FakeSandbox(handle=make_handle(), steps=(steps[0], steps[0]))

    artifact = make_artifact()
    with pytest.raises(FakeSandboxConfigurationError, match="artifact paths"):
        FakeSandbox(
            handle=make_handle(),
            steps=(),
            artifacts=(artifact, artifact),
        )


def test_read_file_request_rejects_escape_and_invalid_line_range() -> None:
    with pytest.raises(ValidationError):
        ReadFileRequest(call_id=READ_CALL_ID, relative_path="../secret")
    with pytest.raises(ValidationError, match="end_line"):
        ReadFileRequest(
            call_id=READ_CALL_ID,
            relative_path="parser.py",
            start_line=5,
            end_line=4,
        )
