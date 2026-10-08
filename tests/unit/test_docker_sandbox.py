"""Docker lifecycle tests using an injected daemon client and tar inspection."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    DockerSandboxErrorCode,
    ErrorType,
    PatchErrorCode,
    PolicyDecision,
    SandboxCommandPhase,
    SandboxLifecycle,
    SandboxNetworkMode,
)
from repopilot.sandbox import (
    DOCKER_WORKSPACE_ROOT,
    ApplyPatchRequest,
    DockerCliClient,
    DockerClientError,
    DockerContainerInspection,
    DockerContainerSpec,
    DockerExecResult,
    DockerExecSpec,
    DockerImageIdentity,
    DockerMountInspection,
    DockerSandbox,
    DockerSandboxError,
    ReadFileRequest,
    RunRequest,
    SandboxHandle,
    SandboxPolicy,
    WorkspaceCopyResult,
)
from repopilot.sandbox.docker import (
    DOCKER_ARCHIVE_MTIME_SECONDS,
    DOCKER_FIXED_ENVIRONMENT,
    _write_filtered_tar,
)
from repopilot.schemas import ArtifactRef, CommandSpec, TaskSpec

TASK_ID = "docker-lifecycle-task"
COMMIT = "a" * 40
IMAGE_DIGEST = "b" * 64
IMAGE = f"alpine@sha256:{IMAGE_DIGEST}"
IMAGE_ID = f"sha256:{'c' * 64}"
CONTAINER_ID = "d" * 64
NOW = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
VALID_PATCH = (
    b"diff --git a/pkg/module.py b/pkg/module.py\n"
    b"--- a/pkg/module.py\n"
    b"+++ b/pkg/module.py\n"
    b"@@ -1 +1 @@\n"
    b"-VALUE = 1\n"
    b"+VALUE = 2\n"
)


def make_task(repository_path: Path) -> TaskSpec:
    return TaskSpec(
        task_id=TASK_ID,
        repository_path=str(repository_path),
        base_commit=COMMIT,
        issue_title="Docker lifecycle fixture",
        issue_body="Prepare and destroy one isolated workspace.",
        test_commands=(CommandSpec(argv=("python", "-m", "pytest", "-q")),),
    )


def make_policy(**overrides: object) -> SandboxPolicy:
    return SandboxPolicy.from_settings(RepoPilotSettings.model_validate(overrides))


class FakeDockerClient:
    """Record lifecycle calls and synthesize inspect from the approved spec."""

    def __init__(self, *, fail_at: set[str] | None = None) -> None:
        self.fail_at = fail_at or set()
        self.events: list[str] = []
        self.spec: DockerContainerSpec | None = None
        self.source: Path | None = None
        self.copy_max_bytes: int | None = None
        self.posture_override: dict[str, object] = {}
        self.exec_results: list[DockerExecResult] = []
        self.exec_specs: list[DockerExecSpec] = []
        self.copied_artifacts: dict[str, bytes] = {}
        self.network_connected = False

    def ping(self) -> None:
        self._event("ping")

    def resolve_image(self, image_reference: str) -> DockerImageIdentity:
        self._event("image")
        assert image_reference == IMAGE
        return DockerImageIdentity(
            image_id=IMAGE_ID,
            repository_digests=(IMAGE,),
        )

    def create_container(self, spec: DockerContainerSpec) -> str:
        self._event("create")
        self.spec = spec
        return CONTAINER_ID

    def start_container(self, container_id: str) -> None:
        self._event("start")
        assert container_id == CONTAINER_ID

    def copy_directory(
        self,
        container_id: str,
        source: Path,
        *,
        max_bytes: int,
    ) -> WorkspaceCopyResult:
        self._event("copy")
        assert container_id == CONTAINER_ID
        self.source = source
        self.copy_max_bytes = max_bytes
        return WorkspaceCopyResult(
            file_count=1,
            directory_count=0,
            byte_count=7,
            skipped_symlink_count=0,
            skipped_denied_count=0,
        )

    def inspect_container(self, container_id: str) -> DockerContainerInspection:
        self._event("inspect")
        assert container_id == CONTAINER_ID
        assert self.spec is not None
        specification = self.spec
        uid, _, gid = specification.user.partition(":")
        payload: dict[str, object] = {
            "container_id": CONTAINER_ID,
            "image_id": IMAGE_ID,
            "running": True,
            "user": specification.user,
            "network_mode": "none",
            "network_names": (("bridge", "none") if self.network_connected else ("none",)),
            "read_only_rootfs": True,
            "privileged": False,
            "cap_drop": ("ALL",),
            "security_options": ("no-new-privileges",),
            "nano_cpus": 1_000_000_000,
            "memory_bytes": 1_024 * 1024 * 1024,
            "pids_limit": 256,
            "tmpfs": {
                DOCKER_WORKSPACE_ROOT: (f"rw,nosuid,nodev,size=256m,uid={uid},gid={gid},mode=0750")
            },
            "mounts": (
                DockerMountInspection(
                    mount_type="tmpfs",
                    destination=DOCKER_WORKSPACE_ROOT,
                ),
            ),
            "environment_names": tuple(sorted({*DOCKER_FIXED_ENVIRONMENT, "PATH"})),
            "labels": specification.labels,
            "workdir": DOCKER_WORKSPACE_ROOT,
        }
        payload.update(self.posture_override)
        return DockerContainerInspection.model_validate(payload)

    def copy_artifact(
        self,
        container_id: str,
        name: str,
        content: bytes,
        *,
        max_bytes: int,
    ) -> None:
        self._event("copy-artifact")
        assert container_id == CONTAINER_ID
        assert len(content) <= max_bytes
        self.copied_artifacts[name] = content

    def remove_container(self, container_id: str) -> None:
        self._event("remove")
        assert container_id == CONTAINER_ID

    def execute(
        self,
        container_id: str,
        specification: DockerExecSpec,
    ) -> DockerExecResult:
        self._event("execute")
        assert container_id == CONTAINER_ID
        self.exec_specs.append(specification)
        if self.exec_results:
            return self.exec_results.pop(0)
        return DockerExecResult(
            exit_code=0,
            duration_ms=17,
            stdout=b"",
            stderr=b"",
        )

    def connect_setup_network(self, container_id: str) -> None:
        self._event("network-connect")
        assert container_id == CONTAINER_ID
        self.network_connected = True

    def disconnect_setup_network(self, container_id: str) -> None:
        self._event("network-disconnect")
        assert container_id == CONTAINER_ID
        self.network_connected = False

    def _event(self, name: str) -> None:
        self.events.append(name)
        if name in self.fail_at:
            raise DockerClientError(name)


def make_sandbox(
    client: FakeDockerClient,
    *,
    image_reference: str = IMAGE,
    suffix: str = "unit-test-0001",
    artifact_root: Path | None = None,
    settings_overrides: dict[str, object] | None = None,
) -> DockerSandbox:
    return DockerSandbox(
        policy=make_policy(**(settings_overrides or {})),
        image_reference=image_reference,
        artifact_root=artifact_root or Path(tempfile.mkdtemp(prefix="repopilot-unit-artifacts-")),
        client=client,
        now=lambda: NOW,
        id_factory=lambda: suffix,
    )


def test_prepare_compiles_and_verifies_minimal_container(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("VALUE=1", encoding="utf-8")
    client = FakeDockerClient()
    sandbox = make_sandbox(client)

    handle = sandbox.prepare(make_task(tmp_path))

    assert sandbox.lifecycle is SandboxLifecycle.PREPARED
    assert handle == SandboxHandle(
        sandbox_id="sbx-unit-test-0001",
        task_id=TASK_ID,
        repository_commit=COMMIT,
        prepared_at=NOW,
    )
    assert client.events == ["ping", "image", "create", "start", "copy", "inspect"]
    assert client.source == tmp_path.resolve()
    assert client.copy_max_bytes == 256 * 1024 * 1024
    assert client.spec is not None
    assert client.spec.image_reference == IMAGE
    assert client.spec.user == "65532:65532"
    assert client.spec.network_mode is SandboxNetworkMode.NONE
    assert client.spec.read_only_rootfs is True
    assert client.spec.privileged is False
    assert client.spec.cap_drop == ("ALL",)
    assert client.spec.security_options == ("no-new-privileges",)
    assert client.spec.environment == DOCKER_FIXED_ENVIRONMENT
    assert all("/Users/" not in value for value in client.spec.model_dump_json())
    assert sandbox.preparation is not None
    assert sandbox.preparation.container_id == CONTAINER_ID
    assert sandbox.preparation.workspace_copy.file_count == 1


def test_two_adapters_receive_independent_container_names(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    first_client = FakeDockerClient()
    second_client = FakeDockerClient()

    first = make_sandbox(first_client, suffix="first-task")
    second = make_sandbox(second_client, suffix="second-task")

    assert first.prepare(task).sandbox_id != second.prepare(task).sandbox_id
    assert first_client.spec is not None
    assert second_client.spec is not None
    assert first_client.spec.name != second_client.spec.name


def test_setup_only_configuration_still_prepares_container_offline(tmp_path: Path) -> None:
    client = FakeDockerClient()
    sandbox = DockerSandbox(
        policy=make_policy(sandbox_network="setup_only"),
        image_reference=IMAGE,
        artifact_root=tmp_path.parent / f"{tmp_path.name}-artifacts",
        client=client,
        now=lambda: NOW,
        id_factory=lambda: "setup-only",
    )

    sandbox.prepare(make_task(tmp_path))

    assert client.spec is not None
    assert client.spec.network_mode is SandboxNetworkMode.NONE


@pytest.mark.parametrize(
    ("stage", "expected_code", "cleanup_count"),
    [
        ("ping", DockerSandboxErrorCode.DAEMON_UNAVAILABLE, 0),
        ("image", DockerSandboxErrorCode.IMAGE_UNAVAILABLE, 0),
        ("create", DockerSandboxErrorCode.CREATE_FAILED, 0),
        ("start", DockerSandboxErrorCode.START_FAILED, 1),
        ("copy", DockerSandboxErrorCode.WORKSPACE_COPY_FAILED, 1),
        ("inspect", DockerSandboxErrorCode.INSPECT_FAILED, 1),
    ],
)
def test_prepare_maps_each_client_failure_and_rolls_back(
    tmp_path: Path,
    stage: str,
    expected_code: DockerSandboxErrorCode,
    cleanup_count: int,
) -> None:
    client = FakeDockerClient(fail_at={stage})
    sandbox = make_sandbox(client)

    with pytest.raises(DockerSandboxError) as captured:
        sandbox.prepare(make_task(tmp_path))

    assert captured.value.code is expected_code
    assert captured.value.error_type is ErrorType.SANDBOX_ERROR
    assert client.events.count("remove") == cleanup_count
    assert sandbox.lifecycle is SandboxLifecycle.NEW
    assert sandbox.preparation is None


def test_prepare_reports_cleanup_failure_instead_of_leaking_container(
    tmp_path: Path,
) -> None:
    client = FakeDockerClient(fail_at={"start", "remove"})

    with pytest.raises(DockerSandboxError) as captured:
        make_sandbox(client).prepare(make_task(tmp_path))

    assert captured.value.code is DockerSandboxErrorCode.CLEANUP_FAILED


def test_posture_mismatch_fails_closed_and_rolls_back(tmp_path: Path) -> None:
    client = FakeDockerClient()
    client.posture_override = {"network_mode": "bridge"}

    with pytest.raises(DockerSandboxError) as captured:
        make_sandbox(client).prepare(make_task(tmp_path))

    assert captured.value.code is DockerSandboxErrorCode.POSTURE_MISMATCH
    assert client.events[-1] == "remove"


@pytest.mark.parametrize(
    "override",
    [
        {"running": False},
        {"user": "0:0"},
        {"read_only_rootfs": False},
        {"privileged": True},
        {"cap_drop": ()},
        {"security_options": ()},
        {"nano_cpus": 0},
        {"memory_bytes": 0},
        {"pids_limit": 0},
        {"tmpfs": {}},
        {"mounts": (DockerMountInspection(mount_type="bind", destination="/workspace"),)},
        {"environment_names": ("OPENAI_API_KEY",)},
        {"labels": {}},
        {"workdir": "/"},
        {"image_id": f"sha256:{'e' * 64}"},
    ],
)
def test_every_inspected_security_control_is_enforced(
    tmp_path: Path,
    override: dict[str, object],
) -> None:
    client = FakeDockerClient()
    client.posture_override = override

    with pytest.raises(DockerSandboxError) as captured:
        make_sandbox(client).prepare(make_task(tmp_path))

    assert captured.value.code is DockerSandboxErrorCode.POSTURE_MISMATCH
    assert client.events[-1] == "remove"


def test_policy_or_source_failure_makes_no_docker_call(tmp_path: Path) -> None:
    denied_client = FakeDockerClient()
    denied = make_sandbox(denied_client, image_reference="alpine:latest")

    with pytest.raises(DockerSandboxError) as denied_error:
        denied.prepare(make_task(tmp_path))

    assert denied_error.value.code is DockerSandboxErrorCode.POLICY_DENIED
    assert denied_error.value.error_type is ErrorType.POLICY_DENIED
    assert denied_client.events == []

    missing_client = FakeDockerClient()
    missing_task = make_task(tmp_path / "missing")
    with pytest.raises(DockerSandboxError) as source_error:
        make_sandbox(missing_client).prepare(missing_task)

    assert source_error.value.code is DockerSandboxErrorCode.SOURCE_UNAVAILABLE
    assert missing_client.events == []


def test_remote_source_is_not_implicitly_cloned() -> None:
    client = FakeDockerClient()
    remote_task = make_task(Path(".")).model_copy(
        update={
            "repository_path": None,
            "repository_url": "https://example.com/repository.git",
        }
    )

    with pytest.raises(DockerSandboxError) as captured:
        make_sandbox(client).prepare(remote_task)

    assert captured.value.code is DockerSandboxErrorCode.SOURCE_UNAVAILABLE
    assert client.events == []


def test_destroy_is_retry_safe_and_rejects_wrong_handle(tmp_path: Path) -> None:
    client = FakeDockerClient()
    sandbox = make_sandbox(client)
    handle = sandbox.prepare(make_task(tmp_path))
    wrong_handle = handle.model_copy(update={"sandbox_id": "sbx-wrong"})

    with pytest.raises(DockerSandboxError) as captured:
        sandbox.destroy(wrong_handle)
    assert captured.value.code is DockerSandboxErrorCode.LIFECYCLE_ERROR

    sandbox.destroy(handle)
    sandbox.destroy(handle)

    assert sandbox.lifecycle is SandboxLifecycle.DESTROYED
    assert client.events.count("remove") == 1


def test_failed_destroy_can_be_retried(tmp_path: Path) -> None:
    client = FakeDockerClient()
    sandbox = make_sandbox(client)
    handle = sandbox.prepare(make_task(tmp_path))
    client.fail_at.add("remove")

    with pytest.raises(DockerSandboxError) as captured:
        sandbox.destroy(handle)
    assert captured.value.code is DockerSandboxErrorCode.CLEANUP_FAILED
    assert sandbox.lifecycle is SandboxLifecycle.PREPARED

    client.fail_at.remove("remove")
    sandbox.destroy(handle)
    assert sandbox.lifecycle.value == SandboxLifecycle.DESTROYED.value


def test_prepare_and_run_obey_lifecycle(tmp_path: Path) -> None:
    client = FakeDockerClient()
    sandbox = make_sandbox(client)
    task = make_task(tmp_path)
    handle = sandbox.prepare(task)

    with pytest.raises(DockerSandboxError) as repeated:
        sandbox.prepare(task)
    assert repeated.value.code is DockerSandboxErrorCode.LIFECYCLE_ERROR

    request = RunRequest(
        call_id=uuid4(),
        command=task.test_commands[0],
    )
    observation = sandbox.run(handle, request)

    assert observation.exit_code == 0
    assert observation.error_type is None
    assert len(sandbox.collect_artifacts(handle)) == 2


def test_run_preserves_exit_code_previews_and_complete_artifacts(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    artifact_root = tmp_path / "artifacts"
    client = FakeDockerClient()
    client.exec_results.append(
        DockerExecResult(
            exit_code=7,
            duration_ms=23,
            stdout=b"abcdefgh",
            stderr=b"uvwxyz",
        )
    )
    sandbox = make_sandbox(
        client,
        artifact_root=artifact_root,
        settings_overrides={
            "sandbox_max_output_bytes": 4,
            "sandbox_max_artifact_bytes": 32,
            "patch_max_bytes": 32,
        },
    )
    task = make_task(repository)
    handle = sandbox.prepare(task)
    call_id = uuid4()

    observation = sandbox.run(
        handle,
        RunRequest(call_id=call_id, command=task.test_commands[0]),
    )

    assert observation.policy_decision is PolicyDecision.ALLOWED
    assert observation.exit_code == 7
    assert observation.error_type is None
    assert observation.duration_ms == 23
    assert observation.stdout_preview == "abcd"
    assert observation.stderr_preview == "uvwx"
    assert observation.truncated is True
    assert observation.stdout_ref is not None
    assert observation.stderr_ref is not None
    assert observation.stdout_ref.size_bytes == 8
    assert observation.stderr_ref.size_bytes == 6
    assert (artifact_root / handle.sandbox_id / f"{call_id}.stdout.txt").read_bytes() == b"abcdefgh"
    assert (artifact_root / handle.sandbox_id / f"{call_id}.stderr.txt").read_bytes() == b"uvwxyz"
    assert client.exec_specs[0].argv[-4:] == task.test_commands[0].argv
    assert client.exec_specs[0].workdir == DOCKER_WORKSPACE_ROOT

    sandbox.destroy(handle)
    assert len(sandbox.collect_artifacts(handle)) == 2


def test_run_policy_denial_never_calls_docker_or_writes_artifacts(
    tmp_path: Path,
) -> None:
    client = FakeDockerClient()
    sandbox = make_sandbox(client)
    task = make_task(tmp_path)
    handle = sandbox.prepare(task)
    events_before = tuple(client.events)

    observation = sandbox.run(
        handle,
        RunRequest(
            call_id=uuid4(),
            command=CommandSpec(argv=("python", "-c", "print('not configured')")),
        ),
    )

    assert observation.policy_decision is PolicyDecision.DENIED
    assert observation.error_type is ErrorType.POLICY_DENIED
    assert observation.exit_code is None
    assert tuple(client.events) == events_before
    assert sandbox.collect_artifacts(handle) == ()


def test_run_rejects_denied_cwd_before_docker_exec(tmp_path: Path) -> None:
    command = CommandSpec(
        argv=("python", "-m", "pytest", "-q"),
        cwd="artifacts/output",
    )
    task = make_task(tmp_path).model_copy(update={"test_commands": (command,)})
    client = FakeDockerClient()
    sandbox = make_sandbox(client)
    handle = sandbox.prepare(task)

    observation = sandbox.run(
        handle,
        RunRequest(call_id=uuid4(), command=command),
    )

    assert observation.policy_decision is PolicyDecision.DENIED
    assert "execute" not in client.events


def test_timeout_or_artifact_overflow_aborts_container_but_keeps_evidence(
    tmp_path: Path,
) -> None:
    for result, expected_error in (
        (
            DockerExecResult(
                exit_code=None,
                duration_ms=1_000,
                stdout=b"partial",
                stderr=b"",
                timed_out=True,
            ),
            ErrorType.TOOL_TIMEOUT,
        ),
        (
            DockerExecResult(
                exit_code=None,
                duration_ms=9,
                stdout=b"12345678",
                stderr=b"",
                artifact_limit_exceeded=True,
            ),
            ErrorType.TOOL_EXECUTION_ERROR,
        ),
    ):
        repository = tmp_path / expected_error.value
        repository.mkdir()
        client = FakeDockerClient()
        client.exec_results.append(result)
        sandbox = make_sandbox(
            client,
            artifact_root=tmp_path / f"{expected_error.value}-artifacts",
            settings_overrides={
                "sandbox_max_output_bytes": 4,
                "sandbox_max_artifact_bytes": 8,
                "patch_max_bytes": 8,
            },
        )
        task = make_task(repository)
        handle = sandbox.prepare(task)

        observation = sandbox.run(
            handle,
            RunRequest(call_id=uuid4(), command=task.test_commands[0]),
        )

        assert observation.error_type is expected_error
        assert observation.exit_code is None
        assert sandbox.lifecycle is SandboxLifecycle.DESTROYED
        assert client.events[-1] == "remove"
        assert len(sandbox.collect_artifacts(handle)) == 2


def test_read_file_uses_policy_and_returns_requested_lines(tmp_path: Path) -> None:
    client = FakeDockerClient()
    client.exec_results.append(
        DockerExecResult(
            exit_code=0,
            duration_ms=4,
            stdout=b"second\nthird\n",
            stderr=b"",
        )
    )
    sandbox = make_sandbox(client)
    task = make_task(tmp_path)
    handle = sandbox.prepare(task)

    observation = sandbox.read_file(
        handle,
        ReadFileRequest(
            call_id=uuid4(),
            relative_path="pkg/module.py",
            start_line=2,
            end_line=3,
        ),
    )

    assert observation.exit_code == 0
    assert observation.stdout_preview == "second\nthird\n"
    assert observation.error_type is None
    assert client.exec_specs[-1].argv[-3:] == ("pkg/module.py", "2", "3")

    events_before = tuple(client.events)
    denied = sandbox.read_file(
        handle,
        ReadFileRequest(call_id=uuid4(), relative_path=".env"),
    )
    assert denied.policy_decision is PolicyDecision.DENIED
    assert tuple(client.events) == events_before


def test_read_file_runtime_no_follow_failure_is_path_violation(tmp_path: Path) -> None:
    client = FakeDockerClient()
    client.exec_results.append(
        DockerExecResult(
            exit_code=40,
            duration_ms=2,
            stdout=b"",
            stderr=b"",
        )
    )
    sandbox = make_sandbox(client)
    task = make_task(tmp_path)
    handle = sandbox.prepare(task)

    observation = sandbox.read_file(
        handle,
        ReadFileRequest(call_id=uuid4(), relative_path="link.py"),
    )

    assert observation.exit_code == 40
    assert observation.error_type is ErrorType.PATH_VIOLATION


def test_patch_is_staged_then_applied_with_verified_structured_output(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    client = FakeDockerClient()
    patch_sha256 = hashlib.sha256(VALID_PATCH).hexdigest()
    client.exec_results.append(
        DockerExecResult(
            exit_code=0,
            duration_ms=8,
            stdout=json.dumps(
                {
                    "files_changed": ["pkg/module.py"],
                    "patch_sha256": patch_sha256,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode(),
            stderr=b"",
        )
    )
    sandbox = make_sandbox(client, artifact_root=tmp_path / "artifacts")
    task = make_task(repository)
    handle = sandbox.prepare(task)
    call_id = uuid4()

    patch_ref = sandbox.stage_patch(handle, call_id, VALID_PATCH)
    observation = sandbox.apply_patch(
        handle,
        ApplyPatchRequest(call_id=call_id, patch_ref=patch_ref),
    )

    assert observation.exit_code == 0
    assert observation.error_type is None
    assert client.events[-2:] == ["copy-artifact", "execute"]
    assert client.copied_artifacts[f"{call_id}.patch.diff"] == VALID_PATCH
    assert client.exec_specs[-1].argv[-5:] == (
        str(patch_ref.size_bytes),
        patch_ref.sha256,
        "20",
        "100",
        "2000",
    )
    assert tuple(item.media_type for item in sandbox.collect_artifacts(handle)) == (
        "text/x-diff; charset=utf-8",
        "text/plain; charset=utf-8",
        "text/plain; charset=utf-8",
    )


def test_patch_conflict_preserves_live_sandbox_for_recovery(tmp_path: Path) -> None:
    client = FakeDockerClient()
    client.exec_results.append(
        DockerExecResult(
            exit_code=45,
            duration_ms=3,
            stdout=b"",
            stderr=b"repopilot_patch_error:apply_conflict\n",
        )
    )
    sandbox = make_sandbox(client)
    task = make_task(tmp_path)
    handle = sandbox.prepare(task)
    call_id = uuid4()
    patch_ref = sandbox.stage_patch(handle, call_id, VALID_PATCH)

    observation = sandbox.apply_patch(
        handle,
        ApplyPatchRequest(call_id=call_id, patch_ref=patch_ref),
    )

    assert observation.error_type is ErrorType.PATCH_ERROR
    assert observation.exit_code == 45
    assert observation.stderr_preview == (
        f"repopilot_patch_error:{PatchErrorCode.APPLY_CONFLICT.value}\n"
    )
    assert sandbox.lifecycle is SandboxLifecycle.PREPARED
    follow_up = sandbox.run(
        handle,
        RunRequest(call_id=uuid4(), command=task.test_commands[0]),
    )
    assert follow_up.exit_code == 0


def test_unproven_patch_rollback_aborts_the_container(tmp_path: Path) -> None:
    client = FakeDockerClient()
    client.exec_results.append(
        DockerExecResult(
            exit_code=46,
            duration_ms=3,
            stdout=b"",
            stderr=b"repopilot_patch_error:rollback_failed\n",
        )
    )
    sandbox = make_sandbox(client)
    task = make_task(tmp_path)
    handle = sandbox.prepare(task)
    call_id = uuid4()
    patch_ref = sandbox.stage_patch(handle, call_id, VALID_PATCH)

    observation = sandbox.apply_patch(
        handle,
        ApplyPatchRequest(call_id=call_id, patch_ref=patch_ref),
    )

    assert observation.error_type is ErrorType.PATCH_ERROR
    assert sandbox.lifecycle is SandboxLifecycle.DESTROYED
    assert client.events[-1] == "remove"


def test_patch_path_policy_denial_occurs_before_docker_copy(tmp_path: Path) -> None:
    denied_patch = (
        b"diff --git a/.git/config b/.git/config\n"
        b"--- a/.git/config\n"
        b"+++ b/.git/config\n"
        b"@@ -1 +1 @@\n"
        b"-old\n"
        b"+new\n"
    )
    client = FakeDockerClient()
    sandbox = make_sandbox(client)
    task = make_task(tmp_path)
    handle = sandbox.prepare(task)
    call_id = uuid4()
    patch_ref = sandbox.stage_patch(handle, call_id, denied_patch)
    events_before = tuple(client.events)

    observation = sandbox.apply_patch(
        handle,
        ApplyPatchRequest(call_id=call_id, patch_ref=patch_ref),
    )

    assert observation.policy_decision is PolicyDecision.DENIED
    assert observation.error_type is ErrorType.POLICY_DENIED
    assert tuple(client.events) == events_before


def test_unregistered_or_tampered_patch_ref_never_reaches_docker(
    tmp_path: Path,
) -> None:
    client = FakeDockerClient()
    sandbox = make_sandbox(client)
    task = make_task(tmp_path)
    handle = sandbox.prepare(task)
    call_id = uuid4()
    original = sandbox.stage_patch(handle, call_id, VALID_PATCH)
    tampered = ArtifactRef(
        relative_path=original.relative_path,
        sha256="f" * 64,
        size_bytes=original.size_bytes,
        media_type=original.media_type,
    )
    events_before = tuple(client.events)

    observation = sandbox.apply_patch(
        handle,
        ApplyPatchRequest(call_id=call_id, patch_ref=tampered),
    )

    assert observation.error_type is ErrorType.PATCH_ERROR
    assert observation.stderr_preview == (
        f"repopilot_patch_error:{PatchErrorCode.ARTIFACT_MISMATCH.value}\n"
    )
    assert tuple(client.events) == events_before


def test_duplicate_call_id_is_rejected_before_second_execution(tmp_path: Path) -> None:
    client = FakeDockerClient()
    sandbox = make_sandbox(client)
    task = make_task(tmp_path)
    handle = sandbox.prepare(task)
    request = RunRequest(call_id=uuid4(), command=task.test_commands[0])
    sandbox.run(handle, request)

    with pytest.raises(DockerSandboxError) as captured:
        sandbox.run(handle, request)

    assert captured.value.code is DockerSandboxErrorCode.DUPLICATE_CALL
    assert client.events.count("execute") == 1


def test_setup_only_network_is_connected_then_proven_disconnected(
    tmp_path: Path,
) -> None:
    setup_command = CommandSpec(argv=("python", "-c", "print('setup')"))
    task = make_task(tmp_path).model_copy(update={"setup_commands": (setup_command,)})
    client = FakeDockerClient()
    sandbox = make_sandbox(
        client,
        settings_overrides={"sandbox_network": "setup_only"},
    )
    handle = sandbox.prepare(task)

    observation = sandbox.run(
        handle,
        RunRequest(
            call_id=uuid4(),
            command=setup_command,
            phase=SandboxCommandPhase.SETUP,
            network_requested=True,
        ),
    )

    assert observation.exit_code == 0
    assert client.events[-5:] == [
        "network-connect",
        "ping",
        "execute",
        "network-disconnect",
        "inspect",
    ]
    assert client.network_connected is False


def test_artifact_root_inside_repository_fails_and_rolls_back(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    client = FakeDockerClient()
    sandbox = make_sandbox(
        client,
        artifact_root=repository / "artifacts",
    )

    with pytest.raises(DockerSandboxError) as captured:
        sandbox.prepare(make_task(repository))

    assert captured.value.code is DockerSandboxErrorCode.ARTIFACT_IO_FAILED
    assert client.events[-1] == "remove"


def test_container_spec_rejects_weakened_posture() -> None:
    valid = DockerContainerSpec(
        name="repopilot-sbx-validation",
        image_reference=IMAGE,
        user="65532:65532",
        network_mode=SandboxNetworkMode.NONE,
        read_only_rootfs=True,
        privileged=False,
        cap_drop=("ALL",),
        security_options=("no-new-privileges",),
        cpu_limit=1.0,
        memory_limit_mb=1024,
        pids_limit=256,
        tmpfs_limit_mb=256,
        environment=dict(DOCKER_FIXED_ENVIRONMENT),
        labels={"com.repopilot.managed": "true"},
    )

    with pytest.raises(ValidationError):
        valid.model_copy(update={"privileged": True}, deep=True).__class__.model_validate(
            {**valid.model_dump(), "privileged": True}
        )


def test_controlled_tar_omits_secrets_and_symlinks_and_sets_sandbox_owner(
    tmp_path: Path,
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=canary", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("canary", encoding="utf-8")
    (tmp_path / "link").symlink_to(tmp_path / "pkg" / "module.py")
    output = io.BytesIO()

    result = _write_filtered_tar(tmp_path, output, max_bytes=1024)

    output.seek(0)
    with tarfile.open(fileobj=output, mode="r:") as archive:
        members = {member.name: member for member in archive.getmembers()}
    assert set(members) == {"pkg", "pkg/module.py"}
    assert all(member.uid == 65_532 and member.gid == 65_532 for member in members.values())
    assert all(member.mtime == DOCKER_ARCHIVE_MTIME_SECONDS for member in members.values())
    assert members["pkg"].mode & 0o700 == 0o700
    assert members["pkg/module.py"].mode & 0o600 == 0o600
    assert result.file_count == 1
    assert result.directory_count == 1
    assert result.skipped_denied_count == 2
    assert result.skipped_symlink_count == 1

    output.seek(0)
    with tarfile.open(fileobj=output, mode="r:") as archive:
        archive.extract("pkg/module.py", path=tmp_path / "extracted", filter="data")
    zip_info = zipfile.ZipInfo.from_file(
        tmp_path / "extracted/pkg/module.py",
        "pkg/module.py",
    )
    assert zip_info.date_time[:3] >= (1980, 1, 1)


def test_controlled_tar_rejects_workspace_larger_than_tmpfs(tmp_path: Path) -> None:
    (tmp_path / "large.bin").write_bytes(b"1234")

    with pytest.raises(DockerClientError, match="workspace copy"):
        _write_filtered_tar(tmp_path, io.BytesIO(), max_bytes=3)


def test_cli_remove_is_idempotent_only_for_explicit_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = DockerCliClient()

    def missing_container(
        *args: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=("docker",),
            returncode=1,
            stdout=b"",
            stderr=b"Error response from daemon: No such container: fixture",
        )

    monkeypatch.setattr(subprocess, "run", missing_container)
    client.remove_container(CONTAINER_ID)

    def daemon_failure(
        *args: object,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=("docker",),
            returncode=1,
            stdout=b"",
            stderr=b"Cannot connect to the Docker daemon",
        )

    monkeypatch.setattr(subprocess, "run", daemon_failure)
    with pytest.raises(DockerClientError, match="container cleanup"):
        client.remove_container(CONTAINER_ID)
