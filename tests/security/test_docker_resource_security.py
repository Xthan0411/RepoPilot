"""Real-Docker resource-abuse, termination, cleanup, and artifact proofs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from uuid import uuid4

import pytest

from repopilot.config import RepoPilotSettings
from repopilot.enums import ErrorType, SandboxLifecycle, SandboxNetworkMode
from repopilot.sandbox import (
    DockerCliClient,
    DockerClientError,
    DockerSandbox,
    RunRequest,
    SandboxHandle,
    SandboxPolicy,
)
from repopilot.schemas import ArtifactRef, CommandSpec, TaskSpec

pytestmark = [pytest.mark.security, pytest.mark.integration]

_IMAGE_ENVIRONMENT_NAME = "REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE"
_MEBIBYTE = 1024 * 1024

_CPU_PRESSURE = """\
import os

os.write(1, b"cpu-pressure-started\\n")
value = 1
while True:
    value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
"""

_PID_PRESSURE = """\
import errno
import json
import os
import signal
import time

children = []
failure_errno = None
try:
    for _ in range(64):
        try:
            child = os.fork()
        except OSError as error:
            failure_errno = error.errno
            break
        if child == 0:
            time.sleep(30)
            os._exit(0)
        children.append(child)
finally:
    for child in children:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for child in children:
        try:
            os.waitpid(child, 0)
        except ChildProcessError:
            pass

result = {
    "failure_errno": failure_errno,
    "limit_hit": failure_errno == errno.EAGAIN,
    "spawned": len(children),
}
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if result["limit_hit"] and 0 < result["spawned"] < 64 else 93)
"""

_MEMORY_PRESSURE = """\
import json

chunks = []
try:
    for _ in range(64):
        chunks.append(bytearray(8 * 1024 * 1024))
except MemoryError:
    print(json.dumps({"limit_hit": True, "allocated_chunks": len(chunks)}, sort_keys=True))
    raise SystemExit(0)

print(json.dumps({"limit_hit": False, "allocated_chunks": len(chunks)}, sort_keys=True))
raise SystemExit(93)
"""

_LOG_PRESSURE = """\
import os

stdout_block = b"O" * 1024
stderr_block = b"E" * 1024
while True:
    os.write(1, stdout_block)
    os.write(2, stderr_block)
"""


@dataclass(frozen=True, slots=True)
class _PreparedAttack:
    sandbox: DockerSandbox
    client: DockerCliClient
    handle: SandboxHandle
    container_id: str
    artifact_root: Path


def _image() -> str:
    image = os.environ.get(_IMAGE_ENVIRONMENT_NAME)
    if image is None:
        pytest.skip("set REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE to a local digest-pinned Python image")
    return image


def _settings(**overrides: object) -> RepoPilotSettings:
    values: dict[str, object] = {
        "sandbox_network": SandboxNetworkMode.NONE,
        "sandbox_cpu_limit": 0.25,
        "sandbox_memory_limit_mb": 128,
        "sandbox_pids_limit": 16,
        "sandbox_tmpfs_limit_mb": 16,
        "sandbox_command_timeout_seconds": 5,
        "sandbox_max_output_bytes": 1_024,
        "sandbox_max_artifact_bytes": 8_192,
        "patch_max_bytes": 8_192,
    }
    values.update(overrides)
    return RepoPilotSettings.model_validate(values)


def _prepare_attack(
    tmp_path: Path,
    *,
    attack_name: str,
    sources: dict[str, str],
    commands: tuple[CommandSpec, ...],
    settings: RepoPilotSettings | None = None,
) -> _PreparedAttack:
    repository = tmp_path / f"{attack_name}-repository"
    repository.mkdir()
    for name, source in sources.items():
        (repository / name).write_text(source, encoding="utf-8")
    task = TaskSpec(
        task_id=f"docker-resource-{attack_name}",
        repository_path=str(repository),
        base_commit="d" * 40,
        issue_title=f"Bound {attack_name} resource abuse",
        issue_body="Prove resource pressure terminates inside Docker and cleans safely.",
        test_commands=commands,
        timeout_seconds=20,
    )
    effective_settings = settings or _settings()
    client = DockerCliClient()
    artifact_root = tmp_path / f"{attack_name}-trusted-artifacts"
    sandbox = DockerSandbox(
        policy=SandboxPolicy.from_settings(effective_settings),
        image_reference=_image(),
        artifact_root=artifact_root,
        client=client,
    )
    handle = sandbox.prepare(task)
    assert sandbox.preparation is not None
    return _PreparedAttack(
        sandbox=sandbox,
        client=client,
        handle=handle,
        container_id=sandbox.preparation.container_id,
        artifact_root=artifact_root,
    )


def _artifact_bytes(
    prepared: _PreparedAttack,
    artifact: ArtifactRef,
) -> bytes:
    path = prepared.artifact_root / prepared.handle.sandbox_id / Path(artifact.relative_path).name
    content = path.read_bytes()
    assert len(content) == artifact.size_bytes
    assert hashlib.sha256(content).hexdigest() == artifact.sha256
    return content


def _assert_container_absent(prepared: _PreparedAttack) -> None:
    with pytest.raises(DockerClientError):
        prepared.client.inspect_container(prepared.container_id)


def test_cpu_pressure_hits_deadline_destroys_container_and_keeps_output(
    tmp_path: Path,
) -> None:
    command = CommandSpec(argv=("python", "cpu_pressure.py"), timeout_seconds=1)
    prepared = _prepare_attack(
        tmp_path,
        attack_name="cpu-timeout",
        sources={"cpu_pressure.py": _CPU_PRESSURE},
        commands=(command,),
    )
    assert prepared.sandbox.preparation is not None
    inspection = prepared.sandbox.preparation.inspection
    assert inspection.nano_cpus == 250_000_000
    assert inspection.memory_bytes == 128 * _MEBIBYTE
    assert inspection.pids_limit == 16

    started = monotonic()
    try:
        observation = prepared.sandbox.run(
            prepared.handle,
            RunRequest(call_id=uuid4(), command=command),
        )
        elapsed = monotonic() - started

        assert observation.error_type is ErrorType.TOOL_TIMEOUT
        assert observation.exit_code is None
        assert observation.duration_ms >= 900
        assert elapsed < 5
        assert observation.stdout_ref is not None
        assert _artifact_bytes(prepared, observation.stdout_ref) == b"cpu-pressure-started\n"
        assert prepared.sandbox.lifecycle is SandboxLifecycle.DESTROYED
        assert len(prepared.sandbox.collect_artifacts(prepared.handle)) == 2
    finally:
        prepared.sandbox.destroy(prepared.handle)

    _assert_container_absent(prepared)


def test_pid_pressure_hits_cgroup_limit_reaps_children_and_recovers(
    tmp_path: Path,
) -> None:
    pressure = CommandSpec(argv=("python", "pid_pressure.py"))
    health = CommandSpec(argv=("python", "-c", "print('pid-capacity-recovered')"))
    prepared = _prepare_attack(
        tmp_path,
        attack_name="pid-limit",
        sources={"pid_pressure.py": _PID_PRESSURE},
        commands=(pressure, health),
    )
    assert prepared.sandbox.preparation is not None
    assert prepared.sandbox.preparation.inspection.pids_limit == 16

    try:
        observation = prepared.sandbox.run(
            prepared.handle,
            RunRequest(call_id=uuid4(), command=pressure),
        )
        assert observation.error_type is None
        assert observation.exit_code == 0
        assert observation.stdout_preview is not None
        evidence = json.loads(observation.stdout_preview)
        assert evidence["limit_hit"] is True
        assert evidence["failure_errno"] == 11
        assert 0 < evidence["spawned"] < 64

        recovered = prepared.sandbox.run(
            prepared.handle,
            RunRequest(call_id=uuid4(), command=health),
        )
        assert recovered.exit_code == 0
        assert recovered.stdout_preview == "pid-capacity-recovered\n"
        assert prepared.client.inspect_container(prepared.container_id).running is True
    finally:
        prepared.sandbox.destroy(prepared.handle)

    assert len(prepared.sandbox.collect_artifacts(prepared.handle)) == 4
    for artifact in prepared.sandbox.collect_artifacts(prepared.handle):
        _artifact_bytes(prepared, artifact)
    _assert_container_absent(prepared)


def test_memory_pressure_cannot_cross_container_limit_and_cleans(
    tmp_path: Path,
) -> None:
    command = CommandSpec(argv=("python", "memory_pressure.py"))
    prepared = _prepare_attack(
        tmp_path,
        attack_name="memory-limit",
        sources={"memory_pressure.py": _MEMORY_PRESSURE},
        commands=(command,),
    )
    assert prepared.sandbox.preparation is not None
    assert prepared.sandbox.preparation.inspection.memory_bytes == 128 * _MEBIBYTE

    started = monotonic()
    try:
        observation = prepared.sandbox.run(
            prepared.handle,
            RunRequest(call_id=uuid4(), command=command),
        )
        elapsed = monotonic() - started

        assert observation.error_type is None
        assert observation.exit_code is not None
        assert observation.exit_code in {0, 137}
        assert elapsed < 8
        if observation.exit_code == 0:
            assert observation.stdout_preview is not None
            evidence = json.loads(observation.stdout_preview)
            assert evidence["limit_hit"] is True
            assert evidence["allocated_chunks"] < 64
    finally:
        prepared.sandbox.destroy(prepared.handle)

    assert len(prepared.sandbox.collect_artifacts(prepared.handle)) == 2
    for artifact in prepared.sandbox.collect_artifacts(prepared.handle):
        _artifact_bytes(prepared, artifact)
    _assert_container_absent(prepared)


def test_log_flood_hits_combined_artifact_limit_destroys_and_keeps_prefix(
    tmp_path: Path,
) -> None:
    command = CommandSpec(argv=("python", "log_pressure.py"))
    prepared = _prepare_attack(
        tmp_path,
        attack_name="log-limit",
        sources={"log_pressure.py": _LOG_PRESSURE},
        commands=(command,),
    )

    started = monotonic()
    try:
        observation = prepared.sandbox.run(
            prepared.handle,
            RunRequest(call_id=uuid4(), command=command),
        )
        elapsed = monotonic() - started

        assert observation.error_type is ErrorType.TOOL_EXECUTION_ERROR
        assert observation.exit_code is None
        assert observation.truncated is True
        assert elapsed < 5
        assert prepared.sandbox.lifecycle is SandboxLifecycle.DESTROYED
        artifacts = prepared.sandbox.collect_artifacts(prepared.handle)
        assert len(artifacts) == 2
        contents = tuple(_artifact_bytes(prepared, artifact) for artifact in artifacts)
        assert sum(map(len, contents)) == 8_192
        assert all(set(content).issubset({ord("E"), ord("O")}) for content in contents)
    finally:
        prepared.sandbox.destroy(prepared.handle)

    _assert_container_absent(prepared)
