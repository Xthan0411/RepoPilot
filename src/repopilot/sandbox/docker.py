"""Docker-backed Sandbox prepare/destroy lifecycle.

The host process only orchestrates Docker with fixed argv. Repository commands are
not executed on the host, and the Docker socket is never exposed to the task container.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import stat
import subprocess
import tarfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import IO, Protocol, Self, TypeVar, cast, runtime_checkable
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from repopilot.enums import (
    DockerSandboxErrorCode,
    ErrorType,
    PatchErrorCode,
    PolicyDecision,
    SandboxLifecycle,
    SandboxNetworkMode,
    SandboxPathOperation,
)
from repopilot.sandbox.artifacts import SandboxArtifactError, SandboxArtifactStore
from repopilot.sandbox.base import (
    ApplyPatchRequest,
    ReadFileRequest,
    RunRequest,
    SandboxError,
    SandboxHandle,
)
from repopilot.sandbox.patching import APPLY_PATCH_HELPER, COPY_PATCH_HELPER
from repopilot.sandbox.policy import SANDBOX_DENIED_PATH_ROOTS, SandboxPolicy
from repopilot.schemas import ArtifactRef, Observation, StrictModel, TaskSpec

DOCKER_LABEL_PREFIX = "com.repopilot"
DOCKER_WORKSPACE_ROOT = "/workspace"
DOCKER_REPOSITORY_ROOT = f"{DOCKER_WORKSPACE_ROOT}/repository"
DOCKER_ARTIFACT_ROOT = f"{DOCKER_WORKSPACE_ROOT}/artifacts"
DOCKER_COMMAND_TIMEOUT_SECONDS = 30
DOCKER_MAX_RESPONSE_BYTES = 2_000_000
DOCKER_EXEC_CHUNK_BYTES = 65_536
# Deterministic 1980-01-02 UTC: accepted by ZIP-based build backends even after
# a worst-case timezone conversion, unlike TarInfo's default Unix epoch.
DOCKER_ARCHIVE_MTIME_SECONDS = 315_619_200
DOCKER_SETUP_NETWORK = "bridge"
DOCKER_BOOTSTRAP_COMMAND = (
    "/bin/sh",
    "-c",
    ("mkdir -p /workspace/repository /workspace/artifacts && exec sleep 86400"),
)
DOCKER_FIXED_ENVIRONMENT = {
    "CI": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
}
DOCKER_SENSITIVE_ENVIRONMENT_NAMES = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_OPENAI_API_KEY",
        "DOCKER_AUTH_CONFIG",
        "GITHUB_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "OPENAI_API_KEY",
        "SSH_AUTH_SOCK",
    }
)
_MEBIBYTE = 1024 * 1024
_NANO_CPUS_PER_CPU = 1_000_000_000
_COPY_EXCLUDED_NAMES = frozenset({*SANDBOX_DENIED_PATH_ROOTS, ".env"})
_T = TypeVar("_T")

_READ_FILE_HELPER = """\
import os
import stat
import sys

root_fd = -1
current_fd = -1
try:
    parts = sys.argv[1].split("/")
    start = int(sys.argv[2])
    end = None if sys.argv[3] == "0" else int(sys.argv[3])
    root_fd = os.open("/workspace/repository", os.O_RDONLY | os.O_DIRECTORY)
    current_fd = root_fd
    for part in parts[:-1]:
        next_fd = os.open(
            part,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=current_fd,
        )
        if current_fd != root_fd:
            os.close(current_fd)
        current_fd = next_fd
    file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current_fd)
    if not stat.S_ISREG(os.fstat(file_fd).st_mode):
        raise OSError
    with os.fdopen(file_fd, "rb") as source:
        for line_number, line in enumerate(source, start=1):
            if line_number >= start and (end is None or line_number <= end):
                sys.stdout.buffer.write(line)
            if end is not None and line_number >= end:
                break
except (OSError, ValueError):
    raise SystemExit(40) from None
finally:
    if current_fd >= 0 and current_fd != root_fd:
        os.close(current_fd)
    if root_fd >= 0:
        os.close(root_fd)
"""

_EXEC_COMMAND_HELPER = """\
import os
import sys

root_fd = -1
current_fd = -1
try:
    root_fd = os.open("/workspace/repository", os.O_RDONLY | os.O_DIRECTORY)
    current_fd = root_fd
    if sys.argv[1]:
        for part in sys.argv[1].split("/"):
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
    os.fchdir(current_fd)
    os.execvp(sys.argv[2], sys.argv[2:])
except (OSError, ValueError):
    raise SystemExit(125) from None
finally:
    if current_fd >= 0 and current_fd != root_fd:
        os.close(current_fd)
    if root_fd >= 0:
        os.close(root_fd)
"""


class DockerSandboxError(SandboxError):
    """Safe Docker lifecycle error with a stable machine-readable code."""

    def __init__(
        self,
        code: DockerSandboxErrorCode,
        *,
        error_type: ErrorType = ErrorType.SANDBOX_ERROR,
    ) -> None:
        super().__init__(_DOCKER_SANDBOX_ERROR_MESSAGES[code], error_type=error_type)
        self.code = code


class DockerClientError(RuntimeError):
    """Sanitized Docker CLI failure that never includes daemon stderr."""

    def __init__(self, operation: str) -> None:
        super().__init__(f"Docker {operation} failed")
        self.operation = operation


_DOCKER_SANDBOX_ERROR_MESSAGES = {
    DockerSandboxErrorCode.LIFECYCLE_ERROR: "sandbox lifecycle operation is invalid",
    DockerSandboxErrorCode.POLICY_DENIED: "sandbox preparation was denied by policy",
    DockerSandboxErrorCode.SOURCE_UNAVAILABLE: "task repository source is unavailable",
    DockerSandboxErrorCode.DAEMON_UNAVAILABLE: "Docker daemon is unavailable",
    DockerSandboxErrorCode.IMAGE_UNAVAILABLE: "sandbox image is unavailable or not immutable",
    DockerSandboxErrorCode.CREATE_FAILED: "sandbox container creation failed",
    DockerSandboxErrorCode.START_FAILED: "sandbox container start failed",
    DockerSandboxErrorCode.WORKSPACE_COPY_FAILED: "controlled workspace copy failed",
    DockerSandboxErrorCode.INSPECT_FAILED: "sandbox container inspection failed",
    DockerSandboxErrorCode.POSTURE_MISMATCH: (
        "created container does not match the approved security posture"
    ),
    DockerSandboxErrorCode.DUPLICATE_CALL: "sandbox call ID has already been used",
    DockerSandboxErrorCode.EXECUTION_FAILED: "sandbox command execution failed",
    DockerSandboxErrorCode.ARTIFACT_IO_FAILED: "sandbox artifact persistence failed",
    DockerSandboxErrorCode.NETWORK_TRANSITION_FAILED: ("sandbox setup network transition failed"),
    DockerSandboxErrorCode.OPERATION_UNAVAILABLE: (
        "sandbox operation is not available in the prepare/destroy adapter"
    ),
    DockerSandboxErrorCode.CLEANUP_FAILED: "sandbox resource cleanup failed",
}


class DockerImageIdentity(StrictModel):
    """Immutable image identity resolved by the daemon."""

    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    repository_digests: tuple[str, ...] = ()


class DockerContainerSpec(StrictModel):
    """Fully compiled create request; no host bind-mount field exists by design."""

    name: str = Field(pattern=r"^repopilot-[a-z0-9-]{1,48}$")
    image_reference: str = Field(min_length=1, max_length=512)
    user: str = Field(pattern=r"^[1-9][0-9]{0,4}:[1-9][0-9]{0,4}$")
    network_mode: SandboxNetworkMode
    read_only_rootfs: bool
    privileged: bool
    cap_drop: tuple[str, ...]
    security_options: tuple[str, ...]
    cpu_limit: float = Field(gt=0.0, allow_inf_nan=False)
    memory_limit_mb: int = Field(gt=0)
    pids_limit: int = Field(gt=0)
    tmpfs_limit_mb: int = Field(gt=0)
    workdir: str = DOCKER_WORKSPACE_ROOT
    environment: dict[str, str]
    labels: dict[str, str]
    command: tuple[str, ...] = DOCKER_BOOTSTRAP_COMMAND

    @model_validator(mode="after")
    def validate_fixed_posture(self) -> Self:
        if (
            self.network_mode is not SandboxNetworkMode.NONE
            or not self.read_only_rootfs
            or self.privileged
            or self.cap_drop != ("ALL",)
            or self.security_options != ("no-new-privileges",)
            or self.workdir != DOCKER_WORKSPACE_ROOT
            or self.environment != DOCKER_FIXED_ENVIRONMENT
            or self.command != DOCKER_BOOTSTRAP_COMMAND
        ):
            raise ValueError("Docker container spec must preserve the fixed security posture")
        return self


class DockerMountInspection(StrictModel):
    """One normalized mount returned by Docker inspect."""

    mount_type: str = Field(min_length=1)
    destination: str = Field(min_length=1)


class DockerContainerInspection(StrictModel):
    """Security-relevant subset of Docker inspect output."""

    container_id: str = Field(min_length=12)
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    running: bool
    user: str
    network_mode: str
    network_names: tuple[str, ...]
    read_only_rootfs: bool
    privileged: bool
    cap_drop: tuple[str, ...]
    security_options: tuple[str, ...]
    nano_cpus: int = Field(ge=0)
    memory_bytes: int = Field(ge=0)
    pids_limit: int = Field(ge=0)
    tmpfs: dict[str, str]
    mounts: tuple[DockerMountInspection, ...]
    environment_names: tuple[str, ...]
    labels: dict[str, str]
    workdir: str


class WorkspaceCopyResult(StrictModel):
    """Bounded, non-sensitive summary of the controlled repository copy."""

    file_count: int = Field(ge=0)
    directory_count: int = Field(ge=0)
    byte_count: int = Field(ge=0)
    skipped_symlink_count: int = Field(ge=0)
    skipped_denied_count: int = Field(ge=0)


class DockerPreparationRecord(StrictModel):
    """Post-create evidence retained for lifecycle tests and later audit traces."""

    container_id: str = Field(min_length=12)
    container_name: str
    policy_request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inspection: DockerContainerInspection
    workspace_copy: WorkspaceCopyResult


class DockerExecSpec(StrictModel):
    """One shell-free Docker exec request with hard output and wall-clock limits."""

    argv: tuple[str, ...] = Field(min_length=1, max_length=128)
    workdir: str = Field(pattern=r"^/workspace(?:/repository(?:/[^\x00]+)?)?$")
    timeout_seconds: int = Field(gt=0, le=3_600)
    max_artifact_bytes: int = Field(gt=0, le=100_000_000)

    @model_validator(mode="after")
    def validate_argv(self) -> Self:
        if any(
            not argument or "\x00" in argument or len(argument) > 16_384 for argument in self.argv
        ):
            raise ValueError("Docker exec argv must contain bounded non-empty values")
        if self.workdir != DOCKER_WORKSPACE_ROOT:
            raise ValueError("Docker exec control-plane workdir must remain /workspace")
        return self


@dataclass(frozen=True, slots=True)
class DockerExecResult:
    """Bounded byte streams and process outcome from one Docker exec."""

    exit_code: int | None
    duration_ms: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    artifact_limit_exceeded: bool = False

    def __post_init__(self) -> None:
        if self.duration_ms < 0:
            raise ValueError("Docker exec duration must be non-negative")
        if self.timed_out or self.artifact_limit_exceeded:
            if self.exit_code is not None:
                raise ValueError("aborted Docker exec cannot have an exit code")
        elif self.exit_code is None:
            raise ValueError("completed Docker exec requires an exit code")


@runtime_checkable
class DockerClient(Protocol):
    """Injectable daemon client used by lifecycle unit tests and the CLI adapter."""

    def ping(self) -> None:
        """Prove that a Docker daemon is reachable."""

    def resolve_image(self, image_reference: str) -> DockerImageIdentity:
        """Resolve an already-local immutable image without pulling it."""

    def create_container(self, spec: DockerContainerSpec) -> str:
        """Create but do not start a container."""

    def start_container(self, container_id: str) -> None:
        """Start one created container."""

    def copy_directory(
        self,
        container_id: str,
        source: Path,
        *,
        max_bytes: int,
    ) -> WorkspaceCopyResult:
        """Copy a filtered repository snapshot into managed container storage."""

    def copy_artifact(
        self,
        container_id: str,
        name: str,
        content: bytes,
        *,
        max_bytes: int,
    ) -> None:
        """Copy one bounded trusted artifact without a host bind mount."""

    def inspect_container(self, container_id: str) -> DockerContainerInspection:
        """Read back the effective security posture."""

    def execute(
        self,
        container_id: str,
        specification: DockerExecSpec,
    ) -> DockerExecResult:
        """Execute argv directly inside the container with bounded streaming output."""

    def connect_setup_network(self, container_id: str) -> None:
        """Temporarily connect the configured setup network."""

    def disconnect_setup_network(self, container_id: str) -> None:
        """Remove the temporary setup network."""

    def remove_container(self, container_id: str) -> None:
        """Force-remove a container; absence is treated as success."""


class DockerCliClient:
    """Docker Engine client implemented with fixed, shell-free CLI argv."""

    def __init__(
        self,
        *,
        executable: str = "docker",
        timeout_seconds: int = DOCKER_COMMAND_TIMEOUT_SECONDS,
        monotonic_clock: Callable[[], float] = monotonic,
    ) -> None:
        if not executable or "\x00" in executable:
            raise ValueError("Docker executable must be bounded text")
        if timeout_seconds <= 0:
            raise ValueError("Docker timeout must be positive")
        self._executable = executable
        self._timeout_seconds = timeout_seconds
        self._monotonic = monotonic_clock

    def ping(self) -> None:
        self._run(("version", "--format", "{{json .Server.Version}}"), operation="ping")

    def resolve_image(self, image_reference: str) -> DockerImageIdentity:
        payload = self._run(
            ("image", "inspect", image_reference),
            operation="image inspection",
        )
        try:
            records = cast(list[object], json.loads(payload))
            record = cast(dict[str, object], records[0])
            image_id = cast(str, record["Id"])
            repository_digests = tuple(sorted(cast(list[str], record.get("RepoDigests") or [])))
            identity = DockerImageIdentity(
                image_id=image_id,
                repository_digests=repository_digests,
            )
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise DockerClientError("image inspection") from error
        if image_reference.startswith("sha256:"):
            if image_reference != identity.image_id:
                raise DockerClientError("image identity verification")
        elif image_reference not in identity.repository_digests:
            raise DockerClientError("image identity verification")
        return identity

    def create_container(self, spec: DockerContainerSpec) -> str:
        arguments = [
            "container",
            "create",
            "--name",
            spec.name,
            "--network",
            spec.network_mode.value,
            "--user",
            spec.user,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--cpus",
            _format_cpu_limit(spec.cpu_limit),
            "--memory",
            f"{spec.memory_limit_mb}m",
            "--pids-limit",
            str(spec.pids_limit),
            "--tmpfs",
            _tmpfs_option(spec),
            "--workdir",
            spec.workdir,
        ]
        for name, value in sorted(spec.environment.items()):
            arguments.extend(("--env", f"{name}={value}"))
        for name, value in sorted(spec.labels.items()):
            arguments.extend(("--label", f"{name}={value}"))
        arguments.extend(("--entrypoint", spec.command[0], spec.image_reference))
        arguments.extend(spec.command[1:])
        container_id = self._run(tuple(arguments), operation="container creation").strip()
        if len(container_id) < 12 or any(
            character not in "0123456789abcdef" for character in container_id
        ):
            raise DockerClientError("container creation")
        return container_id

    def start_container(self, container_id: str) -> None:
        self._run(("container", "start", container_id), operation="container start")

    def copy_directory(
        self,
        container_id: str,
        source: Path,
        *,
        max_bytes: int,
    ) -> WorkspaceCopyResult:
        if max_bytes <= 0:
            raise DockerClientError("workspace copy")
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                (
                    self._executable,
                    "container",
                    "exec",
                    "--interactive",
                    container_id,
                    "tar",
                    "-x",
                    "-C",
                    DOCKER_REPOSITORY_ROOT,
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if process.stdin is None:
                raise DockerClientError("workspace copy")
            result = _write_filtered_tar(source, process.stdin, max_bytes=max_bytes)
            process.stdin.close()
            process.stdin = None
            stdout, stderr = process.communicate(timeout=self._timeout_seconds)
            if (
                process.returncode != 0
                or len(stdout) > DOCKER_MAX_RESPONSE_BYTES
                or len(stderr) > DOCKER_MAX_RESPONSE_BYTES
            ):
                raise DockerClientError("workspace copy")
            return result
        except (
            BrokenPipeError,
            FileNotFoundError,
            OSError,
            subprocess.SubprocessError,
            tarfile.TarError,
        ) as error:
            if process is not None:
                process.kill()
                process.communicate()
            raise DockerClientError("workspace copy") from error

    def copy_artifact(
        self,
        container_id: str,
        name: str,
        content: bytes,
        *,
        max_bytes: int,
    ) -> None:
        if (
            re.fullmatch(r"[0-9a-f-]{36}\.patch\.diff", name) is None
            or max_bytes <= 0
            or len(content) > max_bytes
        ):
            raise DockerClientError("artifact copy")
        try:
            completed = subprocess.run(
                (
                    self._executable,
                    "container",
                    "exec",
                    "--interactive",
                    "--workdir",
                    DOCKER_WORKSPACE_ROOT,
                    container_id,
                    "python",
                    "-I",
                    "-c",
                    COPY_PATCH_HELPER,
                    name,
                    str(max_bytes),
                    hashlib.sha256(content).hexdigest(),
                ),
                input=content,
                check=False,
                capture_output=True,
                timeout=self._timeout_seconds,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError) as error:
            raise DockerClientError("artifact copy") from error
        if (
            completed.returncode != 0
            or len(completed.stdout) > DOCKER_MAX_RESPONSE_BYTES
            or len(completed.stderr) > DOCKER_MAX_RESPONSE_BYTES
        ):
            raise DockerClientError("artifact copy")

    def inspect_container(self, container_id: str) -> DockerContainerInspection:
        payload = self._run(
            ("container", "inspect", container_id),
            operation="container inspection",
        )
        try:
            records = cast(list[object], json.loads(payload))
            record = cast(dict[str, object], records[0])
            config = cast(dict[str, object], record["Config"])
            host_config = cast(dict[str, object], record["HostConfig"])
            state = cast(dict[str, object], record["State"])
            network_settings = cast(dict[str, object], record.get("NetworkSettings") or {})
            networks = cast(dict[str, object], network_settings.get("Networks") or {})
            environment = cast(list[str], config.get("Env") or [])
            mounts = tuple(
                DockerMountInspection(
                    mount_type=cast(str, mount["Type"]),
                    destination=cast(str, mount["Destination"]),
                )
                for mount in cast(list[dict[str, object]], record.get("Mounts") or [])
            )
            return DockerContainerInspection(
                container_id=cast(str, record["Id"]),
                image_id=cast(str, record["Image"]),
                running=cast(bool, state["Running"]),
                user=cast(str, config.get("User") or ""),
                network_mode=cast(str, host_config.get("NetworkMode") or ""),
                network_names=tuple(sorted(networks)),
                read_only_rootfs=cast(bool, host_config.get("ReadonlyRootfs")),
                privileged=cast(bool, host_config.get("Privileged")),
                cap_drop=tuple(sorted(cast(list[str], host_config.get("CapDrop") or []))),
                security_options=tuple(
                    sorted(cast(list[str], host_config.get("SecurityOpt") or []))
                ),
                nano_cpus=cast(int, host_config.get("NanoCpus") or 0),
                memory_bytes=cast(int, host_config.get("Memory") or 0),
                pids_limit=cast(int, host_config.get("PidsLimit") or 0),
                tmpfs=cast(dict[str, str], host_config.get("Tmpfs") or {}),
                mounts=mounts,
                environment_names=tuple(
                    sorted(
                        {item.partition("=")[0] for item in environment if item.partition("=")[0]}
                    )
                ),
                labels=cast(dict[str, str], config.get("Labels") or {}),
                workdir=cast(str, config.get("WorkingDir") or ""),
            )
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise DockerClientError("container inspection") from error

    def execute(
        self,
        container_id: str,
        specification: DockerExecSpec,
    ) -> DockerExecResult:
        started = self._monotonic()
        process: subprocess.Popen[bytes] | None = None
        selector = selectors.DefaultSelector()
        stdout = bytearray()
        stderr = bytearray()
        timed_out = False
        artifact_limit_exceeded = False
        try:
            process = subprocess.Popen(
                (
                    self._executable,
                    "container",
                    "exec",
                    "--workdir",
                    specification.workdir,
                    container_id,
                    *specification.argv,
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if process.stdout is None or process.stderr is None:
                raise DockerClientError("command execution")
            selector.register(process.stdout, selectors.EVENT_READ, stdout)
            selector.register(process.stderr, selectors.EVENT_READ, stderr)
            deadline = started + specification.timeout_seconds
            total_bytes = 0
            while selector.get_map():
                remaining_seconds = deadline - self._monotonic()
                if remaining_seconds <= 0:
                    timed_out = True
                    break
                events = selector.select(timeout=min(remaining_seconds, 0.1))
                for key, _ in events:
                    target = cast(bytearray, key.data)
                    chunk = os.read(key.fd, DOCKER_EXEC_CHUNK_BYTES)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    available = specification.max_artifact_bytes - total_bytes
                    if len(chunk) > available:
                        target.extend(chunk[:available])
                        artifact_limit_exceeded = True
                        break
                    target.extend(chunk)
                    total_bytes += len(chunk)
                if artifact_limit_exceeded:
                    break
            if timed_out or artifact_limit_exceeded:
                process.kill()
                process.wait(timeout=self._timeout_seconds)
                exit_code = None
            else:
                exit_code = process.wait(timeout=self._timeout_seconds)
        except (
            FileNotFoundError,
            OSError,
            subprocess.SubprocessError,
        ) as error:
            if process is not None:
                process.kill()
                process.wait()
            raise DockerClientError("command execution") from error
        finally:
            selector.close()
        return DockerExecResult(
            exit_code=exit_code,
            duration_ms=max(0, round((self._monotonic() - started) * 1000)),
            stdout=bytes(stdout),
            stderr=bytes(stderr),
            timed_out=timed_out,
            artifact_limit_exceeded=artifact_limit_exceeded,
        )

    def connect_setup_network(self, container_id: str) -> None:
        self._run(
            ("network", "disconnect", "none", container_id),
            operation="offline network disconnect",
        )
        self._run(
            ("network", "connect", DOCKER_SETUP_NETWORK, container_id),
            operation="setup network connect",
        )

    def disconnect_setup_network(self, container_id: str) -> None:
        self._run(
            (
                "network",
                "disconnect",
                DOCKER_SETUP_NETWORK,
                container_id,
            ),
            operation="setup network disconnect",
        )
        self._run(
            ("network", "connect", "none", container_id),
            operation="offline network restore",
        )

    def remove_container(self, container_id: str) -> None:
        try:
            completed = subprocess.run(
                (
                    self._executable,
                    "container",
                    "rm",
                    "--force",
                    container_id,
                ),
                check=False,
                capture_output=True,
                timeout=self._timeout_seconds,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError) as error:
            raise DockerClientError("container cleanup") from error
        if (
            len(completed.stdout) > DOCKER_MAX_RESPONSE_BYTES
            or len(completed.stderr) > DOCKER_MAX_RESPONSE_BYTES
        ):
            raise DockerClientError("container cleanup")
        if completed.returncode != 0:
            if b"No such container:" in completed.stderr:
                return
            raise DockerClientError("container cleanup")

    def _run(
        self,
        arguments: Sequence[str],
        *,
        operation: str,
    ) -> str:
        try:
            completed = subprocess.run(
                (self._executable, *arguments),
                check=False,
                capture_output=True,
                timeout=self._timeout_seconds,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError) as error:
            raise DockerClientError(operation) from error
        if (
            completed.returncode != 0
            or len(completed.stdout) > DOCKER_MAX_RESPONSE_BYTES
            or len(completed.stderr) > DOCKER_MAX_RESPONSE_BYTES
        ):
            raise DockerClientError(operation)
        try:
            return completed.stdout.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DockerClientError(operation) from error


class DockerSandbox:
    """Single-task Docker adapter with bounded execution and persistent artifacts."""

    def __init__(
        self,
        *,
        policy: SandboxPolicy,
        image_reference: str,
        artifact_root: Path,
        client: DockerClient | None = None,
        now: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._policy = policy
        self._image_reference = image_reference
        self._client = client or DockerCliClient()
        self._artifact_store = SandboxArtifactStore(artifact_root)
        self._now = now or (lambda: datetime.now(UTC))
        self._id_factory = id_factory or (lambda: uuid4().hex[:20])
        self._lifecycle = SandboxLifecycle.NEW
        self._handle: SandboxHandle | None = None
        self._container_id: str | None = None
        self._preparation: DockerPreparationRecord | None = None
        self._task: TaskSpec | None = None
        self._used_call_ids: set[UUID] = set()

    @property
    def lifecycle(self) -> SandboxLifecycle:
        return self._lifecycle

    @property
    def preparation(self) -> DockerPreparationRecord | None:
        return self._preparation

    def prepare(self, task: TaskSpec) -> SandboxHandle:
        if self._lifecycle is not SandboxLifecycle.NEW:
            raise DockerSandboxError(DockerSandboxErrorCode.LIFECYCLE_ERROR)
        request = self._policy.default_preparation_request(
            task.task_id,
            image_reference=self._image_reference,
        )
        evaluation = self._policy.evaluate_preparation(
            request,
            expected_task_id=task.task_id,
        )
        if evaluation.decision is not PolicyDecision.ALLOWED:
            raise DockerSandboxError(
                DockerSandboxErrorCode.POLICY_DENIED,
                error_type=ErrorType.POLICY_DENIED,
            )
        source = _resolve_repository_source(task)
        suffix = self._id_factory()
        if (
            not suffix
            or len(suffix) > 44
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in suffix)
        ):
            raise DockerSandboxError(DockerSandboxErrorCode.LIFECYCLE_ERROR)
        sandbox_id = f"sbx-{suffix}"
        container_name = f"repopilot-{sandbox_id}"
        labels = _sandbox_labels(
            sandbox_id=sandbox_id,
            task=task,
            policy_request_sha256=evaluation.request_sha256,
        )
        specification = DockerContainerSpec(
            name=container_name,
            image_reference=request.image_reference,
            user=f"{request.run_as_uid}:{request.run_as_gid}",
            # Preparation is always offline. P4-S3 may temporarily grant network
            # to one setup command, but a prepared container starts disconnected.
            network_mode=SandboxNetworkMode.NONE,
            read_only_rootfs=request.read_only_rootfs,
            privileged=request.privileged,
            cap_drop=("ALL",),
            security_options=("no-new-privileges",),
            cpu_limit=request.resources.cpu_limit,
            memory_limit_mb=request.resources.memory_limit_mb,
            pids_limit=request.resources.pids_limit,
            tmpfs_limit_mb=request.resources.tmpfs_limit_mb,
            environment=dict(DOCKER_FIXED_ENVIRONMENT),
            labels=labels,
        )

        container_id: str | None = None
        try:
            self._call_client(self._client.ping, DockerSandboxErrorCode.DAEMON_UNAVAILABLE)
            image: DockerImageIdentity = self._call_client(
                lambda: self._client.resolve_image(request.image_reference),
                DockerSandboxErrorCode.IMAGE_UNAVAILABLE,
            )
            created_container_id: str = self._call_client(
                lambda: self._client.create_container(specification),
                DockerSandboxErrorCode.CREATE_FAILED,
            )
            container_id = created_container_id
            self._call_client(
                lambda: self._client.start_container(created_container_id),
                DockerSandboxErrorCode.START_FAILED,
            )
            copy_result: WorkspaceCopyResult = self._call_client(
                lambda: self._client.copy_directory(
                    created_container_id,
                    source,
                    max_bytes=request.resources.tmpfs_limit_mb * _MEBIBYTE,
                ),
                DockerSandboxErrorCode.WORKSPACE_COPY_FAILED,
            )
            inspection: DockerContainerInspection = self._call_client(
                lambda: self._client.inspect_container(created_container_id),
                DockerSandboxErrorCode.INSPECT_FAILED,
            )
            if not _posture_matches(
                inspection,
                specification,
                image=image,
                container_id=created_container_id,
            ):
                raise DockerSandboxError(DockerSandboxErrorCode.POSTURE_MISMATCH)
            try:
                self._artifact_store.prepare(
                    sandbox_id,
                    repository_source=source,
                )
            except SandboxArtifactError as artifact_error:
                raise DockerSandboxError(
                    DockerSandboxErrorCode.ARTIFACT_IO_FAILED
                ) from artifact_error
        except DockerSandboxError as error:
            if container_id is not None:
                try:
                    self._client.remove_container(container_id)
                except DockerClientError as cleanup_error:
                    raise DockerSandboxError(
                        DockerSandboxErrorCode.CLEANUP_FAILED
                    ) from cleanup_error
            raise error

        handle = SandboxHandle(
            sandbox_id=sandbox_id,
            task_id=task.task_id,
            repository_commit=task.base_commit,
            prepared_at=self._now(),
        )
        self._container_id = container_id
        self._handle = handle
        self._task = task
        self._preparation = DockerPreparationRecord(
            container_id=container_id,
            container_name=container_name,
            policy_request_sha256=evaluation.request_sha256,
            inspection=inspection,
            workspace_copy=copy_result,
        )
        self._lifecycle = SandboxLifecycle.PREPARED
        return handle

    def run(self, handle: SandboxHandle, request: RunRequest) -> Observation:
        self._require_prepared_handle(handle)
        self._require_new_call(request.call_id)
        task, container_id = self._runtime_state()
        started_at = self._now()
        decision = self._policy.evaluate_command(
            task,
            request.command,
            phase=request.phase,
            network_requested=request.network_requested,
        )
        if decision.decision is PolicyDecision.DENIED:
            return _denied_observation(
                request.call_id,
                handle.sandbox_id,
                started_at=started_at,
            )
        if request.command.cwd is not None:
            path_decision = self._policy.evaluate_path(
                task,
                operation=SandboxPathOperation.READ_REPOSITORY,
                relative_path=request.command.cwd,
            )
            if path_decision.decision is PolicyDecision.DENIED:
                return _denied_observation(
                    request.call_id,
                    handle.sandbox_id,
                    started_at=started_at,
                )
        timeout_seconds = decision.effective_timeout_seconds
        if timeout_seconds is None:
            raise DockerSandboxError(DockerSandboxErrorCode.EXECUTION_FAILED)
        specification = DockerExecSpec(
            argv=(
                "python",
                "-I",
                "-c",
                _EXEC_COMMAND_HELPER,
                request.command.cwd or ".",
                *request.command.argv,
            ),
            workdir=DOCKER_WORKSPACE_ROOT,
            timeout_seconds=timeout_seconds,
            max_artifact_bytes=self._policy.configuration.resources.max_artifact_bytes,
        )
        try:
            result = self._execute_with_network(
                container_id,
                specification,
                network_enabled=decision.network_enabled,
            )
        except DockerClientError:
            self._abort_container()
            return Observation(
                call_id=request.call_id,
                started_at=started_at,
                duration_ms=0,
                sandbox_id=handle.sandbox_id,
                policy_decision=PolicyDecision.ALLOWED,
                error_type=ErrorType.SANDBOX_ERROR,
            )
        stdout_ref, stderr_ref = self._persist_streams(
            request.call_id,
            stdout_name="stdout",
            stdout=result.stdout,
            stderr=result.stderr,
        )
        error_type: ErrorType | None = None
        if result.timed_out:
            error_type = ErrorType.TOOL_TIMEOUT
            self._abort_container()
        elif result.artifact_limit_exceeded:
            error_type = ErrorType.TOOL_EXECUTION_ERROR
            self._abort_container()
        output_limit = self._policy.configuration.resources.max_output_bytes
        return Observation(
            call_id=request.call_id,
            started_at=started_at,
            duration_ms=result.duration_ms,
            exit_code=result.exit_code,
            stdout_ref=stdout_ref,
            stderr_ref=stderr_ref,
            stdout_preview=_decode_preview(result.stdout, output_limit),
            stderr_preview=_decode_preview(result.stderr, output_limit),
            truncated=(
                result.artifact_limit_exceeded
                or len(result.stdout) > output_limit
                or len(result.stderr) > output_limit
            ),
            sandbox_id=handle.sandbox_id,
            policy_decision=PolicyDecision.ALLOWED,
            error_type=error_type,
        )

    def read_file(
        self,
        handle: SandboxHandle,
        request: ReadFileRequest,
    ) -> Observation:
        self._require_prepared_handle(handle)
        self._require_new_call(request.call_id)
        task, container_id = self._runtime_state()
        started_at = self._now()
        decision = self._policy.evaluate_path(
            task,
            operation=SandboxPathOperation.READ_REPOSITORY,
            relative_path=request.relative_path,
        )
        if decision.decision is PolicyDecision.DENIED:
            return _denied_observation(
                request.call_id,
                handle.sandbox_id,
                started_at=started_at,
            )
        try:
            result = self._execute_read(
                container_id,
                request,
            )
        except DockerClientError:
            self._abort_container()
            return Observation(
                call_id=request.call_id,
                started_at=started_at,
                duration_ms=0,
                sandbox_id=handle.sandbox_id,
                policy_decision=PolicyDecision.ALLOWED,
                error_type=ErrorType.SANDBOX_ERROR,
            )
        stdout_ref, stderr_ref = self._persist_streams(
            request.call_id,
            stdout_name="read",
            stdout=result.stdout,
            stderr=result.stderr,
        )
        error_type: ErrorType | None = None
        if result.timed_out:
            error_type = ErrorType.TOOL_TIMEOUT
            self._abort_container()
        elif result.artifact_limit_exceeded:
            error_type = ErrorType.TOOL_EXECUTION_ERROR
            self._abort_container()
        elif result.exit_code != 0:
            error_type = ErrorType.PATH_VIOLATION
        output_limit = self._policy.configuration.resources.max_output_bytes
        return Observation(
            call_id=request.call_id,
            started_at=started_at,
            duration_ms=result.duration_ms,
            exit_code=result.exit_code,
            stdout_ref=stdout_ref,
            stderr_ref=stderr_ref,
            stdout_preview=_decode_preview(result.stdout, output_limit),
            stderr_preview=_decode_preview(result.stderr, output_limit),
            truncated=(
                result.artifact_limit_exceeded
                or len(result.stdout) > output_limit
                or len(result.stderr) > output_limit
            ),
            sandbox_id=handle.sandbox_id,
            policy_decision=PolicyDecision.ALLOWED,
            error_type=error_type,
        )

    def apply_patch(
        self,
        handle: SandboxHandle,
        request: ApplyPatchRequest,
    ) -> Observation:
        self._require_prepared_handle(handle)
        self._require_new_call(request.call_id)
        task, container_id = self._runtime_state()
        started_at = self._now()
        limits = self._policy.configuration.resources
        try:
            content = self._artifact_store.read(request.patch_ref)
        except SandboxArtifactError:
            return self._patch_rejection(
                handle,
                request.call_id,
                started_at=started_at,
                code=PatchErrorCode.ARTIFACT_MISMATCH,
                exit_code=42,
            )
        if not content or len(content) > limits.max_patch_bytes:
            return self._patch_rejection(
                handle,
                request.call_id,
                started_at=started_at,
                code=PatchErrorCode.PATCH_TOO_LARGE,
                exit_code=44,
            )
        paths = _candidate_patch_paths(content)
        if paths is None:
            return self._patch_rejection(
                handle,
                request.call_id,
                started_at=started_at,
                code=PatchErrorCode.INVALID_DIFF,
                exit_code=44,
            )
        for path in paths:
            decision = self._policy.evaluate_path(
                task,
                operation=SandboxPathOperation.WRITE_REPOSITORY,
                relative_path=path,
            )
            if decision.decision is PolicyDecision.DENIED:
                return _denied_observation(
                    request.call_id,
                    handle.sandbox_id,
                    started_at=started_at,
                )

        artifact_name = Path(request.patch_ref.relative_path).name
        try:
            self._client.copy_artifact(
                container_id,
                artifact_name,
                content,
                max_bytes=limits.max_patch_bytes,
            )
            result = self._client.execute(
                container_id,
                DockerExecSpec(
                    argv=(
                        "python",
                        "-I",
                        "-c",
                        APPLY_PATCH_HELPER,
                        artifact_name,
                        str(request.patch_ref.size_bytes),
                        request.patch_ref.sha256,
                        str(limits.max_patch_files),
                        str(limits.max_patch_hunks),
                        str(limits.max_patch_changed_lines),
                    ),
                    workdir=DOCKER_WORKSPACE_ROOT,
                    timeout_seconds=limits.command_timeout_seconds,
                    max_artifact_bytes=limits.max_artifact_bytes,
                ),
            )
        except DockerClientError:
            self._abort_container()
            return Observation(
                call_id=request.call_id,
                started_at=started_at,
                duration_ms=0,
                sandbox_id=handle.sandbox_id,
                policy_decision=PolicyDecision.ALLOWED,
                error_type=ErrorType.SANDBOX_ERROR,
            )
        stdout_ref, stderr_ref = self._persist_streams(
            request.call_id,
            stdout_name="stdout",
            stdout=result.stdout,
            stderr=result.stderr,
        )
        error_type: ErrorType | None = None
        if result.timed_out:
            error_type = ErrorType.TOOL_TIMEOUT
            self._abort_container()
        elif result.artifact_limit_exceeded:
            error_type = ErrorType.TOOL_EXECUTION_ERROR
            self._abort_container()
        elif result.exit_code == 0:
            if result.stderr or not _valid_patch_success_output(
                result.stdout,
                expected_sha256=request.patch_ref.sha256,
                expected_paths=paths,
            ):
                error_type = ErrorType.SANDBOX_ERROR
                self._abort_container()
        elif result.exit_code in {42, 43, 44, 45, 46} and _valid_patch_error_output(result.stderr):
            error_type = ErrorType.PATCH_ERROR
            if result.exit_code == 46:
                self._abort_container()
        else:
            error_type = ErrorType.SANDBOX_ERROR
            self._abort_container()
        output_limit = limits.max_output_bytes
        return Observation(
            call_id=request.call_id,
            started_at=started_at,
            duration_ms=result.duration_ms,
            exit_code=result.exit_code,
            stdout_ref=stdout_ref,
            stderr_ref=stderr_ref,
            stdout_preview=_decode_preview(result.stdout, output_limit),
            stderr_preview=_decode_preview(result.stderr, output_limit),
            truncated=(
                result.artifact_limit_exceeded
                or len(result.stdout) > output_limit
                or len(result.stderr) > output_limit
            ),
            sandbox_id=handle.sandbox_id,
            policy_decision=PolicyDecision.ALLOWED,
            error_type=error_type,
        )

    def stage_patch(
        self,
        handle: SandboxHandle,
        call_id: UUID,
        content: bytes,
    ) -> ArtifactRef:
        """Persist validated input before Docker receives an immutable reference."""

        self._require_prepared_handle(handle)
        maximum = self._policy.configuration.resources.max_patch_bytes
        if not content or len(content) > maximum:
            raise DockerSandboxError(
                DockerSandboxErrorCode.ARTIFACT_IO_FAILED,
                error_type=ErrorType.INVALID_TOOL_ARGUMENTS,
            )
        try:
            return self._artifact_store.persist_patch(call_id, content)
        except SandboxArtifactError as error:
            raise DockerSandboxError(DockerSandboxErrorCode.ARTIFACT_IO_FAILED) from error

    def collect_artifacts(self, handle: SandboxHandle) -> tuple[ArtifactRef, ...]:
        self._require_matching_handle(handle)
        try:
            return self._artifact_store.collect()
        except SandboxArtifactError as error:
            raise DockerSandboxError(DockerSandboxErrorCode.ARTIFACT_IO_FAILED) from error

    def destroy(self, handle: SandboxHandle) -> None:
        self._require_matching_handle(handle)
        if self._lifecycle is SandboxLifecycle.DESTROYED:
            return
        if self._lifecycle is not SandboxLifecycle.PREPARED or self._container_id is None:
            raise DockerSandboxError(DockerSandboxErrorCode.LIFECYCLE_ERROR)
        try:
            self._client.remove_container(self._container_id)
        except DockerClientError as error:
            raise DockerSandboxError(DockerSandboxErrorCode.CLEANUP_FAILED) from error
        self._lifecycle = SandboxLifecycle.DESTROYED

    def _runtime_state(self) -> tuple[TaskSpec, str]:
        if self._task is None or self._container_id is None:
            raise DockerSandboxError(DockerSandboxErrorCode.LIFECYCLE_ERROR)
        return self._task, self._container_id

    def _require_new_call(self, call_id: UUID) -> None:
        if call_id in self._used_call_ids:
            raise DockerSandboxError(
                DockerSandboxErrorCode.DUPLICATE_CALL,
                error_type=ErrorType.INVALID_TOOL_ARGUMENTS,
            )
        self._used_call_ids.add(call_id)

    def _execute_with_network(
        self,
        container_id: str,
        specification: DockerExecSpec,
        *,
        network_enabled: bool,
    ) -> DockerExecResult:
        connected = False
        try:
            if network_enabled:
                self._client.connect_setup_network(container_id)
                connected = True
            self._client.ping()
            return self._client.execute(container_id, specification)
        finally:
            if connected:
                try:
                    self._client.disconnect_setup_network(container_id)
                    inspection = self._client.inspect_container(container_id)
                    if inspection.network_names != ("none",):
                        raise DockerClientError("setup network verification")
                except DockerClientError as error:
                    self._abort_container()
                    raise DockerSandboxError(
                        DockerSandboxErrorCode.NETWORK_TRANSITION_FAILED
                    ) from error

    def _execute_read(
        self,
        container_id: str,
        request: ReadFileRequest,
    ) -> DockerExecResult:
        specification = DockerExecSpec(
            argv=(
                "python",
                "-I",
                "-c",
                _READ_FILE_HELPER,
                request.relative_path,
                str(request.start_line),
                "0" if request.end_line is None else str(request.end_line),
            ),
            workdir=DOCKER_WORKSPACE_ROOT,
            timeout_seconds=self._policy.configuration.resources.command_timeout_seconds,
            max_artifact_bytes=self._policy.configuration.resources.max_artifact_bytes,
        )
        return self._client.execute(container_id, specification)

    def _persist_streams(
        self,
        call_id: UUID,
        *,
        stdout_name: str,
        stdout: bytes,
        stderr: bytes,
    ) -> tuple[ArtifactRef, ArtifactRef]:
        maximum = self._policy.configuration.resources.max_artifact_bytes
        if len(stdout) + len(stderr) > maximum:
            self._abort_container()
            raise DockerSandboxError(DockerSandboxErrorCode.EXECUTION_FAILED)
        try:
            stdout_ref = self._artifact_store.persist(call_id, stdout_name, stdout)
            stderr_ref = self._artifact_store.persist(call_id, "stderr", stderr)
        except SandboxArtifactError as error:
            self._abort_container()
            raise DockerSandboxError(DockerSandboxErrorCode.ARTIFACT_IO_FAILED) from error
        return stdout_ref, stderr_ref

    def _patch_rejection(
        self,
        handle: SandboxHandle,
        call_id: UUID,
        *,
        started_at: datetime,
        code: PatchErrorCode,
        exit_code: int,
    ) -> Observation:
        stderr = f"repopilot_patch_error:{code.value}\n".encode()
        stdout_ref, stderr_ref = self._persist_streams(
            call_id,
            stdout_name="stdout",
            stdout=b"",
            stderr=stderr,
        )
        return Observation(
            call_id=call_id,
            started_at=started_at,
            duration_ms=0,
            exit_code=exit_code,
            stdout_ref=stdout_ref,
            stderr_ref=stderr_ref,
            stdout_preview="",
            stderr_preview=stderr.decode(),
            sandbox_id=handle.sandbox_id,
            policy_decision=PolicyDecision.ALLOWED,
            error_type=ErrorType.PATCH_ERROR,
        )

    def _abort_container(self) -> None:
        if self._container_id is None or self._lifecycle is SandboxLifecycle.DESTROYED:
            return
        try:
            self._client.remove_container(self._container_id)
        except DockerClientError as error:
            raise DockerSandboxError(DockerSandboxErrorCode.CLEANUP_FAILED) from error
        self._lifecycle = SandboxLifecycle.DESTROYED

    def _require_prepared_handle(self, handle: SandboxHandle) -> None:
        self._require_matching_handle(handle)
        if self._lifecycle is not SandboxLifecycle.PREPARED:
            raise DockerSandboxError(DockerSandboxErrorCode.LIFECYCLE_ERROR)

    def _require_matching_handle(self, handle: SandboxHandle) -> None:
        if self._handle is None or handle != self._handle:
            raise DockerSandboxError(DockerSandboxErrorCode.LIFECYCLE_ERROR)

    @staticmethod
    def _call_client(
        operation: Callable[[], _T],
        error_code: DockerSandboxErrorCode,
    ) -> _T:
        try:
            return operation()
        except DockerClientError as error:
            raise DockerSandboxError(error_code) from error


def _candidate_patch_paths(content: bytes) -> tuple[str, ...] | None:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return None
    paths: list[str] = []
    for line in text.splitlines():
        if not line.startswith("diff --git "):
            continue
        match = re.fullmatch(r"diff --git a/(\S+) b/(\S+)", line)
        if match is None or match.group(1) != match.group(2):
            return None
        if match.group(1) in paths:
            return None
        paths.append(match.group(1))
    return tuple(paths) if paths else None


def _valid_patch_success_output(
    output: bytes,
    *,
    expected_sha256: str,
    expected_paths: tuple[str, ...],
) -> bool:
    try:
        payload = cast(dict[str, object], json.loads(output.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return False
    return payload == {
        "files_changed": list(expected_paths),
        "patch_sha256": expected_sha256,
    }


def _valid_patch_error_output(output: bytes) -> bool:
    try:
        text = output.decode("utf-8")
        prefix = "repopilot_patch_error:"
        if not text.startswith(prefix) or text.count("\n") != 1 or not text.endswith("\n"):
            return False
        PatchErrorCode(text.removeprefix(prefix).removesuffix("\n"))
    except (UnicodeDecodeError, ValueError):
        return False
    return True


def _resolve_repository_source(task: TaskSpec) -> Path:
    if task.repository_path is None:
        raise DockerSandboxError(DockerSandboxErrorCode.SOURCE_UNAVAILABLE)
    source = Path(task.repository_path).expanduser()
    try:
        resolved = source.resolve(strict=True)
    except OSError as error:
        raise DockerSandboxError(DockerSandboxErrorCode.SOURCE_UNAVAILABLE) from error
    if not resolved.is_dir():
        raise DockerSandboxError(DockerSandboxErrorCode.SOURCE_UNAVAILABLE)
    return resolved


def _sandbox_labels(
    *,
    sandbox_id: str,
    task: TaskSpec,
    policy_request_sha256: str,
) -> dict[str, str]:
    return {
        f"{DOCKER_LABEL_PREFIX}.managed": "true",
        f"{DOCKER_LABEL_PREFIX}.policy_request_sha256": policy_request_sha256,
        f"{DOCKER_LABEL_PREFIX}.repository_commit": task.base_commit,
        f"{DOCKER_LABEL_PREFIX}.sandbox_id": sandbox_id,
        f"{DOCKER_LABEL_PREFIX}.task_id_sha256": hashlib.sha256(
            task.task_id.encode("utf-8")
        ).hexdigest(),
    }


def _posture_matches(
    inspection: DockerContainerInspection,
    specification: DockerContainerSpec,
    *,
    image: DockerImageIdentity,
    container_id: str,
) -> bool:
    tmpfs_options = inspection.tmpfs.get(DOCKER_WORKSPACE_ROOT, "")
    required_tmpfs_options = {
        "rw",
        "nosuid",
        "nodev",
        f"size={specification.tmpfs_limit_mb}m",
        f"uid={specification.user.partition(':')[0]}",
        f"gid={specification.user.partition(':')[2]}",
        "mode=0750",
    }
    return (
        inspection.container_id == container_id
        and inspection.image_id == image.image_id
        and inspection.running
        and inspection.user == specification.user
        and inspection.network_mode == specification.network_mode.value
        and inspection.network_names == ("none",)
        and inspection.read_only_rootfs
        and not inspection.privileged
        and inspection.cap_drop == specification.cap_drop
        and inspection.security_options == specification.security_options
        and inspection.nano_cpus == round(specification.cpu_limit * _NANO_CPUS_PER_CPU)
        and inspection.memory_bytes == specification.memory_limit_mb * _MEBIBYTE
        and inspection.pids_limit == specification.pids_limit
        and required_tmpfs_options.issubset(set(tmpfs_options.split(",")))
        and all(mount.mount_type == "tmpfs" for mount in inspection.mounts)
        and all(mount.destination == DOCKER_WORKSPACE_ROOT for mount in inspection.mounts)
        and not DOCKER_SENSITIVE_ENVIRONMENT_NAMES.intersection(inspection.environment_names)
        and set(specification.environment).issubset(inspection.environment_names)
        and all(
            inspection.labels.get(name) == value for name, value in specification.labels.items()
        )
        and inspection.workdir == specification.workdir
    )


def _tmpfs_option(specification: DockerContainerSpec) -> str:
    uid, _, gid = specification.user.partition(":")
    return (
        f"{DOCKER_WORKSPACE_ROOT}:rw,nosuid,nodev,size={specification.tmpfs_limit_mb}m,"
        f"uid={uid},gid={gid},mode=0750"
    )


def _format_cpu_limit(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _denied_observation(
    call_id: UUID,
    sandbox_id: str,
    *,
    started_at: datetime,
) -> Observation:
    return Observation(
        call_id=call_id,
        started_at=started_at,
        duration_ms=0,
        sandbox_id=sandbox_id,
        policy_decision=PolicyDecision.DENIED,
        error_type=ErrorType.POLICY_DENIED,
    )


def _decode_preview(content: bytes, limit: int) -> str:
    preview = content[:limit].decode("utf-8", errors="replace")
    encoded = preview.encode("utf-8")
    if len(encoded) <= limit:
        return preview
    return encoded[:limit].decode("utf-8", errors="ignore")


def _write_filtered_tar(
    source: Path,
    output: IO[bytes],
    *,
    max_bytes: int,
) -> WorkspaceCopyResult:
    counters = {
        "files": 0,
        "directories": 0,
        "bytes": 0,
        "symlinks": 0,
        "denied": 0,
    }
    with tarfile.open(fileobj=output, mode="w|") as archive:
        _add_directory_to_tar(
            archive,
            source=source,
            relative=Path(),
            max_bytes=max_bytes,
            counters=counters,
        )
    return WorkspaceCopyResult(
        file_count=counters["files"],
        directory_count=counters["directories"],
        byte_count=counters["bytes"],
        skipped_symlink_count=counters["symlinks"],
        skipped_denied_count=counters["denied"],
    )


def _add_directory_to_tar(
    archive: tarfile.TarFile,
    *,
    source: Path,
    relative: Path,
    max_bytes: int,
    counters: dict[str, int],
) -> None:
    current = source / relative
    for entry in sorted(current.iterdir(), key=lambda item: item.name):
        child_relative = relative / entry.name
        if entry.name in _COPY_EXCLUDED_NAMES:
            counters["denied"] += 1
            continue
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            counters["symlinks"] += 1
            continue
        if stat.S_ISDIR(metadata.st_mode):
            _add_tar_directory(archive, child_relative, metadata.st_mode)
            counters["directories"] += 1
            _add_directory_to_tar(
                archive,
                source=source,
                relative=child_relative,
                max_bytes=max_bytes,
                counters=counters,
            )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise DockerClientError("workspace copy")
        counters["bytes"] += metadata.st_size
        if counters["bytes"] > max_bytes:
            raise DockerClientError("workspace copy")
        _add_tar_file(archive, entry, child_relative, metadata.st_mode, metadata.st_size)
        counters["files"] += 1


def _add_tar_directory(
    archive: tarfile.TarFile,
    relative: Path,
    mode: int,
) -> None:
    info = tarfile.TarInfo(relative.as_posix())
    info.type = tarfile.DIRTYPE
    info.mode = (stat.S_IMODE(mode) & 0o755) | 0o700
    info.uid = 65_532
    info.gid = 65_532
    info.mtime = DOCKER_ARCHIVE_MTIME_SECONDS
    archive.addfile(info)


def _add_tar_file(
    archive: tarfile.TarFile,
    source: Path,
    relative: Path,
    mode: int,
    size: int,
) -> None:
    info = tarfile.TarInfo(relative.as_posix())
    info.mode = (stat.S_IMODE(mode) & 0o755) | 0o600
    info.uid = 65_532
    info.gid = 65_532
    info.mtime = DOCKER_ARCHIVE_MTIME_SECONDS
    info.size = size
    with source.open("rb") as stream:
        archive.addfile(info, stream)
