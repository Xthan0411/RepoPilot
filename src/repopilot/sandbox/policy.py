"""Fail-closed Sandbox policy decisions made before any Docker operation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    ErrorType,
    PolicyDecision,
    SandboxCommandPhase,
    SandboxNetworkMode,
    SandboxPathOperation,
    SandboxPolicyErrorCode,
    SandboxPolicyOperation,
)
from repopilot.schemas import (
    CommandSpec,
    NonEmptyText,
    RelativePath,
    Sha256,
    StrictModel,
    TaskId,
    TaskSpec,
)

SANDBOX_POLICY_VERSION: Literal[1] = 1
SANDBOX_DEFAULT_UID = 65_532
SANDBOX_DEFAULT_GID = 65_532
SANDBOX_ALLOWED_EXECUTABLES = (
    "pip",
    "pip3",
    "pytest",
    "python",
    "python3",
    "uv",
)
SANDBOX_ALLOWED_ENVIRONMENT_NAMES = (
    "CI",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUNBUFFERED",
)
SANDBOX_DENIED_PATH_ROOTS = (
    ".aws",
    ".codex",
    ".config",
    ".docker",
    ".git",
    ".ssh",
)
SANDBOX_POLICY_ERROR_MESSAGES = {
    SandboxPolicyErrorCode.INVALID_REQUEST: "sandbox request is invalid",
    SandboxPolicyErrorCode.IMAGE_DENIED: ("sandbox image must use an immutable sha256 digest"),
    SandboxPolicyErrorCode.TASK_MISMATCH: "sandbox request belongs to another task",
    SandboxPolicyErrorCode.COMMAND_NOT_CONFIGURED: (
        "sandbox command is not configured by the task"
    ),
    SandboxPolicyErrorCode.EXECUTABLE_DENIED: "sandbox executable is not allowlisted",
    SandboxPolicyErrorCode.PATH_DENIED: "sandbox path is outside its allowed capability",
    SandboxPolicyErrorCode.NETWORK_DENIED: "sandbox network request is not allowed",
    SandboxPolicyErrorCode.HOST_MOUNT_DENIED: "sandbox host bind mounts are forbidden",
    SandboxPolicyErrorCode.DOCKER_SOCKET_DENIED: "mounting the Docker socket is forbidden",
    SandboxPolicyErrorCode.PRIVILEGED_DENIED: "privileged containers are forbidden",
    SandboxPolicyErrorCode.ROOT_USER_DENIED: "sandbox processes must not run as root",
    SandboxPolicyErrorCode.CAPABILITY_DENIED: "Linux capabilities must be empty",
    SandboxPolicyErrorCode.ROOTFS_WRITE_DENIED: ("the container root filesystem must be read-only"),
    SandboxPolicyErrorCode.NO_NEW_PRIVILEGES_REQUIRED: ("no-new-privileges must be enabled"),
    SandboxPolicyErrorCode.ENVIRONMENT_DENIED: (
        "sandbox environment forwarding is not allowlisted"
    ),
    SandboxPolicyErrorCode.RESOURCE_LIMIT_DENIED: ("sandbox resource request exceeds policy"),
    SandboxPolicyErrorCode.TIMEOUT_DENIED: "sandbox command timeout exceeds policy",
}


class SandboxResourceLimits(StrictModel):
    """Trusted maximums compiled from validated RepoPilot settings."""

    cpu_limit: float = Field(gt=0.0, le=8.0, allow_inf_nan=False)
    memory_limit_mb: int = Field(ge=128, le=8_192)
    pids_limit: int = Field(ge=16, le=1_024)
    tmpfs_limit_mb: int = Field(ge=16, le=2_048)
    command_timeout_seconds: int = Field(gt=0, le=3_600)
    max_output_bytes: int = Field(gt=0, le=10_000_000)
    max_artifact_bytes: int = Field(gt=0, le=100_000_000)
    max_patch_bytes: int = Field(gt=0, le=1_000_000)
    max_patch_files: int = Field(gt=0, le=100)
    max_patch_hunks: int = Field(gt=0, le=1_000)
    max_patch_changed_lines: int = Field(gt=0, le=10_000)

    @model_validator(mode="after")
    def validate_related_limits(self) -> Self:
        if self.tmpfs_limit_mb > self.memory_limit_mb:
            raise ValueError("tmpfs limit cannot exceed memory limit")
        if self.max_output_bytes > self.max_artifact_bytes:
            raise ValueError("output limit cannot exceed artifact limit")
        if self.max_patch_bytes > self.max_artifact_bytes:
            raise ValueError("patch limit cannot exceed artifact limit")
        return self


class RequestedSandboxResources(StrictModel):
    """Untrusted requested values broad enough for policy to return a typed denial."""

    cpu_limit: float = Field(gt=0.0, le=64.0, allow_inf_nan=False)
    memory_limit_mb: int = Field(gt=0, le=131_072)
    pids_limit: int = Field(gt=0, le=1_000_000)
    tmpfs_limit_mb: int = Field(gt=0, le=131_072)
    max_output_bytes: int = Field(gt=0, le=1_000_000_000)
    max_artifact_bytes: int = Field(gt=0, le=10_000_000_000)

    @field_validator("*", mode="before")
    @classmethod
    def reject_booleans(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("sandbox resources must be numeric, not booleans")
        return value


class SandboxPolicyConfiguration(StrictModel):
    """Immutable policy maximums and non-expandable capability allowlists."""

    policy_version: Literal[1] = SANDBOX_POLICY_VERSION
    network_mode: SandboxNetworkMode = SandboxNetworkMode.NONE
    resources: SandboxResourceLimits
    allowed_executables: tuple[str, ...] = SANDBOX_ALLOWED_EXECUTABLES
    allowed_environment_names: tuple[str, ...] = SANDBOX_ALLOWED_ENVIRONMENT_NAMES

    @model_validator(mode="after")
    def validate_allowlists(self) -> Self:
        if (
            self.allowed_executables != SANDBOX_ALLOWED_EXECUTABLES
            or self.allowed_environment_names != SANDBOX_ALLOWED_ENVIRONMENT_NAMES
        ):
            raise ValueError("sandbox capability allowlists cannot be expanded")
        return self

    @classmethod
    def from_settings(
        cls,
        settings: RepoPilotSettings,
    ) -> SandboxPolicyConfiguration:
        return cls(
            network_mode=settings.sandbox_network,
            resources=SandboxResourceLimits(
                cpu_limit=settings.sandbox_cpu_limit,
                memory_limit_mb=settings.sandbox_memory_limit_mb,
                pids_limit=settings.sandbox_pids_limit,
                tmpfs_limit_mb=settings.sandbox_tmpfs_limit_mb,
                command_timeout_seconds=settings.sandbox_command_timeout_seconds,
                max_output_bytes=settings.sandbox_max_output_bytes,
                max_artifact_bytes=settings.sandbox_max_artifact_bytes,
                max_patch_bytes=settings.patch_max_bytes,
                max_patch_files=settings.patch_max_files,
                max_patch_hunks=settings.patch_max_hunks,
                max_patch_changed_lines=settings.patch_max_changed_lines,
            ),
        )


class SandboxPreparationRequest(StrictModel):
    """Container security posture requested from a future Docker adapter."""

    task_id: TaskId
    image_reference: str = Field(min_length=1, max_length=512, repr=False)
    network_mode: SandboxNetworkMode
    run_as_uid: int = Field(ge=0, le=65_535, strict=True)
    run_as_gid: int = Field(ge=0, le=65_535, strict=True)
    privileged: bool
    read_only_rootfs: bool
    no_new_privileges: bool
    capabilities: tuple[str, ...] = Field(default=(), max_length=64)
    host_bind_mounts: tuple[str, ...] = Field(default=(), max_length=16, repr=False)
    mount_docker_socket: bool = False
    environment_names: tuple[str, ...] = Field(default=(), max_length=64)
    resources: RequestedSandboxResources

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(re.fullmatch(r"CAP_[A-Z0-9_]{1,64}", item) is None for item in value):
            raise ValueError("capabilities must use canonical Linux capability names")
        return tuple(sorted(set(value)))

    @field_validator("host_bind_mounts")
    @classmethod
    def validate_host_mount_strings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() or "\x00" in item or len(item) > 4_096 for item in value):
            raise ValueError("host bind mount paths must be bounded text")
        return tuple(value)

    @field_validator("environment_names")
    @classmethod
    def validate_environment_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(re.fullmatch(r"[A-Z_][A-Z0-9_]{0,127}", item) is None for item in value):
            raise ValueError("environment names must be canonical")
        return tuple(sorted(set(value)))


class SandboxPolicyEvaluation(StrictModel):
    """Safe audit record containing no argv, host path, or environment value."""

    policy_version: Literal[1] = SANDBOX_POLICY_VERSION
    operation: SandboxPolicyOperation
    task_id: TaskId
    decision: PolicyDecision
    error_type: ErrorType | None = None
    error_code: SandboxPolicyErrorCode | None = None
    reason: NonEmptyText
    request_sha256: Sha256
    executable: str | None = Field(default=None, min_length=1, max_length=128)
    relative_path: RelativePath | None = None
    effective_timeout_seconds: int | None = Field(default=None, gt=0)
    network_enabled: bool = False

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.decision is PolicyDecision.ALLOWED:
            if self.error_type is not None or self.error_code is not None:
                raise ValueError("allowed policy decisions cannot contain an error")
        elif self.error_type is not ErrorType.POLICY_DENIED or self.error_code is None:
            raise ValueError("denied policy decisions require a policy error")
        if self.operation is not SandboxPolicyOperation.COMMAND and (
            self.executable is not None or self.effective_timeout_seconds is not None
        ):
            raise ValueError("only command decisions contain command metadata")
        if self.operation is not SandboxPolicyOperation.PATH and self.relative_path is not None:
            raise ValueError("only path decisions contain a relative path")
        if self.network_enabled and self.operation is not SandboxPolicyOperation.COMMAND:
            raise ValueError("network can be enabled only for one command decision")
        return self


class _CanonicalPath(StrictModel):
    relative_path: RelativePath


class SandboxPolicy:
    """Evaluate all security-sensitive inputs before a Docker client is called."""

    def __init__(self, configuration: SandboxPolicyConfiguration) -> None:
        self._configuration = configuration

    @classmethod
    def from_settings(cls, settings: RepoPilotSettings) -> SandboxPolicy:
        return cls(SandboxPolicyConfiguration.from_settings(settings))

    @property
    def configuration(self) -> SandboxPolicyConfiguration:
        return self._configuration

    def default_preparation_request(
        self,
        task_id: str,
        *,
        image_reference: str,
    ) -> SandboxPreparationRequest:
        """Build the only default posture P4-S2 should translate to Docker."""

        limits = self._configuration.resources
        return SandboxPreparationRequest(
            task_id=task_id,
            image_reference=image_reference,
            network_mode=self._configuration.network_mode,
            run_as_uid=SANDBOX_DEFAULT_UID,
            run_as_gid=SANDBOX_DEFAULT_GID,
            privileged=False,
            read_only_rootfs=True,
            no_new_privileges=True,
            capabilities=(),
            host_bind_mounts=(),
            mount_docker_socket=False,
            environment_names=SANDBOX_ALLOWED_ENVIRONMENT_NAMES,
            resources=RequestedSandboxResources(
                cpu_limit=limits.cpu_limit,
                memory_limit_mb=limits.memory_limit_mb,
                pids_limit=limits.pids_limit,
                tmpfs_limit_mb=limits.tmpfs_limit_mb,
                max_output_bytes=limits.max_output_bytes,
                max_artifact_bytes=limits.max_artifact_bytes,
            ),
        )

    def evaluate_preparation(
        self,
        request: SandboxPreparationRequest,
        *,
        expected_task_id: str,
    ) -> SandboxPolicyEvaluation:
        """Require the complete non-root, capability-free, minimally mounted posture."""

        fingerprint = _request_sha256(request.model_dump(mode="json"))
        checks = (
            (
                request.task_id != expected_task_id,
                SandboxPolicyErrorCode.TASK_MISMATCH,
            ),
            (
                not _is_digest_pinned_image(request.image_reference),
                SandboxPolicyErrorCode.IMAGE_DENIED,
            ),
            (
                request.network_mode is not self._configuration.network_mode,
                SandboxPolicyErrorCode.NETWORK_DENIED,
            ),
            (
                request.mount_docker_socket,
                SandboxPolicyErrorCode.DOCKER_SOCKET_DENIED,
            ),
            (
                bool(request.host_bind_mounts),
                SandboxPolicyErrorCode.HOST_MOUNT_DENIED,
            ),
            (
                request.privileged,
                SandboxPolicyErrorCode.PRIVILEGED_DENIED,
            ),
            (
                request.run_as_uid == 0 or request.run_as_gid == 0,
                SandboxPolicyErrorCode.ROOT_USER_DENIED,
            ),
            (
                bool(request.capabilities),
                SandboxPolicyErrorCode.CAPABILITY_DENIED,
            ),
            (
                not request.read_only_rootfs,
                SandboxPolicyErrorCode.ROOTFS_WRITE_DENIED,
            ),
            (
                not request.no_new_privileges,
                SandboxPolicyErrorCode.NO_NEW_PRIVILEGES_REQUIRED,
            ),
            (
                not set(request.environment_names).issubset(
                    self._configuration.allowed_environment_names
                ),
                SandboxPolicyErrorCode.ENVIRONMENT_DENIED,
            ),
            (
                not _resources_within(
                    request.resources,
                    self._configuration.resources,
                ),
                SandboxPolicyErrorCode.RESOURCE_LIMIT_DENIED,
            ),
        )
        for denied, code in checks:
            if denied:
                return _denied(
                    operation=SandboxPolicyOperation.PREPARE,
                    task_id=request.task_id,
                    code=code,
                    request_sha256=fingerprint,
                )
        return _allowed(
            operation=SandboxPolicyOperation.PREPARE,
            task_id=request.task_id,
            request_sha256=fingerprint,
        )

    def evaluate_command(
        self,
        task: TaskSpec,
        command: CommandSpec,
        *,
        phase: SandboxCommandPhase,
        network_requested: bool = False,
    ) -> SandboxPolicyEvaluation:
        """Allow only an exact task-owned, shell-free command under policy limits."""

        fingerprint = _request_sha256(
            {
                "task_id": task.task_id,
                "phase": phase,
                "network_requested": network_requested,
                "command": command.model_dump(mode="json"),
            }
        )
        if not isinstance(phase, SandboxCommandPhase) or not isinstance(network_requested, bool):
            return _denied(
                operation=SandboxPolicyOperation.COMMAND,
                task_id=task.task_id,
                code=SandboxPolicyErrorCode.INVALID_REQUEST,
                request_sha256=fingerprint,
                executable=command.argv[0],
            )
        configured = (
            task.setup_commands if phase is SandboxCommandPhase.SETUP else task.test_commands
        )
        executable = command.argv[0]
        if command not in configured:
            return _denied(
                operation=SandboxPolicyOperation.COMMAND,
                task_id=task.task_id,
                code=SandboxPolicyErrorCode.COMMAND_NOT_CONFIGURED,
                request_sha256=fingerprint,
                executable=executable,
            )
        if executable not in self._configuration.allowed_executables:
            return _denied(
                operation=SandboxPolicyOperation.COMMAND,
                task_id=task.task_id,
                code=SandboxPolicyErrorCode.EXECUTABLE_DENIED,
                request_sha256=fingerprint,
                executable=executable,
            )
        network_allowed = (
            self._configuration.network_mode is SandboxNetworkMode.SETUP_ONLY
            and phase is SandboxCommandPhase.SETUP
        )
        if network_requested and not network_allowed:
            return _denied(
                operation=SandboxPolicyOperation.COMMAND,
                task_id=task.task_id,
                code=SandboxPolicyErrorCode.NETWORK_DENIED,
                request_sha256=fingerprint,
                executable=executable,
            )
        maximum_timeout = min(
            task.timeout_seconds,
            self._configuration.resources.command_timeout_seconds,
        )
        if command.timeout_seconds is not None and command.timeout_seconds > maximum_timeout:
            return _denied(
                operation=SandboxPolicyOperation.COMMAND,
                task_id=task.task_id,
                code=SandboxPolicyErrorCode.TIMEOUT_DENIED,
                request_sha256=fingerprint,
                executable=executable,
            )
        effective_timeout = command.timeout_seconds or maximum_timeout
        return _allowed(
            operation=SandboxPolicyOperation.COMMAND,
            task_id=task.task_id,
            request_sha256=fingerprint,
            executable=executable,
            effective_timeout_seconds=effective_timeout,
            network_enabled=network_requested and network_allowed,
        )

    def evaluate_path(
        self,
        task: TaskSpec,
        *,
        operation: SandboxPathOperation,
        relative_path: str,
    ) -> SandboxPolicyEvaluation:
        """Validate a raw path and keep repository/artifact capabilities separate."""

        fingerprint = _request_sha256(
            {
                "task_id": task.task_id,
                "operation": operation,
                "relative_path": relative_path,
            }
        )
        try:
            canonical = _CanonicalPath(relative_path=relative_path).relative_path
        except (TypeError, ValueError):
            return _denied(
                operation=SandboxPolicyOperation.PATH,
                task_id=task.task_id,
                code=SandboxPolicyErrorCode.PATH_DENIED,
                request_sha256=fingerprint,
            )
        parts = canonical.split("/")
        denied_repository_root = parts[0] in SANDBOX_DENIED_PATH_ROOTS or parts[0] == ".env"
        artifact_root = parts[0] == "artifacts"
        artifact_path = len(parts) > 1 and parts[0] == "artifacts"
        if (
            denied_repository_root
            or (operation is SandboxPathOperation.WRITE_ARTIFACT and not artifact_path)
            or (operation is not SandboxPathOperation.WRITE_ARTIFACT and artifact_root)
        ):
            return _denied(
                operation=SandboxPolicyOperation.PATH,
                task_id=task.task_id,
                code=SandboxPolicyErrorCode.PATH_DENIED,
                request_sha256=fingerprint,
                relative_path=canonical,
            )
        return _allowed(
            operation=SandboxPolicyOperation.PATH,
            task_id=task.task_id,
            request_sha256=fingerprint,
            relative_path=canonical,
        )


def _resources_within(
    requested: RequestedSandboxResources,
    limits: SandboxResourceLimits,
) -> bool:
    return (
        requested.cpu_limit <= limits.cpu_limit
        and requested.memory_limit_mb <= limits.memory_limit_mb
        and requested.pids_limit <= limits.pids_limit
        and requested.tmpfs_limit_mb <= limits.tmpfs_limit_mb
        and requested.max_output_bytes <= limits.max_output_bytes
        and requested.max_artifact_bytes <= limits.max_artifact_bytes
        and requested.tmpfs_limit_mb <= requested.memory_limit_mb
        and requested.max_output_bytes <= requested.max_artifact_bytes
    )


def _is_digest_pinned_image(image_reference: str) -> bool:
    if (
        re.fullmatch(
            r"[a-z0-9][a-z0-9._/-]*(?::[A-Za-z0-9._-]+)?@sha256:[0-9a-f]{64}",
            image_reference,
        )
        is None
    ):
        return False
    name = image_reference.split("@", maxsplit=1)[0]
    return "//" not in name and all(segment not in {"", ".", ".."} for segment in name.split("/"))


def _allowed(
    *,
    operation: SandboxPolicyOperation,
    task_id: str,
    request_sha256: str,
    executable: str | None = None,
    relative_path: str | None = None,
    effective_timeout_seconds: int | None = None,
    network_enabled: bool = False,
) -> SandboxPolicyEvaluation:
    return SandboxPolicyEvaluation(
        operation=operation,
        task_id=task_id,
        decision=PolicyDecision.ALLOWED,
        reason="sandbox request satisfies the frozen policy",
        request_sha256=request_sha256,
        executable=executable,
        relative_path=relative_path,
        effective_timeout_seconds=effective_timeout_seconds,
        network_enabled=network_enabled,
    )


def _denied(
    *,
    operation: SandboxPolicyOperation,
    task_id: str,
    code: SandboxPolicyErrorCode,
    request_sha256: str,
    executable: str | None = None,
    relative_path: str | None = None,
) -> SandboxPolicyEvaluation:
    return SandboxPolicyEvaluation(
        operation=operation,
        task_id=task_id,
        decision=PolicyDecision.DENIED,
        error_type=ErrorType.POLICY_DENIED,
        error_code=code,
        reason=SANDBOX_POLICY_ERROR_MESSAGES[code],
        request_sha256=request_sha256,
        executable=executable,
        relative_path=relative_path,
    )


def _request_sha256(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
