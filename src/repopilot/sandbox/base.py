"""Narrow contracts for isolated repository execution."""

from __future__ import annotations

from typing import Annotated, Literal, Protocol, Self, runtime_checkable
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from repopilot.enums import (
    ErrorType,
    SandboxCommandPhase,
    SandboxLifecycle,
    SandboxOperation,
)
from repopilot.schemas import (
    ArtifactRef,
    CommandSpec,
    GitSha,
    Identifier,
    Observation,
    RelativePath,
    StrictModel,
    TaskId,
    TaskSpec,
)


class SandboxError(RuntimeError):
    """Safe sandbox failure carrying the shared runtime error taxonomy."""

    def __init__(self, message: str, *, error_type: ErrorType) -> None:
        super().__init__(message)
        self.error_type = error_type


class SandboxHandle(StrictModel):
    """Opaque identity returned after a task sandbox is prepared."""

    sandbox_id: Identifier
    task_id: TaskId
    repository_commit: GitSha
    prepared_at: AwareDatetime


class RunRequest(StrictModel):
    """Execute one already-validated, shell-free command."""

    operation: Literal[SandboxOperation.RUN] = SandboxOperation.RUN
    call_id: UUID
    command: CommandSpec
    phase: SandboxCommandPhase = SandboxCommandPhase.TEST
    network_requested: bool = Field(default=False, strict=True)


class ReadFileRequest(StrictModel):
    """Read a bounded line range from a contained repository path."""

    operation: Literal[SandboxOperation.READ_FILE] = SandboxOperation.READ_FILE
    call_id: UUID
    relative_path: RelativePath
    start_line: int = Field(default=1, gt=0)
    end_line: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_line_range(self) -> Self:
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ApplyPatchRequest(StrictModel):
    """Apply a candidate diff already stored in controlled artifacts."""

    operation: Literal[SandboxOperation.APPLY_PATCH] = SandboxOperation.APPLY_PATCH
    call_id: UUID
    patch_ref: ArtifactRef


SandboxRequest = Annotated[
    RunRequest | ReadFileRequest | ApplyPatchRequest,
    Field(discriminator="operation"),
]


@runtime_checkable
class SandboxAdapter(Protocol):
    """Minimal lifecycle and operation interface required by RepoPilot."""

    @property
    def lifecycle(self) -> SandboxLifecycle:
        """Return the adapter's current lifecycle state."""

    def prepare(self, task: TaskSpec) -> SandboxHandle:
        """Prepare isolated state for exactly one task."""

    def run(self, handle: SandboxHandle, request: RunRequest) -> Observation:
        """Execute a validated command inside the sandbox."""

    def read_file(
        self,
        handle: SandboxHandle,
        request: ReadFileRequest,
    ) -> Observation:
        """Read a contained file range and return artifact-backed output."""

    def apply_patch(
        self,
        handle: SandboxHandle,
        request: ApplyPatchRequest,
    ) -> Observation:
        """Apply a controlled patch artifact inside the sandbox."""

    def collect_artifacts(self, handle: SandboxHandle) -> tuple[ArtifactRef, ...]:
        """Return artifacts retained for trace and evaluation."""

    def destroy(self, handle: SandboxHandle) -> None:
        """Release sandbox resources; repeated cleanup must be safe."""
