"""Scripted Sandbox implementation that never touches host execution or files."""

from collections.abc import Iterable
from typing import Self

from pydantic import model_validator

from repopilot.enums import ErrorType, SandboxLifecycle
from repopilot.sandbox.base import (
    ApplyPatchRequest,
    ReadFileRequest,
    RunRequest,
    SandboxError,
    SandboxHandle,
    SandboxRequest,
)
from repopilot.schemas import ArtifactRef, Observation, StrictModel, TaskSpec


class FakeSandboxStep(StrictModel):
    """One exact request/observation pair in a deterministic Sandbox script."""

    request: SandboxRequest
    observation: Observation

    @model_validator(mode="after")
    def validate_call_correlation(self) -> Self:
        if self.request.call_id != self.observation.call_id:
            raise ValueError("scripted request and observation call_id values must match")
        return self


class FakeSandboxConfigurationError(SandboxError):
    """The injected Fake Sandbox definition is internally inconsistent."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_type=ErrorType.SANDBOX_ERROR)


class SandboxLifecycleError(SandboxError):
    """An operation was attempted in an invalid lifecycle state."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_type=ErrorType.SANDBOX_ERROR)


class SandboxScriptMismatchError(SandboxError):
    """The received request does not match the next scripted operation."""

    def __init__(self) -> None:
        super().__init__(
            "sandbox request does not match the next scripted operation",
            error_type=ErrorType.INVALID_TOOL_ARGUMENTS,
        )


class SandboxScriptExhaustedError(SandboxError):
    """No scripted observation remains for an operation."""

    def __init__(self) -> None:
        super().__init__(
            "fake sandbox observation script is exhausted",
            error_type=ErrorType.SANDBOX_ERROR,
        )


class FakeSandbox:
    """Enforce lifecycle and return prevalidated observations without side effects."""

    def __init__(
        self,
        *,
        handle: SandboxHandle,
        steps: Iterable[FakeSandboxStep],
        artifacts: Iterable[ArtifactRef] = (),
    ) -> None:
        self._handle = handle
        self._steps = tuple(steps)
        self._artifacts = tuple(artifacts)
        self._validate_configuration()
        self._lifecycle = SandboxLifecycle.NEW
        self._next_step = 0

    @property
    def lifecycle(self) -> SandboxLifecycle:
        return self._lifecycle

    @property
    def remaining_steps(self) -> int:
        return len(self._steps) - self._next_step

    def prepare(self, task: TaskSpec) -> SandboxHandle:
        if self._lifecycle is SandboxLifecycle.DESTROYED:
            raise SandboxLifecycleError("cannot prepare a destroyed sandbox")
        if self._lifecycle is SandboxLifecycle.PREPARED:
            raise SandboxLifecycleError("sandbox is already prepared")
        if (
            task.task_id != self._handle.task_id
            or task.base_commit != self._handle.repository_commit
        ):
            raise FakeSandboxConfigurationError("sandbox handle does not match the requested task")
        self._lifecycle = SandboxLifecycle.PREPARED
        return self._handle

    def run(self, handle: SandboxHandle, request: RunRequest) -> Observation:
        return self._consume(handle, request)

    def read_file(
        self,
        handle: SandboxHandle,
        request: ReadFileRequest,
    ) -> Observation:
        return self._consume(handle, request)

    def apply_patch(
        self,
        handle: SandboxHandle,
        request: ApplyPatchRequest,
    ) -> Observation:
        return self._consume(handle, request)

    def collect_artifacts(self, handle: SandboxHandle) -> tuple[ArtifactRef, ...]:
        self._require_prepared_handle(handle)
        return self._artifacts

    def destroy(self, handle: SandboxHandle) -> None:
        self._require_matching_handle(handle)
        if self._lifecycle is SandboxLifecycle.DESTROYED:
            return
        if self._lifecycle is SandboxLifecycle.NEW:
            raise SandboxLifecycleError("cannot destroy a sandbox before prepare")
        self._lifecycle = SandboxLifecycle.DESTROYED

    def _consume(
        self,
        handle: SandboxHandle,
        request: RunRequest | ReadFileRequest | ApplyPatchRequest,
    ) -> Observation:
        self._require_prepared_handle(handle)
        if self._next_step >= len(self._steps):
            raise SandboxScriptExhaustedError()
        step = self._steps[self._next_step]
        if step.request != request:
            raise SandboxScriptMismatchError()
        self._next_step += 1
        return step.observation

    def _require_prepared_handle(self, handle: SandboxHandle) -> None:
        self._require_matching_handle(handle)
        if self._lifecycle is SandboxLifecycle.NEW:
            raise SandboxLifecycleError("sandbox must be prepared before use")
        if self._lifecycle is SandboxLifecycle.DESTROYED:
            raise SandboxLifecycleError("sandbox has been destroyed")

    def _require_matching_handle(self, handle: SandboxHandle) -> None:
        if handle != self._handle:
            raise SandboxLifecycleError("sandbox handle does not match this adapter")

    def _validate_configuration(self) -> None:
        call_ids = tuple(step.request.call_id for step in self._steps)
        if len(set(call_ids)) != len(call_ids):
            raise FakeSandboxConfigurationError(
                "fake sandbox scripted call_id values must be unique"
            )
        artifact_paths = tuple(artifact.relative_path for artifact in self._artifacts)
        if len(set(artifact_paths)) != len(artifact_paths):
            raise FakeSandboxConfigurationError("fake sandbox artifact paths must be unique")
