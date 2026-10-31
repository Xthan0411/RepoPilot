"""Bounded unified-diff validation and contained patch tool sessions."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol, cast
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    ErrorType,
    PatchErrorCode,
    PatchStatus,
    PolicyDecision,
)
from repopilot.sandbox.artifacts import SandboxArtifactStore
from repopilot.sandbox.base import ApplyPatchRequest, SandboxHandle
from repopilot.sandbox.policy import SANDBOX_DENIED_PATH_ROOTS
from repopilot.schemas import (
    ArtifactRef,
    Observation,
    PatchState,
    RelativePath,
    StrictModel,
    TaskSpec,
    ToolCall,
)
from repopilot.tools.builtin import ReadFileArguments
from repopilot.tools.patch_workspace import (
    CurrentPatchWorkspace,
    CurrentSourceError,
    ExactDiffConflict,
    SourceSandbox,
    cumulative_diff,
)
from repopilot.tools.repository import FileReadResult, ReadOnlyRepository

_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)\n$")
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?P<section> [^\r\n]*)?\n$")
_UNSUPPORTED_PREFIXES = (
    "Binary files ",
    "GIT binary patch",
    "copy from ",
    "copy to ",
    "deleted file mode ",
    "new file mode ",
    "old mode ",
    "new mode ",
    "rename from ",
    "rename to ",
    "similarity index ",
)
_PATCH_ERROR_MARKER = "repopilot_patch_error:"
_ERROR_MESSAGES = {
    PatchErrorCode.INVALID_DIFF: "candidate patch is not a valid bounded unified diff",
    PatchErrorCode.PATCH_TOO_LARGE: "candidate patch exceeds the configured byte limit",
    PatchErrorCode.TOO_MANY_FILES: "candidate patch exceeds the configured file limit",
    PatchErrorCode.TOO_MANY_HUNKS: "candidate patch exceeds the configured hunk limit",
    PatchErrorCode.TOO_MANY_CHANGED_LINES: (
        "candidate patch exceeds the configured changed-line limit"
    ),
    PatchErrorCode.PATH_DENIED: "candidate patch targets a denied path",
    PatchErrorCode.PROTECTED_TEST_PATH: (
        "candidate patch targets read-only test inputs; no candidate source changes were "
        "applied; submit a source-only patch"
    ),
    PatchErrorCode.UNSUPPORTED_PATCH: "candidate patch uses unsupported diff semantics",
    PatchErrorCode.ARTIFACT_MISMATCH: "candidate patch artifact identity does not match",
    PatchErrorCode.APPLY_CONFLICT: "candidate patch does not apply to the current workspace",
    PatchErrorCode.ROLLBACK_FAILED: "candidate patch rollback could not be proven",
    PatchErrorCode.INSPECT_UNAVAILABLE: "no applied candidate diff is available",
    PatchErrorCode.SANDBOX_ERROR: "candidate patch execution failed in the sandbox",
}


class PatchLimits(StrictModel):
    """Trusted complexity limits applied before a patch reaches Docker."""

    max_bytes: int = Field(gt=0, le=1_000_000)
    max_files: int = Field(gt=0, le=100)
    max_hunks: int = Field(gt=0, le=1_000)
    max_changed_lines: int = Field(gt=0, le=10_000)

    @classmethod
    def from_settings(cls, settings: RepoPilotSettings) -> PatchLimits:
        return cls(
            max_bytes=settings.patch_max_bytes,
            max_files=settings.patch_max_files,
            max_hunks=settings.patch_max_hunks,
            max_changed_lines=settings.patch_max_changed_lines,
        )


class UnifiedDiffSummary(StrictModel):
    """Safe structural summary of a validated candidate diff."""

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    files_changed: tuple[RelativePath, ...] = Field(min_length=1)
    hunk_count: int = Field(gt=0)
    added_lines: int = Field(ge=0)
    removed_lines: int = Field(ge=0)
    original_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    original_size_bytes: int | None = Field(default=None, gt=0)
    normalized_hunk_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> UnifiedDiffSummary:
        if len(set(self.files_changed)) != len(self.files_changed):
            raise ValueError("changed paths must be unique")
        if self.added_lines + self.removed_lines == 0:
            raise ValueError("candidate diff must change at least one line")
        if (self.original_sha256 is None) != (self.original_size_bytes is None):
            raise ValueError("original patch identity requires both hash and size")
        if self.normalized_hunk_count > self.hunk_count:
            raise ValueError("normalized hunk count exceeds total hunk count")
        if self.normalized_hunk_count and self.original_sha256 is None:
            raise ValueError("normalized patches require original patch identity")
        return self


@dataclass(frozen=True, slots=True)
class ParsedUnifiedDiff:
    """Internal validated diff body plus its safe summary."""

    content: str
    encoded: bytes
    summary: UnifiedDiffSummary


class PatchValidationError(ValueError):
    """Safe semantic rejection carrying no diff body or repository content."""

    def __init__(self, code: PatchErrorCode) -> None:
        super().__init__(_ERROR_MESSAGES[code])
        self.code = code


class ApplyPatchArguments(StrictModel):
    """One model-proposed unified diff; paths and content are validated again."""

    diff: str = Field(
        min_length=1,
        max_length=1_000_000,
        repr=False,
        description=(
            "Complete canonical git-style unified diff ending with a newline. Start each "
            "file with 'diff --git a/path b/path', then '--- a/path' and '+++ b/path', "
            "followed by a numbered '@@ -old,count +new,count @@' hunk. Do not use "
            "'*** Begin Patch', '*** Update File', or '*** End Patch' wrappers."
        ),
    )


class InspectDiffArguments(StrictModel):
    """The active Sandbox task owns the only candidate diff."""


class PatchApplicationResult(StrictModel):
    """One patch attempt and the unchanged-or-updated current PatchState."""

    status: PatchStatus
    patch_state: PatchState
    summary: UnifiedDiffSummary | None = None
    observation: Observation | None = None
    error_code: PatchErrorCode | None = None
    message: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_attempt(self) -> PatchApplicationResult:
        if self.status is PatchStatus.APPLIED:
            if (
                self.patch_state.status is not PatchStatus.APPLIED
                or self.summary is None
                or self.observation is None
                or self.observation.exit_code != 0
                or self.observation.error_type is not None
                or self.error_code is not None
            ):
                raise ValueError("applied patch result requires successful Sandbox evidence")
        elif self.status is PatchStatus.REJECTED:
            if self.error_code is None:
                raise ValueError("rejected patch result requires a typed error")
        else:
            raise ValueError("patch attempt status must be applied or rejected")
        return self


class DiffInspectionResult(StrictModel):
    """Bounded view of the most recently applied candidate diff."""

    available: bool
    patch_state: PatchState
    preview: str | None = None
    truncated: bool = False
    error_code: PatchErrorCode | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> DiffInspectionResult:
        if self.available:
            if (
                self.patch_state.status is not PatchStatus.APPLIED
                or self.preview is None
                or self.error_code is not None
            ):
                raise ValueError("available diff inspection requires applied patch evidence")
        elif (
            self.patch_state.status is PatchStatus.APPLIED
            or self.preview is not None
            or self.truncated
            or self.error_code is not PatchErrorCode.INSPECT_UNAVAILABLE
        ):
            raise ValueError("unavailable diff inspection contains contradictory evidence")
        return self


class PatchCapableSandbox(Protocol):
    """Additional trusted ingress required above the minimal Sandbox protocol."""

    def stage_patch(
        self,
        handle: SandboxHandle,
        call_id: UUID,
        content: bytes,
    ) -> ArtifactRef:
        """Persist a validated candidate before contained application."""

    def apply_patch(
        self,
        handle: SandboxHandle,
        request: ApplyPatchRequest,
    ) -> Observation:
        """Apply one previously staged patch artifact."""


class PatchToolSession:
    """Share one current patch revision between apply and inspect capabilities."""

    def __init__(
        self,
        *,
        sandbox: PatchCapableSandbox,
        handle: SandboxHandle,
        task: TaskSpec,
        limits: PatchLimits,
        preview_bytes: int,
        repository: ReadOnlyRepository | None = None,
        artifact_store: SandboxArtifactStore | None = None,
    ) -> None:
        if handle.task_id != task.task_id or handle.repository_commit != task.base_commit:
            raise ValueError("sandbox handle does not match task")
        if preview_bytes <= 0:
            raise ValueError("patch preview limit must be positive")
        self._sandbox = sandbox
        self._handle = handle
        self._task = task
        self._limits = limits
        self._preview_bytes = preview_bytes
        self._state = PatchState()
        self._latest: ParsedUnifiedDiff | None = None
        if (repository is None) != (artifact_store is None):
            raise ValueError("current-source tracking requires repository and artifact store")
        self._workspace = (
            CurrentPatchWorkspace(repository, artifact_store, cast(SourceSandbox, sandbox), handle)
            if repository is not None and artifact_store is not None
            else None
        )

    @classmethod
    def from_settings(
        cls,
        *,
        sandbox: PatchCapableSandbox,
        handle: SandboxHandle,
        task: TaskSpec,
        settings: RepoPilotSettings,
        repository: ReadOnlyRepository | None = None,
        artifact_store: SandboxArtifactStore | None = None,
    ) -> PatchToolSession:
        return cls(
            sandbox=sandbox,
            handle=handle,
            task=task,
            limits=PatchLimits.from_settings(settings),
            preview_bytes=max(1, settings.tool_max_output_bytes // 4),
            repository=repository,
            artifact_store=artifact_store,
        )

    @property
    def state(self) -> PatchState:
        return self._state

    def apply(self, diff: str, *, call: ToolCall) -> PatchApplicationResult:
        if self._workspace is not None and self._workspace.poisoned:
            return self._rejected(PatchErrorCode.SANDBOX_ERROR)
        try:
            parsed = parse_unified_diff(diff, self._limits)
        except PatchValidationError as error:
            return self._rejected(error.code)
        if self._workspace is None and self._state.revision:
            # A second incremental patch must never be mislabeled as the complete diff.
            return self._rejected(PatchErrorCode.SANDBOX_ERROR)

        expected: dict[str, str] | None = None
        cumulative: ParsedUnifiedDiff | None = None
        cumulative_ref: ArtifactRef | None = None
        if self._workspace is not None:
            try:
                expected = self._workspace.candidate(
                    parsed.content, parsed.summary.files_changed, call.call_id
                )
                cumulative = parse_unified_diff(
                    cumulative_diff(self._workspace.base, expected), self._limits
                )
            except PatchValidationError as error:
                return self._rejected(error.code)
            except ExactDiffConflict:
                return self._rejected(PatchErrorCode.APPLY_CONFLICT, summary=parsed.summary)
            except CurrentSourceError:
                self._workspace.poisoned = True
                return self._rejected(PatchErrorCode.SANDBOX_ERROR)
            if self._state.revision:
                # Persist the complete candidate before mutation, in a separate public
                # audit store. Only the incremental diff below enters either sandbox.
                cumulative_ref = self._workspace.store.persist_patch(
                    call.call_id, cumulative.encoded
                )

        patch_ref = self._sandbox.stage_patch(
            self._handle,
            call.call_id,
            parsed.encoded,
        )
        if (
            patch_ref.sha256 != parsed.summary.sha256
            or patch_ref.size_bytes != parsed.summary.size_bytes
        ):
            return self._rejected(
                PatchErrorCode.ARTIFACT_MISMATCH,
                summary=parsed.summary,
            )
        observation = self._sandbox.apply_patch(
            self._handle,
            ApplyPatchRequest(call_id=call.call_id, patch_ref=patch_ref),
        )
        error_code = _observation_error(observation)
        if error_code is not None:
            return self._rejected(
                error_code,
                summary=parsed.summary,
                observation=observation,
            )

        if self._workspace is not None and expected is not None:
            try:
                self._workspace.verify(expected, call.call_id, "after")
            except CurrentSourceError:
                self._workspace.poisoned = True
                return self._rejected(PatchErrorCode.SANDBOX_ERROR, observation=observation)
            self._workspace.current = expected

        exported = cumulative if cumulative_ref is not None and cumulative is not None else parsed

        self._state = PatchState(
            status=PatchStatus.APPLIED,
            diff_ref=cumulative_ref or patch_ref,
            files_changed=exported.summary.files_changed,
            revision=self._state.revision + 1,
        )
        self._latest = exported
        return PatchApplicationResult(
            status=PatchStatus.APPLIED,
            patch_state=self._state,
            summary=exported.summary,
            observation=observation,
            message="candidate patch applied atomically in the sandbox",
        )

    def inspect(self) -> DiffInspectionResult:
        if self._workspace is not None and self._workspace.poisoned:
            raise CurrentSourceError
        if self._latest is None:
            return DiffInspectionResult(
                available=False,
                patch_state=self._state,
                error_code=PatchErrorCode.INSPECT_UNAVAILABLE,
            )
        preview, truncated = _bounded_utf8_preview(
            self._latest.encoded,
            self._preview_bytes,
        )
        return DiffInspectionResult(
            available=True,
            patch_state=self._state,
            preview=preview,
            truncated=truncated,
        )

    def read_file(
        self, path: str, *, start_line: int = 1, end_line: int | None = None, call_id: UUID
    ) -> FileReadResult:
        if self._workspace is None:
            raise CurrentSourceError
        return self._workspace.read_file(
            path, start_line=start_line, end_line=end_line, call_id=call_id
        )

    def _rejected(
        self,
        code: PatchErrorCode,
        *,
        summary: UnifiedDiffSummary | None = None,
        observation: Observation | None = None,
    ) -> PatchApplicationResult:
        return PatchApplicationResult(
            status=PatchStatus.REJECTED,
            patch_state=self._state,
            summary=summary,
            observation=observation,
            error_code=code,
            message=_ERROR_MESSAGES[code],
        )


class ApplyPatchTool:
    """Validate, stage, and atomically apply one bounded unified diff."""

    name = "apply_patch"
    description = (
        "Apply a bounded canonical git-style unified diff to existing regular files in "
        "the Sandbox; custom patch wrappers are invalid."
    )
    arguments_model: type[BaseModel] = ApplyPatchArguments

    def __init__(self, session: PatchToolSession) -> None:
        self._session = session

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        if not isinstance(arguments, ApplyPatchArguments):
            raise TypeError("unexpected argument model")
        return self._session.apply(arguments.diff, call=call)


class CurrentSourceReadTool:
    """Read verified modified source, or the unchanged clean repository source."""

    name = "read_file"
    description = "Read a bounded UTF-8 source line window, including current applied edits."
    arguments_model: type[BaseModel] = ReadFileArguments

    def __init__(self, session: PatchToolSession) -> None:
        self._session = session

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        if not isinstance(arguments, ReadFileArguments):
            raise TypeError("unexpected argument model")
        return self._session.read_file(
            arguments.path,
            start_line=arguments.start_line,
            end_line=arguments.end_line,
            call_id=call.call_id,
        )


class InspectDiffTool:
    """Return only a bounded preview plus the full diff ArtifactRef in PatchState."""

    name = "inspect_diff"
    description = "Inspect the current applied diff using a bounded preview and artifact metadata."
    arguments_model: type[BaseModel] = InspectDiffArguments

    def __init__(self, session: PatchToolSession) -> None:
        self._session = session

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        if not isinstance(arguments, InspectDiffArguments):
            raise TypeError("unexpected argument model")
        return self._session.inspect()


def parse_unified_diff(diff: str, limits: PatchLimits) -> ParsedUnifiedDiff:
    """Validate a diff, correcting only hunk counts from the unchanged body.

    Both input and normalized bytes are bounded. The canonical artifact identity is
    ``sha256``; ``original_sha256`` binds the exact model proposal in the tool trace.
    Line numbers, file paths, context and changed lines are never inferred here.
    """

    try:
        encoded = diff.encode("utf-8")
    except UnicodeEncodeError:
        raise PatchValidationError(PatchErrorCode.INVALID_DIFF) from None
    if len(encoded) > limits.max_bytes:
        raise PatchValidationError(PatchErrorCode.PATCH_TOO_LARGE)
    if "\x00" in diff or "\r" in diff or not diff.endswith("\n"):
        raise PatchValidationError(PatchErrorCode.INVALID_DIFF)

    lines = diff.splitlines(keepends=True)
    if not lines or any(
        line.startswith(prefix) for line in lines for prefix in _UNSUPPORTED_PREFIXES
    ):
        raise PatchValidationError(PatchErrorCode.UNSUPPORTED_PATCH)

    canonical_lines = lines.copy()
    normalized_hunk_count = 0
    index = 0
    files: list[str] = []
    hunk_count = 0
    added_lines = 0
    removed_lines = 0
    while index < len(lines):
        header = _DIFF_HEADER.fullmatch(lines[index])
        if header is None:
            raise PatchValidationError(PatchErrorCode.INVALID_DIFF)
        old_path, new_path = header.groups()
        if old_path != new_path:
            raise PatchValidationError(PatchErrorCode.UNSUPPORTED_PATCH)
        path = _validated_patch_path(old_path)
        if path in files:
            raise PatchValidationError(PatchErrorCode.INVALID_DIFF)
        files.append(path)
        if len(files) > limits.max_files:
            raise PatchValidationError(PatchErrorCode.TOO_MANY_FILES)
        index += 1

        if index < len(lines) and lines[index].startswith("index "):
            index += 1
        if (
            index + 1 >= len(lines)
            or lines[index] != f"--- a/{path}\n"
            or lines[index + 1] != f"+++ b/{path}\n"
        ):
            raise PatchValidationError(PatchErrorCode.UNSUPPORTED_PATCH)
        index += 2

        file_hunks = 0
        previous_old_end = 0
        previous_new_end = 0
        while index < len(lines) and not lines[index].startswith("diff --git "):
            hunk = _HUNK_HEADER.fullmatch(lines[index])
            if hunk is None:
                raise PatchValidationError(PatchErrorCode.INVALID_DIFF)
            old_start = int(hunk.group(1))
            old_count = int(hunk.group(2) or "1")
            new_start = int(hunk.group(3))
            new_count = int(hunk.group(4) or "1")
            header_index = index
            file_hunks += 1
            hunk_count += 1
            if hunk_count > limits.max_hunks:
                raise PatchValidationError(PatchErrorCode.TOO_MANY_HUNKS)
            index += 1

            observed_old = 0
            observed_new = 0
            hunk_changed = 0
            while (
                index < len(lines)
                and not lines[index].startswith("@@ ")
                and not lines[index].startswith("diff --git ")
            ):
                line = lines[index]
                if line.startswith("\\"):
                    raise PatchValidationError(PatchErrorCode.UNSUPPORTED_PATCH)
                if not line or line[0] not in {" ", "+", "-"}:
                    raise PatchValidationError(PatchErrorCode.INVALID_DIFF)
                if line[0] in {" ", "-"}:
                    observed_old += 1
                if line[0] in {" ", "+"}:
                    observed_new += 1
                if line[0] == "+":
                    added_lines += 1
                    hunk_changed += 1
                elif line[0] == "-":
                    removed_lines += 1
                    hunk_changed += 1
                index += 1
            if (
                hunk_changed == 0
                or (observed_old > 0 and old_start == 0)
                or (observed_new > 0 and new_start == 0)
                or old_start < previous_old_end
                or new_start < previous_new_end
            ):
                raise PatchValidationError(PatchErrorCode.INVALID_DIFF)
            previous_old_end = old_start + observed_old
            previous_new_end = new_start + observed_new
            if observed_old != old_count or observed_new != new_count:
                canonical_lines[header_index] = (
                    f"@@ -{hunk.group(1)},{observed_old} "
                    f"+{hunk.group(3)},{observed_new} @@{hunk.group('section') or ''}\n"
                )
                normalized_hunk_count += 1
            if added_lines + removed_lines > limits.max_changed_lines:
                raise PatchValidationError(PatchErrorCode.TOO_MANY_CHANGED_LINES)
        if file_hunks == 0:
            raise PatchValidationError(PatchErrorCode.INVALID_DIFF)

    canonical_content = "".join(canonical_lines)
    canonical_encoded = canonical_content.encode("utf-8")
    if len(canonical_encoded) > limits.max_bytes:
        raise PatchValidationError(PatchErrorCode.PATCH_TOO_LARGE)
    summary = UnifiedDiffSummary(
        sha256=hashlib.sha256(canonical_encoded).hexdigest(),
        size_bytes=len(canonical_encoded),
        files_changed=tuple(files),
        hunk_count=hunk_count,
        added_lines=added_lines,
        removed_lines=removed_lines,
        original_sha256=hashlib.sha256(encoded).hexdigest(),
        original_size_bytes=len(encoded),
        normalized_hunk_count=normalized_hunk_count,
    )
    return ParsedUnifiedDiff(
        content=canonical_content,
        encoded=canonical_encoded,
        summary=summary,
    )


def _validated_patch_path(value: str) -> str:
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise PatchValidationError(PatchErrorCode.UNSUPPORTED_PATCH)
    try:
        path = _PatchPath(relative_path=value).relative_path
    except (TypeError, ValueError):
        raise PatchValidationError(PatchErrorCode.PATH_DENIED) from None
    root = path.split("/", maxsplit=1)[0]
    if root in {*SANDBOX_DENIED_PATH_ROOTS, ".env", "artifacts"}:
        raise PatchValidationError(PatchErrorCode.PATH_DENIED)
    return path


class _PatchPath(StrictModel):
    relative_path: RelativePath


def _observation_error(observation: Observation) -> PatchErrorCode | None:
    if observation.policy_decision is PolicyDecision.DENIED:
        return PatchErrorCode.PATH_DENIED
    if observation.exit_code == 0 and observation.error_type is None and not observation.truncated:
        return None
    if observation.error_type is ErrorType.PATCH_ERROR:
        output = (observation.stderr_preview or "") + "\n" + (observation.stdout_preview or "")
        for line in output.splitlines():
            if line.startswith(_PATCH_ERROR_MARKER):
                raw_code = line.removeprefix(_PATCH_ERROR_MARKER)
                try:
                    return PatchErrorCode(raw_code)
                except ValueError:
                    break
    return PatchErrorCode.SANDBOX_ERROR


def _bounded_utf8_preview(content: bytes, limit: int) -> tuple[str, bool]:
    if len(content) <= limit:
        return content.decode("utf-8"), False
    preview = content[:limit].decode("utf-8", errors="ignore")
    while len(preview.encode("utf-8")) > limit:
        preview = preview[:-1]
    return preview, True
