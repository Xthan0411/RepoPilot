"""Versioned Working Memory snapshots and a safe local file store."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, Self, cast
from uuid import UUID, uuid4

from pydantic import AwareDatetime, Field, JsonValue, ValidationError, model_validator
from pydantic_core import to_jsonable_python

from repopilot.context import CONTEXT_POLICY_VERSION
from repopilot.enums import WorkingMemoryErrorCode
from repopilot.schemas import (
    AgentState,
    GitSha,
    Sha256,
    StrictModel,
    TaskId,
    VersionIdentifier,
)
from repopilot.trace import TraceEvent

WORKING_MEMORY_SCHEMA_VERSION: Literal[1] = 1
DEFAULT_MAX_WORKING_SNAPSHOTS = 32
MAX_CONFIGURED_WORKING_SNAPSHOTS = 128
MAX_WORKING_MEMORY_BYTES = 16 * 1024 * 1024
_SNAPSHOT_NAME = re.compile(r"^(?P<sequence>[0-9]{8})\.json$")
_THREAD_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"

WORKING_MEMORY_ERROR_MESSAGES: Mapping[WorkingMemoryErrorCode, str] = {
    WorkingMemoryErrorCode.INVALID_SNAPSHOT: "Working Memory snapshot is invalid",
    WorkingMemoryErrorCode.NAMESPACE_MISMATCH: (
        "Working Memory task or run namespace does not match"
    ),
    WorkingMemoryErrorCode.COMMIT_STALE: ("Working Memory repository commit is no longer current"),
    WorkingMemoryErrorCode.INTEGRITY_MISMATCH: ("Working Memory snapshot integrity check failed"),
    WorkingMemoryErrorCode.SEQUENCE_ROLLBACK: ("Working Memory snapshot sequence is not monotonic"),
    WorkingMemoryErrorCode.SNAPSHOT_LIMIT: "Working Memory snapshot limit was reached",
    WorkingMemoryErrorCode.TERMINAL_SNAPSHOT: (
        "terminal Agent state cannot be saved as Working Memory"
    ),
    WorkingMemoryErrorCode.RECOVERY_CONFLICT: (
        "Working Memory cannot be restored into the requested Agent thread"
    ),
    WorkingMemoryErrorCode.PATH_UNSAFE: "Working Memory storage path is unsafe",
    WorkingMemoryErrorCode.UNSUPPORTED_VERSION: ("Working Memory schema version is unsupported"),
    WorkingMemoryErrorCode.IO_ERROR: "Working Memory snapshot could not be read or written",
}


class WorkingMemoryError(RuntimeError):
    """Safe typed failure from Working Memory persistence or recovery."""

    def __init__(self, code: WorkingMemoryErrorCode) -> None:
        super().__init__(WORKING_MEMORY_ERROR_MESSAGES[code])
        self.code = code


class WorkingMemoryCheckpoint(StrictModel):
    """Self-contained nonterminal Agent checkpoint with its Trace prefix."""

    schema_version: Literal[1] = WORKING_MEMORY_SCHEMA_VERSION
    task_id: TaskId
    run_id: UUID
    source_thread_id: str = Field(pattern=_THREAD_PATTERN)
    repository_commit: GitSha
    context_policy_version: VersionIdentifier
    sequence: int = Field(ge=0, le=99_999_999)
    created_at: AwareDatetime
    state: AgentState
    trace_events: tuple[TraceEvent, ...] = Field(min_length=1)
    state_sha256: Sha256
    trace_sha256: Sha256
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_recovery_contract(self) -> Self:
        if self.state.run_id != self.run_id:
            raise ValueError("state run ID must match snapshot run ID")
        if self.state.task.task_id != self.task_id:
            raise ValueError("state task ID must match snapshot task ID")
        if self.state.task.base_commit != self.repository_commit:
            raise ValueError("state base commit must match snapshot commit")
        if self.state.termination_reason is not None:
            raise ValueError("Working Memory must be nonterminal")
        if not self.state.current_plan:
            raise ValueError("Working Memory requires an initialized plan")
        if self.state.active_step_id is None:
            raise ValueError("Working Memory has no resumable action")
        if self.state.finish_summary is not None and (
            self.state.pending_tool_call is not None or self.state.pending_tool_result is not None
        ):
            raise ValueError("finish recovery cannot contain pending tool work")
        self._validate_trace_prefix()
        latest_policy = context_policy_version_from_events(self.trace_events)
        if latest_policy != self.context_policy_version:
            raise ValueError("context policy version does not match Trace prefix")
        return self

    def _validate_trace_prefix(self) -> None:
        previous: AgentState | None = None
        for expected_sequence, event in enumerate(self.trace_events):
            if event.sequence != expected_sequence:
                raise ValueError("Trace prefix sequence is not contiguous")
            if (
                event.state_before.run_id != self.run_id
                or event.state_after.run_id != self.run_id
                or event.state_before.task.task_id != self.task_id
                or event.state_after.task.task_id != self.task_id
                or event.state_before.task.base_commit != self.repository_commit
                or event.state_after.task.base_commit != self.repository_commit
            ):
                raise ValueError("Trace prefix namespace is inconsistent")
            if previous is not None and event.state_before != previous:
                raise ValueError("Trace prefix state chain is inconsistent")
            previous = event.state_after
        if previous != self.state:
            raise ValueError("snapshot state must equal final Trace-prefix state")


class WorkingMemoryStore(Protocol):
    """Injected persistence boundary; Agent graphs never read environment paths."""

    def save(self, checkpoint: WorkingMemoryCheckpoint) -> None:
        """Persist exactly one monotonic snapshot without overwriting."""

    def load_latest(
        self,
        *,
        task_id: str,
        run_id: UUID,
        repository_commit: str,
        context_policy_version: str = CONTEXT_POLICY_VERSION,
    ) -> WorkingMemoryCheckpoint:
        """Load the newest snapshot inside one validated task/run namespace."""


def build_working_memory_checkpoint(
    *,
    task_id: str,
    run_id: UUID,
    source_thread_id: str,
    repository_commit: str,
    context_policy_version: str,
    sequence: int,
    created_at: datetime,
    state: AgentState,
    trace_events: tuple[TraceEvent, ...],
) -> WorkingMemoryCheckpoint:
    """Build all snapshot digests from canonical JSON."""

    state_digest = _content_sha256(state)
    trace_digest = _content_sha256(trace_events)
    payload: dict[str, object] = {
        "schema_version": WORKING_MEMORY_SCHEMA_VERSION,
        "task_id": task_id,
        "run_id": run_id,
        "source_thread_id": source_thread_id,
        "repository_commit": repository_commit,
        "context_policy_version": context_policy_version,
        "sequence": sequence,
        "created_at": created_at,
        "state": state,
        "trace_events": trace_events,
        "state_sha256": state_digest,
        "trace_sha256": trace_digest,
    }
    try:
        return WorkingMemoryCheckpoint.model_validate(
            {
                **payload,
                "content_sha256": _content_sha256(payload),
            }
        )
    except ValidationError as error:
        if state.termination_reason is not None:
            raise WorkingMemoryError(WorkingMemoryErrorCode.TERMINAL_SNAPSHOT) from None
        raise WorkingMemoryError(WorkingMemoryErrorCode.INVALID_SNAPSHOT) from error


def context_policy_version_from_events(events: tuple[TraceEvent, ...]) -> str:
    """Return the last model-visible context policy or the current default."""

    for event in reversed(events):
        request = event.provider_request
        if (
            request is not None
            and request.context_window is not None
            and request.context_window.policy is not None
        ):
            return request.context_window.policy.policy_version
    return CONTEXT_POLICY_VERSION


def verify_working_memory_integrity(checkpoint: WorkingMemoryCheckpoint) -> None:
    """Reject state, Trace, or envelope tampering with one stable error."""

    payload = checkpoint.model_dump(mode="python", exclude={"content_sha256"})
    if (
        _content_sha256(checkpoint.state) != checkpoint.state_sha256
        or _content_sha256(checkpoint.trace_events) != checkpoint.trace_sha256
        or _content_sha256(payload) != checkpoint.content_sha256
    ):
        raise WorkingMemoryError(WorkingMemoryErrorCode.INTEGRITY_MISMATCH)


class FileWorkingMemoryStore:
    """No-overwrite local JSON store with bounded namespaces and symlink checks."""

    def __init__(
        self,
        root: Path,
        *,
        max_snapshots: int = DEFAULT_MAX_WORKING_SNAPSHOTS,
    ) -> None:
        if not root.is_absolute():
            raise WorkingMemoryError(WorkingMemoryErrorCode.PATH_UNSAFE)
        if (
            isinstance(max_snapshots, bool)
            or not isinstance(max_snapshots, int)
            or not 1 <= max_snapshots <= MAX_CONFIGURED_WORKING_SNAPSHOTS
        ):
            raise ValueError("max_snapshots must be between 1 and 128")
        self._root = root
        self._max_snapshots = max_snapshots

    def save(self, checkpoint: WorkingMemoryCheckpoint) -> None:
        verify_working_memory_integrity(checkpoint)
        namespace = self._namespace(checkpoint.task_id, checkpoint.run_id, create=True)
        snapshots = self._snapshot_paths(namespace)
        if len(snapshots) >= self._max_snapshots:
            raise WorkingMemoryError(WorkingMemoryErrorCode.SNAPSHOT_LIMIT)
        expected_sequence = snapshots[-1][0] + 1 if snapshots else 0
        if checkpoint.sequence != expected_sequence:
            raise WorkingMemoryError(WorkingMemoryErrorCode.SEQUENCE_ROLLBACK)
        destination = namespace / f"{checkpoint.sequence:08d}.json"
        serialized = _canonical_json(checkpoint).encode("utf-8") + b"\n"
        if len(serialized) > MAX_WORKING_MEMORY_BYTES:
            raise WorkingMemoryError(WorkingMemoryErrorCode.INVALID_SNAPSHOT)
        self._atomic_create(destination, serialized)

    def load_latest(
        self,
        *,
        task_id: str,
        run_id: UUID,
        repository_commit: str,
        context_policy_version: str = CONTEXT_POLICY_VERSION,
    ) -> WorkingMemoryCheckpoint:
        namespace = self._namespace(task_id, run_id, create=False)
        snapshots = self._snapshot_paths(namespace)
        if not snapshots:
            raise WorkingMemoryError(WorkingMemoryErrorCode.IO_ERROR)
        checkpoint = self._read(snapshots[-1][1])
        if checkpoint.task_id != task_id or checkpoint.run_id != run_id:
            raise WorkingMemoryError(WorkingMemoryErrorCode.NAMESPACE_MISMATCH)
        if checkpoint.repository_commit != repository_commit:
            raise WorkingMemoryError(WorkingMemoryErrorCode.COMMIT_STALE)
        if checkpoint.context_policy_version != context_policy_version:
            raise WorkingMemoryError(WorkingMemoryErrorCode.RECOVERY_CONFLICT)
        return checkpoint

    def _namespace(self, task_id: str, run_id: UUID, *, create: bool) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", task_id):
            raise WorkingMemoryError(WorkingMemoryErrorCode.NAMESPACE_MISMATCH)
        try:
            root = self._safe_directory(self._root, create=create)
            task_directory = self._safe_directory(root / task_id, create=create)
            return self._safe_directory(task_directory / str(run_id), create=create)
        except FileNotFoundError:
            raise WorkingMemoryError(WorkingMemoryErrorCode.IO_ERROR) from None

    @staticmethod
    def _safe_directory(path: Path, *, create: bool) -> Path:
        try:
            if create:
                path.mkdir(mode=0o700, parents=False, exist_ok=True)
            metadata = path.lstat()
        except FileNotFoundError:
            raise
        except OSError:
            raise WorkingMemoryError(WorkingMemoryErrorCode.IO_ERROR) from None
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise WorkingMemoryError(WorkingMemoryErrorCode.PATH_UNSAFE)
        with suppress(OSError):
            path.chmod(0o700)
        return path

    @staticmethod
    def _snapshot_paths(namespace: Path) -> list[tuple[int, Path]]:
        try:
            entries = tuple(namespace.iterdir())
        except OSError:
            raise WorkingMemoryError(WorkingMemoryErrorCode.IO_ERROR) from None
        snapshots: list[tuple[int, Path]] = []
        for entry in entries:
            match = _SNAPSHOT_NAME.fullmatch(entry.name)
            if match is None:
                continue
            try:
                metadata = entry.lstat()
            except OSError:
                raise WorkingMemoryError(WorkingMemoryErrorCode.IO_ERROR) from None
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise WorkingMemoryError(WorkingMemoryErrorCode.PATH_UNSAFE)
            snapshots.append((int(match.group("sequence")), entry))
        snapshots.sort(key=lambda item: item[0])
        if any(sequence != expected for expected, (sequence, _) in enumerate(snapshots)):
            raise WorkingMemoryError(WorkingMemoryErrorCode.SEQUENCE_ROLLBACK)
        return snapshots

    @staticmethod
    def _atomic_create(destination: Path, serialized: bytes) -> None:
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination, follow_symlinks=False)
            temporary.unlink()
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except FileExistsError:
            raise WorkingMemoryError(WorkingMemoryErrorCode.SEQUENCE_ROLLBACK) from None
        except OSError:
            raise WorkingMemoryError(WorkingMemoryErrorCode.IO_ERROR) from None
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            with suppress(OSError):
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _read(path: Path) -> WorkingMemoryCheckpoint:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                serialized = stream.read(MAX_WORKING_MEMORY_BYTES + 1)
        except OSError:
            raise WorkingMemoryError(WorkingMemoryErrorCode.IO_ERROR) from None
        if len(serialized) > MAX_WORKING_MEMORY_BYTES:
            raise WorkingMemoryError(WorkingMemoryErrorCode.INVALID_SNAPSHOT)
        try:
            raw = json.loads(serialized)
        except (json.JSONDecodeError, UnicodeError):
            raise WorkingMemoryError(WorkingMemoryErrorCode.INVALID_SNAPSHOT) from None
        if not isinstance(raw, dict):
            raise WorkingMemoryError(WorkingMemoryErrorCode.INVALID_SNAPSHOT)
        if raw.get("schema_version") != WORKING_MEMORY_SCHEMA_VERSION:
            raise WorkingMemoryError(WorkingMemoryErrorCode.UNSUPPORTED_VERSION)
        state_digest = raw.get("state_sha256")
        trace_digest = raw.get("trace_sha256")
        content_digest = raw.get("content_sha256")
        if not all(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in (state_digest, trace_digest, content_digest)
        ):
            raise WorkingMemoryError(WorkingMemoryErrorCode.INVALID_SNAPSHOT)
        envelope = {key: value for key, value in raw.items() if key != "content_sha256"}
        if (
            _content_sha256(raw.get("state")) != state_digest
            or _content_sha256(raw.get("trace_events")) != trace_digest
            or _content_sha256(envelope) != content_digest
        ):
            raise WorkingMemoryError(WorkingMemoryErrorCode.INTEGRITY_MISMATCH)
        try:
            checkpoint = WorkingMemoryCheckpoint.model_validate(raw)
        except ValidationError:
            raise WorkingMemoryError(WorkingMemoryErrorCode.INVALID_SNAPSHOT) from None
        verify_working_memory_integrity(checkpoint)
        return checkpoint


def _canonical_json(payload: object) -> str:
    return json.dumps(
        _json_compatible(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _content_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _json_compatible(payload: object) -> JsonValue:
    return cast(JsonValue, to_jsonable_python(payload))


__all__ = [
    "DEFAULT_MAX_WORKING_SNAPSHOTS",
    "WORKING_MEMORY_SCHEMA_VERSION",
    "FileWorkingMemoryStore",
    "WorkingMemoryCheckpoint",
    "WorkingMemoryError",
    "WorkingMemoryStore",
    "build_working_memory_checkpoint",
    "context_policy_version_from_events",
    "verify_working_memory_integrity",
]
