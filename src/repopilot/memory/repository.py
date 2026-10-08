"""Commit/content-scoped Repository Memory with explicit invalidation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, Self, cast
from uuid import uuid4

from pydantic import (
    AwareDatetime,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import to_jsonable_python

from repopilot.enums import (
    RepositoryMemoryConfirmation,
    RepositoryMemoryErrorCode,
    RepositoryMemoryEvidenceKind,
    RepositoryMemoryKind,
    RepositoryMemoryLookupStatus,
)
from repopilot.retrieval.manifest import (
    RepositoryManifest,
    verify_repository_manifest_integrity,
)
from repopilot.schemas import (
    CommandSpec,
    GitSha,
    Identifier,
    RelativePath,
    RepositoryMemoryItem,
    Sha256,
    StrictModel,
    TaskSpec,
    VersionIdentifier,
)

REPOSITORY_MEMORY_SCHEMA_VERSION: Literal[1] = 1
REPOSITORY_MAP_PRODUCER_VERSION = "repository-map-v1"
TEST_COMMAND_PRODUCER_VERSION = "test-command-identity-v1"
CONFIRMED_FACT_PRODUCER_VERSION = "source-confirmation-v1"
DEFAULT_MAX_REPOSITORY_MEMORY_RECORDS = 256
DEFAULT_MAX_REPOSITORY_MEMORY_VERSIONS = 32
DEFAULT_MAX_REPOSITORY_MEMORY_RECORD_BYTES = 256 * 1024
MAX_REPOSITORY_MEMORY_RECORD_BYTES = 1024 * 1024
_MAX_CONTENT_CHARS = 8_000
_RECORD_NAME = re.compile(r"^(?P<record_id>[A-Za-z][A-Za-z0-9_.-]{0,127})\.json$")
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

REPOSITORY_MEMORY_ERROR_MESSAGES: Mapping[RepositoryMemoryErrorCode, str] = {
    RepositoryMemoryErrorCode.INVALID_RECORD: "Repository Memory record is invalid",
    RepositoryMemoryErrorCode.NAMESPACE_MISMATCH: (
        "Repository Memory namespace does not match the current repository"
    ),
    RepositoryMemoryErrorCode.COMMIT_STALE: "Repository Memory commit is stale",
    RepositoryMemoryErrorCode.CONTENT_STALE: "Repository Memory content identity is stale",
    RepositoryMemoryErrorCode.INTEGRITY_MISMATCH: ("Repository Memory integrity check failed"),
    RepositoryMemoryErrorCode.UNCONFIRMED_RECORD: (
        "Repository Memory record lacks trusted confirmation"
    ),
    RepositoryMemoryErrorCode.DUPLICATE_CONFLICT: (
        "Repository Memory record ID has conflicting content"
    ),
    RepositoryMemoryErrorCode.CAPACITY_EXCEEDED: ("Repository Memory capacity was exceeded"),
    RepositoryMemoryErrorCode.PATH_UNSAFE: "Repository Memory storage path is unsafe",
    RepositoryMemoryErrorCode.UNSUPPORTED_VERSION: (
        "Repository Memory schema version is unsupported"
    ),
    RepositoryMemoryErrorCode.IO_ERROR: ("Repository Memory record could not be read or written"),
}


class RepositoryMemoryError(RuntimeError):
    """Safe typed failure that never echoes record content or host paths."""

    def __init__(self, code: RepositoryMemoryErrorCode) -> None:
        super().__init__(REPOSITORY_MEMORY_ERROR_MESSAGES[code])
        self.code = code


class RepositoryMemoryEvidence(StrictModel):
    """Content-addressed evidence authorizing one repository-scoped record."""

    kind: RepositoryMemoryEvidenceKind
    source_id: Identifier
    relative_path: RelativePath | None = None
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.kind is RepositoryMemoryEvidenceKind.SOURCE:
            if self.relative_path is None:
                raise ValueError("source evidence requires relative_path")
        elif self.relative_path is not None:
            raise ValueError("non-source evidence cannot contain relative_path")
        return self


class RepositoryMemoryRecord(StrictModel):
    """One immutable, provenance-bearing repository fact or map entry."""

    schema_version: Literal[1] = REPOSITORY_MEMORY_SCHEMA_VERSION
    record_id: Identifier
    repository_id: Identifier
    repository_commit: GitSha
    repository_content_sha256: Sha256
    kind: RepositoryMemoryKind
    key: Identifier
    confirmation: RepositoryMemoryConfirmation
    producer_version: VersionIdentifier
    content: str = Field(min_length=1, max_length=_MAX_CONTENT_CHARS)
    evidence: tuple[RepositoryMemoryEvidence, ...] = Field(
        min_length=1,
        max_length=32,
    )
    created_at: AwareDatetime
    content_sha256: Sha256
    record_sha256: Sha256

    @field_validator("content")
    @classmethod
    def reject_nul_content(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("Repository Memory content cannot contain NUL")
        return value

    @model_validator(mode="after")
    def validate_identity_and_confirmation(self) -> Self:
        if self.record_id != repository_memory_record_id(
            self.repository_id,
            self.kind,
            self.key,
        ):
            raise ValueError("record ID does not match repository/kind/key")
        if self.content_sha256 != _sha256_text(self.content):
            raise ValueError("content digest does not match record content")
        evidence_keys = tuple(
            (
                item.kind.value,
                item.source_id,
                item.relative_path or "",
                item.content_sha256,
            )
            for item in self.evidence
        )
        if evidence_keys != tuple(sorted(evidence_keys)) or len(set(evidence_keys)) != len(
            evidence_keys
        ):
            raise ValueError("record evidence must be unique and sorted")
        expected_confirmation = {
            RepositoryMemoryKind.REPOSITORY_MAP: (RepositoryMemoryConfirmation.MANIFEST_DERIVED),
            RepositoryMemoryKind.TEST_COMMAND: (RepositoryMemoryConfirmation.TASK_CONFIGURED),
            RepositoryMemoryKind.CONFIRMED_FACT: (RepositoryMemoryConfirmation.SOURCE_CONFIRMED),
        }[self.kind]
        if self.confirmation is not expected_confirmation:
            raise ValueError("record kind has an invalid confirmation source")
        expected_evidence_kind = {
            RepositoryMemoryKind.REPOSITORY_MAP: RepositoryMemoryEvidenceKind.MANIFEST,
            RepositoryMemoryKind.TEST_COMMAND: (RepositoryMemoryEvidenceKind.TASK_CONFIGURATION),
            RepositoryMemoryKind.CONFIRMED_FACT: RepositoryMemoryEvidenceKind.SOURCE,
        }[self.kind]
        if any(item.kind is not expected_evidence_kind for item in self.evidence):
            raise ValueError("record kind has invalid evidence")
        return self


class RepositoryMemoryLookup(StrictModel):
    """Explicit current/miss/stale result; stale records are never returned."""

    repository_id: Identifier
    requested_commit: GitSha
    requested_content_sha256: Sha256
    status: RepositoryMemoryLookupStatus
    records: tuple[RepositoryMemoryRecord, ...] = ()
    stale_record_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status is RepositoryMemoryLookupStatus.CURRENT:
            if not self.records or self.stale_record_count != 0:
                raise ValueError("current lookup requires records and no stale count")
            if any(
                record.repository_id != self.repository_id
                or record.repository_commit != self.requested_commit
                or record.repository_content_sha256 != self.requested_content_sha256
                for record in self.records
            ):
                raise ValueError("current lookup contains a mismatched record")
        elif self.records:
            raise ValueError("non-current lookup cannot expose records")
        if (
            self.status
            in {
                RepositoryMemoryLookupStatus.COMMIT_STALE,
                RepositoryMemoryLookupStatus.CONTENT_STALE,
            }
            and self.stale_record_count == 0
        ):
            raise ValueError("stale lookup requires an invalidated record count")
        if self.status is RepositoryMemoryLookupStatus.EMPTY and self.stale_record_count != 0:
            raise ValueError("empty lookup cannot contain stale records")
        return self


class RepositoryMemoryStore(Protocol):
    """Injected persistence boundary used only by trusted application code."""

    def put(
        self,
        record: RepositoryMemoryRecord,
        *,
        manifest: RepositoryManifest,
        task: TaskSpec | None = None,
    ) -> bool:
        """Create one record, return False for an exact idempotent duplicate."""

    def lookup(self, manifest: RepositoryManifest) -> RepositoryMemoryLookup:
        """Return current records or an explicit empty/stale result."""


def repository_memory_record_id(
    repository_id: str,
    kind: RepositoryMemoryKind,
    key: str,
) -> str:
    """Derive a stable ID that remains comparable across repository versions."""

    return f"rmem_{_canonical_sha256({'repository_id': repository_id, 'kind': kind, 'key': key})}"


def build_repository_map_record(
    manifest: RepositoryManifest,
    *,
    created_at: datetime,
) -> RepositoryMemoryRecord:
    """Build a bounded deterministic map from an integrity-checked Manifest."""

    verify_repository_manifest_integrity(manifest)
    top_level = tuple(sorted({source.relative_path.split("/", 1)[0] for source in manifest.files}))
    suffix_counts = Counter(
        PurePosixPath(source.relative_path).suffix.lower() or "<none>" for source in manifest.files
    )
    content = _canonical_json(
        {
            "memory_version": REPOSITORY_MAP_PRODUCER_VERSION,
            "source_count": len(manifest.files),
            "skipped_source_count": len(manifest.skipped_sources),
            "top_level_entries": top_level[:64],
            "file_suffix_counts": dict(sorted(suffix_counts.items())),
        }
    )
    evidence = (
        RepositoryMemoryEvidence(
            kind=RepositoryMemoryEvidenceKind.MANIFEST,
            source_id=f"manifest_{manifest.content_sha256}",
            content_sha256=manifest.content_sha256,
        ),
    )
    return _build_record(
        repository_id=manifest.repository_id,
        repository_commit=manifest.commit,
        repository_content_sha256=manifest.content_sha256,
        kind=RepositoryMemoryKind.REPOSITORY_MAP,
        key="repository-map",
        confirmation=RepositoryMemoryConfirmation.MANIFEST_DERIVED,
        producer_version=REPOSITORY_MAP_PRODUCER_VERSION,
        content=content,
        evidence=evidence,
        created_at=created_at,
    )


def build_test_command_record(
    manifest: RepositoryManifest,
    task: TaskSpec,
    *,
    test_index: int,
    created_at: datetime,
) -> RepositoryMemoryRecord:
    """Persist command identity/shape without exposing the complete argv."""

    verify_repository_manifest_integrity(manifest)
    if task.base_commit != manifest.commit:
        raise RepositoryMemoryError(RepositoryMemoryErrorCode.NAMESPACE_MISMATCH)
    try:
        command = task.test_commands[test_index]
    except IndexError:
        raise RepositoryMemoryError(RepositoryMemoryErrorCode.INVALID_RECORD) from None
    command_digest = command_spec_sha256(command)
    executable = PurePosixPath(command.argv[0]).name
    content = _canonical_json(
        {
            "memory_version": TEST_COMMAND_PRODUCER_VERSION,
            "command_sha256": command_digest,
            "executable": executable,
            "argument_count": len(command.argv),
            "cwd": command.cwd,
            "timeout_seconds": command.timeout_seconds,
        }
    )
    evidence = (
        RepositoryMemoryEvidence(
            kind=RepositoryMemoryEvidenceKind.TASK_CONFIGURATION,
            source_id=f"test_command_{command_digest}",
            content_sha256=command_digest,
        ),
    )
    return _build_record(
        repository_id=manifest.repository_id,
        repository_commit=manifest.commit,
        repository_content_sha256=manifest.content_sha256,
        kind=RepositoryMemoryKind.TEST_COMMAND,
        key=f"test-command-{command_digest[:16]}",
        confirmation=RepositoryMemoryConfirmation.TASK_CONFIGURED,
        producer_version=TEST_COMMAND_PRODUCER_VERSION,
        content=content,
        evidence=evidence,
        created_at=created_at,
    )


def build_confirmed_fact_record(
    manifest: RepositoryManifest,
    *,
    key: str,
    fact: str,
    evidence_paths: tuple[str, ...],
    created_at: datetime,
    producer_version: str = CONFIRMED_FACT_PRODUCER_VERSION,
) -> RepositoryMemoryRecord:
    """Build a source-backed fact through a trusted, non-Agent write API."""

    verify_repository_manifest_integrity(manifest)
    normalized_fact = fact.strip()
    if not normalized_fact or len(normalized_fact) > 4_000 or "\x00" in normalized_fact:
        raise RepositoryMemoryError(RepositoryMemoryErrorCode.INVALID_RECORD)
    sources_by_path = {source.relative_path: source for source in manifest.files}
    try:
        sources = tuple(sources_by_path[path] for path in evidence_paths)
    except KeyError:
        raise RepositoryMemoryError(RepositoryMemoryErrorCode.UNCONFIRMED_RECORD) from None
    if not sources or len(set(evidence_paths)) != len(evidence_paths):
        raise RepositoryMemoryError(RepositoryMemoryErrorCode.UNCONFIRMED_RECORD)
    evidence = tuple(
        sorted(
            (
                RepositoryMemoryEvidence(
                    kind=RepositoryMemoryEvidenceKind.SOURCE,
                    source_id=source.source_id,
                    relative_path=source.relative_path,
                    content_sha256=source.content_hash,
                )
                for source in sources
            ),
            key=_evidence_sort_key,
        )
    )
    content = _canonical_json(
        {
            "memory_version": CONFIRMED_FACT_PRODUCER_VERSION,
            "fact_key": key,
            "fact": normalized_fact,
        }
    )
    try:
        return _build_record(
            repository_id=manifest.repository_id,
            repository_commit=manifest.commit,
            repository_content_sha256=manifest.content_sha256,
            kind=RepositoryMemoryKind.CONFIRMED_FACT,
            key=key,
            confirmation=RepositoryMemoryConfirmation.SOURCE_CONFIRMED,
            producer_version=producer_version,
            content=content,
            evidence=evidence,
            created_at=created_at,
        )
    except ValidationError:
        raise RepositoryMemoryError(RepositoryMemoryErrorCode.INVALID_RECORD) from None


def command_spec_sha256(command: CommandSpec) -> str:
    """Hash the complete trusted command without projecting it to the model."""

    return _canonical_sha256(command)


def project_repository_memory_record(
    record: RepositoryMemoryRecord,
) -> RepositoryMemoryItem:
    """Project a verified record into bounded, model-visible Agent state."""

    verify_repository_memory_integrity(record)
    return RepositoryMemoryItem(
        record_id=record.record_id,
        repository_id=record.repository_id,
        repository_commit=record.repository_commit,
        repository_content_sha256=record.repository_content_sha256,
        kind=record.kind,
        confirmation=record.confirmation,
        producer_version=record.producer_version,
        content=record.content,
        content_sha256=record.content_sha256,
        evidence_count=len(record.evidence),
        evidence_sha256=_canonical_sha256(record.evidence),
        source_record_sha256=record.record_sha256,
    )


def verify_repository_memory_integrity(record: RepositoryMemoryRecord) -> None:
    """Reject content or envelope tampering."""

    payload = record.model_dump(mode="python", exclude={"record_sha256"})
    if (
        _sha256_text(record.content) != record.content_sha256
        or _canonical_sha256(payload) != record.record_sha256
    ):
        raise RepositoryMemoryError(RepositoryMemoryErrorCode.INTEGRITY_MISMATCH)


def validate_repository_memory_record(
    record: RepositoryMemoryRecord,
    manifest: RepositoryManifest,
    *,
    task: TaskSpec | None = None,
    require_task_configuration: bool = False,
) -> None:
    """Validate record namespace and evidence against trusted current inputs."""

    verify_repository_manifest_integrity(manifest)
    verify_repository_memory_integrity(record)
    if record.repository_id != manifest.repository_id:
        raise RepositoryMemoryError(RepositoryMemoryErrorCode.NAMESPACE_MISMATCH)
    if record.repository_commit != manifest.commit:
        raise RepositoryMemoryError(RepositoryMemoryErrorCode.COMMIT_STALE)
    if record.repository_content_sha256 != manifest.content_sha256:
        raise RepositoryMemoryError(RepositoryMemoryErrorCode.CONTENT_STALE)
    sources = {source.source_id: source for source in manifest.files}
    for evidence in record.evidence:
        if evidence.kind is RepositoryMemoryEvidenceKind.MANIFEST:
            if (
                evidence.source_id != f"manifest_{manifest.content_sha256}"
                or evidence.content_sha256 != manifest.content_sha256
            ):
                raise RepositoryMemoryError(RepositoryMemoryErrorCode.UNCONFIRMED_RECORD)
        elif evidence.kind is RepositoryMemoryEvidenceKind.SOURCE:
            source = sources.get(evidence.source_id)
            if (
                source is None
                or evidence.relative_path != source.relative_path
                or evidence.content_sha256 != source.content_hash
            ):
                raise RepositoryMemoryError(RepositoryMemoryErrorCode.UNCONFIRMED_RECORD)
        elif require_task_configuration:
            if task is None or task.base_commit != manifest.commit:
                raise RepositoryMemoryError(RepositoryMemoryErrorCode.UNCONFIRMED_RECORD)
            command_hashes = {command_spec_sha256(command) for command in task.test_commands}
            if evidence.content_sha256 not in command_hashes:
                raise RepositoryMemoryError(RepositoryMemoryErrorCode.UNCONFIRMED_RECORD)


class FileRepositoryMemoryStore:
    """Bounded no-overwrite store partitioned by repository/commit/content."""

    def __init__(
        self,
        root: Path,
        *,
        max_records_per_version: int = DEFAULT_MAX_REPOSITORY_MEMORY_RECORDS,
        max_versions_per_repository: int = DEFAULT_MAX_REPOSITORY_MEMORY_VERSIONS,
        max_record_bytes: int = DEFAULT_MAX_REPOSITORY_MEMORY_RECORD_BYTES,
    ) -> None:
        if not root.is_absolute():
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.PATH_UNSAFE)
        _validate_limit(max_records_per_version, 1, 4096, "max_records_per_version")
        _validate_limit(max_versions_per_repository, 1, 256, "max_versions_per_repository")
        _validate_limit(
            max_record_bytes,
            1024,
            MAX_REPOSITORY_MEMORY_RECORD_BYTES,
            "max_record_bytes",
        )
        self._root = root
        self._max_records = max_records_per_version
        self._max_versions = max_versions_per_repository
        self._max_record_bytes = max_record_bytes

    def put(
        self,
        record: RepositoryMemoryRecord,
        *,
        manifest: RepositoryManifest,
        task: TaskSpec | None = None,
    ) -> bool:
        validate_repository_memory_record(
            record,
            manifest,
            task=task,
            require_task_configuration=(record.kind is RepositoryMemoryKind.TEST_COMMAND),
        )
        repository_dir = self._repository_directory(record.repository_id, create=True)
        assert repository_dir is not None
        version_directories = self._version_directories(repository_dir)
        version_key = (record.repository_commit, record.repository_content_sha256)
        if (
            version_key not in version_directories
            and len(version_directories) >= self._max_versions
        ):
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.CAPACITY_EXCEEDED)
        namespace = self._version_directory(
            repository_dir,
            record.repository_commit,
            record.repository_content_sha256,
            create=True,
        )
        paths = self._record_paths(namespace)
        destination = namespace / f"{record.record_id}.json"
        existing_path = next(
            (path for record_id, path in paths if record_id == record.record_id),
            None,
        )
        if existing_path is not None:
            existing = self._read(existing_path)
            if _semantic_record_payload(existing) == _semantic_record_payload(record):
                return False
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.DUPLICATE_CONFLICT)
        if len(paths) >= self._max_records:
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.CAPACITY_EXCEEDED)
        serialized = _canonical_json(record).encode("utf-8") + b"\n"
        if len(serialized) > self._max_record_bytes:
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.CAPACITY_EXCEEDED)
        self._atomic_create(destination, serialized)
        return True

    def lookup(self, manifest: RepositoryManifest) -> RepositoryMemoryLookup:
        verify_repository_manifest_integrity(manifest)
        repository_dir = self._repository_directory(
            manifest.repository_id,
            create=False,
            missing_ok=True,
        )
        if repository_dir is None:
            return self._lookup_result(manifest, RepositoryMemoryLookupStatus.EMPTY)
        versions = self._version_directories(repository_dir)
        exact_key = (manifest.commit, manifest.content_sha256)
        namespace = versions.get(exact_key)
        if namespace is not None:
            paths = self._record_paths(namespace)
            if not paths:
                return self._lookup_result(manifest, RepositoryMemoryLookupStatus.EMPTY)
            records = tuple(self._read(path) for _, path in paths)
            for record in records:
                validate_repository_memory_record(record, manifest)
            return RepositoryMemoryLookup(
                repository_id=manifest.repository_id,
                requested_commit=manifest.commit,
                requested_content_sha256=manifest.content_sha256,
                status=RepositoryMemoryLookupStatus.CURRENT,
                records=records,
            )
        same_commit = tuple(
            path for (commit, _), path in versions.items() if commit == manifest.commit
        )
        if same_commit:
            return self._lookup_result(
                manifest,
                RepositoryMemoryLookupStatus.CONTENT_STALE,
                stale_record_count=sum(self._count_records(path) for path in same_commit),
            )
        stale_count = sum(self._count_records(path) for path in versions.values())
        if stale_count:
            return self._lookup_result(
                manifest,
                RepositoryMemoryLookupStatus.COMMIT_STALE,
                stale_record_count=stale_count,
            )
        return self._lookup_result(manifest, RepositoryMemoryLookupStatus.EMPTY)

    def _repository_directory(
        self,
        repository_id: str,
        *,
        create: bool,
        missing_ok: bool = False,
    ) -> Path | None:
        if _IDENTIFIER.fullmatch(repository_id) is None:
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.NAMESPACE_MISMATCH)
        root = self._safe_directory(self._root, create=create, missing_ok=missing_ok)
        if root is None:
            return None
        return self._safe_directory(
            root / repository_id,
            create=create,
            missing_ok=missing_ok,
        )

    def _version_directory(
        self,
        repository_dir: Path,
        commit: str,
        content_sha256: str,
        *,
        create: bool,
    ) -> Path:
        commit_dir = self._safe_directory(repository_dir / commit, create=create)
        assert commit_dir is not None
        version_dir = self._safe_directory(
            commit_dir / content_sha256,
            create=create,
        )
        assert version_dir is not None
        return version_dir

    def _version_directories(
        self,
        repository_dir: Path,
    ) -> dict[tuple[str, str], Path]:
        versions: dict[tuple[str, str], Path] = {}
        for commit_entry in self._entries(repository_dir):
            metadata = self._lstat(commit_entry)
            if not stat.S_ISDIR(metadata.st_mode) or _GIT_SHA.fullmatch(commit_entry.name) is None:
                raise RepositoryMemoryError(RepositoryMemoryErrorCode.PATH_UNSAFE)
            for content_entry in self._entries(commit_entry):
                content_metadata = self._lstat(content_entry)
                if (
                    not stat.S_ISDIR(content_metadata.st_mode)
                    or _SHA256.fullmatch(content_entry.name) is None
                ):
                    raise RepositoryMemoryError(RepositoryMemoryErrorCode.PATH_UNSAFE)
                versions[(commit_entry.name, content_entry.name)] = content_entry
        return versions

    def _record_paths(self, namespace: Path) -> tuple[tuple[str, Path], ...]:
        paths: list[tuple[str, Path]] = []
        for entry in self._entries(namespace):
            match = _RECORD_NAME.fullmatch(entry.name)
            metadata = self._lstat(entry)
            if match is None or not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
                raise RepositoryMemoryError(RepositoryMemoryErrorCode.PATH_UNSAFE)
            paths.append((match.group("record_id"), entry))
        paths.sort(key=lambda item: item[0])
        return tuple(paths)

    def _count_records(self, namespace: Path) -> int:
        return len(self._record_paths(namespace))

    @staticmethod
    def _safe_directory(
        path: Path,
        *,
        create: bool,
        missing_ok: bool = False,
    ) -> Path | None:
        try:
            if create:
                path.mkdir(mode=0o700, parents=False, exist_ok=True)
            metadata = path.lstat()
        except FileNotFoundError:
            if missing_ok:
                return None
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.IO_ERROR) from None
        except OSError:
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.IO_ERROR) from None
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.PATH_UNSAFE)
        with suppress(OSError):
            path.chmod(0o700)
        return path

    @staticmethod
    def _entries(path: Path) -> tuple[Path, ...]:
        try:
            return tuple(sorted(path.iterdir(), key=lambda item: item.name))
        except OSError:
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.IO_ERROR) from None

    @staticmethod
    def _lstat(path: Path) -> os.stat_result:
        try:
            metadata = path.lstat()
        except OSError:
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.IO_ERROR) from None
        if stat.S_ISLNK(metadata.st_mode):
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.PATH_UNSAFE)
        return metadata

    def _read(self, path: Path) -> RepositoryMemoryRecord:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                serialized = stream.read(self._max_record_bytes + 1)
        except OSError:
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.IO_ERROR) from None
        if len(serialized) > self._max_record_bytes:
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.CAPACITY_EXCEEDED)
        try:
            raw = json.loads(serialized)
        except (json.JSONDecodeError, UnicodeError):
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.INVALID_RECORD) from None
        if not isinstance(raw, dict):
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.INVALID_RECORD)
        if raw.get("schema_version") != REPOSITORY_MEMORY_SCHEMA_VERSION:
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.UNSUPPORTED_VERSION)
        record_digest = raw.get("record_sha256")
        if not isinstance(record_digest, str) or _SHA256.fullmatch(record_digest) is None:
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.INVALID_RECORD)
        envelope = {key: value for key, value in raw.items() if key != "record_sha256"}
        if _canonical_sha256(envelope) != record_digest:
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.INTEGRITY_MISMATCH)
        try:
            record = RepositoryMemoryRecord.model_validate(raw)
        except ValidationError:
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.INVALID_RECORD) from None
        verify_repository_memory_integrity(record)
        if path.stem != record.record_id:
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.NAMESPACE_MISMATCH)
        return record

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
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.DUPLICATE_CONFLICT) from None
        except OSError:
            raise RepositoryMemoryError(RepositoryMemoryErrorCode.IO_ERROR) from None
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            with suppress(OSError):
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _lookup_result(
        manifest: RepositoryManifest,
        status: RepositoryMemoryLookupStatus,
        *,
        stale_record_count: int = 0,
    ) -> RepositoryMemoryLookup:
        return RepositoryMemoryLookup(
            repository_id=manifest.repository_id,
            requested_commit=manifest.commit,
            requested_content_sha256=manifest.content_sha256,
            status=status,
            stale_record_count=stale_record_count,
        )


def _build_record(
    *,
    repository_id: str,
    repository_commit: str,
    repository_content_sha256: str,
    kind: RepositoryMemoryKind,
    key: str,
    confirmation: RepositoryMemoryConfirmation,
    producer_version: str,
    content: str,
    evidence: tuple[RepositoryMemoryEvidence, ...],
    created_at: datetime,
) -> RepositoryMemoryRecord:
    payload: dict[str, object] = {
        "schema_version": REPOSITORY_MEMORY_SCHEMA_VERSION,
        "record_id": repository_memory_record_id(repository_id, kind, key),
        "repository_id": repository_id,
        "repository_commit": repository_commit,
        "repository_content_sha256": repository_content_sha256,
        "kind": kind,
        "key": key,
        "confirmation": confirmation,
        "producer_version": producer_version,
        "content": content,
        "evidence": evidence,
        "created_at": created_at,
        "content_sha256": _sha256_text(content),
    }
    try:
        return RepositoryMemoryRecord.model_validate(
            {
                **payload,
                "record_sha256": _canonical_sha256(payload),
            }
        )
    except ValidationError:
        raise RepositoryMemoryError(RepositoryMemoryErrorCode.INVALID_RECORD) from None


def _validate_limit(value: int, minimum: int, maximum: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _evidence_sort_key(
    evidence: RepositoryMemoryEvidence,
) -> tuple[str, str, str, str]:
    return (
        evidence.kind.value,
        evidence.source_id,
        evidence.relative_path or "",
        evidence.content_sha256,
    )


def _semantic_record_payload(record: RepositoryMemoryRecord) -> dict[str, object]:
    """Exclude first-seen time/digest when deciding idempotent semantic equality."""

    return record.model_dump(
        mode="python",
        exclude={"created_at", "record_sha256"},
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        _json_compatible(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_compatible(payload: object) -> JsonValue:
    return cast(JsonValue, to_jsonable_python(payload))


__all__ = [
    "CONFIRMED_FACT_PRODUCER_VERSION",
    "DEFAULT_MAX_REPOSITORY_MEMORY_RECORDS",
    "DEFAULT_MAX_REPOSITORY_MEMORY_RECORD_BYTES",
    "DEFAULT_MAX_REPOSITORY_MEMORY_VERSIONS",
    "REPOSITORY_MAP_PRODUCER_VERSION",
    "REPOSITORY_MEMORY_SCHEMA_VERSION",
    "TEST_COMMAND_PRODUCER_VERSION",
    "FileRepositoryMemoryStore",
    "RepositoryMemoryError",
    "RepositoryMemoryEvidence",
    "RepositoryMemoryLookup",
    "RepositoryMemoryRecord",
    "RepositoryMemoryStore",
    "build_confirmed_fact_record",
    "build_repository_map_record",
    "build_test_command_record",
    "command_spec_sha256",
    "project_repository_memory_record",
    "repository_memory_record_id",
    "validate_repository_memory_record",
    "verify_repository_memory_integrity",
]
