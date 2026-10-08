"""Experiment/snapshot/split namespaces for isolated Memory stores."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Self, cast
from uuid import uuid4

from pydantic import AwareDatetime, ValidationError, model_validator
from pydantic_core import to_jsonable_python

from repopilot.enums import MemoryAccessMode, MemoryNamespaceErrorCode
from repopilot.memory.episodic import (
    DEFAULT_MAX_VERIFIED_EPISODE_BYTES,
    DEFAULT_MAX_VERIFIED_EPISODES_PER_REPOSITORY,
    DEFAULT_VERIFIED_EPISODE_TTL_DAYS,
    FileVerifiedEpisodicMemoryStore,
    VerifiedEpisodeExecutionEvidence,
    VerifiedEpisodeLookup,
    VerifiedEpisodicMemoryStore,
    VerifiedEpisodicRecord,
)
from repopilot.memory.repository import (
    DEFAULT_MAX_REPOSITORY_MEMORY_RECORD_BYTES,
    DEFAULT_MAX_REPOSITORY_MEMORY_RECORDS,
    DEFAULT_MAX_REPOSITORY_MEMORY_VERSIONS,
    FileRepositoryMemoryStore,
    RepositoryMemoryLookup,
    RepositoryMemoryRecord,
    RepositoryMemoryStore,
)
from repopilot.retrieval.manifest import RepositoryManifest
from repopilot.schemas import (
    FinalResult,
    MemoryNamespace,
    Sha256,
    StrictModel,
    TaskSpec,
)
from repopilot.trace import AgentTrace

MEMORY_NAMESPACE_SCHEMA_VERSION: Literal[1] = 1
MEMORY_NAMESPACE_MANIFEST = "namespace.json"

MEMORY_NAMESPACE_ERROR_MESSAGES: Mapping[MemoryNamespaceErrorCode, str] = {
    MemoryNamespaceErrorCode.INVALID_NAMESPACE: "Memory namespace is invalid",
    MemoryNamespaceErrorCode.SPLIT_MISMATCH: "Memory dataset split does not match",
    MemoryNamespaceErrorCode.SNAPSHOT_MISMATCH: "Memory snapshot does not match",
    MemoryNamespaceErrorCode.ACCESS_DENIED: "Memory namespace does not permit writes",
    MemoryNamespaceErrorCode.INTEGRITY_MISMATCH: "Memory namespace integrity check failed",
    MemoryNamespaceErrorCode.PATH_UNSAFE: "Memory namespace path is unsafe",
    MemoryNamespaceErrorCode.UNSUPPORTED_VERSION: (
        "Memory namespace schema version is unsupported"
    ),
    MemoryNamespaceErrorCode.IO_ERROR: "Memory namespace could not be read or written",
}


class MemoryNamespaceError(RuntimeError):
    """Safe namespace failure that never echoes host paths or stored content."""

    def __init__(self, code: MemoryNamespaceErrorCode) -> None:
        super().__init__(MEMORY_NAMESPACE_ERROR_MESSAGES[code])
        self.code = code


class MemoryNamespaceManifest(StrictModel):
    """Integrity-protected declaration stored beside one split-specific snapshot."""

    schema_version: Literal[1] = MEMORY_NAMESPACE_SCHEMA_VERSION
    namespace: MemoryNamespace
    created_at: AwareDatetime
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        payload = self.model_dump(mode="python", exclude={"content_sha256"})
        if _canonical_sha256(payload) != self.content_sha256:
            raise ValueError("Memory namespace manifest digest does not match")
        return self


class NamespacedRepositoryMemoryStore:
    """Repository Memory delegate enforcing namespace write access."""

    def __init__(
        self,
        namespace: MemoryNamespace,
        delegate: RepositoryMemoryStore,
    ) -> None:
        self._namespace = _validated_namespace(namespace)
        self._delegate = delegate

    @property
    def namespace(self) -> MemoryNamespace:
        return self._namespace

    def put(
        self,
        record: RepositoryMemoryRecord,
        *,
        manifest: RepositoryManifest,
        task: TaskSpec | None = None,
    ) -> bool:
        if self._namespace.access_mode is not MemoryAccessMode.READ_WRITE:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.ACCESS_DENIED)
        return self._delegate.put(record, manifest=manifest, task=task)

    def lookup(self, manifest: RepositoryManifest) -> RepositoryMemoryLookup:
        return self._delegate.lookup(manifest)


class NamespacedVerifiedEpisodicMemoryStore:
    """Verified Episodic delegate enforcing namespace write access."""

    def __init__(
        self,
        namespace: MemoryNamespace,
        delegate: VerifiedEpisodicMemoryStore,
    ) -> None:
        self._namespace = _validated_namespace(namespace)
        self._delegate = delegate

    @property
    def namespace(self) -> MemoryNamespace:
        return self._namespace

    def admit(
        self,
        *,
        repository_id: str,
        repository_content_sha256: str,
        result: FinalResult,
        trace: AgentTrace,
        execution: VerifiedEpisodeExecutionEvidence,
        artifact_root: Path,
        created_at: datetime,
    ) -> VerifiedEpisodicRecord:
        if self._namespace.access_mode is not MemoryAccessMode.READ_WRITE:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.ACCESS_DENIED)
        return self._delegate.admit(
            repository_id=repository_id,
            repository_content_sha256=repository_content_sha256,
            result=result,
            trace=trace,
            execution=execution,
            artifact_root=artifact_root,
            created_at=created_at,
        )

    def lookup(
        self,
        *,
        repository_id: str,
        repository_commit: str,
        repository_content_sha256: str,
        as_of: datetime,
    ) -> VerifiedEpisodeLookup:
        return self._delegate.lookup(
            repository_id=repository_id,
            repository_commit=repository_commit,
            repository_content_sha256=repository_content_sha256,
            as_of=as_of,
        )


@dataclass(frozen=True)
class MemoryStoreBundle:
    """One namespace plus its independently typed Repository/Episodic stores."""

    namespace: MemoryNamespace
    repository: NamespacedRepositoryMemoryStore
    episodic: NamespacedVerifiedEpisodicMemoryStore
    manifest: MemoryNamespaceManifest

    def __post_init__(self) -> None:
        if (
            self.repository.namespace != self.namespace
            or self.episodic.namespace != self.namespace
            or self.manifest.namespace != self.namespace
        ):
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.INVALID_NAMESPACE)


class FileMemoryNamespaceManager:
    """Create or reopen immutable namespace manifests beneath one absolute base root."""

    def __init__(self, root: Path) -> None:
        if not root.is_absolute():
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.PATH_UNSAFE)
        self._root = root

    def open(
        self,
        namespace: MemoryNamespace,
        *,
        created_at: datetime,
        repository_max_records_per_version: int = (DEFAULT_MAX_REPOSITORY_MEMORY_RECORDS),
        repository_max_versions: int = DEFAULT_MAX_REPOSITORY_MEMORY_VERSIONS,
        repository_max_record_bytes: int = (DEFAULT_MAX_REPOSITORY_MEMORY_RECORD_BYTES),
        episodic_ttl_days: int = DEFAULT_VERIFIED_EPISODE_TTL_DAYS,
        episodic_max_records: int = DEFAULT_MAX_VERIFIED_EPISODES_PER_REPOSITORY,
        episodic_max_record_bytes: int = DEFAULT_MAX_VERIFIED_EPISODE_BYTES,
    ) -> MemoryStoreBundle:
        """Open one exact namespace, creating only its immutable declaration if absent."""

        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.INVALID_NAMESPACE)
        namespace = _validated_namespace(namespace)
        namespace_root = self._namespace_root(namespace)
        manifest_path = namespace_root / MEMORY_NAMESPACE_MANIFEST
        manifest = self._open_manifest(manifest_path, namespace, created_at)
        repository_store = FileRepositoryMemoryStore(
            namespace_root / "repository",
            max_records_per_version=repository_max_records_per_version,
            max_versions_per_repository=repository_max_versions,
            max_record_bytes=repository_max_record_bytes,
        )
        episodic_store = FileVerifiedEpisodicMemoryStore(
            namespace_root / "episodic",
            ttl_days=episodic_ttl_days,
            max_records_per_repository=episodic_max_records,
            max_record_bytes=episodic_max_record_bytes,
        )
        return MemoryStoreBundle(
            namespace=namespace,
            repository=NamespacedRepositoryMemoryStore(namespace, repository_store),
            episodic=NamespacedVerifiedEpisodicMemoryStore(namespace, episodic_store),
            manifest=manifest,
        )

    def _namespace_root(self, namespace: MemoryNamespace) -> Path:
        root = self._safe_directory(self._root, create=True)
        experiment = self._safe_directory(root / namespace.experiment_id, create=True)
        snapshot = self._safe_directory(experiment / namespace.snapshot_id, create=True)
        return self._safe_directory(snapshot / namespace.split.value, create=True)

    def _open_manifest(
        self,
        path: Path,
        namespace: MemoryNamespace,
        created_at: datetime,
    ) -> MemoryNamespaceManifest:
        try:
            exists = path.exists()
        except OSError:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.IO_ERROR) from None
        if exists:
            manifest = self._read_manifest(path)
            self._validate_manifest_namespace(manifest, namespace)
            return manifest
        siblings = tuple(item for item in self._entries(path.parent) if item.name != path.name)
        if siblings:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.INTEGRITY_MISMATCH)
        payload: dict[str, object] = {
            "schema_version": MEMORY_NAMESPACE_SCHEMA_VERSION,
            "namespace": namespace,
            "created_at": created_at,
        }
        try:
            manifest = MemoryNamespaceManifest.model_validate(
                {
                    **payload,
                    "content_sha256": _canonical_sha256(payload),
                }
            )
        except ValidationError:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.INVALID_NAMESPACE) from None
        serialized = _canonical_json(manifest).encode("utf-8") + b"\n"
        self._atomic_create(path, serialized)
        return manifest

    @staticmethod
    def _validate_manifest_namespace(
        manifest: MemoryNamespaceManifest,
        requested: MemoryNamespace,
    ) -> None:
        current = manifest.namespace
        if current.experiment_id != requested.experiment_id:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.INVALID_NAMESPACE)
        if current.snapshot_id != requested.snapshot_id:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.SNAPSHOT_MISMATCH)
        if current.split is not requested.split:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.SPLIT_MISMATCH)
        if current.access_mode is not requested.access_mode:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.ACCESS_DENIED)

    @staticmethod
    def _read_manifest(path: Path) -> MemoryNamespaceManifest:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            metadata = path.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_mode & 0o077
            ):
                raise MemoryNamespaceError(MemoryNamespaceErrorCode.PATH_UNSAFE)
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                serialized = stream.read(64 * 1024 + 1)
        except MemoryNamespaceError:
            raise
        except OSError:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.IO_ERROR) from None
        if len(serialized) > 64 * 1024:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.INVALID_NAMESPACE)
        try:
            raw = json.loads(serialized)
        except (json.JSONDecodeError, UnicodeError):
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.INVALID_NAMESPACE) from None
        if not isinstance(raw, dict):
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.INVALID_NAMESPACE)
        if raw.get("schema_version") != MEMORY_NAMESPACE_SCHEMA_VERSION:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.UNSUPPORTED_VERSION)
        try:
            return MemoryNamespaceManifest.model_validate(raw)
        except ValidationError:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.INTEGRITY_MISMATCH) from None

    @staticmethod
    def _safe_directory(path: Path, *, create: bool) -> Path:
        try:
            if create:
                path.mkdir(mode=0o700, parents=False, exist_ok=True)
            metadata = path.lstat()
        except OSError:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.IO_ERROR) from None
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.PATH_UNSAFE)
        with suppress(OSError):
            path.chmod(0o700)
        return path

    @staticmethod
    def _entries(path: Path) -> tuple[Path, ...]:
        try:
            return tuple(sorted(path.iterdir(), key=lambda item: item.name))
        except OSError:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.IO_ERROR) from None

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
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.INTEGRITY_MISMATCH) from None
        except OSError:
            raise MemoryNamespaceError(MemoryNamespaceErrorCode.IO_ERROR) from None
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def memory_namespace_sha256(namespace: MemoryNamespace) -> str:
    """Return the stable identity used in Trace assertions and snapshot inventories."""

    return _canonical_sha256(_validated_namespace(namespace))


def _validated_namespace(namespace: MemoryNamespace) -> MemoryNamespace:
    try:
        return MemoryNamespace.model_validate(namespace.model_dump(mode="python"))
    except (AttributeError, ValidationError):
        raise MemoryNamespaceError(MemoryNamespaceErrorCode.INVALID_NAMESPACE) from None


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        cast(object, to_jsonable_python(payload)),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "MEMORY_NAMESPACE_MANIFEST",
    "MEMORY_NAMESPACE_SCHEMA_VERSION",
    "FileMemoryNamespaceManager",
    "MemoryNamespaceError",
    "MemoryNamespaceManifest",
    "MemoryStoreBundle",
    "NamespacedRepositoryMemoryStore",
    "NamespacedVerifiedEpisodicMemoryStore",
    "memory_namespace_sha256",
]
