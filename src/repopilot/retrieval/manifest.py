"""Versioned, deterministic repository-manifest contracts and artifact I/O."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_core import to_jsonable_python

from repopilot.enums import (
    RepositoryEntryType,
    RepositoryErrorCode,
    RepositoryIngestErrorCode,
)
from repopilot.schemas import GitSha, Identifier, RelativePath, Sha256, StrictModel
from repopilot.tools.repository import DEFAULT_DENIED_NAMES, RepositoryAccessPolicy

REPOSITORY_MANIFEST_SCHEMA_VERSION: Literal[1] = 1
DEFAULT_IGNORED_NAMES = tuple(
    sorted(
        {
            *DEFAULT_DENIED_NAMES,
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".tox",
            ".venv",
            "__pycache__",
        }
    )
)
SKIPPABLE_FILE_ERRORS = frozenset(
    {
        RepositoryErrorCode.BINARY_FILE,
        RepositoryErrorCode.DECODE_ERROR,
        RepositoryErrorCode.FILE_TOO_LARGE,
    }
)
INGEST_ERROR_MESSAGES = {
    RepositoryIngestErrorCode.LISTING_TRUNCATED: (
        "repository listing exceeded the configured entry limit"
    ),
    RepositoryIngestErrorCode.SOURCE_CHANGED: (
        "repository source changed while the manifest was being built"
    ),
    RepositoryIngestErrorCode.IO_ERROR: (
        "repository manifest artifact could not be read or written"
    ),
    RepositoryIngestErrorCode.INVALID_DOCUMENT: "repository manifest document is invalid",
    RepositoryIngestErrorCode.UNSUPPORTED_VERSION: (
        "repository manifest schema version is unsupported"
    ),
    RepositoryIngestErrorCode.INTEGRITY_MISMATCH: (
        "repository manifest content digest does not match"
    ),
}


class RepositoryIngestError(RuntimeError):
    """Safe typed failure from ingestion or repository-manifest artifact I/O."""

    def __init__(self, code: RepositoryIngestErrorCode) -> None:
        super().__init__(INGEST_ERROR_MESSAGES[code])
        self.code = code


def _validate_simple_names(value: tuple[str, ...]) -> tuple[str, ...]:
    if len(set(value)) != len(value):
        raise ValueError("ignored_names must be unique")
    for name in value:
        if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
            raise ValueError("ignored_names must contain simple path components")
    return tuple(sorted(value))


class RepositoryIgnorePolicy(StrictModel):
    """Versioned exact-name ignore policy used before repository traversal."""

    policy_version: Literal[1] = 1
    ignored_names: tuple[str, ...] = DEFAULT_IGNORED_NAMES

    @field_validator("ignored_names")
    @classmethod
    def validate_ignored_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_simple_names(value)

    @model_validator(mode="after")
    def preserve_security_denials(self) -> Self:
        if not set(DEFAULT_DENIED_NAMES).issubset(self.ignored_names):
            raise ValueError("ignored_names must include security-denied repository names")
        return self

    def merge_repository_policy(
        self,
        repository_policy: RepositoryAccessPolicy,
    ) -> Self:
        """Return the effective policy, retaining any caller-defined denied names."""

        names = tuple(sorted({*self.ignored_names, *repository_policy.denied_names}))
        return self.model_copy(update={"ignored_names": names})


class RepositorySource(StrictModel):
    """One complete UTF-8 source included in the retrieval corpus."""

    source_id: Identifier
    relative_path: RelativePath
    size_bytes: int = Field(ge=0)
    content_hash: Sha256


class SkippedRepositorySource(StrictModel):
    """One visible repository entry intentionally excluded from the corpus."""

    relative_path: RelativePath
    entry_type: RepositoryEntryType
    reason: RepositoryErrorCode
    size_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_reason(self) -> Self:
        if self.entry_type is RepositoryEntryType.SYMLINK:
            if self.reason is not RepositoryErrorCode.SYMLINK_FORBIDDEN:
                raise ValueError("symlink skips require symlink_forbidden")
            if self.size_bytes is not None:
                raise ValueError("symlink skips cannot contain size_bytes")
            return self
        if self.entry_type is not RepositoryEntryType.FILE:
            raise ValueError("only files and symlinks can be skipped sources")
        if self.reason not in SKIPPABLE_FILE_ERRORS:
            raise ValueError("file skip reason is not supported")
        if self.size_bytes is None:
            raise ValueError("skipped files require size_bytes")
        return self


class RepositoryManifest(StrictModel):
    """Integrity-protected source inventory for one repository and commit."""

    schema_version: Literal[1] = REPOSITORY_MANIFEST_SCHEMA_VERSION
    repository_id: Identifier
    commit: GitSha
    ignore_policy: RepositoryIgnorePolicy
    files: tuple[RepositorySource, ...]
    skipped_sources: tuple[SkippedRepositorySource, ...] = ()
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        file_paths = tuple(source.relative_path for source in self.files)
        skipped_paths = tuple(source.relative_path for source in self.skipped_sources)
        if file_paths != tuple(sorted(file_paths)) or len(set(file_paths)) != len(file_paths):
            raise ValueError("manifest files must have unique sorted paths")
        if skipped_paths != tuple(sorted(skipped_paths)) or len(set(skipped_paths)) != len(
            skipped_paths
        ):
            raise ValueError("skipped sources must have unique sorted paths")
        if set(file_paths).intersection(skipped_paths):
            raise ValueError("included and skipped source paths must not overlap")
        source_ids = tuple(source.source_id for source in self.files)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("manifest source IDs must be unique")
        for source in self.files:
            expected = source_id_for(
                self.repository_id,
                self.commit,
                source.relative_path,
            )
            if source.source_id != expected:
                raise ValueError("manifest source ID does not match repository identity")
        return self

    @classmethod
    def build(
        cls,
        *,
        repository_id: str,
        commit: str,
        ignore_policy: RepositoryIgnorePolicy,
        files: tuple[RepositorySource, ...],
        skipped_sources: tuple[SkippedRepositorySource, ...] = (),
    ) -> Self:
        payload: dict[str, object] = {
            "schema_version": REPOSITORY_MANIFEST_SCHEMA_VERSION,
            "repository_id": repository_id,
            "commit": commit,
            "ignore_policy": ignore_policy,
            "files": files,
            "skipped_sources": skipped_sources,
        }
        return cls(
            schema_version=REPOSITORY_MANIFEST_SCHEMA_VERSION,
            repository_id=repository_id,
            commit=commit,
            ignore_policy=ignore_policy,
            files=files,
            skipped_sources=skipped_sources,
            content_sha256=repository_manifest_content_sha256(payload),
        )


def source_id_for(repository_id: str, commit: str, relative_path: str) -> str:
    """Return a path identity pinned to one repository commit."""

    canonical = json.dumps(
        {
            "commit": commit,
            "relative_path": relative_path,
            "repository_id": repository_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"src_{hashlib.sha256(canonical).hexdigest()}"


def repository_manifest_content_sha256(payload: dict[str, object]) -> str:
    """Hash canonical JSON content excluding the digest field itself."""

    encoded = json.dumps(
        to_jsonable_python(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_repository_manifest_integrity(manifest: RepositoryManifest) -> None:
    """Reject manifest content that no longer matches its digest."""

    payload = manifest.model_dump(mode="python", exclude={"content_sha256"})
    if repository_manifest_content_sha256(payload) != manifest.content_sha256:
        raise RepositoryIngestError(RepositoryIngestErrorCode.INTEGRITY_MISMATCH)


def write_repository_manifest(path: Path, manifest: RepositoryManifest) -> None:
    """Atomically write one deterministic repository Manifest."""

    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
        temporary.replace(path)
    except OSError:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise RepositoryIngestError(RepositoryIngestErrorCode.IO_ERROR) from None


def read_repository_manifest(path: Path) -> RepositoryManifest:
    """Read, validate, and integrity-check one supported Manifest."""

    try:
        serialized = path.read_text(encoding="utf-8")
    except UnicodeError:
        raise RepositoryIngestError(RepositoryIngestErrorCode.INVALID_DOCUMENT) from None
    except OSError:
        raise RepositoryIngestError(RepositoryIngestErrorCode.IO_ERROR) from None
    try:
        raw = json.loads(serialized)
    except json.JSONDecodeError:
        raise RepositoryIngestError(RepositoryIngestErrorCode.INVALID_DOCUMENT) from None
    if not isinstance(raw, dict):
        raise RepositoryIngestError(RepositoryIngestErrorCode.INVALID_DOCUMENT)
    if raw.get("schema_version") != REPOSITORY_MANIFEST_SCHEMA_VERSION:
        raise RepositoryIngestError(RepositoryIngestErrorCode.UNSUPPORTED_VERSION)
    try:
        manifest = RepositoryManifest.model_validate(raw)
    except ValidationError:
        raise RepositoryIngestError(RepositoryIngestErrorCode.INVALID_DOCUMENT) from None
    verify_repository_manifest_integrity(manifest)
    return manifest
