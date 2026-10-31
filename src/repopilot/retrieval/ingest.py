"""Deterministic, contained construction of repository source manifests."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field

from repopilot.enums import (
    RepositoryEntryType,
    RepositoryErrorCode,
    RepositoryIngestErrorCode,
)
from repopilot.retrieval.manifest import (
    RepositoryIgnorePolicy,
    RepositoryIngestError,
    RepositoryManifest,
    RepositorySource,
    SkippedRepositorySource,
    source_id_for,
)
from repopilot.schemas import GitSha, Identifier, StrictModel
from repopilot.tools.repository import (
    ReadOnlyRepository,
    RepositoryAccessError,
    RepositoryAccessPolicy,
)

MUTATION_ERRORS = frozenset(
    {
        RepositoryErrorCode.NOT_FOUND,
        RepositoryErrorCode.NOT_FILE,
        RepositoryErrorCode.SYMLINK_FORBIDDEN,
        RepositoryErrorCode.FILE_TOO_LARGE,
    }
)


class RepositoryIngestScanPolicy(StrictModel):
    """Independent, bounded inventory scan; never expands Agent-facing tool output.

    The default is a safety ceiling, not evidence that a particular repository fits.
    A successful scan is complete regardless of its ceiling, so the ceiling is not
    added to the source manifest or its existing content digest. Runtime qualification
    can record this policy separately without rewriting historical manifests.
    """

    policy_version: Literal[1] = 1
    max_list_entries: int = Field(default=20_000, gt=0, le=20_000, strict=True)


class RepositoryIngestor:
    """Build a stable source inventory without executing repository code."""

    def __init__(
        self,
        repository: ReadOnlyRepository,
        *,
        ignore_policy: RepositoryIgnorePolicy | None = None,
        scan_policy: RepositoryIngestScanPolicy | None = None,
    ) -> None:
        self._scan_policy = scan_policy or RepositoryIngestScanPolicy()
        requested_policy = ignore_policy or RepositoryIgnorePolicy()
        self._ignore_policy = requested_policy.merge_repository_policy(repository.policy)
        source_policy = RepositoryAccessPolicy(
            max_file_bytes=repository.policy.max_file_bytes,
            max_read_bytes=repository.policy.max_read_bytes,
            max_read_lines=repository.policy.max_read_lines,
            max_list_entries=self._scan_policy.max_list_entries,
            denied_names=self._ignore_policy.ignored_names,
        )
        self._repository = ReadOnlyRepository(repository.root, policy=source_policy)

    @property
    def ignore_policy(self) -> RepositoryIgnorePolicy:
        return self._ignore_policy

    @property
    def scan_policy(self) -> RepositoryIngestScanPolicy:
        """Return the effective internal ceiling, separate from the caller's tool policy."""

        return self._scan_policy

    def build(self, *, repository_id: Identifier, commit: GitSha) -> RepositoryManifest:
        """Scan one caller-pinned checkout and return its deterministic inventory."""

        listing = self._repository.list_directory()
        if listing.truncated:
            raise RepositoryIngestError(RepositoryIngestErrorCode.LISTING_TRUNCATED)

        files: list[RepositorySource] = []
        skipped: list[SkippedRepositorySource] = []
        for entry in listing.entries:
            if entry.entry_type is RepositoryEntryType.DIRECTORY:
                continue
            if entry.entry_type is RepositoryEntryType.SYMLINK:
                skipped.append(
                    SkippedRepositorySource(
                        relative_path=entry.relative_path,
                        entry_type=entry.entry_type,
                        reason=RepositoryErrorCode.SYMLINK_FORBIDDEN,
                    )
                )
                continue
            if entry.size_bytes is None:
                raise RepositoryIngestError(RepositoryIngestErrorCode.SOURCE_CHANGED)
            if entry.size_bytes > self._repository.policy.max_file_bytes:
                skipped.append(
                    SkippedRepositorySource(
                        relative_path=entry.relative_path,
                        entry_type=entry.entry_type,
                        reason=RepositoryErrorCode.FILE_TOO_LARGE,
                        size_bytes=entry.size_bytes,
                    )
                )
                continue
            try:
                source = self._repository.read_text_file(entry.relative_path)
            except RepositoryAccessError as error:
                if error.code in {
                    RepositoryErrorCode.BINARY_FILE,
                    RepositoryErrorCode.DECODE_ERROR,
                }:
                    skipped.append(
                        SkippedRepositorySource(
                            relative_path=entry.relative_path,
                            entry_type=entry.entry_type,
                            reason=error.code,
                            size_bytes=entry.size_bytes,
                        )
                    )
                    continue
                if error.code in MUTATION_ERRORS:
                    raise RepositoryIngestError(RepositoryIngestErrorCode.SOURCE_CHANGED) from None
                raise
            if source.size_bytes != entry.size_bytes:
                raise RepositoryIngestError(RepositoryIngestErrorCode.SOURCE_CHANGED)
            content_hash = hashlib.sha256(source.content.encode("utf-8")).hexdigest()
            files.append(
                RepositorySource(
                    source_id=source_id_for(
                        repository_id,
                        commit,
                        entry.relative_path,
                    ),
                    relative_path=entry.relative_path,
                    size_bytes=source.size_bytes,
                    content_hash=content_hash,
                )
            )

        return RepositoryManifest.build(
            repository_id=repository_id,
            commit=commit,
            ignore_policy=self._ignore_policy,
            files=tuple(sorted(files, key=lambda source: source.relative_path)),
            skipped_sources=tuple(sorted(skipped, key=lambda source: source.relative_path)),
        )


def ingest_repository(
    repository: ReadOnlyRepository,
    *,
    repository_id: Identifier,
    commit: GitSha,
    ignore_policy: RepositoryIgnorePolicy | None = None,
    scan_policy: RepositoryIngestScanPolicy | None = None,
) -> RepositoryManifest:
    """Convenience entry point for one deterministic repository scan."""

    return RepositoryIngestor(
        repository, ignore_policy=ignore_policy, scan_policy=scan_policy
    ).build(
        repository_id=repository_id,
        commit=commit,
    )
