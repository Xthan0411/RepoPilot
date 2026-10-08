"""Combined repository chunk corpus, incremental reuse, and versioned JSONL I/O."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, JsonValue, ValidationError, model_validator

from repopilot.config import RepoPilotSettings
from repopilot.enums import ChunkCorpusErrorCode
from repopilot.retrieval.chunker import (
    PYTHON_CHUNKER_VERSION,
    PythonAstChunker,
    PythonChunkPolicy,
    PythonSourceChunkResult,
)
from repopilot.retrieval.manifest import (
    RepositoryManifest,
    RepositorySource,
    verify_repository_manifest_integrity,
)
from repopilot.retrieval.text_chunker import (
    ISSUE_CHUNKER_VERSION,
    TEXT_CHUNKER_VERSION,
    TextChunker,
    TextSourceChunkResult,
    issue_content,
    issue_relative_path,
    issue_source_id_for,
    text_kind_for_path,
)
from repopilot.schemas import (
    CodeChunk,
    GitSha,
    Identifier,
    RelativePath,
    Sha256,
    StrictModel,
    TaskSpec,
)
from repopilot.tools.repository import ReadOnlyRepository

CHUNK_CORPUS_SCHEMA_VERSION: Literal[1] = 1
CORPUS_ERROR_MESSAGES = {
    ChunkCorpusErrorCode.IO_ERROR: "chunk corpus artifact could not be read or written",
    ChunkCorpusErrorCode.INVALID_DOCUMENT: "chunk corpus document is invalid",
    ChunkCorpusErrorCode.UNSUPPORTED_VERSION: "chunk corpus schema version is unsupported",
    ChunkCorpusErrorCode.MIXED_CORPUS: (
        "chunk corpus contains inconsistent repository or commit identity"
    ),
    ChunkCorpusErrorCode.DUPLICATE_CHUNK: "chunk corpus contains duplicate chunk IDs",
    ChunkCorpusErrorCode.UNSORTED_CORPUS: "chunk corpus records are not deterministically sorted",
}


class ChunkCorpusError(RuntimeError):
    """Safe typed failure from corpus construction or JSONL artifact I/O."""

    def __init__(self, code: ChunkCorpusErrorCode) -> None:
        super().__init__(CORPUS_ERROR_MESSAGES[code])
        self.code = code


class ChunkSourceSnapshot(StrictModel):
    """Header metadata required to reuse one source without re-chunking."""

    source_id: Identifier
    relative_path: RelativePath
    source_content_hash: Sha256
    processor: Identifier
    build_fingerprint: Sha256
    chunk_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def validate_chunk_ids(self) -> Self:
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("source snapshot chunk IDs must be unique")
        return self


class ChunkCorpus(StrictModel):
    """One repository+commit-isolated, deterministically ordered chunk corpus."""

    repository_id: Identifier
    commit: GitSha
    sources: tuple[ChunkSourceSnapshot, ...]
    chunks: tuple[CodeChunk, ...]

    @model_validator(mode="after")
    def validate_corpus(self) -> Self:
        source_paths = tuple(source.relative_path for source in self.sources)
        if source_paths != tuple(sorted(source_paths)) or len(set(source_paths)) != len(
            source_paths
        ):
            raise ValueError("corpus sources must have unique sorted paths")
        source_ids = tuple(source.source_id for source in self.sources)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("corpus source IDs must be unique")

        chunk_order = tuple(chunk_corpus_sort_key(chunk) for chunk in self.chunks)
        if chunk_order != tuple(sorted(chunk_order)):
            raise ValueError("corpus chunks must be deterministically sorted")
        chunk_ids = tuple(chunk.chunk_id for chunk in self.chunks)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("corpus chunk IDs must be unique")

        chunks_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        claimed_ids: list[str] = []
        for source in self.sources:
            claimed_ids.extend(source.chunk_ids)
            source_chunks = tuple(chunks_by_id.get(chunk_id) for chunk_id in source.chunk_ids)
            if any(chunk is None for chunk in source_chunks):
                raise ValueError("source snapshot references an unknown chunk")
            for chunk in source_chunks:
                assert chunk is not None
                if (
                    chunk.repository_id != self.repository_id
                    or chunk.commit != self.commit
                    or chunk.relative_path != source.relative_path
                    or chunk.metadata.get("source_id") != source.source_id
                    or chunk.metadata.get("source_content_hash") != source.source_content_hash
                    or chunk.metadata.get("processor") != source.processor
                ):
                    raise ValueError("source snapshot does not match its chunks")
            expected_ids = tuple(
                chunk.chunk_id
                for chunk in self.chunks
                if chunk.metadata.get("source_id") == source.source_id
            )
            if source.chunk_ids != expected_ids:
                raise ValueError("source snapshot chunk IDs must follow corpus order")
        if len(set(claimed_ids)) != len(claimed_ids) or set(claimed_ids) != set(chunk_ids):
            raise ValueError("corpus chunks must belong to exactly one source snapshot")
        return self


class ChunkCorpusHeaderRecord(StrictModel):
    """First JSONL record retaining identity and empty-source snapshots."""

    schema_version: Literal[1] = CHUNK_CORPUS_SCHEMA_VERSION
    record_type: Literal["header"] = "header"
    repository_id: Identifier
    commit: GitSha
    sources: tuple[ChunkSourceSnapshot, ...]


class ChunkCorpusChunkRecord(StrictModel):
    """One versioned JSONL chunk record."""

    schema_version: Literal[1] = CHUNK_CORPUS_SCHEMA_VERSION
    record_type: Literal["chunk"] = "chunk"
    chunk: CodeChunk


class ChunkCorpusBuildStats(StrictModel):
    """Observable incremental behavior for one corpus build."""

    reused_sources: int = Field(ge=0)
    rebuilt_sources: int = Field(ge=0)
    removed_sources: int = Field(ge=0)


class ChunkCorpusBuildResult(StrictModel):
    corpus: ChunkCorpus
    stats: ChunkCorpusBuildStats


class RepositoryChunkCorpusBuilder:
    """Combine Python, text, and optional Issue chunks with exact reuse gates."""

    def __init__(
        self,
        repository: ReadOnlyRepository,
        *,
        policy: PythonChunkPolicy | None = None,
    ) -> None:
        self._policy = policy or PythonChunkPolicy()
        self._python = PythonAstChunker(repository, policy=self._policy)
        self._text = TextChunker(repository, policy=self._policy)

    @classmethod
    def from_settings(
        cls,
        repository: ReadOnlyRepository,
        settings: RepoPilotSettings,
    ) -> Self:
        return cls(repository, policy=PythonChunkPolicy.from_settings(settings))

    @property
    def policy(self) -> PythonChunkPolicy:
        return self._policy

    def build(
        self,
        manifest: RepositoryManifest,
        *,
        task: TaskSpec | None = None,
        previous: ChunkCorpus | None = None,
    ) -> ChunkCorpusBuildResult:
        """Build a combined corpus, reusing only exact source/config snapshots."""

        verify_repository_manifest_integrity(manifest)
        if previous is not None and (
            previous.repository_id != manifest.repository_id or previous.commit != manifest.commit
        ):
            raise ChunkCorpusError(ChunkCorpusErrorCode.MIXED_CORPUS)
        if task is not None and task.base_commit != manifest.commit:
            raise ChunkCorpusError(ChunkCorpusErrorCode.MIXED_CORPUS)

        previous_snapshots = (
            {source.source_id: source for source in previous.sources}
            if previous is not None
            else {}
        )
        previous_chunks = (
            {chunk.chunk_id: chunk for chunk in previous.chunks} if previous is not None else {}
        )
        candidates = self._repository_candidates(manifest)
        if task is not None:
            candidates.append(self._issue_candidate(manifest, task))
        candidates.sort(key=lambda candidate: candidate.relative_path)
        candidate_paths = tuple(candidate.relative_path for candidate in candidates)
        if len(set(candidate_paths)) != len(candidate_paths):
            raise ChunkCorpusError(ChunkCorpusErrorCode.MIXED_CORPUS)

        snapshots: list[ChunkSourceSnapshot] = []
        chunks: list[CodeChunk] = []
        reused_sources = 0
        rebuilt_sources = 0
        current_source_ids: set[str] = set()
        for candidate in candidates:
            current_source_ids.add(candidate.source_id)
            previous_snapshot = previous_snapshots.get(candidate.source_id)
            if (
                previous_snapshot is not None
                and previous_snapshot.relative_path == candidate.relative_path
                and previous_snapshot.source_content_hash == candidate.source_content_hash
                and previous_snapshot.processor == candidate.processor
                and previous_snapshot.build_fingerprint == candidate.build_fingerprint
            ):
                reused = tuple(
                    previous_chunks[chunk_id] for chunk_id in previous_snapshot.chunk_ids
                )
                snapshots.append(previous_snapshot)
                chunks.extend(reused)
                reused_sources += 1
                continue

            rebuilt = candidate.build()
            snapshot = ChunkSourceSnapshot(
                source_id=candidate.source_id,
                relative_path=candidate.relative_path,
                source_content_hash=candidate.source_content_hash,
                processor=candidate.processor,
                build_fingerprint=candidate.build_fingerprint,
                chunk_ids=tuple(chunk.chunk_id for chunk in rebuilt),
            )
            snapshots.append(snapshot)
            chunks.extend(rebuilt)
            rebuilt_sources += 1

        removed_sources = len(set(previous_snapshots).difference(current_source_ids))
        corpus = ChunkCorpus(
            repository_id=manifest.repository_id,
            commit=manifest.commit,
            sources=tuple(sorted(snapshots, key=lambda source: source.relative_path)),
            chunks=tuple(sorted(chunks, key=chunk_corpus_sort_key)),
        )
        return ChunkCorpusBuildResult(
            corpus=corpus,
            stats=ChunkCorpusBuildStats(
                reused_sources=reused_sources,
                rebuilt_sources=rebuilt_sources,
                removed_sources=removed_sources,
            ),
        )

    def _repository_candidates(
        self,
        manifest: RepositoryManifest,
    ) -> list[_SourceCandidate]:
        candidates: list[_SourceCandidate] = []
        for source in manifest.files:
            if source.relative_path.endswith(".py"):
                processor = PYTHON_CHUNKER_VERSION
                fingerprint = processor_fingerprint(
                    processor,
                    self._policy,
                    extra={},
                )
                candidates.append(
                    _SourceCandidate(
                        source_id=source.source_id,
                        relative_path=source.relative_path,
                        source_content_hash=source.content_hash,
                        processor=processor,
                        build_fingerprint=fingerprint,
                        build=_source_builder(
                            self._python.chunk_source,
                            manifest,
                            source,
                        ),
                    )
                )
                continue
            kind_and_language = text_kind_for_path(source.relative_path)
            if kind_and_language is None:
                continue
            source_kind, _ = kind_and_language
            processor = TEXT_CHUNKER_VERSION
            fingerprint = processor_fingerprint(
                processor,
                self._policy,
                extra={"source_kind": source_kind.value},
            )
            candidates.append(
                _SourceCandidate(
                    source_id=source.source_id,
                    relative_path=source.relative_path,
                    source_content_hash=source.content_hash,
                    processor=processor,
                    build_fingerprint=fingerprint,
                    build=_source_builder(
                        self._text.chunk_source,
                        manifest,
                        source,
                    ),
                )
            )
        return candidates

    def _issue_candidate(
        self,
        manifest: RepositoryManifest,
        task: TaskSpec,
    ) -> _SourceCandidate:
        content = issue_content(task)
        source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        processor = ISSUE_CHUNKER_VERSION
        return _SourceCandidate(
            source_id=issue_source_id_for(
                manifest.repository_id,
                manifest.commit,
                task.task_id,
            ),
            relative_path=issue_relative_path(task.task_id),
            source_content_hash=source_hash,
            processor=processor,
            build_fingerprint=processor_fingerprint(
                processor,
                self._policy,
                extra={},
            ),
            build=lambda: self._text.chunk_issue(manifest, task).chunks,
        )


class _SourceCandidate:
    """Internal delayed source build used to prove unchanged sources are not read."""

    def __init__(
        self,
        *,
        source_id: str,
        relative_path: str,
        source_content_hash: str,
        processor: str,
        build_fingerprint: str,
        build: Callable[[], tuple[CodeChunk, ...]],
    ) -> None:
        self.source_id = source_id
        self.relative_path = relative_path
        self.source_content_hash = source_content_hash
        self.processor = processor
        self.build_fingerprint = build_fingerprint
        self.build = build


def _source_builder(
    chunk_source: Callable[
        [RepositoryManifest, RepositorySource],
        PythonSourceChunkResult | TextSourceChunkResult,
    ],
    manifest: RepositoryManifest,
    source: RepositorySource,
) -> Callable[[], tuple[CodeChunk, ...]]:
    def build() -> tuple[CodeChunk, ...]:
        return chunk_source(manifest, source).chunks

    return build


def processor_fingerprint(
    processor: str,
    policy: PythonChunkPolicy,
    *,
    extra: dict[str, JsonValue],
) -> str:
    encoded = json.dumps(
        {
            "extra": extra,
            "max_lines": policy.max_lines,
            "overlap_lines": policy.overlap_lines,
            "processor": processor,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def chunk_corpus_sort_key(
    chunk: CodeChunk,
) -> tuple[str, int, int, str, str, int, str]:
    return (
        chunk.relative_path,
        chunk.start_line,
        chunk.end_line,
        chunk.symbol_type.value,
        chunk.symbol_name or "",
        _part_index(chunk),
        chunk.chunk_id,
    )


def _part_index(chunk: CodeChunk) -> int:
    value = chunk.metadata.get("part_index", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def write_chunk_corpus(path: Path, corpus: ChunkCorpus) -> None:
    """Atomically write a deterministic header-plus-chunks JSONL artifact."""

    header = ChunkCorpusHeaderRecord(
        repository_id=corpus.repository_id,
        commit=corpus.commit,
        sources=corpus.sources,
    )
    records = [header.model_dump_json()]
    records.extend(ChunkCorpusChunkRecord(chunk=chunk).model_dump_json() for chunk in corpus.chunks)
    serialized = "\n".join(records) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(path)
    except OSError:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise ChunkCorpusError(ChunkCorpusErrorCode.IO_ERROR) from None


def read_chunk_corpus(path: Path) -> ChunkCorpus:
    """Read and validate one supported deterministic chunk JSONL artifact."""

    try:
        data = path.read_bytes()
    except OSError:
        raise ChunkCorpusError(ChunkCorpusErrorCode.IO_ERROR) from None
    try:
        serialized = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ChunkCorpusError(ChunkCorpusErrorCode.INVALID_DOCUMENT) from None
    lines = serialized.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ChunkCorpusError(ChunkCorpusErrorCode.INVALID_DOCUMENT)

    raw_records: list[dict[str, object]] = []
    for line in lines:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            raise ChunkCorpusError(ChunkCorpusErrorCode.INVALID_DOCUMENT) from None
        if not isinstance(raw, dict):
            raise ChunkCorpusError(ChunkCorpusErrorCode.INVALID_DOCUMENT)
        if "schema_version" not in raw:
            raise ChunkCorpusError(ChunkCorpusErrorCode.INVALID_DOCUMENT)
        if raw["schema_version"] != CHUNK_CORPUS_SCHEMA_VERSION:
            raise ChunkCorpusError(ChunkCorpusErrorCode.UNSUPPORTED_VERSION)
        raw_records.append(raw)

    try:
        header = ChunkCorpusHeaderRecord.model_validate(raw_records[0])
    except ValidationError:
        raise ChunkCorpusError(ChunkCorpusErrorCode.INVALID_DOCUMENT) from None
    chunks: list[CodeChunk] = []
    for raw in raw_records[1:]:
        try:
            record = ChunkCorpusChunkRecord.model_validate(raw)
        except ValidationError:
            raise ChunkCorpusError(ChunkCorpusErrorCode.INVALID_DOCUMENT) from None
        chunks.append(record.chunk)

    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ChunkCorpusError(ChunkCorpusErrorCode.DUPLICATE_CHUNK)
    if any(
        chunk.repository_id != header.repository_id or chunk.commit != header.commit
        for chunk in chunks
    ):
        raise ChunkCorpusError(ChunkCorpusErrorCode.MIXED_CORPUS)
    order = tuple(chunk_corpus_sort_key(chunk) for chunk in chunks)
    if order != tuple(sorted(order)):
        raise ChunkCorpusError(ChunkCorpusErrorCode.UNSORTED_CORPUS)
    source_paths = tuple(source.relative_path for source in header.sources)
    if source_paths != tuple(sorted(source_paths)):
        raise ChunkCorpusError(ChunkCorpusErrorCode.UNSORTED_CORPUS)
    try:
        return ChunkCorpus(
            repository_id=header.repository_id,
            commit=header.commit,
            sources=header.sources,
            chunks=tuple(chunks),
        )
    except ValidationError:
        raise ChunkCorpusError(ChunkCorpusErrorCode.INVALID_DOCUMENT) from None
