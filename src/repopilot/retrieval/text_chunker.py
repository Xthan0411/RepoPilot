"""Deterministic repository text and Agent-visible Issue chunking."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Self

from pydantic import JsonValue, model_validator

from repopilot.config import RepoPilotSettings
from repopilot.enums import SymbolType, TextChunkErrorCode, TextSourceKind
from repopilot.retrieval.chunker import PythonChunkPolicy
from repopilot.retrieval.manifest import (
    RepositoryManifest,
    RepositorySource,
    verify_repository_manifest_integrity,
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
from repopilot.tools.repository import ReadOnlyRepository, RepositoryAccessError

TEXT_CHUNKER_VERSION = "text_v1"
ISSUE_CHUNKER_VERSION = "issue_v1"
MARKDOWN_SUFFIXES = frozenset({".md"})
TEXT_SUFFIXES = frozenset({".rst", ".txt"})
CONFIG_SUFFIXES = frozenset({".cfg", ".ini", ".json", ".toml", ".yaml", ".yml"})
MARKDOWN_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
TEXT_CHUNK_ERROR_MESSAGES = {
    TextChunkErrorCode.SOURCE_UNAVAILABLE: (
        "manifest source is no longer available for text chunking"
    ),
    TextChunkErrorCode.SOURCE_CHANGED: ("manifest source content changed before text chunking"),
}


class TextChunkError(RuntimeError):
    """Safe typed fatal failure from repository text chunking."""

    def __init__(self, code: TextChunkErrorCode) -> None:
        super().__init__(TEXT_CHUNK_ERROR_MESSAGES[code])
        self.code = code


class TextSourceChunkResult(StrictModel):
    """Chunking outcome for one repository or synthetic Issue text source."""

    source_id: Identifier
    relative_path: RelativePath
    source_kind: TextSourceKind
    source_content_hash: Sha256
    empty: bool
    chunks: tuple[CodeChunk, ...]

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.empty != (not self.chunks):
            raise ValueError("empty must identify exactly the text sources with no chunks")
        chunk_order = tuple(text_chunk_sort_key(chunk) for chunk in self.chunks)
        if chunk_order != tuple(sorted(chunk_order)):
            raise ValueError("text chunks must be deterministically sorted")
        if len({chunk.chunk_id for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("text chunk IDs must be unique")
        for chunk in self.chunks:
            if chunk.relative_path != self.relative_path:
                raise ValueError("text chunk path must match its source result")
            if chunk.symbol_type is not SymbolType.TEXT:
                raise ValueError("text source chunks require text symbol type")
            if chunk.metadata.get("source_id") != self.source_id:
                raise ValueError("text chunk metadata must identify its source")
            if chunk.metadata.get("source_content_hash") != self.source_content_hash:
                raise ValueError("text chunk metadata must identify source content")
            if chunk.metadata.get("source_kind") != self.source_kind.value:
                raise ValueError("text chunk metadata kind must match its source result")
        return self


class TextChunkingResult(StrictModel):
    """Deterministic repository text chunks for one repository commit."""

    repository_id: Identifier
    commit: GitSha
    sources: tuple[TextSourceChunkResult, ...]

    @model_validator(mode="after")
    def validate_corpus(self) -> Self:
        paths = tuple(source.relative_path for source in self.sources)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("text source results must have unique sorted paths")
        chunk_ids: list[str] = []
        for source in self.sources:
            for chunk in source.chunks:
                if chunk.repository_id != self.repository_id or chunk.commit != self.commit:
                    raise ValueError("text chunk repository identity must match its corpus")
                chunk_ids.append(chunk.chunk_id)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("text corpus chunk IDs must be unique")
        return self

    @property
    def chunks(self) -> tuple[CodeChunk, ...]:
        return tuple(chunk for source in self.sources for chunk in source.chunks)


@dataclass(frozen=True)
class _TextUnit:
    start_line: int
    end_line: int
    symbol_name: str | None
    strategy: str
    metadata: dict[str, JsonValue]


class TextChunker:
    """Chunk bounded UTF-8 text without parsing or executing configuration."""

    def __init__(
        self,
        repository: ReadOnlyRepository,
        *,
        policy: PythonChunkPolicy | None = None,
    ) -> None:
        self._repository = repository
        self._policy = policy or PythonChunkPolicy()

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

    def chunk_manifest(self, manifest: RepositoryManifest) -> TextChunkingResult:
        """Chunk all supported non-Python repository text sources."""

        verify_repository_manifest_integrity(manifest)
        results = tuple(
            self._chunk_repository_source(manifest, source)
            for source in manifest.files
            if text_kind_for_path(source.relative_path) is not None
        )
        return TextChunkingResult(
            repository_id=manifest.repository_id,
            commit=manifest.commit,
            sources=results,
        )

    def chunk_source(
        self,
        manifest: RepositoryManifest,
        source: RepositorySource,
    ) -> TextSourceChunkResult:
        """Chunk one supported source after validating Manifest membership."""

        verify_repository_manifest_integrity(manifest)
        if source not in manifest.files or text_kind_for_path(source.relative_path) is None:
            raise TextChunkError(TextChunkErrorCode.SOURCE_UNAVAILABLE)
        return self._chunk_repository_source(manifest, source)

    def chunk_issue(
        self,
        manifest: RepositoryManifest,
        task: TaskSpec,
    ) -> TextSourceChunkResult:
        """Chunk only Agent-visible Issue title/body as a synthetic source."""

        verify_repository_manifest_integrity(manifest)
        if task.base_commit != manifest.commit:
            raise TextChunkError(TextChunkErrorCode.SOURCE_CHANGED)
        relative_path = issue_relative_path(task.task_id)
        source_id = issue_source_id_for(
            manifest.repository_id,
            manifest.commit,
            task.task_id,
        )
        content = issue_content(task)
        source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        lines = content.splitlines(keepends=True)
        unit = _TextUnit(
            start_line=1,
            end_line=len(lines),
            symbol_name=f"Issue {task.task_id}",
            strategy="issue",
            metadata={},
        )
        chunks = self._chunks_for_units(
            manifest,
            source_id=source_id,
            relative_path=relative_path,
            source_hash=source_hash,
            source_kind=TextSourceKind.ISSUE,
            language="issue",
            lines=lines,
            units=(unit,),
            processor=ISSUE_CHUNKER_VERSION,
        )
        return TextSourceChunkResult(
            source_id=source_id,
            relative_path=relative_path,
            source_kind=TextSourceKind.ISSUE,
            source_content_hash=source_hash,
            empty=False,
            chunks=chunks,
        )

    def _chunk_repository_source(
        self,
        manifest: RepositoryManifest,
        source: RepositorySource,
    ) -> TextSourceChunkResult:
        kind_and_language = text_kind_for_path(source.relative_path)
        if kind_and_language is None:
            raise TextChunkError(TextChunkErrorCode.SOURCE_UNAVAILABLE)
        source_kind, language = kind_and_language
        try:
            text_file = self._repository.read_text_file(source.relative_path)
        except RepositoryAccessError:
            raise TextChunkError(TextChunkErrorCode.SOURCE_UNAVAILABLE) from None
        source_hash = hashlib.sha256(text_file.content.encode("utf-8")).hexdigest()
        if text_file.size_bytes != source.size_bytes or source_hash != source.content_hash:
            raise TextChunkError(TextChunkErrorCode.SOURCE_CHANGED)
        if not text_file.content.strip():
            return TextSourceChunkResult(
                source_id=source.source_id,
                relative_path=source.relative_path,
                source_kind=source_kind,
                source_content_hash=source.content_hash,
                empty=True,
                chunks=(),
            )

        lines = text_file.content.splitlines(keepends=True)
        units = (
            _markdown_units(lines)
            if source_kind is TextSourceKind.MARKDOWN
            else _plain_text_units(lines)
        )
        chunks = self._chunks_for_units(
            manifest,
            source_id=source.source_id,
            relative_path=source.relative_path,
            source_hash=source.content_hash,
            source_kind=source_kind,
            language=language,
            lines=lines,
            units=units,
            processor=TEXT_CHUNKER_VERSION,
        )
        return TextSourceChunkResult(
            source_id=source.source_id,
            relative_path=source.relative_path,
            source_kind=source_kind,
            source_content_hash=source.content_hash,
            empty=not chunks,
            chunks=chunks,
        )

    def _chunks_for_units(
        self,
        manifest: RepositoryManifest,
        *,
        source_id: str,
        relative_path: str,
        source_hash: str,
        source_kind: TextSourceKind,
        language: str,
        lines: list[str],
        units: tuple[_TextUnit, ...],
        processor: str,
    ) -> tuple[CodeChunk, ...]:
        chunks: list[CodeChunk] = []
        for unit in units:
            windows = line_windows(
                unit.start_line,
                unit.end_line,
                max_lines=self._policy.max_lines,
                overlap_lines=self._policy.overlap_lines,
            )
            symbol_name, symbol_metadata = _bounded_symbol_name(unit.symbol_name)
            for part_index, (start_line, end_line) in enumerate(windows):
                content = "".join(lines[start_line - 1 : end_line])
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                metadata: dict[str, JsonValue] = {
                    "source_id": source_id,
                    "source_content_hash": source_hash,
                    "source_kind": source_kind.value,
                    "processor": processor,
                    "strategy": unit.strategy,
                    "part_index": part_index,
                    "part_count": len(windows),
                    "max_lines": self._policy.max_lines,
                    "overlap_lines": self._policy.overlap_lines,
                    **unit.metadata,
                    **symbol_metadata,
                }
                chunks.append(
                    CodeChunk(
                        chunk_id=text_chunk_id_for(
                            source_id=source_id,
                            source_kind=source_kind,
                            strategy=unit.strategy,
                            symbol_name=symbol_name,
                            start_line=start_line,
                            end_line=end_line,
                            part_index=part_index,
                            content_hash=content_hash,
                        ),
                        repository_id=manifest.repository_id,
                        commit=manifest.commit,
                        relative_path=relative_path,
                        symbol_name=symbol_name,
                        symbol_type=SymbolType.TEXT,
                        start_line=start_line,
                        end_line=end_line,
                        language=language,
                        content=content,
                        content_hash=content_hash,
                        metadata=metadata,
                    )
                )
        return tuple(sorted(chunks, key=text_chunk_sort_key))


def text_kind_for_path(path: str) -> tuple[TextSourceKind, str] | None:
    """Classify only the exact text suffixes owned by P2-S3."""

    suffix = PurePosixPath(path).suffix
    if suffix in MARKDOWN_SUFFIXES:
        return TextSourceKind.MARKDOWN, "markdown"
    if suffix in TEXT_SUFFIXES:
        return TextSourceKind.TEXT, "text"
    if suffix in CONFIG_SUFFIXES:
        language = suffix[1:]
        return TextSourceKind.CONFIG, language
    return None


def issue_relative_path(task_id: str) -> str:
    return f"_repopilot_issues/{task_id}.md"


def issue_source_id_for(repository_id: str, commit: str, task_id: str) -> str:
    canonical = json.dumps(
        {
            "commit": commit,
            "kind": "issue",
            "repository_id": repository_id,
            "task_id": task_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"issue_{hashlib.sha256(canonical).hexdigest()}"


def issue_content(task: TaskSpec) -> str:
    """Render only Agent-visible TaskSpec fields into deterministic text."""

    return f"# Issue\n\nTitle:\n{task.issue_title}\n\nDescription:\n{task.issue_body}\n"


def text_chunk_id_for(
    *,
    source_id: str,
    source_kind: TextSourceKind,
    strategy: str,
    symbol_name: str | None,
    start_line: int,
    end_line: int,
    part_index: int,
    content_hash: str,
) -> str:
    canonical = json.dumps(
        {
            "content_hash": content_hash,
            "end_line": end_line,
            "part_index": part_index,
            "source_id": source_id,
            "source_kind": source_kind.value,
            "start_line": start_line,
            "strategy": strategy,
            "symbol_name": symbol_name,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"chunk_{hashlib.sha256(canonical).hexdigest()}"


def text_chunk_sort_key(chunk: CodeChunk) -> tuple[int, int, str, int, str]:
    return (
        chunk.start_line,
        chunk.end_line,
        chunk.symbol_name or "",
        _part_index(chunk),
        chunk.chunk_id,
    )


def line_windows(
    start_line: int,
    end_line: int,
    *,
    max_lines: int,
    overlap_lines: int,
) -> tuple[tuple[int, int], ...]:
    windows: list[tuple[int, int]] = []
    cursor = start_line
    while cursor <= end_line:
        window_end = min(end_line, cursor + max_lines - 1)
        windows.append((cursor, window_end))
        if window_end == end_line:
            break
        cursor = window_end - overlap_lines + 1
    return tuple(windows)


def _markdown_units(lines: list[str]) -> tuple[_TextUnit, ...]:
    headings: list[tuple[int, int, str, str]] = []
    stack: list[tuple[int, str]] = []
    fence_marker: str | None = None
    fence_length = 0
    for line_number, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        fence_match = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence_match is not None:
            marker = fence_match.group(1)
            if fence_marker is None:
                fence_marker = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_marker and len(marker) >= fence_length:
                fence_marker = None
                fence_length = 0
            continue
        if fence_marker is not None:
            continue
        match = MARKDOWN_HEADING.match(line.rstrip("\r\n"))
        if match is None:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        if not title:
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        qualified = " > ".join(item[1] for item in stack)
        headings.append((line_number, level, title, qualified))

    if not headings:
        unit = _trimmed_unit(lines, 1, len(lines), strategy="line_window")
        return () if unit is None else (unit,)

    units: list[_TextUnit] = []
    preamble = _trimmed_unit(
        lines,
        1,
        headings[0][0] - 1,
        strategy="markdown_preamble",
    )
    if preamble is not None:
        units.append(preamble)
    for index, (line_number, level, title, qualified) in enumerate(headings):
        end_line = headings[index + 1][0] - 1 if index + 1 < len(headings) else len(lines)
        while end_line >= line_number and not lines[end_line - 1].strip():
            end_line -= 1
        units.append(
            _TextUnit(
                start_line=line_number,
                end_line=end_line,
                symbol_name=qualified,
                strategy="markdown_heading",
                metadata={
                    "heading_level": level,
                    "heading_title": title,
                },
            )
        )
    return tuple(units)


def _plain_text_units(lines: list[str]) -> tuple[_TextUnit, ...]:
    unit = _trimmed_unit(lines, 1, len(lines), strategy="line_window")
    return () if unit is None else (unit,)


def _trimmed_unit(
    lines: list[str],
    start_line: int,
    end_line: int,
    *,
    strategy: str,
) -> _TextUnit | None:
    while start_line <= end_line and not lines[start_line - 1].strip():
        start_line += 1
    while end_line >= start_line and not lines[end_line - 1].strip():
        end_line -= 1
    if start_line > end_line:
        return None
    return _TextUnit(
        start_line=start_line,
        end_line=end_line,
        symbol_name=None,
        strategy=strategy,
        metadata={},
    )


def _bounded_symbol_name(value: str | None) -> tuple[str | None, dict[str, JsonValue]]:
    if value is None or len(value) <= 512:
        return value, {}
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"{value[:447]}.{digest}", {
        "symbol_name_truncated": True,
        "symbol_name_sha256": digest,
    }


def _part_index(chunk: CodeChunk) -> int:
    value = chunk.metadata.get("part_index", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
