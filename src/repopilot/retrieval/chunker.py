"""Deterministic Python AST-aware chunk construction."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from typing import Self

from pydantic import Field, JsonValue, field_validator, model_validator

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    PythonChunkErrorCode,
    PythonChunkMode,
    SymbolType,
)
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
    StrictModel,
)
from repopilot.tools.repository import ReadOnlyRepository, RepositoryAccessError

PYTHON_LANGUAGE = "python"
PYTHON_CHUNKER_VERSION = "python_ast_v1"
CHUNK_ERROR_MESSAGES = {
    PythonChunkErrorCode.SOURCE_UNAVAILABLE: (
        "manifest source is no longer available for Python chunking"
    ),
    PythonChunkErrorCode.SOURCE_CHANGED: ("manifest source content changed before Python chunking"),
}


class PythonChunkError(RuntimeError):
    """Safe typed fatal failure from Python source chunking."""

    def __init__(self, code: PythonChunkErrorCode) -> None:
        super().__init__(CHUNK_ERROR_MESSAGES[code])
        self.code = code


class PythonChunkPolicy(StrictModel):
    """Deterministic line-window limits for large Python symbols."""

    max_lines: int = Field(default=200, gt=0, le=2_000)
    overlap_lines: int = Field(default=20, ge=0, le=500)

    @field_validator("max_lines", "overlap_lines", mode="before")
    @classmethod
    def reject_booleans(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("chunk line limits must be integers, not booleans")
        return value

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.overlap_lines >= self.max_lines:
            raise ValueError("overlap_lines must be less than max_lines")
        return self

    @classmethod
    def from_settings(cls, settings: RepoPilotSettings) -> Self:
        return cls(
            max_lines=settings.chunk_max_lines,
            overlap_lines=settings.chunk_overlap_lines,
        )


class PythonSourceChunkResult(StrictModel):
    """Chunking outcome for one Python source in a Repository Manifest."""

    source_id: Identifier
    relative_path: RelativePath
    mode: PythonChunkMode
    syntax_error_line: int | None = Field(default=None, gt=0)
    chunks: tuple[CodeChunk, ...]

    @model_validator(mode="after")
    def validate_mode_and_chunks(self) -> Self:
        if not self.relative_path.endswith(".py"):
            raise ValueError("Python chunk results require a .py source")
        if self.mode is PythonChunkMode.EMPTY:
            if self.chunks or self.syntax_error_line is not None:
                raise ValueError("empty Python sources cannot contain chunks or syntax errors")
            return self
        if not self.chunks:
            raise ValueError("non-empty Python results require chunks")
        if self.mode is PythonChunkMode.SYNTAX_FALLBACK:
            if any(chunk.symbol_type is not SymbolType.TEXT for chunk in self.chunks):
                raise ValueError("syntax fallback chunks must use text symbol type")
        else:
            if self.syntax_error_line is not None:
                raise ValueError("AST chunk results cannot contain a syntax error line")
            if any(chunk.symbol_type is SymbolType.TEXT for chunk in self.chunks):
                raise ValueError("AST chunks cannot use text symbol type")
        chunk_order = tuple(
            (
                chunk.start_line,
                chunk.end_line,
                chunk.symbol_type.value,
                chunk.symbol_name or "",
                chunk.chunk_id,
            )
            for chunk in self.chunks
        )
        if chunk_order != tuple(sorted(chunk_order)):
            raise ValueError("source chunks must be deterministically sorted")
        if len({chunk.chunk_id for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("source chunk IDs must be unique")
        for chunk in self.chunks:
            if chunk.relative_path != self.relative_path:
                raise ValueError("chunk path must match its source result")
            if chunk.metadata.get("source_id") != self.source_id:
                raise ValueError("chunk metadata must identify its source")
            if chunk.metadata.get("chunk_mode") != self.mode.value:
                raise ValueError("chunk metadata mode must match its source result")
        return self


class PythonChunkingResult(StrictModel):
    """Deterministic Python chunk corpus for one repository commit."""

    repository_id: Identifier
    commit: GitSha
    sources: tuple[PythonSourceChunkResult, ...]

    @model_validator(mode="after")
    def validate_corpus(self) -> Self:
        paths = tuple(source.relative_path for source in self.sources)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("Python source results must have unique sorted paths")
        chunk_ids: list[str] = []
        for source in self.sources:
            for chunk in source.chunks:
                if chunk.repository_id != self.repository_id or chunk.commit != self.commit:
                    raise ValueError("chunk repository identity must match its corpus")
                chunk_ids.append(chunk.chunk_id)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Python chunk IDs must be unique")
        return self

    @property
    def chunks(self) -> tuple[CodeChunk, ...]:
        """Return the flattened corpus in deterministic source/chunk order."""

        return tuple(chunk for source in self.sources for chunk in source.chunks)


@dataclass(frozen=True)
class _Definition:
    qualified_name: str
    symbol_type: SymbolType
    ast_kind: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class _ChunkUnit:
    symbol_name: str | None
    symbol_type: SymbolType
    ast_kind: str
    start_line: int
    end_line: int


class PythonAstChunker:
    """Chunk Manifest-pinned Python files without importing or executing them."""

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

    def chunk_manifest(self, manifest: RepositoryManifest) -> PythonChunkingResult:
        """Validate and chunk every `.py` source in deterministic path order."""

        verify_repository_manifest_integrity(manifest)
        results = tuple(
            self._chunk_source(manifest, source)
            for source in manifest.files
            if source.relative_path.endswith(".py")
        )
        return PythonChunkingResult(
            repository_id=manifest.repository_id,
            commit=manifest.commit,
            sources=results,
        )

    def chunk_source(
        self,
        manifest: RepositoryManifest,
        source: RepositorySource,
    ) -> PythonSourceChunkResult:
        """Chunk one Python source after validating Manifest membership."""

        verify_repository_manifest_integrity(manifest)
        if source not in manifest.files or not source.relative_path.endswith(".py"):
            raise PythonChunkError(PythonChunkErrorCode.SOURCE_UNAVAILABLE)
        return self._chunk_source(manifest, source)

    def _chunk_source(
        self,
        manifest: RepositoryManifest,
        source: RepositorySource,
    ) -> PythonSourceChunkResult:
        try:
            text_file = self._repository.read_text_file(source.relative_path)
        except RepositoryAccessError:
            raise PythonChunkError(PythonChunkErrorCode.SOURCE_UNAVAILABLE) from None
        content_hash = hashlib.sha256(text_file.content.encode("utf-8")).hexdigest()
        if text_file.size_bytes != source.size_bytes or content_hash != source.content_hash:
            raise PythonChunkError(PythonChunkErrorCode.SOURCE_CHANGED)
        if not text_file.content.strip():
            return PythonSourceChunkResult(
                source_id=source.source_id,
                relative_path=source.relative_path,
                mode=PythonChunkMode.EMPTY,
                chunks=(),
            )

        try:
            tree = ast.parse(text_file.content, filename=source.relative_path)
        except SyntaxError as error:
            return self._syntax_fallback(
                manifest,
                source,
                text_file.content,
                syntax_error_line=error.lineno,
            )
        return self._ast_chunks(manifest, source, text_file.content, tree)

    def _ast_chunks(
        self,
        manifest: RepositoryManifest,
        source: RepositorySource,
        content: str,
        tree: ast.Module,
    ) -> PythonSourceChunkResult:
        lines = content.splitlines(keepends=True)
        definitions = _collect_definitions(tree)
        units = [
            _ChunkUnit(
                symbol_name=definition.qualified_name,
                symbol_type=definition.symbol_type,
                ast_kind=definition.ast_kind,
                start_line=definition.start_line,
                end_line=definition.end_line,
            )
            for definition in definitions
        ]
        units.extend(
            _module_units(
                tree,
                lines,
                module_name=_module_name(source.relative_path),
            )
        )
        units.sort(
            key=lambda unit: (
                unit.start_line,
                unit.end_line,
                unit.symbol_type.value,
                unit.symbol_name or "",
            )
        )
        chunks: list[CodeChunk] = []
        for unit in units:
            chunks.extend(
                self._chunks_for_unit(
                    manifest,
                    source,
                    lines,
                    unit,
                    mode=PythonChunkMode.AST,
                    extra_metadata={},
                )
            )
        if not chunks:
            return PythonSourceChunkResult(
                source_id=source.source_id,
                relative_path=source.relative_path,
                mode=PythonChunkMode.EMPTY,
                chunks=(),
            )
        return PythonSourceChunkResult(
            source_id=source.source_id,
            relative_path=source.relative_path,
            mode=PythonChunkMode.AST,
            chunks=tuple(chunks),
        )

    def _syntax_fallback(
        self,
        manifest: RepositoryManifest,
        source: RepositorySource,
        content: str,
        *,
        syntax_error_line: int | None,
    ) -> PythonSourceChunkResult:
        lines = content.splitlines(keepends=True)
        unit = _ChunkUnit(
            symbol_name=None,
            symbol_type=SymbolType.TEXT,
            ast_kind="syntax_fallback",
            start_line=1,
            end_line=len(lines),
        )
        chunks = self._chunks_for_unit(
            manifest,
            source,
            lines,
            unit,
            mode=PythonChunkMode.SYNTAX_FALLBACK,
            extra_metadata={
                "syntax_error_line": syntax_error_line,
            },
        )
        return PythonSourceChunkResult(
            source_id=source.source_id,
            relative_path=source.relative_path,
            mode=PythonChunkMode.SYNTAX_FALLBACK,
            syntax_error_line=syntax_error_line,
            chunks=tuple(chunks),
        )

    def _chunks_for_unit(
        self,
        manifest: RepositoryManifest,
        source: RepositorySource,
        lines: list[str],
        unit: _ChunkUnit,
        *,
        mode: PythonChunkMode,
        extra_metadata: dict[str, JsonValue],
    ) -> list[CodeChunk]:
        windows = _line_windows(
            unit.start_line,
            unit.end_line,
            max_lines=self._policy.max_lines,
            overlap_lines=self._policy.overlap_lines,
        )
        chunks: list[CodeChunk] = []
        symbol_name, symbol_metadata = _bounded_symbol_name(unit.symbol_name)
        for part_index, (start_line, end_line) in enumerate(windows):
            content = "".join(lines[start_line - 1 : end_line])
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            metadata: dict[str, JsonValue] = {
                "source_id": source.source_id,
                "source_content_hash": source.content_hash,
                "chunk_mode": mode.value,
                "ast_kind": unit.ast_kind,
                "processor": PYTHON_CHUNKER_VERSION,
                "part_index": part_index,
                "part_count": len(windows),
                "max_lines": self._policy.max_lines,
                "overlap_lines": self._policy.overlap_lines,
                **symbol_metadata,
                **extra_metadata,
            }
            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id_for(
                        source_id=source.source_id,
                        mode=mode,
                        symbol_type=unit.symbol_type,
                        symbol_name=symbol_name,
                        start_line=start_line,
                        end_line=end_line,
                        part_index=part_index,
                        content_hash=content_hash,
                    ),
                    repository_id=manifest.repository_id,
                    commit=manifest.commit,
                    relative_path=source.relative_path,
                    symbol_name=symbol_name,
                    symbol_type=unit.symbol_type,
                    start_line=start_line,
                    end_line=end_line,
                    language=PYTHON_LANGUAGE,
                    content=content,
                    content_hash=content_hash,
                    metadata=metadata,
                )
            )
        return chunks


def chunk_python_repository(
    repository: ReadOnlyRepository,
    manifest: RepositoryManifest,
    *,
    policy: PythonChunkPolicy | None = None,
) -> PythonChunkingResult:
    """Convenience entry point for deterministic Python AST-aware chunking."""

    return PythonAstChunker(repository, policy=policy).chunk_manifest(manifest)


def chunk_id_for(
    *,
    source_id: str,
    mode: PythonChunkMode,
    symbol_type: SymbolType,
    symbol_name: str | None,
    start_line: int,
    end_line: int,
    part_index: int,
    content_hash: str,
) -> str:
    """Return a stable chunk identity for one exact source span."""

    canonical = json.dumps(
        {
            "content_hash": content_hash,
            "end_line": end_line,
            "mode": mode.value,
            "part_index": part_index,
            "source_id": source_id,
            "start_line": start_line,
            "symbol_name": symbol_name,
            "symbol_type": symbol_type.value,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"chunk_{hashlib.sha256(canonical).hexdigest()}"


def _collect_definitions(tree: ast.Module) -> tuple[_Definition, ...]:
    definitions: list[_Definition] = []

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        child_scope = scope
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            qualified_name = ".".join((*scope, node.name))
            start_line = min(
                (decorator.lineno for decorator in node.decorator_list),
                default=node.lineno,
            )
            end_line = node.end_lineno or node.lineno
            if isinstance(node, ast.ClassDef):
                symbol_type = SymbolType.CLASS
                ast_kind = "class"
            elif isinstance(node, ast.AsyncFunctionDef):
                symbol_type = SymbolType.FUNCTION
                ast_kind = "async_function"
            else:
                symbol_type = SymbolType.FUNCTION
                ast_kind = "function"
            definitions.append(
                _Definition(
                    qualified_name=qualified_name,
                    symbol_type=symbol_type,
                    ast_kind=ast_kind,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            child_scope = (*scope, node.name)
        for child in ast.iter_child_nodes(node):
            visit(child, child_scope)

    visit(tree, ())
    return tuple(
        sorted(
            definitions,
            key=lambda definition: (
                definition.start_line,
                definition.end_line,
                definition.symbol_type.value,
                definition.qualified_name,
            ),
        )
    )


def _module_units(
    tree: ast.Module,
    lines: list[str],
    *,
    module_name: str,
) -> tuple[_ChunkUnit, ...]:
    top_level_spans = sorted(
        (
            min(
                (decorator.lineno for decorator in node.decorator_list),
                default=node.lineno,
            ),
            node.end_lineno or node.lineno,
        )
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    )
    units: list[_ChunkUnit] = []
    cursor = 1
    for start_line, end_line in top_level_spans:
        _append_module_unit(units, lines, module_name, cursor, start_line - 1)
        cursor = end_line + 1
    _append_module_unit(units, lines, module_name, cursor, len(lines))
    return tuple(units)


def _append_module_unit(
    units: list[_ChunkUnit],
    lines: list[str],
    module_name: str,
    start_line: int,
    end_line: int,
) -> None:
    while start_line <= end_line and not lines[start_line - 1].strip():
        start_line += 1
    while end_line >= start_line and not lines[end_line - 1].strip():
        end_line -= 1
    if start_line > end_line:
        return
    units.append(
        _ChunkUnit(
            symbol_name=module_name,
            symbol_type=SymbolType.MODULE,
            ast_kind="module",
            start_line=start_line,
            end_line=end_line,
        )
    )


def _line_windows(
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


def _module_name(relative_path: str) -> str:
    without_suffix = relative_path[:-3]
    parts = without_suffix.split("/")
    if parts[-1] == "__init__" and len(parts) > 1:
        parts = parts[:-1]
    return ".".join(parts)


def _bounded_symbol_name(value: str | None) -> tuple[str | None, dict[str, JsonValue]]:
    if value is None or len(value) <= 512:
        return value, {}
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    bounded = f"{value[:447]}.{digest}"
    return bounded, {
        "symbol_name_truncated": True,
        "symbol_name_sha256": digest,
    }
