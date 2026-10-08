"""Bounded Python AST symbol inspection without an LSP or code execution."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    RepositoryEntryType,
    RepositoryErrorCode,
    SourceInspectionErrorCode,
    SymbolKind,
)
from repopilot.schemas import RelativePath, StrictModel, ToolCall
from repopilot.tools.repository import ReadOnlyRepository, RepositoryAccessError, RepositoryLocation
from repopilot.tools.search import (
    HARD_MAX_MATCHES,
    PartialSourceReadError,
    SkippedSourceFile,
    read_complete_source,
)


class InspectSymbolArguments(StrictModel):
    """Arguments for exact simple-name or qualified-name lookup."""

    symbol: str = Field(min_length=1, max_length=512)
    path: str = Field(
        default=".",
        min_length=1,
        description="Canonical repository-relative path; use '.' for the repository root.",
    )
    kind: SymbolKind | None = None

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\x00" in normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("symbol must be one non-empty text line")
        return normalized


class SymbolLocation(StrictModel):
    """One Python definition with a source span and qualified name."""

    relative_path: RelativePath
    name: str = Field(min_length=1)
    qualified_name: str = Field(min_length=1)
    kind: SymbolKind
    start_line: int = Field(gt=0)
    end_line: int = Field(gt=0)


class InspectSymbolResult(StrictModel):
    """Bounded symbol matches and explicit source-inspection omissions."""

    symbol: str
    path: RepositoryLocation
    symbols: tuple[SymbolLocation, ...]
    files_scanned: int = Field(ge=0)
    skipped_files: tuple[SkippedSourceFile, ...]
    truncated: bool


class InspectSymbolTool:
    """Locate Python definitions with the standard-library AST."""

    name = "inspect_symbol"
    description = (
        "Locate exact Python definitions in one repository Python file or beneath one "
        "repository directory."
    )
    arguments_model: type[BaseModel] = InspectSymbolArguments

    def __init__(
        self,
        repository: ReadOnlyRepository,
        *,
        max_results: int = 100,
    ) -> None:
        if not 0 < max_results <= HARD_MAX_MATCHES:
            raise ValueError("max_results is outside hard limits")
        self._repository = repository
        self._max_results = max_results

    @classmethod
    def from_settings(
        cls,
        repository: ReadOnlyRepository,
        settings: RepoPilotSettings,
    ) -> InspectSymbolTool:
        return cls(repository, max_results=settings.search_max_matches)

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        if not isinstance(arguments, InspectSymbolArguments):
            raise TypeError("unexpected argument model")
        try:
            listing = self._repository.list_directory(arguments.path, recursive=True)
        except RepositoryAccessError as error:
            if error.code is not RepositoryErrorCode.NOT_DIRECTORY:
                raise
            return self._inspect_single_file(arguments)
        symbols: list[SymbolLocation] = []
        skipped: list[SkippedSourceFile] = []
        files_scanned = 0
        found_extra = False
        for entry in listing.entries:
            if entry.entry_type is not RepositoryEntryType.FILE or not entry.relative_path.endswith(
                ".py"
            ):
                continue
            try:
                source = read_complete_source(self._repository, entry.relative_path)
            except PartialSourceReadError:
                skipped.append(
                    SkippedSourceFile(
                        relative_path=entry.relative_path,
                        reason=SourceInspectionErrorCode.PARTIAL_LINE,
                    )
                )
                continue
            except RepositoryAccessError as error:
                skipped.append(
                    SkippedSourceFile(
                        relative_path=entry.relative_path,
                        reason=error.code,
                    )
                )
                continue
            try:
                tree = ast.parse(source, filename=entry.relative_path)
            except SyntaxError:
                skipped.append(
                    SkippedSourceFile(
                        relative_path=entry.relative_path,
                        reason=SourceInspectionErrorCode.SYNTAX_ERROR,
                    )
                )
                continue

            files_scanned += 1
            collector = _SymbolCollector(
                relative_path=entry.relative_path,
                target=arguments.symbol,
                requested_kind=arguments.kind,
                limit=self._max_results + 1 - len(symbols),
            )
            collector.visit(tree)
            symbols.extend(collector.results)
            if len(symbols) > self._max_results:
                found_extra = True
                symbols = symbols[: self._max_results]
                break

        return InspectSymbolResult(
            symbol=arguments.symbol,
            path=listing.directory,
            symbols=tuple(symbols),
            files_scanned=files_scanned,
            skipped_files=tuple(skipped),
            truncated=listing.truncated or found_extra,
        )

    def _inspect_single_file(
        self,
        arguments: InspectSymbolArguments,
    ) -> InspectSymbolResult:
        skipped: tuple[SkippedSourceFile, ...] = ()
        collected: tuple[SymbolLocation, ...] = ()
        try:
            source = read_complete_source(self._repository, arguments.path)
        except PartialSourceReadError:
            skipped = (
                SkippedSourceFile(
                    relative_path=arguments.path,
                    reason=SourceInspectionErrorCode.PARTIAL_LINE,
                ),
            )
        except RepositoryAccessError as error:
            skipped = (
                SkippedSourceFile(
                    relative_path=arguments.path,
                    reason=error.code,
                ),
            )
        else:
            try:
                tree = ast.parse(source, filename=arguments.path)
            except SyntaxError:
                skipped = (
                    SkippedSourceFile(
                        relative_path=arguments.path,
                        reason=SourceInspectionErrorCode.SYNTAX_ERROR,
                    ),
                )
            else:
                collector = _SymbolCollector(
                    relative_path=arguments.path,
                    target=arguments.symbol,
                    requested_kind=arguments.kind,
                    limit=self._max_results + 1,
                )
                collector.visit(tree)
                collected = tuple(collector.results)
        return InspectSymbolResult(
            symbol=arguments.symbol,
            path=arguments.path,
            symbols=collected[: self._max_results],
            files_scanned=0 if skipped else 1,
            skipped_files=skipped,
            truncated=len(collected) > self._max_results,
        )


@dataclass(frozen=True)
class _DefinitionContext:
    name: str
    is_class: bool


class _SymbolCollector(ast.NodeVisitor):
    """Collect matching definitions in deterministic AST source order."""

    def __init__(
        self,
        *,
        relative_path: str,
        target: str,
        requested_kind: SymbolKind | None,
        limit: int,
    ) -> None:
        self._relative_path = relative_path
        self._target = target
        self._requested_kind = requested_kind
        self._limit = limit
        self._contexts: list[_DefinitionContext] = []
        self.results: list[SymbolLocation] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node, SymbolKind.CLASS)
        self._contexts.append(_DefinitionContext(node.name, is_class=True))
        self.generic_visit(node)
        self._contexts.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = (
            SymbolKind.METHOD
            if self._contexts and self._contexts[-1].is_class
            else SymbolKind.FUNCTION
        )
        self._visit_function(node, kind)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        kind = (
            SymbolKind.ASYNC_METHOD
            if self._contexts and self._contexts[-1].is_class
            else SymbolKind.ASYNC_FUNCTION
        )
        self._visit_function(node, kind)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        kind: SymbolKind,
    ) -> None:
        self._record(node, kind)
        self._contexts.append(_DefinitionContext(node.name, is_class=False))
        self.generic_visit(node)
        self._contexts.pop()

    def _record(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        kind: SymbolKind,
    ) -> None:
        qualified_name = ".".join([*(context.name for context in self._contexts), node.name])
        if (
            len(self.results) < self._limit
            and (node.name == self._target or qualified_name == self._target)
            and (self._requested_kind is None or kind is self._requested_kind)
        ):
            self.results.append(
                SymbolLocation(
                    relative_path=self._relative_path,
                    name=node.name,
                    qualified_name=qualified_name,
                    kind=kind,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                )
            )
