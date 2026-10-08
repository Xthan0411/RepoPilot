"""Deterministic bounded text search over contained repository files."""

from __future__ import annotations

import re
from collections.abc import Iterator

from pydantic import BaseModel, Field, StrictBool, field_validator

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    RepositoryEntryType,
    RepositoryErrorCode,
    SourceInspectionErrorCode,
)
from repopilot.schemas import RelativePath, StrictModel, ToolCall
from repopilot.tools.registry import ToolArgumentError
from repopilot.tools.repository import ReadOnlyRepository, RepositoryAccessError, RepositoryLocation

SEARCH_PREVIEW_CHARS = 240
HARD_MAX_MATCHES = 1_000
HARD_MAX_QUERY_LENGTH = 2_000


class PartialSourceReadError(RuntimeError):
    """A configured read-byte boundary split one logical source line."""


class SearchCodeArguments(StrictModel):
    """Arguments for literal, line-oriented repository search."""

    query: str = Field(min_length=1, max_length=HARD_MAX_QUERY_LENGTH)
    path: str = Field(
        default=".",
        min_length=1,
        description="Canonical repository-relative path; use '.' for the repository root.",
    )
    case_sensitive: StrictBool = True
    max_matches: int | None = Field(
        default=None,
        gt=0,
        le=HARD_MAX_MATCHES,
        strict=True,
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip() or "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("query must be one non-empty text line")
        return value


class SearchMatch(StrictModel):
    """One literal match with 1-based inclusive source columns."""

    relative_path: RelativePath
    line: int = Field(gt=0)
    column_start: int = Field(gt=0)
    column_end: int = Field(gt=0)
    preview: str = Field(max_length=SEARCH_PREVIEW_CHARS)
    preview_start_column: int = Field(gt=0)


class SkippedSourceFile(StrictModel):
    """A file omitted because safe complete inspection was impossible."""

    relative_path: RelativePath
    reason: RepositoryErrorCode | SourceInspectionErrorCode


class SearchCodeResult(StrictModel):
    """Bounded search output and explicit completeness metadata."""

    query: str
    path: RepositoryLocation
    matches: tuple[SearchMatch, ...]
    files_scanned: int = Field(ge=0)
    skipped_files: tuple[SkippedSourceFile, ...]
    truncated: bool


class SearchCodeTool:
    """Search repository text without a shell, index, embedding, or model call."""

    name = "search_code"
    description = (
        "Find bounded literal text matches in one readable repository file or beneath "
        "one repository directory."
    )
    arguments_model: type[BaseModel] = SearchCodeArguments

    def __init__(
        self,
        repository: ReadOnlyRepository,
        *,
        max_matches: int = 100,
        max_query_length: int = 256,
    ) -> None:
        if not 0 < max_matches <= HARD_MAX_MATCHES:
            raise ValueError("max_matches is outside hard limits")
        if not 0 < max_query_length <= HARD_MAX_QUERY_LENGTH:
            raise ValueError("max_query_length is outside hard limits")
        self._repository = repository
        self._max_matches = max_matches
        self._max_query_length = max_query_length

    @classmethod
    def from_settings(
        cls,
        repository: ReadOnlyRepository,
        settings: RepoPilotSettings,
    ) -> SearchCodeTool:
        return cls(
            repository,
            max_matches=settings.search_max_matches,
            max_query_length=settings.search_max_query_length,
        )

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        if not isinstance(arguments, SearchCodeArguments):
            raise TypeError("unexpected argument model")
        if len(arguments.query) > self._max_query_length:
            raise ToolArgumentError("query exceeds configured length")
        requested_limit = arguments.max_matches or self._max_matches
        if requested_limit > self._max_matches:
            raise ToolArgumentError("max_matches exceeds configured limit")

        try:
            listing = self._repository.list_directory(arguments.path, recursive=True)
        except RepositoryAccessError as error:
            if error.code is not RepositoryErrorCode.NOT_DIRECTORY:
                raise
            return self._search_single_file(arguments, requested_limit=requested_limit)
        matches: list[SearchMatch] = []
        skipped: list[SkippedSourceFile] = []
        files_scanned = 0
        found_extra_match = False
        for entry in listing.entries:
            if entry.entry_type is not RepositoryEntryType.FILE:
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

            files_scanned += 1
            for match in _find_matches(
                entry.relative_path,
                source,
                query=arguments.query,
                case_sensitive=arguments.case_sensitive,
            ):
                if len(matches) == requested_limit:
                    found_extra_match = True
                    break
                matches.append(match)
            if found_extra_match:
                break

        return SearchCodeResult(
            query=arguments.query,
            path=listing.directory,
            matches=tuple(matches),
            files_scanned=files_scanned,
            skipped_files=tuple(skipped),
            truncated=listing.truncated or found_extra_match,
        )

    def _search_single_file(
        self,
        arguments: SearchCodeArguments,
        *,
        requested_limit: int,
    ) -> SearchCodeResult:
        skipped: tuple[SkippedSourceFile, ...] = ()
        try:
            source = read_complete_source(self._repository, arguments.path)
        except PartialSourceReadError:
            source = ""
            skipped = (
                SkippedSourceFile(
                    relative_path=arguments.path,
                    reason=SourceInspectionErrorCode.PARTIAL_LINE,
                ),
            )
        except RepositoryAccessError as error:
            source = ""
            skipped = (
                SkippedSourceFile(
                    relative_path=arguments.path,
                    reason=error.code,
                ),
            )
        matches = tuple(
            _find_matches(
                arguments.path,
                source,
                query=arguments.query,
                case_sensitive=arguments.case_sensitive,
            )
        )
        return SearchCodeResult(
            query=arguments.query,
            path=arguments.path,
            matches=matches[:requested_limit],
            files_scanned=0 if skipped else 1,
            skipped_files=skipped,
            truncated=len(matches) > requested_limit,
        )


def read_complete_source(repository: ReadOnlyRepository, path: str) -> str:
    """Reassemble bounded line pages, rejecting an unrepresentable partial line."""

    pages: list[str] = []
    start_line = 1
    while True:
        page = repository.read_file(path, start_line=start_line)
        if page.truncated and not page.content.endswith(("\n", "\r")):
            raise PartialSourceReadError
        pages.append(page.content)
        if page.next_start_line is None:
            return "".join(pages)
        start_line = page.next_start_line


def _find_matches(
    relative_path: str,
    source: str,
    *,
    query: str,
    case_sensitive: bool,
) -> Iterator[SearchMatch]:
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(re.escape(query), flags)
    for line_number, line in enumerate(source.splitlines(), start=1):
        for found in pattern.finditer(line):
            preview, preview_start = _make_preview(line, found.start())
            yield SearchMatch(
                relative_path=relative_path,
                line=line_number,
                column_start=found.start() + 1,
                column_end=found.end(),
                preview=preview,
                preview_start_column=preview_start + 1,
            )


def _make_preview(line: str, match_start: int) -> tuple[str, int]:
    if len(line) <= SEARCH_PREVIEW_CHARS:
        return line, 0
    proposed_start = max(0, match_start - SEARCH_PREVIEW_CHARS // 3)
    preview_start = min(proposed_start, len(line) - SEARCH_PREVIEW_CHARS)
    return line[preview_start : preview_start + SEARCH_PREVIEW_CHARS], preview_start
