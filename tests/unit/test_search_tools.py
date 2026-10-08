"""Literal search behavior, completeness metadata, and safety limits."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import JsonValue, ValidationError

from repopilot.enums import (
    ErrorType,
    RepositoryErrorCode,
    SourceInspectionErrorCode,
    ToolErrorCode,
)
from repopilot.schemas import ToolCall
from repopilot.tools import ReadOnlyRepository, ToolRegistry
from repopilot.tools.repository import RepositoryAccessPolicy
from repopilot.tools.search import SearchCodeArguments, SearchCodeResult, SearchCodeTool

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "repositories" / "read_only"


def execute_search(
    tool: SearchCodeTool,
    **arguments: object,
) -> SearchCodeResult:
    validated = SearchCodeArguments.model_validate(arguments)
    result = tool.execute(
        validated,
        call=make_call({"query": validated.query}),
    )
    assert isinstance(result, SearchCodeResult)
    return result


def make_call(arguments: dict[str, JsonValue]) -> ToolCall:
    return ToolCall(
        call_id=uuid4(),
        tool_name="search_code",
        validated_arguments=arguments,
        requested_at=datetime(2026, 7, 27, tzinfo=UTC),
    )


def test_search_is_deterministic_line_oriented_and_round_trippable() -> None:
    tool = SearchCodeTool(ReadOnlyRepository(FIXTURE_ROOT))

    first = execute_search(tool, query="parse")
    second = execute_search(tool, query="parse")

    assert first == second
    assert tuple(
        (match.relative_path, match.line, match.column_start, match.column_end)
        for match in first.matches
    ) == (
        ("pkg/__init__.py", 3, 10, 14),
        ("pkg/__init__.py", 3, 24, 28),
        ("pkg/__init__.py", 5, 13, 17),
        ("pkg/parser.py", 4, 5, 9),
        ("pkg/service.py", 5, 13, 17),
        ("pkg/service.py", 7, 9, 13),
        ("pkg/service.py", 10, 15, 19),
        ("pkg/service.py", 11, 21, 25),
        ("pkg/service.py", 14, 16, 20),
        ("pkg/service.py", 15, 16, 20),
    )
    assert first.files_scanned == 5
    assert first.skipped_files == ()
    assert first.truncated is False
    assert SearchCodeResult.model_validate_json(first.model_dump_json()) == first


def test_search_schema_requires_dot_instead_of_an_empty_root_path() -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        SearchCodeArguments(query="parse", path="")

    schema = SearchCodeArguments.model_json_schema()
    path_schema = schema["properties"]["path"]
    assert path_schema["minLength"] == 1
    assert "use '.'" in path_schema["description"]


def test_case_sensitivity_and_unicode_search_are_explicit() -> None:
    tool = SearchCodeTool(ReadOnlyRepository(FIXTURE_ROOT))

    sensitive = execute_search(tool, query="PARSER")
    insensitive = execute_search(tool, query="PARSER", case_sensitive=False)
    unicode_result = execute_search(tool, query="验证", path="docs")

    assert sensitive.matches == ()
    assert len(insensitive.matches) == 7
    assert tuple(match.relative_path for match in unicode_result.matches) == ("docs/说明.md",)
    assert unicode_result.matches[0].line == 3
    assert unicode_result.matches[0].column_start == 7


def test_search_scope_and_multiple_non_overlapping_matches(tmp_path: Path) -> None:
    source = tmp_path / "nested"
    source.mkdir()
    source.joinpath("sample.txt").write_text("foo foo foo\n", encoding="utf-8")
    tool = SearchCodeTool(ReadOnlyRepository(tmp_path))

    result = execute_search(tool, query="foo", path="nested")

    assert tuple(match.column_start for match in result.matches) == (1, 5, 9)
    assert tuple(match.column_end for match in result.matches) == (3, 7, 11)
    assert all(match.preview == "foo foo foo" for match in result.matches)
    assert all(match.preview_start_column == 1 for match in result.matches)


def test_search_accepts_one_explicit_file_scope() -> None:
    tool = SearchCodeTool(ReadOnlyRepository(FIXTURE_ROOT))

    result = execute_search(tool, query="def parse", path="pkg/parser.py")

    assert result.path == "pkg/parser.py"
    assert result.files_scanned == 1
    assert tuple(match.relative_path for match in result.matches) == ("pkg/parser.py",)
    assert result.skipped_files == ()


def test_literal_search_preserves_meaningful_edge_whitespace(tmp_path: Path) -> None:
    tmp_path.joinpath("spacing.txt").write_text(" foo\nfoo\n", encoding="utf-8")
    tool = SearchCodeTool(ReadOnlyRepository(tmp_path))

    result = execute_search(tool, query=" foo")

    assert result.query == " foo"
    assert tuple((match.line, match.column_start) for match in result.matches) == ((1, 1),)


def test_search_reassembles_repository_line_pages(tmp_path: Path) -> None:
    tmp_path.joinpath("pages.txt").write_text(
        "first line\nsecond line\nneedle on third\n",
        encoding="utf-8",
    )
    repository = ReadOnlyRepository(
        tmp_path,
        policy=RepositoryAccessPolicy(max_read_lines=1),
    )
    tool = SearchCodeTool(repository)

    result = execute_search(tool, query="needle")

    assert len(result.matches) == 1
    assert result.matches[0].line == 3
    assert result.skipped_files == ()


def test_long_preview_is_bounded_around_the_match(tmp_path: Path) -> None:
    line = "a" * 300 + "needle" + "z" * 300
    tmp_path.joinpath("long.txt").write_text(f"{line}\n", encoding="utf-8")
    tool = SearchCodeTool(ReadOnlyRepository(tmp_path))

    result = execute_search(tool, query="needle")
    match = result.matches[0]

    assert len(match.preview) == 240
    assert "needle" in match.preview
    assert match.column_start == 301
    assert match.column_end == 306
    assert match.preview_start_column > 1


def test_match_limit_stops_with_explicit_truncation(tmp_path: Path) -> None:
    tmp_path.joinpath("many.txt").write_text("x x x x x\n", encoding="utf-8")
    tool = SearchCodeTool(ReadOnlyRepository(tmp_path), max_matches=3)

    result = execute_search(tool, query="x")

    assert len(result.matches) == 3
    assert result.truncated is True


def test_configured_query_and_match_limits_return_invalid_arguments() -> None:
    registry = ToolRegistry()
    registry.register(
        SearchCodeTool(
            ReadOnlyRepository(FIXTURE_ROOT),
            max_matches=2,
            max_query_length=5,
        )
    )

    too_many = registry.invoke(make_call({"query": "parse", "max_matches": 3}))
    too_long = registry.invoke(make_call({"query": "parser"}))

    assert too_many.error_code is ToolErrorCode.INVALID_ARGUMENTS
    assert too_long.error_code is ToolErrorCode.INVALID_ARGUMENTS
    assert too_many.error_type is ErrorType.INVALID_TOOL_ARGUMENTS


@pytest.mark.parametrize(
    "query",
    ["", "   ", "two\nlines", "nul\x00query", "x" * 2_001],
)
def test_query_schema_rejects_illegal_text(query: str) -> None:
    with pytest.raises(ValidationError):
        SearchCodeArguments(query=query)


def test_unreadable_files_are_skipped_with_stable_reasons(tmp_path: Path) -> None:
    tmp_path.joinpath("good.txt").write_text("needle\n", encoding="utf-8")
    tmp_path.joinpath("binary.dat").write_bytes(b"needle\x00binary")
    tmp_path.joinpath("invalid.dat").write_bytes(b"\xff\xfe")
    tmp_path.joinpath("large.txt").write_text("needle too large\n", encoding="utf-8")
    policy = RepositoryAccessPolicy(
        max_file_bytes=15,
        max_read_bytes=15,
    )
    tool = SearchCodeTool(ReadOnlyRepository(tmp_path, policy=policy))

    result = execute_search(tool, query="needle")

    assert tuple(match.relative_path for match in result.matches) == ("good.txt",)
    assert tuple((item.relative_path, item.reason) for item in result.skipped_files) == (
        ("binary.dat", RepositoryErrorCode.BINARY_FILE),
        ("invalid.dat", RepositoryErrorCode.DECODE_ERROR),
        ("large.txt", RepositoryErrorCode.FILE_TOO_LARGE),
    )
    assert result.files_scanned == 1


def test_partial_long_line_is_skipped_instead_of_returning_incomplete_text(
    tmp_path: Path,
) -> None:
    tmp_path.joinpath("long.txt").write_text("needle-after-boundary\n", encoding="utf-8")
    policy = RepositoryAccessPolicy(max_file_bytes=100, max_read_bytes=5)
    tool = SearchCodeTool(ReadOnlyRepository(tmp_path, policy=policy))

    result = execute_search(tool, query="needle")

    assert result.matches == ()
    assert tuple((item.relative_path, item.reason) for item in result.skipped_files) == (
        ("long.txt", SourceInspectionErrorCode.PARTIAL_LINE),
    )
    assert result.files_scanned == 0


def test_listing_limit_and_path_violation_are_not_hidden(tmp_path: Path) -> None:
    tmp_path.joinpath("a.txt").write_text("needle\n", encoding="utf-8")
    tmp_path.joinpath("b.txt").write_text("needle\n", encoding="utf-8")
    repository = ReadOnlyRepository(
        tmp_path,
        policy=RepositoryAccessPolicy(max_list_entries=1),
    )
    tool = SearchCodeTool(repository)

    limited = execute_search(tool, query="needle")
    registry = ToolRegistry()
    registry.register(tool)
    escape = registry.invoke(make_call({"query": "needle", "path": "../outside"}))

    assert limited.truncated is True
    assert tuple(match.relative_path for match in limited.matches) == ("a.txt",)
    assert escape.error_type is ErrorType.PATH_VIOLATION
    assert escape.repository_error_code is RepositoryErrorCode.INVALID_PATH
