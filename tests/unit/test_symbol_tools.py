"""Python AST symbol inspection and malformed-source behavior."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from repopilot.enums import (
    ErrorType,
    RepositoryErrorCode,
    SourceInspectionErrorCode,
    SymbolKind,
)
from repopilot.schemas import ToolCall
from repopilot.tools import ReadOnlyRepository, ToolRegistry
from repopilot.tools.repository import RepositoryAccessPolicy
from repopilot.tools.symbols import (
    InspectSymbolArguments,
    InspectSymbolResult,
    InspectSymbolTool,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "repositories" / "read_only"


def execute_inspection(
    tool: InspectSymbolTool,
    **arguments: object,
) -> InspectSymbolResult:
    validated = InspectSymbolArguments.model_validate(arguments)
    result = tool.execute(
        validated,
        call=ToolCall(
            call_id=uuid4(),
            tool_name="inspect_symbol",
            validated_arguments={"symbol": validated.symbol},
            requested_at=datetime(2026, 7, 27, tzinfo=UTC),
        ),
    )
    assert isinstance(result, InspectSymbolResult)
    return result


def test_inspects_function_and_method_with_exact_source_spans() -> None:
    tool = InspectSymbolTool(ReadOnlyRepository(FIXTURE_ROOT))

    result = execute_inspection(tool, symbol="parse")

    assert tuple(
        (
            symbol.relative_path,
            symbol.qualified_name,
            symbol.kind,
            symbol.start_line,
            symbol.end_line,
        )
        for symbol in result.symbols
    ) == (
        ("pkg/parser.py", "parse", SymbolKind.FUNCTION, 4, 11),
        ("pkg/service.py", "ParserService.parse", SymbolKind.METHOD, 7, 8),
    )
    assert result.files_scanned == 3
    assert result.skipped_files == ()
    assert result.truncated is False
    assert InspectSymbolResult.model_validate_json(result.model_dump_json()) == result


def test_inspection_accepts_one_explicit_python_file_scope() -> None:
    tool = InspectSymbolTool(ReadOnlyRepository(FIXTURE_ROOT))

    result = execute_inspection(
        tool,
        symbol="ParserService.parse_async",
        path="pkg/service.py",
        kind=SymbolKind.ASYNC_METHOD,
    )

    assert result.path == "pkg/service.py"
    assert result.files_scanned == 1
    assert tuple(item.qualified_name for item in result.symbols) == ("ParserService.parse_async",)
    assert result.skipped_files == ()


def test_symbol_schema_requires_dot_instead_of_an_empty_root_path() -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        InspectSymbolArguments(symbol="parse", path="")

    schema = InspectSymbolArguments.model_json_schema()
    path_schema = schema["properties"]["path"]
    assert path_schema["minLength"] == 1
    assert "use '.'" in path_schema["description"]


@pytest.mark.parametrize(
    ("symbol", "kind", "qualified_name", "start_line"),
    [
        ("ParserService", SymbolKind.CLASS, "ParserService", 4),
        ("parse_async", SymbolKind.ASYNC_METHOD, "ParserService.parse_async", 10),
        ("load_parser", SymbolKind.ASYNC_FUNCTION, "load_parser", 14),
        ("ParserService.parse", SymbolKind.METHOD, "ParserService.parse", 7),
    ],
)
def test_class_async_and_qualified_symbol_queries(
    symbol: str,
    kind: SymbolKind,
    qualified_name: str,
    start_line: int,
) -> None:
    tool = InspectSymbolTool(ReadOnlyRepository(FIXTURE_ROOT))

    result = execute_inspection(tool, symbol=symbol, kind=kind)

    assert len(result.symbols) == 1
    assert result.symbols[0].qualified_name == qualified_name
    assert result.symbols[0].kind is kind
    assert result.symbols[0].start_line == start_line


def test_kind_filter_distinguishes_top_level_function_and_method() -> None:
    tool = InspectSymbolTool(ReadOnlyRepository(FIXTURE_ROOT))

    function = execute_inspection(tool, symbol="parse", kind=SymbolKind.FUNCTION)
    method = execute_inspection(tool, symbol="parse", kind=SymbolKind.METHOD)

    assert tuple(item.qualified_name for item in function.symbols) == ("parse",)
    assert tuple(item.qualified_name for item in method.symbols) == ("ParserService.parse",)


def test_nested_qualified_names_follow_lexical_definition_context(tmp_path: Path) -> None:
    tmp_path.joinpath("nested.py").write_text(
        "class Outer:\n"
        "    class Inner:\n"
        "        def run(self):\n"
        "            def nested():\n"
        "                return 1\n"
        "            return nested()\n",
        encoding="utf-8",
    )
    tool = InspectSymbolTool(ReadOnlyRepository(tmp_path))

    method = execute_inspection(tool, symbol="Outer.Inner.run")
    nested = execute_inspection(tool, symbol="nested")

    assert method.symbols[0].kind is SymbolKind.METHOD
    assert method.symbols[0].qualified_name == "Outer.Inner.run"
    assert nested.symbols[0].kind is SymbolKind.FUNCTION
    assert nested.symbols[0].qualified_name == "Outer.Inner.run.nested"


def test_syntax_error_is_reported_and_non_python_files_are_ignored(tmp_path: Path) -> None:
    tmp_path.joinpath("broken.py").write_text("def target(:\n", encoding="utf-8")
    tmp_path.joinpath("notes.txt").write_text("def target():\n", encoding="utf-8")
    tmp_path.joinpath("valid.py").write_text(
        "def target():\n    return 'ok'\n",
        encoding="utf-8",
    )
    tool = InspectSymbolTool(ReadOnlyRepository(tmp_path))

    result = execute_inspection(tool, symbol="target")

    assert tuple(item.relative_path for item in result.symbols) == ("valid.py",)
    assert tuple((item.relative_path, item.reason) for item in result.skipped_files) == (
        ("broken.py", SourceInspectionErrorCode.SYNTAX_ERROR),
    )
    assert result.files_scanned == 1


def test_inspection_parses_but_never_executes_source(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    tmp_path.joinpath("danger.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n"
        "def safe_symbol():\n    return True\n",
        encoding="utf-8",
    )
    tool = InspectSymbolTool(ReadOnlyRepository(tmp_path))

    result = execute_inspection(tool, symbol="safe_symbol")

    assert len(result.symbols) == 1
    assert not marker.exists()


def test_symbol_result_limit_is_explicitly_truncated(tmp_path: Path) -> None:
    tmp_path.joinpath("many.py").write_text(
        "def repeated():\n    pass\n" * 3,
        encoding="utf-8",
    )
    tool = InspectSymbolTool(ReadOnlyRepository(tmp_path), max_results=2)

    result = execute_inspection(tool, symbol="repeated")

    assert len(result.symbols) == 2
    assert result.truncated is True


def test_partial_line_and_binary_python_files_are_skipped(tmp_path: Path) -> None:
    tmp_path.joinpath("binary.py").write_bytes(b"def target():\x00\n")
    tmp_path.joinpath("long.py").write_text(
        "def target_with_a_very_long_name():\n    pass\n",
        encoding="utf-8",
    )
    repository = ReadOnlyRepository(
        tmp_path,
        policy=RepositoryAccessPolicy(max_file_bytes=100, max_read_bytes=5),
    )
    tool = InspectSymbolTool(repository)

    result = execute_inspection(tool, symbol="target")

    assert tuple((item.relative_path, item.reason) for item in result.skipped_files) == (
        ("binary.py", RepositoryErrorCode.BINARY_FILE),
        ("long.py", SourceInspectionErrorCode.PARTIAL_LINE),
    )
    assert result.symbols == ()


@pytest.mark.parametrize("symbol", ["", "   ", "bad\nname", "nul\x00name"])
def test_symbol_schema_rejects_illegal_text(symbol: str) -> None:
    with pytest.raises(ValidationError):
        InspectSymbolArguments(symbol=symbol)


def test_inspection_path_violation_is_preserved_by_registry() -> None:
    registry = ToolRegistry()
    registry.register(InspectSymbolTool(ReadOnlyRepository(FIXTURE_ROOT)))
    call = ToolCall(
        call_id=uuid4(),
        tool_name="inspect_symbol",
        validated_arguments={"symbol": "parse", "path": "../outside"},
        requested_at=datetime(2026, 7, 27, tzinfo=UTC),
    )

    result = registry.invoke(call)

    assert result.error_type is ErrorType.PATH_VIOLATION
    assert result.repository_error_code is RepositoryErrorCode.INVALID_PATH
