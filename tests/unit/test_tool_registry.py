"""Tool Registry contracts, dispatch behavior, and safe failure tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, Field, JsonValue

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    ErrorType,
    RepositoryErrorCode,
    ToolErrorCode,
    ToolResultStatus,
)
from repopilot.schemas import StrictModel, ToolCall
from repopilot.tools import (
    ListFilesTool,
    ReadFileTool,
    ReadOnlyRepository,
    ToolExecutionResult,
    ToolRegistrationError,
    ToolRegistry,
    build_read_only_tool_registry,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "repositories" / "read_only"


def make_call(
    tool_name: str,
    arguments: dict[str, JsonValue],
    *,
    call_id: UUID | None = None,
) -> ToolCall:
    return ToolCall(
        call_id=call_id or uuid4(),
        tool_name=tool_name,
        validated_arguments=arguments,
        requested_at=datetime(2026, 7, 27, tzinfo=UTC),
    )


class EchoArguments(StrictModel):
    text: str = Field(min_length=1)


class EchoResult(StrictModel):
    text: str
    truncated: bool = False


class EchoTool:
    name = "echo"
    description = "Return the validated text."
    arguments_model: type[BaseModel] = EchoArguments

    def __init__(self) -> None:
        self.executions = 0

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        assert isinstance(arguments, EchoArguments)
        self.executions += 1
        return EchoResult(text=arguments.text)


class ExplodingTool:
    name = "explode"
    description = "Raise a private implementation error."
    arguments_model: type[BaseModel] = EchoArguments

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        raise RuntimeError("private-exception-detail")


class InvalidOutputTool:
    name = "invalid_output"
    description = "Violate the registered output contract."
    arguments_model: type[BaseModel] = EchoArguments

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        return cast(BaseModel, "not-a-model-private-detail")


def test_registry_lists_sorted_descriptors_and_rejects_duplicate_names() -> None:
    registry = ToolRegistry()
    repository = ReadOnlyRepository(FIXTURE_ROOT)
    registry.register(ReadFileTool(repository))
    registry.register(ListFilesTool(repository))

    descriptors = registry.list_tools()
    list_properties = cast(
        dict[str, JsonValue],
        descriptors[0].argument_schema["properties"],
    )

    assert tuple(item.name for item in descriptors) == ("list_files", "read_file")
    assert descriptors[0].argument_schema["additionalProperties"] is False
    assert "path" in list_properties
    with pytest.raises(ToolRegistrationError, match="already registered"):
        registry.register(ListFilesTool(repository))


@pytest.mark.parametrize("max_output_bytes", [0, -1, True, 1_000_001, "10"])
def test_registry_rejects_invalid_output_limits(max_output_bytes: object) -> None:
    with pytest.raises(ToolRegistrationError, match="max_output_bytes"):
        ToolRegistry(max_output_bytes=max_output_bytes)  # type: ignore[arg-type]


def test_success_is_serializable_and_output_bytes_are_exact() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    call = make_call("echo", {"text": "你好"})

    result = registry.invoke(call)

    assert result.status is ToolResultStatus.SUCCESS
    assert result.payload == {"text": "你好", "truncated": False}
    encoded = json.dumps(
        result.payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert result.output_bytes == len(encoded)
    assert result.truncated is False
    assert ToolExecutionResult.model_validate_json(result.model_dump_json()) == result


def test_unknown_tool_and_invalid_arguments_use_stable_safe_errors() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    unknown = registry.invoke(make_call("missing", {}))
    invalid = registry.invoke(make_call("echo", {"text": "", "private": "do-not-echo"}))

    assert (unknown.error_type, unknown.error_code) == (
        ErrorType.UNKNOWN_TOOL,
        ToolErrorCode.UNKNOWN_TOOL,
    )
    assert (invalid.error_type, invalid.error_code) == (
        ErrorType.INVALID_TOOL_ARGUMENTS,
        ToolErrorCode.INVALID_ARGUMENTS,
    )
    assert "do-not-echo" not in invalid.model_dump_json()


def test_every_first_attempt_claims_call_id_and_duplicate_never_executes() -> None:
    registry = ToolRegistry()
    echo = EchoTool()
    registry.register(echo)
    call_id = uuid4()

    first = registry.invoke(make_call("echo", {"text": "first"}, call_id=call_id))
    duplicate = registry.invoke(make_call("echo", {"text": "second"}, call_id=call_id))

    assert first.status is ToolResultStatus.SUCCESS
    assert duplicate.error_type is ErrorType.DUPLICATE_TOOL_CALL
    assert duplicate.error_code is ToolErrorCode.DUPLICATE_CALL
    assert echo.executions == 1

    unknown_id = uuid4()
    assert registry.invoke(make_call("missing", {}, call_id=unknown_id)).error_code is (
        ToolErrorCode.UNKNOWN_TOOL
    )
    assert (
        registry.invoke(make_call("echo", {"text": "retry"}, call_id=unknown_id)).error_code
        is ToolErrorCode.DUPLICATE_CALL
    )
    assert echo.executions == 1


def test_different_call_ids_may_repeat_identical_arguments() -> None:
    registry = ToolRegistry()
    echo = EchoTool()
    registry.register(echo)

    assert registry.invoke(make_call("echo", {"text": "same"})).status is ToolResultStatus.SUCCESS
    assert registry.invoke(make_call("echo", {"text": "same"})).status is ToolResultStatus.SUCCESS
    assert echo.executions == 2


def test_oversized_output_is_dropped_not_partially_returned() -> None:
    registry = ToolRegistry(max_output_bytes=20)
    registry.register(EchoTool())

    result = registry.invoke(make_call("echo", {"text": "x" * 100}))

    assert result.status is ToolResultStatus.ERROR
    assert result.payload is None
    assert result.output_bytes == 0
    assert result.truncated is True
    assert result.error_code is ToolErrorCode.OUTPUT_TOO_LARGE
    assert "x" * 20 not in result.model_dump_json()


def test_unexpected_exception_is_mapped_without_detail_leakage() -> None:
    registry = ToolRegistry()
    registry.register(ExplodingTool())

    result = registry.invoke(make_call("explode", {"text": "ok"}))

    assert result.error_type is ErrorType.TOOL_EXECUTION_ERROR
    assert result.error_code is ToolErrorCode.TOOL_EXECUTION_ERROR
    assert "private-exception-detail" not in result.model_dump_json()


def test_invalid_tool_output_is_mapped_without_crashing_registry() -> None:
    registry = ToolRegistry()
    registry.register(InvalidOutputTool())

    result = registry.invoke(make_call("invalid_output", {"text": "ok"}))

    assert result.error_type is ErrorType.TOOL_EXECUTION_ERROR
    assert result.error_code is ToolErrorCode.TOOL_EXECUTION_ERROR
    assert "private-detail" not in result.model_dump_json()


def test_repository_errors_preserve_path_or_execution_classification() -> None:
    registry = ToolRegistry()
    repository = ReadOnlyRepository(FIXTURE_ROOT)
    registry.register(ReadFileTool(repository))

    escape = registry.invoke(make_call("read_file", {"path": "../outside"}))
    missing = registry.invoke(make_call("read_file", {"path": "missing.py"}))

    assert (escape.error_type, escape.repository_error_code) == (
        ErrorType.PATH_VIOLATION,
        RepositoryErrorCode.INVALID_PATH,
    )
    assert escape.error_code is ToolErrorCode.REPOSITORY_ERROR
    assert (missing.error_type, missing.repository_error_code) == (
        ErrorType.TOOL_EXECUTION_ERROR,
        RepositoryErrorCode.NOT_FOUND,
    )


def test_read_and_list_tools_preserve_repository_payloads() -> None:
    settings = RepoPilotSettings()
    registry = build_read_only_tool_registry(ReadOnlyRepository(FIXTURE_ROOT), settings)

    names = tuple(descriptor.name for descriptor in registry.list_tools())
    listing = registry.invoke(make_call("list_files", {"path": "pkg", "recursive": False}))
    reading = registry.invoke(
        make_call("read_file", {"path": "pkg/parser.py", "start_line": 4, "end_line": 4})
    )

    assert names == ("inspect_symbol", "list_files", "read_file", "search_code")
    assert listing.payload is not None
    assert len(listing.payload["entries"]) == 3  # type: ignore[arg-type]
    assert reading.payload is not None
    assert reading.payload["content"] == "def parse(raw: str) -> dict[str, str]:\n"


def test_tool_arguments_reject_extra_fields_and_invalid_line_windows() -> None:
    registry = ToolRegistry()
    repository = ReadOnlyRepository(FIXTURE_ROOT)
    registry.register(ReadFileTool(repository))

    extra = registry.invoke(make_call("read_file", {"path": "README.md", "unknown": True}))
    reversed_window = registry.invoke(
        make_call("read_file", {"path": "README.md", "start_line": 3, "end_line": 2})
    )
    boolean_line = registry.invoke(
        make_call("read_file", {"path": "README.md", "start_line": True})
    )

    assert {
        extra.error_code,
        reversed_window.error_code,
        boolean_line.error_code,
    } == {ToolErrorCode.INVALID_ARGUMENTS}
