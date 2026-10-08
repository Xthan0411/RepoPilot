"""Built-in wrappers for contained repository listing and reading."""

from __future__ import annotations

from pydantic import BaseModel, Field, StrictBool, model_validator

from repopilot.config import RepoPilotSettings
from repopilot.schemas import StrictModel, ToolCall
from repopilot.tools.registry import ToolRegistry
from repopilot.tools.repository import ReadOnlyRepository
from repopilot.tools.search import SearchCodeTool
from repopilot.tools.symbols import InspectSymbolTool


class ListFilesArguments(StrictModel):
    """Arguments for deterministic directory listing."""

    path: str = Field(
        default=".",
        min_length=1,
        description="Canonical repository-relative path; use '.' for the repository root.",
    )
    recursive: StrictBool = True


class ReadFileArguments(StrictModel):
    """Arguments for a bounded UTF-8 source-file read."""

    path: str = Field(
        min_length=1,
        description="Canonical repository-relative file path; never use an empty string.",
    )
    start_line: int = Field(default=1, gt=0, strict=True)
    end_line: int | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode="after")
    def validate_line_window(self) -> ReadFileArguments:
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ListFilesTool:
    """Expose :class:`ReadOnlyRepository.list_directory` through the Registry."""

    name = "list_files"
    description = "List bounded repository entries beneath a canonical relative path."
    arguments_model: type[BaseModel] = ListFilesArguments

    def __init__(self, repository: ReadOnlyRepository) -> None:
        self._repository = repository

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        if not isinstance(arguments, ListFilesArguments):
            raise TypeError("unexpected argument model")
        return self._repository.list_directory(
            arguments.path,
            recursive=arguments.recursive,
        )


class ReadFileTool:
    """Expose :class:`ReadOnlyRepository.read_file` through the Registry."""

    name = "read_file"
    description = "Read a bounded UTF-8 line window from one regular repository file."
    arguments_model: type[BaseModel] = ReadFileArguments

    def __init__(self, repository: ReadOnlyRepository) -> None:
        self._repository = repository

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        if not isinstance(arguments, ReadFileArguments):
            raise TypeError("unexpected argument model")
        return self._repository.read_file(
            arguments.path,
            start_line=arguments.start_line,
            end_line=arguments.end_line,
        )


def build_read_only_tool_registry(
    repository: ReadOnlyRepository,
    settings: RepoPilotSettings,
) -> ToolRegistry:
    """Register the complete P1-S2 read-only capability set."""

    registry = ToolRegistry.from_settings(settings)
    registry.register(ListFilesTool(repository))
    registry.register(ReadFileTool(repository))
    registry.register(SearchCodeTool.from_settings(repository, settings))
    registry.register(InspectSymbolTool.from_settings(repository, settings))
    return registry
