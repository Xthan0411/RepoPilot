"""Safe, typed read-only capabilities exposed to the Agent kernel."""

from repopilot.schemas import ToolExecutionResult
from repopilot.tools.builtin import (
    ListFilesArguments,
    ListFilesTool,
    ReadFileArguments,
    ReadFileTool,
    build_read_only_tool_registry,
)
from repopilot.tools.patch import (
    ApplyPatchArguments,
    ApplyPatchTool,
    DiffInspectionResult,
    InspectDiffArguments,
    InspectDiffTool,
    ParsedUnifiedDiff,
    PatchApplicationResult,
    PatchLimits,
    PatchToolSession,
    PatchValidationError,
    UnifiedDiffSummary,
    parse_unified_diff,
)
from repopilot.tools.registry import (
    RegisteredTool,
    ToolArgumentError,
    ToolDescriptor,
    ToolRegistrationError,
    ToolRegistry,
)
from repopilot.tools.repository import (
    DirectoryListing,
    FileReadResult,
    ReadOnlyRepository,
    RepositoryAccessError,
    RepositoryAccessPolicy,
    RepositoryEntry,
    RepositoryTextFile,
)
from repopilot.tools.sandbox import (
    RunTestsArguments,
    RunTestsResult,
    RunTestsTool,
)
from repopilot.tools.search import (
    SearchCodeArguments,
    SearchCodeResult,
    SearchCodeTool,
    SearchMatch,
    SkippedSourceFile,
)
from repopilot.tools.symbols import (
    InspectSymbolArguments,
    InspectSymbolResult,
    InspectSymbolTool,
    SymbolLocation,
)
from repopilot.tools.tests import (
    advance_test_state,
    build_reproduction_result,
    interpret_test_observation,
)

__all__ = [
    "ApplyPatchArguments",
    "ApplyPatchTool",
    "DiffInspectionResult",
    "DirectoryListing",
    "FileReadResult",
    "InspectDiffArguments",
    "InspectDiffTool",
    "InspectSymbolArguments",
    "InspectSymbolResult",
    "InspectSymbolTool",
    "ListFilesArguments",
    "ListFilesTool",
    "ParsedUnifiedDiff",
    "PatchApplicationResult",
    "PatchLimits",
    "PatchToolSession",
    "PatchValidationError",
    "ReadFileArguments",
    "ReadFileTool",
    "ReadOnlyRepository",
    "RegisteredTool",
    "RepositoryAccessError",
    "RepositoryAccessPolicy",
    "RepositoryEntry",
    "RepositoryTextFile",
    "RunTestsArguments",
    "RunTestsResult",
    "RunTestsTool",
    "SearchCodeArguments",
    "SearchCodeResult",
    "SearchCodeTool",
    "SearchMatch",
    "SkippedSourceFile",
    "SymbolLocation",
    "ToolArgumentError",
    "ToolDescriptor",
    "ToolExecutionResult",
    "ToolRegistrationError",
    "ToolRegistry",
    "UnifiedDiffSummary",
    "advance_test_state",
    "build_read_only_tool_registry",
    "build_reproduction_result",
    "interpret_test_observation",
    "parse_unified_diff",
]
