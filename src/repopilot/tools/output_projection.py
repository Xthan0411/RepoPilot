"""Persist large read/search results and expose bounded, usable source previews."""

from __future__ import annotations

import json
from typing import cast

from pydantic import JsonValue

from repopilot.context import ContextBudgetPolicy, estimate_utf8_tokens
from repopilot.enums import ToolResultStatus
from repopilot.sandbox.artifacts import SandboxArtifactStore
from repopilot.schemas import ArtifactRef, ToolCall, ToolExecutionResult
from repopilot.tools.registry import ToolOutputProjectionError
from repopilot.tools.repository import FileReadResult
from repopilot.tools.search import SearchCodeResult

OUTPUT_PROJECTION_VERSION = "source-tool-preview-v1"
_PREVIEW_TOKEN_LIMIT = 1_500


class RepositoryToolOutputProjector:
    """Keep complete original output in immutable artifacts, not in the model window."""

    def __init__(self, store: SandboxArtifactStore) -> None:
        self._store = store
        self._inline_limit = ContextBudgetPolicy().max_inline_tool_tokens

    def __call__(
        self,
        call: ToolCall,
        payload: dict[str, JsonValue],
        encoded: bytes,
    ) -> dict[str, JsonValue]:
        if call.tool_name not in {"read_file", "search_code"}:
            return payload
        if _envelope_tokens(call, payload) <= self._inline_limit:
            return payload
        # The Registry enforces its existing original-output hard cap before this hook.
        reference = self._store.persist(call.call_id, "read", encoded)
        if call.tool_name == "read_file":
            return self._read_preview(call, FileReadResult.model_validate(payload), reference)
        return self._search_preview(call, SearchCodeResult.model_validate(payload), reference)

    def _read_preview(
        self,
        call: ToolCall,
        original: FileReadResult,
        reference: ArtifactRef,
    ) -> dict[str, JsonValue]:
        lines = original.content.splitlines(keepends=True)
        if original.truncated and lines and not lines[-1].endswith(("\n", "\r")):
            # A byte-limited partial source line must be re-read, never silently skipped.
            lines.pop()
        accepted: dict[str, JsonValue] | None = None
        for count in range(1, len(lines) + 1):
            content = "".join(lines[:count])
            end_line = original.start_line + count - 1
            remaining = count < len(original.content.splitlines()) or original.truncated
            value = FileReadResult(
                relative_path=original.relative_path,
                start_line=original.start_line,
                end_line=end_line,
                content=content,
                file_size_bytes=original.file_size_bytes,
                content_bytes=len(content.encode("utf-8")),
                truncated=remaining,
                next_start_line=end_line + 1 if remaining else None,
            )
            candidate = cast(dict[str, JsonValue], value.model_dump(mode="json"))
            candidate["output_projection"] = {
                "version": OUTPUT_PROJECTION_VERSION,
                "artifact": reference.model_dump(mode="json"),
                "original_start_line": original.start_line,
                "original_end_line": original.end_line,
                "source_partial_line_omitted": len(lines) < len(original.content.splitlines()),
                "continuation": (
                    "Use read_file with this path and next_start_line; "
                    "use an explicit end_line for a smaller region."
                ),
            }
            if _envelope_tokens(call, candidate) > _PREVIEW_TOKEN_LIMIT:
                break
            accepted = candidate
        if accepted is None:
            # No misleading empty-source success or incorrect next-line cursor.
            raise ToolOutputProjectionError
        return accepted

    def _search_preview(
        self,
        call: ToolCall,
        original: SearchCodeResult,
        reference: ArtifactRef,
    ) -> dict[str, JsonValue]:
        accepted: dict[str, JsonValue] | None = None
        # Skipped paths are retained in the full artifact; they must not exhaust the preview.
        for count in range(len(original.matches) + 1):
            value = SearchCodeResult(
                query=original.query,
                path=original.path,
                matches=original.matches[:count],
                files_scanned=original.files_scanned,
                skipped_files=(),
                truncated=original.truncated
                or count < len(original.matches)
                or bool(original.skipped_files),
            )
            candidate = cast(dict[str, JsonValue], value.model_dump(mode="json"))
            candidate["output_projection"] = {
                "version": OUTPUT_PROJECTION_VERSION,
                "artifact": reference.model_dump(mode="json"),
                "original_returned_match_count": len(original.matches),
                "omitted_match_count": len(original.matches) - count,
                "omitted_skipped_file_count": len(original.skipped_files),
                "continuation": (
                    "Read a listed path at its matching line with read_file, "
                    "or narrow search_code path/query. "
                    "The artifact is audit evidence, not a repository path."
                ),
            }
            if _envelope_tokens(call, candidate) > _PREVIEW_TOKEN_LIMIT:
                break
            accepted = candidate
        if accepted is None or (original.matches and not accepted.get("matches")):
            raise ToolOutputProjectionError
        return accepted


def _envelope_tokens(call: ToolCall, payload: dict[str, JsonValue]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    result = ToolExecutionResult(
        call_id=call.call_id,
        tool_name=call.tool_name,
        status=ToolResultStatus.SUCCESS,
        payload=payload,
        output_bytes=len(encoded.encode("utf-8")),
        truncated=payload.get("truncated") is True,
    )
    return estimate_utf8_tokens(
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
