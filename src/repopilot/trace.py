"""Versioned Agent Trace contracts, recording, hashing, and artifact I/O."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from pathlib import Path
from typing import Literal, Self, cast
from uuid import UUID

from pydantic import Field, JsonValue, ValidationError, model_validator
from pydantic_core import to_jsonable_python

from repopilot.context import ContextUsageLedger
from repopilot.enums import ErrorType, ProviderErrorCode, TraceErrorCode, TraceNode
from repopilot.providers.base import ProviderRequest, ProviderResponse
from repopilot.schemas import (
    AgentState,
    NonEmptyText,
    StrictModel,
    TaskId,
    ToolCall,
    ToolExecutionResult,
)

TRACE_SCHEMA_VERSION: Literal[1] = 1
TRACE_ERROR_MESSAGES = {
    TraceErrorCode.IO_ERROR: "Trace artifact could not be read or written",
    TraceErrorCode.INVALID_DOCUMENT: "Trace document is invalid",
    TraceErrorCode.UNSUPPORTED_VERSION: "Trace schema version is unsupported",
    TraceErrorCode.INTEGRITY_MISMATCH: "Trace content digest does not match",
    TraceErrorCode.EVENT_ORDER: "Trace event order is invalid",
    TraceErrorCode.STATE_CHAIN_MISMATCH: "Trace state chain is inconsistent",
    TraceErrorCode.CORRELATION_MISMATCH: "Trace request or tool correlation is invalid",
    TraceErrorCode.FINAL_STATE_MISMATCH: "Trace final state does not match replay",
}


class TraceArtifactError(RuntimeError):
    """Safe typed failure from Trace loading, writing, or replay."""

    def __init__(self, code: TraceErrorCode) -> None:
        super().__init__(TRACE_ERROR_MESSAGES[code])
        self.code = code


class TraceEvent(StrictModel):
    """One complete graph-node transition plus its external interaction."""

    sequence: int = Field(ge=0)
    node: TraceNode
    state_before: AgentState
    state_after: AgentState
    provider_request: ProviderRequest | None = None
    provider_response: ProviderResponse | None = None
    provider_error_type: ErrorType | None = None
    provider_error_code: ProviderErrorCode | None = None
    context_usage: ContextUsageLedger | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolExecutionResult | None = None

    @model_validator(mode="after")
    def validate_node_payload(self) -> Self:
        provider_values = (
            self.provider_request,
            self.provider_response,
            self.provider_error_type,
            self.provider_error_code,
            self.context_usage,
        )
        tool_values = (self.tool_call, self.tool_result)
        if self.node is TraceNode.DECIDE:
            if any(value is not None for value in tool_values):
                raise ValueError("decide events cannot contain tool execution payload")
            if self.provider_response is not None and self.provider_request is None:
                raise ValueError("provider response requires its request")
            if self.provider_error_type is not None and self.provider_request is None:
                raise ValueError("provider error requires its request")
            if (self.provider_error_type is None) != (self.provider_error_code is None):
                raise ValueError("provider error type and code must appear together")
            request = self.provider_request
            response = self.provider_response
            usage = self.context_usage
            if (
                request is not None
                and response is not None
                and request.context_window is not None
                and usage is None
            ):
                raise ValueError("context-aware Provider response requires usage ledger")
            if usage is not None:
                if request is None or response is None or request.context_window is None:
                    raise ValueError("context usage requires a complete context-aware exchange")
                window = request.context_window
                if (
                    usage.rendered_prompt_sha256 != window.rendered_prompt_sha256
                    or usage.estimator_version != window.estimator_version
                    or usage.estimated_input_tokens != window.estimated_tokens_after
                    or usage.provider_input_tokens != response.usage.input_tokens
                    or usage.cached_input_tokens != response.usage.cached_input_tokens
                    or usage.cache_write_input_tokens != response.usage.cache_write_input_tokens
                    or usage.attempts != response.attempts
                ):
                    raise ValueError("context usage does not match Provider exchange")
        elif self.node is TraceNode.EXECUTE_TOOL:
            if any(value is not None for value in provider_values):
                raise ValueError("execute events cannot contain Provider payload")
            if self.tool_call is None or self.tool_result is None:
                raise ValueError("execute events require call and result")
        elif any(value is not None for value in (*provider_values, *tool_values)):
            raise ValueError("non-I/O node cannot contain external interaction payload")
        return self


class AgentTrace(StrictModel):
    """Self-contained, integrity-protected Trace for one Agent graph run."""

    schema_version: Literal[1] = TRACE_SCHEMA_VERSION
    run_id: UUID
    task_id: TaskId
    thread_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    provider_name: NonEmptyText
    settings: dict[str, JsonValue]
    events: tuple[TraceEvent, ...] = Field(min_length=1)
    final_state: AgentState
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def build(
        cls,
        *,
        run_id: UUID,
        task_id: str,
        thread_id: str,
        provider_name: str,
        settings: dict[str, JsonValue],
        events: tuple[TraceEvent, ...],
        final_state: AgentState,
    ) -> Self:
        payload: dict[str, object] = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "run_id": run_id,
            "task_id": task_id,
            "thread_id": thread_id,
            "provider_name": provider_name,
            "settings": settings,
            "events": events,
            "final_state": final_state,
        }
        return cls(
            schema_version=TRACE_SCHEMA_VERSION,
            run_id=run_id,
            task_id=task_id,
            thread_id=thread_id,
            provider_name=provider_name,
            settings=settings,
            events=events,
            final_state=final_state,
            content_sha256=trace_content_sha256(payload),
        )


def trace_content_sha256(payload: dict[str, object]) -> str:
    """Hash canonical JSON content excluding the digest field itself."""

    encoded = json.dumps(
        _remove_legacy_optional_nulls(_json_compatible(payload)),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_trace_integrity(trace: AgentTrace) -> None:
    """Reject any content that no longer matches its recorded digest."""

    payload = trace.model_dump(mode="python", exclude={"content_sha256"})
    if trace_content_sha256(payload) != trace.content_sha256:
        raise TraceArtifactError(TraceErrorCode.INTEGRITY_MISMATCH)


def write_trace(path: Path, trace: AgentTrace) -> None:
    """Atomically write canonical, deterministic Trace JSON."""

    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(f"{trace.model_dump_json(indent=2)}\n", encoding="utf-8")
        temporary.replace(path)
    except OSError:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise TraceArtifactError(TraceErrorCode.IO_ERROR) from None


def read_trace(path: Path) -> AgentTrace:
    """Load one supported Trace without echoing rejected file content."""

    try:
        serialized = path.read_text(encoding="utf-8")
    except OSError:
        raise TraceArtifactError(TraceErrorCode.IO_ERROR) from None
    try:
        raw = json.loads(serialized)
    except (json.JSONDecodeError, UnicodeError):
        raise TraceArtifactError(TraceErrorCode.INVALID_DOCUMENT) from None
    if not isinstance(raw, dict):
        raise TraceArtifactError(TraceErrorCode.INVALID_DOCUMENT)
    if raw.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise TraceArtifactError(TraceErrorCode.UNSUPPORTED_VERSION)
    try:
        trace = AgentTrace.model_validate(raw)
    except ValidationError:
        raise TraceArtifactError(TraceErrorCode.INVALID_DOCUMENT) from None
    verify_trace_integrity(trace)
    return trace


def _json_compatible(payload: object) -> JsonValue:
    return cast(JsonValue, to_jsonable_python(payload))


def _remove_legacy_optional_nulls(value: JsonValue) -> JsonValue:
    """Keep schema-v1 hashes compatible across additive Context fields."""

    if isinstance(value, list):
        return [_remove_legacy_optional_nulls(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _remove_legacy_optional_nulls(item)
            for key, item in value.items()
            if not (
                (
                    key
                    in {
                        "context_window",
                        "context_usage",
                        "memory_load_audit",
                        "memory_namespace",
                        "policy",
                        "replacement_item_id",
                    }
                    and item is None
                )
                or (
                    key
                    in {
                        "budget_usages",
                        "episodic_memory",
                        "repository_memory",
                        "transforms",
                    }
                    and item == []
                )
            )
        }
    return value
