"""Offline Trace replay and state-chain verification."""

from __future__ import annotations

from uuid import UUID

from repopilot.enums import TraceErrorCode, TraceNode
from repopilot.schemas import AgentState, StrictModel, TaskId
from repopilot.trace import (
    AgentTrace,
    TraceArtifactError,
    TraceEvent,
    verify_trace_integrity,
)

ALLOWED_NEXT_NODES: dict[TraceNode, frozenset[TraceNode]] = {
    TraceNode.INITIALIZE_PLAN: frozenset({TraceNode.DECIDE}),
    TraceNode.DECIDE: frozenset(
        {
            TraceNode.EXECUTE_TOOL,
            TraceNode.FINISH,
            TraceNode.TERMINATE,
        }
    ),
    TraceNode.EXECUTE_TOOL: frozenset({TraceNode.OBSERVE}),
    TraceNode.OBSERVE: frozenset({TraceNode.DECIDE, TraceNode.TERMINATE}),
    TraceNode.FINISH: frozenset(),
    TraceNode.TERMINATE: frozenset(),
}


class TraceReplayResult(StrictModel):
    """Verified result of replaying one Trace without external capabilities."""

    run_id: UUID
    task_id: TaskId
    event_count: int
    final_state: AgentState
    content_sha256: str


def replay_trace(trace: AgentTrace) -> TraceReplayResult:
    """Validate event order, correlations, and exact state continuity."""

    verify_trace_integrity(trace)
    events = trace.events
    if events[0].node is not TraceNode.INITIALIZE_PLAN:
        raise TraceArtifactError(TraceErrorCode.EVENT_ORDER)

    previous: TraceEvent | None = None
    for expected_sequence, event in enumerate(events):
        if event.sequence != expected_sequence:
            raise TraceArtifactError(TraceErrorCode.EVENT_ORDER)
        if (
            event.state_before.run_id != trace.run_id
            or event.state_after.run_id != trace.run_id
            or event.state_before.task.task_id != trace.task_id
            or event.state_after.task.task_id != trace.task_id
        ):
            raise TraceArtifactError(TraceErrorCode.CORRELATION_MISMATCH)
        if previous is not None:
            if event.node not in ALLOWED_NEXT_NODES[previous.node]:
                raise TraceArtifactError(TraceErrorCode.EVENT_ORDER)
            if previous.state_after != event.state_before:
                raise TraceArtifactError(TraceErrorCode.STATE_CHAIN_MISMATCH)
        _verify_event_correlation(event)
        previous = event

    if events[-1].node not in {TraceNode.FINISH, TraceNode.TERMINATE}:
        raise TraceArtifactError(TraceErrorCode.EVENT_ORDER)
    if events[-1].state_after != trace.final_state:
        raise TraceArtifactError(TraceErrorCode.FINAL_STATE_MISMATCH)
    if trace.final_state.termination_reason is None:
        raise TraceArtifactError(TraceErrorCode.FINAL_STATE_MISMATCH)
    return TraceReplayResult(
        run_id=trace.run_id,
        task_id=trace.task_id,
        event_count=len(events),
        final_state=trace.final_state,
        content_sha256=trace.content_sha256,
    )


def _verify_event_correlation(event: TraceEvent) -> None:
    if event.node is TraceNode.DECIDE:
        request = event.provider_request
        response = event.provider_response
        if request is None:
            if event.state_after.termination_reason is None:
                raise TraceArtifactError(TraceErrorCode.CORRELATION_MISMATCH)
            return
        if request.state != event.state_before or request.run_id != event.state_before.run_id:
            raise TraceArtifactError(TraceErrorCode.CORRELATION_MISMATCH)
        if response is not None and response.request_id != request.request_id:
            raise TraceArtifactError(TraceErrorCode.CORRELATION_MISMATCH)
        usage = event.context_usage
        window = request.context_window
        if usage is not None and (
            response is None
            or window is None
            or usage.request_id != request.request_id
            or usage.rendered_prompt_sha256 != window.rendered_prompt_sha256
            or usage.estimated_input_tokens != window.estimated_tokens_after
            or usage.provider_input_tokens != response.usage.input_tokens
            or usage.cached_input_tokens != response.usage.cached_input_tokens
            or usage.cache_write_input_tokens != response.usage.cache_write_input_tokens
            or usage.attempts != response.attempts
        ):
            raise TraceArtifactError(TraceErrorCode.CORRELATION_MISMATCH)
        return

    if event.node is TraceNode.EXECUTE_TOOL:
        call = event.tool_call
        result = event.tool_result
        if (
            call is None
            or result is None
            or event.state_before.pending_tool_call != call
            or result.call_id != call.call_id
            or result.tool_name != call.tool_name
            or event.state_after.pending_tool_result != result
        ):
            raise TraceArtifactError(TraceErrorCode.CORRELATION_MISMATCH)
        return

    if event.node is TraceNode.OBSERVE and (
        event.state_before.pending_tool_call is None
        or event.state_before.pending_tool_result is None
        or event.state_after.pending_tool_call is not None
        or event.state_after.pending_tool_result is not None
    ):
        raise TraceArtifactError(TraceErrorCode.STATE_CHAIN_MISMATCH)
