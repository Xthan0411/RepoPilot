"""In-process collection of complete Agent node transitions."""

from __future__ import annotations

from uuid import UUID

from repopilot.agent.state import (
    AgentGraphState,
    AgentGraphUpdate,
    apply_state_update,
    from_graph_state,
)
from repopilot.context import ContextUsageLedger
from repopilot.enums import ErrorType, ProviderErrorCode, TraceNode
from repopilot.providers.base import ProviderRequest, ProviderResponse
from repopilot.schemas import ToolCall, ToolExecutionResult
from repopilot.trace import TraceEvent


class TraceRecorder:
    """Collect deterministic node transitions keyed by run ID."""

    def __init__(self) -> None:
        self._events: dict[UUID, list[TraceEvent]] = {}

    def record(
        self,
        *,
        node: TraceNode,
        state: AgentGraphState,
        update: AgentGraphUpdate,
        provider_request: ProviderRequest | None = None,
        provider_response: ProviderResponse | None = None,
        provider_error_type: ErrorType | None = None,
        provider_error_code: ProviderErrorCode | None = None,
        tool_call: ToolCall | None = None,
        tool_result: ToolExecutionResult | None = None,
    ) -> AgentGraphUpdate:
        """Append one transition and return its unchanged graph update."""

        before = from_graph_state(state)
        if before.run_id is None:
            raise ValueError("Trace events require run_id")
        after = apply_state_update(before, update)
        context_usage = _context_usage(provider_request, provider_response)
        events = self._events.setdefault(before.run_id, [])
        events.append(
            TraceEvent(
                sequence=len(events),
                node=node,
                state_before=before,
                state_after=after,
                provider_request=provider_request,
                provider_response=provider_response,
                provider_error_type=provider_error_type,
                provider_error_code=provider_error_code,
                context_usage=context_usage,
                tool_call=tool_call,
                tool_result=tool_result,
            )
        )
        return update

    def events(self, run_id: UUID) -> tuple[TraceEvent, ...]:
        """Return an immutable event sequence for one run."""

        return tuple(self._events.get(run_id, ()))

    def restore(self, run_id: UUID, events: tuple[TraceEvent, ...]) -> None:
        """Install one validated prefix exactly once before graph recovery."""

        if not events:
            raise ValueError("Trace prefix must not be empty")
        previous = None
        for expected_sequence, event in enumerate(events):
            if (
                event.sequence != expected_sequence
                or event.state_before.run_id != run_id
                or event.state_after.run_id != run_id
            ):
                raise ValueError("Trace prefix namespace or sequence is invalid")
            if previous is not None and event.state_before != previous:
                raise ValueError("Trace prefix state chain is invalid")
            previous = event.state_after
        existing = self._events.get(run_id)
        if existing is not None and tuple(existing) != events:
            raise ValueError("Trace recorder already contains a different prefix")
        self._events[run_id] = list(events)


def _context_usage(
    request: ProviderRequest | None,
    response: ProviderResponse | None,
) -> ContextUsageLedger | None:
    if request is None or response is None or request.context_window is None:
        return None
    window = request.context_window
    usage = response.usage
    return ContextUsageLedger(
        request_id=request.request_id,
        rendered_prompt_sha256=window.rendered_prompt_sha256,
        estimator_version=window.estimator_version,
        estimated_input_tokens=window.estimated_tokens_after,
        provider_input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        estimate_delta_tokens=usage.input_tokens - window.estimated_tokens_after,
        attempts=response.attempts,
    )
