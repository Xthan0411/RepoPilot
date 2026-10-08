"""LangGraph state channels, reducers, and core-state conversion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, TypedDict
from uuid import UUID

from pydantic import StringConstraints

from repopilot.enums import TerminationReason
from repopilot.schemas import (
    AgentFailure,
    AgentState,
    BudgetState,
    Identifier,
    MemoryLoadAudit,
    MemoryNamespace,
    Observation,
    PatchState,
    PlanStep,
    RepositoryMemoryItem,
    RetrievalHit,
    Sha256,
    StrictModel,
    TaskSpec,
    TestRunResult,
    TestState,
    ToolCall,
    ToolExecutionResult,
    VerifiedEpisodeMemoryItem,
)

ThreadId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"),
]


class ThreadIdentity(StrictModel):
    """Validated checkpoint namespace supplied per graph invocation."""

    thread_id: ThreadId


class StateReductionConflict(ValueError):
    """A partial update would make checkpointed Agent state ambiguous or invalid."""


def merge_working_memory(
    current: tuple[str, ...],
    update: tuple[str, ...],
) -> tuple[str, ...]:
    """Append exact new notes once while preserving first-seen order."""

    merged = list(current)
    seen = set(current)
    for note in update:
        if note not in seen:
            merged.append(note)
            seen.add(note)
    return tuple(merged)


def merge_observations(
    current: tuple[Observation, ...],
    update: tuple[Observation, ...],
) -> tuple[Observation, ...]:
    """Append observations idempotently and reject conflicting call IDs."""

    merged = list(current)
    by_call_id = {item.call_id: item for item in current}
    for observation in update:
        existing = by_call_id.get(observation.call_id)
        if existing is None:
            merged.append(observation)
            by_call_id[observation.call_id] = observation
        elif existing != observation:
            raise StateReductionConflict("observation call ID has conflicting results")
    return tuple(merged)


def merge_consumed_call_ids(
    current: tuple[UUID, ...],
    update: tuple[UUID, ...],
) -> tuple[UUID, ...]:
    """Append consumed call IDs once in deterministic order."""

    merged = list(current)
    seen = set(current)
    for call_id in update:
        if call_id not in seen:
            merged.append(call_id)
            seen.add(call_id)
    return tuple(merged)


def merge_tool_results(
    current: tuple[ToolExecutionResult, ...],
    update: tuple[ToolExecutionResult, ...],
) -> tuple[ToolExecutionResult, ...]:
    """Append tool results idempotently and reject conflicting call IDs."""

    merged = list(current)
    by_call_id = {item.call_id: item for item in current}
    for result in update:
        existing = by_call_id.get(result.call_id)
        if existing is None:
            merged.append(result)
            by_call_id[result.call_id] = result
        elif existing != result:
            raise StateReductionConflict("tool result call ID has conflicting results")
    return tuple(merged)


def merge_failure_history(
    current: tuple[AgentFailure, ...],
    update: tuple[AgentFailure, ...],
) -> tuple[AgentFailure, ...]:
    """Append failures idempotently and reject conflicting decision records."""

    merged = list(current)
    by_key = {(item.decision_index, item.kind): item for item in current}
    for failure in update:
        key = (failure.decision_index, failure.kind)
        existing = by_key.get(key)
        if existing is None:
            merged.append(failure)
            by_key[key] = failure
        elif existing != failure:
            raise StateReductionConflict("failure history contains conflicting records")
    return tuple(merged)


def merge_action_fingerprints(
    current: tuple[str, ...],
    update: tuple[str, ...],
) -> tuple[str, ...]:
    """Append unique canonical action fingerprints in decision order."""

    merged = list(current)
    seen = set(current)
    for fingerprint in update:
        if fingerprint not in seen:
            merged.append(fingerprint)
            seen.add(fingerprint)
    return tuple(merged)


class AgentGraphState(TypedDict):
    """Full checkpointed state with reducers only on append-style channels."""

    task: TaskSpec
    run_id: UUID | None
    repository_summary: str
    memory_namespace: MemoryNamespace | None
    memory_load_audit: MemoryLoadAudit | None
    repository_memory: tuple[RepositoryMemoryItem, ...]
    episodic_memory: tuple[VerifiedEpisodeMemoryItem, ...]
    current_plan: tuple[PlanStep, ...]
    active_step_id: Identifier | None
    retrieved_context: tuple[RetrievalHit, ...]
    working_memory: Annotated[tuple[str, ...], merge_working_memory]
    observations: Annotated[tuple[Observation, ...], merge_observations]
    pending_tool_call: ToolCall | None
    pending_tool_result: ToolExecutionResult | None
    consumed_call_ids: Annotated[tuple[UUID, ...], merge_consumed_call_ids]
    tool_results: Annotated[tuple[ToolExecutionResult, ...], merge_tool_results]
    failure_history: Annotated[tuple[AgentFailure, ...], merge_failure_history]
    action_fingerprints: Annotated[tuple[Sha256, ...], merge_action_fingerprints]
    replan_count: int
    decision_count: int
    finish_summary: str | None
    patch_state: PatchState
    test_state: TestState
    baseline_test_result: TestRunResult | None
    budget: BudgetState
    termination_reason: TerminationReason | None


class AgentGraphUpdate(TypedDict, total=False):
    """Allowed partial graph update; task identity is intentionally immutable."""

    repository_summary: str
    current_plan: tuple[PlanStep, ...]
    active_step_id: Identifier | None
    retrieved_context: tuple[RetrievalHit, ...]
    working_memory: tuple[str, ...]
    observations: tuple[Observation, ...]
    pending_tool_call: ToolCall | None
    pending_tool_result: ToolExecutionResult | None
    consumed_call_ids: tuple[UUID, ...]
    tool_results: tuple[ToolExecutionResult, ...]
    failure_history: tuple[AgentFailure, ...]
    action_fingerprints: tuple[Sha256, ...]
    replan_count: int
    decision_count: int
    finish_summary: str | None
    patch_state: PatchState
    test_state: TestState
    baseline_test_result: TestRunResult | None
    budget: BudgetState
    termination_reason: TerminationReason | None


ALLOWED_UPDATE_FIELDS = frozenset(AgentGraphUpdate.__annotations__)


def to_graph_state(state: AgentState) -> AgentGraphState:
    """Convert the validated core model to LangGraph channel values."""

    return AgentGraphState(
        task=state.task,
        run_id=state.run_id,
        repository_summary=state.repository_summary,
        memory_namespace=state.memory_namespace,
        memory_load_audit=state.memory_load_audit,
        repository_memory=state.repository_memory,
        episodic_memory=state.episodic_memory,
        current_plan=state.current_plan,
        active_step_id=state.active_step_id,
        retrieved_context=state.retrieved_context,
        working_memory=state.working_memory,
        observations=state.observations,
        pending_tool_call=state.pending_tool_call,
        pending_tool_result=state.pending_tool_result,
        consumed_call_ids=state.consumed_call_ids,
        tool_results=state.tool_results,
        failure_history=state.failure_history,
        action_fingerprints=state.action_fingerprints,
        replan_count=state.replan_count,
        decision_count=state.decision_count,
        finish_summary=state.finish_summary,
        patch_state=state.patch_state,
        test_state=state.test_state,
        baseline_test_result=state.baseline_test_result,
        budget=state.budget,
        termination_reason=state.termination_reason,
    )


def from_graph_state(values: Mapping[str, object]) -> AgentState:
    """Revalidate a checkpoint before returning it to application code."""

    return AgentState.model_validate(values)


def apply_state_update(
    current: AgentState,
    update: AgentGraphUpdate,
) -> AgentState:
    """Preview reducer behavior and reject invalid updates before checkpointing."""

    raw_update: dict[str, object] = dict(update)
    if not set(raw_update).issubset(ALLOWED_UPDATE_FIELDS):
        raise StateReductionConflict("state update contains a forbidden field")

    payload: dict[str, object] = current.model_dump(mode="python")
    if "working_memory" in raw_update:
        working_memory = raw_update.pop("working_memory")
        if not isinstance(working_memory, tuple):
            raise StateReductionConflict("working_memory update must be a tuple")
        payload["working_memory"] = merge_working_memory(
            current.working_memory,
            working_memory,
        )
    if "observations" in raw_update:
        observations = raw_update.pop("observations")
        if not isinstance(observations, tuple) or not all(
            isinstance(item, Observation) for item in observations
        ):
            raise StateReductionConflict("observations update must contain observations")
        payload["observations"] = merge_observations(
            current.observations,
            observations,
        )
    if "consumed_call_ids" in raw_update:
        consumed_ids = raw_update.pop("consumed_call_ids")
        if not isinstance(consumed_ids, tuple) or not all(
            isinstance(item, UUID) for item in consumed_ids
        ):
            raise StateReductionConflict("consumed_call_ids update must contain UUIDs")
        payload["consumed_call_ids"] = merge_consumed_call_ids(
            current.consumed_call_ids,
            consumed_ids,
        )
    if "tool_results" in raw_update:
        tool_results = raw_update.pop("tool_results")
        if not isinstance(tool_results, tuple) or not all(
            isinstance(item, ToolExecutionResult) for item in tool_results
        ):
            raise StateReductionConflict("tool_results update must contain tool results")
        payload["tool_results"] = merge_tool_results(
            current.tool_results,
            tool_results,
        )
    if "failure_history" in raw_update:
        failures = raw_update.pop("failure_history")
        if not isinstance(failures, tuple) or not all(
            isinstance(item, AgentFailure) for item in failures
        ):
            raise StateReductionConflict("failure_history update must contain failures")
        payload["failure_history"] = merge_failure_history(
            current.failure_history,
            failures,
        )
    if "action_fingerprints" in raw_update:
        fingerprints = raw_update.pop("action_fingerprints")
        if not isinstance(fingerprints, tuple) or not all(
            isinstance(item, str) for item in fingerprints
        ):
            raise StateReductionConflict("action_fingerprints update must contain strings")
        payload["action_fingerprints"] = merge_action_fingerprints(
            current.action_fingerprints,
            fingerprints,
        )
    payload.update(raw_update)
    try:
        return AgentState.model_validate(payload)
    except ValueError:
        raise StateReductionConflict("state update violates AgentState invariants") from None
