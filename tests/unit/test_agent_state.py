"""Agent graph state conversion, reducers, and invariant tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from repopilot.agent import (
    AgentGraphUpdate,
    StateReductionConflict,
    ThreadIdentity,
    apply_state_update,
    from_graph_state,
    merge_consumed_call_ids,
    merge_observations,
    merge_tool_results,
    merge_working_memory,
    to_graph_state,
)
from repopilot.enums import TerminationReason, ToolResultStatus
from repopilot.schemas import AgentState, Observation, ToolExecutionResult

FIXTURE = Path(__file__).parents[1] / "fixtures" / "valid_agent_state.json"


def load_state() -> AgentState:
    return AgentState.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def test_core_and_graph_state_round_trip_without_gold_fields() -> None:
    state = load_state()

    graph_state = to_graph_state(state)
    restored = from_graph_state(graph_state)

    assert restored == state
    rendered = restored.model_dump_json()
    assert "expected_patch" not in rendered
    assert "gold_files" not in rendered


def test_working_memory_and_consumed_ids_append_uniquely_in_order() -> None:
    first_id = UUID("11111111-1111-4111-8111-111111111111")
    second_id = UUID("22222222-2222-4222-8222-222222222222")

    notes = merge_working_memory(("first",), ("second", "first", "third"))
    call_ids = merge_consumed_call_ids((first_id,), (second_id, first_id))

    assert notes == ("first", "second", "third")
    assert call_ids == (first_id, second_id)


def test_observation_reducer_is_idempotent_and_rejects_conflicts() -> None:
    observation = load_state().observations[0]
    conflicting = observation.model_copy(update={"duration_ms": observation.duration_ms + 1})

    assert merge_observations((observation,), (observation,)) == (observation,)
    with pytest.raises(StateReductionConflict, match="conflicting"):
        merge_observations((observation,), (conflicting,))


def test_tool_result_reducer_is_idempotent_and_rejects_conflicts() -> None:
    call_id = uuid4()
    result = ToolExecutionResult(
        call_id=call_id,
        tool_name="read_file",
        status=ToolResultStatus.SUCCESS,
        payload={"content": "first"},
        output_bytes=19,
    )
    conflicting = result.model_copy(
        update={
            "payload": {"content": "different"},
            "output_bytes": 23,
        }
    )

    assert merge_tool_results((result,), (result,)) == (result,)
    with pytest.raises(StateReductionConflict, match="conflicting"):
        merge_tool_results((result,), (conflicting,))


def test_partial_update_preserves_unmentioned_state_and_applies_reducers() -> None:
    state = load_state()
    call_id = uuid4()
    update: AgentGraphUpdate = {
        "repository_summary": "Updated summary.",
        "working_memory": ("A new fact.", state.working_memory[0]),
        "consumed_call_ids": (call_id,),
        "termination_reason": TerminationReason.AGENT_STOPPED,
    }

    updated = apply_state_update(state, update)

    assert updated.task == state.task
    assert updated.current_plan == state.current_plan
    assert updated.observations == state.observations
    assert updated.patch_state == state.patch_state
    assert updated.test_state == state.test_state
    assert updated.budget == state.budget
    assert updated.repository_summary == "Updated summary."
    assert updated.working_memory == (*state.working_memory, "A new fact.")
    assert updated.consumed_call_ids == (call_id,)
    assert updated.termination_reason is TerminationReason.AGENT_STOPPED


def test_partial_update_rejects_forbidden_or_invalid_state() -> None:
    state = load_state()
    forbidden = cast(AgentGraphUpdate, {"task": state.task})
    invalid_plan: AgentGraphUpdate = {"current_plan": ()}

    with pytest.raises(StateReductionConflict, match="forbidden"):
        apply_state_update(state, forbidden)
    with pytest.raises(StateReductionConflict, match="invariants"):
        apply_state_update(state, invalid_plan)
    with pytest.raises(StateReductionConflict, match="tuple"):
        apply_state_update(state, cast(AgentGraphUpdate, {"working_memory": []}))


@pytest.mark.parametrize(
    "thread_id",
    [
        "",
        " ",
        "-starts-with-punctuation",
        "contains/slash",
        "contains\\backslash",
        "contains space",
        "nul\x00thread",
        "x" * 129,
    ],
)
def test_thread_identity_rejects_noncanonical_values(thread_id: str) -> None:
    with pytest.raises(ValidationError):
        ThreadIdentity(thread_id=thread_id)


def test_agent_state_rejects_duplicate_or_inconsistent_call_records() -> None:
    state = load_state()
    observation = state.observations[0]

    with pytest.raises(ValidationError, match="working memory entries"):
        AgentState.model_validate(
            {
                **state.model_dump(mode="python"),
                "working_memory": ("duplicate", "duplicate"),
            }
        )
    with pytest.raises(ValidationError, match="observation call IDs"):
        AgentState.model_validate(
            {
                **state.model_dump(mode="python"),
                "observations": (observation, observation),
            }
        )
    with pytest.raises(ValidationError, match="consumed call IDs"):
        AgentState.model_validate(
            {
                **state.model_dump(mode="python"),
                "consumed_call_ids": (observation.call_id, observation.call_id),
            }
        )
    with pytest.raises(ValidationError, match="pending tool call"):
        AgentState.model_validate(
            {
                **state.model_dump(mode="python"),
                "pending_tool_call": {
                    "call_id": observation.call_id,
                    "tool_name": "read_file",
                    "validated_arguments": {"path": "README.md"},
                    "requested_at": observation.started_at,
                },
                "consumed_call_ids": (observation.call_id,),
            }
        )


def test_agent_state_requires_consumption_and_run_identity_for_tool_results() -> None:
    state = load_state()
    result = ToolExecutionResult(
        call_id=uuid4(),
        tool_name="read_file",
        status=ToolResultStatus.SUCCESS,
        payload={"content": "evidence"},
        output_bytes=22,
    )

    with pytest.raises(ValidationError, match="consumed call IDs"):
        AgentState.model_validate(
            {
                **state.model_dump(mode="python"),
                "run_id": uuid4(),
                "tool_results": (result,),
            }
        )
    with pytest.raises(ValidationError, match="requires run_id"):
        AgentState.model_validate(
            {
                **state.model_dump(mode="python"),
                "tool_results": (result,),
                "consumed_call_ids": (result.call_id,),
            }
        )


def test_observation_runtime_type_is_preserved_through_graph_conversion() -> None:
    restored = from_graph_state(to_graph_state(load_state()))

    assert isinstance(restored.observations[0], Observation)
