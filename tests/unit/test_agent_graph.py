"""Thread-scoped LangGraph checkpoint creation, isolation, and recovery tests."""

from __future__ import annotations

import hashlib
import socket
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from repopilot.agent import (
    AgentGraphUpdate,
    AgentStateGraph,
    AgentStateGraphError,
)
from repopilot.enums import (
    AgentStateErrorCode,
    DatasetSplit,
    ErrorType,
    MemoryAccessMode,
    RepositoryMemoryLookupStatus,
    VerifiedEpisodeLookupStatus,
)
from repopilot.enums import (
    TestRunCategory as RepoTestRunCategory,
)
from repopilot.enums import (
    TestStatus as RepoTestStatus,
)
from repopilot.schemas import (
    AgentState,
    MemoryLoadAudit,
    MemoryNamespace,
    RepositoryMemoryItem,
    ToolCall,
    VerifiedEpisodeMemoryItem,
)
from repopilot.schemas import (
    TestRunResult as RepoTestRunResult,
)
from repopilot.schemas import (
    TestState as RepoTestState,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "valid_agent_state.json"


def load_state(*, task_id: str = "fixture-001") -> AgentState:
    state = AgentState.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    if task_id == state.task.task_id:
        return state
    return AgentState.model_validate(
        {
            **state.model_dump(mode="python"),
            "task": {
                **state.task.model_dump(mode="python"),
                "task_id": task_id,
            },
        }
    )


def assert_graph_error(
    captured: pytest.ExceptionInfo[AgentStateGraphError],
    code: AgentStateErrorCode,
    error_type: ErrorType = ErrorType.VALIDATION_ERROR,
) -> None:
    assert captured.value.code is code
    assert captured.value.error_type is error_type


def test_start_and_load_round_trip_complete_state() -> None:
    graph = AgentStateGraph()
    state = load_state()

    started = graph.start("run-001", state)
    loaded = graph.load("run-001")

    assert started == state
    assert loaded == state
    assert isinstance(graph.checkpointer, InMemorySaver)


def test_checkpoint_round_trips_repository_memory_projection() -> None:
    original = load_state()
    content = '{"memory_version":"repository-map-v1","source_count":3}'
    memory = RepositoryMemoryItem(
        record_id="rmem_" + ("a" * 64),
        repository_id="repo_agent_graph",
        repository_commit=original.task.base_commit,
        repository_content_sha256="b" * 64,
        kind="repository_map",
        confirmation="manifest_derived",
        producer_version="repository-map-v1",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        evidence_count=1,
        evidence_sha256="c" * 64,
        source_record_sha256="d" * 64,
    )
    state = original.model_copy(update={"repository_memory": (memory,)})
    graph = AgentStateGraph()

    graph.start("repository-memory", state)

    assert graph.load("repository-memory").repository_memory == (memory,)


def test_checkpoint_round_trips_namespaced_verified_episode_projection() -> None:
    original = load_state()
    created_at = datetime(2026, 8, 6, tzinfo=UTC)
    namespace = MemoryNamespace(
        experiment_id="checkpoint-memory",
        snapshot_id="snapshot-v1",
        split=DatasetSplit.DEV,
        access_mode=MemoryAccessMode.READ_ONLY,
    )
    episode = VerifiedEpisodeMemoryItem(
        episode_id="episode-checkpoint-001",
        source_task_id="source-task-001",
        source_run_id=uuid4(),
        repository_id="repo_agent_graph",
        repository_commit=original.task.base_commit,
        repository_content_sha256="b" * 64,
        problem_statement="Parser rejected blank input.",
        issue_fingerprint="c" * 64,
        repair_summary="Ignored blank input before parsing.",
        files_changed=("pkg/parser.py",),
        test_command_sha256="d" * 64,
        patch_artifact_sha256="e" * 64,
        source_result_sha256="f" * 64,
        source_trace_sha256="1" * 64,
        artifacts_sha256="2" * 64,
        verifier_version="rule-verifier-v1",
        model_provider="fake-model-v1",
        validation_tests_passed=1,
        created_at=created_at,
        expires_at=created_at + timedelta(days=30),
        source_record_sha256="3" * 64,
    )
    audit = MemoryLoadAudit(
        namespace=namespace,
        repository_status=RepositoryMemoryLookupStatus.EMPTY,
        episodic_status=VerifiedEpisodeLookupStatus.CURRENT,
        episodic_records_loaded=1,
    )
    state = original.model_copy(
        update={
            "memory_namespace": namespace,
            "memory_load_audit": audit,
            "episodic_memory": (episode,),
        }
    )
    graph = AgentStateGraph()

    graph.start("verified-episode-memory", state)

    assert graph.load("verified-episode-memory") == state


def test_shared_in_memory_saver_recovers_across_graph_instances() -> None:
    saver = InMemorySaver()
    first_graph = AgentStateGraph(checkpointer=saver)
    state = load_state()
    first_graph.start("recoverable", state)

    second_graph = AgentStateGraph(checkpointer=saver)

    assert second_graph.load("recoverable") == state
    with pytest.raises(AgentStateGraphError) as missing:
        AgentStateGraph().load("recoverable")
    assert_graph_error(missing, AgentStateErrorCode.THREAD_NOT_FOUND)


def test_checkpoint_round_trips_pending_call_and_optional_enums() -> None:
    original = load_state()
    observation = original.observations[0].model_copy(
        update={"error_type": ErrorType.TOOL_EXECUTION_ERROR}
    )
    pending = ToolCall(
        call_id=uuid4(),
        tool_name="read_file",
        validated_arguments={"path": "README.md"},
        requested_at=datetime(2026, 8, 5, tzinfo=UTC),
    )
    state = AgentState.model_validate(
        {
            **original.model_dump(mode="python"),
            "run_id": uuid4(),
            "observations": (observation,),
            "pending_tool_call": pending,
            "termination_reason": None,
        }
    )
    graph = AgentStateGraph()

    graph.start("optional-fields", state)

    assert graph.load("optional-fields") == state


def test_checkpoint_round_trips_structured_test_category() -> None:
    original = load_state()
    observation = original.observations[0]
    test_result = RepoTestRunResult(
        observation_call_id=observation.call_id,
        command_index=0,
        command_sha256="a" * 64,
        status=RepoTestStatus.FAILED,
        category=RepoTestRunCategory.ASSERTION_FAILURE,
        exit_code=1,
        duration_ms=observation.duration_ms,
        tests_run=1,
        failures=1,
        stdout_ref=observation.stdout_ref,
        summary="assertion_failure; tests=1; failures=1; errors=0; exit=1",
    )
    state = original.model_copy(
        update={
            "test_state": RepoTestState(
                status=RepoTestStatus.FAILED,
                last_observation_call_id=observation.call_id,
                commands_run=1,
                last_result=test_result,
            ),
            "baseline_test_result": test_result,
        }
    )
    graph = AgentStateGraph()

    graph.start("test-category", state)

    assert graph.load("test-category") == state


def test_threads_are_isolated_and_expected_task_is_checked() -> None:
    graph = AgentStateGraph()
    first = load_state(task_id="fixture-001")
    second = load_state(task_id="fixture-002")
    graph.start("thread-one", first)
    graph.start("thread-two", second)

    graph.update("thread-one", {"repository_summary": "Only thread one."})

    assert graph.load("thread-one").repository_summary == "Only thread one."
    assert graph.load("thread-two") == second
    with pytest.raises(AgentStateGraphError) as mismatch:
        graph.load("thread-one", expected_task_id="fixture-002")
    assert_graph_error(mismatch, AgentStateErrorCode.TASK_MISMATCH)


def test_existing_thread_rejects_same_or_different_task() -> None:
    graph = AgentStateGraph()
    graph.start("occupied", load_state(task_id="fixture-001"))

    with pytest.raises(AgentStateGraphError) as duplicate:
        graph.start("occupied", load_state(task_id="fixture-001"))
    with pytest.raises(AgentStateGraphError) as mismatch:
        graph.start("occupied", load_state(task_id="fixture-002"))

    assert_graph_error(duplicate, AgentStateErrorCode.THREAD_ALREADY_EXISTS)
    assert_graph_error(mismatch, AgentStateErrorCode.TASK_MISMATCH)


@pytest.mark.parametrize(
    "thread_id",
    ["", "../outside", "contains space", "x" * 129, "secret-shaped\x00thread"],
)
def test_invalid_thread_ids_are_safe_and_never_echoed(thread_id: str) -> None:
    graph = AgentStateGraph()

    with pytest.raises(AgentStateGraphError) as captured:
        graph.start(thread_id, load_state())

    assert_graph_error(captured, AgentStateErrorCode.INVALID_THREAD_ID)
    if thread_id:
        assert thread_id not in str(captured.value)


def test_missing_thread_has_stable_error() -> None:
    with pytest.raises(AgentStateGraphError) as captured:
        AgentStateGraph().load("missing")

    assert_graph_error(captured, AgentStateErrorCode.THREAD_NOT_FOUND)


def test_partial_update_preserves_unrelated_fields_and_checkpoints_reducers() -> None:
    graph = AgentStateGraph()
    original = load_state()
    graph.start("updates", original)
    call_id = uuid4()

    updated = graph.update(
        "updates",
        {
            "repository_summary": "New summary.",
            "working_memory": ("New evidence.", original.working_memory[0]),
            "consumed_call_ids": (call_id, call_id),
        },
    )

    assert updated.repository_summary == "New summary."
    assert updated.working_memory == (*original.working_memory, "New evidence.")
    assert updated.consumed_call_ids == (call_id,)
    assert updated.task == original.task
    assert updated.current_plan == original.current_plan
    assert updated.observations == original.observations
    assert updated.patch_state == original.patch_state
    assert updated.test_state == original.test_state
    assert updated.budget == original.budget
    assert graph.load("updates") == updated


def test_idempotent_observation_update_and_conflict_do_not_corrupt_checkpoint() -> None:
    graph = AgentStateGraph()
    original = load_state()
    graph.start("observation", original)
    observation = original.observations[0]

    assert graph.update("observation", {"observations": (observation,)}) == original
    conflicting = observation.model_copy(update={"duration_ms": observation.duration_ms + 1})
    with pytest.raises(AgentStateGraphError) as captured:
        graph.update("observation", {"observations": (conflicting,)})

    assert_graph_error(captured, AgentStateErrorCode.STATE_CONFLICT)
    assert graph.load("observation") == original


def test_invalid_updates_are_rejected_before_checkpoint_mutation() -> None:
    graph = AgentStateGraph()
    original = load_state()
    graph.start("invalid-update", original)
    forbidden = cast(AgentGraphUpdate, {"task": load_state(task_id="fixture-002").task})

    with pytest.raises(AgentStateGraphError) as empty:
        graph.update("invalid-update", {})
    with pytest.raises(AgentStateGraphError) as invalid_plan:
        graph.update("invalid-update", {"current_plan": ()})
    with pytest.raises(AgentStateGraphError) as task_change:
        graph.update("invalid-update", forbidden)

    assert_graph_error(empty, AgentStateErrorCode.STATE_CONFLICT)
    assert_graph_error(invalid_plan, AgentStateErrorCode.STATE_CONFLICT)
    assert_graph_error(task_change, AgentStateErrorCode.STATE_CONFLICT)
    assert graph.load("invalid-update") == original


def test_state_graph_does_not_use_network_or_host_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("external capability must not be used")

    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setattr(subprocess, "run", fail)
    graph = AgentStateGraph()

    graph.start("offline", load_state())
    graph.update("offline", {"working_memory": ("Offline checkpoint.",)})

    assert graph.load("offline").working_memory[-1] == "Offline checkpoint."
