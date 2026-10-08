"""End-to-end deterministic Happy-path Agent loop tests."""

from __future__ import annotations

import json
import socket
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn, cast
from uuid import UUID

import pytest

from repopilot.agent import (
    AgentLoopError,
    AgentStateGraphError,
    HappyPathAgentGraph,
    HappyPathNodes,
    apply_state_update,
    decision_request_id,
    initialize_plan,
    to_graph_state,
)
from repopilot.config import RepoPilotSettings
from repopilot.context import CONTEXT_POLICY_VERSION
from repopilot.enums import (
    AgentFailureKind,
    AgentLoopErrorCode,
    AgentStateErrorCode,
    ErrorType,
    PlanStepStatus,
    PolicyDecision,
    TerminationReason,
    TraceErrorCode,
    TraceNode,
    WorkingMemoryErrorCode,
)
from repopilot.enums import TestStatus as AgentTestStatus
from repopilot.memory import (
    FileWorkingMemoryStore,
    WorkingMemoryCheckpoint,
    WorkingMemoryError,
    build_working_memory_checkpoint,
    context_policy_version_from_events,
)
from repopilot.providers import (
    FakeProvider,
    FinishAction,
    ProviderResponse,
    ProviderUsageDelta,
    ToolAction,
)
from repopilot.replay import replay_trace
from repopilot.sandbox import FakeSandbox, FakeSandboxStep, RunRequest, SandboxHandle
from repopilot.schemas import (
    AgentState,
    ArtifactRef,
    BudgetState,
    Observation,
    TaskSpec,
    ToolCall,
)
from repopilot.tools import (
    ReadOnlyRepository,
    RunTestsTool,
    ToolRegistry,
    build_read_only_tool_registry,
)
from repopilot.trace import (
    AgentTrace,
    TraceArtifactError,
    read_trace,
    trace_content_sha256,
    write_trace,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
REPOSITORY_ROOT = FIXTURES / "repositories" / "read_only"
RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
SEARCH_CALL_ID = UUID("11111111-2222-4333-8444-555555555555")
EMPTY_SEARCH_CALL_ID = UUID("00000000-1111-4222-8333-444444444444")
READ_CALL_ID = UUID("22222222-3333-4444-8555-666666666666")
TEST_CALL_ID = UUID("33333333-4444-4555-8666-777777777777")
STARTED_AT = datetime(2026, 8, 5, 8, tzinfo=UTC)
ALLOWED_TOOLS = (
    "inspect_symbol",
    "list_files",
    "read_file",
    "run_tests",
    "search_code",
)


def load_fixture(name: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((FIXTURES / name).read_text(encoding="utf-8")),
    )


def make_task(*, task_id: str = "fixture-001") -> TaskSpec:
    payload = load_fixture("valid_task.json")
    payload["task_id"] = task_id
    return TaskSpec.model_validate(payload)


def make_state(*, task: TaskSpec | None = None) -> AgentState:
    return AgentState(
        task=task or make_task(),
        repository_summary="A tiny parser repository used by the offline Fake run.",
        budget=BudgetState(
            max_steps=30,
            max_tokens=100_000,
            max_cost_usd=Decimal("1"),
            started_at=STARTED_AT,
            deadline_at=STARTED_AT + timedelta(minutes=15),
        ),
    )


def replace_budget(state: AgentState, **updates: object) -> AgentState:
    budget = BudgetState.model_validate(
        {
            **state.budget.model_dump(mode="python"),
            **updates,
        }
    )
    return AgentState.model_validate(
        {
            **state.model_dump(mode="python"),
            "budget": budget,
        }
    )


def make_call(
    call_id: UUID,
    tool_name: str,
    arguments: dict[str, object],
) -> ToolCall:
    return ToolCall.model_validate(
        {
            "call_id": call_id,
            "tool_name": tool_name,
            "validated_arguments": arguments,
            "requested_at": STARTED_AT,
        }
    )


def usage(
    *,
    input_tokens: int = 10,
    output_tokens: int = 2,
    cost: Decimal = Decimal("0.0001"),
) -> ProviderUsageDelta:
    return ProviderUsageDelta(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        estimated_cost_usd=cost,
        latency_ms=1,
    )


def tool_response(
    index: int,
    *,
    call_id: UUID,
    tool_name: str,
    arguments: dict[str, object],
    usage_delta: ProviderUsageDelta | None = None,
) -> ProviderResponse:
    return ProviderResponse(
        request_id=decision_request_id(RUN_ID, index),
        action=ToolAction(
            tool_call=make_call(call_id, tool_name, arguments),
            considered_tools=(tool_name,),
            selection_reason=f"Collect evidence with {tool_name}.",
        ),
        usage=usage_delta or usage(),
    )


def finish_response(index: int = 3) -> ProviderResponse:
    return ProviderResponse(
        request_id=decision_request_id(RUN_ID, index),
        action=FinishAction(summary="Search, source read, and Fake test evidence collected."),
        usage=usage(),
    )


def happy_responses() -> tuple[ProviderResponse, ...]:
    return (
        tool_response(
            0,
            call_id=SEARCH_CALL_ID,
            tool_name="search_code",
            arguments={"query": "parse"},
        ),
        tool_response(
            1,
            call_id=READ_CALL_ID,
            tool_name="read_file",
            arguments={"path": "pkg/parser.py", "start_line": 4, "end_line": 10},
        ),
        tool_response(
            2,
            call_id=TEST_CALL_ID,
            tool_name="run_tests",
            arguments={"test_index": 0},
        ),
        finish_response(),
    )


def make_observation(*, call_id: UUID = TEST_CALL_ID) -> Observation:
    return Observation(
        call_id=call_id,
        started_at=STARTED_AT + timedelta(seconds=2),
        duration_ms=23,
        exit_code=0,
        sandbox_id="sandbox-happy-path",
        policy_decision=PolicyDecision.ALLOWED,
    )


def make_registry(
    task: TaskSpec,
    *,
    observation: Observation | None = None,
) -> tuple[ToolRegistry, FakeSandbox]:
    settings = RepoPilotSettings()
    repository = ReadOnlyRepository.from_settings(REPOSITORY_ROOT, settings)
    registry = build_read_only_tool_registry(repository, settings)
    handle = SandboxHandle(
        sandbox_id="sandbox-happy-path",
        task_id=task.task_id,
        repository_commit=task.base_commit,
        prepared_at=STARTED_AT,
    )
    sandbox = FakeSandbox(
        handle=handle,
        steps=(
            FakeSandboxStep(
                request=RunRequest(
                    call_id=TEST_CALL_ID,
                    command=task.test_commands[0],
                ),
                observation=observation or make_observation(),
            ),
        ),
    )
    registry.register(RunTestsTool.prepare(sandbox=sandbox, task=task))
    return registry, sandbox


def fixed_clock() -> datetime:
    return STARTED_AT


def make_graph(
    responses: tuple[ProviderResponse, ...],
    *,
    task: TaskSpec | None = None,
    max_replans: int = 1,
    clock: Callable[[], datetime] = fixed_clock,
    observation: Observation | None = None,
) -> tuple[HappyPathAgentGraph, FakeProvider, FakeSandbox, AgentState]:
    initial = make_state(task=task)
    registry, sandbox = make_registry(initial.task, observation=observation)
    provider = FakeProvider(responses)
    return (
        HappyPathAgentGraph(
            provider=provider,
            registry=registry,
            max_replans=max_replans,
            clock=clock,
        ),
        provider,
        sandbox,
        initial,
    )


def test_happy_path_search_read_fake_test_finish_and_observation_order() -> None:
    graph, provider, sandbox, initial = make_graph(happy_responses())

    final = graph.run(thread_id="happy-path", run_id=RUN_ID, state=initial)

    assert final == graph.load("happy-path")
    assert final.run_id == RUN_ID
    assert final.termination_reason is TerminationReason.AGENT_STOPPED
    assert final.finish_summary == "Search, source read, and Fake test evidence collected."
    assert final.active_step_id is None
    assert tuple(step.step_id for step in final.current_plan) == (
        "search",
        "read",
        "test",
        "finish",
    )
    assert all(step.status is PlanStepStatus.COMPLETED for step in final.current_plan)
    assert tuple(result.tool_name for result in final.tool_results) == (
        "search_code",
        "read_file",
        "run_tests",
    )
    assert final.consumed_call_ids == (SEARCH_CALL_ID, READ_CALL_ID, TEST_CALL_ID)
    assert final.pending_tool_call is None
    assert final.pending_tool_result is None
    assert final.observations == (make_observation(),)
    assert final.test_state.status is AgentTestStatus.PASSED
    assert final.test_state.last_observation_call_id == TEST_CALL_ID
    assert final.test_state.commands_run == 1
    assert final.decision_count == 4
    assert final.budget.steps_used == 4
    assert final.budget.tokens_used == 48
    assert final.budget.cost_used_usd == Decimal("0.0004")
    assert sandbox.remaining_steps == 0

    assert provider.calls_made == 4
    assert tuple(request.request_id for request in provider.requests) == tuple(
        decision_request_id(RUN_ID, index) for index in range(4)
    )
    assert all(request.allowed_tools == ALLOWED_TOOLS for request in provider.requests)
    assert tuple(len(request.state.tool_results) for request in provider.requests) == (
        0,
        1,
        2,
        3,
    )
    assert tuple(request.state.active_step_id for request in provider.requests) == (
        "search",
        "read",
        "test",
        "finish",
    )
    assert provider.requests[3].state.test_state.status is AgentTestStatus.PASSED


def test_happy_path_is_deterministic_and_uses_no_network_or_host_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("happy-path Fake run attempted an external side effect")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    first_graph, _, _, first_initial = make_graph(happy_responses())
    second_graph, _, _, second_initial = make_graph(happy_responses())

    first = first_graph.run(thread_id="repeat-one", run_id=RUN_ID, state=first_initial)
    second = second_graph.run(thread_id="repeat-two", run_id=RUN_ID, state=second_initial)

    assert first == second


def test_existing_thread_is_rejected_before_another_provider_decision() -> None:
    graph, provider, _, initial = make_graph(happy_responses())
    graph.run(thread_id="occupied", run_id=RUN_ID, state=initial)

    with pytest.raises(AgentStateGraphError) as captured:
        graph.run(thread_id="occupied", run_id=RUN_ID, state=initial)

    assert captured.value.code is AgentStateErrorCode.THREAD_ALREADY_EXISTS
    assert provider.calls_made == 4


def test_finish_before_evidence_is_a_terminal_provider_failure() -> None:
    graph, provider, _, initial = make_graph((finish_response(index=0),))

    checkpoint = graph.run(
        thread_id="early-finish",
        run_id=RUN_ID,
        state=initial,
    )

    assert checkpoint.termination_reason is TerminationReason.PROVIDER_ERROR
    assert checkpoint.active_step_id is None
    assert checkpoint.decision_count == 1
    assert checkpoint.tool_results == ()
    assert provider.calls_made == 1


def test_consumed_call_id_cannot_be_selected_again_after_observation() -> None:
    responses = (
        happy_responses()[0],
        tool_response(
            1,
            call_id=SEARCH_CALL_ID,
            tool_name="read_file",
            arguments={"path": "pkg/parser.py"},
        ),
    )
    graph, provider, _, initial = make_graph(responses)

    checkpoint = graph.run(
        thread_id="duplicate-call",
        run_id=RUN_ID,
        state=initial,
    )

    assert checkpoint.termination_reason is TerminationReason.REPEATED_ACTION
    assert checkpoint.consumed_call_ids == (SEARCH_CALL_ID,)
    assert tuple(result.tool_name for result in checkpoint.tool_results) == ("search_code",)
    assert checkpoint.active_step_id is None
    assert checkpoint.current_plan[1].status is PlanStepStatus.BLOCKED
    assert checkpoint.pending_tool_call is None
    assert provider.calls_made == 2


def test_wrong_tool_order_is_a_terminal_provider_failure_without_execution() -> None:
    response = tool_response(
        0,
        call_id=READ_CALL_ID,
        tool_name="read_file",
        arguments={"path": "pkg/parser.py"},
    )
    graph, _, sandbox, initial = make_graph((response,))

    checkpoint = graph.run(
        thread_id="wrong-order",
        run_id=RUN_ID,
        state=initial,
    )

    assert checkpoint.termination_reason is TerminationReason.PROVIDER_ERROR
    assert checkpoint.failure_history[0].kind is AgentFailureKind.PROVIDER_FAILURE
    assert checkpoint.tool_results == ()
    assert sandbox.remaining_steps == 1


def test_provider_failure_has_stable_code_and_preserves_initialized_checkpoint() -> None:
    mismatched = ProviderResponse(
        request_id=UUID("ffffffff-ffff-4fff-8fff-ffffffffffff"),
        action=FinishAction(summary="This response is correlated to another request."),
        usage=usage(),
    )
    graph, _, _, initial = make_graph((mismatched,))

    checkpoint = graph.run(
        thread_id="provider-failure",
        run_id=RUN_ID,
        state=initial,
    )

    assert checkpoint.termination_reason is TerminationReason.PROVIDER_ERROR
    assert checkpoint.active_step_id is None
    assert checkpoint.failure_history[0].error_type is ErrorType.MODEL_FORMAT_ERROR
    assert checkpoint.decision_count == 1
    assert checkpoint.budget.steps_used == 1


def test_tool_failure_keeps_exact_pending_call_for_diagnosis() -> None:
    invalid_search = tool_response(
        0,
        call_id=SEARCH_CALL_ID,
        tool_name="search_code",
        arguments={"query": ""},
    )
    graph, _, _, initial = make_graph((invalid_search,))

    checkpoint = graph.run(
        thread_id="tool-failure",
        run_id=RUN_ID,
        state=initial,
    )

    assert checkpoint.termination_reason is TerminationReason.TOOL_ERROR
    assert checkpoint.failure_history[0].error_type is ErrorType.INVALID_TOOL_ARGUMENTS
    assert checkpoint.pending_tool_call is None
    assert checkpoint.pending_tool_result is None
    assert checkpoint.consumed_call_ids == (SEARCH_CALL_ID,)
    assert len(checkpoint.tool_results) == 1
    assert checkpoint.decision_count == 1


def test_decide_rejects_unfinished_pending_action_before_provider_call() -> None:
    initial = AgentState.model_validate(
        {
            **make_state().model_dump(mode="python"),
            "run_id": RUN_ID,
        }
    )
    planned = apply_state_update(initial, initialize_plan(to_graph_state(initial)))
    pending = AgentState.model_validate(
        {
            **planned.model_dump(mode="python"),
            "pending_tool_call": make_call(
                SEARCH_CALL_ID,
                "search_code",
                {"query": "parse"},
            ),
        }
    )
    registry, _ = make_registry(initial.task)
    provider = FakeProvider((happy_responses()[0],))
    nodes = HappyPathNodes(provider=provider, registry=registry)

    with pytest.raises(AgentLoopError) as captured:
        nodes.decide(to_graph_state(pending))

    assert captured.value.code is AgentLoopErrorCode.UNFINISHED_PENDING_ACTION
    assert provider.calls_made == 0


@pytest.mark.parametrize(
    ("budget_updates", "expected_reason"),
    [
        ({"steps_used": 30}, TerminationReason.MAX_STEPS),
        ({"tokens_used": 100_000}, TerminationReason.TOKEN_LIMIT),
        ({"cost_used_usd": Decimal("1")}, TerminationReason.COST_LIMIT),
    ],
)
def test_pre_decision_budget_limit_stops_without_provider_or_tool(
    budget_updates: dict[str, object],
    expected_reason: TerminationReason,
) -> None:
    graph, provider, sandbox, initial = make_graph(happy_responses())
    initial = replace_budget(initial, **budget_updates)

    final = graph.run(
        thread_id=f"pre-{expected_reason.value}",
        run_id=RUN_ID,
        state=initial,
    )

    assert final.termination_reason is expected_reason
    assert final.active_step_id is None
    assert final.current_plan[0].status is PlanStepStatus.BLOCKED
    assert provider.calls_made == 0
    assert sandbox.remaining_steps == 1


def test_deadline_stops_before_provider_call() -> None:
    graph, provider, sandbox, initial = make_graph(
        happy_responses(),
        clock=lambda: STARTED_AT + timedelta(minutes=15),
    )

    final = graph.run(thread_id="pre-timeout", run_id=RUN_ID, state=initial)

    assert final.termination_reason is TerminationReason.TIMEOUT
    assert provider.calls_made == 0
    assert sandbox.remaining_steps == 1


def test_max_steps_allows_exactly_that_many_decisions_then_stops() -> None:
    graph, provider, _, initial = make_graph(happy_responses())
    initial = replace_budget(initial, max_steps=3)

    final = graph.run(thread_id="exact-three-steps", run_id=RUN_ID, state=initial)

    assert final.termination_reason is TerminationReason.MAX_STEPS
    assert final.decision_count == 3
    assert final.budget.steps_used == 3
    assert final.test_state.status is AgentTestStatus.PASSED
    assert final.current_plan[3].status is PlanStepStatus.BLOCKED
    assert provider.calls_made == 3


def test_deadline_crossed_during_provider_call_stops_before_tool() -> None:
    clock_values = iter(
        (
            STARTED_AT,
            STARTED_AT + timedelta(minutes=15),
        )
    )
    graph, provider, sandbox, initial = make_graph(
        (happy_responses()[0],),
        clock=lambda: next(clock_values),
    )

    final = graph.run(thread_id="post-timeout", run_id=RUN_ID, state=initial)

    assert final.termination_reason is TerminationReason.TIMEOUT
    assert final.budget.steps_used == 1
    assert final.tool_results == ()
    assert provider.calls_made == 1
    assert sandbox.remaining_steps == 1


@pytest.mark.parametrize(
    ("budget_updates", "decision_usage", "expected_reason"),
    [
        (
            {"max_tokens": 11},
            usage(input_tokens=10, output_tokens=2),
            TerminationReason.TOKEN_LIMIT,
        ),
        (
            {"max_cost_usd": Decimal("0.00005")},
            usage(cost=Decimal("0.0001")),
            TerminationReason.COST_LIMIT,
        ),
    ],
)
def test_post_decision_overshoot_is_recorded_but_action_is_not_executed(
    budget_updates: dict[str, object],
    decision_usage: ProviderUsageDelta,
    expected_reason: TerminationReason,
) -> None:
    response = tool_response(
        0,
        call_id=SEARCH_CALL_ID,
        tool_name="search_code",
        arguments={"query": "parse"},
        usage_delta=decision_usage,
    )
    graph, provider, sandbox, initial = make_graph((response,))
    initial = replace_budget(initial, **budget_updates)

    final = graph.run(
        thread_id=f"post-{expected_reason.value}",
        run_id=RUN_ID,
        state=initial,
    )

    assert final.termination_reason is expected_reason
    assert final.decision_count == 1
    assert final.budget.steps_used == 1
    assert final.budget.tokens_used == decision_usage.total_tokens
    assert final.budget.cost_used_usd == decision_usage.estimated_cost_usd
    assert final.pending_tool_call is None
    assert final.tool_results == ()
    assert provider.calls_made == 1
    assert sandbox.remaining_steps == 1


def test_empty_search_replans_once_then_happy_path_finishes() -> None:
    responses = (
        tool_response(
            0,
            call_id=EMPTY_SEARCH_CALL_ID,
            tool_name="search_code",
            arguments={"query": "definitely-not-present"},
        ),
        tool_response(
            1,
            call_id=SEARCH_CALL_ID,
            tool_name="search_code",
            arguments={"query": "parse"},
        ),
        tool_response(
            2,
            call_id=READ_CALL_ID,
            tool_name="read_file",
            arguments={"path": "pkg/parser.py", "start_line": 4, "end_line": 10},
        ),
        tool_response(
            3,
            call_id=TEST_CALL_ID,
            tool_name="run_tests",
            arguments={"test_index": 0},
        ),
        finish_response(index=4),
    )
    graph, provider, _, initial = make_graph(responses)

    final = graph.run(thread_id="empty-then-broader", run_id=RUN_ID, state=initial)

    assert final.termination_reason is TerminationReason.AGENT_STOPPED
    assert final.replan_count == 1
    assert final.decision_count == 5
    assert len(final.tool_results) == 4
    assert len(final.action_fingerprints) == 4
    assert tuple(failure.kind for failure in final.failure_history) == (
        AgentFailureKind.EMPTY_SEARCH,
    )
    assert final.current_plan[0].revision_reason == (
        "search returned no matches; broaden the query"
    )
    assert provider.calls_made == 5


def test_empty_search_without_replan_budget_terminates_as_retrieval_exhausted() -> None:
    response = tool_response(
        0,
        call_id=EMPTY_SEARCH_CALL_ID,
        tool_name="search_code",
        arguments={"query": "definitely-not-present"},
    )
    graph, provider, _, initial = make_graph((response,), max_replans=0)

    final = graph.run(thread_id="no-replans", run_id=RUN_ID, state=initial)

    assert final.termination_reason is TerminationReason.RETRIEVAL_EXHAUSTED
    assert final.replan_count == 0
    assert final.current_plan[0].status is PlanStepStatus.BLOCKED
    assert final.failure_history[0].kind is AgentFailureKind.EMPTY_SEARCH
    assert provider.calls_made == 1


def test_graph_from_settings_applies_replan_limit() -> None:
    response = tool_response(
        0,
        call_id=EMPTY_SEARCH_CALL_ID,
        tool_name="search_code",
        arguments={"query": "definitely-not-present"},
    )
    initial = make_state()
    registry, _ = make_registry(initial.task)
    provider = FakeProvider((response,))
    graph = HappyPathAgentGraph.from_settings(
        provider=provider,
        registry=registry,
        settings=RepoPilotSettings(max_replans=0),
        clock=fixed_clock,
    )

    final = graph.run(thread_id="settings-replans", run_id=RUN_ID, state=initial)

    assert final.termination_reason is TerminationReason.RETRIEVAL_EXHAUSTED
    assert final.replan_count == 0


def test_second_distinct_empty_search_exhausts_single_replan() -> None:
    responses = (
        tool_response(
            0,
            call_id=EMPTY_SEARCH_CALL_ID,
            tool_name="search_code",
            arguments={"query": "definitely-not-present"},
        ),
        tool_response(
            1,
            call_id=SEARCH_CALL_ID,
            tool_name="search_code",
            arguments={"query": "still-not-present"},
        ),
    )
    graph, provider, _, initial = make_graph(responses)

    final = graph.run(thread_id="replans-exhausted", run_id=RUN_ID, state=initial)

    assert final.termination_reason is TerminationReason.RETRIEVAL_EXHAUSTED
    assert final.replan_count == 1
    assert len(final.tool_results) == 2
    assert tuple(failure.kind for failure in final.failure_history) == (
        AgentFailureKind.EMPTY_SEARCH,
        AgentFailureKind.EMPTY_SEARCH,
    )
    assert provider.calls_made == 2


def test_same_search_after_empty_result_terminates_before_repeat_execution() -> None:
    responses = (
        tool_response(
            0,
            call_id=EMPTY_SEARCH_CALL_ID,
            tool_name="search_code",
            arguments={"query": "definitely-not-present"},
        ),
        tool_response(
            1,
            call_id=SEARCH_CALL_ID,
            tool_name="search_code",
            arguments={"query": "definitely-not-present"},
        ),
    )
    graph, provider, _, initial = make_graph(responses)

    final = graph.run(thread_id="same-search", run_id=RUN_ID, state=initial)

    assert final.termination_reason is TerminationReason.REPEATED_ACTION
    assert len(final.tool_results) == 1
    assert final.consumed_call_ids == (EMPTY_SEARCH_CALL_ID,)
    assert tuple(failure.kind for failure in final.failure_history) == (
        AgentFailureKind.EMPTY_SEARCH,
        AgentFailureKind.REPEATED_ACTION,
    )
    assert provider.calls_made == 2


def test_nonzero_test_observation_terminates_as_tests_failed() -> None:
    failed_observation = make_observation().model_copy(
        update={
            "exit_code": 1,
            "stderr_ref": ArtifactRef(
                relative_path="stderr.txt",
                sha256="a" * 64,
                size_bytes=141,
                media_type="text/plain",
            ),
            "stderr_preview": (
                "FAIL: test_regression (checks.ParserTests.test_regression)\n"
                "----------------------------------------------------------------------\n"
                "Ran 1 test in 0.001s\n\n"
                "FAILED (failures=1)\n"
            ),
        }
    )
    graph, provider, _, initial = make_graph(
        happy_responses()[:3],
        observation=failed_observation,
    )

    final = graph.run(thread_id="tests-failed", run_id=RUN_ID, state=initial)

    assert final.termination_reason is TerminationReason.TESTS_FAILED
    assert final.test_state.status is AgentTestStatus.FAILED
    assert final.test_state.last_result is not None
    assert final.test_state.last_result.failing_tests == ("checks.ParserTests.test_regression",)
    assert final.current_plan[2].status is PlanStepStatus.BLOCKED
    assert final.failure_history[0].kind is AgentFailureKind.TEST_FAILURE
    assert provider.calls_made == 3


def test_sandbox_error_observation_has_distinct_terminal_reason() -> None:
    sandbox_error = make_observation().model_copy(
        update={
            "exit_code": None,
            "error_type": ErrorType.SANDBOX_ERROR,
        }
    )
    graph, _, _, initial = make_graph(
        happy_responses()[:3],
        observation=sandbox_error,
    )

    final = graph.run(thread_id="sandbox-error", run_id=RUN_ID, state=initial)

    assert final.termination_reason is TerminationReason.SANDBOX_ERROR
    assert final.test_state.status is AgentTestStatus.ERROR
    assert final.failure_history[0].error_type is ErrorType.SANDBOX_ERROR


def build_happy_trace() -> AgentTrace:
    graph, _, _, initial = make_graph(happy_responses())
    graph.run(thread_id="trace-happy", run_id=RUN_ID, state=initial)
    return graph.build_trace("trace-happy", settings=RepoPilotSettings())


def test_trace_contains_complete_node_and_external_interaction_sequence() -> None:
    trace = build_happy_trace()

    assert tuple(event.sequence for event in trace.events) == tuple(range(12))
    assert tuple(event.node for event in trace.events) == (
        TraceNode.INITIALIZE_PLAN,
        TraceNode.DECIDE,
        TraceNode.EXECUTE_TOOL,
        TraceNode.OBSERVE,
        TraceNode.DECIDE,
        TraceNode.EXECUTE_TOOL,
        TraceNode.OBSERVE,
        TraceNode.DECIDE,
        TraceNode.EXECUTE_TOOL,
        TraceNode.OBSERVE,
        TraceNode.DECIDE,
        TraceNode.FINISH,
    )
    decision_events = tuple(event for event in trace.events if event.node is TraceNode.DECIDE)
    execute_events = tuple(event for event in trace.events if event.node is TraceNode.EXECUTE_TOOL)
    assert all(event.provider_request is not None for event in decision_events)
    assert all(event.provider_response is not None for event in decision_events)
    for event in decision_events:
        assert event.provider_request is not None
        assert event.provider_response is not None
        assert event.provider_request.context_window is not None
        assert event.context_usage is not None
        assert (
            event.context_usage.estimated_input_tokens
            == event.provider_request.context_window.estimated_tokens_after
        )
        assert (
            event.context_usage.provider_input_tokens == event.provider_response.usage.input_tokens
        )
        assert event.context_usage.estimate_delta_tokens == (
            event.provider_response.usage.input_tokens
            - event.provider_request.context_window.estimated_tokens_after
        )
    assert tuple(event.tool_call.tool_name for event in execute_events if event.tool_call) == (
        "search_code",
        "read_file",
        "run_tests",
    )
    assert trace.final_state.termination_reason is TerminationReason.AGENT_STOPPED
    assert replay_trace(trace).final_state == trace.final_state


def test_trace_write_read_is_deterministic_and_replay_has_no_capability_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = build_happy_trace()
    trace_path = tmp_path / "trace.json"

    write_trace(trace_path, trace)
    first_bytes = trace_path.read_bytes()
    write_trace(trace_path, trace)
    assert trace_path.read_bytes() == first_bytes
    loaded = read_trace(trace_path)

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("offline replay attempted an external capability")

    monkeypatch.setattr(FakeProvider, "generate", forbidden)
    monkeypatch.setattr(ToolRegistry, "invoke", forbidden)
    monkeypatch.setattr(FakeSandbox, "run", forbidden)

    replayed = replay_trace(loaded)

    assert replayed.event_count == 12
    assert replayed.content_sha256 == trace.content_sha256


def test_trace_reader_preserves_pre_p6_schema_v1_hash_compatibility(
    tmp_path: Path,
) -> None:
    raw = build_happy_trace().model_dump(mode="json")
    for event in raw["events"]:
        event.pop("context_usage", None)
        request = event.get("provider_request")
        if isinstance(request, dict):
            request.pop("context_window", None)
    payload = {key: value for key, value in raw.items() if key != "content_sha256"}
    raw["content_sha256"] = trace_content_sha256(payload)
    path = tmp_path / "legacy-trace.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = read_trace(path)

    assert all(
        event.context_usage is None
        and (event.provider_request is None or event.provider_request.context_window is None)
        for event in loaded.events
    )
    assert replay_trace(loaded).final_state == loaded.final_state


def test_trace_reader_preserves_p6_s1_context_hash_compatibility(
    tmp_path: Path,
) -> None:
    raw = build_happy_trace().model_dump(mode="json")
    for event in raw["events"]:
        request = event.get("provider_request")
        if not isinstance(request, dict):
            continue
        window = request.get("context_window")
        if not isinstance(window, dict):
            continue
        window.pop("policy", None)
        window.pop("budget_usages", None)
        window.pop("transforms", None)
        for exclusion in window["exclusions"]:
            exclusion.pop("replacement_item_id", None)
    payload = {key: value for key, value in raw.items() if key != "content_sha256"}
    raw["content_sha256"] = trace_content_sha256(payload)
    path = tmp_path / "p6-s1-trace.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = read_trace(path)

    decision_windows = [
        event.provider_request.context_window
        for event in loaded.events
        if event.provider_request is not None and event.provider_request.context_window is not None
    ]
    assert decision_windows
    assert all(window.policy is None for window in decision_windows)
    assert replay_trace(loaded).final_state == loaded.final_state


def test_trace_integrity_detects_content_tampering() -> None:
    trace = build_happy_trace()
    changed_final = trace.final_state.model_copy(update={"repository_summary": "tampered"})
    tampered = trace.model_copy(update={"final_state": changed_final})

    with pytest.raises(TraceArtifactError) as captured:
        replay_trace(tampered)

    assert captured.value.code is TraceErrorCode.INTEGRITY_MISMATCH


def test_trace_replay_rejects_rehashed_event_order_and_state_chain_tampering() -> None:
    trace = build_happy_trace()
    swapped = (
        trace.events[0],
        trace.events[2],
        trace.events[1],
        *trace.events[3:],
    )
    reordered = AgentTrace.build(
        run_id=trace.run_id,
        task_id=trace.task_id,
        thread_id=trace.thread_id,
        provider_name=trace.provider_name,
        settings=trace.settings,
        events=swapped,
        final_state=trace.final_state,
    )
    broken_event = trace.events[1].model_copy(update={"state_before": trace.events[0].state_before})
    broken_events = (trace.events[0], broken_event, *trace.events[2:])
    broken_chain = AgentTrace.build(
        run_id=trace.run_id,
        task_id=trace.task_id,
        thread_id=trace.thread_id,
        provider_name=trace.provider_name,
        settings=trace.settings,
        events=broken_events,
        final_state=trace.final_state,
    )

    with pytest.raises(TraceArtifactError) as order_error:
        replay_trace(reordered)
    with pytest.raises(TraceArtifactError) as chain_error:
        replay_trace(broken_chain)

    assert order_error.value.code is TraceErrorCode.EVENT_ORDER
    assert chain_error.value.code is TraceErrorCode.STATE_CHAIN_MISMATCH


def test_trace_replay_rejects_rehashed_correlation_and_final_state_tampering() -> None:
    trace = build_happy_trace()
    decision = trace.events[1]
    assert decision.provider_request is not None
    mismatched_request = decision.provider_request.model_copy(
        update={"request_id": UUID("ffffffff-ffff-4fff-8fff-ffffffffffff")}
    )
    mismatched_event = decision.model_copy(update={"provider_request": mismatched_request})
    mismatched_trace = AgentTrace.build(
        run_id=trace.run_id,
        task_id=trace.task_id,
        thread_id=trace.thread_id,
        provider_name=trace.provider_name,
        settings=trace.settings,
        events=(trace.events[0], mismatched_event, *trace.events[2:]),
        final_state=trace.final_state,
    )
    wrong_final = AgentTrace.build(
        run_id=trace.run_id,
        task_id=trace.task_id,
        thread_id=trace.thread_id,
        provider_name=trace.provider_name,
        settings=trace.settings,
        events=trace.events,
        final_state=trace.events[-2].state_after,
    )

    with pytest.raises(TraceArtifactError) as correlation_error:
        replay_trace(mismatched_trace)
    with pytest.raises(TraceArtifactError) as final_error:
        replay_trace(wrong_final)

    assert correlation_error.value.code is TraceErrorCode.CORRELATION_MISMATCH
    assert final_error.value.code is TraceErrorCode.FINAL_STATE_MISMATCH


def make_working_checkpoint(
    *,
    execute_pending_tool: bool = False,
    sequence: int = 0,
) -> tuple[
    WorkingMemoryCheckpoint,
    HappyPathAgentGraph,
    FakeProvider,
    FakeSandbox,
]:
    """Stop after decide or execute to simulate a process interruption."""

    initial = AgentState.model_validate(
        {
            **make_state().model_dump(mode="python"),
            "run_id": RUN_ID,
        }
    )
    registry, sandbox = make_registry(initial.task)
    provider = FakeProvider(happy_responses())
    nodes = HappyPathNodes(
        provider=provider,
        registry=registry,
        clock=fixed_clock,
    )
    graph_state = to_graph_state(initial)
    initialized = nodes.initialize(graph_state)
    state = apply_state_update(initial, initialized)
    decided = nodes.decide(to_graph_state(state))
    state = apply_state_update(state, decided)
    if execute_pending_tool:
        executed = nodes.execute_tool(to_graph_state(state))
        state = apply_state_update(state, executed)
    events = nodes.trace_recorder.events(RUN_ID)
    checkpoint = build_working_memory_checkpoint(
        task_id=state.task.task_id,
        run_id=RUN_ID,
        source_thread_id="interrupted-thread",
        repository_commit=state.task.base_commit,
        context_policy_version=context_policy_version_from_events(events),
        sequence=sequence,
        created_at=STARTED_AT,
        state=state,
        trace_events=events,
    )
    graph = HappyPathAgentGraph(
        provider=provider,
        registry=registry,
        clock=fixed_clock,
    )
    return checkpoint, graph, provider, sandbox


def test_working_memory_file_store_round_trips_canonical_private_snapshot(
    tmp_path: Path,
) -> None:
    checkpoint, _, _, _ = make_working_checkpoint()
    root = tmp_path / "working-memory"
    store = FileWorkingMemoryStore(root)

    store.save(checkpoint)
    loaded = store.load_latest(
        task_id=checkpoint.task_id,
        run_id=checkpoint.run_id,
        repository_commit=checkpoint.repository_commit,
    )

    path = root / checkpoint.task_id / str(checkpoint.run_id) / "00000000.json"
    assert loaded == checkpoint
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert json.loads(path.read_text(encoding="utf-8"))["content_sha256"] == (
        checkpoint.content_sha256
    )


def test_working_memory_store_rejects_tampering_stale_commit_and_sequence_rollback(
    tmp_path: Path,
) -> None:
    checkpoint, _, _, _ = make_working_checkpoint()
    root = tmp_path / "working-memory"
    store = FileWorkingMemoryStore(root)
    store.save(checkpoint)
    path = root / checkpoint.task_id / str(checkpoint.run_id) / "00000000.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["state"]["repository_summary"] = "tampered"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(WorkingMemoryError) as integrity_error:
        store.load_latest(
            task_id=checkpoint.task_id,
            run_id=checkpoint.run_id,
            repository_commit=checkpoint.repository_commit,
        )

    assert integrity_error.value.code is WorkingMemoryErrorCode.INTEGRITY_MISMATCH

    clean_store = FileWorkingMemoryStore(tmp_path / "clean-working-memory")
    clean_store.save(checkpoint)
    with pytest.raises(WorkingMemoryError) as commit_error:
        clean_store.load_latest(
            task_id=checkpoint.task_id,
            run_id=checkpoint.run_id,
            repository_commit="f" * 40,
        )
    with pytest.raises(WorkingMemoryError) as task_isolation_error:
        clean_store.load_latest(
            task_id="another-task",
            run_id=checkpoint.run_id,
            repository_commit=checkpoint.repository_commit,
        )
    with pytest.raises(WorkingMemoryError) as run_isolation_error:
        clean_store.load_latest(
            task_id=checkpoint.task_id,
            run_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
            repository_commit=checkpoint.repository_commit,
        )
    with pytest.raises(WorkingMemoryError) as context_error:
        clean_store.load_latest(
            task_id=checkpoint.task_id,
            run_id=checkpoint.run_id,
            repository_commit=checkpoint.repository_commit,
            context_policy_version="future-context-policy-v2",
        )
    with pytest.raises(WorkingMemoryError) as sequence_error:
        clean_store.save(checkpoint)

    assert commit_error.value.code is WorkingMemoryErrorCode.COMMIT_STALE
    assert task_isolation_error.value.code is WorkingMemoryErrorCode.IO_ERROR
    assert run_isolation_error.value.code is WorkingMemoryErrorCode.IO_ERROR
    assert context_error.value.code is WorkingMemoryErrorCode.RECOVERY_CONFLICT
    assert sequence_error.value.code is WorkingMemoryErrorCode.SEQUENCE_ROLLBACK


def test_working_memory_store_enforces_snapshot_limit_and_rejects_symlink_root(
    tmp_path: Path,
) -> None:
    first, _, _, _ = make_working_checkpoint()
    second, _, _, _ = make_working_checkpoint(sequence=1)
    limited = FileWorkingMemoryStore(tmp_path / "limited", max_snapshots=1)
    limited.save(first)

    with pytest.raises(WorkingMemoryError) as limit_error:
        limited.save(second)

    target = tmp_path / "actual-memory"
    target.mkdir()
    link = tmp_path / "linked-memory"
    link.symlink_to(target, target_is_directory=True)
    unsafe = FileWorkingMemoryStore(link)
    with pytest.raises(WorkingMemoryError) as path_error:
        unsafe.save(first)

    assert limit_error.value.code is WorkingMemoryErrorCode.SNAPSHOT_LIMIT
    assert path_error.value.code is WorkingMemoryErrorCode.PATH_UNSAFE


@pytest.mark.parametrize("execute_pending_tool", [False, True])
def test_graph_resumes_working_memory_without_reinitializing_or_repeating_tool(
    tmp_path: Path,
    *,
    execute_pending_tool: bool,
) -> None:
    checkpoint, graph, provider, sandbox = make_working_checkpoint(
        execute_pending_tool=execute_pending_tool
    )
    store = FileWorkingMemoryStore(tmp_path / "working-memory")
    store.save(checkpoint)
    loaded = store.load_latest(
        task_id=checkpoint.task_id,
        run_id=checkpoint.run_id,
        repository_commit=checkpoint.repository_commit,
        context_policy_version=CONTEXT_POLICY_VERSION,
    )

    final = graph.resume_from_working_memory(
        thread_id="recovered-thread",
        checkpoint=loaded,
    )
    trace = graph.build_trace("recovered-thread", settings=RepoPilotSettings())

    assert final == graph.load("recovered-thread")
    assert final.termination_reason is TerminationReason.AGENT_STOPPED
    assert provider.calls_made == 4
    assert sandbox.remaining_steps == 0
    assert tuple(event.sequence for event in trace.events) == tuple(range(len(trace.events)))
    assert sum(event.node is TraceNode.INITIALIZE_PLAN for event in trace.events) == 1
    assert replay_trace(trace).final_state == final

    with pytest.raises(WorkingMemoryError) as occupied_error:
        graph.resume_from_working_memory(
            thread_id="recovered-thread",
            checkpoint=loaded,
        )

    assert occupied_error.value.code is WorkingMemoryErrorCode.RECOVERY_CONFLICT


def test_working_memory_recovery_rejects_source_thread_and_terminal_save(
    tmp_path: Path,
) -> None:
    checkpoint, recovery_graph, _, _ = make_working_checkpoint()

    with pytest.raises(WorkingMemoryError) as recovery_error:
        recovery_graph.resume_from_working_memory(
            thread_id=checkpoint.source_thread_id,
            checkpoint=checkpoint,
        )

    completed_graph, _, _, initial = make_graph(happy_responses())
    completed_graph.run(thread_id="completed-thread", run_id=RUN_ID, state=initial)
    with pytest.raises(WorkingMemoryError) as terminal_error:
        completed_graph.save_working_memory(
            "completed-thread",
            store=FileWorkingMemoryStore(tmp_path / "working-memory"),
            sequence=0,
            created_at=STARTED_AT,
        )

    assert recovery_error.value.code is WorkingMemoryErrorCode.RECOVERY_CONFLICT
    assert terminal_error.value.code is WorkingMemoryErrorCode.TERMINAL_SNAPSHOT
