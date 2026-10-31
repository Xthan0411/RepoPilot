"""Deterministic tests for the complete repair Agent state machine."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid5

import pytest
from pydantic import BaseModel, Field, JsonValue, ValidationError

from repopilot.agent import (
    REPAIR_STEP_IDS,
    RepairAgentGraph,
    RepairAgentNodes,
    apply_state_update,
    repair_decision_request_id,
    to_graph_state,
)
from repopilot.config import RepoPilotSettings
from repopilot.context import ContextAccountingError
from repopilot.enums import (
    AgentFailureKind,
    ContextErrorCode,
    ErrorType,
    PatchErrorCode,
    PatchStatus,
    PolicyDecision,
    ProviderErrorCode,
    RunStatus,
    TerminationReason,
    TraceNode,
    VerifiedEpisodeErrorCode,
    VerifiedEpisodeLookupStatus,
)
from repopilot.enums import TestRunCategory as RepoTestRunCategory
from repopilot.enums import TestStatus as RepoTestStatus
from repopilot.memory import (
    FileVerifiedEpisodicMemoryStore,
    VerifiedEpisodeError,
    VerifiedEpisodeExecutionEvidence,
    build_verified_episode_record,
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
from repopilot.providers.prompts import SYSTEM_INSTRUCTIONS, build_agent_prompt
from repopilot.replay import replay_trace
from repopilot.schemas import (
    AgentState,
    ArtifactRef,
    BudgetState,
    CommandSpec,
    FinalResult,
    LocalizationMetrics,
    Observation,
    PatchState,
    StrictModel,
    TaskSpec,
    ToolCall,
    UsageMetrics,
)
from repopilot.schemas import TestRunResult as RepoTestRunResult
from repopilot.tools import (
    PatchApplicationResult,
    RunTestsArguments,
    RunTestsResult,
    ToolRegistry,
    UnifiedDiffSummary,
    build_reproduction_result,
)
from repopilot.trace import AgentTrace
from repopilot.verifier import (
    FinalResultContext,
    RuleBasedVerifier,
    VerificationEvidence,
    build_final_result,
)

RUN_ID = UUID("aaaaaaaa-1111-4222-8333-bbbbbbbbbbbb")
STARTED_AT = datetime(2026, 8, 6, 15, 0, tzinfo=UTC)
CALL_IDS = tuple(
    UUID(value)
    for value in (
        "10000000-0000-4000-8000-000000000001",
        "10000000-0000-4000-8000-000000000002",
        "10000000-0000-4000-8000-000000000003",
        "10000000-0000-4000-8000-000000000004",
        "10000000-0000-4000-8000-000000000005",
        "10000000-0000-4000-8000-000000000006",
        "10000000-0000-4000-8000-000000000007",
        "10000000-0000-4000-8000-000000000008",
        "10000000-0000-4000-8000-000000000009",
    )
)


class SearchArguments(StrictModel):
    query: str = Field(min_length=1)


class SearchResult(StrictModel):
    matches: tuple[str, ...]


class SearchTool:
    name = "search_code"
    description = "Return one deterministic source match."
    arguments_model: type[BaseModel] = SearchArguments

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        del arguments, call
        return SearchResult(matches=("pkg/parser.py:1",))


class ReadArguments(StrictModel):
    path: str = Field(min_length=1)


class ReadResult(StrictModel):
    path: str
    content: str


class ReadTool:
    name = "read_file"
    description = "Return one deterministic source file."
    arguments_model: type[BaseModel] = ReadArguments

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        del call
        assert isinstance(arguments, ReadArguments)
        return ReadResult(path=arguments.path, content="VALUE = 1\n")


class InspectArguments(StrictModel):
    path: str
    symbol_name: str


class InspectResult(StrictModel):
    found: bool


class InspectTool:
    name = "inspect_symbol"
    description = "Inspect one deterministic source symbol."
    arguments_model: type[BaseModel] = InspectArguments

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        del arguments, call
        return InspectResult(found=True)


class ApplyArguments(StrictModel):
    diff: str


class ApplyTool:
    name = "apply_patch"
    description = "Return an applied or rejected structured patch result."
    arguments_model: type[BaseModel] = ApplyArguments

    def __init__(self, *, rejection: PatchErrorCode | None = None) -> None:
        self.rejection = rejection

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        del arguments
        if self.rejection is not None:
            return PatchApplicationResult(
                status=PatchStatus.REJECTED,
                patch_state=PatchState(),
                error_code=self.rejection,
                observation=(
                    observation(call.call_id, exit_code=44, suffix="protected-test").model_copy(
                        update={
                            "error_type": ErrorType.PATCH_ERROR,
                            "stdout_preview": "",
                            "stderr_ref": artifact("protected-test.stderr.txt", "e", 41),
                            "stderr_preview": "repopilot_patch_error:protected_test_path\n",
                        }
                    )
                    if self.rejection is PatchErrorCode.PROTECTED_TEST_PATH
                    else None
                ),
                message="candidate patch rejected",
            )
        patch_ref = artifact("patch.diff", "d", 64)
        return PatchApplicationResult(
            status=PatchStatus.APPLIED,
            patch_state=PatchState(
                status=PatchStatus.APPLIED,
                diff_ref=patch_ref,
                files_changed=("pkg/parser.py",),
                revision=1,
            ),
            summary=UnifiedDiffSummary(
                sha256=patch_ref.sha256,
                size_bytes=64,
                files_changed=("pkg/parser.py",),
                hunk_count=1,
                added_lines=1,
                removed_lines=1,
            ),
            observation=observation(call.call_id, exit_code=0, suffix="patch"),
            message="candidate patch applied atomically in the sandbox",
        )


class RunTestsTool:
    name = "run_tests"
    description = "Return baseline failure followed by validation pass."
    arguments_model: type[BaseModel] = RunTestsArguments

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        assert isinstance(arguments, RunTestsArguments)
        self.calls += 1
        passed = self.calls == 2
        current = observation(
            call.call_id,
            exit_code=0 if passed else 1,
            suffix=f"test-{self.calls}",
        )
        result = RepoTestRunResult(
            observation_call_id=call.call_id,
            command_index=arguments.test_index,
            command_sha256="c" * 64,
            status=RepoTestStatus.PASSED if passed else RepoTestStatus.FAILED,
            category=(
                RepoTestRunCategory.PASSED if passed else RepoTestRunCategory.ASSERTION_FAILURE
            ),
            exit_code=0 if passed else 1,
            duration_ms=5,
            tests_run=2,
            failures=0 if passed else 1,
            errors=0,
            failing_tests=() if passed else ("checks.test_regression",),
            stdout_ref=current.stdout_ref,
            stderr_ref=current.stderr_ref,
            summary=(
                "passed; tests=2; failures=0; errors=0; exit=0"
                if passed
                else "assertion_failure; tests=2; failures=1; errors=0; exit=1"
            ),
        )
        return RunTestsResult(observation=current, test_run=result)


def artifact(name: str, fill: str, size: int) -> ArtifactRef:
    content = fill.encode("ascii") * size
    return ArtifactRef(
        relative_path=f"runs/repair/{name}",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=size,
        media_type="text/plain",
    )


def observation(
    call_id: UUID,
    *,
    exit_code: int,
    suffix: str,
) -> Observation:
    output = artifact(f"{suffix}.txt", "b", 20)
    return Observation(
        call_id=call_id,
        started_at=STARTED_AT,
        duration_ms=5,
        exit_code=exit_code,
        stdout_ref=output,
        stdout_preview="bounded test output",
        sandbox_id="sandbox-repair-unit",
        policy_decision=PolicyDecision.ALLOWED,
    )


def make_task() -> TaskSpec:
    return TaskSpec(
        task_id="repair-agent-unit",
        repository_path="/tmp/repair-agent-unit",
        base_commit="a" * 40,
        issue_title="Repair parser regression",
        issue_body="The parser should normalize empty values.",
        test_commands=(
            CommandSpec(
                argv=("python", "-m", "unittest", "-q"),
                timeout_seconds=30,
            ),
        ),
        timeout_seconds=60,
    )


def make_state() -> AgentState:
    return AgentState(
        task=make_task(),
        repository_summary="Top BM25 source: pkg/parser.py",
        budget=BudgetState(
            max_steps=20,
            max_tokens=10_000,
            max_cost_usd=Decimal("1"),
            started_at=STARTED_AT,
            deadline_at=STARTED_AT + timedelta(minutes=5),
        ),
    )


def usage() -> ProviderUsageDelta:
    return ProviderUsageDelta(
        input_tokens=10,
        output_tokens=2,
        total_tokens=12,
        estimated_cost_usd=Decimal("0.001"),
        latency_ms=3,
    )


def tool_response(
    decision: int,
    call_index: int,
    tool_name: str,
    arguments: dict[str, JsonValue],
    considered_tools: tuple[str, ...],
) -> ProviderResponse:
    return ProviderResponse(
        request_id=repair_decision_request_id(RUN_ID, decision),
        action=ToolAction(
            tool_call=ToolCall(
                call_id=CALL_IDS[call_index],
                tool_name=tool_name,
                validated_arguments=arguments,
                requested_at=STARTED_AT,
            ),
            considered_tools=considered_tools,
            selection_reason=f"Select {tool_name} for this repair step.",
        ),
        usage=usage(),
        model="fake-repair-v1",
    )


def successful_responses() -> tuple[ProviderResponse, ...]:
    return (
        tool_response(0, 0, "run_tests", {"test_index": 0}, ("run_tests",)),
        tool_response(1, 1, "search_code", {"query": "parser"}, ("search_code",)),
        tool_response(
            2,
            2,
            "read_file",
            {"path": "pkg/parser.py"},
            ("inspect_symbol", "read_file"),
        ),
        tool_response(
            3,
            3,
            "apply_patch",
            {"diff": "bounded-diff"},
            ("apply_patch",),
        ),
        tool_response(4, 4, "run_tests", {"test_index": 0}, ("run_tests",)),
        ProviderResponse(
            request_id=repair_decision_request_id(RUN_ID, 5),
            action=FinishAction(summary="Evidence is ready for deterministic verification."),
            usage=usage(),
            model="fake-repair-v1",
        ),
    )


def registry(
    *,
    patch_rejection: PatchErrorCode | None = None,
    patch_tool: ApplyTool | None = None,
    read_tool: ReadTool | None = None,
    test_tool: RunTestsTool | None = None,
    search_tool: SearchTool | None = None,
) -> ToolRegistry:
    result = ToolRegistry()
    for tool in (
        search_tool or SearchTool(),
        read_tool or ReadTool(),
        InspectTool(),
        patch_tool or ApplyTool(rejection=patch_rejection),
        test_tool or RunTestsTool(),
    ):
        result.register(tool)
    return result


def verified_source() -> tuple[
    AgentState,
    AgentTrace,
    FinalResult,
    VerifiedEpisodeExecutionEvidence,
]:
    settings = RepoPilotSettings()
    provider = FakeProvider(successful_responses())
    graph = RepairAgentGraph.from_settings(
        provider=provider,
        registry=registry(),
        settings=settings,
        clock=lambda: STARTED_AT,
    )
    final = graph.run(
        thread_id="repair-verified-episode",
        run_id=RUN_ID,
        state=make_state(),
    )
    trace = graph.build_trace("repair-verified-episode", settings=settings)
    assert final.baseline_test_result is not None
    assert final.test_state.last_result is not None
    decision = RuleBasedVerifier().verify(
        VerificationEvidence(
            reproduction_result=build_reproduction_result(final.baseline_test_result),
            patch_state=final.patch_state,
            validation_test_run=final.test_state.last_result,
            tool_steps=len(final.tool_results),
        )
    )
    result = build_final_result(
        decision,
        FinalResultContext(
            task_id=final.task.task_id,
            run_id=RUN_ID,
            localization_metrics=LocalizationMetrics(
                file_recall_at_1=0,
                file_recall_at_3=0,
                file_recall_at_5=0,
                mean_reciprocal_rank=0,
                indexing_latency_ms=0,
                retrieval_latency_ms=0,
            ),
            usage=UsageMetrics(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                estimated_cost_usd=Decimal("0"),
                latency_ms=0,
            ),
            trace_path="trace.json",
        ),
    )
    assert final.patch_state.diff_ref is not None
    artifact_refs = (
        final.patch_state.diff_ref,
        final.baseline_test_result.stdout_ref,
        final.baseline_test_result.stderr_ref,
        final.test_state.last_result.stdout_ref,
        final.test_state.last_result.stderr_ref,
    )
    artifacts = tuple(
        sorted(
            (artifact_ref for artifact_ref in artifact_refs if artifact_ref is not None),
            key=lambda item: item.relative_path,
        )
    )
    execution = VerifiedEpisodeExecutionEvidence(
        sandbox_image_reference=f"python@sha256:{'e' * 64}",
        settings_sha256="f" * 64,
        context_policy_versions=(context_policy_version_from_events(trace.events),),
        artifacts=artifacts,
    )
    return final, trace, result, execution


def materialize_verified_artifacts(
    root: Path,
    execution: VerifiedEpisodeExecutionEvidence,
) -> None:
    for artifact_ref in execution.artifacts:
        path = root / artifact_ref.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fill = b"d" if path.name == "patch.diff" else b"b"
        content = fill * artifact_ref.size_bytes
        assert hashlib.sha256(content).hexdigest() == artifact_ref.sha256
        path.write_bytes(content)


def test_repair_graph_completes_all_evidence_steps_and_replays() -> None:
    settings = RepoPilotSettings()
    provider = FakeProvider(successful_responses())
    graph = RepairAgentGraph.from_settings(
        provider=provider,
        registry=registry(),
        settings=settings,
        clock=lambda: STARTED_AT,
    )

    final = graph.run(
        thread_id="repair-success",
        run_id=RUN_ID,
        state=make_state(),
    )
    trace = graph.build_trace("repair-success", settings=settings)

    assert final.termination_reason is TerminationReason.AGENT_STOPPED
    assert tuple(step.step_id for step in final.current_plan) == REPAIR_STEP_IDS
    assert all(step.status.value == "completed" for step in final.current_plan)
    assert final.baseline_test_result is not None
    assert final.baseline_test_result.status is RepoTestStatus.FAILED
    assert final.test_state.status is RepoTestStatus.PASSED
    assert final.test_state.commands_run == 2
    assert final.patch_state.status is PatchStatus.APPLIED
    assert len(final.tool_results) == 5
    assert len(final.observations) == 3
    assert final.decision_count == 6
    assert final.budget.steps_used == 6
    assert final.budget.tokens_used == 72
    assert provider.requests[0].allowed_tools == ("run_tests",)
    assert provider.requests[2].allowed_tools == ("inspect_symbol", "read_file")
    assert provider.requests[-1].allowed_tools == ()
    assert all(request.context_window is not None for request in provider.requests)
    assert all(
        event.context_usage is not None
        for event in trace.events
        if event.provider_response is not None
    )
    assert final.action_fingerprints[0] != final.action_fingerprints[-1]
    assert replay_trace(trace).final_state == final


def test_provider_exhaustion_yields_replayable_typed_failure() -> None:
    settings = RepoPilotSettings()
    provider = FakeProvider((successful_responses()[0],))
    graph = RepairAgentGraph.from_settings(
        provider=provider,
        registry=registry(),
        settings=settings,
        clock=lambda: STARTED_AT,
    )

    final = graph.run(
        thread_id="repair-provider-failure",
        run_id=RUN_ID,
        state=make_state(),
    )
    trace = graph.build_trace("repair-provider-failure", settings=settings)

    assert final.termination_reason is TerminationReason.PROVIDER_ERROR
    assert final.baseline_test_result is not None
    assert final.failure_history[-1].kind is AgentFailureKind.PROVIDER_FAILURE
    assert final.failure_history[-1].provider_error_code is ProviderErrorCode.SCRIPT_EXHAUSTED
    failed_decision = next(
        event
        for event in reversed(trace.events)
        if event.provider_error_code is ProviderErrorCode.SCRIPT_EXHAUSTED
    )
    assert failed_decision.provider_request is not None
    assert failed_decision.provider_request.context_window is not None
    assert failed_decision.provider_response is None
    assert failed_decision.context_usage is None
    assert replay_trace(trace).final_state == final


@pytest.mark.parametrize(
    "rejection",
    [
        code
        for code in PatchErrorCode
        if code
        not in (
            PatchErrorCode.INVALID_DIFF,
            PatchErrorCode.APPLY_CONFLICT,
            PatchErrorCode.PROTECTED_TEST_PATH,
        )
    ],
)
def test_unsafe_patch_records_code_and_stops_without_retry(rejection: PatchErrorCode) -> None:
    settings = RepoPilotSettings()
    provider = FakeProvider(successful_responses()[:4])
    graph = RepairAgentGraph.from_settings(
        provider=provider,
        registry=registry(patch_rejection=rejection),
        settings=settings,
        clock=lambda: STARTED_AT,
    )

    final = graph.run(
        thread_id="repair-patch-failure",
        run_id=RUN_ID,
        state=make_state(),
    )

    assert final.termination_reason is TerminationReason.INVALID_PATCH
    assert final.patch_state.status is PatchStatus.NONE
    assert final.test_state.commands_run == 1
    assert final.failure_history[-1].kind is AgentFailureKind.PATCH_FAILURE
    assert final.failure_history[-1].error_type is ErrorType.PATCH_ERROR
    assert final.failure_history[-1].patch_error_code is rejection
    assert len(provider.requests) == 4


def _resequence(responses: list[ProviderResponse]) -> tuple[ProviderResponse, ...]:
    result = []
    for index, response in enumerate(responses):
        updates: dict[str, object] = {
            "request_id": repair_decision_request_id(RUN_ID, index),
        }
        if isinstance(response.action, ToolAction):
            updates["action"] = response.action.model_copy(
                update={
                    "tool_call": response.action.tool_call.model_copy(
                        update={
                            "call_id": uuid5(RUN_ID, f"test-call:{index}"),
                        }
                    ),
                }
            )
        result.append(response.model_copy(update=updates))
    return tuple(result)


def _run_script(
    responses: tuple[ProviderResponse, ...],
    *,
    tools: ToolRegistry | None = None,
    state: AgentState | None = None,
    settings: RepoPilotSettings | None = None,
) -> tuple[AgentState, FakeProvider, AgentTrace]:
    settings = settings or RepoPilotSettings()
    provider = FakeProvider(responses)
    graph = RepairAgentGraph.from_settings(
        provider=provider,
        registry=tools or registry(),
        settings=settings,
        clock=lambda: STARTED_AT,
    )
    final = graph.run(thread_id="repair-correction", run_id=RUN_ID, state=state or make_state())
    trace = graph.build_trace("repair-correction", settings=settings)
    assert replay_trace(trace).final_state == final
    return final, provider, trace


class ValidationSequenceTool(RunTestsTool):
    """Synthetic baseline and candidate outcomes without executing a subprocess."""

    def __init__(self, *, final_passes: bool = True) -> None:
        super().__init__()
        self.final_passes = final_passes

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        assert isinstance(arguments, RunTestsArguments)
        self.calls += 1
        passed = self.calls == 3 and self.final_passes
        current = observation(
            call.call_id, exit_code=0 if passed else 1, suffix=f"validation-{self.calls}"
        )
        result = RepoTestRunResult(
            observation_call_id=call.call_id,
            command_index=arguments.test_index,
            command_sha256="c" * 64,
            status=RepoTestStatus.PASSED if passed else RepoTestStatus.FAILED,
            category=(
                RepoTestRunCategory.PASSED if passed else RepoTestRunCategory.ASSERTION_FAILURE
            ),
            exit_code=current.exit_code,
            duration_ms=5,
            tests_run=2,
            failures=0 if passed else 1,
            failing_tests=() if passed else ("checks.test_regression",),
            stdout_ref=current.stdout_ref,
            summary="synthetic candidate pass" if passed else "synthetic candidate failure",
        )
        return RunTestsResult(observation=current, test_run=result)


class RevisionApplyTool(ApplyTool):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        result = super().execute(arguments, call=call)
        assert isinstance(result, PatchApplicationResult)
        self.calls += 1
        patch_ref = artifact(f"patch-{self.calls}.diff", str(self.calls), 64)
        assert result.summary is not None
        return result.model_copy(
            update={
                "patch_state": result.patch_state.model_copy(
                    update={"revision": self.calls, "diff_ref": patch_ref}
                ),
                "summary": result.summary.model_copy(update={"sha256": patch_ref.sha256}),
            }
        )


def _source_diff(before: int, after: int, *, path: str = "pkg/parser.py") -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n+++ b/{path}\n"
        f"@@ -1 +1 @@\n-VALUE = {before}\n+VALUE = {after}\n"
    )


def _validation_repair_responses() -> tuple[ProviderResponse, ...]:
    first = list(successful_responses()[:5])
    first[3] = tool_response(3, 3, "apply_patch", {"diff": _source_diff(1, 2)}, ("apply_patch",))
    return _resequence(
        [
            *first,
            tool_response(5, 5, "read_file", {"path": "pkg/parser.py"}, ("read_file",)),
            tool_response(6, 6, "apply_patch", {"diff": _source_diff(2, 3)}, ("apply_patch",)),
            successful_responses()[4],
            successful_responses()[5],
        ]
    )


def test_failed_candidate_can_read_current_source_repair_and_validate_once() -> None:
    patch_tool = RevisionApplyTool()

    class CurrentReadTool(ReadTool):
        def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
            del call
            assert isinstance(arguments, ReadArguments)
            return ReadResult(path=arguments.path, content=f"VALUE = {patch_tool.calls + 1}\n")

    test_tool = ValidationSequenceTool()
    final, provider, trace = _run_script(
        _validation_repair_responses(),
        tools=registry(patch_tool=patch_tool, test_tool=test_tool, read_tool=CurrentReadTool()),
    )

    assert final.termination_reason is TerminationReason.AGENT_STOPPED
    assert final.replan_count == 1
    assert final.patch_state.revision == 2
    assert patch_tool.calls == 2
    assert test_tool.calls == 3
    assert final.test_state.commands_run == 3
    assert final.baseline_test_result is not None
    assert final.baseline_test_result.status is RepoTestStatus.FAILED
    assert final.test_state.status is RepoTestStatus.PASSED
    assert [failure.kind for failure in final.failure_history] == [AgentFailureKind.TEST_FAILURE]
    assert final.failure_history[0].step_id == "validate"
    assert len(final.observations) == 5
    assert final.budget.tokens_used == 9 * usage().total_tokens
    assert final.budget.steps_used == final.decision_count == 9
    assert final.budget.started_at == make_state().budget.started_at
    assert final.budget.deadline_at == make_state().budget.deadline_at
    retry = provider.requests[5].state
    assert provider.requests[5].allowed_tools == ("read_file", "apply_patch")
    assert retry.active_step_id == "patch"
    assert tuple(step.status.value for step in retry.current_plan) == (
        "completed",
        "completed",
        "completed",
        "in_progress",
        "pending",
        "pending",
    )
    assert retry.current_plan[3].revision_reason is not None
    assert "current patched source" in retry.current_plan[3].revision_reason
    assert "incremental diff" in retry.current_plan[3].revision_reason
    assert retry.baseline_test_result == final.baseline_test_result
    assert provider.requests[6].state.tool_results[-1].payload == {
        "path": "pkg/parser.py",
        "content": "VALUE = 2\n",
    }
    assert all(step.status.value == "completed" for step in final.current_plan)
    assert replay_trace(trace).final_state == final


def test_same_patch_phase_read_can_repeat_only_after_applied_revision_changes() -> None:
    responses = list(_validation_repair_responses())
    # One PATCH-phase read before revision 1, then the exact same PATCH read after it.
    responses.insert(3, successful_responses()[2])
    final, provider, _ = _run_script(
        _resequence(responses),
        tools=registry(patch_tool=RevisionApplyTool(), test_tool=ValidationSequenceTool()),
    )
    assert final.termination_reason is TerminationReason.AGENT_STOPPED
    assert provider.requests[3].state.active_step_id == "patch"
    assert provider.requests[6].state.active_step_id == "patch"
    assert provider.requests[3].state.patch_state.revision == 0
    assert provider.requests[6].state.patch_state.revision == 1


@pytest.mark.parametrize("max_replans", [1, 3])
def test_second_failed_candidate_terminates_even_with_more_replan_budget(max_replans: int) -> None:
    patch_tool = RevisionApplyTool()
    final, provider, _ = _run_script(
        _validation_repair_responses(),
        tools=registry(patch_tool=patch_tool, test_tool=ValidationSequenceTool(final_passes=False)),
        settings=RepoPilotSettings(max_replans=max_replans),
    )
    assert final.termination_reason is TerminationReason.TESTS_FAILED
    assert final.replan_count == 1
    assert final.patch_state.revision == 2
    assert final.test_state.commands_run == 3
    assert len(provider.requests) == 8
    assert [failure.kind for failure in final.failure_history] == [
        AgentFailureKind.TEST_FAILURE,
        AgentFailureKind.TEST_FAILURE,
    ]


def test_validation_retry_is_disabled_with_zero_replans() -> None:
    final, provider, _ = _run_script(
        _validation_repair_responses(),
        tools=registry(patch_tool=RevisionApplyTool(), test_tool=ValidationSequenceTool()),
        settings=RepoPilotSettings(max_replans=0),
    )
    assert final.termination_reason is TerminationReason.TESTS_FAILED
    assert final.replan_count == 0
    assert final.patch_state.revision == 1
    assert len(provider.requests) == 5


def test_empty_search_and_validation_retry_share_replan_allowance() -> None:
    class EmptyOnceSearch(SearchTool):
        calls = 0

        def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
            self.calls += 1
            if self.calls == 1:
                return SearchResult(matches=())
            return super().execute(arguments, call=call)

    responses = list(_validation_repair_responses())
    responses.insert(2, tool_response(2, 8, "search_code", {"query": "broader"}, ("search_code",)))
    final, provider, _ = _run_script(
        _resequence(responses),
        tools=registry(
            patch_tool=RevisionApplyTool(),
            test_tool=ValidationSequenceTool(),
            search_tool=EmptyOnceSearch(),
        ),
    )
    assert final.termination_reason is TerminationReason.TESTS_FAILED
    assert final.replan_count == 1
    assert final.patch_state.revision == 1
    assert len(provider.requests) == 6
    assert [failure.kind for failure in final.failure_history] == [
        AgentFailureKind.EMPTY_SEARCH,
        AgentFailureKind.TEST_FAILURE,
    ]


@pytest.mark.parametrize(
    ("category", "error_type", "termination"),
    [
        (RepoTestRunCategory.TIMEOUT, ErrorType.TOOL_TIMEOUT, TerminationReason.TIMEOUT),
        (
            RepoTestRunCategory.SANDBOX_ERROR,
            ErrorType.SANDBOX_ERROR,
            TerminationReason.SANDBOX_ERROR,
        ),
        (RepoTestRunCategory.EXECUTION_ERROR, ErrorType.TEST_ERROR, TerminationReason.TOOL_ERROR),
    ],
)
def test_validation_infrastructure_error_never_enters_candidate_repair(
    category: RepoTestRunCategory, error_type: ErrorType, termination: TerminationReason
) -> None:
    class InfrastructureFailureTool(ValidationSequenceTool):
        def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
            result = super().execute(arguments, call=call)
            assert isinstance(result, RunTestsResult)
            if self.calls != 2:
                return result
            return RunTestsResult(
                observation=result.observation.model_copy(update={"error_type": error_type}),
                test_run=result.test_run.model_copy(
                    update={
                        "status": RepoTestStatus.ERROR,
                        "category": category,
                        "error_type": error_type,
                    }
                ),
            )

    final, provider, _ = _run_script(
        _validation_repair_responses(),
        tools=registry(patch_tool=RevisionApplyTool(), test_tool=InfrastructureFailureTool()),
    )
    assert final.termination_reason is termination
    assert final.replan_count == 0
    assert final.patch_state.revision == 1
    assert len(provider.requests) == 5


@pytest.mark.parametrize("read_kind", ["missing", "different_file", "empty"])
def test_second_patch_requires_nonempty_current_read_of_every_target(read_kind: str) -> None:
    patch_tool = RevisionApplyTool()

    class MaybeEmptyReadTool(ReadTool):
        def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
            result = super().execute(arguments, call=call)
            assert isinstance(result, ReadResult)
            if read_kind == "empty" and patch_tool.calls:
                return result.model_copy(update={"content": ""})
            return result

    responses = list(_validation_repair_responses()[:7])
    if read_kind == "missing":
        del responses[5]
    elif read_kind == "different_file":
        responses[5] = tool_response(5, 5, "read_file", {"path": "pkg/other.py"}, ("read_file",))
    final, _, trace = _run_script(
        _resequence(responses),
        tools=registry(
            patch_tool=patch_tool,
            test_tool=ValidationSequenceTool(),
            read_tool=MaybeEmptyReadTool(),
        ),
    )
    assert patch_tool.calls == 1  # The second proposal never reached the mutation tool.
    assert final.patch_state.revision == 1
    assert final.failure_history[1].patch_error_code is PatchErrorCode.APPLY_CONFLICT
    patch_event = [
        event
        for event in trace.events
        if event.tool_call is not None and event.tool_call.tool_name == "apply_patch"
    ][-1]
    assert patch_event.tool_result is not None and patch_event.tool_result.payload is not None
    assert "after the latest applied patch" in str(patch_event.tool_result.payload["message"])


def test_retry_disallows_base_only_symbol_inspection_after_source_was_patched() -> None:
    responses = list(_validation_repair_responses()[:5])
    responses.append(
        tool_response(
            5,
            5,
            "inspect_symbol",
            {"path": "pkg/parser.py", "symbol_name": "parse"},
            ("inspect_symbol",),
        )
    )
    final, provider, _ = _run_script(
        _resequence(responses),
        tools=registry(patch_tool=RevisionApplyTool(), test_tool=ValidationSequenceTool()),
    )
    assert provider.requests[-1].allowed_tools == ("read_file", "apply_patch")
    assert final.termination_reason is TerminationReason.PROVIDER_ERROR
    assert final.failure_history[-1].provider_error_code is ProviderErrorCode.TOOL_NOT_ALLOWED
    assert not any(result.tool_name == "inspect_symbol" for result in final.tool_results)
    assert final.patch_state.revision == 1


@pytest.mark.parametrize("repeat", ["patch", "read", "call_id"])
def test_validation_retry_preserves_repeat_action_and_global_call_id_guards(repeat: str) -> None:
    responses = list(_validation_repair_responses()[:7])
    if repeat == "patch":
        responses[6] = responses[3]
    elif repeat == "read":
        responses[6] = responses[5]
    responses = list(_resequence(responses))
    if repeat == "call_id":
        repeated = responses[5].action
        original = responses[2].action
        assert isinstance(repeated, ToolAction) and isinstance(original, ToolAction)
        responses[5] = responses[5].model_copy(
            update={
                "action": repeated.model_copy(
                    update={
                        "tool_call": repeated.tool_call.model_copy(
                            update={"call_id": original.tool_call.call_id}
                        )
                    }
                )
            }
        )
    patch_tool = RevisionApplyTool()
    final, _, _ = _run_script(
        tuple(responses),
        tools=registry(patch_tool=patch_tool, test_tool=ValidationSequenceTool()),
    )
    assert final.termination_reason is TerminationReason.REPEATED_ACTION
    assert final.failure_history[-1].kind is AgentFailureKind.REPEATED_ACTION
    assert patch_tool.calls == 1


@pytest.mark.parametrize(
    ("limits", "termination"),
    [
        ({"max_steps": 5}, TerminationReason.MAX_STEPS),
        ({"max_tokens": 60}, TerminationReason.TOKEN_LIMIT),
        ({"max_cost_usd": Decimal("0.005")}, TerminationReason.COST_LIMIT),
    ],
)
def test_validation_retry_does_not_reset_original_budget(
    limits: dict[str, object], termination: TerminationReason
) -> None:
    initial = make_state()
    limited = initial.model_copy(update={"budget": initial.budget.model_copy(update=limits)})
    final, provider, _ = _run_script(
        _validation_repair_responses(),
        state=limited,
        tools=registry(patch_tool=RevisionApplyTool(), test_tool=ValidationSequenceTool()),
    )
    assert final.termination_reason is termination
    assert final.replan_count == 1
    assert len(provider.requests) == 5
    assert final.patch_state.revision == 1
    assert final.budget.max_tokens == limited.budget.max_tokens
    assert final.budget.max_cost_usd == limited.budget.max_cost_usd
    assert final.budget.max_steps == limited.budget.max_steps


def test_symbol_lookup_requires_source_read_before_patch_is_allowed() -> None:
    original = list(successful_responses())
    original.insert(
        2,
        tool_response(
            2,
            5,
            "inspect_symbol",
            {"path": "pkg/parser.py", "symbol_name": "parse"},
            ("inspect_symbol", "read_file"),
        ),
    )
    final, provider, _ = _run_script(_resequence(original))

    assert final.termination_reason is TerminationReason.AGENT_STOPPED
    assert provider.requests[3].allowed_tools == ("inspect_symbol", "read_file")
    assert provider.requests[4].allowed_tools == ("read_file", "inspect_symbol", "apply_patch")
    assert final.decision_count == 7


def test_empty_source_read_does_not_advance_to_patch() -> None:
    class EmptyReadTool(ReadTool):
        def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
            del call
            assert isinstance(arguments, ReadArguments)
            return ReadResult(path=arguments.path, content="")

    final, provider, _ = _run_script(
        successful_responses()[:3], tools=registry(read_tool=EmptyReadTool())
    )

    assert final.patch_state.status is PatchStatus.NONE
    assert provider.requests[-1].allowed_tools == ("inspect_symbol", "read_file")
    assert final.termination_reason is TerminationReason.PROVIDER_ERROR


@pytest.mark.parametrize(
    "rejection",
    [
        PatchErrorCode.INVALID_DIFF,
        PatchErrorCode.APPLY_CONFLICT,
        PatchErrorCode.PROTECTED_TEST_PATH,
    ],
)
def test_recoverable_patch_error_allows_exactly_one_correction(rejection: PatchErrorCode) -> None:
    responses = list(successful_responses()[:4])
    responses.append(
        tool_response(4, 5, "apply_patch", {"diff": "different-corrected-diff"}, ("apply_patch",))
    )
    final, provider, _ = _run_script(tuple(responses), tools=registry(patch_rejection=rejection))

    assert final.termination_reason is TerminationReason.INVALID_PATCH
    assert len(provider.requests) == 5
    assert [failure.patch_error_code for failure in final.failure_history] == [rejection, rejection]
    assert final.test_state.commands_run == 1


@pytest.mark.parametrize(
    "rejection", [PatchErrorCode.APPLY_CONFLICT, PatchErrorCode.PROTECTED_TEST_PATH]
)
def test_patch_correction_can_read_more_source_then_succeed(rejection: PatchErrorCode) -> None:
    class RejectOnceTool(ApplyTool):
        def __init__(self) -> None:
            super().__init__(rejection=rejection)
            self.calls = 0

        def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
            self.calls += 1
            result = super().execute(arguments, call=call)
            self.rejection = None
            return result

    responses = [
        *successful_responses()[:4],
        tool_response(4, 5, "read_file", {"path": "pkg/parser.py"}, ("read_file",)),
        tool_response(5, 6, "apply_patch", {"diff": "corrected-diff"}, ("apply_patch",)),
        *successful_responses()[4:],
    ]
    patch_tool = RejectOnceTool()
    final, provider, _ = _run_script(_resequence(responses), tools=registry(patch_tool=patch_tool))

    assert final.termination_reason is TerminationReason.AGENT_STOPPED
    assert patch_tool.calls == 2
    assert len(final.failure_history) == 1
    assert provider.requests[5].allowed_tools == ("read_file", "inspect_symbol", "apply_patch")
    if rejection is PatchErrorCode.PROTECTED_TEST_PATH:
        reason = provider.requests[4].state.current_plan[3].revision_reason
        assert reason is not None
        assert "before any source changes were applied" in reason
        assert "source-only diff" in reason
        assert "read-only" in reason


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (PatchErrorCode.INVALID_DIFF, PatchErrorCode.PROTECTED_TEST_PATH),
        (PatchErrorCode.PROTECTED_TEST_PATH, PatchErrorCode.APPLY_CONFLICT),
    ],
)
def test_protected_tests_share_one_correction_with_format_errors(
    first: PatchErrorCode, second: PatchErrorCode
) -> None:
    class SequentialRejectionTool(ApplyTool):
        def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
            result = super().execute(arguments, call=call)
            self.rejection = second
            return result

    responses = [
        *successful_responses()[:4],
        tool_response(4, 5, "apply_patch", {"diff": "distinct-corrected-diff"}, ("apply_patch",)),
    ]
    final, provider, _ = _run_script(
        _resequence(responses), tools=registry(patch_tool=SequentialRejectionTool(rejection=first))
    )

    assert final.termination_reason is TerminationReason.INVALID_PATCH
    assert len(provider.requests) == 5
    assert [failure.patch_error_code for failure in final.failure_history] == [first, second]
    assert final.patch_state.status is PatchStatus.NONE


def test_protected_test_correction_requires_a_new_whole_source_only_candidate() -> None:
    source_diff = (
        "diff --git a/pkg/parser.py b/pkg/parser.py\n"
        "--- a/pkg/parser.py\n"
        "+++ b/pkg/parser.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )
    mixed_diff = source_diff + source_diff.replace("pkg/parser.py", "tests/test_parser.py")

    class RecordingProtectedTool(ApplyTool):
        def __init__(self) -> None:
            super().__init__()
            self.proposals: list[str] = []

        def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
            assert isinstance(arguments, ApplyArguments)
            self.proposals.append(arguments.diff)
            self.rejection = (
                PatchErrorCode.PROTECTED_TEST_PATH
                if "tests/test_parser.py" in arguments.diff
                else None
            )
            return super().execute(arguments, call=call)

    responses = [
        *successful_responses()[:3],
        tool_response(3, 5, "read_file", {"path": "tests/test_parser.py"}, ("read_file",)),
        tool_response(4, 6, "apply_patch", {"diff": mixed_diff}, ("apply_patch",)),
        tool_response(5, 7, "apply_patch", {"diff": source_diff}, ("apply_patch",)),
        *successful_responses()[4:],
    ]
    tool = RecordingProtectedTool()
    final, provider, trace = _run_script(_resequence(responses), tools=registry(patch_tool=tool))

    assert final.termination_reason is TerminationReason.AGENT_STOPPED
    assert final.patch_state.files_changed == ("pkg/parser.py",)
    assert final.test_state.status is RepoTestStatus.PASSED
    assert tool.proposals == [mixed_diff, source_diff]  # Nothing silently strips the first diff.
    assert provider.requests[5].state.patch_state.status is PatchStatus.NONE
    assert len(final.failure_history) == 1
    assert final.failure_history[0].patch_error_code is PatchErrorCode.PROTECTED_TEST_PATH
    assert replay_trace(trace).final_state == final


@pytest.mark.parametrize("uncertain", ["missing_observation", "wrong_exit", "changed_state"])
def test_protected_test_code_without_no_write_evidence_is_not_retried(uncertain: str) -> None:
    class UnconfirmedTool(ApplyTool):
        def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
            result = super().execute(arguments, call=call)
            assert isinstance(result, PatchApplicationResult)
            if uncertain == "missing_observation":
                return result.model_copy(update={"observation": None})
            if uncertain == "wrong_exit":
                assert result.observation is not None
                return result.model_copy(
                    update={"observation": result.observation.model_copy(update={"exit_code": 1})}
                )
            applied = ApplyTool().execute(arguments, call=call)
            assert isinstance(applied, PatchApplicationResult)
            return result.model_copy(update={"patch_state": applied.patch_state})

    final, provider, _ = _run_script(
        successful_responses()[:4],
        tools=registry(patch_tool=UnconfirmedTool(rejection=PatchErrorCode.PROTECTED_TEST_PATH)),
    )
    assert final.termination_reason is TerminationReason.INVALID_PATCH
    assert len(provider.requests) == 4


def test_each_fake_decision_exposes_remaining_budget_and_bounded_read_guidance() -> None:
    final, provider, _ = _run_script(successful_responses())

    assert final.termination_reason is TerminationReason.AGENT_STOPPED
    remaining_tokens = []
    for request in provider.requests:
        assert request.context_window is not None
        rendered = build_agent_prompt(
            request, max_chars=RepoPilotSettings().openai_prompt_max_chars
        )
        prompt = json.loads(rendered.split("\n", 1)[1].rsplit("\n", 1)[0])
        budget = prompt["state"]["budget"]
        original = request.state.budget
        assert budget["steps_remaining"] == original.max_steps - original.steps_used
        assert budget["tokens_remaining"] == original.max_tokens - original.tokens_used
        assert Decimal(budget["cost_remaining_usd"]) == (
            original.max_cost_usd - original.cost_used_usd
        )
        assert budget["max_tokens"] == make_state().budget.max_tokens
        remaining_tokens.append(budget["tokens_remaining"])
    assert remaining_tokens == sorted(remaining_tokens, reverse=True)
    assert len(set(remaining_tokens)) == len(remaining_tokens)
    assert "small, targeted read_file windows" in SYSTEM_INSTRUCTIONS
    assert "apply_patch, run_tests, and finish" in SYSTEM_INSTRUCTIONS
    assert "Tests and evaluator inputs are read-only" in SYSTEM_INSTRUCTIONS


def test_graph_preserves_typed_context_accounting_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = ContextAccountingError(ContextErrorCode.ARTIFACT_REQUIRED)

    def fail_context(**_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("repopilot.agent.repair.build_context_aware_provider_request", fail_context)
    provider = FakeProvider(successful_responses())
    graph = RepairAgentGraph.from_settings(
        provider=provider,
        registry=registry(),
        settings=RepoPilotSettings(),
        clock=lambda: STARTED_AT,
    )

    with pytest.raises(ContextAccountingError) as caught:
        graph.run(thread_id="typed-context-failure", run_id=RUN_ID, state=make_state())

    assert caught.value is failure
    assert caught.value.code is ContextErrorCode.ARTIFACT_REQUIRED
    assert not provider.requests
    assert graph.partial_trace_events(RUN_ID)


def test_patch_correction_does_not_bypass_repeated_action_guard() -> None:
    responses = list(successful_responses()[:4])
    responses.append(successful_responses()[3])
    final, _, _ = _run_script(
        _resequence(responses), tools=registry(patch_rejection=PatchErrorCode.INVALID_DIFF)
    )
    assert final.termination_reason is TerminationReason.REPEATED_ACTION
    assert (
        sum(failure.kind is AgentFailureKind.PATCH_FAILURE for failure in final.failure_history)
        == 1
    )


@pytest.mark.parametrize(
    ("budget_update", "termination"),
    [
        ({"max_steps": 4}, TerminationReason.MAX_STEPS),
        ({"max_tokens": 48}, TerminationReason.TOKEN_LIMIT),
        ({"max_cost_usd": Decimal("0.004")}, TerminationReason.COST_LIMIT),
    ],
)
@pytest.mark.parametrize(
    "rejection", [PatchErrorCode.INVALID_DIFF, PatchErrorCode.PROTECTED_TEST_PATH]
)
def test_patch_correction_remains_inside_original_budget(
    budget_update: dict[str, object], termination: TerminationReason, rejection: PatchErrorCode
) -> None:
    initial = make_state()
    limited = initial.model_copy(
        update={
            "budget": initial.budget.model_copy(update=budget_update),
        }
    )
    final, provider, _ = _run_script(
        successful_responses()[:4],
        tools=registry(patch_rejection=rejection),
        state=limited,
    )
    assert final.termination_reason is termination
    assert len(provider.requests) == 4
    assert len(final.failure_history) == 1
    assert final.test_state.commands_run == 1


@pytest.mark.parametrize("include_read_target", [False, True])
def test_patch_target_must_have_source_read_even_after_reading_another_file(
    include_read_target: bool,
) -> None:
    unread_diff = (
        "diff --git a/pkg/other.py b/pkg/other.py\n"
        "--- a/pkg/other.py\n+++ b/pkg/other.py\n"
        "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
    )
    if include_read_target:
        unread_diff = (
            "diff --git a/pkg/parser.py b/pkg/parser.py\n"
            "--- a/pkg/parser.py\n+++ b/pkg/parser.py\n"
            "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n" + unread_diff
        )

    class CountingApplyTool(ApplyTool):
        calls = 0

        def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
            self.calls += 1
            return super().execute(arguments, call=call)

    patch_tool = CountingApplyTool()
    responses = [
        *successful_responses()[:3],
        tool_response(3, 3, "apply_patch", {"diff": unread_diff}, ("apply_patch",)),
    ]
    final, _, trace = _run_script(tuple(responses), tools=registry(patch_tool=patch_tool))

    assert patch_tool.calls == 0
    assert final.patch_state.status is PatchStatus.NONE
    assert final.failure_history[0].patch_error_code is PatchErrorCode.APPLY_CONFLICT
    tool_event = next(
        event
        for event in trace.events
        if event.tool_call is not None and event.tool_call.tool_name == "apply_patch"
    )
    assert tool_event.tool_result is not None and tool_event.tool_result.payload is not None
    assert "pkg/other.py" in str(tool_event.tool_result.payload["message"])


def test_patch_safety_rejection_is_not_masked_by_unread_target_check() -> None:
    settings = RepoPilotSettings(patch_max_changed_lines=1)
    responses = [
        *successful_responses()[:3],
        tool_response(
            3,
            3,
            "apply_patch",
            {
                "diff": (
                    "diff --git a/pkg/unread.py b/pkg/unread.py\n"
                    "--- a/pkg/unread.py\n+++ b/pkg/unread.py\n"
                    "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
                )
            },
            ("apply_patch",),
        ),
    ]
    provider = FakeProvider(responses)
    graph = RepairAgentGraph.from_settings(
        provider=provider,
        registry=registry(patch_rejection=PatchErrorCode.TOO_MANY_CHANGED_LINES),
        settings=settings,
        clock=lambda: STARTED_AT,
    )
    final = graph.run(thread_id="patch-size-priority", run_id=RUN_ID, state=make_state())
    assert final.termination_reason is TerminationReason.INVALID_PATCH
    assert len(provider.requests) == 4
    assert final.failure_history[-1].patch_error_code is PatchErrorCode.TOO_MANY_CHANGED_LINES


def test_patch_phase_source_read_keeps_patch_pending_and_allows_its_target() -> None:
    read_diff = (
        "diff --git a/pkg/parser.py b/pkg/parser.py\n"
        "--- a/pkg/parser.py\n+++ b/pkg/parser.py\n"
        "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
    )
    responses = [
        *successful_responses()[:2],
        tool_response(2, 2, "read_file", {"path": "pkg/another.py"}, ("read_file",)),
        tool_response(3, 5, "read_file", {"path": "pkg/parser.py"}, ("read_file",)),
        tool_response(4, 3, "apply_patch", {"diff": read_diff}, ("apply_patch",)),
        *successful_responses()[4:],
    ]
    final, provider, _ = _run_script(_resequence(responses))

    assert final.termination_reason is TerminationReason.AGENT_STOPPED
    assert not final.failure_history
    assert provider.requests[4].allowed_tools == ("read_file", "inspect_symbol", "apply_patch")


def test_repair_graph_resumes_initialized_working_memory_with_continuous_trace() -> None:
    initial = AgentState.model_validate(
        {
            **make_state().model_dump(mode="python"),
            "run_id": RUN_ID,
        }
    )
    provider = FakeProvider(successful_responses())
    tools = registry()
    nodes = RepairAgentNodes(
        provider=provider,
        registry=tools,
        clock=lambda: STARTED_AT,
    )
    initialized = apply_state_update(
        initial,
        nodes.initialize(to_graph_state(initial)),
    )
    pending = apply_state_update(
        initialized,
        nodes.decide(to_graph_state(initialized)),
    )
    events = nodes.trace_recorder.events(RUN_ID)
    checkpoint = build_working_memory_checkpoint(
        task_id=pending.task.task_id,
        run_id=RUN_ID,
        source_thread_id="repair-interrupted",
        repository_commit=pending.task.base_commit,
        context_policy_version=context_policy_version_from_events(events),
        sequence=0,
        created_at=STARTED_AT,
        state=pending,
        trace_events=events,
    )
    graph = RepairAgentGraph(
        provider=provider,
        registry=tools,
        clock=lambda: STARTED_AT,
    )

    final = graph.resume_from_working_memory(
        thread_id="repair-recovered",
        checkpoint=checkpoint,
    )
    trace = graph.build_trace("repair-recovered", settings=RepoPilotSettings())

    assert final.termination_reason is TerminationReason.AGENT_STOPPED
    assert provider.calls_made == 6
    assert sum(event.node is TraceNode.INITIALIZE_PLAN for event in trace.events) == 1
    assert tuple(event.sequence for event in trace.events) == tuple(range(len(trace.events)))
    assert replay_trace(trace).final_state == final


def test_verified_episode_gate_persists_only_replayed_verifier_success(
    tmp_path: Path,
) -> None:
    final, trace, result, execution = verified_source()
    store = FileVerifiedEpisodicMemoryStore(tmp_path / "verified-episodes")
    artifact_root = tmp_path / "source-artifacts"
    materialize_verified_artifacts(artifact_root, execution)

    record = store.admit(
        repository_id="repo_verified_unit",
        repository_content_sha256="9" * 64,
        result=result,
        trace=trace,
        execution=execution,
        artifact_root=artifact_root,
        created_at=STARTED_AT,
    )
    duplicate = store.admit(
        repository_id="repo_verified_unit",
        repository_content_sha256="9" * 64,
        result=result,
        trace=trace,
        execution=execution,
        artifact_root=artifact_root,
        created_at=STARTED_AT + timedelta(days=1),
    )
    current = store.lookup(
        repository_id="repo_verified_unit",
        repository_commit=final.task.base_commit,
        repository_content_sha256="9" * 64,
        as_of=STARTED_AT + timedelta(days=29),
    )
    expired = store.lookup(
        repository_id="repo_verified_unit",
        repository_commit=final.task.base_commit,
        repository_content_sha256="9" * 64,
        as_of=STARTED_AT + timedelta(days=30),
    )
    wrong_content = store.lookup(
        repository_id="repo_verified_unit",
        repository_commit=final.task.base_commit,
        repository_content_sha256="8" * 64,
        as_of=STARTED_AT,
    )

    assert duplicate == record
    assert current.status is VerifiedEpisodeLookupStatus.CURRENT
    assert current.records == (record,)
    assert expired.status is VerifiedEpisodeLookupStatus.EXPIRED
    assert expired.records == ()
    assert expired.expired_record_count == 1
    assert wrong_content.status is VerifiedEpisodeLookupStatus.EMPTY
    assert wrong_content.records == ()
    assert record.problem_statement == final.task.issue_title
    assert record.repair_summary == "Verified repair changed 1 file(s) and passed 2 test(s)."
    assert record.evidence.source_trace_sha256 == trace.content_sha256
    assert record.provenance.model_provider == "fake-deterministic"
    assert record.provenance.model_versions == ("fake-repair-v1",)
    assert "Evidence is ready for deterministic verification." not in record.model_dump_json()
    stored_path = next((tmp_path / "verified-episodes").rglob("*.json"))
    assert stored_path.stat().st_mode & 0o077 == 0


def test_verified_episode_gate_rejects_failure_tampering_and_missing_artifacts(
    tmp_path: Path,
) -> None:
    _, trace, result, execution = verified_source()
    artifact_root = tmp_path / "source-artifacts"
    materialize_verified_artifacts(artifact_root, execution)

    forged_failure = result.model_copy(update={"status": RunStatus.FAILED})
    with pytest.raises(VerifiedEpisodeError) as captured:
        build_verified_episode_record(
            repository_id="repo_verified_unit",
            repository_content_sha256="9" * 64,
            result=forged_failure,
            trace=trace,
            execution=execution,
            artifact_root=artifact_root,
            created_at=STARTED_AT,
        )
    assert captured.value.code is VerifiedEpisodeErrorCode.VERIFIER_REJECTED

    forged_result = result.model_copy(update={"patch_path": "artifacts/forged.diff"})
    with pytest.raises(VerifiedEpisodeError) as captured:
        build_verified_episode_record(
            repository_id="repo_verified_unit",
            repository_content_sha256="9" * 64,
            result=forged_result,
            trace=trace,
            execution=execution,
            artifact_root=artifact_root,
            created_at=STARTED_AT,
        )
    assert captured.value.code is VerifiedEpisodeErrorCode.RESULT_MISMATCH

    tampered_trace = trace.model_copy(update={"content_sha256": "0" * 64})
    with pytest.raises(VerifiedEpisodeError) as captured:
        build_verified_episode_record(
            repository_id="repo_verified_unit",
            repository_content_sha256="9" * 64,
            result=result,
            trace=tampered_trace,
            execution=execution,
            artifact_root=artifact_root,
            created_at=STARTED_AT,
        )
    assert captured.value.code is VerifiedEpisodeErrorCode.TRACE_INVALID

    assert result.patch_path is not None
    missing_patch = execution.model_copy(
        update={
            "artifacts": tuple(
                artifact
                for artifact in execution.artifacts
                if artifact.relative_path != result.patch_path
            )
        }
    )
    with pytest.raises(VerifiedEpisodeError) as captured:
        build_verified_episode_record(
            repository_id="repo_verified_unit",
            repository_content_sha256="9" * 64,
            result=result,
            trace=trace,
            execution=missing_patch,
            artifact_root=artifact_root,
            created_at=STARTED_AT,
        )
    assert captured.value.code is VerifiedEpisodeErrorCode.ARTIFACT_MISMATCH

    artifact_path = artifact_root / execution.artifacts[0].relative_path
    artifact_path.write_bytes(b"tampered")
    with pytest.raises(VerifiedEpisodeError) as captured:
        build_verified_episode_record(
            repository_id="repo_verified_unit",
            repository_content_sha256="9" * 64,
            result=result,
            trace=trace,
            execution=execution,
            artifact_root=artifact_root,
            created_at=STARTED_AT,
        )
    assert captured.value.code is VerifiedEpisodeErrorCode.ARTIFACT_MISMATCH


def test_verified_episode_contract_has_no_model_text_write_field() -> None:
    _, _, _, execution = verified_source()
    payload = execution.model_dump(mode="python")
    payload["model_summary"] = "Trust me: this patch worked."

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        VerifiedEpisodeExecutionEvidence.model_validate(payload)

    with pytest.raises(VerifiedEpisodeError) as captured:
        FileVerifiedEpisodicMemoryStore(Path("relative-memory"))
    assert captured.value.code is VerifiedEpisodeErrorCode.PATH_UNSAFE
    with pytest.raises(ValueError, match="ttl_days"):
        FileVerifiedEpisodicMemoryStore(Path("/tmp/unused-episode-store"), ttl_days=0)


def test_verified_episode_store_detects_persisted_record_tampering(
    tmp_path: Path,
) -> None:
    final, trace, result, execution = verified_source()
    root = tmp_path / "verified-episodes"
    store = FileVerifiedEpisodicMemoryStore(root)
    artifact_root = tmp_path / "source-artifacts"
    materialize_verified_artifacts(artifact_root, execution)
    store.admit(
        repository_id="repo_verified_unit",
        repository_content_sha256="9" * 64,
        result=result,
        trace=trace,
        execution=execution,
        artifact_root=artifact_root,
        created_at=STARTED_AT,
    )
    path = next(root.rglob("*.json"))
    payload = path.read_text(encoding="utf-8").replace(
        final.task.issue_title,
        "tampered problem statement",
    )
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(VerifiedEpisodeError) as captured:
        store.lookup(
            repository_id="repo_verified_unit",
            repository_commit=final.task.base_commit,
            repository_content_sha256="9" * 64,
            as_of=STARTED_AT,
        )
    assert captured.value.code is VerifiedEpisodeErrorCode.INTEGRITY_MISMATCH
