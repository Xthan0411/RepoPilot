"""Deterministic tests for the complete repair Agent state machine."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID

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
from repopilot.enums import (
    AgentFailureKind,
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


def registry(*, patch_rejection: PatchErrorCode | None = None) -> ToolRegistry:
    result = ToolRegistry()
    for tool in (
        SearchTool(),
        ReadTool(),
        InspectTool(),
        ApplyTool(rejection=patch_rejection),
        RunTestsTool(),
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


def test_rejected_patch_records_code_and_stops_before_validation() -> None:
    settings = RepoPilotSettings()
    provider = FakeProvider(successful_responses()[:4])
    graph = RepairAgentGraph.from_settings(
        provider=provider,
        registry=registry(patch_rejection=PatchErrorCode.APPLY_CONFLICT),
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
    assert final.failure_history[-1].patch_error_code is PatchErrorCode.APPLY_CONFLICT


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
