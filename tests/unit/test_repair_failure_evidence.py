"""Failed Agent prefixes retain useful metadata without prompts, secrets, or success claims."""

from __future__ import annotations

import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from test_repair_workflow import IMAGE, RUN_ID, STARTED_AT, _repository, _response, _task

import repopilot.repair_workflow as workflow
from repopilot.agent import AgentLoopError, AgentStateGraphError, RepairAgentGraph
from repopilot.config import RepoPilotSettings
from repopilot.context import ContextAccountingError
from repopilot.enums import (
    AgentLoopErrorCode,
    AgentStateErrorCode,
    ContextErrorCode,
    ErrorType,
    RepairContextMode,
    RepairWorkflowErrorCode,
    SandboxLifecycle,
    ToolResultStatus,
    TraceNode,
)
from repopilot.providers import (
    FakeProvider,
    FinishAction,
    ProviderRequest,
    ProviderResponse,
    ProviderUsageDelta,
)
from repopilot.sandbox import FakeSandbox, SandboxHandle
from repopilot.schemas import AgentState, BudgetState, ToolCall, ToolExecutionResult
from repopilot.trace import TraceEvent

SECRET = "sk-private-key postgresql://private:secret@host/db PROMPT_PRIVATE"


@pytest.mark.parametrize(
    ("error", "cause"),
    (
        (ContextAccountingError(ContextErrorCode.ARTIFACT_REQUIRED), "artifact_required"),
        (AgentStateGraphError(AgentStateErrorCode.CHECKPOINT_ERROR), "checkpoint_error"),
        (
            AgentLoopError(
                AgentLoopErrorCode.INVALID_STATE_TRANSITION, error_type=ErrorType.INTERNAL_ERROR
            ),
            "invalid_state_transition",
        ),
    ),
)
def test_agent_enum_causes_survive_workflow_classification(
    error: BaseException, cause: str
) -> None:
    assert workflow.safe_repair_failure_detail(error, stage="agent").cause_code == cause


def test_graph_cause_rejects_arbitrary_payload() -> None:
    with pytest.raises(ValueError, match="unapproved"):
        AgentStateGraphError(AgentStateErrorCode.CHECKPOINT_ERROR, cause_type=SECRET)  # type: ignore[arg-type]


def test_context_error_has_safe_original_taxonomy_in_incomplete_evidence(tmp_path: Path) -> None:
    task = _task(str(_repository(tmp_path)))
    request = workflow.RepairRunRequest(
        thread_id="context-failure-prefix",
        run_id=RUN_ID,
        started_at=STARTED_AT,
        task=task,
        sandbox_image_reference=IMAGE,
        provider_responses=(_response(),),
    )
    error = ContextAccountingError(ContextErrorCode.ARTIFACT_REQUIRED)
    evidence = workflow._incomplete_agent_evidence(
        request=request,
        error=error,
        stage="agent",
        events=(),
    )
    assert evidence.failure.cause_code == "artifact_required"
    assert evidence.exception_kind == "context_accounting_error"
    assert evidence.provider_requests_made == "unknown"
    assert evidence.budget_reserve_unchanged


def test_workflow_failure_persists_only_safe_prefix_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task(str(_repository(tmp_path)))
    request = workflow.RepairRunRequest(
        thread_id="repair-failure-prefix",
        run_id=RUN_ID,
        started_at=STARTED_AT,
        task=task,
        sandbox_image_reference=IMAGE,
        provider_responses=(_response(),),
        context_mode=RepairContextMode.NONE,
    )
    state = AgentState(
        task=task,
        run_id=RUN_ID,
        repository_summary=SECRET,
        budget=BudgetState(
            max_steps=30,
            max_tokens=1000,
            max_cost_usd=Decimal("1"),
            started_at=STARTED_AT,
            deadline_at=STARTED_AT + timedelta(minutes=15),
        ),
    )
    response = ProviderResponse(
        request_id=RUN_ID,
        action=FinishAction(summary=SECRET),
        usage=ProviderUsageDelta(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            estimated_cost_usd=Decimal("0.1"),
            latency_ms=3,
        ),
    )
    tool_call = ToolCall(
        call_id=RUN_ID,
        tool_name="run_tests",
        validated_arguments={"private": SECRET},
        requested_at=STARTED_AT,
    )
    events = (
        TraceEvent(
            sequence=0,
            node=TraceNode.DECIDE,
            state_before=state,
            state_after=state,
            provider_request=ProviderRequest(
                request_id=RUN_ID,
                run_id=RUN_ID,
                state=state,
                requested_at=STARTED_AT,
            ),
            provider_response=response,
        ),
        TraceEvent(
            sequence=1,
            node=TraceNode.EXECUTE_TOOL,
            state_before=state,
            state_after=state,
            tool_call=tool_call,
            tool_result=ToolExecutionResult(
                call_id=RUN_ID,
                tool_name="run_tests",
                status=ToolResultStatus.SUCCESS,
                payload={"private": SECRET},
            ),
        ),
    )

    graphs: list[RepairAgentGraph] = []

    def fail_run(self: RepairAgentGraph, **kwargs: object) -> AgentState:
        graphs.append(self)
        raise AgentStateGraphError(
            AgentStateErrorCode.CHECKPOINT_ERROR, cause_type="graph_recursion_limit"
        ) from RuntimeError(SECRET)

    monkeypatch.setattr(RepairAgentGraph, "run", fail_run)
    monkeypatch.setattr(RepairAgentGraph, "partial_trace_events", lambda self, run_id: events)
    sandbox = FakeSandbox(
        handle=SandboxHandle(
            sandbox_id="sandbox-failure-prefix",
            task_id=task.task_id,
            repository_commit=task.base_commit,
            prepared_at=STARTED_AT,
        ),
        steps=(),
    )
    output = tmp_path / "run"
    with pytest.raises(workflow.RepairWorkflowError) as failure:
        workflow.run_repair_request(
            request,
            output_dir=output,
            settings=RepoPilotSettings(),
            sandbox=sandbox,
        )
    assert failure.value.code is RepairWorkflowErrorCode.AGENT_FAILURE
    assert failure.value.detail.cause_code == "checkpoint_error"
    assert sandbox.lifecycle is SandboxLifecycle.DESTROYED
    path = output / workflow.INCOMPLETE_AGENT_EVIDENCE_FILENAME
    payload = json.loads(path.read_text())
    assert payload["exception_kind"] == "agent_state_graph_error"
    assert payload["graph_cause_type"] == "graph_recursion_limit"
    assert payload["last_completed_node"] == "execute_tool"
    assert payload["last_tool_name"] == "run_tests"
    assert payload["recorded_provider_responses"] == 1
    assert payload["recorded_usage_lower_bound"]["estimated_cost_usd"] == "0.1"
    assert payload["recorded_usage_lower_bound"]["total_tokens"] == 15
    assert payload["provider_requests_made"] == "unknown"
    assert payload["budget_reserve_unchanged"] is True
    assert payload["complete"] is False
    assert payload["replayable"] is False
    assert path.stat().st_mode & 0o777 == 0o600
    assert not any(
        (output / name).exists() for name in ("result.json", "trace.json", "artifacts.json")
    )
    for forbidden in (
        SECRET,
        "PROMPT_PRIVATE",
        "postgresql://",
        "validated_arguments",
        "RuntimeError",
    ):
        assert forbidden not in path.read_text()

    original = path.read_bytes()
    workflow._persist_incomplete_agent_evidence(
        root=output,
        request=request,
        graph=graphs[0],
        error=RuntimeError(SECRET),
        stage="agent",
    )
    assert path.read_bytes() == original


def test_unrecognized_exception_class_is_not_serialized(tmp_path: Path) -> None:
    class PrivateExceptionName(RuntimeError):
        pass

    task = _task(str(_repository(tmp_path)))
    request = workflow.RepairRunRequest(
        thread_id="repair-failure-prefix",
        run_id=RUN_ID,
        started_at=STARTED_AT,
        task=task,
        sandbox_image_reference=IMAGE,
        provider_responses=(_response(),),
    )
    evidence = workflow._incomplete_agent_evidence(
        request=request,
        error=PrivateExceptionName(SECRET),
        stage="agent",
        events=(),
    )
    assert evidence.exception_kind == "unexpected_exception"
    assert evidence.failure.cause_code == "unexpected_exception"
    assert evidence.provider_requests_made == "unknown"
    assert evidence.recorded_provider_responses == 0
    assert SECRET not in evidence.model_dump_json()
    assert "PrivateExceptionName" not in evidence.model_dump_json()


def test_real_graph_failure_exposes_only_completed_prefix_and_safe_cause() -> None:
    from test_repair_agent import RUN_ID as GRAPH_RUN_ID
    from test_repair_agent import STARTED_AT as GRAPH_STARTED_AT
    from test_repair_agent import make_state, registry, successful_responses

    class InterruptedProvider(FakeProvider):
        def generate(self, request: ProviderRequest) -> ProviderResponse:
            if self.requests:
                raise OSError(SECRET)
            return super().generate(request)

    graph = RepairAgentGraph.from_settings(
        provider=InterruptedProvider(successful_responses()),
        registry=registry(),
        settings=RepoPilotSettings(),
        clock=lambda: GRAPH_STARTED_AT,
    )
    with pytest.raises(AgentStateGraphError) as failure:
        graph.run(thread_id="real-interrupted-prefix", run_id=GRAPH_RUN_ID, state=make_state())

    assert failure.value.code is AgentStateErrorCode.CHECKPOINT_ERROR
    assert failure.value.cause_type == "local_io_error"
    assert SECRET not in str(failure.value)
    events = graph.partial_trace_events(GRAPH_RUN_ID)
    assert [event.node for event in events] == [
        TraceNode.INITIALIZE_PLAN,
        TraceNode.DECIDE,
        TraceNode.EXECUTE_TOOL,
        TraceNode.OBSERVE,
    ]
    assert events[2].tool_call is not None
    assert events[2].tool_call.tool_name == "run_tests"
    assert sum(event.provider_response is not None for event in events) == 1
