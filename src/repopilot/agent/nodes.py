"""Deterministic Agent nodes, recovery policy, and terminal routing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID, uuid5

from repopilot.agent.budget import BudgetController
from repopilot.agent.state import AgentGraphState, AgentGraphUpdate, from_graph_state
from repopilot.agent.trace_recorder import TraceRecorder
from repopilot.enums import (
    AgentFailureKind,
    AgentLoopErrorCode,
    ErrorType,
    PlanStepStatus,
    ProviderErrorCode,
    TerminationReason,
    TestStatus,
    ToolResultStatus,
    TraceNode,
)
from repopilot.providers.base import (
    FinishAction,
    ModelProviderAdapter,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    ProviderUsageDelta,
    ToolAction,
)
from repopilot.providers.prompts import build_context_aware_provider_request
from repopilot.schemas import (
    AgentFailure,
    AgentState,
    Observation,
    PlanStep,
    TestRunResult,
    ToolCall,
    ToolExecutionResult,
)
from repopilot.tools.registry import ToolRegistry
from repopilot.tools.tests import advance_test_state

SEARCH_STEP = "search"
READ_STEP = "read"
TEST_STEP = "test"
FINISH_STEP = "finish"
EXPECTED_TOOL_BY_STEP = {
    SEARCH_STEP: "search_code",
    READ_STEP: "read_file",
    TEST_STEP: "run_tests",
}
LOOP_ERROR_MESSAGES = {
    AgentLoopErrorCode.INVALID_STATE_TRANSITION: "Agent state transition is invalid",
    AgentLoopErrorCode.PROVIDER_FAILURE: "Provider decision failed",
    AgentLoopErrorCode.TOOL_FAILURE: "tool execution failed",
    AgentLoopErrorCode.UNFINISHED_PENDING_ACTION: "previous Agent action is unfinished",
}
ZERO_USAGE = ProviderUsageDelta(
    input_tokens=0,
    output_tokens=0,
    total_tokens=0,
    estimated_cost_usd=Decimal("0"),
    latency_ms=0,
)
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class _DecisionOutcome:
    update: AgentGraphUpdate
    request: ProviderRequest | None = None
    response: ProviderResponse | None = None
    error_type: ErrorType | None = None
    error_code: ProviderErrorCode | None = None


class AgentLoopError(RuntimeError):
    """Safe typed failure from an invalid graph transition."""

    def __init__(
        self,
        code: AgentLoopErrorCode,
        *,
        error_type: ErrorType,
    ) -> None:
        super().__init__(LOOP_ERROR_MESSAGES[code])
        self.code = code
        self.error_type = error_type


def decision_request_id(run_id: UUID, decision_index: int) -> UUID:
    """Derive a stable request ID from the run and zero-based decision index."""

    if decision_index < 0:
        raise ValueError("decision_index must not be negative")
    return uuid5(run_id, f"decision:{decision_index}")


def initialize_plan(state: AgentGraphState) -> AgentGraphUpdate:
    """Create the fixed evidence path for an otherwise fresh run."""

    snapshot = from_graph_state(state)
    if snapshot.run_id is None or snapshot.termination_reason is not None:
        raise AgentLoopError(
            AgentLoopErrorCode.INVALID_STATE_TRANSITION,
            error_type=ErrorType.VALIDATION_ERROR,
        )
    if (
        snapshot.current_plan
        or snapshot.active_step_id is not None
        or snapshot.pending_tool_call is not None
        or snapshot.pending_tool_result is not None
        or snapshot.tool_results
        or snapshot.failure_history
        or snapshot.action_fingerprints
        or snapshot.replan_count != 0
        or snapshot.decision_count != 0
        or snapshot.finish_summary is not None
    ):
        raise AgentLoopError(
            AgentLoopErrorCode.INVALID_STATE_TRANSITION,
            error_type=ErrorType.VALIDATION_ERROR,
        )
    return {
        "current_plan": (
            PlanStep(
                step_id=SEARCH_STEP,
                objective="Locate issue-relevant repository text.",
                status=PlanStepStatus.IN_PROGRESS,
                evidence_required=("A non-empty successful search_code result",),
            ),
            PlanStep(
                step_id=READ_STEP,
                objective="Read the most relevant source file.",
                evidence_required=("A successful read_file result",),
                dependencies=(SEARCH_STEP,),
            ),
            PlanStep(
                step_id=TEST_STEP,
                objective="Run the configured regression test command.",
                evidence_required=("A passing Sandbox test observation",),
                dependencies=(READ_STEP,),
            ),
            PlanStep(
                step_id=FINISH_STEP,
                objective="Stop after collecting the happy-path evidence.",
                evidence_required=("A structured Provider finish action",),
                dependencies=(TEST_STEP,),
            ),
        ),
        "active_step_id": SEARCH_STEP,
    }


def system_utc_now() -> datetime:
    """Return a timezone-aware production clock value."""

    return datetime.now(UTC)


class HappyPathNodes:
    """Node methods bound to one Provider, Registry, clock, and recovery limit."""

    def __init__(
        self,
        *,
        provider: ModelProviderAdapter,
        registry: ToolRegistry,
        max_replans: int = 1,
        prompt_max_chars: int = 50_000,
        clock: Clock = system_utc_now,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        if isinstance(max_replans, bool) or not 0 <= max_replans <= 10:
            raise ValueError("max_replans must be between 0 and 10")
        if isinstance(prompt_max_chars, bool) or not 4_096 <= prompt_max_chars <= 200_000:
            raise ValueError("prompt_max_chars must be between 4096 and 200000")
        self._provider = provider
        self._registry = registry
        self._max_replans = max_replans
        self._prompt_max_chars = prompt_max_chars
        self._clock = clock
        self._trace_recorder = trace_recorder or TraceRecorder()

    @property
    def trace_recorder(self) -> TraceRecorder:
        return self._trace_recorder

    def initialize(self, state: AgentGraphState) -> AgentGraphUpdate:
        """Initialize and record the fixed evidence plan."""

        update = initialize_plan(state)
        return self._trace_recorder.record(
            node=TraceNode.INITIALIZE_PLAN,
            state=state,
            update=update,
        )

    def decide(self, state: AgentGraphState) -> AgentGraphUpdate:
        """Check budget, then request exactly one next action."""

        outcome = self._decide(state)
        return self._trace_recorder.record(
            node=TraceNode.DECIDE,
            state=state,
            update=outcome.update,
            provider_request=outcome.request,
            provider_response=outcome.response,
            provider_error_type=outcome.error_type,
            provider_error_code=outcome.error_code,
        )

    def _decide(self, state: AgentGraphState) -> _DecisionOutcome:
        """Build one decision update and its optional Provider exchange."""

        snapshot = from_graph_state(state)
        if snapshot.run_id is None or snapshot.termination_reason is not None:
            raise self._invalid_transition()
        if snapshot.pending_tool_call is not None or snapshot.pending_tool_result is not None:
            raise AgentLoopError(
                AgentLoopErrorCode.UNFINISHED_PENDING_ACTION,
                error_type=ErrorType.VALIDATION_ERROR,
            )
        if snapshot.finish_summary is not None or snapshot.active_step_id is None:
            raise self._invalid_transition()

        now = self._clock()
        budget_stop = BudgetController.before_decision(snapshot.budget, now=now)
        if budget_stop is not None:
            return _DecisionOutcome(self._terminal_update(snapshot, budget_stop))

        allowed_tools = tuple(descriptor.name for descriptor in self._registry.list_tools())
        request = build_context_aware_provider_request(
            request_id=decision_request_id(snapshot.run_id, snapshot.decision_count),
            run_id=snapshot.run_id,
            state=snapshot,
            allowed_tools=allowed_tools,
            requested_at=now,
            max_chars=self._prompt_max_chars,
        )
        try:
            response = self._provider.generate(request)
        except ProviderError as error:
            return _DecisionOutcome(
                self._provider_failure_update(
                    snapshot,
                    error.error_type,
                    error.code,
                ),
                request=request,
                error_type=error.error_type,
                error_code=error.code,
            )
        if response.request_id != request.request_id:
            return _DecisionOutcome(
                self._provider_failure_update(
                    snapshot,
                    ErrorType.MODEL_FORMAT_ERROR,
                    ProviderErrorCode.RESPONSE_MALFORMED,
                    usage=response.usage,
                ),
                request=request,
                response=response,
                error_type=ErrorType.MODEL_FORMAT_ERROR,
                error_code=ProviderErrorCode.RESPONSE_MALFORMED,
            )

        budget = BudgetController.charge_decision(snapshot.budget, response.usage)
        common_update: AgentGraphUpdate = {
            "budget": budget,
            "decision_count": snapshot.decision_count + 1,
        }
        budget_stop = BudgetController.after_decision(budget, now=self._clock())
        if budget_stop is not None:
            return _DecisionOutcome(
                {
                    **common_update,
                    **self._terminal_update(snapshot, budget_stop),
                },
                request=request,
                response=response,
            )

        if isinstance(response.action, ToolAction):
            call = response.action.tool_call
            expected_tool = EXPECTED_TOOL_BY_STEP.get(snapshot.active_step_id)
            if (
                expected_tool is None
                or call.tool_name != expected_tool
                or not set(response.action.considered_tools).issubset(allowed_tools)
            ):
                return _DecisionOutcome(
                    self._provider_failure_update(
                        snapshot,
                        ErrorType.MODEL_FORMAT_ERROR,
                        ProviderErrorCode.TOOL_NOT_ALLOWED,
                        usage=response.usage,
                    ),
                    request=request,
                    response=response,
                    error_type=ErrorType.MODEL_FORMAT_ERROR,
                    error_code=ProviderErrorCode.TOOL_NOT_ALLOWED,
                )
            fingerprint = action_fingerprint(call)
            if (
                call.call_id in snapshot.consumed_call_ids
                or fingerprint in snapshot.action_fingerprints
            ):
                failure = AgentFailure(
                    kind=AgentFailureKind.REPEATED_ACTION,
                    decision_index=snapshot.decision_count,
                    step_id=snapshot.active_step_id,
                    call_id=call.call_id,
                )
                return _DecisionOutcome(
                    {
                        **common_update,
                        "failure_history": (failure,),
                        **self._terminal_update(
                            snapshot,
                            TerminationReason.REPEATED_ACTION,
                        ),
                    },
                    request=request,
                    response=response,
                )
            return _DecisionOutcome(
                {
                    **common_update,
                    "action_fingerprints": (fingerprint,),
                    "pending_tool_call": call,
                },
                request=request,
                response=response,
            )
        if not isinstance(response.action, FinishAction):
            raise self._invalid_transition()
        if snapshot.active_step_id != FINISH_STEP:
            return _DecisionOutcome(
                self._provider_failure_update(
                    snapshot,
                    ErrorType.MODEL_FORMAT_ERROR,
                    ProviderErrorCode.RESPONSE_MALFORMED,
                    usage=response.usage,
                ),
                request=request,
                response=response,
                error_type=ErrorType.MODEL_FORMAT_ERROR,
                error_code=ProviderErrorCode.RESPONSE_MALFORMED,
            )
        return _DecisionOutcome(
            {
                **common_update,
                "finish_summary": response.action.summary,
            },
            request=request,
            response=response,
        )

    def execute_tool(self, state: AgentGraphState) -> AgentGraphUpdate:
        """Execute and checkpoint the result of the single pending ToolCall."""

        snapshot = from_graph_state(state)
        call = snapshot.pending_tool_call
        if (
            call is None
            or snapshot.pending_tool_result is not None
            or call.call_id in snapshot.consumed_call_ids
        ):
            raise self._invalid_transition()
        result = self._registry.invoke(call)
        update: AgentGraphUpdate = {"pending_tool_result": result}
        return self._trace_recorder.record(
            node=TraceNode.EXECUTE_TOOL,
            state=state,
            update=update,
            tool_call=call,
            tool_result=result,
        )

    def observe(self, state: AgentGraphState) -> AgentGraphUpdate:
        """Commit one result, then advance, replan, or terminate."""

        update = self._observe(state)
        return self._trace_recorder.record(
            node=TraceNode.OBSERVE,
            state=state,
            update=update,
        )

    def _observe(self, state: AgentGraphState) -> AgentGraphUpdate:
        """Build one observation update before it is recorded."""

        snapshot = from_graph_state(state)
        call = snapshot.pending_tool_call
        result = snapshot.pending_tool_result
        if (
            call is None
            or result is None
            or result.call_id != call.call_id
            or result.tool_name != call.tool_name
        ):
            raise self._invalid_transition()

        committed: AgentGraphUpdate = {
            "pending_tool_call": None,
            "pending_tool_result": None,
            "consumed_call_ids": (call.call_id,),
            "tool_results": (result,),
        }
        if result.status is not ToolResultStatus.SUCCESS:
            return {
                **committed,
                "failure_history": (self._tool_failure(snapshot, result),),
                **self._terminal_update(
                    snapshot,
                    self._tool_failure_termination(result),
                ),
            }

        if call.tool_name == "search_code" and self._search_is_empty(result.payload):
            failure = AgentFailure(
                kind=AgentFailureKind.EMPTY_SEARCH,
                decision_index=snapshot.decision_count - 1,
                step_id=SEARCH_STEP,
                call_id=call.call_id,
            )
            if snapshot.replan_count < self._max_replans:
                return {
                    **committed,
                    "current_plan": self._revise_active_plan(
                        snapshot,
                        "search returned no matches; broaden the query",
                    ),
                    "failure_history": (failure,),
                    "replan_count": snapshot.replan_count + 1,
                }
            return {
                **committed,
                "failure_history": (failure,),
                **self._terminal_update(
                    snapshot,
                    TerminationReason.RETRIEVAL_EXHAUSTED,
                ),
            }

        if call.tool_name == "run_tests":
            return self._observe_test(snapshot, result, committed)

        plan, next_step = self._advance_plan(
            snapshot.current_plan,
            snapshot.active_step_id,
        )
        return {
            **committed,
            "current_plan": plan,
            "active_step_id": next_step,
        }

    def finish(self, state: AgentGraphState) -> AgentGraphUpdate:
        """Complete the final plan step without claiming verifier success."""

        snapshot = from_graph_state(state)
        if (
            snapshot.active_step_id != FINISH_STEP
            or snapshot.finish_summary is None
            or snapshot.pending_tool_call is not None
            or snapshot.pending_tool_result is not None
        ):
            raise HappyPathNodes._invalid_transition()
        if tuple(step.step_id for step in snapshot.current_plan) != (
            SEARCH_STEP,
            READ_STEP,
            TEST_STEP,
            FINISH_STEP,
        ) or tuple(step.status for step in snapshot.current_plan) != (
            PlanStepStatus.COMPLETED,
            PlanStepStatus.COMPLETED,
            PlanStepStatus.COMPLETED,
            PlanStepStatus.IN_PROGRESS,
        ):
            raise HappyPathNodes._invalid_transition()
        plan = tuple(
            step.model_copy(update={"status": PlanStepStatus.COMPLETED})
            if step.step_id == FINISH_STEP
            else step
            for step in snapshot.current_plan
        )
        update: AgentGraphUpdate = {
            "current_plan": plan,
            "active_step_id": None,
            "termination_reason": TerminationReason.AGENT_STOPPED,
        }
        return self._trace_recorder.record(
            node=TraceNode.FINISH,
            state=state,
            update=update,
        )

    def terminate(self, state: AgentGraphState) -> AgentGraphUpdate:
        """Validate a failure/budget terminal checkpoint before graph END."""

        snapshot = from_graph_state(state)
        if (
            snapshot.termination_reason is None
            or snapshot.pending_tool_call is not None
            or snapshot.pending_tool_result is not None
            or snapshot.active_step_id is not None
        ):
            raise HappyPathNodes._invalid_transition()
        return self._trace_recorder.record(
            node=TraceNode.TERMINATE,
            state=state,
            update={},
        )

    @staticmethod
    def route_after_decision(
        state: AgentGraphState,
    ) -> Literal["execute_tool", "finish", "terminate"]:
        """Route from mutually exclusive action, finish, or terminal channels."""

        snapshot = from_graph_state(state)
        if snapshot.termination_reason is not None:
            return "terminate"
        if snapshot.pending_tool_call is not None and snapshot.finish_summary is None:
            return "execute_tool"
        if snapshot.pending_tool_call is None and snapshot.finish_summary is not None:
            return "finish"
        raise HappyPathNodes._invalid_transition()

    @staticmethod
    def route_after_observation(
        state: AgentGraphState,
    ) -> Literal["decide", "terminate"]:
        """Continue only after a complete nonterminal observation."""

        snapshot = from_graph_state(state)
        if snapshot.termination_reason is not None:
            return "terminate"
        if (
            snapshot.pending_tool_call is None
            and snapshot.pending_tool_result is None
            and snapshot.active_step_id is not None
        ):
            return "decide"
        raise HappyPathNodes._invalid_transition()

    def _observe_test(
        self,
        snapshot: AgentState,
        result: ToolExecutionResult,
        committed: AgentGraphUpdate,
    ) -> AgentGraphUpdate:
        observation = self._test_observation(result.payload)
        test_run = self._test_run_result(result.payload)
        if test_run.observation_call_id != observation.call_id:
            raise self._invalid_transition()
        test_status = test_run.status
        test_update: AgentGraphUpdate = {
            **committed,
            "observations": (observation,),
            "test_state": advance_test_state(snapshot.test_state, test_run),
        }
        if test_status is TestStatus.PASSED:
            plan, next_step = self._advance_plan(
                snapshot.current_plan,
                snapshot.active_step_id,
            )
            return {
                **test_update,
                "current_plan": plan,
                "active_step_id": next_step,
            }

        failure = AgentFailure(
            kind=AgentFailureKind.TEST_FAILURE,
            decision_index=snapshot.decision_count - 1,
            step_id=TEST_STEP,
            call_id=observation.call_id,
            error_type=test_run.error_type,
        )
        if test_run.error_type is ErrorType.SANDBOX_ERROR:
            reason = TerminationReason.SANDBOX_ERROR
        elif test_run.error_type is ErrorType.TOOL_TIMEOUT:
            reason = TerminationReason.TIMEOUT
        elif test_status is TestStatus.ERROR:
            reason = TerminationReason.TOOL_ERROR
        else:
            reason = TerminationReason.TESTS_FAILED
        return {
            **test_update,
            "failure_history": (failure,),
            **self._terminal_update(snapshot, reason),
        }

    @staticmethod
    def _advance_plan(
        plan: tuple[PlanStep, ...],
        active_step_id: str | None,
    ) -> tuple[tuple[PlanStep, ...], str]:
        step_ids = tuple(step.step_id for step in plan)
        if (
            step_ids != (SEARCH_STEP, READ_STEP, TEST_STEP, FINISH_STEP)
            or active_step_id not in EXPECTED_TOOL_BY_STEP
        ):
            raise HappyPathNodes._invalid_transition()
        active_index = step_ids.index(active_step_id)
        if active_index + 1 >= len(plan):
            raise HappyPathNodes._invalid_transition()
        expected_statuses = (PlanStepStatus.COMPLETED,) * active_index + (
            PlanStepStatus.IN_PROGRESS,
        )
        if tuple(step.status for step in plan[: active_index + 1]) != expected_statuses or any(
            step.status is not PlanStepStatus.PENDING for step in plan[active_index + 1 :]
        ):
            raise HappyPathNodes._invalid_transition()
        next_step_id = plan[active_index + 1].step_id
        updated = tuple(
            step.model_copy(update={"status": PlanStepStatus.COMPLETED})
            if index == active_index
            else step.model_copy(update={"status": PlanStepStatus.IN_PROGRESS})
            if index == active_index + 1
            else step
            for index, step in enumerate(plan)
        )
        return updated, next_step_id

    @staticmethod
    def _revise_active_plan(
        snapshot: AgentState,
        reason: str,
    ) -> tuple[PlanStep, ...]:
        if snapshot.active_step_id != SEARCH_STEP:
            raise HappyPathNodes._invalid_transition()
        return tuple(
            step.model_copy(update={"revision_reason": reason})
            if step.step_id == SEARCH_STEP
            else step
            for step in snapshot.current_plan
        )

    @staticmethod
    def _search_is_empty(payload: Mapping[str, object] | None) -> bool:
        if payload is None:
            raise HappyPathNodes._invalid_transition()
        matches = payload.get("matches")
        if not isinstance(matches, list):
            raise HappyPathNodes._invalid_transition()
        return not matches

    @staticmethod
    def _test_observation(
        payload: Mapping[str, object] | None,
    ) -> Observation:
        if payload is None or "observation" not in payload:
            raise HappyPathNodes._invalid_transition()
        try:
            return Observation.model_validate(payload["observation"])
        except ValueError:
            raise HappyPathNodes._invalid_transition() from None

    @staticmethod
    def _test_run_result(
        payload: Mapping[str, object] | None,
    ) -> TestRunResult:
        if payload is None or "test_run" not in payload:
            raise HappyPathNodes._invalid_transition()
        try:
            return TestRunResult.model_validate(payload["test_run"])
        except ValueError:
            raise HappyPathNodes._invalid_transition() from None

    def _provider_failure_update(
        self,
        snapshot: AgentState,
        error_type: ErrorType,
        error_code: ProviderErrorCode,
        *,
        usage: ProviderUsageDelta = ZERO_USAGE,
    ) -> AgentGraphUpdate:
        failure = AgentFailure(
            kind=AgentFailureKind.PROVIDER_FAILURE,
            decision_index=snapshot.decision_count,
            step_id=snapshot.active_step_id or SEARCH_STEP,
            error_type=error_type,
            provider_error_code=error_code,
        )
        return {
            "budget": BudgetController.charge_decision(snapshot.budget, usage),
            "decision_count": snapshot.decision_count + 1,
            "failure_history": (failure,),
            **self._terminal_update(snapshot, TerminationReason.PROVIDER_ERROR),
        }

    @staticmethod
    def _tool_failure(
        snapshot: AgentState,
        result: ToolExecutionResult,
    ) -> AgentFailure:
        if result.error_type is None or result.error_code is None:
            raise HappyPathNodes._invalid_transition()
        return AgentFailure(
            kind=AgentFailureKind.TOOL_ERROR,
            decision_index=snapshot.decision_count - 1,
            step_id=snapshot.active_step_id or SEARCH_STEP,
            call_id=result.call_id,
            error_type=result.error_type,
            tool_error_code=result.error_code,
        )

    @staticmethod
    def _tool_failure_termination(
        result: ToolExecutionResult,
    ) -> TerminationReason:
        if result.error_type is ErrorType.SANDBOX_ERROR:
            return TerminationReason.SANDBOX_ERROR
        if result.error_type is ErrorType.TOOL_TIMEOUT:
            return TerminationReason.TIMEOUT
        return TerminationReason.TOOL_ERROR

    @staticmethod
    def _terminal_update(
        snapshot: AgentState,
        reason: TerminationReason,
    ) -> AgentGraphUpdate:
        plan = tuple(
            step.model_copy(
                update={
                    "status": PlanStepStatus.BLOCKED,
                    "revision_reason": f"terminated: {reason.value}",
                }
            )
            if step.step_id == snapshot.active_step_id
            else step
            for step in snapshot.current_plan
        )
        return {
            "current_plan": plan,
            "active_step_id": None,
            "pending_tool_call": None,
            "pending_tool_result": None,
            "termination_reason": reason,
        }

    @staticmethod
    def _invalid_transition() -> AgentLoopError:
        return AgentLoopError(
            AgentLoopErrorCode.INVALID_STATE_TRANSITION,
            error_type=ErrorType.VALIDATION_ERROR,
        )


def action_fingerprint(call: ToolCall) -> str:
    """Hash one canonical capability request without retaining raw arguments."""

    encoded = json.dumps(
        {
            "tool_name": call.tool_name,
            "arguments": call.validated_arguments,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
