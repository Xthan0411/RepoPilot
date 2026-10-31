"""Complete repair Agent graph from baseline reproduction through structured finish."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, Self, cast
from uuid import UUID, uuid5

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import JsonValue, ValidationError

from repopilot.agent.budget import BudgetController
from repopilot.agent.graph import (
    AgentStateGraphError,
    build_in_memory_checkpointer,
)
from repopilot.agent.nodes import AgentLoopError
from repopilot.agent.state import (
    AgentGraphState,
    AgentGraphUpdate,
    ThreadIdentity,
    from_graph_state,
    to_graph_state,
)
from repopilot.agent.trace_recorder import TraceRecorder
from repopilot.config import RepoPilotSettings
from repopilot.context import CONTEXT_POLICY_VERSION, ContextAccountingError
from repopilot.enums import (
    AgentFailureKind,
    AgentLoopErrorCode,
    AgentStateErrorCode,
    ErrorType,
    PatchErrorCode,
    PatchStatus,
    PlanStepStatus,
    PolicyDecision,
    ProviderErrorCode,
    TerminationReason,
    TestStatus,
    ToolResultStatus,
    TraceErrorCode,
    TraceNode,
    WorkingMemoryErrorCode,
)
from repopilot.memory import (
    WorkingMemoryCheckpoint,
    WorkingMemoryError,
    WorkingMemoryStore,
    build_working_memory_checkpoint,
    context_policy_version_from_events,
    verify_working_memory_integrity,
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
from repopilot.tools.patch import (
    PatchApplicationResult,
    PatchLimits,
    PatchValidationError,
    parse_unified_diff,
)
from repopilot.tools.registry import ToolRegistry
from repopilot.tools.tests import advance_test_state
from repopilot.trace import AgentTrace, TraceArtifactError, TraceEvent

REPRODUCE_STEP = "reproduce"
SEARCH_STEP = "search"
INSPECT_STEP = "inspect"
PATCH_STEP = "patch"
VALIDATE_STEP = "validate"
FINISH_STEP = "finish"
REPAIR_STEP_IDS = (
    REPRODUCE_STEP,
    SEARCH_STEP,
    INSPECT_STEP,
    PATCH_STEP,
    VALIDATE_STEP,
    FINISH_STEP,
)
STEP_TOOL_CAPABILITIES: dict[str, tuple[str, ...]] = {
    REPRODUCE_STEP: ("run_tests",),
    SEARCH_STEP: ("search_code",),
    INSPECT_STEP: ("inspect_symbol", "read_file"),
    PATCH_STEP: ("read_file", "inspect_symbol", "apply_patch"),
    VALIDATE_STEP: ("run_tests",),
    FINISH_STEP: (),
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


def system_utc_now() -> datetime:
    return datetime.now(UTC)


def repair_decision_request_id(run_id: UUID, decision_index: int) -> UUID:
    """Derive the same stable decision namespace used by all Agent graphs."""

    if decision_index < 0:
        raise ValueError("decision_index must not be negative")
    return uuid5(run_id, f"decision:{decision_index}")


def initialize_repair_plan(state: AgentGraphState) -> AgentGraphUpdate:
    """Install the only P5 repair evidence path on a fresh checkpoint."""

    snapshot = from_graph_state(state)
    if (
        snapshot.run_id is None
        or snapshot.termination_reason is not None
        or snapshot.current_plan
        or snapshot.active_step_id is not None
        or snapshot.pending_tool_call is not None
        or snapshot.pending_tool_result is not None
        or snapshot.tool_results
        or snapshot.failure_history
        or snapshot.action_fingerprints
        or snapshot.replan_count != 0
        or snapshot.decision_count != 0
        or snapshot.finish_summary is not None
        or snapshot.baseline_test_result is not None
        or snapshot.test_state.status is not TestStatus.NOT_RUN
        or snapshot.patch_state.status is not PatchStatus.NONE
    ):
        raise _invalid_transition()
    return {
        "current_plan": (
            PlanStep(
                step_id=REPRODUCE_STEP,
                objective="Run the configured baseline test and reproduce the issue.",
                status=PlanStepStatus.IN_PROGRESS,
                evidence_required=("Artifact-backed failing baseline test evidence",),
            ),
            PlanStep(
                step_id=SEARCH_STEP,
                objective="Search for issue-relevant repository code.",
                evidence_required=("A non-empty successful search result",),
                dependencies=(REPRODUCE_STEP,),
            ),
            PlanStep(
                step_id=INSPECT_STEP,
                objective=(
                    "Locate relevant symbols if needed, then use read_file to inspect "
                    "the actual source text before proposing a patch."
                ),
                evidence_required=("A successful read_file result with non-empty source text",),
                dependencies=(SEARCH_STEP,),
            ),
            PlanStep(
                step_id=PATCH_STEP,
                objective=(
                    "Read every file to be modified, then apply one bounded unified diff "
                    "in the Sandbox. Read additional source regions when needed."
                ),
                evidence_required=("An artifact-backed applied PatchState",),
                dependencies=(INSPECT_STEP,),
            ),
            PlanStep(
                step_id=VALIDATE_STEP,
                objective="Run the same configured test command after the patch.",
                evidence_required=("An independent artifact-backed passing test run",),
                dependencies=(PATCH_STEP,),
            ),
            PlanStep(
                step_id=FINISH_STEP,
                objective="Request deterministic verification of collected evidence.",
                evidence_required=("A structured Provider finish action",),
                dependencies=(VALIDATE_STEP,),
            ),
        ),
        "active_step_id": REPRODUCE_STEP,
    }


class RepairAgentNodes:
    """Step-gated model decisions and deterministic repair evidence transitions."""

    def __init__(
        self,
        *,
        provider: ModelProviderAdapter,
        registry: ToolRegistry,
        max_replans: int = 1,
        prompt_max_chars: int = 50_000,
        patch_limits: PatchLimits | None = None,
        clock: Clock = system_utc_now,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        if isinstance(max_replans, bool) or not 0 <= max_replans <= 10:
            raise ValueError("max_replans must be between 0 and 10")
        if isinstance(prompt_max_chars, bool) or not 4_096 <= prompt_max_chars <= 200_000:
            raise ValueError("prompt_max_chars must be between 4096 and 200000")
        available = {descriptor.name for descriptor in registry.list_tools()}
        required = {
            capability
            for capabilities in STEP_TOOL_CAPABILITIES.values()
            for capability in capabilities
        }
        if not required.issubset(available):
            raise ValueError("repair Registry is missing a required capability")
        self._provider = provider
        self._registry = registry
        self._max_replans = max_replans
        self._prompt_max_chars = prompt_max_chars
        self._patch_limits = patch_limits or PatchLimits(
            max_bytes=1_000_000,
            max_files=100,
            max_hunks=1_000,
            max_changed_lines=10_000,
        )
        self._clock = clock
        self._trace_recorder = trace_recorder or TraceRecorder()

    @property
    def trace_recorder(self) -> TraceRecorder:
        return self._trace_recorder

    def initialize(self, state: AgentGraphState) -> AgentGraphUpdate:
        update = initialize_repair_plan(state)
        return self._trace_recorder.record(
            node=TraceNode.INITIALIZE_PLAN,
            state=state,
            update=update,
        )

    def decide(self, state: AgentGraphState) -> AgentGraphUpdate:
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
        snapshot = from_graph_state(state)
        if (
            snapshot.run_id is None
            or snapshot.termination_reason is not None
            or snapshot.pending_tool_call is not None
            or snapshot.pending_tool_result is not None
            or snapshot.finish_summary is not None
            or snapshot.active_step_id not in STEP_TOOL_CAPABILITIES
        ):
            raise _invalid_transition()

        now = self._clock()
        budget_stop = BudgetController.before_decision(snapshot.budget, now=now)
        if budget_stop is not None:
            return _DecisionOutcome(self._terminal_update(snapshot, budget_stop))

        allowed_tools = self._allowed_tools(snapshot)
        request = build_context_aware_provider_request(
            request_id=repair_decision_request_id(
                snapshot.run_id,
                snapshot.decision_count,
            ),
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

        if isinstance(response.action, FinishAction):
            if snapshot.active_step_id != FINISH_STEP:
                return self._format_failure(
                    snapshot,
                    request,
                    response,
                    common_update,
                    ProviderErrorCode.RESPONSE_MALFORMED,
                )
            return _DecisionOutcome(
                {
                    **common_update,
                    "finish_summary": response.action.summary,
                },
                request=request,
                response=response,
            )
        if not isinstance(response.action, ToolAction):
            raise _invalid_transition()

        call = response.action.tool_call
        if (
            snapshot.active_step_id == FINISH_STEP
            or call.tool_name not in allowed_tools
            or not set(response.action.considered_tools).issubset(allowed_tools)
        ):
            return self._format_failure(
                snapshot,
                request,
                response,
                common_update,
                ProviderErrorCode.TOOL_NOT_ALLOWED,
            )
        fingerprint = repair_action_fingerprint(
            snapshot.active_step_id,
            call,
            patch_revision=snapshot.patch_state.revision,
        )
        if call.call_id in snapshot.consumed_call_ids or fingerprint in (
            snapshot.action_fingerprints
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

    def _format_failure(
        self,
        snapshot: AgentState,
        request: ProviderRequest,
        response: ProviderResponse,
        common_update: AgentGraphUpdate,
        code: ProviderErrorCode,
    ) -> _DecisionOutcome:
        provider_update = self._provider_failure_update(
            snapshot,
            ErrorType.MODEL_FORMAT_ERROR,
            code,
            usage=response.usage,
        )
        provider_update.pop("budget", None)
        provider_update.pop("decision_count", None)
        return _DecisionOutcome(
            {**common_update, **provider_update},
            request=request,
            response=response,
            error_type=ErrorType.MODEL_FORMAT_ERROR,
            error_code=code,
        )

    def execute_tool(self, state: AgentGraphState) -> AgentGraphUpdate:
        snapshot = from_graph_state(state)
        call = snapshot.pending_tool_call
        if (
            call is None
            or snapshot.pending_tool_result is not None
            or call.call_id in snapshot.consumed_call_ids
        ):
            raise _invalid_transition()
        result = self._unread_patch_target_result(snapshot, call)
        if result is None:
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
        update = self._observe(state)
        return self._trace_recorder.record(
            node=TraceNode.OBSERVE,
            state=state,
            update=update,
        )

    def _observe(self, state: AgentGraphState) -> AgentGraphUpdate:
        snapshot = from_graph_state(state)
        call = snapshot.pending_tool_call
        result = snapshot.pending_tool_result
        if (
            call is None
            or result is None
            or result.call_id != call.call_id
            or result.tool_name != call.tool_name
        ):
            raise _invalid_transition()
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
        if snapshot.active_step_id == REPRODUCE_STEP and call.tool_name == "run_tests":
            return self._observe_reproduction(snapshot, result, committed)
        if snapshot.active_step_id == VALIDATE_STEP and call.tool_name == "run_tests":
            return self._observe_validation(snapshot, result, committed)
        if snapshot.active_step_id == PATCH_STEP and call.tool_name == "apply_patch":
            return self._observe_patch(snapshot, result, committed)
        if (
            snapshot.active_step_id == SEARCH_STEP
            and call.tool_name == "search_code"
            and self._search_is_empty(result.payload)
        ):
            return self._empty_search_update(snapshot, call, committed)
        if call.tool_name not in self._allowed_tools(snapshot):
            raise _invalid_transition()
        if snapshot.active_step_id == INSPECT_STEP and (
            call.tool_name != "read_file" or not self._read_source_path(result)
        ):
            return {
                **committed,
                "current_plan": self._revise_active_step(
                    snapshot,
                    "Symbol locations or empty reads are not source evidence; use read_file "
                    "on the relevant source region and inspect non-empty text.",
                ),
            }
        if snapshot.active_step_id == PATCH_STEP:
            # Additional inspection must never substitute for an applied patch.
            return committed
        plan, next_step = self._advance_plan(snapshot)
        return {
            **committed,
            "current_plan": plan,
            "active_step_id": next_step,
        }

    def _observe_reproduction(
        self,
        snapshot: AgentState,
        result: ToolExecutionResult,
        committed: AgentGraphUpdate,
    ) -> AgentGraphUpdate:
        observation, test_run = self._test_evidence(result.payload)
        test_update: AgentGraphUpdate = {
            **committed,
            "observations": (observation,),
            "test_state": advance_test_state(snapshot.test_state, test_run),
            "baseline_test_result": test_run,
        }
        if test_run.status is TestStatus.FAILED:
            plan, next_step = self._advance_plan(snapshot)
            return {
                **test_update,
                "current_plan": plan,
                "active_step_id": next_step,
            }
        failure = AgentFailure(
            kind=AgentFailureKind.TEST_FAILURE,
            decision_index=snapshot.decision_count - 1,
            step_id=REPRODUCE_STEP,
            call_id=observation.call_id,
            error_type=test_run.error_type,
        )
        return {
            **test_update,
            "failure_history": (failure,),
            **self._terminal_update(
                snapshot,
                self._test_termination(test_run),
            ),
        }

    def _observe_validation(
        self,
        snapshot: AgentState,
        result: ToolExecutionResult,
        committed: AgentGraphUpdate,
    ) -> AgentGraphUpdate:
        observation, test_run = self._test_evidence(result.payload)
        test_update: AgentGraphUpdate = {
            **committed,
            "observations": (observation,),
            "test_state": advance_test_state(snapshot.test_state, test_run),
        }
        if test_run.status is TestStatus.PASSED:
            plan, next_step = self._advance_plan(snapshot)
            return {
                **test_update,
                "current_plan": plan,
                "active_step_id": next_step,
            }
        failure = AgentFailure(
            kind=AgentFailureKind.TEST_FAILURE,
            decision_index=snapshot.decision_count - 1,
            step_id=VALIDATE_STEP,
            call_id=observation.call_id,
            error_type=test_run.error_type,
        )
        if (
            test_run.status is TestStatus.FAILED
            and test_run.error_type is None
            and observation.error_type is None
            and observation.policy_decision is PolicyDecision.ALLOWED
            and not test_run.truncated
            and not observation.truncated
            and snapshot.patch_state.status is PatchStatus.APPLIED
            and snapshot.replan_count < self._max_replans
            and not any(
                prior.kind is AgentFailureKind.TEST_FAILURE and prior.step_id == VALIDATE_STEP
                for prior in snapshot.failure_history
            )
        ):
            reason = (
                "The applied candidate failed validation; one bounded repair is allowed. "
                "Read the current patched source again for every target file before editing. "
                "Use the available failure evidence to submit a distinct incremental diff "
                "against the current workspace, then run the configured validation again. "
                "Tests remain read-only; all existing budgets and safety checks still apply."
            )
            return {
                **test_update,
                "failure_history": (failure,),
                "replan_count": snapshot.replan_count + 1,
                "current_plan": tuple(
                    step.model_copy(
                        update={
                            "status": PlanStepStatus.IN_PROGRESS,
                            "revision_reason": reason,
                        }
                    )
                    if step.step_id == PATCH_STEP
                    else step.model_copy(
                        update={"status": PlanStepStatus.PENDING, "revision_reason": reason}
                    )
                    if step.step_id in (VALIDATE_STEP, FINISH_STEP)
                    else step
                    for step in snapshot.current_plan
                ),
                "active_step_id": PATCH_STEP,
            }
        return {
            **test_update,
            "failure_history": (failure,),
            **self._terminal_update(
                snapshot,
                self._test_termination(test_run),
            ),
        }

    def _observe_patch(
        self,
        snapshot: AgentState,
        result: ToolExecutionResult,
        committed: AgentGraphUpdate,
    ) -> AgentGraphUpdate:
        if result.payload is None:
            raise _invalid_transition()
        try:
            patch = PatchApplicationResult.model_validate(result.payload)
        except ValidationError:
            raise _invalid_transition() from None
        observation_update: tuple[Observation, ...] = (
            () if patch.observation is None else (patch.observation,)
        )
        patch_update: AgentGraphUpdate = {
            **committed,
            "patch_state": patch.patch_state,
            "observations": observation_update,
        }
        if patch.status is PatchStatus.APPLIED:
            plan, next_step = self._advance_plan(snapshot)
            return {
                **patch_update,
                "current_plan": plan,
                "active_step_id": next_step,
            }
        if patch.error_code is None:
            raise _invalid_transition()
        failure = AgentFailure(
            kind=AgentFailureKind.PATCH_FAILURE,
            decision_index=snapshot.decision_count - 1,
            step_id=PATCH_STEP,
            call_id=result.call_id,
            error_type=ErrorType.PATCH_ERROR,
            patch_error_code=patch.error_code,
        )
        recoverable = patch.error_code in (
            PatchErrorCode.INVALID_DIFF,
            PatchErrorCode.APPLY_CONFLICT,
        )
        protected_test_rejection = (
            patch.error_code is PatchErrorCode.PROTECTED_TEST_PATH
            and patch.patch_state == snapshot.patch_state
            and patch.observation is not None
            and patch.observation.call_id == result.call_id
            and patch.observation.exit_code == 44
            and patch.observation.error_type is ErrorType.PATCH_ERROR
            and patch.observation.policy_decision is PolicyDecision.ALLOWED
        )
        recoverable = recoverable or protected_test_rejection
        if recoverable and not any(
            prior.kind is AgentFailureKind.PATCH_FAILURE for prior in snapshot.failure_history
        ):
            correction_instruction = (
                "The entire candidate was rejected before any source changes were applied. "
                "Tests and evaluator inputs are read-only. Submit a new source-only diff; "
                "do not modify, delete, or disable tests. "
                if protected_test_rejection
                else "Read the actual target source if needed, then submit a distinct "
                "corrected diff. "
            )
            return {
                **patch_update,
                "failure_history": (failure,),
                "current_plan": self._revise_active_step(
                    snapshot,
                    f"Patch rejected ({patch.error_code.value}); one correction remains. "
                    + correction_instruction
                    + "Do not repeat the rejected patch.",
                ),
            }
        return {
            **patch_update,
            "failure_history": (failure,),
            **self._terminal_update(snapshot, TerminationReason.INVALID_PATCH),
        }

    @staticmethod
    def _allowed_tools(snapshot: AgentState) -> tuple[str, ...]:
        allowed = STEP_TOOL_CAPABILITIES.get(snapshot.active_step_id or "", ())
        if snapshot.active_step_id == PATCH_STEP and snapshot.patch_state.revision > 0:
            # Symbol indexes describe the immutable base, not the patched workspace.
            return tuple(name for name in allowed if name != "inspect_symbol")
        return allowed

    @staticmethod
    def _read_source_path(result: ToolExecutionResult) -> str | None:
        if result.tool_name != "read_file" or result.status is not ToolResultStatus.SUCCESS:
            return None
        payload = result.payload
        if payload is None:
            return None
        content = payload.get("content")
        path = payload.get("relative_path", payload.get("path"))
        if isinstance(content, str) and content.strip() and isinstance(path, str) and path:
            return path
        return None

    def _unread_patch_target_result(
        self,
        snapshot: AgentState,
        call: ToolCall,
    ) -> ToolExecutionResult | None:
        """Require source evidence for each target without bypassing patch safety checks."""

        if call.tool_name != "apply_patch":
            return None
        diff = call.validated_arguments.get("diff")
        if not isinstance(diff, str):
            return None
        try:
            parsed = parse_unified_diff(diff, self._patch_limits)
        except PatchValidationError:
            # The real tool still owns malformed, unsupported and unsafe patch rejection.
            return None
        # A read of the original checkout is not evidence about a newly patched file.
        # Rejected patches do not invalidate reads because no source change occurred.
        latest_applied = -1
        for index, result in enumerate(snapshot.tool_results):
            if (
                result.tool_name == "apply_patch"
                and result.status is ToolResultStatus.SUCCESS
                and result.payload is not None
                and result.payload.get("status") == PatchStatus.APPLIED.value
            ):
                latest_applied = index
        read_paths = {
            path
            for result in snapshot.tool_results[latest_applied + 1 :]
            if (path := self._read_source_path(result)) is not None
        }
        unread = sorted(set(parsed.summary.files_changed) - read_paths)
        if not unread:
            return None
        rejected = PatchApplicationResult(
            status=PatchStatus.REJECTED,
            patch_state=snapshot.patch_state,
            error_code=PatchErrorCode.APPLY_CONFLICT,
            message=(
                "Read the current source with read_file after the latest applied patch "
                "before modifying: " + ", ".join(unread)
            ),
        )
        payload = cast(dict[str, JsonValue], rejected.model_dump(mode="json"))
        return ToolExecutionResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolResultStatus.SUCCESS,
            payload=payload,
            output_bytes=len(
                json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
            ),
        )

    @staticmethod
    def _revise_active_step(snapshot: AgentState, reason: str) -> tuple[PlanStep, ...]:
        return tuple(
            step.model_copy(update={"revision_reason": reason})
            if step.step_id == snapshot.active_step_id
            else step
            for step in snapshot.current_plan
        )

    def finish(self, state: AgentGraphState) -> AgentGraphUpdate:
        snapshot = from_graph_state(state)
        if (
            snapshot.active_step_id != FINISH_STEP
            or snapshot.finish_summary is None
            or snapshot.pending_tool_call is not None
            or snapshot.pending_tool_result is not None
            or tuple(step.step_id for step in snapshot.current_plan) != REPAIR_STEP_IDS
            or tuple(step.status for step in snapshot.current_plan)
            != (
                PlanStepStatus.COMPLETED,
                PlanStepStatus.COMPLETED,
                PlanStepStatus.COMPLETED,
                PlanStepStatus.COMPLETED,
                PlanStepStatus.COMPLETED,
                PlanStepStatus.IN_PROGRESS,
            )
        ):
            raise _invalid_transition()
        plan = tuple(
            step.model_copy(update={"status": PlanStepStatus.COMPLETED})
            if step.step_id == FINISH_STEP
            else step
            for step in snapshot.current_plan
        )
        return self._trace_recorder.record(
            node=TraceNode.FINISH,
            state=state,
            update={
                "current_plan": plan,
                "active_step_id": None,
                "termination_reason": TerminationReason.AGENT_STOPPED,
            },
        )

    def terminate(self, state: AgentGraphState) -> AgentGraphUpdate:
        snapshot = from_graph_state(state)
        if (
            snapshot.termination_reason is None
            or snapshot.pending_tool_call is not None
            or snapshot.pending_tool_result is not None
            or snapshot.active_step_id is not None
        ):
            raise _invalid_transition()
        return self._trace_recorder.record(
            node=TraceNode.TERMINATE,
            state=state,
            update={},
        )

    @staticmethod
    def route_after_decision(
        state: AgentGraphState,
    ) -> Literal["execute_tool", "finish", "terminate"]:
        snapshot = from_graph_state(state)
        if snapshot.termination_reason is not None:
            return "terminate"
        if snapshot.pending_tool_call is not None and snapshot.finish_summary is None:
            return "execute_tool"
        if snapshot.pending_tool_call is None and snapshot.finish_summary is not None:
            return "finish"
        raise _invalid_transition()

    @staticmethod
    def route_after_observation(
        state: AgentGraphState,
    ) -> Literal["decide", "terminate"]:
        snapshot = from_graph_state(state)
        if snapshot.termination_reason is not None:
            return "terminate"
        if (
            snapshot.pending_tool_call is None
            and snapshot.pending_tool_result is None
            and snapshot.active_step_id is not None
        ):
            return "decide"
        raise _invalid_transition()

    def _empty_search_update(
        self,
        snapshot: AgentState,
        call: ToolCall,
        committed: AgentGraphUpdate,
    ) -> AgentGraphUpdate:
        failure = AgentFailure(
            kind=AgentFailureKind.EMPTY_SEARCH,
            decision_index=snapshot.decision_count - 1,
            step_id=SEARCH_STEP,
            call_id=call.call_id,
        )
        if snapshot.replan_count < self._max_replans:
            return {
                **committed,
                "current_plan": tuple(
                    step.model_copy(
                        update={
                            "revision_reason": (
                                "search returned no matches; use a broader distinct query"
                            )
                        }
                    )
                    if step.step_id == SEARCH_STEP
                    else step
                    for step in snapshot.current_plan
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

    @staticmethod
    def _advance_plan(
        snapshot: AgentState,
    ) -> tuple[tuple[PlanStep, ...], str]:
        plan = snapshot.current_plan
        active_step_id = snapshot.active_step_id
        if (
            tuple(step.step_id for step in plan) != REPAIR_STEP_IDS
            or active_step_id not in REPAIR_STEP_IDS[:-1]
        ):
            raise _invalid_transition()
        active_index = REPAIR_STEP_IDS.index(active_step_id)
        expected = (PlanStepStatus.COMPLETED,) * active_index + (PlanStepStatus.IN_PROGRESS,)
        if tuple(step.status for step in plan[: active_index + 1]) != expected or any(
            step.status is not PlanStepStatus.PENDING for step in plan[active_index + 1 :]
        ):
            raise _invalid_transition()
        next_step = REPAIR_STEP_IDS[active_index + 1]
        updated = tuple(
            step.model_copy(update={"status": PlanStepStatus.COMPLETED})
            if index == active_index
            else step.model_copy(update={"status": PlanStepStatus.IN_PROGRESS})
            if index == active_index + 1
            else step
            for index, step in enumerate(plan)
        )
        return updated, next_step

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
            step_id=snapshot.active_step_id or REPRODUCE_STEP,
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
            raise _invalid_transition()
        return AgentFailure(
            kind=AgentFailureKind.TOOL_ERROR,
            decision_index=snapshot.decision_count - 1,
            step_id=snapshot.active_step_id or REPRODUCE_STEP,
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
    def _test_termination(test_run: TestRunResult) -> TerminationReason:
        if test_run.error_type is ErrorType.SANDBOX_ERROR:
            return TerminationReason.SANDBOX_ERROR
        if test_run.error_type is ErrorType.TOOL_TIMEOUT:
            return TerminationReason.TIMEOUT
        if test_run.status is TestStatus.ERROR:
            return TerminationReason.TOOL_ERROR
        return TerminationReason.TESTS_FAILED

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
    def _search_is_empty(payload: Mapping[str, object] | None) -> bool:
        if payload is None:
            raise _invalid_transition()
        matches = payload.get("matches")
        if not isinstance(matches, list):
            raise _invalid_transition()
        return not matches

    @staticmethod
    def _test_evidence(
        payload: Mapping[str, object] | None,
    ) -> tuple[Observation, TestRunResult]:
        if payload is None or "observation" not in payload or "test_run" not in payload:
            raise _invalid_transition()
        try:
            observation = Observation.model_validate(payload["observation"])
            test_run = TestRunResult.model_validate(payload["test_run"])
        except ValidationError:
            raise _invalid_transition() from None
        if test_run.observation_call_id != observation.call_id:
            raise _invalid_transition()
        return observation, test_run


class RepairAgentGraph:
    """LangGraph repair loop with injected Provider, Registry, and checkpoint store."""

    def __init__(
        self,
        *,
        provider: ModelProviderAdapter,
        registry: ToolRegistry,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        max_replans: int = 1,
        prompt_max_chars: int = 50_000,
        patch_limits: PatchLimits | None = None,
        clock: Clock = system_utc_now,
    ) -> None:
        self._checkpointer = checkpointer or build_in_memory_checkpointer()
        self._provider_name = provider.provider_name
        self._clock = clock
        self._nodes = RepairAgentNodes(
            provider=provider,
            registry=registry,
            max_replans=max_replans,
            prompt_max_chars=prompt_max_chars,
            patch_limits=patch_limits,
            clock=clock,
        )
        builder = StateGraph(AgentGraphState)
        builder.add_node("initialize_plan", RunnableLambda(self._nodes.initialize))
        builder.add_node("decide", RunnableLambda(self._nodes.decide))
        builder.add_node("execute_tool", RunnableLambda(self._nodes.execute_tool))
        builder.add_node("observe", RunnableLambda(self._nodes.observe))
        builder.add_node("finish", RunnableLambda(self._nodes.finish))
        builder.add_node("terminate", RunnableLambda(self._nodes.terminate))
        builder.add_edge(START, "initialize_plan")
        builder.add_edge("initialize_plan", "decide")
        builder.add_conditional_edges(
            "decide",
            self._nodes.route_after_decision,
            {
                "execute_tool": "execute_tool",
                "finish": "finish",
                "terminate": "terminate",
            },
        )
        builder.add_edge("execute_tool", "observe")
        builder.add_conditional_edges(
            "observe",
            self._nodes.route_after_observation,
            {
                "decide": "decide",
                "terminate": "terminate",
            },
        )
        builder.add_edge("finish", END)
        builder.add_edge("terminate", END)
        self._graph: CompiledStateGraph[
            AgentGraphState,
            None,
            AgentGraphState,
            AgentGraphState,
        ] = builder.compile(
            checkpointer=self._checkpointer,
            name="repopilot-repair",
        )
        recovery_builder = StateGraph(AgentGraphState)
        recovery_builder.add_node("decide", RunnableLambda(self._nodes.decide))
        recovery_builder.add_node(
            "execute_tool",
            RunnableLambda(self._nodes.execute_tool),
        )
        recovery_builder.add_node("observe", RunnableLambda(self._nodes.observe))
        recovery_builder.add_node("finish", RunnableLambda(self._nodes.finish))
        recovery_builder.add_node(
            "terminate",
            RunnableLambda(self._nodes.terminate),
        )
        recovery_builder.add_conditional_edges(
            START,
            _route_repair_recovery,
            {
                "decide": "decide",
                "execute_tool": "execute_tool",
                "observe": "observe",
                "finish": "finish",
            },
        )
        recovery_builder.add_conditional_edges(
            "decide",
            self._nodes.route_after_decision,
            {
                "execute_tool": "execute_tool",
                "finish": "finish",
                "terminate": "terminate",
            },
        )
        recovery_builder.add_edge("execute_tool", "observe")
        recovery_builder.add_conditional_edges(
            "observe",
            self._nodes.route_after_observation,
            {
                "decide": "decide",
                "terminate": "terminate",
            },
        )
        recovery_builder.add_edge("finish", END)
        recovery_builder.add_edge("terminate", END)
        self._recovery_graph: CompiledStateGraph[
            AgentGraphState,
            None,
            AgentGraphState,
            AgentGraphState,
        ] = recovery_builder.compile(
            checkpointer=self._checkpointer,
            name="repopilot-repair-recovery",
        )

    @classmethod
    def from_settings(
        cls,
        *,
        provider: ModelProviderAdapter,
        registry: ToolRegistry,
        settings: RepoPilotSettings,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        clock: Clock = system_utc_now,
    ) -> Self:
        return cls(
            provider=provider,
            registry=registry,
            checkpointer=checkpointer,
            max_replans=settings.max_replans,
            prompt_max_chars=settings.openai_prompt_max_chars,
            patch_limits=PatchLimits.from_settings(settings),
            clock=clock,
        )

    def run(
        self,
        *,
        thread_id: str,
        run_id: UUID,
        state: AgentState,
    ) -> AgentState:
        config = self._thread_config(thread_id)
        existing = self._read_optional(config)
        if existing is not None:
            code = (
                AgentStateErrorCode.TASK_MISMATCH
                if existing.task.task_id != state.task.task_id
                else AgentStateErrorCode.THREAD_ALREADY_EXISTS
            )
            raise AgentStateGraphError(code)
        if state.run_id is not None and state.run_id != run_id:
            raise AgentStateGraphError(AgentStateErrorCode.STATE_CONFLICT)
        try:
            initial = AgentState.model_validate(
                {
                    **state.model_dump(mode="python"),
                    "run_id": run_id,
                }
            )
            initialize_repair_plan(to_graph_state(initial))
            output = self._graph.invoke(to_graph_state(initial), config)
            if not isinstance(output, Mapping):
                raise TypeError
            return from_graph_state(cast(Mapping[str, object], output))
        except (AgentLoopError, AgentStateGraphError):
            raise
        except ContextAccountingError:
            # Preserve the typed context boundary for sanitized failure evidence.
            raise
        except GraphRecursionError:
            raise AgentStateGraphError(
                AgentStateErrorCode.CHECKPOINT_ERROR, cause_type="graph_recursion_limit"
            ) from None
        except ValidationError:
            raise AgentStateGraphError(
                AgentStateErrorCode.STATE_CONFLICT, cause_type="validation_error"
            ) from None
        except ValueError:
            raise AgentStateGraphError(
                AgentStateErrorCode.STATE_CONFLICT, cause_type="invalid_value"
            ) from None
        except OSError:
            raise AgentStateGraphError(
                AgentStateErrorCode.CHECKPOINT_ERROR, cause_type="local_io_error"
            ) from None
        except Exception:
            raise AgentStateGraphError(
                AgentStateErrorCode.CHECKPOINT_ERROR, cause_type="unexpected_exception"
            ) from None

    def partial_trace_events(self, run_id: UUID) -> tuple[TraceEvent, ...]:
        """Return the recorded prefix even if the graph could not reach a final state."""

        return self._nodes.trace_recorder.events(run_id)

    def load(self, thread_id: str) -> AgentState:
        state = self._read_optional(self._thread_config(thread_id))
        if state is None:
            raise AgentStateGraphError(AgentStateErrorCode.THREAD_NOT_FOUND)
        return state

    def save_working_memory(
        self,
        thread_id: str,
        *,
        store: WorkingMemoryStore,
        sequence: int,
        created_at: datetime | None = None,
    ) -> WorkingMemoryCheckpoint:
        """Explicitly persist one nonterminal repair state and Trace prefix."""

        state = self.load(thread_id)
        if state.run_id is None:
            raise WorkingMemoryError(WorkingMemoryErrorCode.INVALID_SNAPSHOT)
        events = self._nodes.trace_recorder.events(state.run_id)
        checkpoint = build_working_memory_checkpoint(
            task_id=state.task.task_id,
            run_id=state.run_id,
            source_thread_id=thread_id,
            repository_commit=state.task.base_commit,
            context_policy_version=context_policy_version_from_events(events),
            sequence=sequence,
            created_at=created_at or self._clock(),
            state=state,
            trace_events=events,
        )
        store.save(checkpoint)
        return checkpoint

    def resume_from_working_memory(
        self,
        *,
        thread_id: str,
        checkpoint: WorkingMemoryCheckpoint,
    ) -> AgentState:
        """Resume repair in a distinct thread without repeating initialization."""

        verify_working_memory_integrity(checkpoint)
        config = self._thread_config(thread_id)
        if (
            thread_id == checkpoint.source_thread_id
            or checkpoint.context_policy_version != CONTEXT_POLICY_VERSION
            or self._read_optional(config) is not None
        ):
            raise WorkingMemoryError(WorkingMemoryErrorCode.RECOVERY_CONFLICT)
        graph_state = to_graph_state(checkpoint.state)
        _route_repair_recovery(graph_state)
        try:
            self._nodes.trace_recorder.restore(
                checkpoint.run_id,
                checkpoint.trace_events,
            )
            output = self._recovery_graph.invoke(graph_state, config)
            if not isinstance(output, Mapping):
                raise TypeError
            return from_graph_state(cast(Mapping[str, object], output))
        except (AgentLoopError, AgentStateGraphError, WorkingMemoryError, ContextAccountingError):
            raise
        except (ValidationError, ValueError):
            raise WorkingMemoryError(WorkingMemoryErrorCode.INVALID_SNAPSHOT) from None
        except Exception:
            raise AgentStateGraphError(AgentStateErrorCode.CHECKPOINT_ERROR) from None

    def build_trace(
        self,
        thread_id: str,
        *,
        settings: RepoPilotSettings,
    ) -> AgentTrace:
        final_state = self.load(thread_id)
        if final_state.run_id is None:
            raise TraceArtifactError(TraceErrorCode.INVALID_DOCUMENT)
        events = self._nodes.trace_recorder.events(final_state.run_id)
        if not events:
            raise TraceArtifactError(TraceErrorCode.INVALID_DOCUMENT)
        return AgentTrace.build(
            run_id=final_state.run_id,
            task_id=final_state.task.task_id,
            thread_id=thread_id,
            provider_name=self._provider_name,
            settings=cast(dict[str, JsonValue], settings.redacted_snapshot()),
            events=events,
            final_state=final_state,
        )

    def _read_optional(self, config: RunnableConfig) -> AgentState | None:
        try:
            snapshot = self._graph.get_state(config)
            if not snapshot.values:
                return None
            return from_graph_state(cast(Mapping[str, object], snapshot.values))
        except ValidationError:
            raise AgentStateGraphError(AgentStateErrorCode.CHECKPOINT_ERROR) from None
        except AgentStateGraphError:
            raise
        except Exception:
            raise AgentStateGraphError(AgentStateErrorCode.CHECKPOINT_ERROR) from None

    @staticmethod
    def _thread_config(thread_id: str) -> RunnableConfig:
        try:
            identity = ThreadIdentity(thread_id=thread_id)
        except ValidationError:
            raise AgentStateGraphError(AgentStateErrorCode.INVALID_THREAD_ID) from None
        return {"configurable": {"thread_id": identity.thread_id}}


def repair_action_fingerprint(step_id: str, call: ToolCall, *, patch_revision: int = 0) -> str:
    """Scope source reads and validation to a revision; keep patch proposals global."""

    payload: dict[str, object] = {
        "step_id": step_id,
        "tool_name": call.tool_name,
        "validated_arguments": call.validated_arguments,
    }
    if patch_revision > 0 and (
        call.tool_name in ("read_file", "inspect_symbol")
        or (step_id == VALIDATE_STEP and call.tool_name == "run_tests")
    ):
        payload["patch_revision"] = patch_revision
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _route_repair_recovery(
    state: AgentGraphState,
) -> Literal["decide", "execute_tool", "observe", "finish"]:
    snapshot = from_graph_state(state)
    if (
        snapshot.termination_reason is not None
        or snapshot.active_step_id not in STEP_TOOL_CAPABILITIES
    ):
        raise WorkingMemoryError(WorkingMemoryErrorCode.TERMINAL_SNAPSHOT)
    if snapshot.pending_tool_result is not None:
        return "observe"
    if snapshot.pending_tool_call is not None:
        return "execute_tool"
    if snapshot.finish_summary is not None:
        return "finish"
    return "decide"


def _invalid_transition() -> AgentLoopError:
    return AgentLoopError(
        AgentLoopErrorCode.INVALID_STATE_TRANSITION,
        error_type=ErrorType.VALIDATION_ERROR,
    )


__all__ = [
    "FINISH_STEP",
    "INSPECT_STEP",
    "PATCH_STEP",
    "REPAIR_STEP_IDS",
    "REPRODUCE_STEP",
    "SEARCH_STEP",
    "STEP_TOOL_CAPABILITIES",
    "VALIDATE_STEP",
    "RepairAgentGraph",
    "RepairAgentNodes",
    "initialize_repair_plan",
    "repair_action_fingerprint",
    "repair_decision_request_id",
]
