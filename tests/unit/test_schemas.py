"""Contract, round-trip, and hostile-input tests for RepoPilot core schemas."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from repopilot.enums import (
    AgentFailureKind,
    ErrorType,
    PatchErrorCode,
    PatchStatus,
    PlanStepStatus,
    PolicyDecision,
    ProviderErrorCode,
    ReproductionStatus,
    RunStatus,
    SymbolType,
    ToolErrorCode,
    ValidationStatus,
)
from repopilot.enums import (
    TestStatus as RepoTestStatus,
)
from repopilot.schemas import (
    AgentFailure,
    AgentState,
    ArtifactRef,
    BudgetState,
    CodeChunk,
    CommandSpec,
    EvaluationTaskSpec,
    FinalResult,
    LocalizationMetrics,
    Observation,
    PatchState,
    PlanStep,
    ReproductionResult,
    RetrievalHit,
    TaskMetrics,
    TaskSpec,
    ToolCall,
    UsageMetrics,
    ValidationResult,
)
from repopilot.schemas import (
    TestState as RepoTestState,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
CALL_ID = UUID("11111111-1111-4111-8111-111111111111")
STARTED_AT = datetime(2026, 7, 27, 8, 0, tzinfo=UTC)


def load_fixture(name: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((FIXTURES / name).read_text(encoding="utf-8")),
    )


@pytest.mark.parametrize(
    ("model_type", "fixture_name"),
    [
        (TaskSpec, "valid_task.json"),
        (EvaluationTaskSpec, "valid_evaluation_task.json"),
        (AgentState, "valid_agent_state.json"),
        (FinalResult, "valid_final_result.json"),
    ],
)
def test_json_fixtures_round_trip(
    model_type: type[TaskSpec] | type[EvaluationTaskSpec] | type[AgentState] | type[FinalResult],
    fixture_name: str,
) -> None:
    serialized = (FIXTURES / fixture_name).read_text(encoding="utf-8")
    model = model_type.model_validate_json(serialized)

    assert model_type.model_validate_json(model.model_dump_json()) == model


def test_unknown_fields_are_rejected_and_models_are_frozen() -> None:
    payload = load_fixture("valid_task.json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TaskSpec.model_validate(payload)

    task = TaskSpec.model_validate(load_fixture("valid_task.json"))
    with pytest.raises(ValidationError, match="Instance is frozen"):
        task.issue_title = "mutated"


def test_agent_task_rejects_and_never_serializes_gold_data() -> None:
    task_payload = load_fixture("valid_task.json")
    task_payload["expected_patch"] = "secret evaluator data"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TaskSpec.model_validate(task_payload)

    evaluation = EvaluationTaskSpec.model_validate(load_fixture("valid_evaluation_task.json"))
    agent_json = evaluation.task.model_dump_json()

    assert "expected_patch" not in agent_json
    assert "gold_files" not in agent_json
    assert "return {}" not in repr(evaluation)


@pytest.mark.parametrize(
    ("repository_path", "repository_url"),
    [
        (None, None),
        ("/repo", "https://example.com/repo.git"),
    ],
)
def test_task_requires_exactly_one_repository_location(
    repository_path: str | None,
    repository_url: str | None,
) -> None:
    payload = load_fixture("valid_task.json")
    payload["repository_path"] = repository_path
    payload["repository_url"] = repository_url

    with pytest.raises(ValidationError, match="exactly one"):
        TaskSpec.model_validate(payload)


@pytest.mark.parametrize("base_commit", ["abc", "A" * 40, "g" * 40])
def test_task_rejects_noncanonical_git_sha(base_commit: str) -> None:
    payload = load_fixture("valid_task.json")
    payload["base_commit"] = base_commit

    with pytest.raises(ValidationError):
        TaskSpec.model_validate(payload)


@pytest.mark.parametrize("relative_path", ["", "/etc/passwd", "../secret", "a/../b", "a\\b"])
def test_relative_paths_reject_escape_and_ambiguous_forms(relative_path: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactRef(
            relative_path=relative_path,
            sha256="b" * 64,
            size_bytes=1,
        )


@pytest.mark.parametrize(
    "argv",
    [
        (),
        ("",),
        ("   ",),
        ("python\x00bad",),
        ("x" * 4097,),
    ],
)
def test_command_rejects_invalid_argv(argv: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        CommandSpec(argv=argv)


def test_command_rejects_nonpositive_timeout_and_traversing_cwd() -> None:
    with pytest.raises(ValidationError):
        CommandSpec(argv=("pytest",), timeout_seconds=0)
    with pytest.raises(ValidationError):
        CommandSpec(argv=("pytest",), cwd="../outside")


def test_agent_failure_requires_kind_specific_evidence() -> None:
    tool_failure = AgentFailure(
        kind=AgentFailureKind.TOOL_ERROR,
        decision_index=2,
        step_id="test",
        call_id=CALL_ID,
        error_type=ErrorType.SANDBOX_ERROR,
        tool_error_code=ToolErrorCode.TOOL_EXECUTION_ERROR,
    )

    assert AgentFailure.model_validate_json(tool_failure.model_dump_json()) == tool_failure
    with pytest.raises(ValidationError, match="Registry error details"):
        AgentFailure(
            kind=AgentFailureKind.TOOL_ERROR,
            decision_index=2,
            step_id="test",
            call_id=CALL_ID,
        )
    provider_failure = AgentFailure(
        kind=AgentFailureKind.PROVIDER_FAILURE,
        decision_index=0,
        step_id="search",
        error_type=ErrorType.PROVIDER_ERROR,
        provider_error_code=ProviderErrorCode.CONNECTION_ERROR,
    )
    assert AgentFailure.model_validate_json(provider_failure.model_dump_json()) == provider_failure
    with pytest.raises(ValidationError, match="typed provider error details"):
        AgentFailure(
            kind=AgentFailureKind.PROVIDER_FAILURE,
            decision_index=0,
            step_id="search",
            error_type=ErrorType.PROVIDER_ERROR,
        )
    with pytest.raises(ValidationError, match="only valid for provider failures"):
        AgentFailure(
            kind=AgentFailureKind.TOOL_ERROR,
            decision_index=2,
            step_id="test",
            call_id=CALL_ID,
            error_type=ErrorType.SANDBOX_ERROR,
            provider_error_code=ProviderErrorCode.CONNECTION_ERROR,
            tool_error_code=ToolErrorCode.TOOL_EXECUTION_ERROR,
        )
    patch_failure = AgentFailure(
        kind=AgentFailureKind.PATCH_FAILURE,
        decision_index=3,
        step_id="patch",
        call_id=CALL_ID,
        patch_error_code=PatchErrorCode.APPLY_CONFLICT,
    )
    assert AgentFailure.model_validate_json(patch_failure.model_dump_json()) == patch_failure
    with pytest.raises(ValidationError, match="typed patch error"):
        AgentFailure(
            kind=AgentFailureKind.PATCH_FAILURE,
            decision_index=3,
            step_id="patch",
            call_id=CALL_ID,
        )


def test_code_chunk_validates_hash_and_line_span() -> None:
    content = "def parse(): ..."
    base = {
        "chunk_id": "chunk.parser",
        "repository_id": "fixture.repo",
        "commit": "a" * 40,
        "relative_path": "parser.py",
        "symbol_name": "parse",
        "symbol_type": "function",
        "start_line": 10,
        "end_line": 12,
        "language": "python",
        "content": content,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
    assert CodeChunk.model_validate(base).start_line == 10

    with pytest.raises(ValidationError):
        CodeChunk.model_validate({**base, "content_hash": "short"})
    with pytest.raises(ValidationError, match="must match"):
        CodeChunk.model_validate({**base, "content_hash": "b" * 64})
    with pytest.raises(ValidationError, match="end_line"):
        CodeChunk.model_validate({**base, "end_line": 9})


def test_code_chunk_and_tool_call_round_trip_independently() -> None:
    content = "def parse(): ..."
    chunk = CodeChunk(
        chunk_id="chunk.parser",
        repository_id="fixture.repo",
        commit="a" * 40,
        relative_path="parser.py",
        symbol_name="parse",
        symbol_type=SymbolType.FUNCTION,
        start_line=1,
        end_line=1,
        language="python",
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        metadata={"is_test": False},
    )
    call = ToolCall(
        call_id=CALL_ID,
        tool_name="read_file",
        validated_arguments={"relative_path": "parser.py", "line": 1},
        requested_at=STARTED_AT,
    )

    assert CodeChunk.model_validate_json(chunk.model_dump_json()) == chunk
    assert ToolCall.model_validate_json(call.model_dump_json()) == call


def test_tool_call_arguments_must_be_json_values() -> None:
    with pytest.raises(ValidationError):
        ToolCall(
            call_id=CALL_ID,
            tool_name="read_file",
            validated_arguments={"invalid": cast(Any, object())},
            requested_at=STARTED_AT,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"sparse_rank": 1},
        {"dense_score": float("nan"), "dense_rank": 1},
        {"sparse_rank": 0, "sparse_score": 1.0},
    ],
)
def test_retrieval_hit_requires_valid_rank_score_pairs(changes: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "chunk_id": "chunk.parser",
        "query": "parser failure",
        "retrieval_reason": "identifier match",
    }
    payload.update(changes)

    with pytest.raises(ValidationError):
        RetrievalHit.model_validate(payload)


def test_plan_step_rejects_self_dependency_and_unexplained_cancellation() -> None:
    with pytest.raises(ValidationError, match="depend on itself"):
        PlanStep(
            step_id="inspect",
            objective="Inspect the parser.",
            evidence_required=("Relevant implementation lines",),
            dependencies=("inspect",),
        )

    with pytest.raises(ValidationError, match="revision_reason"):
        PlanStep(
            step_id="inspect",
            objective="Inspect the parser.",
            status=PlanStepStatus.CANCELLED,
            evidence_required=("Relevant implementation lines",),
        )


def test_tool_call_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValidationError):
        ToolCall(
            call_id=CALL_ID,
            tool_name="read_file",
            requested_at=datetime(2026, 7, 27, 8, 0),
        )


def test_denied_observation_has_no_process_result() -> None:
    denied = Observation(
        call_id=CALL_ID,
        started_at=STARTED_AT,
        duration_ms=0,
        policy_decision=PolicyDecision.DENIED,
        error_type=ErrorType.POLICY_DENIED,
    )
    assert denied.exit_code is None

    with pytest.raises(ValidationError, match="policy_denied"):
        Observation(
            call_id=CALL_ID,
            started_at=STARTED_AT,
            duration_ms=0,
            policy_decision=PolicyDecision.DENIED,
        )
    with pytest.raises(ValidationError, match="exit_code"):
        Observation(
            call_id=CALL_ID,
            started_at=STARTED_AT,
            duration_ms=0,
            exit_code=126,
            policy_decision=PolicyDecision.DENIED,
            error_type=ErrorType.POLICY_DENIED,
        )


def test_budget_rejects_negative_usage_and_invalid_deadline() -> None:
    base = {
        "max_steps": 30,
        "max_tokens": 1000,
        "max_cost_usd": "1.0",
        "started_at": "2026-07-27T08:00:00Z",
        "deadline_at": "2026-07-27T08:10:00Z",
    }
    with pytest.raises(ValidationError):
        BudgetState.model_validate({**base, "tokens_used": -1})
    with pytest.raises(ValidationError, match="deadline_at"):
        BudgetState.model_validate({**base, "deadline_at": "2026-07-27T07:59:00Z"})


def test_budget_can_record_the_terminal_transition_that_exceeded_a_limit() -> None:
    budget = BudgetState(
        steps_used=31,
        max_steps=30,
        tokens_used=1001,
        max_tokens=1000,
        cost_used_usd=Decimal("1.01"),
        max_cost_usd=Decimal("1.00"),
        started_at=STARTED_AT,
        deadline_at=datetime(2026, 7, 27, 8, 10, tzinfo=UTC),
    )

    assert budget.steps_used > budget.max_steps


def test_agent_state_validates_active_and_unique_plan_step_ids() -> None:
    payload = load_fixture("valid_agent_state.json")
    payload["active_step_id"] = "missing"
    with pytest.raises(ValidationError, match="active_step_id"):
        AgentState.model_validate(payload)

    payload = load_fixture("valid_agent_state.json")
    payload["current_plan"].append(dict(payload["current_plan"][0]))
    with pytest.raises(ValidationError, match="step IDs must be unique"):
        AgentState.model_validate(payload)


def test_patch_and_test_state_validate_artifact_references() -> None:
    with pytest.raises(ValidationError, match="requires a diff"):
        PatchState(status=PatchStatus.PROPOSED, files_changed=("parser.py",))
    with pytest.raises(ValidationError, match="cannot have diff"):
        PatchState(
            status=PatchStatus.NONE,
            diff_ref=ArtifactRef(
                relative_path="candidate.diff",
                sha256="c" * 64,
                size_bytes=1,
            ),
        )
    with pytest.raises(ValidationError, match="call reference"):
        RepoTestState(status=RepoTestStatus.PASSED, commands_run=1)


def test_result_components_reject_inconsistent_metrics() -> None:
    with pytest.raises(ValidationError, match="monotonic"):
        LocalizationMetrics(
            file_recall_at_1=1,
            file_recall_at_3=0,
            file_recall_at_5=1,
            mean_reciprocal_rank=1,
            first_relevant_file_rank=1,
            indexing_latency_ms=0,
            retrieval_latency_ms=0,
        )
    with pytest.raises(ValidationError, match="reciprocal"):
        LocalizationMetrics(
            file_recall_at_1=0,
            file_recall_at_3=1,
            file_recall_at_5=1,
            mean_reciprocal_rank=0.25,
            first_relevant_file_rank=2,
            indexing_latency_ms=0,
            retrieval_latency_ms=0,
        )
    with pytest.raises(ValidationError, match="applied patch"):
        TaskMetrics(
            issue_reproduced=True,
            patch_applied=False,
            tests_passed=True,
            task_success=True,
            tool_steps=1,
        )
    with pytest.raises(ValidationError, match="reproduced issue"):
        TaskMetrics(
            issue_reproduced=False,
            patch_applied=True,
            tests_passed=True,
            task_success=True,
            tool_steps=1,
        )
    with pytest.raises(ValidationError, match="total_tokens"):
        UsageMetrics(
            input_tokens=10,
            output_tokens=5,
            total_tokens=14,
            estimated_cost_usd=Decimal("0"),
            latency_ms=1,
        )


def test_reproduction_and_validation_require_observation_evidence() -> None:
    with pytest.raises(ValidationError, match="observation"):
        ReproductionResult(status=ReproductionStatus.REPRODUCED)
    with pytest.raises(ValidationError, match="observation"):
        ValidationResult(status=ValidationStatus.PASSED, tests_passed=1)
    with pytest.raises(ValidationError, match="failed tests"):
        ValidationResult(
            status=ValidationStatus.PASSED,
            observation_call_id=CALL_ID,
            tests_passed=1,
            tests_failed=1,
        )


def test_successful_final_result_rejects_contradictory_evidence() -> None:
    payload = load_fixture("valid_final_result.json")
    payload["validation_result"]["status"] = "failed"

    with pytest.raises(ValidationError, match="structured test result"):
        FinalResult.model_validate(payload)

    payload = load_fixture("valid_final_result.json")
    payload["patch_path"] = None
    with pytest.raises(ValidationError, match="patch_path"):
        FinalResult.model_validate(payload)

    payload = load_fixture("valid_final_result.json")
    payload["status"] = RunStatus.FAILED
    with pytest.raises(ValidationError, match="non-successful"):
        FinalResult.model_validate(payload)
