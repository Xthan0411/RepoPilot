"""Immutable, JSON-serializable data contracts shared across RepoPilot modules.

This module defines data only. Provider calls, tool execution, state reduction,
verification, and metric computation belong to later phases.
"""

from __future__ import annotations

import hashlib
import math
from decimal import Decimal
from pathlib import PurePosixPath
from typing import Annotated, Self
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from repopilot.enums import (
    AgentFailureKind,
    DatasetSplit,
    ErrorCategory,
    ErrorType,
    MemoryAccessMode,
    PatchErrorCode,
    PatchStatus,
    PlanStepStatus,
    PolicyDecision,
    ProviderErrorCode,
    RepositoryErrorCode,
    RepositoryMemoryConfirmation,
    RepositoryMemoryKind,
    RepositoryMemoryLookupStatus,
    ReproductionStatus,
    RunStatus,
    SymbolType,
    TerminationReason,
    TestRunCategory,
    TestStatus,
    ToolErrorCode,
    ToolResultStatus,
    ValidationStatus,
    VerifiedEpisodeLookupStatus,
    VerifierErrorCode,
)


def _validate_relative_path(value: str) -> str:
    """Accept one canonical POSIX path contained beneath a future workspace root."""

    if not value or "\x00" in value or "\\" in value:
        raise ValueError("path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("path must stay beneath the workspace root")
    if value in {".", ".."} or path.as_posix() != value:
        raise ValueError("path must be canonical")
    return value


RelativePath = Annotated[str, AfterValidator(_validate_relative_path)]
TaskId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"),
]
GitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$"),
]
ModelIdentifier = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$"),
]
VersionIdentifier = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"),
]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FiniteScore = Annotated[float, Field(allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class StrictModel(BaseModel):
    """Base contract: reject unknown fields and prevent field reassignment."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ArtifactRef(StrictModel):
    """Metadata for a file stored in the controlled artifact directory."""

    relative_path: RelativePath
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    media_type: str | None = Field(default=None, min_length=1, max_length=255)


class CommandSpec(StrictModel):
    """A shell-free command represented as an executable and argument vector."""

    argv: tuple[str, ...] = Field(min_length=1, max_length=128)
    cwd: RelativePath | None = None
    timeout_seconds: int | None = Field(default=None, gt=0)

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for argument in value:
            if not argument.strip():
                raise ValueError("command arguments must not be blank")
            if "\x00" in argument:
                raise ValueError("command arguments must not contain NUL bytes")
            if len(argument) > 4096:
                raise ValueError("command arguments must not exceed 4096 characters")
        return value


class TaskSpec(StrictModel):
    """Agent-visible task input with no evaluator-only gold data."""

    task_id: TaskId
    repository_path: str | None = None
    repository_url: HttpUrl | None = None
    base_commit: GitSha
    issue_title: NonEmptyText
    issue_body: NonEmptyText
    setup_commands: tuple[CommandSpec, ...] = ()
    test_commands: tuple[CommandSpec, ...] = Field(min_length=1)
    timeout_seconds: int = Field(default=900, gt=0)

    @field_validator("repository_path")
    @classmethod
    def validate_repository_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or "\x00" in normalized:
            raise ValueError("repository_path must not be blank or contain NUL bytes")
        return normalized

    @model_validator(mode="after")
    def validate_repository_location(self) -> Self:
        if (self.repository_path is None) == (self.repository_url is None):
            raise ValueError("exactly one of repository_path or repository_url is required")
        return self


class EvaluationTaskSpec(StrictModel):
    """Evaluator-owned task wrapper containing gold data hidden from the Agent."""

    task: TaskSpec
    expected_patch: str = Field(min_length=1, repr=False)
    gold_files: tuple[RelativePath, ...] = Field(min_length=1)
    split: DatasetSplit

    @field_validator("gold_files")
    @classmethod
    def validate_unique_gold_files(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("gold_files must be unique")
        return value


class CodeChunk(StrictModel):
    """A version-pinned repository chunk produced by an indexer."""

    chunk_id: Identifier
    repository_id: Identifier
    commit: GitSha
    relative_path: RelativePath
    symbol_name: str | None = Field(default=None, min_length=1, max_length=512)
    symbol_type: SymbolType
    start_line: int = Field(gt=0)
    end_line: int = Field(gt=0)
    language: Identifier
    content: str
    content_hash: Sha256
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_line_span(self) -> Self:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        expected_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError("content_hash must match UTF-8 chunk content")
        return self


class RetrievalHit(StrictModel):
    """Normalized sparse, dense, or fused retrieval output."""

    chunk_id: Identifier
    sparse_rank: int | None = Field(default=None, gt=0)
    sparse_score: FiniteScore | None = None
    dense_rank: int | None = Field(default=None, gt=0)
    dense_score: FiniteScore | None = None
    fused_score: FiniteScore | None = None
    query: NonEmptyText
    retrieval_reason: NonEmptyText

    @model_validator(mode="after")
    def validate_rank_score_pairs(self) -> Self:
        if self.sparse_rank is None and self.dense_rank is None:
            raise ValueError("at least one retrieval rank is required")
        if (self.sparse_rank is None) != (self.sparse_score is None):
            raise ValueError("sparse_rank and sparse_score must be provided together")
        if (self.dense_rank is None) != (self.dense_score is None):
            raise ValueError("dense_rank and dense_score must be provided together")
        return self


class PlanStep(StrictModel):
    """One explicit, evidence-seeking step in the Agent plan."""

    step_id: Identifier
    objective: NonEmptyText
    status: PlanStepStatus = PlanStepStatus.PENDING
    evidence_required: tuple[NonEmptyText, ...] = Field(min_length=1)
    dependencies: tuple[Identifier, ...] = ()
    revision_reason: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_dependencies_and_revision(self) -> Self:
        if self.step_id in self.dependencies:
            raise ValueError("a plan step cannot depend on itself")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("plan step dependencies must be unique")
        if self.status is PlanStepStatus.CANCELLED and self.revision_reason is None:
            raise ValueError("cancelled plan steps require a revision_reason")
        return self


class ToolCall(StrictModel):
    """A model-requested tool invocation after argument validation."""

    call_id: UUID
    tool_name: Identifier
    validated_arguments: dict[str, JsonValue] = Field(default_factory=dict)
    requested_at: AwareDatetime


class ToolExecutionResult(StrictModel):
    """Stable success/error envelope returned for every Registry invocation."""

    call_id: UUID
    tool_name: Identifier
    status: ToolResultStatus
    payload: dict[str, JsonValue] | None = None
    output_bytes: int = Field(default=0, ge=0)
    truncated: bool = False
    error_type: ErrorType | None = None
    error_code: ToolErrorCode | None = None
    repository_error_code: RepositoryErrorCode | None = None
    message: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.status is ToolResultStatus.SUCCESS:
            if self.payload is None:
                raise ValueError("successful tool results require a payload")
            if any(
                value is not None
                for value in (
                    self.error_type,
                    self.error_code,
                    self.repository_error_code,
                    self.message,
                )
            ):
                raise ValueError("successful tool results cannot contain error details")
        else:
            if self.payload is not None or self.output_bytes != 0:
                raise ValueError("failed tool results cannot contain output")
            if self.error_type is None or self.error_code is None or self.message is None:
                raise ValueError("failed tool results require structured error details")
            if (self.error_code is ToolErrorCode.REPOSITORY_ERROR) != (
                self.repository_error_code is not None
            ):
                raise ValueError("repository_error_code is required only for repository failures")
            if self.truncated and self.error_code is not ToolErrorCode.OUTPUT_TOO_LARGE:
                raise ValueError("only oversized output errors may be truncated")
        return self


class AgentFailure(StrictModel):
    """One structured loop failure used for recovery and bad-case diagnosis."""

    kind: AgentFailureKind
    decision_index: int = Field(ge=0)
    step_id: Identifier
    call_id: UUID | None = None
    error_type: ErrorType | None = None
    provider_error_code: ProviderErrorCode | None = None
    patch_error_code: PatchErrorCode | None = None
    tool_error_code: ToolErrorCode | None = None

    @model_validator(mode="after")
    def validate_failure_evidence(self) -> Self:
        if self.kind is AgentFailureKind.PROVIDER_FAILURE:
            if (
                self.call_id is not None
                or self.error_type is None
                or self.provider_error_code is None
            ):
                raise ValueError("provider failure requires typed provider error details")
        elif self.call_id is None:
            raise ValueError("action failures require a tool call ID")
        if (
            self.kind is not AgentFailureKind.PROVIDER_FAILURE
            and self.provider_error_code is not None
        ):
            raise ValueError("provider_error_code is only valid for provider failures")
        if self.kind is AgentFailureKind.PATCH_FAILURE:
            if self.call_id is None or self.patch_error_code is None:
                raise ValueError("patch failure requires a call and typed patch error")
        elif self.patch_error_code is not None:
            raise ValueError("patch_error_code is only valid for patch failures")
        if self.kind is AgentFailureKind.TOOL_ERROR:
            if self.error_type is None or self.tool_error_code is None:
                raise ValueError("tool failure requires typed Registry error details")
        elif self.tool_error_code is not None:
            raise ValueError("tool_error_code is only valid for tool failures")
        return self


class Observation(StrictModel):
    """Auditable result of one allowed or denied tool invocation."""

    call_id: UUID
    started_at: AwareDatetime
    duration_ms: int = Field(ge=0)
    exit_code: int | None = None
    stdout_ref: ArtifactRef | None = None
    stderr_ref: ArtifactRef | None = None
    stdout_preview: str | None = None
    stderr_preview: str | None = None
    truncated: bool = False
    sandbox_id: str | None = Field(default=None, min_length=1, max_length=255)
    policy_decision: PolicyDecision
    error_type: ErrorType | None = None

    @model_validator(mode="after")
    def validate_policy_outcome(self) -> Self:
        if self.policy_decision is PolicyDecision.DENIED:
            if self.error_type is not ErrorType.POLICY_DENIED:
                raise ValueError("denied observations require policy_denied error_type")
            if self.exit_code is not None:
                raise ValueError("denied observations cannot have an exit_code")
            if any(
                value is not None
                for value in (
                    self.stdout_ref,
                    self.stderr_ref,
                    self.stdout_preview,
                    self.stderr_preview,
                )
            ):
                raise ValueError("denied observations cannot contain execution output")
        elif self.error_type is ErrorType.POLICY_DENIED:
            raise ValueError("allowed observations cannot use policy_denied error_type")
        if self.stdout_preview is not None and self.stdout_ref is None:
            raise ValueError("stdout preview requires an artifact reference")
        if self.stderr_preview is not None and self.stderr_ref is None:
            raise ValueError("stderr preview requires an artifact reference")
        if any(
            len(value.encode("utf-8")) > 10_000_000
            for value in (self.stdout_preview, self.stderr_preview)
            if value is not None
        ):
            raise ValueError("observation preview exceeds the global byte limit")
        if self.truncated and self.stdout_ref is None and self.stderr_ref is None:
            raise ValueError("truncated observations require an artifact reference")
        return self


class BudgetState(StrictModel):
    """Budget counters and limits captured at one Agent-state transition."""

    steps_used: int = Field(default=0, ge=0)
    max_steps: int = Field(gt=0)
    tokens_used: int = Field(default=0, ge=0)
    max_tokens: int = Field(gt=0)
    cost_used_usd: Decimal = Field(default=Decimal("0"), ge=0)
    max_cost_usd: Decimal = Field(gt=0)
    started_at: AwareDatetime
    deadline_at: AwareDatetime

    @model_validator(mode="after")
    def validate_time_window(self) -> Self:
        if self.deadline_at <= self.started_at:
            raise ValueError("deadline_at must be later than started_at")
        return self


class PatchState(StrictModel):
    """Current candidate-patch metadata without embedding diff contents."""

    status: PatchStatus = PatchStatus.NONE
    diff_ref: ArtifactRef | None = None
    files_changed: tuple[RelativePath, ...] = ()
    revision: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_patch_artifacts(self) -> Self:
        if len(set(self.files_changed)) != len(self.files_changed):
            raise ValueError("files_changed must be unique")
        if self.status is PatchStatus.NONE:
            if self.diff_ref is not None or self.files_changed:
                raise ValueError("an absent patch cannot have diff metadata")
        elif self.diff_ref is None or not self.files_changed:
            raise ValueError("a candidate patch requires a diff and changed files")
        return self


class TestRunResult(StrictModel):
    """Structured meaning extracted from one artifact-backed test Observation."""

    observation_call_id: UUID
    command_index: int = Field(ge=0, strict=True)
    command_sha256: Sha256
    status: TestStatus
    category: TestRunCategory
    exit_code: int | None = None
    duration_ms: int = Field(ge=0)
    tests_run: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    failing_tests: tuple[NonEmptyText, ...] = ()
    stdout_ref: ArtifactRef | None = None
    stderr_ref: ArtifactRef | None = None
    truncated: bool = False
    error_type: ErrorType | None = None
    summary: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if len(set(self.failing_tests)) != len(self.failing_tests):
            raise ValueError("failing test names must be unique")
        if self.failures + self.errors > self.tests_run and self.tests_run > 0:
            raise ValueError("test failure counts cannot exceed tests run")
        if self.truncated and self.stdout_ref is None and self.stderr_ref is None:
            raise ValueError("truncated test output requires an artifact reference")

        failed_categories = {
            TestRunCategory.ASSERTION_FAILURE,
            TestRunCategory.TEST_ERROR,
        }
        error_categories = {
            TestRunCategory.IMPORT_ERROR,
            TestRunCategory.NO_TESTS_COLLECTED,
            TestRunCategory.OUTPUT_TRUNCATED,
            TestRunCategory.TIMEOUT,
            TestRunCategory.POLICY_DENIED,
            TestRunCategory.SANDBOX_ERROR,
            TestRunCategory.EXECUTION_ERROR,
            TestRunCategory.UNKNOWN_NONZERO,
        }
        if self.category is TestRunCategory.PASSED:
            if (
                self.status is not TestStatus.PASSED
                or self.exit_code != 0
                or self.failures
                or self.errors
                or self.error_type is not None
            ):
                raise ValueError("passed test result contains contradictory evidence")
        elif self.category in failed_categories:
            if (
                self.status is not TestStatus.FAILED
                or self.exit_code in {None, 0}
                or self.error_type is not None
            ):
                raise ValueError("failed test result contains contradictory evidence")
        elif self.category in error_categories:
            if self.status is not TestStatus.ERROR or self.error_type is None:
                raise ValueError("test infrastructure errors require typed error evidence")
        else:
            raise ValueError("test result category is unsupported")

        required_errors = {
            TestRunCategory.IMPORT_ERROR: ErrorType.TEST_ERROR,
            TestRunCategory.NO_TESTS_COLLECTED: ErrorType.TEST_ERROR,
            TestRunCategory.OUTPUT_TRUNCATED: ErrorType.TEST_ERROR,
            TestRunCategory.TIMEOUT: ErrorType.TOOL_TIMEOUT,
            TestRunCategory.POLICY_DENIED: ErrorType.POLICY_DENIED,
            TestRunCategory.SANDBOX_ERROR: ErrorType.SANDBOX_ERROR,
            TestRunCategory.UNKNOWN_NONZERO: ErrorType.TEST_ERROR,
        }
        required_error = required_errors.get(self.category)
        if required_error is not None and self.error_type is not required_error:
            raise ValueError("test result category has the wrong error type")
        return self


class TestState(StrictModel):
    """Current test-execution summary held in Agent state."""

    status: TestStatus = TestStatus.NOT_RUN
    last_observation_call_id: UUID | None = None
    commands_run: int = Field(default=0, ge=0)
    last_result: TestRunResult | None = None

    @model_validator(mode="after")
    def validate_execution_reference(self) -> Self:
        if self.status is TestStatus.NOT_RUN:
            if (
                self.last_observation_call_id is not None
                or self.commands_run != 0
                or self.last_result is not None
            ):
                raise ValueError("not_run test state cannot reference executed commands")
        elif self.last_observation_call_id is None or self.commands_run == 0:
            raise ValueError("an executed test state requires a call reference and count")
        if self.last_result is not None and (
            self.status is not self.last_result.status
            or self.last_observation_call_id != self.last_result.observation_call_id
        ):
            raise ValueError("test state must agree with its structured result")
        return self


class MemoryNamespace(StrictModel):
    """Trusted experiment/snapshot/split boundary kept outside Agent task input."""

    experiment_id: Identifier
    snapshot_id: Identifier
    split: DatasetSplit
    access_mode: MemoryAccessMode

    @field_validator("snapshot_id")
    @classmethod
    def reject_mutable_snapshot_aliases(cls, value: str) -> str:
        if value.casefold() in {"current", "head", "latest"}:
            raise ValueError("Memory snapshot ID must identify one frozen snapshot")
        return value

    @model_validator(mode="after")
    def require_read_only_holdout(self) -> Self:
        if (
            self.split is DatasetSplit.HOLDOUT
            and self.access_mode is not MemoryAccessMode.READ_ONLY
        ):
            raise ValueError("holdout Memory namespace must be read-only")
        return self


class MemoryLoadAudit(StrictModel):
    """Trace-visible result of namespace-scoped Memory lookup and projection."""

    namespace: MemoryNamespace
    repository_status: RepositoryMemoryLookupStatus | None = None
    episodic_status: VerifiedEpisodeLookupStatus | None = None
    repository_records_loaded: int = Field(default=0, ge=0)
    episodic_records_loaded: int = Field(default=0, ge=0)
    expired_episode_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.repository_status is None and self.repository_records_loaded != 0:
            raise ValueError("Repository Memory count requires a lookup status")
        if (self.repository_status is RepositoryMemoryLookupStatus.CURRENT) != (
            self.repository_records_loaded > 0
        ):
            raise ValueError("Repository Memory current status must match loaded records")
        if self.episodic_status is None and (
            self.episodic_records_loaded != 0 or self.expired_episode_count != 0
        ):
            raise ValueError("Episodic Memory counts require a lookup status")
        if (self.episodic_status is VerifiedEpisodeLookupStatus.CURRENT) != (
            self.episodic_records_loaded > 0
        ):
            raise ValueError("Episodic Memory current status must match loaded records")
        if (
            self.episodic_status is VerifiedEpisodeLookupStatus.EXPIRED
            and self.expired_episode_count == 0
        ):
            raise ValueError("expired Episodic Memory status requires an expired count")
        return self


class RepositoryMemoryItem(StrictModel):
    """Bounded Agent-visible projection of one admitted Repository Memory record."""

    record_id: Identifier
    repository_id: Identifier
    repository_commit: GitSha
    repository_content_sha256: Sha256
    kind: RepositoryMemoryKind
    confirmation: RepositoryMemoryConfirmation
    producer_version: VersionIdentifier
    content: str = Field(min_length=1, max_length=8_000)
    content_sha256: Sha256
    evidence_count: int = Field(ge=1, le=32)
    evidence_sha256: Sha256
    source_record_sha256: Sha256

    @model_validator(mode="after")
    def validate_content_digest(self) -> Self:
        expected = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.content_sha256 != expected:
            raise ValueError("Repository Memory content digest does not match")
        return self


class VerifiedEpisodeMemoryItem(StrictModel):
    """Bounded low-trust Context projection of one unexpired verified episode."""

    episode_id: Identifier
    source_task_id: TaskId
    source_run_id: UUID
    repository_id: Identifier
    repository_commit: GitSha
    repository_content_sha256: Sha256
    problem_statement: str = Field(min_length=1, max_length=500)
    issue_fingerprint: Sha256
    repair_summary: str = Field(min_length=1, max_length=500)
    files_changed: tuple[RelativePath, ...] = Field(min_length=1, max_length=128)
    test_command_sha256: Sha256
    patch_artifact_sha256: Sha256
    source_result_sha256: Sha256
    source_trace_sha256: Sha256
    artifacts_sha256: Sha256
    verifier_version: VersionIdentifier
    model_provider: str = Field(min_length=1, max_length=256)
    validation_tests_passed: int = Field(ge=1)
    created_at: AwareDatetime
    expires_at: AwareDatetime
    source_record_sha256: Sha256

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        if self.files_changed != tuple(sorted(self.files_changed)) or len(
            set(self.files_changed)
        ) != len(self.files_changed):
            raise ValueError("verified episode files must be unique and sorted")
        if self.expires_at <= self.created_at:
            raise ValueError("verified episode projection must have a positive TTL")
        return self


class AgentState(StrictModel):
    """Checkpointable state shared by future deterministic Agent nodes."""

    task: TaskSpec
    run_id: UUID | None = None
    repository_summary: str = ""
    memory_namespace: MemoryNamespace | None = None
    memory_load_audit: MemoryLoadAudit | None = None
    repository_memory: tuple[RepositoryMemoryItem, ...] = ()
    episodic_memory: tuple[VerifiedEpisodeMemoryItem, ...] = ()
    current_plan: tuple[PlanStep, ...] = ()
    active_step_id: Identifier | None = None
    retrieved_context: tuple[RetrievalHit, ...] = ()
    working_memory: tuple[str, ...] = ()
    observations: tuple[Observation, ...] = ()
    pending_tool_call: ToolCall | None = None
    pending_tool_result: ToolExecutionResult | None = None
    consumed_call_ids: tuple[UUID, ...] = ()
    tool_results: tuple[ToolExecutionResult, ...] = ()
    failure_history: tuple[AgentFailure, ...] = ()
    action_fingerprints: tuple[Sha256, ...] = ()
    replan_count: int = Field(default=0, ge=0)
    decision_count: int = Field(default=0, ge=0)
    finish_summary: str | None = Field(default=None, min_length=1)
    patch_state: PatchState = Field(default_factory=PatchState)
    test_state: TestState = Field(default_factory=TestState)
    baseline_test_result: TestRunResult | None = None
    budget: BudgetState
    termination_reason: TerminationReason | None = None

    @model_validator(mode="after")
    def validate_plan_references(self) -> Self:
        step_ids = tuple(step.step_id for step in self.current_plan)
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("current_plan step IDs must be unique")
        if self.active_step_id is not None and self.active_step_id not in step_ids:
            raise ValueError("active_step_id must reference a current plan step")
        observation_ids = tuple(observation.call_id for observation in self.observations)
        if len(set(self.working_memory)) != len(self.working_memory):
            raise ValueError("working memory entries must be unique")
        memory_ids = tuple(item.record_id for item in self.repository_memory)
        if len(set(memory_ids)) != len(memory_ids):
            raise ValueError("Repository Memory record IDs must be unique")
        if any(item.repository_commit != self.task.base_commit for item in self.repository_memory):
            raise ValueError("Repository Memory commit must match Agent task")
        memory_namespaces = {
            (item.repository_id, item.repository_content_sha256) for item in self.repository_memory
        }
        if len(memory_namespaces) > 1:
            raise ValueError("Repository Memory must use one current content namespace")
        episode_ids = tuple(item.episode_id for item in self.episodic_memory)
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("Verified Episodic Memory episode IDs must be unique")
        if any(item.repository_commit != self.task.base_commit for item in self.episodic_memory):
            raise ValueError("Verified Episodic Memory commit must match Agent task")
        episode_namespaces = {
            (item.repository_id, item.repository_content_sha256) for item in self.episodic_memory
        }
        if len(episode_namespaces) > 1:
            raise ValueError("Verified Episodic Memory must use one current content namespace")
        if memory_namespaces and episode_namespaces and memory_namespaces != episode_namespaces:
            raise ValueError("Repository and Episodic Memory namespaces must match")
        if (
            self.memory_load_audit is not None
            and self.memory_load_audit.namespace != self.memory_namespace
        ):
            raise ValueError("Memory audit must match the Agent Memory namespace")
        if self.episodic_memory and self.memory_namespace is None:
            raise ValueError("Verified Episodic Memory requires an explicit namespace")
        if self.memory_load_audit is not None and (
            self.memory_load_audit.repository_records_loaded != len(self.repository_memory)
            or self.memory_load_audit.episodic_records_loaded != len(self.episodic_memory)
        ):
            raise ValueError("Memory audit counts must match Agent-visible records")
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("observation call IDs must be unique")
        if len(set(self.consumed_call_ids)) != len(self.consumed_call_ids):
            raise ValueError("consumed call IDs must be unique")
        result_ids = tuple(result.call_id for result in self.tool_results)
        if len(set(result_ids)) != len(result_ids):
            raise ValueError("tool result call IDs must be unique")
        if not set(result_ids).issubset(self.consumed_call_ids):
            raise ValueError("recorded tool results must reference consumed call IDs")
        failure_keys = tuple(
            (failure.decision_index, failure.kind) for failure in self.failure_history
        )
        if len(set(failure_keys)) != len(failure_keys):
            raise ValueError("failure history decision/kind pairs must be unique")
        if len(set(self.action_fingerprints)) != len(self.action_fingerprints):
            raise ValueError("action fingerprints must be unique")
        if self.pending_tool_result is not None:
            if self.pending_tool_call is None:
                raise ValueError("a pending tool result requires a pending tool call")
            if (
                self.pending_tool_result.call_id != self.pending_tool_call.call_id
                or self.pending_tool_result.tool_name != self.pending_tool_call.tool_name
            ):
                raise ValueError("pending tool result must match its pending call")
        if self.baseline_test_result is not None:
            if self.baseline_test_result.observation_call_id not in observation_ids:
                raise ValueError("baseline test result must reference a recorded observation")
            if self.test_state.commands_run == 0:
                raise ValueError("baseline test result requires an executed test state")
        if (
            self.pending_tool_call is not None
            and self.pending_tool_call.call_id in self.consumed_call_ids
        ):
            raise ValueError("a pending tool call cannot already be consumed")
        if self.run_id is None and (
            self.decision_count != 0
            or self.pending_tool_call is not None
            or self.tool_results
            or self.failure_history
            or self.action_fingerprints
            or self.replan_count != 0
            or self.finish_summary is not None
        ):
            raise ValueError("active Agent-loop state requires run_id")
        if self.termination_reason is not None and (
            self.pending_tool_call is not None or self.pending_tool_result is not None
        ):
            raise ValueError("terminal Agent state cannot contain pending tool work")
        return self


class ReproductionResult(StrictModel):
    """Structured issue-reproduction outcome used by the future Verifier."""

    status: ReproductionStatus
    observation_call_id: UUID | None = None
    failing_tests: tuple[NonEmptyText, ...] = ()
    summary: str = ""
    test_run: TestRunResult | None = None

    @model_validator(mode="after")
    def validate_reproduction_evidence(self) -> Self:
        if self.status is ReproductionStatus.NOT_ATTEMPTED:
            if (
                self.observation_call_id is not None
                or self.failing_tests
                or self.test_run is not None
            ):
                raise ValueError("not_attempted reproduction cannot contain run evidence")
        elif self.observation_call_id is None:
            raise ValueError("attempted reproduction requires an observation reference")
        if self.test_run is not None:
            if self.observation_call_id != self.test_run.observation_call_id:
                raise ValueError("reproduction and test result observations must match")
            expected_status = {
                TestStatus.PASSED: ReproductionStatus.NOT_REPRODUCED,
                TestStatus.FAILED: ReproductionStatus.REPRODUCED,
                TestStatus.ERROR: ReproductionStatus.ERROR,
            }.get(self.test_run.status)
            if self.status is not expected_status:
                raise ValueError("reproduction status must match its structured test result")
            if self.failing_tests != self.test_run.failing_tests:
                raise ValueError("reproduction failing tests must match structured evidence")
        return self


class ValidationResult(StrictModel):
    """Structured candidate-validation outcome used by the future Verifier."""

    status: ValidationStatus
    observation_call_id: UUID | None = None
    tests_passed: int = Field(default=0, ge=0)
    tests_failed: int = Field(default=0, ge=0)
    summary: str = ""
    test_run: TestRunResult | None = None

    @model_validator(mode="after")
    def validate_test_counts(self) -> Self:
        if self.status is ValidationStatus.NOT_RUN:
            if (
                self.observation_call_id is not None
                or self.tests_passed != 0
                or self.tests_failed != 0
                or self.test_run is not None
            ):
                raise ValueError("not_run validation cannot contain run evidence")
        elif self.observation_call_id is None:
            raise ValueError("attempted validation requires an observation reference")
        if self.status is ValidationStatus.PASSED and self.tests_failed != 0:
            raise ValueError("passed validation cannot contain failed tests")
        if self.test_run is not None:
            if self.observation_call_id != self.test_run.observation_call_id:
                raise ValueError("validation and test result observations must match")
            expected_status = {
                TestStatus.PASSED: ValidationStatus.PASSED,
                TestStatus.FAILED: ValidationStatus.FAILED,
                TestStatus.ERROR: ValidationStatus.ERROR,
            }.get(self.test_run.status)
            if self.status is not expected_status:
                raise ValueError("validation status must match its structured test result")
            expected_failed = self.test_run.failures + self.test_run.errors
            if self.tests_failed != expected_failed:
                raise ValueError("validation failed-test count must match structured evidence")
            expected_passed = max(self.test_run.tests_run - expected_failed, 0)
            if self.tests_passed != expected_passed:
                raise ValueError("validation passed-test count must match structured evidence")
        return self


class LocalizationMetrics(StrictModel):
    """File-localization metrics computed by an evaluator, not by the Agent."""

    file_recall_at_1: UnitInterval
    file_recall_at_3: UnitInterval
    file_recall_at_5: UnitInterval
    mean_reciprocal_rank: UnitInterval
    first_relevant_file_rank: int | None = Field(default=None, gt=0)
    indexing_latency_ms: int = Field(ge=0)
    retrieval_latency_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_recall_order(self) -> Self:
        if not (self.file_recall_at_1 <= self.file_recall_at_3 <= self.file_recall_at_5):
            raise ValueError("file recall must be monotonic as K increases")
        if self.mean_reciprocal_rank > 0 and self.first_relevant_file_rank is None:
            raise ValueError("positive MRR requires first_relevant_file_rank")
        if self.first_relevant_file_rank is not None:
            expected_mrr = 1.0 / self.first_relevant_file_rank
            if not math.isclose(self.mean_reciprocal_rank, expected_mrr, abs_tol=1e-9):
                raise ValueError("MRR must equal the reciprocal of first relevant rank")
        return self


class TaskMetrics(StrictModel):
    """Execution-level task outcomes calculated from evidence."""

    issue_reproduced: bool
    patch_applied: bool
    tests_passed: bool
    task_success: bool
    tool_steps: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_success_components(self) -> Self:
        if self.task_success and not (
            self.issue_reproduced and self.patch_applied and self.tests_passed
        ):
            raise ValueError(
                "task success requires reproduced issue, applied patch, and passing tests"
            )
        return self


class UsageMetrics(StrictModel):
    """Provider usage, cost, and end-to-end latency for one run."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: Decimal = Field(ge=0)
    latency_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total_tokens(self) -> Self:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens plus output_tokens")
        return self


class FinalResult(StrictModel):
    """Stable handoff from a completed run to reporting and evaluation."""

    task_id: TaskId
    run_id: UUID
    status: RunStatus
    termination_reason: TerminationReason
    patch_path: RelativePath | None = None
    reproduction_result: ReproductionResult
    validation_result: ValidationResult
    localization_metrics: LocalizationMetrics
    task_metrics: TaskMetrics
    usage: UsageMetrics
    trace_path: RelativePath
    error_category: ErrorCategory | None = None
    verifier_error_code: VerifierErrorCode | None = None

    @model_validator(mode="after")
    def validate_outcome_consistency(self) -> Self:
        if self.task_metrics.patch_applied and self.patch_path is None:
            raise ValueError("an applied patch requires patch_path")
        if self.status is RunStatus.SUCCESS:
            if self.termination_reason is not TerminationReason.SUCCESS:
                raise ValueError("a successful run requires success termination")
            if self.validation_result.status is not ValidationStatus.PASSED:
                raise ValueError("a successful run requires passed validation")
            if self.reproduction_result.status is not ReproductionStatus.REPRODUCED:
                raise ValueError("a successful run requires reproduced issue evidence")
            if self.reproduction_result.test_run is None:
                raise ValueError("a successful run requires structured reproduction evidence")
            if self.validation_result.test_run is None:
                raise ValueError("a successful run requires structured validation evidence")
            reproduction_test = self.reproduction_result.test_run
            validation_test = self.validation_result.test_run
            if (
                reproduction_test.tests_run == 0
                or reproduction_test.failures + reproduction_test.errors == 0
                or not reproduction_test.failing_tests
                or reproduction_test.truncated
                or (reproduction_test.stdout_ref is None and reproduction_test.stderr_ref is None)
            ):
                raise ValueError("a successful run requires complete reproduction artifacts")
            if (
                validation_test.tests_run == 0
                or validation_test.truncated
                or (validation_test.stdout_ref is None and validation_test.stderr_ref is None)
            ):
                raise ValueError("a successful run requires complete validation artifacts")
            if (
                reproduction_test.command_index != validation_test.command_index
                or reproduction_test.command_sha256 != validation_test.command_sha256
            ):
                raise ValueError("a successful run requires matching test commands")
            if (
                self.reproduction_result.observation_call_id
                == self.validation_result.observation_call_id
            ):
                raise ValueError("a successful run requires distinct test observations")
            if not self.task_metrics.task_success:
                raise ValueError("a successful run requires task_success metrics")
            if self.error_category is not None:
                raise ValueError("a successful run cannot have an error_category")
            if self.verifier_error_code is not None:
                raise ValueError("a successful run cannot have a verifier_error_code")
        elif self.termination_reason is TerminationReason.SUCCESS:
            raise ValueError("a non-successful run cannot use success termination")
        return self
