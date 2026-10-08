"""Append-only evaluation runner with resumable, provenance-bound task slots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, Self, cast
from uuid import UUID, uuid4, uuid5

from pydantic import AwareDatetime, Field, JsonValue, ValidationError, model_validator
from pydantic_core import to_jsonable_python

from repopilot.config import RepoPilotSettings
from repopilot.embeddings import EmbeddingProvider, LocalEmbeddingCache
from repopilot.enums import (
    DatasetSplit,
    EvaluationAttemptStatus,
    EvaluationRunnerErrorCode,
    ModelProvider,
    RepairContextMode,
    RunStatus,
    TerminationReason,
)
from repopilot.memory import MemoryStoreBundle
from repopilot.providers import ProviderResponse
from repopilot.repair_workflow import (
    RepairRunArtifacts,
    RepairRunRequest,
    RepairSandboxFactory,
    RepairWorkflowError,
    run_repair_request,
)
from repopilot.schemas import (
    ArtifactRef,
    EvaluationTaskSpec,
    FinalResult,
    GitSha,
    Identifier,
    MemoryNamespace,
    ModelIdentifier,
    RelativePath,
    Sha256,
    StrictModel,
    TaskId,
    TaskSpec,
    UsageMetrics,
    VersionIdentifier,
)
from repopilot.storage import VectorStore

EVALUATION_RUN_SCHEMA_VERSION: Literal[1] = 1
EVALUATION_EVENT_SCHEMA_VERSION: Literal[1] = 1
EVALUATION_RUNNER_VERSION: Literal["eval-runner-v1"] = "eval-runner-v1"
EVALUATION_MANIFEST_FILENAME = "manifest.json"
EVALUATION_EVENTS_DIRECTORY = "events"
EVALUATION_EXECUTIONS_DIRECTORY = "executions"
_MAX_MANIFEST_BYTES = 1_000_000
_MAX_EVENT_BYTES = 1_000_000
_MAX_RESULT_ARTIFACT_BYTES = 32 * 1024 * 1024
_EVENT_NAME = re.compile(r"^(?P<sequence>[0-9]{8})\.json$")
_IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:~-]{0,439}@sha256:[0-9a-f]{64}$")
_SENSITIVE_CONFIGURATION_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "auth_token",
        "authorization",
        "credential",
        "credentials",
        "database_url",
        "password",
        "secret",
    }
)
_SENSITIVE_CONFIGURATION_SUFFIXES = (
    "api_key",
    "password",
    "secret",
    "access_token",
    "auth_token",
)

EVALUATION_RUNNER_ERROR_MESSAGES: Mapping[EvaluationRunnerErrorCode, str] = {
    EvaluationRunnerErrorCode.INVALID_MANIFEST: "evaluation run manifest is invalid",
    EvaluationRunnerErrorCode.UNSUPPORTED_VERSION: ("evaluation run schema version is unsupported"),
    EvaluationRunnerErrorCode.IDENTITY_MISMATCH: (
        "evaluation task or execution identity does not match"
    ),
    EvaluationRunnerErrorCode.CONFIGURATION_DRIFT: (
        "evaluation run provenance changed during resume"
    ),
    EvaluationRunnerErrorCode.DUPLICATE_ATTEMPT: (
        "evaluation attempt history contains a duplicate or invalid transition"
    ),
    EvaluationRunnerErrorCode.RESUME_LIMIT: "evaluation attempt resume limit was reached",
    EvaluationRunnerErrorCode.OUTPUT_CONFLICT: (
        "evaluation output contains an unexpected existing entry"
    ),
    EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY: (
        "evaluation result or trace integrity check failed"
    ),
    EvaluationRunnerErrorCode.EXECUTION_FAILURE: "evaluation task execution failed",
    EvaluationRunnerErrorCode.PATH_UNSAFE: "evaluation output path is unsafe",
    EvaluationRunnerErrorCode.CONCURRENT_WRITE: ("evaluation run was modified by another writer"),
    EvaluationRunnerErrorCode.INPUT_IO_ERROR: "evaluation input could not be read",
    EvaluationRunnerErrorCode.OUTPUT_IO_ERROR: "evaluation output could not be written",
}


class EvaluationRunnerError(RuntimeError):
    """Safe typed failure that never includes paths, task text, or artifact content."""

    def __init__(self, code: EvaluationRunnerErrorCode) -> None:
        super().__init__(EVALUATION_RUNNER_ERROR_MESSAGES[code])
        self.code = code


class EvaluationTaskExecutionError(RuntimeError):
    """Executor-owned technical failure with no model or repository text attached."""

    def __init__(self) -> None:
        super().__init__(
            EVALUATION_RUNNER_ERROR_MESSAGES[EvaluationRunnerErrorCode.EXECUTION_FAILURE]
        )


class EvaluationRunManifest(StrictModel):
    """Immutable identity and sanitized configuration for one experiment run."""

    schema_version: Literal[1] = EVALUATION_RUN_SCHEMA_VERSION
    runner_version: Literal["eval-runner-v1"] = EVALUATION_RUNNER_VERSION
    experiment_id: Identifier
    run_id: UUID
    created_at: AwareDatetime
    source_commit: GitSha
    dependency_lock_sha256: Sha256
    dataset_manifest_sha256: Sha256
    task_set_sha256: Sha256
    configuration: dict[str, JsonValue]
    configuration_sha256: Sha256
    split: DatasetSplit
    arm_id: Identifier
    repetitions: int = Field(default=1, ge=1, le=10, strict=True)
    max_execution_attempts: int = Field(default=3, ge=1, le=10, strict=True)
    task_ids: tuple[TaskId, ...] = Field(min_length=1, max_length=100)
    memory_namespace: MemoryNamespace | None = None
    model_provider: ModelProvider
    model: ModelIdentifier
    prompt_versions: tuple[VersionIdentifier, ...] = Field(min_length=1, max_length=32)
    sandbox_image_reference: str = Field(min_length=1, max_length=512)
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.task_ids != tuple(sorted(self.task_ids)) or len(set(self.task_ids)) != len(
            self.task_ids
        ):
            raise ValueError("evaluation task IDs must be sorted and unique")
        if len(set(self.prompt_versions)) != len(self.prompt_versions):
            raise ValueError("evaluation prompt versions must be unique")
        if _IMAGE_PATTERN.fullmatch(self.sandbox_image_reference) is None:
            raise ValueError("evaluation Sandbox image must use an immutable digest")
        if self.memory_namespace is not None and self.memory_namespace.split is not self.split:
            raise ValueError("evaluation Memory namespace split must match the task split")
        if len(_canonical_json(self.configuration).encode("utf-8")) > 512_000:
            raise ValueError("evaluation configuration snapshot is too large")
        _validate_redacted_configuration(self.configuration)
        if _canonical_sha256(self.configuration) != self.configuration_sha256:
            raise ValueError("evaluation configuration digest does not match")
        payload = self.model_dump(mode="python", exclude={"content_sha256"})
        if _canonical_sha256(payload) != self.content_sha256:
            raise ValueError("evaluation run manifest digest does not match")
        return self


class EvaluationExecutionEvidence(StrictModel):
    """Terminal executor handoff whose referenced bytes are rechecked by the runner."""

    task_id: TaskId
    run_id: UUID
    status: RunStatus
    termination_reason: TerminationReason
    task_success: bool
    usage: UsageMetrics
    result_ref: ArtifactRef
    trace_ref: ArtifactRef
    artifact_manifest_ref: ArtifactRef

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        expected_paths = ("result.json", "trace.json", "artifacts.json")
        actual_paths = (
            self.result_ref.relative_path,
            self.trace_ref.relative_path,
            self.artifact_manifest_ref.relative_path,
        )
        if actual_paths != expected_paths:
            raise ValueError("evaluation evidence must use fixed result artifact names")
        if any(
            item.media_type != "application/json"
            for item in (
                self.result_ref,
                self.trace_ref,
                self.artifact_manifest_ref,
            )
        ):
            raise ValueError("evaluation evidence artifacts must be JSON")
        if self.status is RunStatus.SUCCESS:
            if self.termination_reason is not TerminationReason.SUCCESS or not self.task_success:
                raise ValueError("successful evaluation evidence must be verifier-successful")
        elif self.task_success:
            raise ValueError("non-success evaluation evidence cannot claim task success")
        return self


class EvaluationAttemptEvent(StrictModel):
    """One no-overwrite lifecycle transition in the global run event sequence."""

    schema_version: Literal[1] = EVALUATION_EVENT_SCHEMA_VERSION
    sequence: int = Field(ge=0, strict=True)
    attempt_id: Identifier
    task_id: TaskId
    repetition: int = Field(ge=0, le=9, strict=True)
    execution_index: int = Field(ge=0, le=9, strict=True)
    execution_run_id: UUID
    thread_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    output_path: RelativePath
    status: EvaluationAttemptStatus
    recorded_at: AwareDatetime
    evidence: EvaluationExecutionEvidence | None = None
    failure_code: EvaluationRunnerErrorCode | None = None
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if self.status is EvaluationAttemptStatus.COMPLETED:
            if self.evidence is None or self.failure_code is not None:
                raise ValueError("completed evaluation event requires only evidence")
        elif self.status is EvaluationAttemptStatus.FAILED:
            if self.evidence is not None or self.failure_code is None:
                raise ValueError("failed evaluation event requires only a failure code")
        elif self.evidence is not None or self.failure_code is not None:
            raise ValueError("nonterminal evaluation events cannot carry outcomes")
        payload = self.model_dump(mode="python", exclude={"content_sha256"})
        if _canonical_sha256(payload) != self.content_sha256:
            raise ValueError("evaluation event digest does not match")
        return self


class EvaluationAttemptSummary(StrictModel):
    """Latest reconstructed state for one stable task/repetition slot."""

    attempt_id: Identifier
    task_id: TaskId
    repetition: int = Field(ge=0, le=9, strict=True)
    execution_count: int = Field(ge=0, le=10, strict=True)
    status: EvaluationAttemptStatus | None = None
    evidence: EvaluationExecutionEvidence | None = None
    failure_code: EvaluationRunnerErrorCode | None = None

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.status is None:
            if (
                self.execution_count != 0
                or self.evidence is not None
                or self.failure_code is not None
            ):
                raise ValueError("pending evaluation summary cannot contain execution state")
        elif self.status is EvaluationAttemptStatus.COMPLETED:
            if self.evidence is None or self.failure_code is not None:
                raise ValueError("completed evaluation summary requires evidence")
        elif self.status is EvaluationAttemptStatus.FAILED:
            if self.evidence is not None or self.failure_code is None:
                raise ValueError("failed evaluation summary requires a failure code")
        elif self.evidence is not None or self.failure_code is not None:
            raise ValueError("nonterminal evaluation summary cannot contain an outcome")
        return self


class EvaluationRunReport(StrictModel):
    """Deterministic report rebuilt only from the manifest and append-only events."""

    run_id: UUID
    total_attempts: int = Field(ge=1, le=1_000, strict=True)
    completed_attempts: int = Field(ge=0, strict=True)
    failed_attempts: int = Field(ge=0, strict=True)
    pending_attempts: int = Field(ge=0, strict=True)
    attempts: tuple[EvaluationAttemptSummary, ...] = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.total_attempts != len(self.attempts):
            raise ValueError("evaluation report total does not match attempts")
        if self.completed_attempts != sum(
            item.status is EvaluationAttemptStatus.COMPLETED for item in self.attempts
        ):
            raise ValueError("evaluation report completed count does not match")
        if self.failed_attempts != sum(
            item.status is EvaluationAttemptStatus.FAILED for item in self.attempts
        ):
            raise ValueError("evaluation report failed count does not match")
        if self.pending_attempts != self.total_attempts - (
            self.completed_attempts + self.failed_attempts
        ):
            raise ValueError("evaluation report pending count does not match")
        return self


@dataclass(frozen=True)
class EvaluationExecutionRequest:
    """Agent-only executor input; evaluator gold is intentionally absent."""

    task: TaskSpec
    attempt_id: str
    execution_index: int
    run_id: UUID
    thread_id: str
    started_at: datetime
    output_dir: Path


class EvaluationTaskExecutor(Protocol):
    """Replaceable adapter from one frozen task slot to repair artifacts."""

    def execute(self, request: EvaluationExecutionRequest) -> EvaluationExecutionEvidence:
        """Execute one task without receiving expected patch or gold files."""


Clock = Callable[[], datetime]
ProviderResponsesFactory = Callable[[TaskSpec, UUID], tuple[ProviderResponse, ...]]
AttemptStartGuard = Callable[[EvaluationRunReport], None]


@dataclass(frozen=True)
class RepairWorkflowEvaluationExecutor:
    """Production adapter from one Agent-only eval request to repair workflow evidence."""

    settings: RepoPilotSettings
    sandbox_image_reference: str
    context_mode: RepairContextMode = RepairContextMode.BM25
    evaluation_configuration_sha256: Sha256 | None = None
    provider_responses_factory: ProviderResponsesFactory | None = None
    sandbox_factory: RepairSandboxFactory | None = None
    memory_stores: MemoryStoreBundle | None = None
    embedding_provider: EmbeddingProvider | None = None
    embedding_cache: LocalEmbeddingCache | None = None
    vector_store: VectorStore | None = None

    def execute(self, request: EvaluationExecutionRequest) -> EvaluationExecutionEvidence:
        try:
            provider_responses = (
                ()
                if self.provider_responses_factory is None
                else self.provider_responses_factory(request.task, request.run_id)
            )
            artifacts = run_repair_request(
                RepairRunRequest(
                    thread_id=request.thread_id,
                    run_id=request.run_id,
                    started_at=request.started_at,
                    task=request.task,
                    sandbox_image_reference=self.sandbox_image_reference,
                    context_mode=self.context_mode,
                    evaluation_configuration_sha256=self.evaluation_configuration_sha256,
                    provider_responses=provider_responses,
                ),
                output_dir=request.output_dir,
                settings=self.settings,
                sandbox_factory=self.sandbox_factory,
                memory_stores=self.memory_stores,
                embedding_provider=self.embedding_provider,
                embedding_cache=self.embedding_cache,
                vector_store=self.vector_store,
            )
            return evaluation_evidence_from_repair_artifacts(artifacts)
        except (RepairWorkflowError, ValidationError, ValueError):
            raise EvaluationTaskExecutionError from None


def evaluation_evidence_from_repair_artifacts(
    artifacts: RepairRunArtifacts,
) -> EvaluationExecutionEvidence:
    """Convert persisted repair files into a runner handoff without trusting metadata alone."""

    return EvaluationExecutionEvidence(
        task_id=artifacts.result.task_id,
        run_id=artifacts.result.run_id,
        status=artifacts.result.status,
        termination_reason=artifacts.result.termination_reason,
        task_success=artifacts.result.task_metrics.task_success,
        usage=artifacts.result.usage,
        result_ref=_json_artifact_ref(artifacts.result_path),
        trace_ref=_json_artifact_ref(artifacts.trace_path),
        artifact_manifest_ref=_json_artifact_ref(artifacts.artifact_manifest_path),
    )


def build_evaluation_run_manifest(
    *,
    experiment_id: str,
    run_id: UUID,
    created_at: datetime,
    source_commit: str,
    dependency_lock_sha256: str,
    dataset_manifest_sha256: str,
    configuration: Mapping[str, object],
    split: DatasetSplit,
    arm_id: str,
    tasks: tuple[EvaluationTaskSpec, ...],
    memory_namespace: MemoryNamespace | None,
    model_provider: ModelProvider,
    model: str,
    prompt_versions: tuple[str, ...],
    sandbox_image_reference: str,
    repetitions: int = 1,
    max_execution_attempts: int = 3,
) -> EvaluationRunManifest:
    """Build one content-addressed run declaration from sanitized trusted inputs."""

    validated_tasks = _validated_tasks(tasks)
    if any(item.split is not split for item in validated_tasks):
        raise EvaluationRunnerError(EvaluationRunnerErrorCode.IDENTITY_MISMATCH)
    configuration_json = cast(dict[str, JsonValue], to_jsonable_python(dict(configuration)))
    payload: dict[str, object] = {
        "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
        "runner_version": EVALUATION_RUNNER_VERSION,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "created_at": created_at,
        "source_commit": source_commit,
        "dependency_lock_sha256": dependency_lock_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "task_set_sha256": evaluation_task_set_sha256(validated_tasks),
        "configuration": configuration_json,
        "configuration_sha256": _canonical_sha256(configuration_json),
        "split": split,
        "arm_id": arm_id,
        "repetitions": repetitions,
        "max_execution_attempts": max_execution_attempts,
        "task_ids": tuple(sorted(item.task.task_id for item in validated_tasks)),
        "memory_namespace": memory_namespace,
        "model_provider": model_provider,
        "model": model,
        "prompt_versions": prompt_versions,
        "sandbox_image_reference": sandbox_image_reference,
    }
    return EvaluationRunManifest.model_validate(
        {
            **payload,
            "content_sha256": _canonical_sha256(payload),
        }
    )


def evaluation_task_set_sha256(tasks: tuple[EvaluationTaskSpec, ...]) -> str:
    """Hash evaluator-owned task semantics without persisting a local repository path."""

    projections: list[dict[str, object]] = []
    for evaluation_task in sorted(tasks, key=lambda item: item.task.task_id):
        payload = evaluation_task.model_dump(mode="python")
        task = cast(dict[str, object], payload["task"])
        if task["repository_path"] is not None:
            task["repository_path"] = "<local-repository>"
        projections.append(payload)
    return _canonical_sha256(projections)


def evaluation_attempt_id(run_id: UUID, task_id: str, repetition: int) -> str:
    """Return the stable slot identity shared by retries and resume executions."""

    encoded = f"{run_id}:{task_id}:{repetition}".encode()
    return f"attempt_{hashlib.sha256(encoded).hexdigest()}"


def run_evaluation(
    *,
    manifest: EvaluationRunManifest,
    tasks: tuple[EvaluationTaskSpec, ...],
    output_dir: Path,
    executor: EvaluationTaskExecutor,
    clock: Clock | None = None,
    before_attempt: AttemptStartGuard | None = None,
) -> EvaluationRunReport:
    """Execute all unfinished slots and reconstruct a deterministic terminal report."""

    validated_manifest = _validated_manifest(manifest)
    validated_tasks = _validated_tasks(tasks)
    _validate_task_set(validated_manifest, validated_tasks)
    store = FileEvaluationRunStore(output_dir)
    store.open(validated_manifest)
    active_clock = clock or (lambda: datetime.now(UTC))
    events = store.read_events(validated_manifest)
    store.verify_completed_history(validated_manifest, events)
    by_task = {item.task.task_id: item for item in validated_tasks}

    for task_id in validated_manifest.task_ids:
        for repetition in range(validated_manifest.repetitions):
            attempt_id = evaluation_attempt_id(
                validated_manifest.run_id,
                task_id,
                repetition,
            )
            history = tuple(event for event in events if event.attempt_id == attempt_id)
            if history and history[-1].status in {
                EvaluationAttemptStatus.COMPLETED,
                EvaluationAttemptStatus.FAILED,
            }:
                continue
            if history and history[-1].status is EvaluationAttemptStatus.STARTED:
                interrupted = _build_event(
                    sequence=len(events),
                    manifest=validated_manifest,
                    task_id=task_id,
                    repetition=repetition,
                    execution_index=history[-1].execution_index,
                    status=EvaluationAttemptStatus.INTERRUPTED,
                    recorded_at=_aware_now(active_clock),
                )
                store.append_event(interrupted)
                events = (*events, interrupted)
                history = (*history, interrupted)
            execution_count = sum(
                event.status is EvaluationAttemptStatus.STARTED for event in history
            )
            if execution_count >= validated_manifest.max_execution_attempts:
                raise EvaluationRunnerError(EvaluationRunnerErrorCode.RESUME_LIMIT)
            if before_attempt is not None:
                before_attempt(build_evaluation_run_report(validated_manifest, events))
            execution_index = execution_count
            output_path = _execution_output_path(attempt_id, execution_index)
            store.require_execution_absent(output_path)
            started_at = _aware_now(active_clock)
            started = _build_event(
                sequence=len(events),
                manifest=validated_manifest,
                task_id=task_id,
                repetition=repetition,
                execution_index=execution_index,
                status=EvaluationAttemptStatus.STARTED,
                recorded_at=started_at,
            )
            store.append_event(started)
            events = (*events, started)
            execution_root = store.create_execution_directory(output_path)
            request = EvaluationExecutionRequest(
                task=by_task[task_id].task,
                attempt_id=attempt_id,
                execution_index=execution_index,
                run_id=started.execution_run_id,
                thread_id=started.thread_id,
                started_at=started_at,
                output_dir=execution_root,
            )
            try:
                evidence = EvaluationExecutionEvidence.model_validate(
                    executor.execute(request).model_dump(mode="python")
                )
                store.verify_execution_evidence(
                    manifest=validated_manifest,
                    event=started,
                    execution_root=execution_root,
                    evidence=evidence,
                )
            except EvaluationTaskExecutionError:
                failed = _build_event(
                    sequence=len(events),
                    manifest=validated_manifest,
                    task_id=task_id,
                    repetition=repetition,
                    execution_index=execution_index,
                    status=EvaluationAttemptStatus.FAILED,
                    recorded_at=_aware_now(active_clock),
                    failure_code=EvaluationRunnerErrorCode.EXECUTION_FAILURE,
                )
                store.append_event(failed)
                events = (*events, failed)
                continue
            except EvaluationRunnerError as error:
                if error.code is not EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY:
                    raise
                failed = _build_event(
                    sequence=len(events),
                    manifest=validated_manifest,
                    task_id=task_id,
                    repetition=repetition,
                    execution_index=execution_index,
                    status=EvaluationAttemptStatus.FAILED,
                    recorded_at=_aware_now(active_clock),
                    failure_code=error.code,
                )
                store.append_event(failed)
                events = (*events, failed)
                continue
            except Exception:
                failed = _build_event(
                    sequence=len(events),
                    manifest=validated_manifest,
                    task_id=task_id,
                    repetition=repetition,
                    execution_index=execution_index,
                    status=EvaluationAttemptStatus.FAILED,
                    recorded_at=_aware_now(active_clock),
                    failure_code=EvaluationRunnerErrorCode.EXECUTION_FAILURE,
                )
                store.append_event(failed)
                events = (*events, failed)
                continue
            completed = _build_event(
                sequence=len(events),
                manifest=validated_manifest,
                task_id=task_id,
                repetition=repetition,
                execution_index=execution_index,
                status=EvaluationAttemptStatus.COMPLETED,
                recorded_at=_aware_now(active_clock),
                evidence=evidence,
            )
            store.append_event(completed)
            events = (*events, completed)

    store.validate_event_history(validated_manifest, events)
    return build_evaluation_run_report(validated_manifest, events)


def load_evaluation_run_manifest(output_dir: Path) -> EvaluationRunManifest:
    """Load a persisted run identity for an explicit process-restart resume."""

    return FileEvaluationRunStore(output_dir).load_manifest()


def inspect_evaluation_run(output_dir: Path) -> EvaluationRunReport:
    """Verify retained history and rebuild progress without executing a task."""

    store = FileEvaluationRunStore(output_dir)
    manifest = store.load_manifest()
    events = store.read_events(manifest)
    store.verify_completed_history(manifest, events)
    return build_evaluation_run_report(manifest, events)


def build_evaluation_run_report(
    manifest: EvaluationRunManifest,
    events: tuple[EvaluationAttemptEvent, ...],
) -> EvaluationRunReport:
    """Rebuild run progress without trusting a mutable summary file."""

    summaries: list[EvaluationAttemptSummary] = []
    for task_id in manifest.task_ids:
        for repetition in range(manifest.repetitions):
            attempt_id = evaluation_attempt_id(manifest.run_id, task_id, repetition)
            history = tuple(event for event in events if event.attempt_id == attempt_id)
            latest = history[-1] if history else None
            summaries.append(
                EvaluationAttemptSummary(
                    attempt_id=attempt_id,
                    task_id=task_id,
                    repetition=repetition,
                    execution_count=sum(
                        event.status is EvaluationAttemptStatus.STARTED for event in history
                    ),
                    status=None if latest is None else latest.status,
                    evidence=None if latest is None else latest.evidence,
                    failure_code=None if latest is None else latest.failure_code,
                )
            )
    attempts = tuple(summaries)
    return EvaluationRunReport(
        run_id=manifest.run_id,
        total_attempts=len(attempts),
        completed_attempts=sum(
            item.status is EvaluationAttemptStatus.COMPLETED for item in attempts
        ),
        failed_attempts=sum(item.status is EvaluationAttemptStatus.FAILED for item in attempts),
        pending_attempts=sum(
            item.status
            not in {
                EvaluationAttemptStatus.COMPLETED,
                EvaluationAttemptStatus.FAILED,
            }
            for item in attempts
        ),
        attempts=attempts,
    )


class FileEvaluationRunStore:
    """Private no-overwrite manifest/event store rooted at one absolute run directory."""

    def __init__(self, root: Path) -> None:
        if not root.is_absolute():
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.PATH_UNSAFE)
        self._root = root
        self._events = root / EVALUATION_EVENTS_DIRECTORY
        self._executions = root / EVALUATION_EXECUTIONS_DIRECTORY

    def open(self, manifest: EvaluationRunManifest) -> None:
        """Create or reopen exactly one immutable run declaration."""

        root = self._safe_directory(self._root, create=True)
        manifest_path = root / EVALUATION_MANIFEST_FILENAME
        entries = self._entries(root)
        if manifest_path.exists():
            persisted = self._read_manifest(manifest_path)
            if persisted != manifest:
                raise EvaluationRunnerError(EvaluationRunnerErrorCode.CONFIGURATION_DRIFT)
            unexpected = {
                entry.name
                for entry in entries
                if entry.name
                not in {
                    EVALUATION_MANIFEST_FILENAME,
                    EVALUATION_EVENTS_DIRECTORY,
                    EVALUATION_EXECUTIONS_DIRECTORY,
                }
            }
            if unexpected:
                raise EvaluationRunnerError(EvaluationRunnerErrorCode.OUTPUT_CONFLICT)
        else:
            if entries:
                raise EvaluationRunnerError(EvaluationRunnerErrorCode.OUTPUT_CONFLICT)
            serialized = _model_bytes(manifest)
            if len(serialized) > _MAX_MANIFEST_BYTES:
                raise EvaluationRunnerError(EvaluationRunnerErrorCode.INVALID_MANIFEST)
            self._atomic_create(manifest_path, serialized)
        self._safe_directory(self._events, create=True)
        self._safe_directory(self._executions, create=True)

    def load_manifest(self) -> EvaluationRunManifest:
        """Read an existing declaration without creating or repairing its directory."""

        self._safe_directory(self._root, create=False)
        return self._read_manifest(self._root / EVALUATION_MANIFEST_FILENAME)

    def read_events(
        self,
        manifest: EvaluationRunManifest,
    ) -> tuple[EvaluationAttemptEvent, ...]:
        """Read, integrity-check, and transition-check the full event history."""

        events: list[EvaluationAttemptEvent] = []
        for expected_sequence, path in enumerate(self._entries(self._events)):
            match = _EVENT_NAME.fullmatch(path.name)
            if match is None or int(match.group("sequence")) != expected_sequence:
                raise EvaluationRunnerError(EvaluationRunnerErrorCode.DUPLICATE_ATTEMPT)
            event = self._read_event(path)
            if event.sequence != expected_sequence:
                raise EvaluationRunnerError(EvaluationRunnerErrorCode.DUPLICATE_ATTEMPT)
            events.append(event)
        result = tuple(events)
        self.validate_event_history(manifest, result)
        return result

    def append_event(self, event: EvaluationAttemptEvent) -> None:
        """Publish exactly the next event or reject a concurrent writer."""

        current = self._entries(self._events)
        if event.sequence != len(current):
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.CONCURRENT_WRITE)
        destination = self._events / f"{event.sequence:08d}.json"
        self._atomic_create(destination, _model_bytes(event))

    def validate_event_history(
        self,
        manifest: EvaluationRunManifest,
        events: tuple[EvaluationAttemptEvent, ...],
    ) -> None:
        """Reject impossible identities, retries, or terminal transitions."""

        valid_attempts = {
            evaluation_attempt_id(manifest.run_id, task_id, repetition): (
                task_id,
                repetition,
            )
            for task_id in manifest.task_ids
            for repetition in range(manifest.repetitions)
        }
        if tuple(event.sequence for event in events) != tuple(range(len(events))):
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.DUPLICATE_ATTEMPT)
        for attempt_id, (task_id, repetition) in valid_attempts.items():
            history = tuple(event for event in events if event.attempt_id == attempt_id)
            starts = 0
            awaiting_terminal = False
            terminal = False
            for event in history:
                if (
                    event.task_id != task_id
                    or event.repetition != repetition
                    or event.execution_index != starts - int(awaiting_terminal)
                    or event.execution_run_id
                    != _execution_run_id(manifest.run_id, attempt_id, event.execution_index)
                    or event.thread_id != _execution_thread_id(attempt_id, event.execution_index)
                    or event.output_path
                    != _execution_output_path(attempt_id, event.execution_index)
                ):
                    raise EvaluationRunnerError(EvaluationRunnerErrorCode.IDENTITY_MISMATCH)
                if terminal:
                    raise EvaluationRunnerError(EvaluationRunnerErrorCode.DUPLICATE_ATTEMPT)
                if event.status is EvaluationAttemptStatus.STARTED:
                    if awaiting_terminal or event.execution_index != starts:
                        raise EvaluationRunnerError(EvaluationRunnerErrorCode.DUPLICATE_ATTEMPT)
                    starts += 1
                    awaiting_terminal = True
                elif event.status in {
                    EvaluationAttemptStatus.INTERRUPTED,
                    EvaluationAttemptStatus.COMPLETED,
                    EvaluationAttemptStatus.FAILED,
                }:
                    if not awaiting_terminal or event.execution_index != starts - 1:
                        raise EvaluationRunnerError(EvaluationRunnerErrorCode.DUPLICATE_ATTEMPT)
                    awaiting_terminal = False
                    terminal = event.status in {
                        EvaluationAttemptStatus.COMPLETED,
                        EvaluationAttemptStatus.FAILED,
                    }
            if starts > manifest.max_execution_attempts:
                raise EvaluationRunnerError(EvaluationRunnerErrorCode.RESUME_LIMIT)
        if any(event.attempt_id not in valid_attempts for event in events):
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.IDENTITY_MISMATCH)

    def require_execution_absent(self, output_path: str) -> None:
        path = self._root / output_path
        try:
            exists = path.exists() or path.is_symlink()
        except OSError:
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.OUTPUT_IO_ERROR) from None
        if exists:
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.OUTPUT_CONFLICT)

    def create_execution_directory(self, output_path: str) -> Path:
        path = self._root / output_path
        attempt_root = path.parent
        self._safe_directory(attempt_root, create=True)
        return self._safe_directory(path, create=True)

    def verify_execution_evidence(
        self,
        *,
        manifest: EvaluationRunManifest,
        event: EvaluationAttemptEvent,
        execution_root: Path,
        evidence: EvaluationExecutionEvidence,
    ) -> None:
        """Reopen exact output bytes and correlate result, Trace, and provenance."""

        if evidence.task_id != event.task_id or evidence.run_id != event.execution_run_id:
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY)
        serialized: dict[str, bytes] = {}
        for reference in (
            evidence.result_ref,
            evidence.trace_ref,
            evidence.artifact_manifest_ref,
        ):
            content = self._read_artifact(execution_root, reference.relative_path)
            if (
                len(content) != reference.size_bytes
                or hashlib.sha256(content).hexdigest() != reference.sha256
            ):
                raise EvaluationRunnerError(EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY)
            serialized[reference.relative_path] = content
        try:
            result = FinalResult.model_validate_json(serialized["result.json"])
            trace = _json_mapping(serialized["trace.json"])
            artifact_manifest = _json_mapping(serialized["artifacts.json"])
        except (ValidationError, ValueError, UnicodeError):
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY) from None
        if (
            result.task_id != event.task_id
            or result.run_id != event.execution_run_id
            or result.status is not evidence.status
            or result.termination_reason is not evidence.termination_reason
            or result.task_metrics.task_success is not evidence.task_success
            or result.usage != evidence.usage
            or result.trace_path != "trace.json"
        ):
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY)
        expected_identity = {
            "task_id": event.task_id,
            "run_id": str(event.execution_run_id),
        }
        if any(trace.get(key) != value for key, value in expected_identity.items()):
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY)
        if any(artifact_manifest.get(key) != value for key, value in expected_identity.items()):
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY)
        if (
            artifact_manifest.get("provider_name") != self._expected_provider_name(manifest)
            or artifact_manifest.get("sandbox_image_reference") != manifest.sandbox_image_reference
            or (
                artifact_manifest.get("evaluation_configuration_sha256")
                or artifact_manifest.get("settings_sha256")
            )
            != manifest.configuration_sha256
            or artifact_manifest.get("prompt_versions") != list(manifest.prompt_versions)
            or artifact_manifest.get("result_path") != "result.json"
            or artifact_manifest.get("trace_path") != "trace.json"
        ):
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY)

    def recover_execution_evidence(
        self,
        *,
        manifest: EvaluationRunManifest,
        event: EvaluationAttemptEvent,
    ) -> EvaluationExecutionEvidence:
        """Revalidate retained bytes when an earlier verifier defect rejected them."""

        if self.load_manifest() != manifest:
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.CONFIGURATION_DRIFT)
        execution_root = self._existing_execution_directory(event.output_path)
        serialized = {
            name: self._read_artifact(execution_root, name)
            for name in ("result.json", "trace.json", "artifacts.json")
        }
        try:
            result = FinalResult.model_validate_json(serialized["result.json"])
        except (ValidationError, UnicodeError):
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY) from None

        def reference(name: str) -> ArtifactRef:
            content = serialized[name]
            return ArtifactRef(
                relative_path=name,
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                media_type="application/json",
            )

        evidence = EvaluationExecutionEvidence(
            task_id=result.task_id,
            run_id=result.run_id,
            status=result.status,
            termination_reason=result.termination_reason,
            task_success=result.task_metrics.task_success,
            usage=result.usage,
            result_ref=reference("result.json"),
            trace_ref=reference("trace.json"),
            artifact_manifest_ref=reference("artifacts.json"),
        )
        self.verify_execution_evidence(
            manifest=manifest,
            event=event,
            execution_root=execution_root,
            evidence=evidence,
        )
        return evidence

    @staticmethod
    def _expected_provider_name(manifest: EvaluationRunManifest) -> str:
        """Derive the exact trace-visible adapter identity from frozen provenance."""

        if manifest.model_provider is ModelProvider.OPENAI:
            if len(manifest.prompt_versions) != 1:
                raise EvaluationRunnerError(EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY)
            return f"openai-responses:{manifest.model}:{manifest.prompt_versions[0]}"
        return manifest.model

    def verify_completed_history(
        self,
        manifest: EvaluationRunManifest,
        events: tuple[EvaluationAttemptEvent, ...],
    ) -> None:
        """Recheck retained terminal bytes before treating a slot as deduplicated."""

        for event in events:
            if event.status is not EvaluationAttemptStatus.COMPLETED:
                continue
            if event.evidence is None:
                raise EvaluationRunnerError(EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY)
            execution_root = self._existing_execution_directory(event.output_path)
            self.verify_execution_evidence(
                manifest=manifest,
                event=event,
                execution_root=execution_root,
                evidence=event.evidence,
            )

    @staticmethod
    def _read_artifact(root: Path, relative_path: str) -> bytes:
        path = root / relative_path
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise EvaluationRunnerError(EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY)
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                content = stream.read(_MAX_RESULT_ARTIFACT_BYTES + 1)
        except EvaluationRunnerError:
            raise
        except OSError:
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY) from None
        if len(content) > _MAX_RESULT_ARTIFACT_BYTES:
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY)
        return content

    def _existing_execution_directory(self, output_path: str) -> Path:
        path = self._root / output_path
        return self._safe_directory(path, create=False)

    @staticmethod
    def _read_manifest(path: Path) -> EvaluationRunManifest:
        raw = _read_private_file(path, _MAX_MANIFEST_BYTES)
        version = _schema_version(raw)
        if version != EVALUATION_RUN_SCHEMA_VERSION:
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.UNSUPPORTED_VERSION)
        try:
            return EvaluationRunManifest.model_validate_json(raw)
        except (ValidationError, UnicodeError):
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.INVALID_MANIFEST) from None

    @staticmethod
    def _read_event(path: Path) -> EvaluationAttemptEvent:
        raw = _read_private_file(path, _MAX_EVENT_BYTES)
        version = _schema_version(raw)
        if version != EVALUATION_EVENT_SCHEMA_VERSION:
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.UNSUPPORTED_VERSION)
        try:
            return EvaluationAttemptEvent.model_validate_json(raw)
        except (ValidationError, UnicodeError):
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY) from None

    @staticmethod
    def _safe_directory(path: Path, *, create: bool) -> Path:
        try:
            if create:
                path.mkdir(mode=0o700, parents=False, exist_ok=True)
            metadata = path.lstat()
        except OSError:
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.OUTPUT_IO_ERROR) from None
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.PATH_UNSAFE)
        with suppress(OSError):
            path.chmod(0o700)
        return path

    @staticmethod
    def _entries(path: Path) -> tuple[Path, ...]:
        try:
            return tuple(sorted(path.iterdir(), key=lambda item: item.name))
        except OSError:
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.INPUT_IO_ERROR) from None

    @staticmethod
    def _atomic_create(destination: Path, serialized: bytes) -> None:
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination, follow_symlinks=False)
            temporary.unlink()
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except FileExistsError:
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.CONCURRENT_WRITE) from None
        except OSError:
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.OUTPUT_IO_ERROR) from None
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _build_event(
    *,
    sequence: int,
    manifest: EvaluationRunManifest,
    task_id: str,
    repetition: int,
    execution_index: int,
    status: EvaluationAttemptStatus,
    recorded_at: datetime,
    evidence: EvaluationExecutionEvidence | None = None,
    failure_code: EvaluationRunnerErrorCode | None = None,
) -> EvaluationAttemptEvent:
    attempt_id = evaluation_attempt_id(manifest.run_id, task_id, repetition)
    payload: dict[str, object] = {
        "schema_version": EVALUATION_EVENT_SCHEMA_VERSION,
        "sequence": sequence,
        "attempt_id": attempt_id,
        "task_id": task_id,
        "repetition": repetition,
        "execution_index": execution_index,
        "execution_run_id": _execution_run_id(
            manifest.run_id,
            attempt_id,
            execution_index,
        ),
        "thread_id": _execution_thread_id(attempt_id, execution_index),
        "output_path": _execution_output_path(attempt_id, execution_index),
        "status": status,
        "recorded_at": recorded_at,
        "evidence": evidence,
        "failure_code": failure_code,
    }
    return EvaluationAttemptEvent.model_validate(
        {
            **payload,
            "content_sha256": _canonical_sha256(payload),
        }
    )


def _execution_run_id(run_id: UUID, attempt_id: str, execution_index: int) -> UUID:
    return uuid5(run_id, f"{attempt_id}:execution:{execution_index}")


def _execution_thread_id(attempt_id: str, execution_index: int) -> str:
    return f"eval-{attempt_id[8:40]}-{execution_index:02d}"


def _execution_output_path(attempt_id: str, execution_index: int) -> str:
    return f"{EVALUATION_EXECUTIONS_DIRECTORY}/{attempt_id}/{execution_index:04d}"


def _validate_task_set(
    manifest: EvaluationRunManifest,
    tasks: tuple[EvaluationTaskSpec, ...],
) -> None:
    task_ids = tuple(sorted(item.task.task_id for item in tasks))
    if task_ids != manifest.task_ids or any(item.split is not manifest.split for item in tasks):
        raise EvaluationRunnerError(EvaluationRunnerErrorCode.IDENTITY_MISMATCH)
    if evaluation_task_set_sha256(tasks) != manifest.task_set_sha256:
        raise EvaluationRunnerError(EvaluationRunnerErrorCode.IDENTITY_MISMATCH)


def _validated_manifest(manifest: EvaluationRunManifest) -> EvaluationRunManifest:
    try:
        return EvaluationRunManifest.model_validate(manifest.model_dump(mode="python"))
    except (AttributeError, ValidationError):
        raise EvaluationRunnerError(EvaluationRunnerErrorCode.INVALID_MANIFEST) from None


def _validated_tasks(
    tasks: tuple[EvaluationTaskSpec, ...],
) -> tuple[EvaluationTaskSpec, ...]:
    try:
        validated = tuple(
            EvaluationTaskSpec.model_validate(item.model_dump(mode="python")) for item in tasks
        )
    except (AttributeError, ValidationError):
        raise EvaluationRunnerError(EvaluationRunnerErrorCode.IDENTITY_MISMATCH) from None
    task_ids = tuple(item.task.task_id for item in validated)
    if not validated or len(set(task_ids)) != len(task_ids):
        raise EvaluationRunnerError(EvaluationRunnerErrorCode.IDENTITY_MISMATCH)
    return validated


def _validate_redacted_configuration(configuration: Mapping[str, JsonValue]) -> None:
    def visit(value: JsonValue, key: str | None = None) -> None:
        if (
            key is not None
            and (
                key.casefold() in _SENSITIVE_CONFIGURATION_KEYS
                or key.casefold().endswith(_SENSITIVE_CONFIGURATION_SUFFIXES)
            )
            and value is not None
            and value != "<redacted>"
        ):
            raise ValueError("evaluation configuration contains an unredacted secret")
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and re.search(r"\bsk-[A-Za-z0-9_-]{8,}\b", value):
            raise ValueError("evaluation configuration contains secret-shaped text")

    visit(cast(JsonValue, dict(configuration)))


def _aware_now(clock: Clock) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise EvaluationRunnerError(EvaluationRunnerErrorCode.INVALID_MANIFEST)
    return value


def _json_mapping(serialized: bytes) -> dict[str, object]:
    value = json.loads(serialized)
    if not isinstance(value, dict):
        raise ValueError
    return cast(dict[str, object], value)


def _schema_version(serialized: bytes) -> object:
    try:
        return _json_mapping(serialized).get("schema_version")
    except (json.JSONDecodeError, UnicodeError, ValueError):
        raise EvaluationRunnerError(EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY) from None


def _read_private_file(path: Path, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & 0o077
        ):
            raise EvaluationRunnerError(EvaluationRunnerErrorCode.PATH_UNSAFE)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            serialized = stream.read(max_bytes + 1)
    except EvaluationRunnerError:
        raise
    except OSError:
        raise EvaluationRunnerError(EvaluationRunnerErrorCode.INPUT_IO_ERROR) from None
    if len(serialized) > max_bytes:
        raise EvaluationRunnerError(EvaluationRunnerErrorCode.INPUT_IO_ERROR)
    return serialized


def _model_bytes(model: StrictModel) -> bytes:
    return _canonical_json(model).encode("utf-8") + b"\n"


def _json_artifact_ref(path: Path) -> ArtifactRef:
    try:
        if path.is_symlink():
            raise OSError
        content = path.read_bytes()
    except OSError:
        raise EvaluationTaskExecutionError from None
    if len(content) > _MAX_RESULT_ARTIFACT_BYTES:
        raise EvaluationTaskExecutionError
    return ArtifactRef(
        relative_path=path.name,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        media_type="application/json",
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        cast(object, to_jsonable_python(value)),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "EVALUATION_EVENT_SCHEMA_VERSION",
    "EVALUATION_MANIFEST_FILENAME",
    "EVALUATION_RUNNER_VERSION",
    "EVALUATION_RUN_SCHEMA_VERSION",
    "EvaluationAttemptEvent",
    "EvaluationAttemptSummary",
    "EvaluationExecutionEvidence",
    "EvaluationExecutionRequest",
    "EvaluationRunManifest",
    "EvaluationRunReport",
    "EvaluationRunnerError",
    "EvaluationTaskExecutionError",
    "EvaluationTaskExecutor",
    "FileEvaluationRunStore",
    "RepairWorkflowEvaluationExecutor",
    "build_evaluation_run_manifest",
    "build_evaluation_run_report",
    "evaluation_attempt_id",
    "evaluation_evidence_from_repair_artifacts",
    "evaluation_task_set_sha256",
    "inspect_evaluation_run",
    "load_evaluation_run_manifest",
    "run_evaluation",
]
