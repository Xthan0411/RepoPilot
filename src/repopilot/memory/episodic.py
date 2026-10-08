"""Verifier-gated, immutable episodic memory for successful repair runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, Self, cast
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import to_jsonable_python

from repopilot.context import CONTEXT_POLICY_VERSION
from repopilot.enums import (
    RunStatus,
    TerminationReason,
    VerifiedEpisodeErrorCode,
    VerifiedEpisodeLookupStatus,
)
from repopilot.memory.working import context_policy_version_from_events
from repopilot.replay import replay_trace
from repopilot.schemas import (
    ArtifactRef,
    FinalResult,
    GitSha,
    Identifier,
    ModelIdentifier,
    RelativePath,
    ReproductionResult,
    Sha256,
    StrictModel,
    TaskId,
    ValidationResult,
    VerifiedEpisodeMemoryItem,
    VersionIdentifier,
)
from repopilot.tools import build_reproduction_result
from repopilot.trace import AgentTrace, TraceArtifactError
from repopilot.verifier import (
    RULE_BASED_VERIFIER_VERSION,
    RuleBasedVerifier,
    VerificationEvidence,
)

VERIFIED_EPISODIC_MEMORY_SCHEMA_VERSION: Literal[1] = 1
VERIFIED_REPAIR_PRODUCER_VERSION = "verified-repair-episode-v1"
SANDBOX_POLICY_VERSION_ID = "repopilot-sandbox-policy-v1"
DEFAULT_VERIFIED_EPISODE_TTL_DAYS = 30
MAX_VERIFIED_EPISODE_TTL_DAYS = 365
DEFAULT_MAX_VERIFIED_EPISODES_PER_REPOSITORY = 256
DEFAULT_MAX_VERIFIED_EPISODE_BYTES = 512 * 1024
MAX_VERIFIED_EPISODE_BYTES = 2 * 1024 * 1024
_MAX_ARTIFACTS = 256
_MAX_SOURCE_ARTIFACT_BYTES = 100_000_000
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORD_NAME = re.compile(r"^(?P<episode_id>[A-Za-z][A-Za-z0-9_.-]{0,127})\.json$")

VERIFIED_EPISODE_ERROR_MESSAGES: Mapping[VerifiedEpisodeErrorCode, str] = {
    VerifiedEpisodeErrorCode.INVALID_WRITE: "verified episode write request is invalid",
    VerifiedEpisodeErrorCode.VERIFIER_REJECTED: (
        "repair outcome is not eligible for verified episodic memory"
    ),
    VerifiedEpisodeErrorCode.TRACE_INVALID: "verified episode source Trace is invalid",
    VerifiedEpisodeErrorCode.RESULT_MISMATCH: (
        "FinalResult does not match replayed verifier evidence"
    ),
    VerifiedEpisodeErrorCode.EVIDENCE_INCOMPLETE: (
        "verified episode is missing required repair evidence"
    ),
    VerifiedEpisodeErrorCode.ARTIFACT_MISMATCH: (
        "verified episode evidence does not match retained artifacts"
    ),
    VerifiedEpisodeErrorCode.NAMESPACE_MISMATCH: (
        "verified episode namespace does not match its source run"
    ),
    VerifiedEpisodeErrorCode.EXPIRED: "verified episode is already expired",
    VerifiedEpisodeErrorCode.INTEGRITY_MISMATCH: ("verified episode integrity check failed"),
    VerifiedEpisodeErrorCode.DUPLICATE_CONFLICT: ("verified episode ID has conflicting content"),
    VerifiedEpisodeErrorCode.CAPACITY_EXCEEDED: ("verified episodic memory capacity was exceeded"),
    VerifiedEpisodeErrorCode.PATH_UNSAFE: ("verified episodic memory storage path is unsafe"),
    VerifiedEpisodeErrorCode.UNSUPPORTED_VERSION: (
        "verified episodic memory schema version is unsupported"
    ),
    VerifiedEpisodeErrorCode.IO_ERROR: ("verified episodic memory could not be read or written"),
}


class VerifiedEpisodeError(RuntimeError):
    """Safe failure that never echoes model text, repository content, or host paths."""

    def __init__(self, code: VerifiedEpisodeErrorCode) -> None:
        super().__init__(VERIFIED_EPISODE_ERROR_MESSAGES[code])
        self.code = code


class VerifiedEpisodeExecutionEvidence(StrictModel):
    """Trusted run metadata and retained artifacts supplied by the workflow."""

    sandbox_image_reference: str = Field(min_length=1, max_length=512, repr=False)
    sandbox_policy_version: VersionIdentifier = SANDBOX_POLICY_VERSION_ID
    settings_sha256: Sha256
    prompt_versions: tuple[VersionIdentifier, ...] = Field(default=(), max_length=32)
    context_policy_versions: tuple[VersionIdentifier, ...] = Field(
        default=(CONTEXT_POLICY_VERSION,),
        min_length=1,
        max_length=8,
    )
    artifacts: tuple[ArtifactRef, ...] = Field(min_length=1, max_length=_MAX_ARTIFACTS)

    @field_validator("prompt_versions", "context_policy_versions")
    @classmethod
    def require_unique_versions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("execution versions must be unique")
        return value

    @model_validator(mode="after")
    def validate_artifact_index(self) -> Self:
        paths = tuple(artifact.relative_path for artifact in self.artifacts)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("artifacts must be unique and sorted by relative path")
        if any(artifact.size_bytes > _MAX_SOURCE_ARTIFACT_BYTES for artifact in self.artifacts):
            raise ValueError("source artifact exceeds the supported byte limit")
        return self


class VerifiedEpisodeScope(StrictModel):
    """Exact applicability boundary; P6-S5 performs no fuzzy cross-version reuse."""

    repository_id: Identifier
    repository_commit: GitSha
    repository_content_sha256: Sha256
    files_changed: tuple[RelativePath, ...] = Field(min_length=1, max_length=128)
    test_command_sha256: Sha256

    @field_validator("files_changed")
    @classmethod
    def require_sorted_unique_files(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(value)) or len(set(value)) != len(value):
            raise ValueError("episode scope files must be unique and sorted")
        return value


class VerifiedEpisodeEvidence(StrictModel):
    """Content-addressed evidence that authorized the verified namespace write."""

    source_result_sha256: Sha256
    source_trace_sha256: Sha256
    patch_artifact: ArtifactRef
    reproduction_result: ReproductionResult
    validation_result: ValidationResult
    artifacts: tuple[ArtifactRef, ...] = Field(min_length=1, max_length=_MAX_ARTIFACTS)
    artifacts_sha256: Sha256

    @model_validator(mode="after")
    def validate_evidence_artifacts(self) -> Self:
        paths = tuple(artifact.relative_path for artifact in self.artifacts)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("episode artifacts must be unique and sorted")
        if _canonical_sha256(self.artifacts) != self.artifacts_sha256:
            raise ValueError("episode artifact index digest does not match")
        required = _required_artifacts(
            self.patch_artifact,
            self.reproduction_result,
            self.validation_result,
        )
        if any(artifact not in self.artifacts for artifact in required):
            raise ValueError("episode evidence artifact is absent from the artifact index")
        return self


class VerifiedEpisodeProvenance(StrictModel):
    """Versions needed to reproduce and audit the admission decision."""

    producer_version: VersionIdentifier = VERIFIED_REPAIR_PRODUCER_VERSION
    verifier_version: VersionIdentifier = RULE_BASED_VERIFIER_VERSION
    model_provider: str = Field(min_length=1, max_length=256)
    model_versions: tuple[ModelIdentifier, ...] = Field(default=(), max_length=32)
    prompt_versions: tuple[VersionIdentifier, ...] = Field(default=(), max_length=32)
    context_policy_versions: tuple[VersionIdentifier, ...] = Field(
        min_length=1,
        max_length=8,
    )
    sandbox_policy_version: VersionIdentifier
    sandbox_image_reference: str = Field(min_length=1, max_length=512, repr=False)
    settings_sha256: Sha256

    @field_validator(
        "model_versions",
        "prompt_versions",
        "context_policy_versions",
    )
    @classmethod
    def require_unique_versions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("provenance versions must be unique")
        return value


class VerifiedEpisodicRecord(StrictModel):
    """Immutable successful-repair experience admitted by trusted application code."""

    schema_version: Literal[1] = VERIFIED_EPISODIC_MEMORY_SCHEMA_VERSION
    episode_id: Identifier
    source_task_id: TaskId
    source_run_id: UUID
    problem_statement: str = Field(min_length=1, max_length=500)
    issue_fingerprint: Sha256
    repair_summary: str = Field(min_length=1, max_length=500)
    scope: VerifiedEpisodeScope
    evidence: VerifiedEpisodeEvidence
    provenance: VerifiedEpisodeProvenance
    created_at: AwareDatetime
    expires_at: AwareDatetime
    record_sha256: Sha256

    @model_validator(mode="after")
    def validate_identity_and_lifetime(self) -> Self:
        if self.episode_id != verified_episode_id(
            self.scope.repository_id,
            self.source_task_id,
            self.source_run_id,
        ):
            raise ValueError("episode ID does not match source namespace")
        if self.expires_at <= self.created_at:
            raise ValueError("episode expiry must follow creation")
        if self.expires_at - self.created_at > timedelta(days=MAX_VERIFIED_EPISODE_TTL_DAYS):
            raise ValueError("episode TTL exceeds the supported maximum")
        reproduction_test = self.evidence.reproduction_result.test_run
        validation_test = self.evidence.validation_result.test_run
        if reproduction_test is None or validation_test is None:
            raise ValueError("verified episode requires two structured test runs")
        if (
            reproduction_test.command_sha256 != self.scope.test_command_sha256
            or validation_test.command_sha256 != self.scope.test_command_sha256
        ):
            raise ValueError("episode scope command does not match test evidence")
        return self


class VerifiedEpisodeLookup(StrictModel):
    """Exact-namespace lookup; expired record content is never returned."""

    repository_id: Identifier
    repository_commit: GitSha
    repository_content_sha256: Sha256
    as_of: AwareDatetime
    status: VerifiedEpisodeLookupStatus
    records: tuple[VerifiedEpisodicRecord, ...] = ()
    expired_record_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_lookup_result(self) -> Self:
        if self.status is VerifiedEpisodeLookupStatus.CURRENT:
            if not self.records:
                raise ValueError("current lookup requires at least one record")
            if any(
                record.scope.repository_id != self.repository_id
                or record.scope.repository_commit != self.repository_commit
                or record.scope.repository_content_sha256 != self.repository_content_sha256
                or record.expires_at <= self.as_of
                for record in self.records
            ):
                raise ValueError("current lookup contains an unusable record")
        elif self.records:
            raise ValueError("non-current lookup cannot expose records")
        if self.status is VerifiedEpisodeLookupStatus.EXPIRED and self.expired_record_count == 0:
            raise ValueError("expired lookup requires an expired record count")
        if self.status is VerifiedEpisodeLookupStatus.EMPTY and self.expired_record_count != 0:
            raise ValueError("empty lookup cannot contain expired records")
        return self


class VerifiedEpisodicMemoryStore(Protocol):
    """Trusted admission boundary; there is deliberately no arbitrary public put."""

    def admit(
        self,
        *,
        repository_id: str,
        repository_content_sha256: str,
        result: FinalResult,
        trace: AgentTrace,
        execution: VerifiedEpisodeExecutionEvidence,
        artifact_root: Path,
        created_at: datetime,
    ) -> VerifiedEpisodicRecord:
        """Revalidate and persist one successful run."""

    def lookup(
        self,
        *,
        repository_id: str,
        repository_commit: str,
        repository_content_sha256: str,
        as_of: datetime,
    ) -> VerifiedEpisodeLookup:
        """Return only unexpired records from one exact content namespace."""


def verified_episode_id(repository_id: str, task_id: str, run_id: UUID) -> str:
    """Derive an immutable source-run identity."""

    identity = {
        "repository_id": repository_id,
        "task_id": task_id,
        "run_id": run_id,
    }
    return f"vep_{_canonical_sha256(identity)}"


def build_verified_episode_record(
    *,
    repository_id: str,
    repository_content_sha256: str,
    result: FinalResult,
    trace: AgentTrace,
    execution: VerifiedEpisodeExecutionEvidence,
    artifact_root: Path,
    created_at: datetime,
    ttl_days: int = DEFAULT_VERIFIED_EPISODE_TTL_DAYS,
) -> VerifiedEpisodicRecord:
    """Fail closed unless replay, Verifier, result, namespace, and artifacts agree."""

    _validate_ttl(ttl_days)
    try:
        replayed = replay_trace(trace)
    except TraceArtifactError:
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.TRACE_INVALID) from None

    state = replayed.final_state
    if (
        _IDENTIFIER.fullmatch(repository_id) is None
        or _SHA256.fullmatch(repository_content_sha256) is None
        or state.task.base_commit != trace.final_state.task.base_commit
        or result.task_id != trace.task_id
        or result.run_id != trace.run_id
        or state.task.task_id != result.task_id
        or state.run_id != result.run_id
    ):
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.NAMESPACE_MISMATCH)
    if (
        result.status is not RunStatus.SUCCESS
        or result.termination_reason is not TerminationReason.SUCCESS
        or not result.task_metrics.task_success
        or result.verifier_error_code is not None
        or result.error_category is not None
        or state.termination_reason is not TerminationReason.AGENT_STOPPED
    ):
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.VERIFIER_REJECTED)

    baseline = state.baseline_test_result
    validation_test = state.test_state.last_result
    patch = state.patch_state.diff_ref
    if (
        baseline is None
        or validation_test is None
        or state.test_state.commands_run < 2
        or patch is None
        or not state.patch_state.files_changed
    ):
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.EVIDENCE_INCOMPLETE)

    try:
        reproduction = build_reproduction_result(baseline)
        decision = RuleBasedVerifier().verify(
            VerificationEvidence(
                reproduction_result=reproduction,
                patch_state=state.patch_state,
                validation_test_run=validation_test,
                tool_steps=len(state.tool_results),
            )
        )
    except (ValidationError, ValueError):
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.EVIDENCE_INCOMPLETE) from None
    if decision.status is not RunStatus.SUCCESS:
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.VERIFIER_REJECTED)
    if (
        result.patch_path != decision.patch_path
        or result.reproduction_result != decision.reproduction_result
        or result.validation_result != decision.validation_result
        or result.task_metrics != decision.task_metrics
    ):
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.RESULT_MISMATCH)

    expected_prompt_versions = tuple(
        dict.fromkeys(
            response.prompt_version
            for event in trace.events
            if (response := event.provider_response) is not None
            and response.prompt_version is not None
        )
    )
    expected_context_versions = (context_policy_version_from_events(trace.events),)
    if (
        execution.prompt_versions != expected_prompt_versions
        or execution.context_policy_versions != expected_context_versions
    ):
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.RESULT_MISMATCH)
    _verify_source_artifacts(artifact_root, execution.artifacts)
    required_artifacts = _required_artifacts(
        patch,
        result.reproduction_result,
        result.validation_result,
    )
    if any(artifact not in execution.artifacts for artifact in required_artifacts):
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.ARTIFACT_MISMATCH)

    expires_at = created_at + timedelta(days=ttl_days)
    if expires_at <= created_at:
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.EXPIRED)
    scope = VerifiedEpisodeScope(
        repository_id=repository_id,
        repository_commit=state.task.base_commit,
        repository_content_sha256=repository_content_sha256,
        files_changed=tuple(sorted(state.patch_state.files_changed)),
        test_command_sha256=validation_test.command_sha256,
    )
    evidence = VerifiedEpisodeEvidence(
        source_result_sha256=_canonical_sha256(result),
        source_trace_sha256=trace.content_sha256,
        patch_artifact=patch,
        reproduction_result=result.reproduction_result,
        validation_result=result.validation_result,
        artifacts=execution.artifacts,
        artifacts_sha256=_canonical_sha256(execution.artifacts),
    )
    provenance = VerifiedEpisodeProvenance(
        model_provider=trace.provider_name,
        model_versions=tuple(
            dict.fromkeys(
                response.model
                for event in trace.events
                if (response := event.provider_response) is not None and response.model is not None
            )
        ),
        prompt_versions=execution.prompt_versions,
        context_policy_versions=execution.context_policy_versions,
        sandbox_policy_version=execution.sandbox_policy_version,
        sandbox_image_reference=execution.sandbox_image_reference,
        settings_sha256=execution.settings_sha256,
    )
    payload: dict[str, object] = {
        "schema_version": VERIFIED_EPISODIC_MEMORY_SCHEMA_VERSION,
        "episode_id": verified_episode_id(repository_id, result.task_id, result.run_id),
        "source_task_id": result.task_id,
        "source_run_id": result.run_id,
        "problem_statement": state.task.issue_title,
        "issue_fingerprint": _canonical_sha256(
            {
                "issue_title": state.task.issue_title,
                "issue_body": state.task.issue_body,
            }
        ),
        "repair_summary": (
            f"Verified repair changed {len(scope.files_changed)} file(s) and passed "
            f"{result.validation_result.tests_passed} test(s)."
        ),
        "scope": scope,
        "evidence": evidence,
        "provenance": provenance,
        "created_at": created_at,
        "expires_at": expires_at,
    }
    try:
        return VerifiedEpisodicRecord.model_validate(
            {
                **payload,
                "record_sha256": _canonical_sha256(payload),
            }
        )
    except ValidationError:
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.INVALID_WRITE) from None


def verify_verified_episode_integrity(record: VerifiedEpisodicRecord) -> None:
    """Reject envelope or nested evidence tampering."""

    payload = record.model_dump(mode="python", exclude={"record_sha256"})
    if (
        _canonical_sha256(record.evidence.artifacts) != record.evidence.artifacts_sha256
        or _canonical_sha256(payload) != record.record_sha256
    ):
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.INTEGRITY_MISMATCH)


def project_verified_episode_record(
    record: VerifiedEpisodicRecord,
    *,
    as_of: datetime,
) -> VerifiedEpisodeMemoryItem:
    """Project one integrity-checked, unexpired record into bounded Agent state."""

    verify_verified_episode_integrity(record)
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.INVALID_WRITE)
    if record.expires_at <= as_of:
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.EXPIRED)
    return VerifiedEpisodeMemoryItem(
        episode_id=record.episode_id,
        source_task_id=record.source_task_id,
        source_run_id=record.source_run_id,
        repository_id=record.scope.repository_id,
        repository_commit=record.scope.repository_commit,
        repository_content_sha256=record.scope.repository_content_sha256,
        problem_statement=record.problem_statement,
        issue_fingerprint=record.issue_fingerprint,
        repair_summary=record.repair_summary,
        files_changed=record.scope.files_changed,
        test_command_sha256=record.scope.test_command_sha256,
        patch_artifact_sha256=record.evidence.patch_artifact.sha256,
        source_result_sha256=record.evidence.source_result_sha256,
        source_trace_sha256=record.evidence.source_trace_sha256,
        artifacts_sha256=record.evidence.artifacts_sha256,
        verifier_version=record.provenance.verifier_version,
        model_provider=record.provenance.model_provider,
        validation_tests_passed=record.evidence.validation_result.tests_passed,
        created_at=record.created_at,
        expires_at=record.expires_at,
        source_record_sha256=record.record_sha256,
    )


class FileVerifiedEpisodicMemoryStore:
    """Bounded no-overwrite store partitioned by repository/commit/content."""

    def __init__(
        self,
        root: Path,
        *,
        ttl_days: int = DEFAULT_VERIFIED_EPISODE_TTL_DAYS,
        max_records_per_repository: int = (DEFAULT_MAX_VERIFIED_EPISODES_PER_REPOSITORY),
        max_record_bytes: int = DEFAULT_MAX_VERIFIED_EPISODE_BYTES,
    ) -> None:
        if not root.is_absolute():
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.PATH_UNSAFE)
        _validate_ttl(ttl_days)
        _validate_limit(max_records_per_repository, 1, 4096, "max_records_per_repository")
        _validate_limit(
            max_record_bytes,
            1024,
            MAX_VERIFIED_EPISODE_BYTES,
            "max_record_bytes",
        )
        self._root = root
        self._ttl_days = ttl_days
        self._max_records = max_records_per_repository
        self._max_record_bytes = max_record_bytes

    def admit(
        self,
        *,
        repository_id: str,
        repository_content_sha256: str,
        result: FinalResult,
        trace: AgentTrace,
        execution: VerifiedEpisodeExecutionEvidence,
        artifact_root: Path,
        created_at: datetime,
    ) -> VerifiedEpisodicRecord:
        record = build_verified_episode_record(
            repository_id=repository_id,
            repository_content_sha256=repository_content_sha256,
            result=result,
            trace=trace,
            execution=execution,
            artifact_root=artifact_root,
            created_at=created_at,
            ttl_days=self._ttl_days,
        )
        return self._store(record)

    def lookup(
        self,
        *,
        repository_id: str,
        repository_commit: str,
        repository_content_sha256: str,
        as_of: datetime,
    ) -> VerifiedEpisodeLookup:
        _validate_namespace(repository_id, repository_commit, repository_content_sha256)
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.INVALID_WRITE)
        namespace = self._namespace(
            repository_id,
            repository_commit,
            repository_content_sha256,
            create=False,
            missing_ok=True,
        )
        if namespace is None:
            return self._lookup_result(
                repository_id,
                repository_commit,
                repository_content_sha256,
                as_of,
                VerifiedEpisodeLookupStatus.EMPTY,
            )
        records = tuple(self._read(path) for _, path in self._record_paths(namespace))
        for record in records:
            self._validate_record(
                record,
                repository_id,
                repository_commit,
                repository_content_sha256,
            )
        current = tuple(record for record in records if record.expires_at > as_of)
        expired_count = len(records) - len(current)
        if current:
            return VerifiedEpisodeLookup(
                repository_id=repository_id,
                repository_commit=repository_commit,
                repository_content_sha256=repository_content_sha256,
                as_of=as_of,
                status=VerifiedEpisodeLookupStatus.CURRENT,
                records=current,
                expired_record_count=expired_count,
            )
        if expired_count:
            return self._lookup_result(
                repository_id,
                repository_commit,
                repository_content_sha256,
                as_of,
                VerifiedEpisodeLookupStatus.EXPIRED,
                expired_record_count=expired_count,
            )
        return self._lookup_result(
            repository_id,
            repository_commit,
            repository_content_sha256,
            as_of,
            VerifiedEpisodeLookupStatus.EMPTY,
        )

    def _store(self, record: VerifiedEpisodicRecord) -> VerifiedEpisodicRecord:
        self._validate_record(
            record,
            record.scope.repository_id,
            record.scope.repository_commit,
            record.scope.repository_content_sha256,
        )
        repository_dir = self._repository_directory(
            record.scope.repository_id,
            create=True,
        )
        assert repository_dir is not None
        namespace = self._namespace(
            record.scope.repository_id,
            record.scope.repository_commit,
            record.scope.repository_content_sha256,
            create=True,
        )
        assert namespace is not None
        existing_path = namespace / f"{record.episode_id}.json"
        if existing_path.exists():
            existing = self._read(existing_path)
            if _semantic_record_payload(existing) == _semantic_record_payload(record):
                return existing
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.DUPLICATE_CONFLICT)
        if self._repository_record_count(repository_dir) >= self._max_records:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.CAPACITY_EXCEEDED)
        serialized = _canonical_json(record).encode("utf-8") + b"\n"
        if len(serialized) > self._max_record_bytes:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.CAPACITY_EXCEEDED)
        self._atomic_create(existing_path, serialized)
        return record

    def _validate_record(
        self,
        record: VerifiedEpisodicRecord,
        repository_id: str,
        repository_commit: str,
        repository_content_sha256: str,
    ) -> None:
        verify_verified_episode_integrity(record)
        if (
            record.scope.repository_id != repository_id
            or record.scope.repository_commit != repository_commit
            or record.scope.repository_content_sha256 != repository_content_sha256
        ):
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.NAMESPACE_MISMATCH)
        if record.expires_at - record.created_at > timedelta(days=MAX_VERIFIED_EPISODE_TTL_DAYS):
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.INVALID_WRITE)

    def _repository_directory(
        self,
        repository_id: str,
        *,
        create: bool,
        missing_ok: bool = False,
    ) -> Path | None:
        if _IDENTIFIER.fullmatch(repository_id) is None:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.NAMESPACE_MISMATCH)
        root = self._safe_directory(self._root, create=create, missing_ok=missing_ok)
        if root is None:
            return None
        return self._safe_directory(
            root / repository_id,
            create=create,
            missing_ok=missing_ok,
        )

    def _namespace(
        self,
        repository_id: str,
        repository_commit: str,
        repository_content_sha256: str,
        *,
        create: bool,
        missing_ok: bool = False,
    ) -> Path | None:
        _validate_namespace(repository_id, repository_commit, repository_content_sha256)
        repository_dir = self._repository_directory(
            repository_id,
            create=create,
            missing_ok=missing_ok,
        )
        if repository_dir is None:
            return None
        commit_dir = self._safe_directory(
            repository_dir / repository_commit,
            create=create,
            missing_ok=missing_ok,
        )
        if commit_dir is None:
            return None
        return self._safe_directory(
            commit_dir / repository_content_sha256,
            create=create,
            missing_ok=missing_ok,
        )

    def _repository_record_count(self, repository_dir: Path) -> int:
        count = 0
        for commit_dir in self._entries(repository_dir):
            metadata = self._lstat(commit_dir)
            if not stat.S_ISDIR(metadata.st_mode) or _GIT_SHA.fullmatch(commit_dir.name) is None:
                raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.PATH_UNSAFE)
            for content_dir in self._entries(commit_dir):
                content_metadata = self._lstat(content_dir)
                if (
                    not stat.S_ISDIR(content_metadata.st_mode)
                    or _SHA256.fullmatch(content_dir.name) is None
                ):
                    raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.PATH_UNSAFE)
                count += len(self._record_paths(content_dir))
        return count

    def _record_paths(self, namespace: Path) -> tuple[tuple[str, Path], ...]:
        paths: list[tuple[str, Path]] = []
        for entry in self._entries(namespace):
            match = _RECORD_NAME.fullmatch(entry.name)
            metadata = self._lstat(entry)
            if match is None or not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
                raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.PATH_UNSAFE)
            paths.append((match.group("episode_id"), entry))
        paths.sort(key=lambda item: item[0])
        return tuple(paths)

    def _read(self, path: Path) -> VerifiedEpisodicRecord:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as stream:
                serialized = stream.read(self._max_record_bytes + 1)
        except OSError:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.IO_ERROR) from None
        if len(serialized) > self._max_record_bytes:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.CAPACITY_EXCEEDED)
        try:
            raw = json.loads(serialized)
        except (json.JSONDecodeError, UnicodeError):
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.INVALID_WRITE) from None
        if not isinstance(raw, dict):
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.INVALID_WRITE)
        if raw.get("schema_version") != VERIFIED_EPISODIC_MEMORY_SCHEMA_VERSION:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.UNSUPPORTED_VERSION)
        digest = raw.get("record_sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.INVALID_WRITE)
        envelope = {key: value for key, value in raw.items() if key != "record_sha256"}
        if _canonical_sha256(envelope) != digest:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.INTEGRITY_MISMATCH)
        try:
            record = VerifiedEpisodicRecord.model_validate(raw)
        except ValidationError:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.INVALID_WRITE) from None
        verify_verified_episode_integrity(record)
        if path.stem != record.episode_id:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.NAMESPACE_MISMATCH)
        return record

    @staticmethod
    def _safe_directory(
        path: Path,
        *,
        create: bool,
        missing_ok: bool = False,
    ) -> Path | None:
        try:
            if create:
                path.mkdir(mode=0o700, parents=False, exist_ok=True)
            metadata = path.lstat()
        except FileNotFoundError:
            if missing_ok:
                return None
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.IO_ERROR) from None
        except OSError:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.IO_ERROR) from None
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.PATH_UNSAFE)
        with suppress(OSError):
            path.chmod(0o700)
        return path

    @staticmethod
    def _entries(path: Path) -> tuple[Path, ...]:
        try:
            return tuple(sorted(path.iterdir(), key=lambda item: item.name))
        except OSError:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.IO_ERROR) from None

    @staticmethod
    def _lstat(path: Path) -> os.stat_result:
        try:
            metadata = path.lstat()
        except OSError:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.IO_ERROR) from None
        if stat.S_ISLNK(metadata.st_mode):
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.PATH_UNSAFE)
        return metadata

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
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.DUPLICATE_CONFLICT) from None
        except OSError:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.IO_ERROR) from None
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
            with suppress(OSError):
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _lookup_result(
        repository_id: str,
        repository_commit: str,
        repository_content_sha256: str,
        as_of: datetime,
        status: VerifiedEpisodeLookupStatus,
        *,
        expired_record_count: int = 0,
    ) -> VerifiedEpisodeLookup:
        return VerifiedEpisodeLookup(
            repository_id=repository_id,
            repository_commit=repository_commit,
            repository_content_sha256=repository_content_sha256,
            as_of=as_of,
            status=status,
            expired_record_count=expired_record_count,
        )


def _required_artifacts(
    patch: ArtifactRef,
    reproduction: ReproductionResult,
    validation: ValidationResult,
) -> tuple[ArtifactRef, ...]:
    refs: list[ArtifactRef] = [patch]
    for test_result in (reproduction.test_run, validation.test_run):
        if test_result is None:
            continue
        refs.extend(
            artifact
            for artifact in (test_result.stdout_ref, test_result.stderr_ref)
            if artifact is not None
        )
    return tuple(refs)


def _verify_source_artifacts(
    artifact_root: Path,
    artifacts: tuple[ArtifactRef, ...],
) -> None:
    if not artifact_root.is_absolute():
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.PATH_UNSAFE)
    try:
        root_metadata = artifact_root.lstat()
        resolved_root = artifact_root.resolve(strict=True)
    except OSError:
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.ARTIFACT_MISMATCH) from None
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.PATH_UNSAFE)
    for artifact in artifacts:
        candidate = artifact_root / artifact.relative_path
        try:
            metadata = candidate.lstat()
            resolved = candidate.resolve(strict=True)
        except OSError:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.ARTIFACT_MISMATCH) from None
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or not resolved.is_relative_to(resolved_root)
        ):
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.PATH_UNSAFE)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(candidate, flags)
            with os.fdopen(descriptor, "rb") as stream:
                content = stream.read(artifact.size_bytes + 1)
        except OSError:
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.ARTIFACT_MISMATCH) from None
        if (
            len(content) != artifact.size_bytes
            or hashlib.sha256(content).hexdigest() != artifact.sha256
        ):
            raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.ARTIFACT_MISMATCH)


def _validate_namespace(
    repository_id: str,
    repository_commit: str,
    repository_content_sha256: str,
) -> None:
    if (
        _IDENTIFIER.fullmatch(repository_id) is None
        or _GIT_SHA.fullmatch(repository_commit) is None
        or _SHA256.fullmatch(repository_content_sha256) is None
    ):
        raise VerifiedEpisodeError(VerifiedEpisodeErrorCode.NAMESPACE_MISMATCH)


def _validate_ttl(ttl_days: int) -> None:
    _validate_limit(ttl_days, 1, MAX_VERIFIED_EPISODE_TTL_DAYS, "ttl_days")


def _validate_limit(value: int, minimum: int, maximum: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _semantic_record_payload(record: VerifiedEpisodicRecord) -> dict[str, object]:
    return record.model_dump(
        mode="python",
        exclude={"created_at", "expires_at", "record_sha256"},
    )


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        _json_compatible(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_compatible(payload: object) -> JsonValue:
    return cast(JsonValue, to_jsonable_python(payload))


__all__ = [
    "DEFAULT_MAX_VERIFIED_EPISODES_PER_REPOSITORY",
    "DEFAULT_MAX_VERIFIED_EPISODE_BYTES",
    "DEFAULT_VERIFIED_EPISODE_TTL_DAYS",
    "MAX_VERIFIED_EPISODE_TTL_DAYS",
    "SANDBOX_POLICY_VERSION_ID",
    "VERIFIED_EPISODIC_MEMORY_SCHEMA_VERSION",
    "VERIFIED_REPAIR_PRODUCER_VERSION",
    "FileVerifiedEpisodicMemoryStore",
    "VerifiedEpisodeError",
    "VerifiedEpisodeEvidence",
    "VerifiedEpisodeExecutionEvidence",
    "VerifiedEpisodeLookup",
    "VerifiedEpisodeProvenance",
    "VerifiedEpisodeScope",
    "VerifiedEpisodicMemoryStore",
    "VerifiedEpisodicRecord",
    "build_verified_episode_record",
    "project_verified_episode_record",
    "verified_episode_id",
    "verify_verified_episode_integrity",
]
