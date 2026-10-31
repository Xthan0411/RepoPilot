"""Frozen P7 experiment matrix, arm selection, and fail-closed preflight."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Self, cast

import yaml
from pydantic import AwareDatetime, Field, JsonValue, SecretStr, ValidationError, model_validator
from pydantic_core import to_jsonable_python

from repopilot.config import DEFAULT_DATABASE_URL, RepoPilotSettings
from repopilot.embeddings import EmbeddingProvider, LocalEmbeddingCache
from repopilot.enums import (
    DatasetSplit,
    EmbeddingProviderKind,
    ExperimentArmKind,
    ExperimentErrorCode,
    ExperimentPreflightCheckStatus,
    ExperimentPreflightStatus,
    MemoryAccessMode,
    ModelProvider,
    RepairContextMode,
    RetrievalMode,
)
from repopilot.evals.baselines import (
    BaselineEvaluationExecutor,
    load_baseline_definition,
)
from repopilot.evals.dataset import DatasetManifestError, load_frozen_dataset
from repopilot.evals.runner import (
    EvaluationTaskExecutor,
    ProviderResponsesFactory,
    RepairWorkflowEvaluationExecutor,
    evaluation_task_set_sha256,
)
from repopilot.memory import MemoryNamespaceManifest, MemoryStoreBundle
from repopilot.providers import PROMPT_VERSION
from repopilot.repair_workflow import RepairSandboxFactory
from repopilot.schemas import (
    ArtifactRef,
    Identifier,
    MemoryNamespace,
    Sha256,
    StrictModel,
    TaskId,
    VersionIdentifier,
)
from repopilot.storage import VectorStore

EXPERIMENT_MATRIX_SCHEMA_VERSION: Literal[1] = 1
EXPERIMENT_PREFLIGHT_SCHEMA_VERSION: Literal[1] = 1
EXPERIMENT_MATRIX_VERSION: Literal["experiment-matrix-v2"] = "experiment-matrix-v2"
EXPERIMENT_PREFLIGHT_VERSION: Literal["experiment-preflight-v1"] = "experiment-preflight-v1"
EXPERIMENT_SUCCESS_CRITERION: Literal["rule-based-verifier-v1"] = "rule-based-verifier-v1"
FROZEN_OPENAI_MODEL: Literal["gpt-5.6-terra"] = "gpt-5.6-terra"
FROZEN_PROMPT_VERSION: Literal["repopilot-agent-v1"] = "repopilot-agent-v1"
FROZEN_EMBEDDING_MODEL: Literal["jinaai/jina-embeddings-v2-base-code"] = (
    "jinaai/jina-embeddings-v2-base-code"
)
FROZEN_EMBEDDING_REVISION: Literal["516f4baf13dec4ddddda8631e019b5737c8bc250"] = (
    "516f4baf13dec4ddddda8631e019b5737c8bc250"
)
FROZEN_EMBEDDING_CODE_REVISION: Literal["3baf9e3ac750e76e8edd3019170176884695fb94"] = (
    "3baf9e3ac750e76e8edd3019170176884695fb94"
)
ALLOWED_SECRET_ENVIRONMENT = ("DATABASE_URL", "OPENAI_API_KEY")
_MAX_MATRIX_BYTES = 512 * 1024
_MAX_PREFLIGHT_BYTES = 512 * 1024
_IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:~-]{0,439}@sha256:[0-9a-f]{64}$")
_SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "database_url",
        "gold",
        "gold_files",
        "gold_patch",
        "openai_api_key",
        "oracle",
        "password",
        "secret",
        "test_patch",
        "token",
    }
)
_ARM_IDS = (
    "dense-only-v1",
    "hybrid-memory-v1",
    "hybrid-rrf-v1",
    "mini-swe-safe-v1",
    "no-semantic-rag-v1",
)
_SHARED_SETTING_KEYS = frozenset(RepoPilotSettings.model_fields) - {
    "database_url",
    "langsmith_api_key",
    "openai_api_key",
    "retrieval_mode",
}


EXPERIMENT_ERROR_MESSAGES: Mapping[ExperimentErrorCode, str] = {
    ExperimentErrorCode.INPUT_IO_ERROR: "experiment input is unavailable",
    ExperimentErrorCode.INVALID_CONFIGURATION: "experiment configuration is invalid",
    ExperimentErrorCode.UNSUPPORTED_VERSION: "experiment schema version is unsupported",
    ExperimentErrorCode.INTEGRITY_MISMATCH: "experiment integrity check failed",
    ExperimentErrorCode.DATASET_MISMATCH: "experiment dataset identity does not match",
    ExperimentErrorCode.FAIRNESS_VIOLATION: "experiment arms violate the fairness contract",
    ExperimentErrorCode.ARM_UNAVAILABLE: "experiment arm is unavailable",
    ExperimentErrorCode.MODEL_SNAPSHOT_INVALID: "model snapshot is not immutable",
    ExperimentErrorCode.EMBEDDING_SNAPSHOT_INVALID: "embedding snapshot is not immutable",
    ExperimentErrorCode.MEMORY_SNAPSHOT_INVALID: "Memory snapshot is invalid",
    ExperimentErrorCode.DEPENDENCY_LOCK_MISMATCH: "dependency lock does not match",
    ExperimentErrorCode.SOURCE_SNAPSHOT_UNCLEAN: "source snapshot is not clean and frozen",
    ExperimentErrorCode.SANDBOX_IMAGE_UNAVAILABLE: "Sandbox image is unavailable",
    ExperimentErrorCode.PROVIDER_UNAVAILABLE: "model Provider is unavailable",
    ExperimentErrorCode.VECTOR_BACKEND_UNAVAILABLE: "vector backend is unavailable",
    ExperimentErrorCode.GOLD_BOUNDARY_VIOLATION: "evaluator gold crossed the Agent boundary",
    ExperimentErrorCode.OUTPUT_CONFLICT: "experiment output root is not ready",
    ExperimentErrorCode.SECRET_EXPOSURE: "experiment artifact contains a secret field",
}


class ExperimentError(RuntimeError):
    """Safe typed error that never includes configuration values or host paths."""

    def __init__(self, code: ExperimentErrorCode) -> None:
        super().__init__(EXPERIMENT_ERROR_MESSAGES[code])
        self.code = code


class ExperimentTaskSet(StrictModel):
    """One split's stable task selection and Runner-compatible identity."""

    split: DatasetSplit
    task_ids: tuple[TaskId, ...] = Field(min_length=1, max_length=100)
    task_set_sha256: Sha256

    @model_validator(mode="after")
    def validate_task_ids(self) -> Self:
        if self.task_ids != tuple(sorted(self.task_ids)) or len(set(self.task_ids)) != len(
            self.task_ids
        ):
            raise ValueError("experiment task IDs must be sorted and unique")
        return self


class ExperimentDatasetSnapshot(StrictModel):
    """Frozen dataset file, semantic manifest, and split task identities."""

    dataset_id: Identifier
    manifest_path: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")
    manifest_file_sha256: Sha256
    manifest_content_sha256: Sha256
    task_sets: tuple[ExperimentTaskSet, ExperimentTaskSet]

    @model_validator(mode="after")
    def validate_splits(self) -> Self:
        if tuple(item.split for item in self.task_sets) != (
            DatasetSplit.DEV,
            DatasetSplit.HOLDOUT,
        ):
            raise ValueError("experiment task sets must contain dev then holdout")
        return self


class ExperimentSharedSettings(StrictModel):
    """Secret-free settings common to every arm; retrieval mode is arm-owned."""

    values: dict[str, JsonValue]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_settings(self) -> Self:
        if set(self.values) != _SHARED_SETTING_KEYS:
            raise ValueError("shared settings are incomplete or include arm-owned fields")
        _reject_sensitive_keys(self.values)
        if _canonical_sha256(self.values) != self.content_sha256:
            raise ValueError("shared settings digest does not match")
        try:
            settings = RepoPilotSettings.model_validate(
                {
                    **self.values,
                    "database_url": DEFAULT_DATABASE_URL,
                    "openai_api_key": "preflight-placeholder",
                    "retrieval_mode": RetrievalMode.BM25,
                }
            )
        except ValidationError:
            raise ValueError("shared settings do not form a valid runtime") from None
        if (
            settings.model_provider is not ModelProvider.OPENAI
            or settings.model != FROZEN_OPENAI_MODEL
            or settings.temperature != 0.0
            or settings.embedding_provider is not EmbeddingProviderKind.JINA_LOCAL
            or settings.embedding_model != FROZEN_EMBEDDING_MODEL
            or settings.embedding_revision != FROZEN_EMBEDDING_REVISION
            or settings.embedding_code_revision != FROZEN_EMBEDDING_CODE_REVISION
            or settings.query_rewrite_enabled
            or settings.langsmith_tracing
        ):
            raise ValueError("shared model, embedding, or tracing settings are not frozen")
        return self


class ExperimentMemorySnapshot(StrictModel):
    """One split-isolated Memory declaration and exact on-disk manifest."""

    snapshot_id: Identifier
    namespace: MemoryNamespace
    manifest_ref: ArtifactRef
    repository_record_count: Literal[0]
    episodic_record_count: Literal[0]

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if self.namespace.snapshot_id != self.snapshot_id:
            raise ValueError("Memory snapshot identity does not match its namespace")
        if self.namespace.access_mode is not MemoryAccessMode.READ_ONLY:
            raise ValueError("frozen experiment Memory snapshots must be read-only")
        if self.manifest_ref.media_type != "application/json":
            raise ValueError("Memory namespace manifest must be JSON")
        return self


class ExperimentArmDefinition(StrictModel):
    """Only retrieval/context/Memory may differ between experiment arms."""

    arm_id: Identifier
    kind: ExperimentArmKind
    enabled_by_default: bool
    context_mode: RepairContextMode
    retrieval_mode: RetrievalMode | None
    query_rewrite_enabled: Literal[False]
    memory_enabled: bool
    baseline_config_ref: ArtifactRef | None = None
    dev_memory_snapshot_id: Identifier | None = None
    holdout_memory_snapshot_id: Identifier | None = None

    @model_validator(mode="after")
    def validate_capabilities(self) -> Self:
        expected = {
            ExperimentArmKind.MINI_SWE_SAFE: (
                "mini-swe-safe-v1",
                RepairContextMode.NONE,
                None,
                False,
                True,
            ),
            ExperimentArmKind.NO_SEMANTIC_RAG: (
                "no-semantic-rag-v1",
                RepairContextMode.BM25,
                RetrievalMode.BM25,
                False,
                True,
            ),
            ExperimentArmKind.DENSE_ONLY: (
                "dense-only-v1",
                RepairContextMode.CONFIGURED,
                RetrievalMode.DENSE,
                False,
                True,
            ),
            ExperimentArmKind.HYBRID_RRF: (
                "hybrid-rrf-v1",
                RepairContextMode.CONFIGURED,
                RetrievalMode.HYBRID,
                False,
                True,
            ),
            ExperimentArmKind.HYBRID_MEMORY: (
                "hybrid-memory-v1",
                RepairContextMode.CONFIGURED,
                RetrievalMode.HYBRID,
                True,
                False,
            ),
        }[self.kind]
        if (
            self.arm_id,
            self.context_mode,
            self.retrieval_mode,
            self.memory_enabled,
            self.enabled_by_default,
        ) != expected:
            raise ValueError("experiment arm capabilities do not match its identity")
        is_baseline = self.kind in {
            ExperimentArmKind.MINI_SWE_SAFE,
            ExperimentArmKind.NO_SEMANTIC_RAG,
        }
        if is_baseline != (self.baseline_config_ref is not None):
            raise ValueError("only baseline arms require a baseline configuration")
        if self.baseline_config_ref is not None and (
            self.baseline_config_ref.media_type != "application/yaml"
        ):
            raise ValueError("baseline configuration must be YAML")
        snapshot_ids = (self.dev_memory_snapshot_id, self.holdout_memory_snapshot_id)
        if self.memory_enabled != all(item is not None for item in snapshot_ids):
            raise ValueError("Memory arms require both split snapshots")
        if not self.memory_enabled and any(item is not None for item in snapshot_ids):
            raise ValueError("non-Memory arms cannot reference Memory snapshots")
        return self


class ExperimentMatrix(StrictModel):
    """Integrity-protected P7 comparison matrix with structural fairness."""

    schema_version: Literal[1] = EXPERIMENT_MATRIX_SCHEMA_VERSION
    matrix_version: Literal["experiment-matrix-v2"] = EXPERIMENT_MATRIX_VERSION
    experiment_id: Identifier
    dataset: ExperimentDatasetSnapshot
    dependency_lock_path: Literal["uv.lock"]
    dependency_lock_sha256: Sha256
    shared_settings: ExperimentSharedSettings
    prompt_versions: tuple[VersionIdentifier, ...] = Field(min_length=1, max_length=8)
    temperature_transport: Literal["frozen_setting_not_sent_to_reasoning_provider"]
    random_seed: int = Field(ge=0, le=2**32 - 1, strict=True)
    dev_repetitions: int = Field(ge=1, le=10, strict=True)
    holdout_repetitions: int = Field(ge=1, le=10, strict=True)
    repetition_policy: Literal["budget_constrained_single_run_no_variance_claim"]
    sandbox_image_reference: str = Field(min_length=1, max_length=512)
    success_criterion: Literal["rule-based-verifier-v1"] = EXPERIMENT_SUCCESS_CRITERION
    allowed_secret_environment: tuple[str, ...]
    memory_snapshots: tuple[ExperimentMemorySnapshot, ...]
    arms: tuple[ExperimentArmDefinition, ...]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        # A frozen artifact must remain readable after runtime Prompt upgrades. Runtime
        # compatibility is a separate fail-closed preflight check, not schema validity.
        if self.prompt_versions != (FROZEN_PROMPT_VERSION,):
            raise ValueError("experiment Prompt version is not the frozen matrix version")
        if _IMAGE_PATTERN.fullmatch(self.sandbox_image_reference) is None:
            raise ValueError("experiment Sandbox image must use an immutable digest")
        if self.allowed_secret_environment != ALLOWED_SECRET_ENVIRONMENT:
            raise ValueError("formal experiment environment overrides are not allowlisted")
        arm_ids = tuple(arm.arm_id for arm in self.arms)
        if arm_ids != _ARM_IDS:
            raise ValueError("experiment arms must be complete and sorted")
        snapshot_ids = tuple(item.snapshot_id for item in self.memory_snapshots)
        if snapshot_ids != tuple(sorted(snapshot_ids)) or len(set(snapshot_ids)) != len(
            snapshot_ids
        ):
            raise ValueError("Memory snapshots must be sorted and unique")
        snapshots = {item.snapshot_id: item for item in self.memory_snapshots}
        for arm in self.arms:
            if not arm.memory_enabled:
                continue
            dev = snapshots.get(cast(str, arm.dev_memory_snapshot_id))
            holdout = snapshots.get(cast(str, arm.holdout_memory_snapshot_id))
            if (
                dev is None
                or holdout is None
                or dev.namespace.split is not DatasetSplit.DEV
                or holdout.namespace.split is not DatasetSplit.HOLDOUT
                or dev.namespace.experiment_id != self.experiment_id
                or holdout.namespace.experiment_id != self.experiment_id
            ):
                raise ValueError("Memory arm snapshots violate split isolation")
        payload = self.model_dump(mode="python", exclude={"content_sha256"})
        _reject_sensitive_keys(payload, allow_environment_names=True)
        if _canonical_sha256(payload) != self.content_sha256:
            raise ValueError("experiment matrix digest does not match")
        return self


class ExperimentPreflightCheck(StrictModel):
    """One bounded readiness decision with a stable failure category."""

    check_id: Identifier
    status: ExperimentPreflightCheckStatus
    error_code: ExperimentErrorCode | None = None
    detail: Literal["verified", "blocked", "not_run"]

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.status is ExperimentPreflightCheckStatus.PASSED:
            if self.error_code is not None or self.detail != "verified":
                raise ValueError("passed preflight checks cannot carry an error")
        elif self.error_code is None or self.detail == "verified":
            raise ValueError("non-passing preflight checks require an error")
        return self


class ExperimentPreflightReport(StrictModel):
    """Integrity-protected formal-run readiness report."""

    schema_version: Literal[1] = EXPERIMENT_PREFLIGHT_SCHEMA_VERSION
    preflight_version: Literal["experiment-preflight-v1"] = EXPERIMENT_PREFLIGHT_VERSION
    experiment_id: Identifier
    matrix_sha256: Sha256
    checked_at: AwareDatetime
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    status: ExperimentPreflightStatus
    ready_for_formal_run: bool
    checks: tuple[ExperimentPreflightCheck, ...] = Field(min_length=10, max_length=20)
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        check_ids = tuple(item.check_id for item in self.checks)
        if check_ids != tuple(sorted(check_ids)) or len(set(check_ids)) != len(check_ids):
            raise ValueError("preflight checks must be sorted and unique")
        all_passed = all(
            item.status is ExperimentPreflightCheckStatus.PASSED for item in self.checks
        )
        if self.ready_for_formal_run != all_passed:
            raise ValueError("preflight readiness does not match checks")
        expected_status = (
            ExperimentPreflightStatus.READY
            if self.ready_for_formal_run
            else ExperimentPreflightStatus.BLOCKED
        )
        if self.status is not expected_status:
            raise ValueError("preflight status does not match readiness")
        payload = self.model_dump(mode="python", exclude={"content_sha256"})
        if _canonical_sha256(payload) != self.content_sha256:
            raise ValueError("preflight report digest does not match")
        return self


@dataclass(frozen=True)
class ExperimentPreflightFacts:
    """Machine-specific facts kept outside the immutable experiment matrix."""

    source_commit: str
    source_clean: bool
    local_dependencies_available: bool | None
    sandbox_image_available: bool | None
    provider_credential_available: bool
    vector_backend_available: bool | None
    embedding_snapshot_available: bool | None
    output_root_ready: bool | None


@dataclass(frozen=True)
class ExperimentArmSelection:
    """One selected arm whose only runtime variables are two secret connections."""

    matrix: ExperimentMatrix
    arm: ExperimentArmDefinition

    def settings(
        self,
        *,
        openai_api_key: str | SecretStr,
        database_url: str | SecretStr,
    ) -> RepoPilotSettings:
        """Materialize settings without consulting REPOPILOT_* environment overrides."""

        try:
            return RepoPilotSettings.model_validate(
                {
                    **self.matrix.shared_settings.values,
                    "retrieval_mode": self.arm.retrieval_mode or RetrievalMode.BM25,
                    "openai_api_key": openai_api_key,
                    "database_url": database_url,
                }
            )
        except ValidationError:
            raise ExperimentError(ExperimentErrorCode.INVALID_CONFIGURATION) from None

    @property
    def configuration_snapshot(self) -> dict[str, object]:
        """Return the exact secret-free arm declaration persisted by the Runner."""

        return {
            "matrix_sha256": self.matrix.content_sha256,
            "arm": self.arm.model_dump(mode="python"),
        }

    @property
    def configuration_sha256(self) -> Sha256:
        return _canonical_sha256(self.configuration_snapshot)


def load_experiment_matrix(path: Path, *, repository_root: Path) -> ExperimentMatrix:
    """Load a bounded matrix and verify every referenced project artifact."""

    root = _validated_root(repository_root)
    content = _read_contained(root, path, _MAX_MATRIX_BYTES)
    try:
        loaded = yaml.safe_load(content)
    except (UnicodeError, yaml.YAMLError):
        raise ExperimentError(ExperimentErrorCode.INVALID_CONFIGURATION) from None
    if not isinstance(loaded, dict):
        raise ExperimentError(ExperimentErrorCode.INVALID_CONFIGURATION)
    if loaded.get("schema_version") != EXPERIMENT_MATRIX_SCHEMA_VERSION:
        raise ExperimentError(ExperimentErrorCode.UNSUPPORTED_VERSION)
    try:
        matrix = ExperimentMatrix.model_validate(loaded)
    except ValidationError:
        raise ExperimentError(ExperimentErrorCode.INVALID_CONFIGURATION) from None
    _verify_file_sha256(
        root,
        matrix.dataset.manifest_path,
        matrix.dataset.manifest_file_sha256,
    )
    _verify_file_sha256(root, matrix.dependency_lock_path, matrix.dependency_lock_sha256)
    for arm in matrix.arms:
        if arm.baseline_config_ref is not None:
            _verify_artifact_ref(root, arm.baseline_config_ref)
    for snapshot in matrix.memory_snapshots:
        _verify_artifact_ref(root, snapshot.manifest_ref)
        try:
            manifest = MemoryNamespaceManifest.model_validate_json(
                _read_contained(
                    root,
                    Path(snapshot.manifest_ref.relative_path),
                    snapshot.manifest_ref.size_bytes,
                )
            )
        except (UnicodeError, ValidationError):
            raise ExperimentError(ExperimentErrorCode.MEMORY_SNAPSHOT_INVALID) from None
        if manifest.namespace != snapshot.namespace:
            raise ExperimentError(ExperimentErrorCode.MEMORY_SNAPSHOT_INVALID)
    return matrix


def select_experiment_arm(
    matrix: ExperimentMatrix,
    arm_id: str,
    *,
    include_optional: bool = False,
) -> ExperimentArmSelection:
    """Select one complete arm by ID; optional Memory requires explicit opt-in."""

    arm = next((candidate for candidate in matrix.arms if candidate.arm_id == arm_id), None)
    if arm is None or (not arm.enabled_by_default and not include_optional):
        raise ExperimentError(ExperimentErrorCode.ARM_UNAVAILABLE)
    return ExperimentArmSelection(matrix=matrix, arm=arm)


def load_experiment_preflight(
    path: Path,
    *,
    repository_root: Path,
    matrix: ExperimentMatrix,
) -> ExperimentPreflightReport:
    """Load a bounded report and bind it to the exact matrix under review."""

    root = _validated_root(repository_root)
    content = _read_contained(root, path, _MAX_PREFLIGHT_BYTES)
    try:
        loaded = json.loads(content)
    except (UnicodeError, json.JSONDecodeError):
        raise ExperimentError(ExperimentErrorCode.INVALID_CONFIGURATION) from None
    if not isinstance(loaded, dict):
        raise ExperimentError(ExperimentErrorCode.INVALID_CONFIGURATION)
    if loaded.get("schema_version") != EXPERIMENT_PREFLIGHT_SCHEMA_VERSION:
        raise ExperimentError(ExperimentErrorCode.UNSUPPORTED_VERSION)
    try:
        report = ExperimentPreflightReport.model_validate(loaded)
    except ValidationError:
        raise ExperimentError(ExperimentErrorCode.INVALID_CONFIGURATION) from None
    if (
        report.experiment_id != matrix.experiment_id
        or report.matrix_sha256 != matrix.content_sha256
    ):
        raise ExperimentError(ExperimentErrorCode.INTEGRITY_MISMATCH)
    return report


def build_experiment_executor(
    selection: ExperimentArmSelection,
    *,
    repository_root: Path,
    split: DatasetSplit,
    openai_api_key: str | SecretStr,
    database_url: str | SecretStr,
    provider_responses_factory: ProviderResponsesFactory | None = None,
    sandbox_factory: RepairSandboxFactory | None = None,
    memory_stores: MemoryStoreBundle | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_cache: LocalEmbeddingCache | None = None,
    vector_store: VectorStore | None = None,
) -> EvaluationTaskExecutor:
    """Bind a selected arm to the common repair executor without reading env overrides."""

    if selection.matrix.prompt_versions != (PROMPT_VERSION,):
        # Loading historical controls is safe; executing them with a changed Prompt is not.
        raise ExperimentError(ExperimentErrorCode.FAIRNESS_VIOLATION)
    settings = selection.settings(
        openai_api_key=openai_api_key,
        database_url=database_url,
    )
    arm = selection.arm
    if arm.memory_enabled:
        expected_snapshot = (
            arm.dev_memory_snapshot_id
            if split is DatasetSplit.DEV
            else arm.holdout_memory_snapshot_id
        )
        if (
            memory_stores is None
            or memory_stores.namespace.experiment_id != selection.matrix.experiment_id
            or memory_stores.namespace.snapshot_id != expected_snapshot
            or memory_stores.namespace.split is not split
            or memory_stores.namespace.access_mode is not MemoryAccessMode.READ_ONLY
        ):
            raise ExperimentError(ExperimentErrorCode.MEMORY_SNAPSHOT_INVALID)
    elif memory_stores is not None:
        raise ExperimentError(ExperimentErrorCode.FAIRNESS_VIOLATION)
    if arm.baseline_config_ref is not None:
        definition = load_baseline_definition(
            repository_root / Path(arm.baseline_config_ref.relative_path)
        )
        return BaselineEvaluationExecutor(
            definition=definition,
            settings=settings,
            sandbox_image_reference=selection.matrix.sandbox_image_reference,
            evaluation_configuration_sha256=selection.configuration_sha256,
            provider_responses_factory=provider_responses_factory,
            sandbox_factory=sandbox_factory,
        )
    return RepairWorkflowEvaluationExecutor(
        settings=settings,
        sandbox_image_reference=selection.matrix.sandbox_image_reference,
        context_mode=arm.context_mode,
        evaluation_configuration_sha256=selection.configuration_sha256,
        provider_responses_factory=provider_responses_factory,
        sandbox_factory=sandbox_factory,
        memory_stores=memory_stores,
        embedding_provider=embedding_provider,
        embedding_cache=embedding_cache,
        vector_store=vector_store,
    )


def build_experiment_preflight(
    matrix: ExperimentMatrix,
    *,
    repository_root: Path,
    checked_at: datetime,
    facts: ExperimentPreflightFacts,
) -> ExperimentPreflightReport:
    """Check immutable inputs and supplied machine facts without making paid calls."""

    if checked_at.tzinfo is None or checked_at.utcoffset() is None:
        raise ExperimentError(ExperimentErrorCode.INVALID_CONFIGURATION)
    checks: dict[str, tuple[ExperimentPreflightCheckStatus, ExperimentErrorCode | None]] = {}

    def record(
        check_id: str,
        passed: bool | None,
        error_code: ExperimentErrorCode,
    ) -> None:
        status = (
            ExperimentPreflightCheckStatus.NOT_RUN
            if passed is None
            else (
                ExperimentPreflightCheckStatus.PASSED
                if passed
                else ExperimentPreflightCheckStatus.BLOCKED
            )
        )
        checks[check_id] = (status, None if passed else error_code)

    record(
        "dependency-lock",
        _file_sha256(repository_root / matrix.dependency_lock_path)
        == matrix.dependency_lock_sha256,
        ExperimentErrorCode.DEPENDENCY_LOCK_MISMATCH,
    )
    dataset_ok = _dataset_matches(matrix, repository_root)
    record("dataset", dataset_ok, ExperimentErrorCode.DATASET_MISMATCH)
    record("fairness", True, ExperimentErrorCode.FAIRNESS_VIOLATION)
    record(
        "gold-boundary",
        not _contains_sensitive_key(
            matrix.model_dump(mode="python"),
            allow_environment_names=True,
        ),
        ExperimentErrorCode.GOLD_BOUNDARY_VIOLATION,
    )
    record(
        "local-dependencies",
        (
            facts.local_dependencies_available
            if facts.local_dependencies_available is not None
            else all(
                importlib.util.find_spec(module) is not None
                for module in ("openai", "psycopg", "sentence_transformers")
            )
        ),
        ExperimentErrorCode.PROVIDER_UNAVAILABLE,
    )
    record(
        "memory-snapshots",
        _memory_snapshots_match(matrix, repository_root),
        ExperimentErrorCode.MEMORY_SNAPSHOT_INVALID,
    )
    record(
        "model-snapshot",
        (
            matrix.shared_settings.values.get("model_provider") == ModelProvider.OPENAI
            and matrix.shared_settings.values.get("model") == FROZEN_OPENAI_MODEL
        ),
        ExperimentErrorCode.MODEL_SNAPSHOT_INVALID,
    )
    record(
        "embedding-configuration",
        (
            matrix.shared_settings.values.get("embedding_provider")
            == EmbeddingProviderKind.JINA_LOCAL
            and matrix.shared_settings.values.get("embedding_model") == FROZEN_EMBEDDING_MODEL
            and matrix.shared_settings.values.get("embedding_revision") == FROZEN_EMBEDDING_REVISION
            and matrix.shared_settings.values.get("embedding_code_revision")
            == FROZEN_EMBEDDING_CODE_REVISION
        ),
        ExperimentErrorCode.EMBEDDING_SNAPSHOT_INVALID,
    )
    record(
        "prompt-version",
        matrix.prompt_versions == (PROMPT_VERSION,),
        ExperimentErrorCode.FAIRNESS_VIOLATION,
    )
    record(
        "source-snapshot",
        facts.source_clean and re.fullmatch(r"[0-9a-f]{40}", facts.source_commit) is not None,
        ExperimentErrorCode.SOURCE_SNAPSHOT_UNCLEAN,
    )
    record(
        "sandbox-image",
        facts.sandbox_image_available,
        ExperimentErrorCode.SANDBOX_IMAGE_UNAVAILABLE,
    )
    record(
        "provider-credential",
        facts.provider_credential_available,
        ExperimentErrorCode.PROVIDER_UNAVAILABLE,
    )
    record(
        "vector-backend",
        facts.vector_backend_available,
        ExperimentErrorCode.VECTOR_BACKEND_UNAVAILABLE,
    )
    record(
        "embedding-snapshot",
        facts.embedding_snapshot_available,
        ExperimentErrorCode.EMBEDDING_SNAPSHOT_INVALID,
    )
    record(
        "output-root",
        facts.output_root_ready,
        ExperimentErrorCode.OUTPUT_CONFLICT,
    )
    items = tuple(
        ExperimentPreflightCheck(
            check_id=check_id,
            status=status,
            error_code=error_code,
            detail=(
                "verified"
                if status is ExperimentPreflightCheckStatus.PASSED
                else ("not_run" if status is ExperimentPreflightCheckStatus.NOT_RUN else "blocked")
            ),
        )
        for check_id, (status, error_code) in sorted(checks.items())
    )
    ready = all(item.status is ExperimentPreflightCheckStatus.PASSED for item in items)
    payload: dict[str, object] = {
        "schema_version": EXPERIMENT_PREFLIGHT_SCHEMA_VERSION,
        "preflight_version": EXPERIMENT_PREFLIGHT_VERSION,
        "experiment_id": matrix.experiment_id,
        "matrix_sha256": matrix.content_sha256,
        "checked_at": checked_at,
        "source_commit": facts.source_commit,
        "status": (ExperimentPreflightStatus.READY if ready else ExperimentPreflightStatus.BLOCKED),
        "ready_for_formal_run": ready,
        "checks": items,
    }
    return ExperimentPreflightReport.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def write_experiment_preflight(path: Path, report: ExperimentPreflightReport) -> None:
    """Create one no-overwrite report with deterministic JSON bytes."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(report.model_dump_json(indent=2))
            stream.write("\n")
    except OSError:
        raise ExperimentError(ExperimentErrorCode.OUTPUT_CONFLICT) from None


def require_ready_preflight(report: ExperimentPreflightReport) -> None:
    """Stop formal execution unless every required check passed."""

    if not report.ready_for_formal_run:
        blocker = next(
            item.error_code
            for item in report.checks
            if item.status is not ExperimentPreflightCheckStatus.PASSED
        )
        assert blocker is not None
        raise ExperimentError(blocker)


def _dataset_matches(matrix: ExperimentMatrix, root: Path) -> bool:
    try:
        bundle = load_frozen_dataset(
            Path(matrix.dataset.manifest_path),
            repository_root=root,
        )
    except DatasetManifestError:
        return False
    if (
        bundle.manifest.dataset_id != matrix.dataset.dataset_id
        or bundle.manifest.content_sha256 != matrix.dataset.manifest_content_sha256
    ):
        return False
    for declared in matrix.dataset.task_sets:
        tasks = tuple(task for task in bundle.evaluation_tasks() if task.split is declared.split)
        if (
            tuple(task.task.task_id for task in tasks) != declared.task_ids
            or evaluation_task_set_sha256(tasks) != declared.task_set_sha256
        ):
            return False
    return True


def _memory_snapshots_match(matrix: ExperimentMatrix, root: Path) -> bool:
    for snapshot in matrix.memory_snapshots:
        try:
            _verify_artifact_ref(root, snapshot.manifest_ref)
            manifest = MemoryNamespaceManifest.model_validate_json(
                _read_contained(
                    root,
                    Path(snapshot.manifest_ref.relative_path),
                    snapshot.manifest_ref.size_bytes,
                )
            )
        except (ExperimentError, UnicodeError, ValidationError):
            return False
        if manifest.namespace != snapshot.namespace:
            return False
    return True


def _validated_root(root: Path) -> Path:
    try:
        if not root.is_absolute() or root.is_symlink():
            raise OSError
        resolved = root.resolve(strict=True)
    except OSError:
        raise ExperimentError(ExperimentErrorCode.INPUT_IO_ERROR) from None
    if resolved != root or not root.is_dir():
        raise ExperimentError(ExperimentErrorCode.INPUT_IO_ERROR)
    return root


def _read_contained(root: Path, path: Path, max_bytes: int) -> bytes:
    candidate = path if path.is_absolute() else root / path
    try:
        if candidate.is_symlink():
            raise OSError
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        content = resolved.read_bytes()
    except (OSError, ValueError):
        raise ExperimentError(ExperimentErrorCode.INPUT_IO_ERROR) from None
    if not content or len(content) > max_bytes:
        raise ExperimentError(ExperimentErrorCode.INPUT_IO_ERROR)
    return content


def _verify_artifact_ref(root: Path, reference: ArtifactRef) -> None:
    content = _read_contained(
        root,
        Path(reference.relative_path),
        max(reference.size_bytes, 1),
    )
    if len(content) != reference.size_bytes or hashlib.sha256(content).hexdigest() != (
        reference.sha256
    ):
        raise ExperimentError(ExperimentErrorCode.INTEGRITY_MISMATCH)


def _verify_file_sha256(root: Path, path: str, expected: str) -> None:
    content = _read_contained(root, Path(path), _MAX_MATRIX_BYTES * 100)
    if hashlib.sha256(content).hexdigest() != expected:
        raise ExperimentError(ExperimentErrorCode.INTEGRITY_MISMATCH)


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "0" * 64


def _reject_sensitive_keys(
    value: object,
    *,
    allow_environment_names: bool = False,
) -> None:
    if _contains_sensitive_key(value, allow_environment_names=allow_environment_names):
        raise ValueError("sensitive or evaluator-only keys are forbidden")


def _contains_sensitive_key(
    value: object,
    *,
    allow_environment_names: bool = False,
) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in _SENSITIVE_KEYS:
                return True
            if _contains_sensitive_key(
                item,
                allow_environment_names=allow_environment_names,
            ):
                return True
    elif isinstance(value, (list, tuple)):
        for item in value:
            if (
                allow_environment_names
                and isinstance(item, str)
                and item in ALLOWED_SECRET_ENVIRONMENT
            ):
                continue
            if _contains_sensitive_key(
                item,
                allow_environment_names=allow_environment_names,
            ):
                return True
    elif isinstance(value, str) and not allow_environment_names:
        return value in ALLOWED_SECRET_ENVIRONMENT
    return False


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ALLOWED_SECRET_ENVIRONMENT",
    "EXPERIMENT_MATRIX_SCHEMA_VERSION",
    "EXPERIMENT_MATRIX_VERSION",
    "EXPERIMENT_PREFLIGHT_SCHEMA_VERSION",
    "EXPERIMENT_PREFLIGHT_VERSION",
    "FROZEN_EMBEDDING_CODE_REVISION",
    "FROZEN_EMBEDDING_MODEL",
    "FROZEN_EMBEDDING_REVISION",
    "FROZEN_OPENAI_MODEL",
    "FROZEN_PROMPT_VERSION",
    "ExperimentArmDefinition",
    "ExperimentArmSelection",
    "ExperimentDatasetSnapshot",
    "ExperimentError",
    "ExperimentMatrix",
    "ExperimentMemorySnapshot",
    "ExperimentPreflightCheck",
    "ExperimentPreflightFacts",
    "ExperimentPreflightReport",
    "ExperimentSharedSettings",
    "ExperimentTaskSet",
    "build_experiment_executor",
    "build_experiment_preflight",
    "load_experiment_matrix",
    "load_experiment_preflight",
    "require_ready_preflight",
    "select_experiment_arm",
    "write_experiment_preflight",
]
