"""Dev-only P7 calibration, paid-run authorization, and formal freeze contracts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Literal, Self
from uuid import UUID, uuid5

from pydantic import AwareDatetime, Field, JsonValue, SecretStr, ValidationError, model_validator
from pydantic_core import to_jsonable_python

from repopilot.enums import (
    CalibrationErrorCode,
    DatasetSplit,
    EvaluationAttemptStatus,
    ExperimentPreflightCheckStatus,
    TerminationReason,
)
from repopilot.evals.experiments import (
    ExperimentArmSelection,
    ExperimentMatrix,
    build_experiment_executor,
    select_experiment_arm,
)
from repopilot.evals.repair_fixtures import RepairFixtureError, load_repair_fixture_manifest
from repopilot.evals.runner import (
    EvaluationRunReport,
    EvaluationTaskExecutor,
    build_evaluation_run_manifest,
    evaluation_task_set_sha256,
    run_evaluation,
)
from repopilot.schemas import (
    ArtifactRef,
    EvaluationTaskSpec,
    Identifier,
    Sha256,
    StrictModel,
    TaskId,
)

CALIBRATION_SCHEMA_VERSION: Literal[1] = 1
CALIBRATION_AUTHORIZATION_VERSION: Literal["calibration-authorization-v1"] = (
    "calibration-authorization-v1"
)
CALIBRATION_PLAN_VERSION: Literal["dev-calibration-plan-v1"] = "dev-calibration-plan-v1"
CALIBRATION_REPORT_VERSION: Literal["dev-calibration-report-v1"] = "dev-calibration-report-v1"
FORMAL_FREEZE_VERSION: Literal["formal-freeze-v1"] = "formal-freeze-v1"
MAX_APPROVED_CALIBRATION_COST_USD = Decimal("8.00")
DEV_FIXTURE_MANIFEST_PATH = Path("tests/fixtures/p5_repair_tasks/manifest.json")
_RUN_NAMESPACE = UUID("491b8d96-46cf-4bdd-84df-d2862d9ed793")
_MAX_ARTIFACT_BYTES = 2_000_000

CALIBRATION_ERROR_MESSAGES: Mapping[CalibrationErrorCode, str] = {
    CalibrationErrorCode.INVALID_CONFIGURATION: "calibration configuration is invalid",
    CalibrationErrorCode.UNSUPPORTED_VERSION: "calibration schema version is unsupported",
    CalibrationErrorCode.INTEGRITY_MISMATCH: "calibration integrity check failed",
    CalibrationErrorCode.DEV_TASK_SET_MISMATCH: "calibration dev task set does not match",
    CalibrationErrorCode.HOLDOUT_ACCESS_DENIED: "calibration cannot access holdout tasks",
    CalibrationErrorCode.BUDGET_APPROVAL_REQUIRED: "paid calibration requires explicit approval",
    CalibrationErrorCode.BUDGET_APPROVAL_MISMATCH: "calibration approval does not match the plan",
    CalibrationErrorCode.BUDGET_LIMIT_EXCEEDED: "calibration reported cost exceeded approval",
    CalibrationErrorCode.PROVIDER_CREDENTIAL_REQUIRED: "paid calibration credential is unavailable",
    CalibrationErrorCode.PREFLIGHT_BLOCKED: "calibration preflight is blocked",
    CalibrationErrorCode.SOURCE_SNAPSHOT_UNCLEAN: "formal freeze requires a clean source snapshot",
    CalibrationErrorCode.RUN_INCOMPLETE: "calibration run set is incomplete",
    CalibrationErrorCode.OUTPUT_CONFLICT: "calibration output already exists",
    CalibrationErrorCode.PATH_UNSAFE: "calibration path is unsafe",
    CalibrationErrorCode.INPUT_IO_ERROR: "calibration input is unavailable",
}


class CalibrationError(RuntimeError):
    """Safe calibration failure without secrets, task text, or host paths."""

    def __init__(self, code: CalibrationErrorCode) -> None:
        super().__init__(CALIBRATION_ERROR_MESSAGES[code])
        self.code = code


class PaidCalibrationAuthorization(StrictModel):
    """User approval bound to one matrix and its exact worst-case paid ceiling."""

    schema_version: Literal[1] = CALIBRATION_SCHEMA_VERSION
    authorization_version: Literal["calibration-authorization-v1"] = (
        CALIBRATION_AUTHORIZATION_VERSION
    )
    authorization_id: Identifier
    experiment_id: Identifier
    matrix_sha256: Sha256
    approved_by: Identifier
    approved_at: AwareDatetime
    paid_execution_approved: Literal[True]
    max_total_cost_usd: Decimal = Field(gt=0, max_digits=6, decimal_places=2)
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_authorization(self) -> Self:
        if self.max_total_cost_usd > MAX_APPROVED_CALIBRATION_COST_USD:
            raise ValueError("calibration approval exceeds the P7-S5 ceiling")
        _validate_content_digest(self)
        return self


class DevCalibrationRunSlot(StrictModel):
    """One immutable default-arm Runner invocation on the dev task set."""

    arm_id: Identifier
    run_id: UUID
    configuration_sha256: Sha256
    output_path: str = Field(pattern=r"^results/p7-dev/[A-Za-z0-9._/-]{1,400}$")


class CalibrationChangeRecord(StrictModel):
    """One dev-discovered change that is explicitly independent of holdout evidence."""

    change_id: Identifier
    category: Literal["dependency_compatibility", "embedding_identity"]
    before: JsonValue
    after: JsonValue
    reason: str = Field(min_length=1, max_length=1_000)
    evidence: str = Field(min_length=1, max_length=1_000)
    holdout_informed: Literal[False] = False


class CalibrationChangeLog(StrictModel):
    """Version transition audit retained before any paid dev or holdout execution."""

    schema_version: Literal[1] = CALIBRATION_SCHEMA_VERSION
    change_log_version: Literal["calibration-change-log-v1"] = "calibration-change-log-v1"
    from_matrix_version: Literal["experiment-matrix-v1"]
    from_matrix_sha256: Sha256
    to_matrix_version: Literal["experiment-matrix-v2"]
    to_matrix_sha256: Sha256
    recorded_at: AwareDatetime
    holdout_accessed: Literal[False] = False
    changes: tuple[CalibrationChangeRecord, ...]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_change_log(self) -> Self:
        change_ids = tuple(item.change_id for item in self.changes)
        if change_ids != tuple(sorted(change_ids)) or len(set(change_ids)) != len(change_ids):
            raise ValueError("calibration changes must be sorted and unique")
        _validate_content_digest(self)
        return self


class DevCalibrationPreflightCheck(StrictModel):
    """One dev-only readiness decision with no holdout input."""

    check_id: Identifier
    status: ExperimentPreflightCheckStatus
    error_code: CalibrationErrorCode | None = None

    @model_validator(mode="after")
    def validate_check(self) -> Self:
        if (self.status is ExperimentPreflightCheckStatus.PASSED) != (self.error_code is None):
            raise ValueError("calibration preflight outcome is inconsistent")
        return self


class DevCalibrationPreflight(StrictModel):
    """Readiness report that proves the holdout source artifact was never opened."""

    schema_version: Literal[1] = CALIBRATION_SCHEMA_VERSION
    preflight_version: Literal["dev-calibration-preflight-v1"] = "dev-calibration-preflight-v1"
    experiment_id: Identifier
    matrix_sha256: Sha256
    checked_at: AwareDatetime
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_dirty: bool
    ready_for_paid_dev: bool
    holdout_accessed: Literal[False] = False
    checks: tuple[DevCalibrationPreflightCheck, ...]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_preflight(self) -> Self:
        check_ids = tuple(item.check_id for item in self.checks)
        if check_ids != tuple(sorted(check_ids)) or len(set(check_ids)) != len(check_ids):
            raise ValueError("calibration preflight checks must be sorted and unique")
        required = tuple(item for item in self.checks if item.check_id != "source-snapshot")
        ready = all(item.status is ExperimentPreflightCheckStatus.PASSED for item in required)
        if ready != self.ready_for_paid_dev:
            raise ValueError("calibration preflight readiness does not match checks")
        _validate_content_digest(self)
        return self


@dataclass(frozen=True)
class DevCalibrationPreflightFacts:
    """Machine facts supplied without loading the frozen holdout dataset."""

    source_commit: str
    source_clean: bool
    local_dependencies_available: bool
    sandbox_image_available: bool
    provider_credential_available: bool
    vector_backend_available: bool
    embedding_snapshot_available: bool
    output_root_ready: bool


class DevCalibrationPlan(StrictModel):
    """Paid execution declaration that contains no Provider or database secret."""

    schema_version: Literal[1] = CALIBRATION_SCHEMA_VERSION
    plan_version: Literal["dev-calibration-plan-v1"] = CALIBRATION_PLAN_VERSION
    calibration_id: Identifier
    experiment_id: Identifier
    matrix_sha256: Sha256
    authorization_sha256: Sha256
    created_at: AwareDatetime
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_dirty: bool
    split: Literal[DatasetSplit.DEV] = DatasetSplit.DEV
    task_ids: tuple[TaskId, ...]
    task_set_sha256: Sha256
    repetitions: int = Field(ge=1, le=10, strict=True)
    per_task_cost_limit_usd: Decimal = Field(gt=0, max_digits=6, decimal_places=2)
    theoretical_max_cost_usd: Decimal = Field(gt=0, max_digits=6, decimal_places=2)
    carried_reported_cost_usd: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        max_digits=16,
        decimal_places=12,
    )
    holdout_accessed: Literal[False] = False
    run_slots: tuple[DevCalibrationRunSlot, ...]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.task_ids != tuple(sorted(self.task_ids)) or not self.task_ids:
            raise ValueError("calibration tasks must be sorted and non-empty")
        arm_ids = tuple(slot.arm_id for slot in self.run_slots)
        if arm_ids != tuple(sorted(arm_ids)) or len(set(arm_ids)) != len(arm_ids):
            raise ValueError("calibration run slots must be sorted and unique")
        expected_cost = (
            self.per_task_cost_limit_usd
            * len(self.task_ids)
            * len(self.run_slots)
            * self.repetitions
        )
        if self.theoretical_max_cost_usd != expected_cost:
            raise ValueError("calibration theoretical cost does not match its slots")
        if self.carried_reported_cost_usd >= MAX_APPROVED_CALIBRATION_COST_USD:
            raise ValueError("carried calibration cost already exhausts the approval")
        _validate_content_digest(self)
        return self


class DevCalibrationRunRecord(StrictModel):
    """One arm's raw Runner location and non-success-rate operational totals."""

    arm_id: Identifier
    run_id: UUID
    run_manifest_ref: ArtifactRef
    total_attempts: int = Field(ge=1, le=100)
    completed_attempts: int = Field(ge=0, le=100)
    failed_attempts: int = Field(ge=0, le=100)
    estimated_cost_usd: Decimal = Field(ge=0, max_digits=16, decimal_places=12)
    termination_reasons: tuple[TerminationReason, ...]
    runner_failure_codes: tuple[str, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.completed_attempts + self.failed_attempts != self.total_attempts:
            raise ValueError("calibration arm must retain every terminal attempt")
        if len(self.termination_reasons) != self.completed_attempts:
            raise ValueError("each completed attempt needs one termination reason")
        if len(self.runner_failure_codes) != self.failed_attempts:
            raise ValueError("each failed attempt needs one Runner failure code")
        return self


class DevCalibrationReport(StrictModel):
    """Integrity-protected index of raw dev runs; it deliberately has no success rate."""

    schema_version: Literal[1] = CALIBRATION_SCHEMA_VERSION
    report_version: Literal["dev-calibration-report-v1"] = CALIBRATION_REPORT_VERSION
    calibration_id: Identifier
    experiment_id: Identifier
    matrix_sha256: Sha256
    plan_sha256: Sha256
    completed_at: AwareDatetime
    split: Literal[DatasetSplit.DEV] = DatasetSplit.DEV
    task_ids: tuple[TaskId, ...]
    holdout_accessed: Literal[False] = False
    runs: tuple[DevCalibrationRunRecord, ...]
    total_estimated_cost_usd: Decimal = Field(ge=0, max_digits=16, decimal_places=12)
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        arm_ids = tuple(item.arm_id for item in self.runs)
        if arm_ids != tuple(sorted(arm_ids)) or len(set(arm_ids)) != len(arm_ids):
            raise ValueError("calibration report arms must be sorted and unique")
        if self.total_estimated_cost_usd != sum(
            (item.estimated_cost_usd for item in self.runs),
            Decimal("0"),
        ):
            raise ValueError("calibration report cost does not match raw runs")
        _validate_content_digest(self)
        return self


class FormalFreezeMarker(StrictModel):
    """Final clean-commit identity consumed by the P7-S6 holdout runner."""

    schema_version: Literal[1] = CALIBRATION_SCHEMA_VERSION
    freeze_version: Literal["formal-freeze-v1"] = FORMAL_FREEZE_VERSION
    experiment_id: Identifier
    matrix_version: Literal["experiment-matrix-v2"]
    matrix_sha256: Sha256
    dependency_lock_sha256: Sha256
    dataset_manifest_sha256: Sha256
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_clean: Literal[True]
    prompt_versions: tuple[str, ...]
    model: str
    embedding_model: str
    embedding_revision: str
    embedding_code_revision: str
    sandbox_image_reference: str
    arm_configuration_sha256: dict[str, Sha256]
    calibration_report_ref: ArtifactRef
    holdout_accessed: Literal[False] = False
    frozen_at: AwareDatetime
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_marker(self) -> Self:
        if tuple(self.arm_configuration_sha256) != tuple(sorted(self.arm_configuration_sha256)):
            raise ValueError("freeze arm identities must be sorted")
        _validate_content_digest(self)
        return self


ExecutorFactory = Callable[[ExperimentArmSelection], EvaluationTaskExecutor]


def build_paid_calibration_authorization(
    *,
    matrix: ExperimentMatrix,
    authorization_id: str,
    approved_by: str,
    approved_at: datetime,
    max_total_cost_usd: Decimal,
) -> PaidCalibrationAuthorization:
    """Build explicit approval; calling this function is the auditable consent event."""

    payload: dict[str, object] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "authorization_version": CALIBRATION_AUTHORIZATION_VERSION,
        "authorization_id": authorization_id,
        "experiment_id": matrix.experiment_id,
        "matrix_sha256": matrix.content_sha256,
        "approved_by": approved_by,
        "approved_at": approved_at,
        "paid_execution_approved": True,
        "max_total_cost_usd": max_total_cost_usd,
    }
    return PaidCalibrationAuthorization.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def load_dev_calibration_tasks(
    matrix: ExperimentMatrix,
    *,
    repository_root: Path,
) -> tuple[EvaluationTaskSpec, ...]:
    """Load only the checked-in dev fixtures; never open the holdout source-row artifact."""

    root = _validated_root(repository_root)
    try:
        fixture_manifest = load_repair_fixture_manifest(root / DEV_FIXTURE_MANIFEST_PATH)
    except RepairFixtureError:
        raise CalibrationError(CalibrationErrorCode.INPUT_IO_ERROR) from None
    tasks = tuple(case.evaluation_task for case in fixture_manifest.cases)
    declared = next(
        (item for item in matrix.dataset.task_sets if item.split is DatasetSplit.DEV),
        None,
    )
    if declared is None:
        raise CalibrationError(CalibrationErrorCode.DEV_TASK_SET_MISMATCH)
    if (
        any(item.split is not DatasetSplit.DEV for item in tasks)
        or tuple(item.task.task_id for item in tasks) != declared.task_ids
        or evaluation_task_set_sha256(tasks) != declared.task_set_sha256
    ):
        raise CalibrationError(CalibrationErrorCode.DEV_TASK_SET_MISMATCH)
    return tasks


def build_dev_calibration_plan(
    *,
    matrix: ExperimentMatrix,
    authorization: PaidCalibrationAuthorization | None,
    tasks: tuple[EvaluationTaskSpec, ...],
    created_at: datetime,
    source_commit: str,
    source_dirty: bool,
    carried_reported_cost_usd: Decimal = Decimal("0"),
) -> DevCalibrationPlan:
    """Bind four default arms and two dev tasks to the exact USD 8 authorization."""

    if authorization is None:
        raise CalibrationError(CalibrationErrorCode.BUDGET_APPROVAL_REQUIRED)
    if (
        authorization.experiment_id != matrix.experiment_id
        or authorization.matrix_sha256 != matrix.content_sha256
    ):
        raise CalibrationError(CalibrationErrorCode.BUDGET_APPROVAL_MISMATCH)
    evaluation_tasks = _validate_dev_tasks(matrix, tasks)
    arms = tuple(
        sorted(
            (arm for arm in matrix.arms if arm.enabled_by_default),
            key=lambda item: item.arm_id,
        )
    )
    per_task_cost = Decimal(str(matrix.shared_settings.values["max_cost_usd"]))
    theoretical_cost = per_task_cost * len(evaluation_tasks) * len(arms) * matrix.dev_repetitions
    if (
        theoretical_cost != MAX_APPROVED_CALIBRATION_COST_USD
        or authorization.max_total_cost_usd != theoretical_cost
    ):
        raise CalibrationError(CalibrationErrorCode.BUDGET_APPROVAL_MISMATCH)
    calibration_id = f"p7-dev-{matrix.content_sha256[:12]}-{source_commit[:12]}"
    slots = tuple(
        DevCalibrationRunSlot(
            arm_id=arm.arm_id,
            run_id=uuid5(_RUN_NAMESPACE, f"{matrix.content_sha256}:{arm.arm_id}:dev"),
            configuration_sha256=ExperimentArmSelection(
                matrix=matrix,
                arm=arm,
            ).configuration_sha256,
            output_path=f"results/p7-dev/{calibration_id}/{arm.arm_id}",
        )
        for arm in arms
    )
    payload: dict[str, object] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "plan_version": CALIBRATION_PLAN_VERSION,
        "calibration_id": calibration_id,
        "experiment_id": matrix.experiment_id,
        "matrix_sha256": matrix.content_sha256,
        "authorization_sha256": authorization.content_sha256,
        "created_at": created_at,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "split": DatasetSplit.DEV,
        "task_ids": tuple(item.task.task_id for item in evaluation_tasks),
        "task_set_sha256": evaluation_task_set_sha256(evaluation_tasks),
        "repetitions": matrix.dev_repetitions,
        "per_task_cost_limit_usd": per_task_cost,
        "theoretical_max_cost_usd": theoretical_cost,
        "carried_reported_cost_usd": carried_reported_cost_usd,
        "holdout_accessed": False,
        "run_slots": slots,
    }
    return DevCalibrationPlan.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def build_dev_calibration_preflight(
    *,
    matrix: ExperimentMatrix,
    tasks: tuple[EvaluationTaskSpec, ...],
    repository_root: Path,
    checked_at: datetime,
    facts: DevCalibrationPreflightFacts,
) -> DevCalibrationPreflight:
    """Check dev runtime readiness without loading dataset source rows or holdout gold."""

    root = _validated_root(repository_root)
    _validate_dev_tasks(matrix, tasks)
    checks: dict[str, tuple[bool, CalibrationErrorCode]] = {
        "dependency-lock": (
            _file_sha256(root / matrix.dependency_lock_path) == matrix.dependency_lock_sha256,
            CalibrationErrorCode.INTEGRITY_MISMATCH,
        ),
        "dev-task-set": (True, CalibrationErrorCode.DEV_TASK_SET_MISMATCH),
        "embedding-snapshot": (
            facts.embedding_snapshot_available,
            CalibrationErrorCode.PREFLIGHT_BLOCKED,
        ),
        "local-dependencies": (
            facts.local_dependencies_available,
            CalibrationErrorCode.PREFLIGHT_BLOCKED,
        ),
        "output-root": (
            facts.output_root_ready,
            CalibrationErrorCode.OUTPUT_CONFLICT,
        ),
        "provider-credential": (
            facts.provider_credential_available,
            CalibrationErrorCode.PROVIDER_CREDENTIAL_REQUIRED,
        ),
        "sandbox-image": (
            facts.sandbox_image_available,
            CalibrationErrorCode.PREFLIGHT_BLOCKED,
        ),
        "source-snapshot": (
            facts.source_clean,
            CalibrationErrorCode.SOURCE_SNAPSHOT_UNCLEAN,
        ),
        "vector-backend": (
            facts.vector_backend_available,
            CalibrationErrorCode.PREFLIGHT_BLOCKED,
        ),
    }
    items = tuple(
        DevCalibrationPreflightCheck(
            check_id=check_id,
            status=(
                ExperimentPreflightCheckStatus.PASSED
                if passed
                else ExperimentPreflightCheckStatus.BLOCKED
            ),
            error_code=None if passed else error,
        )
        for check_id, (passed, error) in sorted(checks.items())
    )
    ready = all(
        item.status is ExperimentPreflightCheckStatus.PASSED
        for item in items
        if item.check_id != "source-snapshot"
    )
    payload: dict[str, object] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "preflight_version": "dev-calibration-preflight-v1",
        "experiment_id": matrix.experiment_id,
        "matrix_sha256": matrix.content_sha256,
        "checked_at": checked_at,
        "source_commit": facts.source_commit,
        "source_dirty": not facts.source_clean,
        "ready_for_paid_dev": ready,
        "holdout_accessed": False,
        "checks": items,
    }
    return DevCalibrationPreflight.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def require_dev_calibration_preflight(
    report: DevCalibrationPreflight,
    *,
    matrix: ExperimentMatrix,
) -> None:
    """Permit dirty-source dev debugging, but require every runtime dependency."""

    if (
        report.experiment_id != matrix.experiment_id
        or report.matrix_sha256 != matrix.content_sha256
    ):
        raise CalibrationError(CalibrationErrorCode.INTEGRITY_MISMATCH)
    if not report.ready_for_paid_dev:
        raise CalibrationError(CalibrationErrorCode.PREFLIGHT_BLOCKED)


def run_dev_calibration(
    *,
    plan: DevCalibrationPlan,
    matrix: ExperimentMatrix,
    authorization: PaidCalibrationAuthorization,
    preflight: DevCalibrationPreflight,
    tasks: tuple[EvaluationTaskSpec, ...],
    repository_root: Path,
    openai_api_key: str | SecretStr,
    database_url: str | SecretStr,
    completed_at: datetime | None = None,
    executor_factory: ExecutorFactory | None = None,
) -> DevCalibrationReport:
    """Run only the authorized default-arm dev matrix through the append-only Runner."""

    root = _validated_root(repository_root)
    secret = (
        openai_api_key.get_secret_value()
        if isinstance(openai_api_key, SecretStr)
        else openai_api_key
    )
    if not secret.strip():
        raise CalibrationError(CalibrationErrorCode.PROVIDER_CREDENTIAL_REQUIRED)
    if plan.authorization_sha256 != authorization.content_sha256:
        raise CalibrationError(CalibrationErrorCode.BUDGET_APPROVAL_MISMATCH)
    expected = build_dev_calibration_plan(
        matrix=matrix,
        authorization=authorization,
        tasks=tasks,
        created_at=plan.created_at,
        source_commit=plan.source_commit,
        source_dirty=plan.source_dirty,
        carried_reported_cost_usd=plan.carried_reported_cost_usd,
    )
    if expected != plan:
        raise CalibrationError(CalibrationErrorCode.INTEGRITY_MISMATCH)
    require_dev_calibration_preflight(preflight, matrix=matrix)
    if preflight.source_commit != plan.source_commit or preflight.source_dirty != plan.source_dirty:
        raise CalibrationError(CalibrationErrorCode.INTEGRITY_MISMATCH)
    evaluation_tasks = _validate_dev_tasks(matrix, tasks)
    records: list[DevCalibrationRunRecord] = []
    for slot in plan.run_slots:
        completed_cost = sum(
            (item.estimated_cost_usd for item in records),
            plan.carried_reported_cost_usd,
        )
        if completed_cost >= authorization.max_total_cost_usd:
            raise CalibrationError(CalibrationErrorCode.BUDGET_LIMIT_EXCEEDED)
        selection = select_experiment_arm(matrix, slot.arm_id)
        executor = (
            executor_factory(selection)
            if executor_factory is not None
            else build_experiment_executor(
                selection,
                repository_root=root,
                split=DatasetSplit.DEV,
                openai_api_key=openai_api_key,
                database_url=database_url,
            )
        )
        settings = selection.settings(
            openai_api_key=openai_api_key,
            database_url=database_url,
        )
        manifest = build_evaluation_run_manifest(
            experiment_id=matrix.experiment_id,
            run_id=slot.run_id,
            created_at=plan.created_at,
            source_commit=plan.source_commit,
            dependency_lock_sha256=matrix.dependency_lock_sha256,
            dataset_manifest_sha256=matrix.dataset.manifest_content_sha256,
            configuration=selection.configuration_snapshot,
            split=DatasetSplit.DEV,
            arm_id=slot.arm_id,
            tasks=evaluation_tasks,
            memory_namespace=None,
            model_provider=settings.model_provider,
            model=settings.model,
            prompt_versions=matrix.prompt_versions,
            sandbox_image_reference=matrix.sandbox_image_reference,
            repetitions=matrix.dev_repetitions,
        )
        if manifest.configuration_sha256 != slot.configuration_sha256:
            raise CalibrationError(CalibrationErrorCode.INTEGRITY_MISMATCH)
        output = root / slot.output_path
        _ensure_output_parent(root, output.parent)
        report = run_evaluation(
            manifest=manifest,
            tasks=evaluation_tasks,
            output_dir=output,
            executor=executor,
            before_attempt=partial(
                _guard_attempt_budget,
                completed_cost=completed_cost,
                approved_cost=authorization.max_total_cost_usd,
            ),
        )
        records.append(_run_record(root, slot, report))
        reported_cost = sum(
            (item.estimated_cost_usd for item in records),
            plan.carried_reported_cost_usd,
        )
        if reported_cost > authorization.max_total_cost_usd:
            raise CalibrationError(CalibrationErrorCode.BUDGET_LIMIT_EXCEEDED)
    runs = tuple(sorted(records, key=lambda item: item.arm_id))
    payload: dict[str, object] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "report_version": CALIBRATION_REPORT_VERSION,
        "calibration_id": plan.calibration_id,
        "experiment_id": matrix.experiment_id,
        "matrix_sha256": matrix.content_sha256,
        "plan_sha256": plan.content_sha256,
        "completed_at": completed_at or datetime.now(UTC),
        "split": DatasetSplit.DEV,
        "task_ids": plan.task_ids,
        "holdout_accessed": False,
        "runs": runs,
        "total_estimated_cost_usd": sum((item.estimated_cost_usd for item in runs), Decimal("0")),
    }
    return DevCalibrationReport.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def build_formal_freeze_marker(
    *,
    matrix: ExperimentMatrix,
    plan: DevCalibrationPlan,
    report: DevCalibrationReport,
    calibration_report_ref: ArtifactRef,
    source_commit: str,
    source_clean: bool,
    frozen_at: datetime,
) -> FormalFreezeMarker:
    """Freeze only a complete dev run set at the exact clean source commit."""

    if not source_clean or plan.source_dirty or source_commit != plan.source_commit:
        raise CalibrationError(CalibrationErrorCode.SOURCE_SNAPSHOT_UNCLEAN)
    if (
        report.plan_sha256 != plan.content_sha256
        or report.matrix_sha256 != matrix.content_sha256
        or tuple(item.arm_id for item in report.runs)
        != tuple(slot.arm_id for slot in plan.run_slots)
        or any(
            item.total_attempts != item.completed_attempts + item.failed_attempts
            for item in report.runs
        )
    ):
        raise CalibrationError(CalibrationErrorCode.RUN_INCOMPLETE)
    expected_report = f"{report.model_dump_json(indent=2)}\n".encode()
    if (
        calibration_report_ref.sha256 != hashlib.sha256(expected_report).hexdigest()
        or calibration_report_ref.size_bytes != len(expected_report)
        or calibration_report_ref.media_type != "application/json"
    ):
        raise CalibrationError(CalibrationErrorCode.INTEGRITY_MISMATCH)
    payload: dict[str, object] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "freeze_version": FORMAL_FREEZE_VERSION,
        "experiment_id": matrix.experiment_id,
        "matrix_version": matrix.matrix_version,
        "matrix_sha256": matrix.content_sha256,
        "dependency_lock_sha256": matrix.dependency_lock_sha256,
        "dataset_manifest_sha256": matrix.dataset.manifest_content_sha256,
        "source_commit": source_commit,
        "source_clean": True,
        "prompt_versions": matrix.prompt_versions,
        "model": matrix.shared_settings.values["model"],
        "embedding_model": matrix.shared_settings.values["embedding_model"],
        "embedding_revision": matrix.shared_settings.values["embedding_revision"],
        "embedding_code_revision": matrix.shared_settings.values["embedding_code_revision"],
        "sandbox_image_reference": matrix.sandbox_image_reference,
        "arm_configuration_sha256": {
            slot.arm_id: slot.configuration_sha256 for slot in plan.run_slots
        },
        "calibration_report_ref": calibration_report_ref,
        "holdout_accessed": False,
        "frozen_at": frozen_at,
    }
    return FormalFreezeMarker.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def write_calibration_artifact(path: Path, value: StrictModel) -> None:
    """Persist one private JSON artifact without overwriting history."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value.model_dump_json(indent=2))
            stream.write("\n")
    except OSError:
        raise CalibrationError(CalibrationErrorCode.OUTPUT_CONFLICT) from None


def _validate_dev_tasks(
    matrix: ExperimentMatrix,
    tasks: tuple[EvaluationTaskSpec, ...],
) -> tuple[EvaluationTaskSpec, ...]:
    try:
        validated = tuple(
            EvaluationTaskSpec.model_validate(item.model_dump(mode="python")) for item in tasks
        )
        task_ids = tuple(item.task.task_id for item in validated)
        splits = tuple(item.split for item in validated)
        digest = evaluation_task_set_sha256(validated)
    except (AttributeError, TypeError, ValidationError):
        raise CalibrationError(CalibrationErrorCode.DEV_TASK_SET_MISMATCH) from None
    if any(split is not DatasetSplit.DEV for split in splits):
        raise CalibrationError(CalibrationErrorCode.HOLDOUT_ACCESS_DENIED)
    declared = next(item for item in matrix.dataset.task_sets if item.split is DatasetSplit.DEV)
    if task_ids != declared.task_ids or digest != declared.task_set_sha256:
        raise CalibrationError(CalibrationErrorCode.DEV_TASK_SET_MISMATCH)
    return validated


def _run_record(
    root: Path,
    slot: DevCalibrationRunSlot,
    report: EvaluationRunReport,
) -> DevCalibrationRunRecord:
    completed = tuple(
        item for item in report.attempts if item.status is EvaluationAttemptStatus.COMPLETED
    )
    failed = tuple(
        item for item in report.attempts if item.status is EvaluationAttemptStatus.FAILED
    )
    if report.pending_attempts or len(completed) + len(failed) != report.total_attempts:
        raise CalibrationError(CalibrationErrorCode.RUN_INCOMPLETE)
    return DevCalibrationRunRecord(
        arm_id=slot.arm_id,
        run_id=slot.run_id,
        run_manifest_ref=_artifact_ref(root, root / slot.output_path / "manifest.json"),
        total_attempts=report.total_attempts,
        completed_attempts=len(completed),
        failed_attempts=len(failed),
        estimated_cost_usd=sum(
            (item.evidence.usage.estimated_cost_usd for item in completed if item.evidence),
            Decimal("0"),
        ),
        termination_reasons=tuple(
            item.evidence.termination_reason for item in completed if item.evidence
        ),
        runner_failure_codes=tuple(item.failure_code.value for item in failed if item.failure_code),
    )


def _guard_attempt_budget(
    report: EvaluationRunReport,
    *,
    completed_cost: Decimal,
    approved_cost: Decimal,
) -> None:
    current_cost = sum(
        (
            item.evidence.usage.estimated_cost_usd
            for item in report.attempts
            if item.status is EvaluationAttemptStatus.COMPLETED and item.evidence is not None
        ),
        completed_cost,
    )
    if current_cost >= approved_cost:
        raise CalibrationError(CalibrationErrorCode.BUDGET_LIMIT_EXCEEDED)


def _artifact_ref(root: Path, path: Path) -> ArtifactRef:
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
        content = resolved.read_bytes()
    except (OSError, ValueError):
        raise CalibrationError(CalibrationErrorCode.PATH_UNSAFE) from None
    if not content or len(content) > _MAX_ARTIFACT_BYTES:
        raise CalibrationError(CalibrationErrorCode.INTEGRITY_MISMATCH)
    return ArtifactRef(
        relative_path=relative,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        media_type="application/json",
    )


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "0" * 64


def _ensure_output_parent(root: Path, parent: Path) -> None:
    try:
        relative = parent.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            if current.exists() or current.is_symlink():
                if current.is_symlink() or not current.is_dir():
                    raise OSError
            else:
                current.mkdir(mode=0o700)
    except (OSError, ValueError):
        raise CalibrationError(CalibrationErrorCode.PATH_UNSAFE) from None


def _validated_root(root: Path) -> Path:
    try:
        if not root.is_absolute() or root.is_symlink():
            raise OSError
        resolved = root.resolve(strict=True)
    except OSError:
        raise CalibrationError(CalibrationErrorCode.PATH_UNSAFE) from None
    if resolved != root or not root.is_dir():
        raise CalibrationError(CalibrationErrorCode.PATH_UNSAFE)
    return root


def _validate_content_digest(value: StrictModel) -> None:
    payload = value.model_dump(mode="python", exclude={"content_sha256"})
    if _canonical_sha256(payload) != value.content_sha256:  # type: ignore[attr-defined]
        raise ValueError("calibration content digest does not match")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CALIBRATION_AUTHORIZATION_VERSION",
    "CALIBRATION_PLAN_VERSION",
    "CALIBRATION_REPORT_VERSION",
    "CALIBRATION_SCHEMA_VERSION",
    "DEV_FIXTURE_MANIFEST_PATH",
    "FORMAL_FREEZE_VERSION",
    "MAX_APPROVED_CALIBRATION_COST_USD",
    "CalibrationChangeLog",
    "CalibrationChangeRecord",
    "CalibrationError",
    "DevCalibrationPlan",
    "DevCalibrationPreflight",
    "DevCalibrationPreflightCheck",
    "DevCalibrationPreflightFacts",
    "DevCalibrationReport",
    "DevCalibrationRunRecord",
    "DevCalibrationRunSlot",
    "FormalFreezeMarker",
    "PaidCalibrationAuthorization",
    "build_dev_calibration_plan",
    "build_dev_calibration_preflight",
    "build_formal_freeze_marker",
    "build_paid_calibration_authorization",
    "load_dev_calibration_tasks",
    "require_dev_calibration_preflight",
    "run_dev_calibration",
    "write_calibration_artifact",
]
