"""Run the authorized P7-S6 32-slot holdout without loading evaluator Gold.

The default command is read-only. Paid execution requires ``--execute`` and the exact
approved plan SHA-256. Every task is persisted through the P7-S1 append-only Runner; a
restart reconstructs progress from manifests and events rather than trusting a mutable
checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum, unique
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, ValidationError, model_validator
from pydantic_core import to_jsonable_python

from repopilot.embeddings import (
    EmbeddingProvider,
    LocalEmbeddingCache,
    embedding_cache_from_settings,
    embedding_provider_from_settings,
)
from repopilot.enums import (
    DatasetSplit,
    EvaluationAttemptStatus,
    EvaluationRunnerErrorCode,
    RunStatus,
    TerminationReason,
)
from repopilot.evals import load_experiment_matrix, select_experiment_arm
from repopilot.evals.baselines import BaselineEvaluationExecutor, load_baseline_definition
from repopilot.evals.runner import (
    EvaluationRunManifest,
    EvaluationRunnerError,
    EvaluationRunReport,
    EvaluationTaskExecutor,
    RepairWorkflowEvaluationExecutor,
    build_evaluation_run_manifest,
    inspect_evaluation_run,
    run_evaluation,
)
from repopilot.sandbox import DockerCliClient, DockerClientError
from repopilot.schemas import (
    ArtifactRef,
    EvaluationTaskSpec,
    GitSha,
    Identifier,
    Sha256,
    StrictModel,
    TaskId,
    TaskSpec,
    UsageMetrics,
)
from repopilot.storage import PostgresVectorStore, VectorStore
from scripts.p7_holdout_agent_inputs import (
    AGENT_INPUTS_PATH,
    HoldoutAgentInputs,
)
from scripts.p7_holdout_control import (
    AUTHORIZATION_VERSION,
    ENVIRONMENT_PATH,
    EXPECTED_ARM_IDS,
    EXPECTED_TASK_IDS,
    MATRIX_PATH,
    PLAN_PATH,
    HoldoutApplicationPlan,
    HoldoutEnvironmentBinding,
    HoldoutEnvironmentManifest,
    HoldoutRunSlot,
    PaidHoldoutAuthorization,
)
from scripts.p7_holdout_readiness import (
    AUTHORIZATION_PATH,
    READINESS_PATH,
    HoldoutRuntimeReadiness,
    ReadinessError,
    _verify_frozen_runtime,
)

EXECUTION_REPORT_PATH = Path("results/p7-holdout/execution-report-v1.json")
EXECUTION_VERSION: Literal["holdout-execution-v1"] = "holdout-execution-v1"
READINESS_RAW_SHA256: Literal[
    "6608e286837149b44f27a0d4d378afa4f1985326db6394608ab3f768329cb90c"
] = "6608e286837149b44f27a0d4d378afa4f1985326db6394608ab3f768329cb90c"
AGENT_INPUTS_RAW_SHA256: Literal[
    "2a3a197ea7318d42e4b681db8b56b554fc241c19923ba8482566a4bdc9c7661f"
] = "2a3a197ea7318d42e4b681db8b56b554fc241c19923ba8482566a4bdc9c7661f"
WITHHELD_EVALUATOR_PATCH = "EVALUATOR_GOLD_WITHHELD_FROM_P7_S6_AGENT"
WITHHELD_EVALUATOR_FILE = "evaluator-gold-withheld"
MAX_EXECUTION_ATTEMPTS = 3
PER_ATTEMPT_UNKNOWN_COST_RESERVE_USD = Decimal("1.00")


@unique
class HoldoutRunnerErrorCode(StrEnum):
    """Stable, secret-free formal-run failures."""

    INVALID_CONFIGURATION = "invalid_configuration"
    HOST_PLATFORM_MISMATCH = "host_platform_mismatch"
    CONTROLLER_SOURCE_DIRTY = "controller_source_dirty"
    SOURCE_ARTIFACT_INVALID = "source_artifact_invalid"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    UNAPPROVED_ENVIRONMENT_OVERRIDE = "unapproved_environment_override"
    PROVIDER_CREDENTIAL_REQUIRED = "provider_credential_required"
    CHECKOUT_UNAVAILABLE = "checkout_unavailable"
    SANDBOX_IMAGE_UNAVAILABLE = "sandbox_image_unavailable"
    OUTPUT_CONFLICT = "output_conflict"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TECHNICAL_FAILURE = "technical_failure"


class HoldoutRunnerError(RuntimeError):
    """Typed formal-run failure without credentials, DSNs, Issue text, or host paths."""

    def __init__(self, code: HoldoutRunnerErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


class HoldoutSlotOutcome(StrictModel):
    """One terminal raw slot reconstructed from immutable Runner files."""

    arm_id: Identifier
    task_id: TaskId
    run_id: UUID
    output_path: str
    manifest_ref: ArtifactRef
    attempt_status: Literal["completed", "technical_failure"]
    execution_count: int = Field(ge=1, le=MAX_EXECUTION_ATTEMPTS, strict=True)
    run_status: RunStatus | None = None
    termination_reason: TerminationReason | None = None
    task_success: bool | None = None
    usage: UsageMetrics | None = None
    technical_failure_code: EvaluationRunnerErrorCode | None = None
    reserved_unknown_cost_usd: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.attempt_status == "completed":
            if (
                self.run_status is None
                or self.termination_reason is None
                or self.task_success is None
                or self.usage is None
                or self.technical_failure_code is not None
            ):
                raise ValueError("completed holdout outcome lacks execution evidence")
            expected_reserve = Decimal(self.execution_count - 1)
        else:
            if (
                self.run_status is not None
                or self.termination_reason is not None
                or self.task_success is not None
                or self.usage is not None
                or self.technical_failure_code is None
            ):
                raise ValueError("technical holdout outcome contains invalid evidence")
            expected_reserve = Decimal(self.execution_count)
        if self.reserved_unknown_cost_usd != expected_reserve:
            raise ValueError("unknown-cost reserve does not match execution history")
        return self


class HoldoutExecutionReport(StrictModel):
    """Final P7-S6 raw-run index; quality aggregation remains P7-S7."""

    schema_version: Literal[1] = 1
    execution_version: Literal["holdout-execution-v1"] = EXECUTION_VERSION
    experiment_id: Identifier
    plan_content_sha256: Sha256
    authorization_content_sha256: Sha256
    readiness_raw_sha256: Sha256
    readiness_content_sha256: Sha256
    agent_inputs_raw_sha256: Sha256
    agent_inputs_content_sha256: Sha256
    controller_commit: GitSha
    completed_at: AwareDatetime
    slot_count: Literal[32] = 32
    completed_slot_count: int = Field(ge=0, le=32, strict=True)
    technical_failure_slot_count: int = Field(ge=0, le=32, strict=True)
    task_success_count: int = Field(ge=0, le=32, strict=True)
    known_estimated_cost_usd: Decimal = Field(ge=0)
    reserved_unknown_cost_usd: Decimal = Field(ge=0)
    application_accounted_cost_usd: Decimal = Field(ge=0)
    authorized_cost_usd: Decimal = Field(ge=0)
    final_in_flight_overshoot_observed: bool
    outcomes: tuple[HoldoutSlotOutcome, ...]
    aggregation_deferred_to: Literal["P7-S7"] = "P7-S7"
    gold_visible_to_agent: Literal[False] = False
    source_rows_loaded_by_formal_runner: Literal[False] = False
    formal_execution_complete: Literal[True] = True
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if len(self.outcomes) != self.slot_count:
            raise ValueError("formal execution report does not cover all slots")
        completed = sum(item.attempt_status == "completed" for item in self.outcomes)
        technical = sum(item.attempt_status == "technical_failure" for item in self.outcomes)
        successes = sum(item.task_success is True for item in self.outcomes)
        known = sum(
            (item.usage.estimated_cost_usd for item in self.outcomes if item.usage is not None),
            Decimal("0"),
        )
        reserved = sum((item.reserved_unknown_cost_usd for item in self.outcomes), Decimal("0"))
        if (
            self.authorized_cost_usd != Decimal("32.00")
            or self.completed_slot_count != completed
            or self.technical_failure_slot_count != technical
            or self.task_success_count != successes
            or self.known_estimated_cost_usd != known
            or self.reserved_unknown_cost_usd != reserved
            or self.application_accounted_cost_usd != known + reserved
            or self.final_in_flight_overshoot_observed
            != (self.application_accounted_cost_usd > self.authorized_cost_usd)
        ):
            raise ValueError("formal execution report totals are inconsistent")
        _validate_content_digest(self)
        return self


@dataclass(frozen=True)
class _Controls:
    plan: HoldoutApplicationPlan
    authorization: PaidHoldoutAuthorization
    environment: HoldoutEnvironmentManifest
    readiness: HoldoutRuntimeReadiness
    agent_inputs: HoldoutAgentInputs
    readiness_raw: bytes
    agent_inputs_raw: bytes
    controller_commit: str


@dataclass(frozen=True)
class _Progress:
    outcomes: tuple[HoldoutSlotOutcome, ...]
    pending_slots: tuple[HoldoutRunSlot, ...]
    interrupted_slots: tuple[HoldoutRunSlot, ...]
    known_cost_usd: Decimal
    reserved_unknown_cost_usd: Decimal

    @property
    def accounted_cost_usd(self) -> Decimal:
        return self.known_cost_usd + self.reserved_unknown_cost_usd

    @property
    def technical_failures(self) -> int:
        return sum(item.attempt_status == "technical_failure" for item in self.outcomes)


ExecutorFactory = Callable[[str, str], EvaluationTaskExecutor]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-plan-sha256")
    parser.add_argument(
        "--max-new-slots",
        type=int,
        default=32,
        choices=range(1, 33),
        metavar="1..32",
    )
    parser.add_argument(
        "--continue-after-technical-failure",
        action="store_true",
        help="continue other slots using the conservative USD 1 reserve per failed attempt",
    )
    arguments = parser.parse_args()
    try:
        root = arguments.repository_root.resolve(strict=True)
    except OSError:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.INVALID_CONFIGURATION) from None
    controls = _load_and_verify_controls(root, require_credential=arguments.execute)
    progress = _inspect_progress(root, controls)
    _print_progress(progress, execute=arguments.execute, slot_executions_started=0)
    if not arguments.execute:
        print(
            "holdout_execution_preflight_passed=true execute=false external_mutations=0 "
            "provider_requests=0"
        )
        return 0
    if arguments.confirm_plan_sha256 != controls.plan.content_sha256:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.INVALID_CONFIGURATION)
    if progress.technical_failures and not arguments.continue_after_technical_failure:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.TECHNICAL_FAILURE)
    if not progress.pending_slots and not progress.interrupted_slots:
        _write_final_report_if_complete(root, controls, progress)
        return 0

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    api_key = os.environ["OPENAI_API_KEY"]
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://repopilot:repopilot@localhost:55433/repopilot",
    )
    try:
        executor_factory = _real_executor_factory(
            root=root,
            controls=controls,
            openai_api_key=api_key,
            database_url=database_url,
        )
    except Exception:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.TECHNICAL_FAILURE) from None
    executed = 0
    for slot in controls.plan.run_slots:
        if executed >= arguments.max_new_slots:
            break
        current = _inspect_progress(root, controls)
        if any(item.run_id == slot.run_id for item in current.outcomes):
            continue
        if current.technical_failures and not arguments.continue_after_technical_failure:
            raise HoldoutRunnerError(HoldoutRunnerErrorCode.TECHNICAL_FAILURE)
        if (
            current.accounted_cost_usd + PER_ATTEMPT_UNKNOWN_COST_RESERVE_USD
            > controls.authorization.max_total_cost_usd
        ):
            raise HoldoutRunnerError(HoldoutRunnerErrorCode.BUDGET_EXHAUSTED)
        _execute_slot(
            root=root,
            controls=controls,
            slot=slot,
            executor=_executor_for_slot(executor_factory, slot),
            accounted_cost_before=current.accounted_cost_usd,
        )
        executed += 1
        after = _inspect_progress(root, controls)
        outcome = next(item for item in after.outcomes if item.run_id == slot.run_id)
        print(
            f"holdout_slot_terminal=true arm={slot.arm_id} task={slot.task_id} "
            f"status={outcome.attempt_status} accounted_cost_usd={after.accounted_cost_usd}",
            flush=True,
        )
        if outcome.attempt_status == "technical_failure" and not (
            arguments.continue_after_technical_failure
        ):
            raise HoldoutRunnerError(HoldoutRunnerErrorCode.TECHNICAL_FAILURE)
        if after.accounted_cost_usd > controls.authorization.max_total_cost_usd:
            raise HoldoutRunnerError(HoldoutRunnerErrorCode.BUDGET_EXHAUSTED)

    final = _inspect_progress(root, controls)
    _print_progress(final, execute=True, slot_executions_started=executed)
    if not final.pending_slots and not final.interrupted_slots:
        _write_final_report_if_complete(root, controls, final)
    else:
        print(
            f"holdout_execution_invocation_complete=true slot_executions_started={executed} "
            f"remaining_slots={len(final.pending_slots) + len(final.interrupted_slots)}"
        )
    return 0


def _load_and_verify_controls(root: Path, *, require_credential: bool) -> _Controls:
    if platform.system().lower() != "linux" or platform.machine().lower() not in {
        "amd64",
        "x86_64",
    }:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.HOST_PLATFORM_MISMATCH)
    if any(name.startswith("REPOPILOT_") for name in os.environ):
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.UNAPPROVED_ENVIRONMENT_OVERRIDE)
    try:
        tracked_status = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    except subprocess.SubprocessError:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.SOURCE_ARTIFACT_INVALID) from None
    if tracked_status:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.CONTROLLER_SOURCE_DIRTY)
    if require_credential and not os.environ.get("OPENAI_API_KEY", "").strip():
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.PROVIDER_CREDENTIAL_REQUIRED)
    try:
        plan_raw = (root / PLAN_PATH).read_bytes()
        authorization_raw = (root / AUTHORIZATION_PATH).read_bytes()
        environment_raw = (root / ENVIRONMENT_PATH).read_bytes()
        readiness_raw = (root / READINESS_PATH).read_bytes()
        agent_inputs_raw = (root / AGENT_INPUTS_PATH).read_bytes()
        plan = HoldoutApplicationPlan.model_validate_json(plan_raw)
        authorization = PaidHoldoutAuthorization.model_validate_json(authorization_raw)
        environment = HoldoutEnvironmentManifest.model_validate_json(environment_raw)
        readiness = HoldoutRuntimeReadiness.model_validate_json(readiness_raw)
        agent_inputs = HoldoutAgentInputs.model_validate_json(agent_inputs_raw)
        matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
    except (OSError, ValidationError, ValueError):
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.SOURCE_ARTIFACT_INVALID) from None
    try:
        controller_commit = _git(root, "rev-parse", "HEAD")
    except subprocess.SubprocessError:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.SOURCE_ARTIFACT_INVALID) from None
    if (
        hashlib.sha256(readiness_raw).hexdigest() != READINESS_RAW_SHA256
        or hashlib.sha256(agent_inputs_raw).hexdigest() != AGENT_INPUTS_RAW_SHA256
        or hashlib.sha256(plan_raw).hexdigest() != readiness.plan_raw_sha256
        or hashlib.sha256(authorization_raw).hexdigest() != readiness.authorization_raw_sha256
        or hashlib.sha256(environment_raw).hexdigest() != readiness.environment_raw_sha256
        or plan.content_sha256 != readiness.plan_content_sha256
        or authorization.content_sha256 != readiness.authorization_content_sha256
        or environment.content_sha256 != readiness.environment_content_sha256
        or not readiness.ready_for_paid_holdout
        or readiness.paid_requests_made != 0
        or readiness.gold_visible_to_agent
        or authorization.authorization_version != AUTHORIZATION_VERSION
        or authorization.plan_sha256 != plan.content_sha256
        or authorization.max_total_cost_usd != Decimal("32.00")
        or matrix.content_sha256 != plan.matrix_sha256
        or agent_inputs.matrix_sha256 != matrix.content_sha256
        or agent_inputs.dataset_manifest_content_sha256 != plan.dataset_manifest_sha256
        or agent_inputs.runtime_command_contract_sha256
        != environment.runtime_command_contract_sha256
        or tuple(item.task_id for item in environment.bindings) != EXPECTED_TASK_IDS
        or tuple(item.arm_id for item in plan.run_slots[::8]) != EXPECTED_ARM_IDS
        or len(plan.run_slots) != 32
    ):
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.SOURCE_IDENTITY_MISMATCH)
    try:
        _verify_frozen_runtime(
            root=root,
            source_commit=readiness.frozen_runtime.frozen_source_commit,
            environment_commit=environment.controller_commit,
            controller_commit=controller_commit,
        )
    except ReadinessError:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.SOURCE_IDENTITY_MISMATCH) from None
    controls = _Controls(
        plan=plan,
        authorization=authorization,
        environment=environment,
        readiness=readiness,
        agent_inputs=agent_inputs,
        readiness_raw=readiness_raw,
        agent_inputs_raw=agent_inputs_raw,
        controller_commit=controller_commit,
    )
    _verify_runtime_assets(root, controls)
    return controls


def _verify_runtime_assets(root: Path, controls: _Controls) -> None:
    tasks = {item.task_id: item for item in controls.agent_inputs.tasks}
    readiness_images = {item.task_id: item for item in controls.readiness.images}
    for binding in controls.environment.bindings:
        task = tasks.get(binding.task_id)
        image = readiness_images.get(binding.task_id)
        slot = next(item for item in controls.plan.run_slots if item.task_id == binding.task_id)
        if (
            task is None
            or image is None
            or task.base_commit != binding.base_commit
            or slot.base_commit != binding.base_commit
            or image.registry_digest_reference != binding.sandbox_image_reference
        ):
            raise HoldoutRunnerError(HoldoutRunnerErrorCode.SOURCE_IDENTITY_MISMATCH)
        checkout = _checkout(root, binding)
        if task.repository_path is not None or task.repository_url is None or not checkout.is_dir():
            raise HoldoutRunnerError(HoldoutRunnerErrorCode.SOURCE_IDENTITY_MISMATCH)
        try:
            resolved = DockerCliClient().resolve_image(binding.sandbox_image_reference)
        except DockerClientError:
            raise HoldoutRunnerError(HoldoutRunnerErrorCode.SANDBOX_IMAGE_UNAVAILABLE) from None
        if resolved.image_id != image.local_image_id:
            raise HoldoutRunnerError(HoldoutRunnerErrorCode.SOURCE_IDENTITY_MISMATCH)


def _checkout(root: Path, binding: HoldoutEnvironmentBinding) -> Path:
    try:
        checkout = (root / binding.checkout_relative_path).resolve(strict=True)
        valid = (
            checkout.is_relative_to(root)
            and _git(checkout, "rev-parse", "HEAD") == binding.base_commit
            and not _git(checkout, "status", "--porcelain=v1")
        )
    except (OSError, subprocess.SubprocessError):
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.CHECKOUT_UNAVAILABLE) from None
    if not valid:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.CHECKOUT_UNAVAILABLE)
    return checkout


def _runtime_task(root: Path, controls: _Controls, task_id: str) -> TaskSpec:
    source = next(item for item in controls.agent_inputs.tasks if item.task_id == task_id)
    binding = next(item for item in controls.environment.bindings if item.task_id == task_id)
    checkout = _checkout(root, binding)
    return source.model_copy(update={"repository_path": str(checkout), "repository_url": None})


def _runner_task(task: TaskSpec) -> EvaluationTaskSpec:
    """Satisfy the generic runner wrapper without making evaluator Gold available."""

    return EvaluationTaskSpec(
        task=task,
        expected_patch=WITHHELD_EVALUATOR_PATCH,
        gold_files=(WITHHELD_EVALUATOR_FILE,),
        split=DatasetSplit.HOLDOUT,
    )


def _expected_manifest(
    root: Path,
    controls: _Controls,
    slot: HoldoutRunSlot,
) -> EvaluationRunManifest:
    matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
    selection = select_experiment_arm(matrix, slot.arm_id)
    binding = next(item for item in controls.environment.bindings if item.task_id == slot.task_id)
    task = _runner_task(_runtime_task(root, controls, slot.task_id))
    settings = selection.settings(
        openai_api_key="formal-manifest-secret-not-serialized",
        database_url="postgresql://repopilot:repopilot@localhost:55433/repopilot",
    )
    manifest = build_evaluation_run_manifest(
        experiment_id=controls.plan.experiment_id,
        run_id=slot.run_id,
        created_at=controls.plan.created_at,
        source_commit=controls.plan.frozen_source_commit,
        dependency_lock_sha256=controls.plan.dependency_lock_sha256,
        dataset_manifest_sha256=controls.plan.dataset_manifest_sha256,
        configuration=selection.configuration_snapshot,
        split=DatasetSplit.HOLDOUT,
        arm_id=slot.arm_id,
        tasks=(task,),
        memory_namespace=None,
        model_provider=settings.model_provider,
        model=settings.model,
        prompt_versions=controls.plan.prompt_versions,
        sandbox_image_reference=binding.sandbox_image_reference,
        repetitions=1,
        max_execution_attempts=MAX_EXECUTION_ATTEMPTS,
    )
    if manifest.configuration_sha256 != slot.configuration_sha256:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.SOURCE_IDENTITY_MISMATCH)
    return manifest


def _inspect_progress(root: Path, controls: _Controls) -> _Progress:
    outcomes: list[HoldoutSlotOutcome] = []
    pending: list[HoldoutRunSlot] = []
    interrupted: list[HoldoutRunSlot] = []
    interrupted_execution_counts: dict[UUID, int] = {}
    for slot in controls.plan.run_slots:
        output = root / slot.output_path
        if not output.exists() and not output.is_symlink():
            pending.append(slot)
            continue
        output = _existing_output_directory(root, slot.output_path)
        expected = _expected_manifest(root, controls, slot)
        try:
            report = inspect_evaluation_run(output)
            actual = EvaluationRunManifest.model_validate_json(
                (output / "manifest.json").read_bytes()
            )
        except EvaluationRunnerError as error:
            raise _map_runner_error(error) from None
        except (OSError, ValidationError, ValueError):
            raise HoldoutRunnerError(HoldoutRunnerErrorCode.OUTPUT_CONFLICT) from None
        if actual != expected or report.run_id != slot.run_id or len(report.attempts) != 1:
            raise HoldoutRunnerError(HoldoutRunnerErrorCode.SOURCE_IDENTITY_MISMATCH)
        attempt = report.attempts[0]
        if attempt.status is EvaluationAttemptStatus.COMPLETED and attempt.evidence is not None:
            outcomes.append(
                _completed_outcome(root, slot, attempt.execution_count, attempt.evidence)
            )
        elif attempt.status is EvaluationAttemptStatus.FAILED and attempt.failure_code is not None:
            outcomes.append(
                _technical_outcome(root, slot, attempt.execution_count, attempt.failure_code)
            )
        else:
            interrupted.append(slot)
            interrupted_execution_counts[slot.run_id] = attempt.execution_count
    known = sum(
        (item.usage.estimated_cost_usd for item in outcomes if item.usage is not None),
        Decimal("0"),
    )
    reserved = sum((item.reserved_unknown_cost_usd for item in outcomes), Decimal("0"))
    for slot in interrupted:
        reserved += Decimal(interrupted_execution_counts[slot.run_id])
    return _Progress(
        outcomes=tuple(outcomes),
        pending_slots=tuple(pending),
        interrupted_slots=tuple(interrupted),
        known_cost_usd=known,
        reserved_unknown_cost_usd=reserved,
    )


def _completed_outcome(
    root: Path,
    slot: HoldoutRunSlot,
    execution_count: int,
    evidence: object,
) -> HoldoutSlotOutcome:
    from repopilot.evals.runner import EvaluationExecutionEvidence

    validated = EvaluationExecutionEvidence.model_validate(evidence)
    return HoldoutSlotOutcome(
        arm_id=slot.arm_id,
        task_id=slot.task_id,
        run_id=slot.run_id,
        output_path=slot.output_path,
        manifest_ref=_artifact_ref(root, root / slot.output_path / "manifest.json"),
        attempt_status="completed",
        execution_count=execution_count,
        run_status=validated.status,
        termination_reason=validated.termination_reason,
        task_success=validated.task_success,
        usage=validated.usage,
        reserved_unknown_cost_usd=Decimal(execution_count - 1),
    )


def _technical_outcome(
    root: Path,
    slot: HoldoutRunSlot,
    execution_count: int,
    failure_code: EvaluationRunnerErrorCode,
) -> HoldoutSlotOutcome:
    return HoldoutSlotOutcome(
        arm_id=slot.arm_id,
        task_id=slot.task_id,
        run_id=slot.run_id,
        output_path=slot.output_path,
        manifest_ref=_artifact_ref(root, root / slot.output_path / "manifest.json"),
        attempt_status="technical_failure",
        execution_count=execution_count,
        technical_failure_code=failure_code,
        reserved_unknown_cost_usd=Decimal(execution_count),
    )


def _execute_slot(
    *,
    root: Path,
    controls: _Controls,
    slot: HoldoutRunSlot,
    executor: EvaluationTaskExecutor,
    accounted_cost_before: Decimal,
) -> EvaluationRunReport:
    manifest = _expected_manifest(root, controls, slot)
    task = _runner_task(_runtime_task(root, controls, slot.task_id))
    _prepare_output_parent(root, slot.output_path)

    def guard(_: EvaluationRunReport) -> None:
        if (
            accounted_cost_before + PER_ATTEMPT_UNKNOWN_COST_RESERVE_USD
            > controls.authorization.max_total_cost_usd
        ):
            raise HoldoutRunnerError(HoldoutRunnerErrorCode.BUDGET_EXHAUSTED)

    try:
        return run_evaluation(
            manifest=manifest,
            tasks=(task,),
            output_dir=root / slot.output_path,
            executor=executor,
            before_attempt=guard,
        )
    except HoldoutRunnerError:
        raise
    except EvaluationRunnerError as error:
        raise _map_runner_error(error) from None


def _prepare_output_parent(root: Path, output_path: str) -> None:
    """Create one relative output hierarchy without traversing a symlink."""

    try:
        resolved_root = root.resolve(strict=True)
        current = resolved_root
        for part in Path(output_path).parent.parts:
            current = current / part
            if current.exists() or current.is_symlink():
                if current.is_symlink() or not current.is_dir():
                    raise OSError
            else:
                current.mkdir(mode=0o700)
        if not current.resolve(strict=True).is_relative_to(resolved_root):
            raise OSError
    except OSError:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.OUTPUT_CONFLICT) from None


def _existing_output_directory(root: Path, output_path: str) -> Path:
    """Resolve an existing output only when every component is a real directory."""

    try:
        resolved_root = root.resolve(strict=True)
        current = resolved_root
        for part in Path(output_path).parts:
            current = current / part
            if current.is_symlink() or not current.is_dir():
                raise OSError
        if not current.resolve(strict=True).is_relative_to(resolved_root):
            raise OSError
    except OSError:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.OUTPUT_CONFLICT) from None
    return current


def _real_executor_factory(
    *,
    root: Path,
    controls: _Controls,
    openai_api_key: str,
    database_url: str,
) -> ExecutorFactory:
    matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
    dense_settings = select_experiment_arm(matrix, "dense-only-v1").settings(
        openai_api_key=openai_api_key,
        database_url=database_url,
    )
    shared_embedding_provider: EmbeddingProvider = embedding_provider_from_settings(dense_settings)
    shared_embedding_cache: LocalEmbeddingCache = embedding_cache_from_settings(
        dense_settings,
        project_root=root,
    )
    shared_vector_store: VectorStore = PostgresVectorStore(dense_settings)

    def build(arm_id: str, task_id: str) -> EvaluationTaskExecutor:
        selection = select_experiment_arm(matrix, arm_id)
        settings = selection.settings(
            openai_api_key=openai_api_key,
            database_url=database_url,
        )
        binding = next(item for item in controls.environment.bindings if item.task_id == task_id)
        if selection.arm.baseline_config_ref is not None:
            definition = load_baseline_definition(
                root / selection.arm.baseline_config_ref.relative_path
            )
            return BaselineEvaluationExecutor(
                definition=definition,
                settings=settings,
                sandbox_image_reference=binding.sandbox_image_reference,
                evaluation_configuration_sha256=selection.configuration_sha256,
            )
        return RepairWorkflowEvaluationExecutor(
            settings=settings,
            sandbox_image_reference=binding.sandbox_image_reference,
            context_mode=selection.arm.context_mode,
            evaluation_configuration_sha256=selection.configuration_sha256,
            embedding_provider=shared_embedding_provider,
            embedding_cache=shared_embedding_cache,
            vector_store=shared_vector_store,
        )

    return build


def _executor_for_slot(
    executor_factory: ExecutorFactory,
    slot: HoldoutRunSlot,
) -> EvaluationTaskExecutor:
    try:
        return executor_factory(slot.arm_id, slot.task_id)
    except HoldoutRunnerError:
        raise
    except Exception:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.TECHNICAL_FAILURE) from None


def _write_final_report_if_complete(
    root: Path,
    controls: _Controls,
    progress: _Progress,
) -> HoldoutExecutionReport:
    if progress.pending_slots or progress.interrupted_slots or len(progress.outcomes) != 32:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.OUTPUT_CONFLICT)
    ordered = tuple(
        next(item for item in progress.outcomes if item.run_id == slot.run_id)
        for slot in controls.plan.run_slots
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "execution_version": EXECUTION_VERSION,
        "experiment_id": controls.plan.experiment_id,
        "plan_content_sha256": controls.plan.content_sha256,
        "authorization_content_sha256": controls.authorization.content_sha256,
        "readiness_raw_sha256": hashlib.sha256(controls.readiness_raw).hexdigest(),
        "readiness_content_sha256": controls.readiness.content_sha256,
        "agent_inputs_raw_sha256": hashlib.sha256(controls.agent_inputs_raw).hexdigest(),
        "agent_inputs_content_sha256": controls.agent_inputs.content_sha256,
        "controller_commit": controls.controller_commit,
        "completed_at": datetime.now(UTC),
        "slot_count": 32,
        "completed_slot_count": sum(item.attempt_status == "completed" for item in ordered),
        "technical_failure_slot_count": sum(
            item.attempt_status == "technical_failure" for item in ordered
        ),
        "task_success_count": sum(item.task_success is True for item in ordered),
        "known_estimated_cost_usd": progress.known_cost_usd,
        "reserved_unknown_cost_usd": progress.reserved_unknown_cost_usd,
        "application_accounted_cost_usd": progress.accounted_cost_usd,
        "authorized_cost_usd": Decimal("32.00"),
        "final_in_flight_overshoot_observed": (
            progress.accounted_cost_usd > controls.authorization.max_total_cost_usd
        ),
        "outcomes": ordered,
        "aggregation_deferred_to": "P7-S7",
        "gold_visible_to_agent": False,
        "source_rows_loaded_by_formal_runner": False,
        "formal_execution_complete": True,
    }
    report = HoldoutExecutionReport.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )
    target = root / EXECUTION_REPORT_PATH
    if target.exists():
        try:
            existing = HoldoutExecutionReport.model_validate_json(target.read_bytes())
        except (OSError, ValidationError):
            raise HoldoutRunnerError(HoldoutRunnerErrorCode.OUTPUT_CONFLICT) from None
        if existing != report.model_copy(
            update={
                "completed_at": existing.completed_at,
                "content_sha256": existing.content_sha256,
            }
        ):
            raise HoldoutRunnerError(HoldoutRunnerErrorCode.OUTPUT_CONFLICT)
        print(
            "holdout_execution_complete=true slots=32 "
            f"known_cost_usd={existing.known_estimated_cost_usd} "
            f"reserved_unknown_cost_usd={existing.reserved_unknown_cost_usd} "
            "aggregation_deferred_to=P7-S7"
        )
        return existing
    _write_private_create_only(target, report.model_dump_json(indent=2).encode() + b"\n")
    print(
        "holdout_execution_complete=true slots=32 "
        f"known_cost_usd={report.known_estimated_cost_usd} "
        f"reserved_unknown_cost_usd={report.reserved_unknown_cost_usd} "
        "aggregation_deferred_to=P7-S7"
    )
    return report


def _print_progress(
    progress: _Progress,
    *,
    execute: bool,
    slot_executions_started: int,
) -> None:
    print(
        "holdout_execution_progress="
        + json.dumps(
            {
                "accounted_cost_usd": str(progress.accounted_cost_usd),
                "completed_or_failed_slots": len(progress.outcomes),
                "execute": execute,
                "interrupted_slots": len(progress.interrupted_slots),
                "known_cost_usd": str(progress.known_cost_usd),
                "pending_slots": len(progress.pending_slots),
                "reserved_unknown_cost_usd": str(progress.reserved_unknown_cost_usd),
                "slot_executions_started_by_this_invocation": slot_executions_started,
                "technical_failure_slots": progress.technical_failures,
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def _map_runner_error(error: EvaluationRunnerError) -> HoldoutRunnerError:
    if error.code in {
        EvaluationRunnerErrorCode.OUTPUT_CONFLICT,
        EvaluationRunnerErrorCode.OUTPUT_IO_ERROR,
        EvaluationRunnerErrorCode.PATH_UNSAFE,
        EvaluationRunnerErrorCode.INPUT_IO_ERROR,
        EvaluationRunnerErrorCode.CONCURRENT_WRITE,
    }:
        return HoldoutRunnerError(HoldoutRunnerErrorCode.OUTPUT_CONFLICT)
    if error.code in {
        EvaluationRunnerErrorCode.CONFIGURATION_DRIFT,
        EvaluationRunnerErrorCode.IDENTITY_MISMATCH,
        EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY,
        EvaluationRunnerErrorCode.INVALID_MANIFEST,
        EvaluationRunnerErrorCode.UNSUPPORTED_VERSION,
    }:
        return HoldoutRunnerError(HoldoutRunnerErrorCode.SOURCE_IDENTITY_MISMATCH)
    return HoldoutRunnerError(HoldoutRunnerErrorCode.TECHNICAL_FAILURE)


def _artifact_ref(root: Path, path: Path) -> ArtifactRef:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
            raise OSError
        content = resolved.read_bytes()
        relative = resolved.relative_to(resolved_root).as_posix()
    except OSError:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.OUTPUT_CONFLICT) from None
    return ArtifactRef(
        relative_path=relative,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        media_type="application/json",
    )


def _write_private_create_only(path: Path, content: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
    except OSError:
        raise HoldoutRunnerError(HoldoutRunnerErrorCode.OUTPUT_CONFLICT) from None


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_content_digest(value: StrictModel) -> None:
    payload = value.model_dump(mode="python", exclude={"content_sha256"})
    if _canonical_sha256(payload) != getattr(value, "content_sha256", None):
        raise ValueError("holdout execution report content digest does not match")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HoldoutRunnerError as error:
        raise SystemExit(str(error)) from None
