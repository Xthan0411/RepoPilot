"""Build the fail-closed P7-S6 holdout plan and readiness report.

This module deliberately has no paid execution entry point.  A formal holdout run is
allowed only after an exact authorization and a task-specific environment manifest both
pass this control layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum, unique
from pathlib import Path
from typing import Literal, Self
from uuid import UUID, uuid5

from pydantic import AwareDatetime, Field, ValidationError, model_validator
from pydantic_core import to_jsonable_python

from repopilot.enums import DatasetPreflightStatus, DatasetSplit, ExperimentPreflightCheckStatus
from repopilot.evals import (
    ExperimentMatrix,
    FormalFreezeMarker,
    load_experiment_matrix,
    select_experiment_arm,
)
from repopilot.evals.dataset import DatasetPreflightReport, FrozenDatasetManifest
from repopilot.schemas import CommandSpec, GitSha, Identifier, Sha256, StrictModel, TaskId, TaskSpec

SCHEMA_VERSION: Literal[1] = 1
PLAN_VERSION: Literal["holdout-application-plan-v1"] = "holdout-application-plan-v1"
AUTHORIZATION_VERSION: Literal["holdout-authorization-v1"] = "holdout-authorization-v1"
PREFLIGHT_VERSION: Literal["holdout-preflight-v1"] = "holdout-preflight-v1"
ENVIRONMENT_VERSION: Literal["holdout-environments-v1"] = "holdout-environments-v1"
TIMEOUT_CONTRACT_VERSION: Literal["holdout-command-timeout-contract-v1"] = (
    "holdout-command-timeout-contract-v1"
)
COMMAND_TIMEOUT_CAP_SECONDS: Literal[300] = 300
MATRIX_PATH = Path("configs/experiments/p7-v2.yaml")
FREEZE_PATH = Path("results/p7-dev/formal-freeze-v1.json")
DATASET_PREFLIGHT_PATH = Path("results/p7-dataset-v1/preflight-report.json")
PLAN_PATH = Path("results/p7-holdout/plan-v1.json")
PREFLIGHT_PATH = Path("results/p7-holdout/preflight-v1.json")
ENVIRONMENT_PATH = Path("results/p7-holdout/environments-v1.json")
EXPECTED_ARM_IDS = (
    "dense-only-v1",
    "hybrid-rrf-v1",
    "mini-swe-safe-v1",
    "no-semantic-rag-v1",
)
EXPECTED_TASK_IDS = (
    "django__django-15902",
    "mwaskom__seaborn-3190",
    "pallets__flask-4992",
    "pydata__xarray-4094",
    "pydata__xarray-5131",
    "pytest-dev__pytest-11143",
    "pytest-dev__pytest-11148",
    "sympy__sympy-20590",
)
PER_TASK_COST_LIMIT_USD = Decimal("1.00")
THEORETICAL_APPLICATION_COST_USD = Decimal("32.00")
_RUN_NAMESPACE = UUID("1db83bc9-9712-49f4-a61e-b76475f50ce0")
_DIGEST_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:~-]{0,439}@sha256:[0-9a-f]{64}$")
_RELATIVE_CHECKOUT = re.compile(r"^artifacts/p7-holdout/checkouts/[A-Za-z0-9._/-]{1,300}$")
_FROZEN_RUNTIME_PATHS = (
    "src/repopilot",
    "pyproject.toml",
    "uv.lock",
    "configs",
    "data/memory",
    "tests/fixtures/p5_repair_tasks",
)


@unique
class HoldoutErrorCode(StrEnum):
    """Stable, secret-free reasons that prevent or terminate formal holdout control."""

    INVALID_CONFIGURATION = "invalid_configuration"
    FREEZE_IDENTITY_MISMATCH = "freeze_identity_mismatch"
    FROZEN_RUNTIME_DRIFT = "frozen_runtime_drift"
    SOURCE_SNAPSHOT_UNCLEAN = "source_snapshot_unclean"
    DATASET_MISMATCH = "dataset_mismatch"
    HOLDOUT_TASK_SET_MISMATCH = "holdout_task_set_mismatch"
    DEFAULT_ARM_SET_MISMATCH = "default_arm_set_mismatch"
    OPTIONAL_ARM_FORBIDDEN = "optional_arm_forbidden"
    BUDGET_AUTHORIZATION_REQUIRED = "budget_authorization_required"
    BUDGET_AUTHORIZATION_MISMATCH = "budget_authorization_mismatch"
    BUDGET_LIMIT_EXCEEDED = "budget_limit_exceeded"
    FINAL_RESPONSE_OVERSHOOT_UNACKNOWLEDGED = "final_response_overshoot_unacknowledged"
    PROVIDER_CREDENTIAL_REQUIRED = "provider_credential_required"
    TASK_ENVIRONMENT_UNVERIFIED = "task_environment_unverified"
    TASK_CHECKOUT_UNAVAILABLE = "task_checkout_unavailable"
    TASK_CHECKOUT_IDENTITY_MISMATCH = "task_checkout_identity_mismatch"
    SANDBOX_IMAGE_UNAVAILABLE = "sandbox_image_unavailable"
    SANDBOX_IMAGE_UNPINNED = "sandbox_image_unpinned"
    SANDBOX_PLATFORM_MISMATCH = "sandbox_platform_mismatch"
    TASK_PYTHON_MISMATCH = "task_python_mismatch"
    VECTOR_BACKEND_UNAVAILABLE = "vector_backend_unavailable"
    EMBEDDING_SNAPSHOT_UNAVAILABLE = "embedding_snapshot_unavailable"
    GOLD_BOUNDARY_VIOLATION = "gold_boundary_violation"
    OUTPUT_CONFLICT = "output_conflict"
    PATH_UNSAFE = "path_unsafe"
    RUN_INCOMPLETE = "run_incomplete"
    ARTIFACT_INTEGRITY = "artifact_integrity"


class HoldoutControlError(RuntimeError):
    """Typed control-plane rejection that never contains task, secret, or host text."""

    def __init__(self, code: HoldoutErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


class HoldoutRunSlot(StrictModel):
    """One task-specific Runner slot; grouping by task permits per-task images."""

    arm_id: Identifier
    task_id: TaskId
    run_id: UUID
    configuration_sha256: Sha256
    base_commit: GitSha
    python_version: str = Field(pattern=r"^3\.(?:9|10|11|12)(?:\.[0-9]+)?$")
    output_path: str = Field(pattern=r"^results/p7-holdout/raw/[A-Za-z0-9._/-]{1,500}$")


class HoldoutApplicationPlan(StrictModel):
    """Non-secret 32-slot declaration created before any authorization or paid call."""

    schema_version: Literal[1] = SCHEMA_VERSION
    plan_version: Literal["holdout-application-plan-v1"] = PLAN_VERSION
    plan_id: Identifier
    experiment_id: Identifier
    matrix_sha256: Sha256
    freeze_marker_sha256: Sha256
    frozen_source_commit: GitSha
    controller_commit: GitSha
    dependency_lock_sha256: Sha256
    dataset_manifest_sha256: Sha256
    prompt_versions: tuple[Identifier, ...]
    model: Identifier
    embedding_model: str
    embedding_revision: GitSha
    embedding_code_revision: GitSha
    frozen_sandbox_image_reference: str
    created_at: AwareDatetime
    split: Literal[DatasetSplit.HOLDOUT] = DatasetSplit.HOLDOUT
    arm_ids: tuple[Identifier, ...]
    task_ids: tuple[TaskId, ...]
    repetitions: Literal[1] = 1
    per_task_cost_limit_usd: Decimal
    theoretical_application_cost_usd: Decimal
    final_in_flight_response_may_overshoot: Literal[True] = True
    authorization_required: Literal[True] = True
    aggregation_deferred_to: Literal["P7-S7"] = "P7-S7"
    tuning_from_holdout_forbidden: Literal[True] = True
    run_slots: tuple[HoldoutRunSlot, ...]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.arm_ids != EXPECTED_ARM_IDS:
            raise ValueError("formal holdout requires the exact four default arms")
        if self.task_ids != EXPECTED_TASK_IDS:
            raise ValueError("formal holdout requires the exact frozen eight tasks")
        if self.per_task_cost_limit_usd != PER_TASK_COST_LIMIT_USD:
            raise ValueError("formal holdout per-task cost must remain USD 1.00")
        expected_cost = (
            self.per_task_cost_limit_usd * len(self.arm_ids) * len(self.task_ids) * self.repetitions
        )
        if (
            expected_cost != THEORETICAL_APPLICATION_COST_USD
            or self.theoretical_application_cost_usd != expected_cost
        ):
            raise ValueError("formal holdout application plan must equal USD 32.00")
        expected_pairs = tuple(
            (arm_id, task_id) for arm_id in self.arm_ids for task_id in self.task_ids
        )
        actual_pairs = tuple((slot.arm_id, slot.task_id) for slot in self.run_slots)
        if actual_pairs != expected_pairs or len({slot.run_id for slot in self.run_slots}) != 32:
            raise ValueError("formal holdout must contain exactly 32 stable run slots")
        _validate_content_digest(self)
        return self


class PaidHoldoutAuthorization(StrictModel):
    """Separate user consent bound to one exact P7-S6 plan and overshoot warning."""

    schema_version: Literal[1] = SCHEMA_VERSION
    authorization_version: Literal["holdout-authorization-v1"] = AUTHORIZATION_VERSION
    authorization_id: Identifier
    experiment_id: Identifier
    matrix_sha256: Sha256
    freeze_marker_sha256: Sha256
    plan_sha256: Sha256
    approved_by: Identifier
    approved_at: AwareDatetime
    paid_execution_approved: Literal[True]
    max_total_cost_usd: Decimal
    final_in_flight_response_overshoot_acknowledged: Literal[True]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_authorization(self) -> Self:
        if self.max_total_cost_usd != THEORETICAL_APPLICATION_COST_USD:
            raise ValueError("formal holdout authorization must approve exactly USD 32.00")
        _validate_content_digest(self)
        return self


class HoldoutCommandTimeoutBinding(StrictModel):
    """One declared command timeout and its deterministic formal projection."""

    phase: Literal["setup", "test"]
    command_index: int = Field(ge=0, strict=True)
    declared_timeout_seconds: int = Field(gt=0, le=3_600, strict=True)
    effective_timeout_seconds: int = Field(gt=0, le=300, strict=True)
    capped_by_sandbox_policy: bool

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        expected = min(self.declared_timeout_seconds, COMMAND_TIMEOUT_CAP_SECONDS)
        if self.effective_timeout_seconds != expected:
            raise ValueError("effective command timeout does not match the frozen cap")
        if self.capped_by_sandbox_policy != (
            self.declared_timeout_seconds > COMMAND_TIMEOUT_CAP_SECONDS
        ):
            raise ValueError("command timeout cap evidence is inconsistent")
        return self


class HoldoutEnvironmentBinding(StrictModel):
    """Evaluator-qualified checkout and immutable image for one frozen task."""

    task_id: TaskId
    base_commit: GitSha
    python_version: str = Field(pattern=r"^3\.(?:9|10|11|12)(?:\.[0-9]+)?$")
    checkout_relative_path: str
    checkout_commit: GitSha
    sandbox_image_reference: str
    platform: Literal["linux/amd64", "linux/arm64"]
    setup_network: Literal["none"] = "none"
    agent_network: Literal["none"] = "none"
    timeout_contract_version: Literal["holdout-command-timeout-contract-v1"] = (
        TIMEOUT_CONTRACT_VERSION
    )
    task_timeout_seconds: int = Field(gt=0, le=3_600, strict=True)
    command_timeout_cap_seconds: Literal[300] = COMMAND_TIMEOUT_CAP_SECONDS
    setup_command_timeouts: tuple[HoldoutCommandTimeoutBinding, ...]
    test_command_timeouts: tuple[HoldoutCommandTimeoutBinding, ...]
    timeout_projection_sha256: Sha256
    setup_commands_passed_offline: Literal[True]
    baseline_failed: Literal[True]
    gold_passed: Literal[True]
    gold_visible_to_agent: Literal[False] = False

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if _RELATIVE_CHECKOUT.fullmatch(self.checkout_relative_path) is None:
            raise ValueError("holdout checkout path is outside the dedicated root")
        if self.checkout_commit != self.base_commit:
            raise ValueError("holdout checkout must be at the exact base commit")
        if _DIGEST_IMAGE.fullmatch(self.sandbox_image_reference) is None:
            raise ValueError("holdout task image must use an immutable digest")
        if tuple(item.command_index for item in self.setup_command_timeouts) != tuple(
            range(len(self.setup_command_timeouts))
        ) or any(item.phase != "setup" for item in self.setup_command_timeouts):
            raise ValueError("setup timeout bindings are unordered or mislabeled")
        if tuple(item.command_index for item in self.test_command_timeouts) != tuple(
            range(len(self.test_command_timeouts))
        ) or any(item.phase != "test" for item in self.test_command_timeouts):
            raise ValueError("test timeout bindings are unordered or mislabeled")
        payload = _timeout_projection_payload(
            task_id=self.task_id,
            task_timeout_seconds=self.task_timeout_seconds,
            setup=self.setup_command_timeouts,
            test=self.test_command_timeouts,
        )
        if _canonical_sha256(payload) != self.timeout_projection_sha256:
            raise ValueError("timeout projection digest does not match")
        return self


class HoldoutEnvironmentManifest(StrictModel):
    """All eight evaluator-qualified environments; partial manifests are invalid."""

    schema_version: Literal[1] = SCHEMA_VERSION
    environment_version: Literal["holdout-environments-v1"] = ENVIRONMENT_VERSION
    experiment_id: Identifier
    matrix_sha256: Sha256
    dataset_manifest_sha256: Sha256
    execution_harness: Literal["swe-bench"] = "swe-bench"
    execution_harness_revision: GitSha
    controller_commit: GitSha
    created_at: AwareDatetime
    compatibility_preflight_raw_sha256: Sha256
    compatibility_preflight_content_sha256: Sha256
    registry_manifest_raw_sha256: Sha256
    registry_manifest_content_sha256: Sha256
    timeout_contract_version: Literal["holdout-command-timeout-contract-v1"] = (
        TIMEOUT_CONTRACT_VERSION
    )
    sandbox_command_timeout_seconds: Literal[300] = COMMAND_TIMEOUT_CAP_SECONDS
    source_command_contract_sha256: Sha256
    runtime_command_contract_sha256: Sha256
    bindings: tuple[HoldoutEnvironmentBinding, ...]
    paid_requests_made: Literal[0] = 0
    gold_visible_to_agent: Literal[False] = False
    formal_environment_ready: Literal[True] = True
    formal_blockers: tuple[()] = ()
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if tuple(item.task_id for item in self.bindings) != EXPECTED_TASK_IDS:
            raise ValueError("holdout environment manifest must cover the exact task set")
        if len({item.sandbox_image_reference for item in self.bindings}) != 8:
            raise ValueError("holdout environment images must be task-specific")
        if not any(
            timeout.capped_by_sandbox_policy
            for binding in self.bindings
            for timeout in (*binding.setup_command_timeouts, *binding.test_command_timeouts)
        ):
            raise ValueError("timeout contract did not record any capped command")
        _validate_content_digest(self)
        return self


def build_holdout_timeout_bindings(
    task: TaskSpec,
) -> tuple[
    tuple[HoldoutCommandTimeoutBinding, ...],
    tuple[HoldoutCommandTimeoutBinding, ...],
]:
    """Bind every frozen command to the already-frozen 300-second Sandbox ceiling."""

    def bind(
        commands: tuple[CommandSpec, ...],
        phase: Literal["setup", "test"],
    ) -> tuple[HoldoutCommandTimeoutBinding, ...]:
        bindings: list[HoldoutCommandTimeoutBinding] = []
        for index, command in enumerate(commands):
            if command.timeout_seconds is None:
                raise HoldoutControlError(HoldoutErrorCode.INVALID_CONFIGURATION)
            declared = command.timeout_seconds
            bindings.append(
                HoldoutCommandTimeoutBinding(
                    phase=phase,
                    command_index=index,
                    declared_timeout_seconds=declared,
                    effective_timeout_seconds=min(declared, COMMAND_TIMEOUT_CAP_SECONDS),
                    capped_by_sandbox_policy=declared > COMMAND_TIMEOUT_CAP_SECONDS,
                )
            )
        return tuple(bindings)

    return bind(task.setup_commands, "setup"), bind(task.test_commands, "test")


def holdout_timeout_projection_sha256(task: TaskSpec) -> Sha256:
    """Hash the timeout-only projection without duplicating issue or evaluator Gold."""

    setup, test = build_holdout_timeout_bindings(task)
    return _canonical_sha256(
        _timeout_projection_payload(
            task_id=task.task_id,
            task_timeout_seconds=task.timeout_seconds,
            setup=setup,
            test=test,
        )
    )


def project_holdout_task_timeouts(task: TaskSpec) -> TaskSpec:
    """Project command timeouts only; preserve all Agent-visible task semantics."""

    setup, test = build_holdout_timeout_bindings(task)

    def project(
        commands: tuple[CommandSpec, ...],
        bindings: tuple[HoldoutCommandTimeoutBinding, ...],
    ) -> tuple[CommandSpec, ...]:
        return tuple(
            command.model_copy(update={"timeout_seconds": binding.effective_timeout_seconds})
            for command, binding in zip(commands, bindings, strict=True)
        )

    return task.model_copy(
        update={
            "setup_commands": project(task.setup_commands, setup),
            "test_commands": project(task.test_commands, test),
        }
    )


def holdout_command_contract_sha256(tasks: tuple[TaskSpec, ...]) -> Sha256:
    """Hash only task command semantics; issue text and evaluator Gold are excluded."""

    payload = tuple(
        {
            "task_id": task.task_id,
            "task_timeout_seconds": task.timeout_seconds,
            "setup_commands": task.setup_commands,
            "test_commands": task.test_commands,
        }
        for task in tasks
    )
    return _canonical_sha256(payload)


def _timeout_projection_payload(
    *,
    task_id: str,
    task_timeout_seconds: int,
    setup: tuple[HoldoutCommandTimeoutBinding, ...],
    test: tuple[HoldoutCommandTimeoutBinding, ...],
) -> dict[str, object]:
    return {
        "timeout_contract_version": TIMEOUT_CONTRACT_VERSION,
        "task_id": task_id,
        "task_timeout_seconds": task_timeout_seconds,
        "command_timeout_cap_seconds": COMMAND_TIMEOUT_CAP_SECONDS,
        "setup_command_timeouts": setup,
        "test_command_timeouts": test,
    }


class HoldoutRuntimeObservation(StrictModel):
    """Gold-free summary of existing evaluator environment evidence depth."""

    task_id: TaskId
    python_version: str
    dataset_preflight_status: DatasetPreflightStatus
    runtime_gold_verified: bool
    checkout_ready: bool
    task_image_ready: bool


class HoldoutPreflightCheck(StrictModel):
    """One deterministic allow/block/not-run decision."""

    check_id: Identifier
    status: ExperimentPreflightCheckStatus
    error_code: HoldoutErrorCode | None = None

    @model_validator(mode="after")
    def validate_check(self) -> Self:
        if self.status is ExperimentPreflightCheckStatus.PASSED:
            if self.error_code is not None:
                raise ValueError("passed holdout check cannot contain an error")
        elif self.error_code is None:
            raise ValueError("blocked or not-run holdout check requires a stable reason")
        return self


class HoldoutPreflight(StrictModel):
    """Machine-readable gate proving that no paid call was made while blocked."""

    schema_version: Literal[1] = SCHEMA_VERSION
    preflight_version: Literal["holdout-preflight-v1"] = PREFLIGHT_VERSION
    plan_sha256: Sha256
    checked_at: AwareDatetime
    controller_commit: GitSha
    ready_for_paid_holdout: bool
    paid_requests_made: Literal[0] = 0
    gold_visible_to_agent: Literal[False] = False
    runtime_verified_task_count: int = Field(ge=0, le=8, strict=True)
    required_task_count: Literal[8] = 8
    checks: tuple[HoldoutPreflightCheck, ...]
    task_observations: tuple[HoldoutRuntimeObservation, ...]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_preflight(self) -> Self:
        check_ids = tuple(item.check_id for item in self.checks)
        if check_ids != tuple(sorted(check_ids)) or len(set(check_ids)) != len(check_ids):
            raise ValueError("holdout preflight checks must be sorted and unique")
        ready = all(item.status is ExperimentPreflightCheckStatus.PASSED for item in self.checks)
        if ready != self.ready_for_paid_holdout:
            raise ValueError("holdout readiness does not match its checks")
        if tuple(item.task_id for item in self.task_observations) != EXPECTED_TASK_IDS:
            raise ValueError("holdout observations must cover the exact task set")
        if self.runtime_verified_task_count != sum(
            item.runtime_gold_verified for item in self.task_observations
        ):
            raise ValueError("runtime-verified count does not match observations")
        _validate_content_digest(self)
        return self


def build_holdout_application_plan(
    *,
    matrix: ExperimentMatrix,
    marker: FormalFreezeMarker,
    dataset_manifest: FrozenDatasetManifest,
    controller_commit: str,
    created_at: datetime,
) -> HoldoutApplicationPlan:
    """Bind the frozen four-by-eight matrix without loading source rows or gold patches."""

    _validate_frozen_identity(matrix, marker, dataset_manifest)
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise HoldoutControlError(HoldoutErrorCode.INVALID_CONFIGURATION)
    if re.fullmatch(r"[0-9a-f]{40}", controller_commit) is None:
        raise HoldoutControlError(HoldoutErrorCode.INVALID_CONFIGURATION)
    arm_ids = tuple(sorted(arm.arm_id for arm in matrix.arms if arm.enabled_by_default))
    if arm_ids != EXPECTED_ARM_IDS:
        raise HoldoutControlError(HoldoutErrorCode.DEFAULT_ARM_SET_MISMATCH)
    declared = next(item for item in matrix.dataset.task_sets if item.split is DatasetSplit.HOLDOUT)
    if declared.task_ids != EXPECTED_TASK_IDS:
        raise HoldoutControlError(HoldoutErrorCode.HOLDOUT_TASK_SET_MISMATCH)
    cases = {case.case_id: case for case in dataset_manifest.external_cases}
    if tuple(sorted(cases)) != EXPECTED_TASK_IDS:
        raise HoldoutControlError(HoldoutErrorCode.HOLDOUT_TASK_SET_MISMATCH)
    plan_id = f"p7-holdout-{matrix.content_sha256[:12]}-{marker.source_commit[:12]}"
    slots = tuple(
        HoldoutRunSlot(
            arm_id=arm_id,
            task_id=task_id,
            run_id=uuid5(
                _RUN_NAMESPACE,
                f"{plan_id}:{arm_id}:{task_id}:{matrix.holdout_repetitions}",
            ),
            configuration_sha256=select_experiment_arm(matrix, arm_id).configuration_sha256,
            base_commit=cases[task_id].base_commit,
            python_version=cases[task_id].python_version,
            output_path=f"results/p7-holdout/raw/{plan_id}/{arm_id}/{task_id}",
        )
        for arm_id in arm_ids
        for task_id in declared.task_ids
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "plan_version": PLAN_VERSION,
        "plan_id": plan_id,
        "experiment_id": matrix.experiment_id,
        "matrix_sha256": matrix.content_sha256,
        "freeze_marker_sha256": marker.content_sha256,
        "frozen_source_commit": marker.source_commit,
        "controller_commit": controller_commit,
        "dependency_lock_sha256": marker.dependency_lock_sha256,
        "dataset_manifest_sha256": marker.dataset_manifest_sha256,
        "prompt_versions": marker.prompt_versions,
        "model": marker.model,
        "embedding_model": marker.embedding_model,
        "embedding_revision": marker.embedding_revision,
        "embedding_code_revision": marker.embedding_code_revision,
        "frozen_sandbox_image_reference": marker.sandbox_image_reference,
        "created_at": created_at,
        "split": DatasetSplit.HOLDOUT,
        "arm_ids": arm_ids,
        "task_ids": declared.task_ids,
        "repetitions": matrix.holdout_repetitions,
        "per_task_cost_limit_usd": PER_TASK_COST_LIMIT_USD,
        "theoretical_application_cost_usd": THEORETICAL_APPLICATION_COST_USD,
        "final_in_flight_response_may_overshoot": True,
        "authorization_required": True,
        "aggregation_deferred_to": "P7-S7",
        "tuning_from_holdout_forbidden": True,
        "run_slots": slots,
    }
    return HoldoutApplicationPlan.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def build_paid_holdout_authorization(
    *,
    plan: HoldoutApplicationPlan,
    authorization_id: str,
    approved_by: str,
    approved_at: datetime,
    max_total_cost_usd: Decimal,
    overshoot_acknowledged: bool,
) -> PaidHoldoutAuthorization:
    """Create consent only from an exact USD 32 approval and explicit overshoot ack."""

    if max_total_cost_usd != THEORETICAL_APPLICATION_COST_USD:
        raise HoldoutControlError(HoldoutErrorCode.BUDGET_AUTHORIZATION_MISMATCH)
    if not overshoot_acknowledged:
        raise HoldoutControlError(HoldoutErrorCode.FINAL_RESPONSE_OVERSHOOT_UNACKNOWLEDGED)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "authorization_version": AUTHORIZATION_VERSION,
        "authorization_id": authorization_id,
        "experiment_id": plan.experiment_id,
        "matrix_sha256": plan.matrix_sha256,
        "freeze_marker_sha256": plan.freeze_marker_sha256,
        "plan_sha256": plan.content_sha256,
        "approved_by": approved_by,
        "approved_at": approved_at,
        "paid_execution_approved": True,
        "max_total_cost_usd": max_total_cost_usd,
        "final_in_flight_response_overshoot_acknowledged": True,
    }
    return PaidHoldoutAuthorization.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def build_holdout_preflight(
    *,
    root: Path,
    plan: HoldoutApplicationPlan,
    marker: FormalFreezeMarker,
    dataset_preflight: DatasetPreflightReport,
    checked_at: datetime,
    controller_source_clean: bool,
    frozen_runtime_matches: bool,
    authorization: PaidHoldoutAuthorization | None,
    environment_manifest: HoldoutEnvironmentManifest | None,
    provider_credential_available: bool | None = None,
    vector_backend_available: bool | None = None,
    embedding_snapshot_available: bool | None = None,
    controller_commit: str | None = None,
) -> HoldoutPreflight:
    """Evaluate all gates; downstream runtime checks remain not-run until prerequisites pass."""

    effective_controller_commit = controller_commit or plan.controller_commit
    if re.fullmatch(r"[0-9a-f]{40}", effective_controller_commit) is None:
        raise HoldoutControlError(HoldoutErrorCode.INVALID_CONFIGURATION)

    observations = _runtime_observations(dataset_preflight, environment_manifest)
    environment_matches = _environment_matches(plan, environment_manifest)
    checkout_ready = environment_matches and _checkouts_match(root, environment_manifest)
    images_ready = (
        environment_matches
        and all(
            _DIGEST_IMAGE.fullmatch(binding.sandbox_image_reference) is not None
            for binding in environment_manifest.bindings
        )
        if environment_manifest is not None
        else False
    )
    authorization_matches = _authorization_matches(plan, authorization)
    output_ready = all(not (root / slot.output_path).exists() for slot in plan.run_slots)
    runtime_evidence_ready = all(item.runtime_gold_verified for item in observations)
    checks: dict[str, HoldoutPreflightCheck] = {}

    def record(
        check_id: str,
        passed: bool | None,
        error_code: HoldoutErrorCode,
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
        checks[check_id] = HoldoutPreflightCheck(
            check_id=check_id,
            status=status,
            error_code=None if passed else error_code,
        )

    authorization_error = (
        HoldoutErrorCode.BUDGET_AUTHORIZATION_REQUIRED
        if authorization is None
        else HoldoutErrorCode.BUDGET_AUTHORIZATION_MISMATCH
    )
    record("authorization", authorization_matches, authorization_error)
    record("controller-source", controller_source_clean, HoldoutErrorCode.SOURCE_SNAPSHOT_UNCLEAN)
    record("frozen-runtime", frozen_runtime_matches, HoldoutErrorCode.FROZEN_RUNTIME_DRIFT)
    record(
        "freeze-identity",
        plan.freeze_marker_sha256 == marker.content_sha256,
        HoldoutErrorCode.FREEZE_IDENTITY_MISMATCH,
    )
    record("gold-boundary", _plan_is_gold_free(plan), HoldoutErrorCode.GOLD_BOUNDARY_VIOLATION)
    record("output-root", output_ready, HoldoutErrorCode.OUTPUT_CONFLICT)
    record("task-environments", environment_matches, HoldoutErrorCode.TASK_ENVIRONMENT_UNVERIFIED)
    record("task-checkouts", checkout_ready, HoldoutErrorCode.TASK_CHECKOUT_UNAVAILABLE)
    record("task-images", images_ready, HoldoutErrorCode.SANDBOX_IMAGE_UNAVAILABLE)
    record(
        "runtime-gold-validation",
        runtime_evidence_ready,
        HoldoutErrorCode.TASK_ENVIRONMENT_UNVERIFIED,
    )
    prerequisites = (
        authorization_matches and environment_matches and checkout_ready and images_ready
    )
    record(
        "provider-credential",
        provider_credential_available if prerequisites else None,
        HoldoutErrorCode.PROVIDER_CREDENTIAL_REQUIRED,
    )
    record(
        "vector-backend",
        vector_backend_available if prerequisites else None,
        HoldoutErrorCode.VECTOR_BACKEND_UNAVAILABLE,
    )
    record(
        "embedding-snapshot",
        embedding_snapshot_available if prerequisites else None,
        HoldoutErrorCode.EMBEDDING_SNAPSHOT_UNAVAILABLE,
    )
    ordered = tuple(checks[key] for key in sorted(checks))
    ready = all(item.status is ExperimentPreflightCheckStatus.PASSED for item in ordered)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "preflight_version": PREFLIGHT_VERSION,
        "plan_sha256": plan.content_sha256,
        "checked_at": checked_at,
        "controller_commit": effective_controller_commit,
        "ready_for_paid_holdout": ready,
        "paid_requests_made": 0,
        "gold_visible_to_agent": False,
        "runtime_verified_task_count": sum(item.runtime_gold_verified for item in observations),
        "required_task_count": 8,
        "checks": ordered,
        "task_observations": observations,
    }
    return HoldoutPreflight.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def write_control_artifact(path: Path, value: StrictModel) -> None:
    """Create one private, no-overwrite JSON control artifact."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value.model_dump_json(indent=2))
            stream.write("\n")
    except OSError:
        raise HoldoutControlError(HoldoutErrorCode.OUTPUT_CONFLICT) from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve(strict=True)
    matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
    marker = FormalFreezeMarker.model_validate_json((root / FREEZE_PATH).read_bytes())
    dataset_manifest = FrozenDatasetManifest.model_validate_json(
        (root / matrix.dataset.manifest_path).read_bytes()
    )
    dataset_preflight = DatasetPreflightReport.model_validate_json(
        (root / DATASET_PREFLIGHT_PATH).read_bytes()
    )
    controller_commit = _git(root, "rev-parse", "HEAD")
    plan = build_holdout_application_plan(
        matrix=matrix,
        marker=marker,
        dataset_manifest=dataset_manifest,
        controller_commit=controller_commit,
        created_at=datetime.now(UTC),
    )
    environment_manifest = _load_optional_environment(root)
    preflight = build_holdout_preflight(
        root=root,
        plan=plan,
        marker=marker,
        dataset_preflight=dataset_preflight,
        checked_at=plan.created_at,
        controller_source_clean=not bool(_git(root, "status", "--porcelain=v1")),
        frozen_runtime_matches=_frozen_runtime_matches(root, marker.source_commit),
        authorization=None,
        environment_manifest=environment_manifest,
    )
    write_control_artifact(root / PLAN_PATH, plan)
    write_control_artifact(root / PREFLIGHT_PATH, preflight)
    blocked = tuple(
        check.error_code.value
        for check in preflight.checks
        if check.status is not ExperimentPreflightCheckStatus.PASSED
        and check.error_code is not None
    )
    print(
        f"plan_id={plan.plan_id} slots={len(plan.run_slots)} "
        f"application_limit_usd={plan.theoretical_application_cost_usd} "
        f"ready={str(preflight.ready_for_paid_holdout).lower()} "
        f"paid_requests={preflight.paid_requests_made} "
        f"blocked_checks={len(blocked)}"
    )
    return 0


def _validate_frozen_identity(
    matrix: ExperimentMatrix,
    marker: FormalFreezeMarker,
    dataset_manifest: FrozenDatasetManifest,
) -> None:
    arm_hashes = {
        arm_id: select_experiment_arm(matrix, arm_id).configuration_sha256
        for arm_id in EXPECTED_ARM_IDS
    }
    if (
        marker.experiment_id != matrix.experiment_id
        or marker.matrix_version != matrix.matrix_version
        or marker.matrix_sha256 != matrix.content_sha256
        or marker.dependency_lock_sha256 != matrix.dependency_lock_sha256
        or marker.dataset_manifest_sha256 != matrix.dataset.manifest_content_sha256
        or marker.prompt_versions != matrix.prompt_versions
        or marker.model != matrix.shared_settings.values.get("model")
        or marker.embedding_model != matrix.shared_settings.values.get("embedding_model")
        or marker.embedding_revision != matrix.shared_settings.values.get("embedding_revision")
        or marker.embedding_code_revision
        != matrix.shared_settings.values.get("embedding_code_revision")
        or marker.sandbox_image_reference != matrix.sandbox_image_reference
        or marker.arm_configuration_sha256 != arm_hashes
    ):
        raise HoldoutControlError(HoldoutErrorCode.FREEZE_IDENTITY_MISMATCH)
    if (
        dataset_manifest.content_sha256 != matrix.dataset.manifest_content_sha256
        or dataset_manifest.dataset_id != matrix.dataset.dataset_id
    ):
        raise HoldoutControlError(HoldoutErrorCode.DATASET_MISMATCH)


def _runtime_observations(
    report: DatasetPreflightReport,
    environment_manifest: HoldoutEnvironmentManifest | None,
) -> tuple[HoldoutRuntimeObservation, ...]:
    evidence = {entry.case_id: entry for entry in report.entries}
    bindings = (
        {binding.task_id: binding for binding in environment_manifest.bindings}
        if environment_manifest is not None
        else {}
    )
    if any(task_id not in evidence for task_id in EXPECTED_TASK_IDS):
        raise HoldoutControlError(HoldoutErrorCode.DATASET_MISMATCH)
    return tuple(
        HoldoutRuntimeObservation(
            task_id=task_id,
            python_version=evidence[task_id].python_version,
            dataset_preflight_status=evidence[task_id].status,
            runtime_gold_verified=(
                (
                    bindings[task_id].baseline_failed
                    and bindings[task_id].gold_passed
                    and bindings[task_id].setup_commands_passed_offline
                )
                if task_id in bindings
                else (
                    evidence[task_id].status is DatasetPreflightStatus.BASELINE_FAILED_GOLD_PASSED
                )
            ),
            checkout_ready=task_id in bindings,
            task_image_ready=task_id in bindings,
        )
        for task_id in EXPECTED_TASK_IDS
    )


def _environment_matches(
    plan: HoldoutApplicationPlan,
    manifest: HoldoutEnvironmentManifest | None,
) -> bool:
    if manifest is None:
        return False
    expected = {slot.task_id: (slot.base_commit, slot.python_version) for slot in plan.run_slots}
    return (
        manifest.experiment_id == plan.experiment_id
        and manifest.matrix_sha256 == plan.matrix_sha256
        and manifest.dataset_manifest_sha256 == plan.dataset_manifest_sha256
        and tuple(binding.task_id for binding in manifest.bindings) == plan.task_ids
        and all(
            expected[binding.task_id] == (binding.base_commit, binding.python_version)
            for binding in manifest.bindings
        )
    )


def _checkouts_match(root: Path, manifest: HoldoutEnvironmentManifest | None) -> bool:
    if manifest is None:
        return False
    for binding in manifest.bindings:
        checkout = root / binding.checkout_relative_path
        try:
            if _git(checkout, "rev-parse", "HEAD") != binding.base_commit:
                return False
            if _git(checkout, "status", "--porcelain=v1"):
                return False
        except (OSError, subprocess.CalledProcessError):
            return False
    return True


def _authorization_matches(
    plan: HoldoutApplicationPlan,
    authorization: PaidHoldoutAuthorization | None,
) -> bool | None:
    if authorization is None:
        return None
    return (
        authorization.experiment_id == plan.experiment_id
        and authorization.matrix_sha256 == plan.matrix_sha256
        and authorization.freeze_marker_sha256 == plan.freeze_marker_sha256
        and authorization.plan_sha256 == plan.content_sha256
        and authorization.max_total_cost_usd == plan.theoretical_application_cost_usd
    )


def _load_optional_environment(root: Path) -> HoldoutEnvironmentManifest | None:
    path = root / ENVIRONMENT_PATH
    if not path.is_file():
        return None
    try:
        return HoldoutEnvironmentManifest.model_validate_json(path.read_bytes())
    except (OSError, ValidationError):
        raise HoldoutControlError(HoldoutErrorCode.TASK_ENVIRONMENT_UNVERIFIED) from None


def _frozen_runtime_matches(root: Path, source_commit: str) -> bool:
    completed = subprocess.run(
        ("git", "diff", "--quiet", source_commit, "--", *_FROZEN_RUNTIME_PATHS),
        cwd=root,
        check=False,
    )
    return completed.returncode == 0


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


def _plan_is_gold_free(plan: HoldoutApplicationPlan) -> bool:
    forbidden = frozenset({"fail_to_pass", "gold_files", "gold_patch", "issue_body"})

    def contains(value: object) -> bool:
        if isinstance(value, dict):
            return any(
                str(key).lower() in forbidden or contains(item) for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(contains(item) for item in value)
        return False

    return not contains(plan.model_dump(mode="python"))


def _validate_content_digest(value: StrictModel) -> None:
    payload = value.model_dump(mode="python", exclude={"content_sha256"})
    if _canonical_sha256(payload) != getattr(value, "content_sha256", None):
        raise ValueError("holdout control artifact digest does not match")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HoldoutControlError, ValidationError) as error:
        raise SystemExit(str(error)) from None
