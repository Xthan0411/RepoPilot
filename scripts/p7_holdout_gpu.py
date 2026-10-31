"""Freeze the GPU-cache execution profile without creating paid authorization.

The historical matrix is unchanged; the new source/cache/CPU profile is explicit.
Consent must bind this new plan and include the old USD 1 unknown-cost reserve.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self
from uuid import UUID, uuid5

from pydantic import AwareDatetime, JsonValue, TypeAdapter, model_validator

from repopilot.config import DEFAULT_DATABASE_URL
from repopilot.embeddings import embedding_identity_from_settings
from repopilot.enums import DatasetSplit
from repopilot.evals import select_experiment_arm
from repopilot.evals.dataset import FrozenDatasetManifest
from repopilot.schemas import GitSha, Identifier, Sha256, StrictModel, TaskId
from scripts import p7_holdout_readiness as legacy
from scripts import p7_repository_workload_preflight as workload
from scripts.p7_gpu_execution_profile import (
    QUALIFICATION_RAW_SHA256,
    verify_qualification,
)
from scripts.p7_holdout_agent_inputs import (
    AGENT_INPUTS_PATH,
    HoldoutAgentInputs,
    _canonical_sha256,
    _validate_content_digest,
)
from scripts.p7_holdout_control import (
    ENVIRONMENT_PATH,
    EXPECTED_ARM_IDS,
    EXPECTED_TASK_IDS,
    HoldoutEnvironmentManifest,
    HoldoutRunSlot,
    write_control_artifact,
)
from scripts.p7_holdout_requalification import (
    PLAN_V2_PATH,
    HoldoutApplicationPlanV2,
    _git,
    _sealed,
    _sha,
)
from scripts.p7_holdout_requalification import (
    _read as _read,
)
from scripts.p7_holdout_v2 import verify_runtime_environment as verify_runtime_environment

FREEZE_PATH = Path("results/p7-holdout/gpu-runtime-freeze-v3.json")
PLAN_PATH = Path("results/p7-holdout/plan-v3.json")
AUTHORIZATION_PATH = Path("results/p7-holdout/authorization-v3.json")
READINESS_PATH = Path("results/p7-holdout/runtime-readiness-v3.json")
READINESS_V3_PATH = READINESS_PATH
TEMPLATE_RAW_SHA256 = "b0654760cea101d8d1b295ca1c2c56e05e924a88f8d70c730e326706414253a2"
RUNTIME_PROFILE: Literal["gpu-corpus-cache-cpu-query-v1"] = "gpu-corpus-cache-cpu-query-v1"
_RUN_NAMESPACE = UUID("a181b55f-8080-5a4d-9476-df7513683129")
REQUIRED_EXECUTION_PATHS = (
    "scripts/p7_holdout_gpu.py",
    "scripts/p7_gpu_execution_profile.py",
    "scripts/p7_holdout_readiness_v3.py",
    "scripts/p7_holdout_runner_v3.py",
    "scripts/p7_holdout_runner.py",
    "scripts/p7_gpu_cache_acceptance.py",
    "scripts/p7_gpu_cpu_probe.py",
    "scripts/p7_repository_workload_preflight.py",
)
READINESS_CHECKS = (
    "authorization",
    "checkouts",
    "controller",
    "embedding",
    "gpu-cache",
    "images",
    "local-dependencies",
    "output-paths",
    "provider-presence",
    "qualified-runtime",
    "vector-backend",
)


class CarriedUnknownAttempt(StrictModel):
    """A conservative declaration, not a fabricated import of the old raw evidence."""

    plan_id: Literal["p7-holdout-v2-d26d33699d59b6fc"] = "p7-holdout-v2-d26d33699d59b6fc"
    plan_raw_sha256: Sha256 = TEMPLATE_RAW_SHA256
    run_id: Literal["c06c245c-3fe4-5df5-ba6b-34bda2ce00ce"] = "c06c245c-3fe4-5df5-ba6b-34bda2ce00ce"
    output_path: Literal[
        "results/p7-holdout/raw/p7-holdout-v2-d26d33699d59b6fc/dense-only-v1/django__django-15902"
    ] = "results/p7-holdout/raw/p7-holdout-v2-d26d33699d59b6fc/dense-only-v1/django__django-15902"
    status: Literal["technical_failure"] = "technical_failure"
    evidence_basis: Literal["user_reported_original_raw_not_imported"] = (
        "user_reported_original_raw_not_imported"
    )
    original_raw_evidence_revalidated: Literal[False] = False
    provider_request_count: Literal["unknown"] = "unknown"
    reserved_unknown_cost_usd: Decimal = Decimal("1.00")

    @model_validator(mode="after")
    def validate_reserve(self) -> Self:
        if (
            self.reserved_unknown_cost_usd != Decimal("1.00")
            or self.plan_raw_sha256 != TEMPLATE_RAW_SHA256
        ):
            raise ValueError("old unknown reserve must remain USD 1")
        return self


class GpuFreeze(StrictModel):
    schema_version: Literal[3] = 3
    freeze_version: Literal["gpu-runtime-freeze-v3"] = "gpu-runtime-freeze-v3"
    controller_commit: GitSha
    frozen_at: AwareDatetime
    execution_source_objects: dict[str, GitSha]
    execution_source_sha256: Sha256
    qualification_raw_sha256: Sha256
    qualification_controller_commit: GitSha
    qualification_execution_source_sha256: Sha256
    qualification_checked_at: AwareDatetime
    runtime_profile: Literal["gpu-corpus-cache-cpu-query-v1"] = RUNTIME_PROFILE
    runtime_profile_sha256: Sha256
    cpu_runtime: dict[str, JsonValue]
    cache_evidence: dict[str, JsonValue]
    template_plan_raw_sha256: Sha256 = TEMPLATE_RAW_SHA256
    matrix_sha256: Sha256
    agent_inputs_raw_sha256: Sha256
    environment_raw_sha256: Sha256
    test_inputs_raw_sha256: Sha256
    canary_raw_sha256: Sha256
    dependency_lock_sha256: Sha256
    dataset_manifest_sha256: Sha256
    task_ids: tuple[TaskId, ...] = EXPECTED_TASK_IDS
    carried_attempt: CarriedUnknownAttempt = CarriedUnknownAttempt()
    current_source_calibration_performed: Literal[False] = False
    new_matrix_claimed: Literal[False] = False
    provider_requests_made: Literal[0] = 0
    paid_requests_made: Literal[0] = 0
    formal_environment_ready: Literal[False] = False
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_freeze(self) -> Self:
        if (
            self.qualification_raw_sha256 != QUALIFICATION_RAW_SHA256
            or self.template_plan_raw_sha256 != TEMPLATE_RAW_SHA256
            or self.execution_source_sha256 != _canonical_sha256(self.execution_source_objects)
            or tuple(self.execution_source_objects) != tuple(sorted(self.execution_source_objects))
            or not set(REQUIRED_EXECUTION_PATHS).issubset(self.execution_source_objects)
            or self.task_ids != EXPECTED_TASK_IDS
            or self.qualification_checked_at > self.frozen_at
            or self.runtime_profile_sha256 != _profile_digest(self.cpu_runtime, self.cache_evidence)
        ):
            raise ValueError("GPU freeze identity mismatch")
        _validate_content_digest(self)
        return self


class GpuPlan(StrictModel):
    """The same 32 experiment slots under a newly frozen runtime and total budget."""

    schema_version: Literal[3] = 3
    plan_version: Literal["holdout-application-plan-v3"] = "holdout-application-plan-v3"
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
    carried_unknown_cost_usd: Decimal = Decimal("1.00")
    application_budget_usd: Decimal = Decimal("32.00")
    new_execution_budget_usd: Decimal = Decimal("31.00")
    carry_forward_sha256: Sha256
    runtime_profile: Literal["gpu-corpus-cache-cpu-query-v1"] = RUNTIME_PROFILE
    runtime_profile_sha256: Sha256
    runtime_profile_overrides_historical_matrix: Literal[True] = True
    new_matrix_claimed: Literal[False] = False
    all_slots_completion_within_budget_guaranteed: Literal[False] = False
    final_in_flight_response_may_overshoot: Literal[True] = True
    authorization_required: Literal[True] = True
    aggregation_deferred_to: Literal["P7-S7"] = "P7-S7"
    tuning_from_holdout_forbidden: Literal[True] = True
    run_slots: tuple[HoldoutRunSlot, ...]
    test_protocol_version: Literal["isolated-regression-tests-v1"]
    canary_raw_sha256: Sha256
    environment_raw_sha256: Sha256
    agent_inputs_raw_sha256: Sha256
    test_inputs_raw_sha256: Sha256
    qualification_raw_sha256: Sha256
    ready_for_paid_holdout: Literal[False] = False
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if (
            self.arm_ids != EXPECTED_ARM_IDS
            or self.task_ids != EXPECTED_TASK_IDS
            or self.per_task_cost_limit_usd != Decimal("1.00")
            or self.theoretical_application_cost_usd != Decimal("32.00")
            or self.carried_unknown_cost_usd != Decimal("1.00")
            or self.application_budget_usd != Decimal("32.00")
            or self.new_execution_budget_usd != Decimal("31.00")
            or self.qualification_raw_sha256 != QUALIFICATION_RAW_SHA256
            or self.plan_id != f"p7-holdout-v3-{self.freeze_marker_sha256[:16]}"
            or self.experiment_id != "repopilot-p7-v3"
            or tuple((s.arm_id, s.task_id) for s in self.run_slots)
            != tuple((arm, task) for arm in self.arm_ids for task in self.task_ids)
            or any(
                slot.run_id != _run_id(self.plan_id, slot.arm_id, slot.task_id)
                or slot.output_path != _output_path(self.plan_id, slot.arm_id, slot.task_id)
                for slot in self.run_slots
            )
        ):
            raise ValueError("GPU plan identity or budget mismatch")
        _validate_content_digest(self)
        return self


class GpuAuthorization(StrictModel):
    authorization_version: Literal["holdout-authorization-v3"] = "holdout-authorization-v3"
    authorization_id: Identifier
    approved_by: Literal["ethan"] = "ethan"
    approved_at: AwareDatetime
    experiment_id: Identifier
    plan_id: Identifier
    plan_sha256: Sha256
    plan_raw_sha256: Sha256
    freeze_marker_sha256: Sha256
    freeze_raw_sha256: Sha256
    matrix_sha256: Sha256
    runtime_profile_sha256: Sha256
    carry_forward_sha256: Sha256
    carried_unknown_cost_usd: Decimal
    max_total_cost_usd: Decimal
    new_execution_budget_usd: Decimal
    paid_execution_approved: Literal[True]
    final_in_flight_response_overshoot_acknowledged: Literal[True]
    readiness_required_before_paid: Literal[True]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_authorization(self) -> Self:
        if (
            self.max_total_cost_usd != Decimal("32.00")
            or self.carried_unknown_cost_usd != Decimal("1.00")
            or self.new_execution_budget_usd != Decimal("31.00")
            or self.authorization_id != f"{self.plan_id}-usd32-ethan"
        ):
            raise ValueError("GPU consent must preserve the total budget and old reserve")
        _validate_content_digest(self)
        return self


@dataclass(frozen=True)
class GpuControls:
    freeze: GpuFreeze
    plan: GpuPlan
    authorization: GpuAuthorization
    environment: HoldoutEnvironmentManifest
    agent_inputs: HoldoutAgentInputs
    controller_commit: str
    freeze_raw: bytes
    plan_raw: bytes
    authorization_raw: bytes
    environment_raw: bytes
    agent_inputs_raw: bytes


V3Controls = GpuControls


class GpuReadiness(StrictModel):
    schema_version: Literal[3] = 3
    readiness_version: Literal["holdout-runtime-readiness-v3"] = "holdout-runtime-readiness-v3"
    experiment_id: Identifier
    controller_commit: GitSha
    checked_at: AwareDatetime
    host_platform: Literal["linux/amd64"] = "linux/amd64"
    plan_raw_sha256: Sha256
    plan_content_sha256: Sha256
    authorization_raw_sha256: Sha256
    authorization_content_sha256: Sha256
    environment_raw_sha256: Sha256
    environment_content_sha256: Sha256
    freeze_raw_sha256: Sha256
    freeze_content_sha256: Sha256
    agent_inputs_raw_sha256: Sha256
    test_inputs_raw_sha256: Sha256
    canary_raw_sha256: Sha256
    qualified_runtime_commit: GitSha
    semantic_objects_sha256: Sha256
    qualification_raw_sha256: Sha256
    runtime_profile_sha256: Sha256
    execution_controller_objects: dict[str, GitSha]
    allowed_secret_environment: tuple[str, ...] = legacy.ALLOWED_SECRET_ENVIRONMENT
    provider: legacy.ProviderCredentialEvidence
    embedding: legacy.EmbeddingReadinessEvidence
    cpu_profile: dict[str, JsonValue]
    gpu_cache: dict[str, JsonValue]
    vector_backend: legacy.VectorReadinessEvidence
    images: tuple[legacy.ReadinessImageEvidence, ...]
    completed_checks: tuple[str, ...]
    paid_requests_made: Literal[0] = 0
    gold_visible_to_agent: Literal[False] = False
    ready_for_paid_holdout: Literal[True] = True
    formal_blockers: tuple[()] = ()
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_readiness(self) -> Self:
        if (
            self.completed_checks != READINESS_CHECKS
            or self.allowed_secret_environment != legacy.ALLOWED_SECRET_ENVIRONMENT
            or tuple(image.task_id for image in self.images) != EXPECTED_TASK_IDS
            or tuple(self.execution_controller_objects)
            != tuple(sorted(self.execution_controller_objects))
            or not set(REQUIRED_EXECUTION_PATHS).issubset(self.execution_controller_objects)
            or self.qualification_raw_sha256 != QUALIFICATION_RAW_SHA256
            or _profile_digest(self.cpu_profile, self.gpu_cache) != self.runtime_profile_sha256
        ):
            raise ValueError("GPU readiness incomplete or profile mismatch")
        _validate_content_digest(self)
        return self


def _profile_digest(cpu: dict[str, JsonValue], cache: dict[str, JsonValue]) -> str:
    return _canonical_sha256({"runtime_profile": RUNTIME_PROFILE, "cpu": cpu, "cache": cache})


def _run_id(plan_id: str, arm: str, task: str) -> UUID:
    return uuid5(_RUN_NAMESPACE, f"{plan_id}:{arm}:{task}:1")


def _output_path(plan_id: str, arm: str, task: str) -> str:
    return f"results/p7-holdout/raw/{plan_id}/{arm}/{task}"


def verify_execution_controller(root: Path) -> dict[str, str]:
    _, objects = workload.verify_source(root)
    if not set(REQUIRED_EXECUTION_PATHS).issubset(objects):
        raise ValueError("GPU execution controller incomplete")
    return objects


def _objects_at(root: Path, commit: str) -> dict[str, str]:
    objects: dict[str, str] = {}
    for entry in _git(root, "ls-tree", "-rz", commit, "--", *workload.SOURCE_PATHS).split("\0"):
        if not entry:
            continue
        metadata, path = entry.split("\t", 1)
        mode, kind, blob = metadata.split()
        if kind != "blob" or mode not in ("100644", "100755"):
            raise ValueError("unsupported frozen source entry")
        objects[path] = blob
    return dict(sorted(objects.items()))


def verify_frozen_source(root: Path, freeze: GpuFreeze) -> None:
    current, objects = workload.verify_source(root)
    _git(root, "merge-base", "--is-ancestor", freeze.controller_commit, current)
    if (
        objects != freeze.execution_source_objects
        or _objects_at(root, freeze.controller_commit) != objects
    ):
        raise ValueError("GPU frozen source drift")


def build_plan(
    root: Path, *, frozen_at: datetime, recorded_controller: str | None = None
) -> tuple[GpuFreeze, GpuPlan]:
    """Build unpaid controls from pinned qualification, without loading source rows."""
    root = root.resolve(strict=True)
    report = verify_qualification(root)
    current, objects = workload.verify_source(root)
    controller = recorded_controller or current
    _git(root, "merge-base", "--is-ancestor", controller, current)
    if _objects_at(root, controller) != objects:
        raise ValueError("GPU recorded controller source drift")
    template_raw = _read(root, PLAN_V2_PATH)
    if _sha(template_raw) != TEMPLATE_RAW_SHA256:
        raise ValueError("historical plan template mismatch")
    template = HoldoutApplicationPlanV2.model_validate_json(template_raw)
    inputs = workload.load_inputs(root)
    dataset = FrozenDatasetManifest.model_validate_json(
        _read(root, Path(inputs.matrix.dataset.manifest_path))
    )
    if (
        template.matrix_sha256 != inputs.matrix.content_sha256
        or _sha(_read(root, Path("uv.lock"))) != template.dependency_lock_sha256
        or dataset.content_sha256 != template.dataset_manifest_sha256
    ):
        raise ValueError("GPU experimental inputs drift")
    cpu = report["cpu_probe"]["runtime"]
    cache = {
        key: report["coverage"][key]
        for key in (
            "status",
            "total_vectors",
            "completed_vectors",
            "job_sha256",
            "inputs_sha256",
            "complete_sha256",
        )
    }
    freeze = _sealed(
        GpuFreeze,
        {
            "controller_commit": controller,
            "frozen_at": frozen_at,
            "execution_source_objects": objects,
            "execution_source_sha256": _canonical_sha256(objects),
            "qualification_raw_sha256": QUALIFICATION_RAW_SHA256,
            "qualification_controller_commit": report["controller_commit"],
            "qualification_execution_source_sha256": report["execution_source_sha256"],
            "qualification_checked_at": TypeAdapter(AwareDatetime).validate_python(
                report["checked_at"]
            ),
            "runtime_profile_sha256": _profile_digest(cpu, cache),
            "cpu_runtime": cpu,
            "cache_evidence": cache,
            **{
                key: getattr(template, key)
                for key in (
                    "matrix_sha256",
                    "agent_inputs_raw_sha256",
                    "environment_raw_sha256",
                    "test_inputs_raw_sha256",
                    "canary_raw_sha256",
                    "dependency_lock_sha256",
                    "dataset_manifest_sha256",
                )
            },
        },
    )
    plan_id = f"p7-holdout-v3-{freeze.content_sha256[:16]}"
    payload = template.model_dump(exclude={"content_sha256"})
    payload.update(
        schema_version=3,
        plan_version="holdout-application-plan-v3",
        plan_id=plan_id,
        experiment_id="repopilot-p7-v3",
        freeze_marker_sha256=freeze.content_sha256,
        frozen_source_commit=controller,
        controller_commit=controller,
        created_at=frozen_at,
        runtime_profile_sha256=freeze.runtime_profile_sha256,
        carry_forward_sha256=_canonical_sha256(freeze.carried_attempt),
        qualification_raw_sha256=QUALIFICATION_RAW_SHA256,
        run_slots=tuple(
            slot.model_copy(
                update={
                    "run_id": _run_id(plan_id, slot.arm_id, slot.task_id),
                    "output_path": _output_path(plan_id, slot.arm_id, slot.task_id),
                }
            )
            for slot in template.run_slots
        ),
    )
    return freeze, _sealed(GpuPlan, payload)


def load_plan(root: Path) -> tuple[GpuFreeze, GpuPlan]:
    freeze = GpuFreeze.model_validate_json(_read(root, FREEZE_PATH))
    plan = GpuPlan.model_validate_json(_read(root, PLAN_PATH))
    verify_frozen_source(root, freeze)
    if (freeze, plan) != build_plan(
        root, frozen_at=freeze.frozen_at, recorded_controller=freeze.controller_commit
    ):
        raise ValueError("GPU freeze/plan binding mismatch")
    return freeze, plan


def build_authorization(
    root: Path,
    *,
    approved_at: datetime,
    max_total_cost_usd: Decimal,
    overshoot_acknowledged: bool,
    readiness_required_before_paid: bool,
    confirm_plan_sha256: str,
) -> GpuAuthorization:
    """Call only after direct human approval of this exact plan; never called by CLI."""
    freeze, plan = load_plan(root)
    if confirm_plan_sha256 != plan.content_sha256 or approved_at < plan.created_at:
        raise ValueError("GPU exact-plan approval missing or stale")
    return _sealed(
        GpuAuthorization,
        {
            "authorization_id": f"{plan.plan_id}-usd32-ethan",
            "approved_at": approved_at,
            "experiment_id": plan.experiment_id,
            "plan_id": plan.plan_id,
            "plan_sha256": plan.content_sha256,
            "plan_raw_sha256": _sha(_read(root, PLAN_PATH)),
            "freeze_marker_sha256": freeze.content_sha256,
            "freeze_raw_sha256": _sha(_read(root, FREEZE_PATH)),
            "matrix_sha256": plan.matrix_sha256,
            "runtime_profile_sha256": plan.runtime_profile_sha256,
            "carry_forward_sha256": plan.carry_forward_sha256,
            "carried_unknown_cost_usd": plan.carried_unknown_cost_usd,
            "max_total_cost_usd": max_total_cost_usd,
            "new_execution_budget_usd": plan.new_execution_budget_usd,
            "paid_execution_approved": True,
            "final_in_flight_response_overshoot_acknowledged": overshoot_acknowledged,
            "readiness_required_before_paid": readiness_required_before_paid,
        },
    )


def load_controls(root: Path) -> GpuControls:
    freeze, plan = load_plan(root)
    freeze_raw, plan_raw = _read(root, FREEZE_PATH), _read(root, PLAN_PATH)
    auth_raw = _read(root, AUTHORIZATION_PATH)
    environment_raw, agent_raw = _read(root, ENVIRONMENT_PATH), _read(root, AGENT_INPUTS_PATH)
    auth = GpuAuthorization.model_validate_json(auth_raw)
    bindings = {
        "experiment_id": plan.experiment_id,
        "plan_id": plan.plan_id,
        "plan_sha256": plan.content_sha256,
        "plan_raw_sha256": _sha(plan_raw),
        "freeze_marker_sha256": freeze.content_sha256,
        "freeze_raw_sha256": _sha(freeze_raw),
        "matrix_sha256": plan.matrix_sha256,
        "runtime_profile_sha256": plan.runtime_profile_sha256,
        "carry_forward_sha256": plan.carry_forward_sha256,
        "carried_unknown_cost_usd": plan.carried_unknown_cost_usd,
        "max_total_cost_usd": plan.application_budget_usd,
        "new_execution_budget_usd": plan.new_execution_budget_usd,
    }
    if (
        any(getattr(auth, key) != value for key, value in bindings.items())
        or auth.approved_at < plan.created_at
        or _sha(environment_raw) != plan.environment_raw_sha256
        or _sha(agent_raw) != plan.agent_inputs_raw_sha256
    ):
        raise ValueError("GPU authorization or source binding mismatch")
    return GpuControls(
        freeze,
        plan,
        auth,
        HoldoutEnvironmentManifest.model_validate_json(environment_raw),
        HoldoutAgentInputs.model_validate_json(agent_raw),
        _git(root, "rev-parse", "HEAD"),
        freeze_raw,
        plan_raw,
        auth_raw,
        environment_raw,
        agent_raw,
    )


def _readiness_bindings(controls: GpuControls) -> dict[str, object]:
    return {
        "experiment_id": controls.plan.experiment_id,
        "plan_raw_sha256": _sha(controls.plan_raw),
        "plan_content_sha256": controls.plan.content_sha256,
        "authorization_raw_sha256": _sha(controls.authorization_raw),
        "authorization_content_sha256": controls.authorization.content_sha256,
        "environment_raw_sha256": _sha(controls.environment_raw),
        "environment_content_sha256": controls.environment.content_sha256,
        "freeze_raw_sha256": _sha(controls.freeze_raw),
        "freeze_content_sha256": controls.freeze.content_sha256,
        "agent_inputs_raw_sha256": _sha(controls.agent_inputs_raw),
        "test_inputs_raw_sha256": controls.freeze.test_inputs_raw_sha256,
        "canary_raw_sha256": controls.freeze.canary_raw_sha256,
        "qualified_runtime_commit": controls.freeze.controller_commit,
        "semantic_objects_sha256": controls.freeze.execution_source_sha256,
        "qualification_raw_sha256": controls.freeze.qualification_raw_sha256,
        "runtime_profile_sha256": controls.freeze.runtime_profile_sha256,
        "cpu_profile": controls.freeze.cpu_runtime,
        "gpu_cache": controls.freeze.cache_evidence,
    }


def build_readiness(
    controls: GpuControls,
    *,
    checked_at: datetime,
    execution_controller_objects: dict[str, str],
    embedding: legacy.EmbeddingReadinessEvidence,
    vector_backend: legacy.VectorReadinessEvidence,
    images: tuple[legacy.ReadinessImageEvidence, ...],
    cpu_profile: dict[str, JsonValue],
    gpu_cache: dict[str, JsonValue],
) -> GpuReadiness:
    if cpu_profile != controls.freeze.cpu_runtime or gpu_cache != controls.freeze.cache_evidence:
        raise ValueError("GPU readiness observed profile mismatch")
    return _sealed(
        GpuReadiness,
        {
            **_readiness_bindings(controls),
            "controller_commit": controls.controller_commit,
            "checked_at": checked_at,
            "execution_controller_objects": execution_controller_objects,
            "provider": legacy.ProviderCredentialEvidence(),
            "embedding": embedding,
            "vector_backend": vector_backend,
            "images": images,
            "completed_checks": READINESS_CHECKS,
        },
    )


def validate_readiness_bindings(root: Path, controls: GpuControls, readiness: GpuReadiness) -> None:
    if any(
        getattr(readiness, key) != value for key, value in _readiness_bindings(controls).items()
    ):
        raise ValueError("GPU readiness binding mismatch")
    verify_frozen_source(root, controls.freeze)
    _git(
        root,
        "merge-base",
        "--is-ancestor",
        controls.plan.controller_commit,
        readiness.controller_commit,
    )
    _git(
        root, "merge-base", "--is-ancestor", readiness.controller_commit, controls.controller_commit
    )
    if (
        readiness.execution_controller_objects != verify_execution_controller(root)
        or _objects_at(root, readiness.controller_commit) != readiness.execution_controller_objects
    ):
        raise ValueError("GPU readiness execution controller drift")
    if readiness.checked_at < controls.authorization.approved_at:
        raise ValueError("GPU readiness predates authorization")
    if tuple(
        (image.task_id, image.registry_digest_reference) for image in readiness.images
    ) != tuple(
        (item.task_id, item.sandbox_image_reference) for item in controls.environment.bindings
    ):
        raise ValueError("GPU readiness image binding mismatch")
    inputs = workload.load_inputs(root)
    settings = select_experiment_arm(inputs.matrix, "dense-only-v1").settings(
        openai_api_key="readiness-identity-placeholder",
        database_url=DEFAULT_DATABASE_URL,
    )
    identity = embedding_identity_from_settings(settings)
    if (
        readiness.embedding.identity_sha256 != identity.fingerprint
        or readiness.embedding.model != identity.model
        or readiness.embedding.revision != identity.revision
        or readiness.embedding.code_revision != identity.code_revision
        or readiness.vector_backend.schema_name != settings.vector_schema
        or readiness.vector_backend.table_name != settings.vector_table
    ):
        raise ValueError("GPU readiness embedding/vector identity mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    for relative in (FREEZE_PATH, PLAN_PATH):
        path = root / relative
        if (
            path.exists()
            or path.is_symlink()
            or any(parent.is_symlink() for parent in path.parents)
        ):
            raise ValueError("GPU control output conflict or unsafe path")
    freeze, plan = build_plan(root, frozen_at=datetime.now(UTC))
    write_control_artifact(root / FREEZE_PATH, freeze)
    write_control_artifact(root / PLAN_PATH, plan)
    print(
        "holdout_gpu_plan_created=true tasks=8 slots=32 total_budget_usd=32.00 "
        "carried_unknown_cost_usd=1.00 new_execution_budget_usd=31.00 "
        f"plan_sha256={plan.content_sha256} authorization_created=false paid_requests=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
