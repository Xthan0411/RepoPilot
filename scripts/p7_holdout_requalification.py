"""Accept the replacement-VM protocol evidence and freeze a separate unpaid v2 plan.

No source rows, Provider, Docker, credentials, or authorization are accessed here.
Historical dev calibration remains historical; protocol qualification is not a new
Agent calibration or a measured Holdout result. The legacy v1 guards remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self
from uuid import UUID, uuid5

from pydantic import AwareDatetime, TypeAdapter, model_validator

from repopilot.enums import DatasetSplit, TestRunCategory, TestStatus
from repopilot.evals import FormalFreezeMarker, load_experiment_matrix
from repopilot.evals.dataset import FrozenDatasetManifest
from repopilot.schemas import GitSha, Identifier, Sha256, StrictModel, TaskId
from scripts.p7_holdout_agent_inputs import (
    AGENT_INPUTS_PATH,
    HoldoutAgentInputs,
    _canonical_sha256,
    _validate_content_digest,
)
from scripts.p7_holdout_control import (
    _FROZEN_RUNTIME_PATHS,
    ENVIRONMENT_PATH,
    EXPECTED_ARM_IDS,
    EXPECTED_TASK_IDS,
    FREEZE_PATH,
    MATRIX_PATH,
    HoldoutEnvironmentManifest,
    HoldoutRunSlot,
    build_holdout_application_plan,
    write_control_artifact,
)
from scripts.p7_holdout_test_inputs import (
    TEST_INPUTS_RAW_SHA256,
    load_test_inputs,
)
from scripts.p7_test_protocol_canary import (
    REQUIRED_CONTROLLER_PATHS,
    RESULT_PATH,
    ProtocolCanaryReport,
    ProtocolTestEvidence,
)

QUALIFIED_COMMIT = "19cb83dd0370c4f450319040fe604f39cd002b3c"
CANARY_RAW_SHA256 = "732ae6863bc3d28f9a5c584e775672e97c87f9844026ac83df0f299e907fae46"
POST_REPAIR_FREEZE_PATH = Path("results/p7-holdout/post-repair-freeze-v2.json")
PLAN_V2_PATH = Path("results/p7-holdout/plan-v2.json")
CONTROLLER_PATH = "scripts/p7_holdout_requalification.py"
SEMANTIC_PATHS = tuple(
    sorted(
        set(_FROZEN_RUNTIME_PATHS)
        | set(REQUIRED_CONTROLLER_PATHS)
        | {
            "scripts/__init__.py",
            AGENT_INPUTS_PATH.as_posix(),
            ENVIRONMENT_PATH.as_posix(),
            FREEZE_PATH.as_posix(),
            "data/manifests/p7-v1/manifest.json",
            "results/p7-dev/calibration-report-v5.json",
        }
    )
)
_RUN_NAMESPACE_V2 = UUID("d5ed7a1c-a45d-5f71-8c34-9e1252cd7d31")


class PostRepairFreeze(StrictModel):
    """Qualified runtime identity, explicitly separate from historical dev calibration."""

    freeze_version: Literal["post-repair-freeze-v2"] = "post-repair-freeze-v2"
    qualified_runtime_commit: Literal["19cb83dd0370c4f450319040fe604f39cd002b3c"]
    controller_commit: GitSha
    frozen_at: AwareDatetime
    semantic_objects: dict[str, GitSha]
    semantic_objects_sha256: Sha256
    canary_raw_sha256: Literal["732ae6863bc3d28f9a5c584e775672e97c87f9844026ac83df0f299e907fae46"]
    canary_content_sha256: Sha256
    canary_checked_at: AwareDatetime
    historical_freeze_raw_sha256: Sha256
    historical_freeze_content_sha256: Sha256
    historical_calibration_source_commit: GitSha
    historical_calibration_report_raw_sha256: Sha256
    current_source_calibration_performed: Literal[False] = False
    evaluator_only_holdout_qualification_performed: Literal[True] = True
    environment_raw_sha256: Sha256
    agent_inputs_raw_sha256: Sha256
    test_inputs_raw_sha256: Sha256
    matrix_sha256: Sha256
    dependency_lock_sha256: Sha256
    dataset_manifest_sha256: Sha256
    test_protocol_version: Literal["isolated-regression-tests-v1"]
    task_ids: tuple[TaskId, ...]
    provider_requests_made: Literal[0] = 0
    paid_requests_made: Literal[0] = 0
    formal_environment_ready: Literal[False] = False
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_freeze(self) -> Self:
        if (
            tuple(self.semantic_objects) != SEMANTIC_PATHS
            or _canonical_sha256(self.semantic_objects) != self.semantic_objects_sha256
            or self.task_ids != EXPECTED_TASK_IDS
            or self.canary_checked_at > self.frozen_at
        ):
            raise ValueError("post-repair freeze identity mismatch")
        _validate_content_digest(self)
        return self


class HoldoutApplicationPlanV2(StrictModel):
    """New 32-slot namespace; neither v1 consent nor this plan permits paid execution."""

    schema_version: Literal[2] = 2
    plan_version: Literal["holdout-application-plan-v2"] = "holdout-application-plan-v2"
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
    test_protocol_version: Literal["isolated-regression-tests-v1"]
    canary_raw_sha256: Sha256
    environment_raw_sha256: Sha256
    agent_inputs_raw_sha256: Sha256
    test_inputs_raw_sha256: Sha256
    ready_for_paid_holdout: Literal[False] = False
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.arm_ids != EXPECTED_ARM_IDS or self.task_ids != EXPECTED_TASK_IDS:
            raise ValueError("v2 plan task/arm set mismatch")
        if self.per_task_cost_limit_usd != Decimal("1.00") or (
            self.theoretical_application_cost_usd != Decimal("32.00")
        ):
            raise ValueError("v2 plan budget mismatch")
        expected = tuple((arm, task) for arm in self.arm_ids for task in self.task_ids)
        if tuple((s.arm_id, s.task_id) for s in self.run_slots) != expected:
            raise ValueError("v2 plan requires exactly 32 ordered slots")
        for slot in self.run_slots:
            if slot.run_id != _run_id(self.plan_id, slot.arm_id, slot.task_id) or (
                slot.output_path != _output_path(self.plan_id, slot.arm_id, slot.task_id)
            ):
                raise ValueError("v2 slot namespace mismatch")
        _validate_content_digest(self)
        return self


def _run_id(plan_id: str, arm: str, task: str) -> UUID:
    return uuid5(_RUN_NAMESPACE_V2, f"{plan_id}:{arm}:{task}:1")


def _output_path(plan_id: str, arm: str, task: str) -> str:
    return f"results/p7-holdout/raw/{plan_id}/{arm}/{task}"


def _read(root: Path, relative: Path, limit: int = 4_000_000) -> bytes:
    path = root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or any(parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent)
    ):
        raise ValueError("unsafe requalification path")
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("unsafe requalification path")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("requalification input too large")
    return raw


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def verify_qualified_runtime(root: Path) -> tuple[str, dict[str, str]]:
    """Reject runtime drift; later controller commits cannot change qualified semantics."""

    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("repository root mismatch")
    controller = _git(root, "rev-parse", "HEAD")
    _git(root, "merge-base", "--is-ancestor", QUALIFIED_COMMIT, controller)
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no") or _git(
        root, "status", "--porcelain=v1", "--untracked-files=all", "--", *SEMANTIC_PATHS
    ):
        raise ValueError("source snapshot unclean")
    if _git(root, "diff", "--name-only", QUALIFIED_COMMIT, controller, "--", *SEMANTIC_PATHS):
        raise ValueError("qualified runtime drift")
    # Compare disk bytes, not only Git's index/stat cache (assume-unchanged can hide edits).
    for entry in _git(root, "ls-tree", "-rz", QUALIFIED_COMMIT, "--", *SEMANTIC_PATHS).split("\0"):
        if not entry:
            continue
        metadata, relative = entry.split("\t", 1)
        mode, kind, blob = metadata.split()
        if kind != "blob" or mode not in ("100644", "100755"):
            raise ValueError("unsupported semantic source entry")
        raw = _read(root, Path(relative))
        actual = hashlib.sha1(
            f"blob {len(raw)}\0".encode() + raw, usedforsecurity=False
        ).hexdigest()
        if actual != blob:
            raise ValueError("qualified runtime bytes changed")
    committed = subprocess.run(
        ("git", "show", f"{controller}:{CONTROLLER_PATH}"),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if _read(root, Path(CONTROLLER_PATH)) != committed:
        raise ValueError("executed controller is not committed")
    objects = {
        path: _git(root, "rev-parse", f"{QUALIFIED_COMMIT}:{path}") for path in SEMANTIC_PATHS
    }
    return controller, objects


def _validate_completed_evidence(item: ProtocolTestEvidence) -> None:
    if (
        not item.verifier_accepts_evidence
        or item.truncated
        or item.tests_run <= 0
        or item.failures + item.errors > item.tests_run
    ):
        raise ValueError("invalid completed test evidence")
    if item.status is TestStatus.PASSED:
        valid = (
            item.category is TestRunCategory.PASSED
            and item.exit_code == 0
            and (item.failures == item.errors == item.failing_test_count == 0)
        )
    else:
        valid = (
            item.status is TestStatus.FAILED
            and item.category in (TestRunCategory.ASSERTION_FAILURE, TestRunCategory.TEST_ERROR)
            and item.exit_code not in (None, 0)
            and item.failures + item.errors > 0
            and 0 < item.failing_test_count <= item.failures + item.errors
        )
    if not valid:
        raise ValueError("inconsistent completed test evidence")


def build_requalification(
    root: Path, *, frozen_at: datetime, recorded_controller: str | None = None
) -> tuple[PostRepairFreeze, HoldoutApplicationPlanV2]:
    """Cross-check original evidence and metadata only, never evaluator Gold/source rows."""

    root = root.resolve(strict=True)
    controller, objects = verify_qualified_runtime(root)
    if recorded_controller is not None:
        _git(root, "merge-base", "--is-ancestor", QUALIFIED_COMMIT, recorded_controller)
        _git(root, "merge-base", "--is-ancestor", recorded_controller, controller)
        if _git(
            root,
            "diff",
            "--name-only",
            QUALIFIED_COMMIT,
            recorded_controller,
            "--",
            *SEMANTIC_PATHS,
        ):
            raise ValueError("recorded controller runtime drift")
        if _git(root, "rev-parse", f"{recorded_controller}:{CONTROLLER_PATH}") != _git(
            root, "rev-parse", f"{controller}:{CONTROLLER_PATH}"
        ):
            raise ValueError("requalification builder changed")
        controller = recorded_controller
    raw = _read(root, RESULT_PATH, 100_000)
    if _sha(raw) != CANARY_RAW_SHA256:
        raise ValueError("unrecognized canary raw digest")
    canary = ProtocolCanaryReport.model_validate_json(raw)
    checked_at = TypeAdapter(AwareDatetime).validate_python(canary.checked_at)
    environment_raw = _read(root, ENVIRONMENT_PATH)
    environment = HoldoutEnvironmentManifest.model_validate_json(environment_raw)
    agent_raw = _read(root, AGENT_INPUTS_PATH)
    agent = HoldoutAgentInputs.model_validate_json(agent_raw)
    tests = load_test_inputs(root, agent_inputs=agent, expected_raw_sha256=TEST_INPUTS_RAW_SHA256)
    matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
    dataset_raw = _read(root, Path(matrix.dataset.manifest_path))
    dataset = FrozenDatasetManifest.model_validate_json(dataset_raw)
    cases = {case.case_id: case for case in dataset.external_cases}
    historical_raw = _read(root, FREEZE_PATH)
    historical = FormalFreezeMarker.model_validate_json(historical_raw)
    calibration_raw = _read(root, Path(historical.calibration_report_ref.relative_path))
    if (
        canary.controller_commit != QUALIFIED_COMMIT
        or canary.environment_raw_sha256 != _sha(environment_raw)
        or canary.test_inputs_raw_sha256 != TEST_INPUTS_RAW_SHA256
        or agent.matrix_sha256 != environment.matrix_sha256
        or agent.matrix_sha256 != matrix.content_sha256
        or agent.dataset_manifest_raw_sha256 != _sha(dataset_raw)
        or agent.dataset_manifest_content_sha256 != dataset.content_sha256
        or environment.dataset_manifest_sha256 != dataset.content_sha256
        or agent.source_rows_raw_sha256 != dataset.source_rows_ref.sha256
        or agent.runtime_command_contract_sha256 != environment.runtime_command_contract_sha256
        or _sha(calibration_raw) != historical.calibration_report_ref.sha256
        or len(calibration_raw) != historical.calibration_report_ref.size_bytes
        or _sha(_read(root, Path("uv.lock"))) != historical.dependency_lock_sha256
    ):
        raise ValueError("requalification artifact binding mismatch")
    for evidence, task, regression, binding in zip(
        canary.tasks, agent.tasks, tests.tasks, environment.bindings, strict=True
    ):
        if (
            evidence.base_commit != task.base_commit
            or binding.base_commit != task.base_commit
            or binding.python_version != cases[task.task_id].python_version
            or task.base_commit != cases[task.task_id].base_commit
            or evidence.regression_patch_sha256 != regression.regression_patch_sha256
            or len(evidence.baseline) != len(task.test_commands)
        ):
            raise ValueError("requalification task binding mismatch")
        for item in (*evidence.baseline, *evidence.gold):
            _validate_completed_evidence(item)
    # Use the historical builder only to validate the unchanged experimental choices.
    # Its old-source plan is never saved or treated as authorization for repaired code.
    template = build_holdout_application_plan(
        matrix=matrix,
        marker=historical,
        dataset_manifest=dataset,
        controller_commit=controller,
        created_at=frozen_at,
    )
    if any(
        slot.base_commit != cases[slot.task_id].base_commit
        or slot.python_version != cases[slot.task_id].python_version
        for slot in template.run_slots
    ):
        raise ValueError("requalification slot binding mismatch")
    payload: dict[str, object] = {
        "qualified_runtime_commit": QUALIFIED_COMMIT,
        "controller_commit": controller,
        "frozen_at": frozen_at,
        "semantic_objects": objects,
        "semantic_objects_sha256": _canonical_sha256(objects),
        "canary_raw_sha256": CANARY_RAW_SHA256,
        "canary_content_sha256": canary.content_sha256,
        "canary_checked_at": checked_at,
        "historical_freeze_raw_sha256": _sha(historical_raw),
        "historical_freeze_content_sha256": historical.content_sha256,
        "historical_calibration_source_commit": historical.source_commit,
        "historical_calibration_report_raw_sha256": _sha(calibration_raw),
        "environment_raw_sha256": _sha(environment_raw),
        "agent_inputs_raw_sha256": _sha(agent_raw),
        "test_inputs_raw_sha256": TEST_INPUTS_RAW_SHA256,
        "matrix_sha256": matrix.content_sha256,
        "dependency_lock_sha256": historical.dependency_lock_sha256,
        "dataset_manifest_sha256": dataset.content_sha256,
        "test_protocol_version": canary.protocol_version,
        "task_ids": EXPECTED_TASK_IDS,
    }
    # Materialize defaults before hashing; content digests cover the whole strict model.
    freeze = _sealed(PostRepairFreeze, payload)
    plan_id = f"p7-holdout-v2-{freeze.content_sha256[:16]}"
    plan_payload = template.model_dump(exclude={"content_sha256"})
    plan_payload.update(
        schema_version=2,
        plan_version="holdout-application-plan-v2",
        plan_id=plan_id,
        freeze_marker_sha256=freeze.content_sha256,
        frozen_source_commit=QUALIFIED_COMMIT,
        test_protocol_version=canary.protocol_version,
        canary_raw_sha256=CANARY_RAW_SHA256,
        environment_raw_sha256=_sha(environment_raw),
        agent_inputs_raw_sha256=_sha(agent_raw),
        test_inputs_raw_sha256=TEST_INPUTS_RAW_SHA256,
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
    return freeze, _sealed(HoldoutApplicationPlanV2, plan_payload)


def load_requalification(root: Path) -> tuple[PostRepairFreeze, HoldoutApplicationPlanV2]:
    """Rebuild every binding from pinned inputs; a self-rehashed plan is insufficient."""

    root = root.resolve(strict=True)
    freeze = PostRepairFreeze.model_validate_json(_read(root, POST_REPAIR_FREEZE_PATH))
    plan = HoldoutApplicationPlanV2.model_validate_json(_read(root, PLAN_V2_PATH))
    expected = build_requalification(
        root, frozen_at=freeze.frozen_at, recorded_controller=freeze.controller_commit
    )
    if (freeze, plan) != expected:
        raise ValueError("requalification freeze/plan binding mismatch")
    return freeze, plan


def _sealed[T: StrictModel](model: type[T], payload: dict[str, object]) -> T:
    draft = model.model_construct(_fields_set=None, **payload, content_sha256="0" * 64)
    values = draft.model_dump(exclude={"content_sha256"})
    return model.model_validate({**values, "content_sha256": _canonical_sha256(values)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    for relative in (POST_REPAIR_FREEZE_PATH, PLAN_V2_PATH):
        path = root / relative
        if (
            path.exists()
            or path.is_symlink()
            or any(parent.is_symlink() for parent in path.parents if parent != root.parent)
        ):
            raise ValueError("requalification output conflict or unsafe path")
    freeze, plan = build_requalification(root, frozen_at=datetime.now(UTC))
    write_control_artifact(root / POST_REPAIR_FREEZE_PATH, freeze)
    write_control_artifact(root / PLAN_V2_PATH, plan)
    print(
        "holdout_requalification_created=true tasks=8 slots=32 "
        "current_source_calibration_performed=false authorization_created=false "
        f"plan_sha256={plan.content_sha256} provider_requests=0 paid_requests=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
