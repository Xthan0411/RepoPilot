"""P7-S6 fail-closed holdout planning and environment-boundary tests."""

from __future__ import annotations

import json
import stat
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from repopilot.enums import ExperimentPreflightCheckStatus
from repopilot.evals import ExperimentMatrix, FormalFreezeMarker, load_experiment_matrix
from repopilot.evals.dataset import DatasetPreflightReport, FrozenDatasetManifest

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p7_holdout_control import (  # noqa: E402
    EXPECTED_ARM_IDS,
    EXPECTED_TASK_IDS,
    FREEZE_PATH,
    MATRIX_PATH,
    THEORETICAL_APPLICATION_COST_USD,
    HoldoutApplicationPlan,
    HoldoutControlError,
    HoldoutEnvironmentBinding,
    HoldoutErrorCode,
    build_holdout_application_plan,
    build_holdout_preflight,
    build_paid_holdout_authorization,
    write_control_artifact,
)

NOW = datetime(2026, 8, 8, 12, tzinfo=UTC)
CONTROLLER_COMMIT = "b" * 40


def _inputs() -> tuple[
    ExperimentMatrix,
    FormalFreezeMarker,
    FrozenDatasetManifest,
    DatasetPreflightReport,
]:
    matrix = load_experiment_matrix(PROJECT_ROOT / MATRIX_PATH, repository_root=PROJECT_ROOT)
    marker = FormalFreezeMarker.model_validate_json(PROJECT_ROOT.joinpath(FREEZE_PATH).read_bytes())
    dataset = FrozenDatasetManifest.model_validate_json(
        PROJECT_ROOT.joinpath(matrix.dataset.manifest_path).read_bytes()
    )
    preflight = DatasetPreflightReport.model_validate_json(
        PROJECT_ROOT.joinpath("results/p7-dataset-v1/preflight-report.json").read_bytes()
    )
    return matrix, marker, dataset, preflight


def _plan() -> HoldoutApplicationPlan:
    matrix, marker, dataset, _ = _inputs()
    return build_holdout_application_plan(
        matrix=matrix,
        marker=marker,
        dataset_manifest=dataset,
        controller_commit=CONTROLLER_COMMIT,
        created_at=NOW,
    )


def test_holdout_plan_is_exact_four_by_eight_gold_free_and_stable() -> None:
    plan = _plan()

    assert plan.arm_ids == EXPECTED_ARM_IDS
    assert plan.task_ids == EXPECTED_TASK_IDS
    assert len(plan.run_slots) == 32
    assert len({slot.run_id for slot in plan.run_slots}) == 32
    assert plan.theoretical_application_cost_usd == THEORETICAL_APPLICATION_COST_USD
    assert plan.repetitions == 1
    assert "hybrid-memory-v1" not in plan.arm_ids
    assert {slot.python_version for slot in plan.run_slots} == {"3.9", "3.10", "3.12.13"}
    serialized = plan.model_dump_json()
    assert "gold_files" not in serialized
    assert "gold_patch" not in serialized
    assert "issue_body" not in serialized
    assert tuple(slot.run_id for slot in plan.run_slots) == tuple(
        slot.run_id for slot in _plan().run_slots
    )


def test_holdout_plan_rejects_freeze_drift() -> None:
    matrix, marker, dataset, _ = _inputs()
    drifted = marker.model_copy(update={"model": "different-model"})

    with pytest.raises(HoldoutControlError) as captured:
        build_holdout_application_plan(
            matrix=matrix,
            marker=drifted,
            dataset_manifest=dataset,
            controller_commit=CONTROLLER_COMMIT,
            created_at=NOW,
        )

    assert captured.value.code is HoldoutErrorCode.FREEZE_IDENTITY_MISMATCH


@pytest.mark.parametrize("approved", (Decimal("31.99"), Decimal("32.01"), Decimal("8.00")))
def test_paid_authorization_rejects_any_non_exact_budget(approved: Decimal) -> None:
    with pytest.raises(HoldoutControlError) as captured:
        build_paid_holdout_authorization(
            plan=_plan(),
            authorization_id="p7-holdout-user-approved-v1",
            approved_by="ethan",
            approved_at=NOW,
            max_total_cost_usd=approved,
            overshoot_acknowledged=True,
        )

    assert captured.value.code is HoldoutErrorCode.BUDGET_AUTHORIZATION_MISMATCH


def test_paid_authorization_requires_final_response_overshoot_ack() -> None:
    with pytest.raises(HoldoutControlError) as captured:
        build_paid_holdout_authorization(
            plan=_plan(),
            authorization_id="p7-holdout-user-approved-v1",
            approved_by="ethan",
            approved_at=NOW,
            max_total_cost_usd=THEORETICAL_APPLICATION_COST_USD,
            overshoot_acknowledged=False,
        )

    assert captured.value.code is HoldoutErrorCode.FINAL_RESPONSE_OVERSHOOT_UNACKNOWLEDGED


def test_missing_environment_and_authorization_block_without_paid_requests(
    tmp_path: Path,
) -> None:
    _, marker, _, dataset_preflight = _inputs()
    preflight = build_holdout_preflight(
        root=tmp_path,
        plan=_plan(),
        marker=marker,
        dataset_preflight=dataset_preflight,
        checked_at=NOW,
        controller_source_clean=True,
        frozen_runtime_matches=True,
        authorization=None,
        environment_manifest=None,
    )
    checks = {item.check_id: item for item in preflight.checks}

    assert not preflight.ready_for_paid_holdout
    assert preflight.paid_requests_made == 0
    assert not preflight.gold_visible_to_agent
    assert preflight.runtime_verified_task_count == 1
    assert checks["authorization"].status is ExperimentPreflightCheckStatus.NOT_RUN
    assert checks["authorization"].error_code is HoldoutErrorCode.BUDGET_AUTHORIZATION_REQUIRED
    assert checks["task-environments"].status is ExperimentPreflightCheckStatus.BLOCKED
    assert checks["task-checkouts"].status is ExperimentPreflightCheckStatus.BLOCKED
    assert checks["task-images"].status is ExperimentPreflightCheckStatus.BLOCKED
    assert checks["provider-credential"].status is ExperimentPreflightCheckStatus.NOT_RUN
    assert sum(item.runtime_gold_verified for item in preflight.task_observations) == 1


def test_environment_binding_rejects_mutable_image_and_wrong_checkout() -> None:
    values = {
        "task_id": EXPECTED_TASK_IDS[0],
        "base_commit": "a" * 40,
        "python_version": "3.9",
        "checkout_relative_path": (f"artifacts/p7-holdout/checkouts/{EXPECTED_TASK_IDS[0]}"),
        "checkout_commit": "a" * 40,
        "sandbox_image_reference": "python:3.9",
        "platform": "linux/amd64",
        "setup_commands_passed_offline": True,
        "baseline_failed": True,
        "gold_passed": True,
    }
    with pytest.raises(ValidationError):
        HoldoutEnvironmentBinding.model_validate(values)

    values["sandbox_image_reference"] = f"python@sha256:{'c' * 64}"
    values["checkout_commit"] = "d" * 40
    with pytest.raises(ValidationError):
        HoldoutEnvironmentBinding.model_validate(values)


def test_control_artifacts_are_private_and_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    plan = _plan()
    write_control_artifact(path, plan)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text(encoding="utf-8"))["content_sha256"] == (plan.content_sha256)
    with pytest.raises(HoldoutControlError) as captured:
        write_control_artifact(path, plan)
    assert captured.value.code is HoldoutErrorCode.OUTPUT_CONFLICT
