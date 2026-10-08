"""P7-S5 dev-only calibration, authorization, and freeze tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from repopilot.enums import (
    CalibrationErrorCode,
    DatasetSplit,
    ReproductionStatus,
    RunStatus,
    TerminationReason,
    ValidationStatus,
)
from repopilot.evals import (
    CalibrationChangeLog,
    CalibrationError,
    DevCalibrationPreflight,
    DevCalibrationPreflightFacts,
    EvaluationExecutionEvidence,
    EvaluationExecutionRequest,
    ExperimentArmSelection,
    ExperimentMatrix,
    PaidCalibrationAuthorization,
    build_dev_calibration_plan,
    build_dev_calibration_preflight,
    build_formal_freeze_marker,
    build_paid_calibration_authorization,
    load_dev_calibration_tasks,
    load_experiment_matrix,
    run_dev_calibration,
    write_calibration_artifact,
)
from repopilot.schemas import (
    ArtifactRef,
    FinalResult,
    LocalizationMetrics,
    ReproductionResult,
    TaskMetrics,
    UsageMetrics,
    ValidationResult,
)

PROJECT_ROOT = Path(__file__).parents[2]
MATRIX_PATH = PROJECT_ROOT / "configs/experiments/p7-v2.yaml"
NOW = datetime(2026, 8, 7, 23, tzinfo=UTC)
COMMIT = "a" * 40


def _artifact_ref(path: Path, *, relative_path: str | None = None) -> ArtifactRef:
    content = path.read_bytes()
    return ArtifactRef(
        relative_path=relative_path or path.name,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        media_type="application/json",
    )


class CalibrationFakeExecutor:
    """No-network executor that emits Runner-verifiable raw dev artifacts."""

    def __init__(
        self,
        selection: ExperimentArmSelection,
        *,
        reported_cost_usd: Decimal = Decimal("0"),
    ) -> None:
        self.selection = selection
        self.reported_cost_usd = reported_cost_usd

    def execute(self, request: EvaluationExecutionRequest) -> EvaluationExecutionEvidence:
        usage = UsageMetrics(
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            estimated_cost_usd=self.reported_cost_usd,
            latency_ms=5,
        )
        result = FinalResult(
            task_id=request.task.task_id,
            run_id=request.run_id,
            status=RunStatus.STOPPED,
            termination_reason=TerminationReason.AGENT_STOPPED,
            reproduction_result=ReproductionResult(status=ReproductionStatus.NOT_ATTEMPTED),
            validation_result=ValidationResult(status=ValidationStatus.NOT_RUN),
            localization_metrics=LocalizationMetrics(
                file_recall_at_1=0,
                file_recall_at_3=0,
                file_recall_at_5=0,
                mean_reciprocal_rank=0,
                indexing_latency_ms=0,
                retrieval_latency_ms=0,
            ),
            task_metrics=TaskMetrics(
                issue_reproduced=False,
                patch_applied=False,
                tests_passed=False,
                task_success=False,
                tool_steps=0,
            ),
            usage=usage,
            trace_path="trace.json",
        )
        result_path = request.output_dir / "result.json"
        trace_path = request.output_dir / "trace.json"
        artifact_path = request.output_dir / "artifacts.json"
        result_path.write_text(f"{result.model_dump_json(indent=2)}\n", encoding="utf-8")
        trace_path.write_text(
            json.dumps(
                {"task_id": request.task.task_id, "run_id": str(request.run_id)},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        artifact_path.write_text(
            json.dumps(
                {
                    "task_id": request.task.task_id,
                    "run_id": str(request.run_id),
                    "provider_name": (
                        "openai-responses:"
                        f"{self.selection.matrix.shared_settings.values['model']}:"
                        f"{self.selection.matrix.prompt_versions[0]}"
                    ),
                    "sandbox_image_reference": (self.selection.matrix.sandbox_image_reference),
                    "evaluation_configuration_sha256": (self.selection.configuration_sha256),
                    "prompt_versions": list(self.selection.matrix.prompt_versions),
                    "result_path": "result.json",
                    "trace_path": "trace.json",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return EvaluationExecutionEvidence(
            task_id=request.task.task_id,
            run_id=request.run_id,
            status=result.status,
            termination_reason=result.termination_reason,
            task_success=False,
            usage=usage,
            result_ref=_artifact_ref(result_path),
            trace_ref=_artifact_ref(trace_path),
            artifact_manifest_ref=_artifact_ref(artifact_path),
        )


def _authorization(
    matrix: ExperimentMatrix,
    *,
    cost: Decimal = Decimal("8.00"),
) -> PaidCalibrationAuthorization:
    return build_paid_calibration_authorization(
        matrix=matrix,
        authorization_id="p7-dev-user-approved-v1",
        approved_by="ethan",
        approved_at=NOW,
        max_total_cost_usd=cost,
    )


def _facts(*, clean: bool) -> DevCalibrationPreflightFacts:
    return DevCalibrationPreflightFacts(
        source_commit=COMMIT,
        source_clean=clean,
        local_dependencies_available=True,
        sandbox_image_available=True,
        provider_credential_available=True,
        vector_backend_available=True,
        embedding_snapshot_available=True,
        output_root_ready=True,
    )


def test_dev_loader_plan_and_preflight_never_load_holdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matrix = load_experiment_matrix(MATRIX_PATH, repository_root=PROJECT_ROOT)
    monkeypatch.setattr(
        "repopilot.evals.dataset.load_frozen_dataset",
        lambda *_args, **_kwargs: pytest.fail("holdout loader must remain unreachable"),
    )
    tasks = load_dev_calibration_tasks(matrix, repository_root=PROJECT_ROOT)
    tmp_path.joinpath("uv.lock").write_bytes(PROJECT_ROOT.joinpath("uv.lock").read_bytes())
    preflight = build_dev_calibration_preflight(
        matrix=matrix,
        tasks=tasks,
        repository_root=tmp_path,
        checked_at=NOW,
        facts=_facts(clean=False),
    )
    plan = build_dev_calibration_plan(
        matrix=matrix,
        authorization=_authorization(matrix),
        tasks=tasks,
        created_at=NOW,
        source_commit=COMMIT,
        source_dirty=True,
    )

    assert tuple(item.task.task_id for item in tasks) == (
        "p5-config-parser-blank",
        "p5-parser-service-async",
    )
    assert preflight.ready_for_paid_dev
    assert preflight.source_dirty
    assert not preflight.holdout_accessed
    assert plan.theoretical_max_cost_usd == Decimal("8.00")
    assert plan.carried_reported_cost_usd == 0
    assert plan.calibration_id.endswith(COMMIT[:12])
    assert len(plan.run_slots) == 4
    assert not plan.holdout_accessed

    tracked_preflight = DevCalibrationPreflight.model_validate_json(
        PROJECT_ROOT.joinpath("results/p7-dev/preflight-v2.json").read_bytes()
    )
    change_log = CalibrationChangeLog.model_validate_json(
        PROJECT_ROOT.joinpath("results/p7-dev/calibration-change-log-v1.json").read_bytes()
    )
    assert not tracked_preflight.ready_for_paid_dev
    assert not tracked_preflight.holdout_accessed
    assert change_log.to_matrix_sha256 == matrix.content_sha256
    assert not change_log.holdout_accessed
    assert all(not item.holdout_informed for item in change_log.changes)


def test_plan_rejects_wrong_approval_and_holdout_task() -> None:
    matrix = load_experiment_matrix(MATRIX_PATH, repository_root=PROJECT_ROOT)
    tasks = load_dev_calibration_tasks(matrix, repository_root=PROJECT_ROOT)

    with pytest.raises(CalibrationError) as captured:
        build_dev_calibration_plan(
            matrix=matrix,
            authorization=None,
            tasks=tasks,
            created_at=NOW,
            source_commit=COMMIT,
            source_dirty=True,
        )
    assert captured.value.code is CalibrationErrorCode.BUDGET_APPROVAL_REQUIRED

    with pytest.raises(CalibrationError) as captured:
        build_dev_calibration_plan(
            matrix=matrix,
            authorization=_authorization(matrix, cost=Decimal("7.00")),
            tasks=tasks,
            created_at=NOW,
            source_commit=COMMIT,
            source_dirty=True,
        )
    assert captured.value.code is CalibrationErrorCode.BUDGET_APPROVAL_MISMATCH

    holdout = tasks[0].model_copy(update={"split": DatasetSplit.HOLDOUT})
    with pytest.raises(CalibrationError) as captured:
        build_dev_calibration_plan(
            matrix=matrix,
            authorization=_authorization(matrix),
            tasks=(holdout, tasks[1]),
            created_at=NOW,
            source_commit=COMMIT,
            source_dirty=True,
        )
    assert captured.value.code is CalibrationErrorCode.HOLDOUT_ACCESS_DENIED


def test_fake_dev_matrix_retains_all_raw_runs_and_builds_clean_freeze(
    tmp_path: Path,
) -> None:
    matrix = load_experiment_matrix(MATRIX_PATH, repository_root=PROJECT_ROOT)
    tasks = load_dev_calibration_tasks(matrix, repository_root=PROJECT_ROOT)
    authorization = _authorization(matrix)
    tmp_path.joinpath("uv.lock").write_bytes(PROJECT_ROOT.joinpath("uv.lock").read_bytes())
    preflight = build_dev_calibration_preflight(
        matrix=matrix,
        tasks=tasks,
        repository_root=tmp_path,
        checked_at=NOW,
        facts=_facts(clean=True),
    )
    plan = build_dev_calibration_plan(
        matrix=matrix,
        authorization=authorization,
        tasks=tasks,
        created_at=NOW,
        source_commit=COMMIT,
        source_dirty=False,
    )
    report = run_dev_calibration(
        plan=plan,
        matrix=matrix,
        authorization=authorization,
        preflight=preflight,
        tasks=tasks,
        repository_root=tmp_path,
        openai_api_key="test-placeholder-not-sent",
        database_url="postgresql://repopilot:repopilot@localhost:55433/repopilot",
        completed_at=NOW,
        executor_factory=CalibrationFakeExecutor,
    )

    assert len(report.runs) == 4
    assert sum(item.total_attempts for item in report.runs) == 8
    assert all(item.completed_attempts == 2 for item in report.runs)
    assert report.total_estimated_cost_usd == 0
    assert not report.holdout_accessed
    assert not hasattr(report, "success_rate")

    report_path = tmp_path / "results/p7-dev/calibration-report.json"
    write_calibration_artifact(report_path, report)
    reference = _artifact_ref(
        report_path,
        relative_path="results/p7-dev/calibration-report.json",
    )
    marker = build_formal_freeze_marker(
        matrix=matrix,
        plan=plan,
        report=report,
        calibration_report_ref=reference,
        source_commit=COMMIT,
        source_clean=True,
        frozen_at=NOW,
    )
    assert marker.source_clean
    assert marker.embedding_code_revision == ("3baf9e3ac750e76e8edd3019170176884695fb94")
    assert tuple(marker.arm_configuration_sha256) == tuple(sorted(marker.arm_configuration_sha256))

    dirty = build_dev_calibration_plan(
        matrix=matrix,
        authorization=authorization,
        tasks=tasks,
        created_at=NOW,
        source_commit=COMMIT,
        source_dirty=True,
    )
    with pytest.raises(CalibrationError) as captured:
        build_formal_freeze_marker(
            matrix=matrix,
            plan=dirty,
            report=report,
            calibration_report_ref=reference,
            source_commit=COMMIT,
            source_clean=True,
            frozen_at=NOW,
        )
    assert captured.value.code is CalibrationErrorCode.SOURCE_SNAPSHOT_UNCLEAN


def test_reported_cost_overrun_stops_before_another_arm_and_retains_raw_run(
    tmp_path: Path,
) -> None:
    matrix = load_experiment_matrix(MATRIX_PATH, repository_root=PROJECT_ROOT)
    tasks = load_dev_calibration_tasks(matrix, repository_root=PROJECT_ROOT)
    authorization = _authorization(matrix)
    tmp_path.joinpath("uv.lock").write_bytes(PROJECT_ROOT.joinpath("uv.lock").read_bytes())
    preflight = build_dev_calibration_preflight(
        matrix=matrix,
        tasks=tasks,
        repository_root=tmp_path,
        checked_at=NOW,
        facts=_facts(clean=True),
    )
    plan = build_dev_calibration_plan(
        matrix=matrix,
        authorization=authorization,
        tasks=tasks,
        created_at=NOW,
        source_commit=COMMIT,
        source_dirty=False,
    )

    with pytest.raises(CalibrationError) as captured:
        run_dev_calibration(
            plan=plan,
            matrix=matrix,
            authorization=authorization,
            preflight=preflight,
            tasks=tasks,
            repository_root=tmp_path,
            openai_api_key="test-placeholder-not-sent",
            database_url="postgresql://repopilot:repopilot@localhost:55433/repopilot",
            completed_at=NOW,
            executor_factory=lambda selection: CalibrationFakeExecutor(
                selection,
                reported_cost_usd=Decimal("5.00"),
            ),
        )
    assert captured.value.code is CalibrationErrorCode.BUDGET_LIMIT_EXCEEDED
    first = tmp_path / plan.run_slots[0].output_path
    second = tmp_path / plan.run_slots[1].output_path
    assert first.joinpath("manifest.json").is_file()
    assert not second.exists()
