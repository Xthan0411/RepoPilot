"""P7-S6 Gold-free, budgeted, resumable formal-run controller tests."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.p7_holdout_agent_inputs as agent_inputs  # noqa: E402
import scripts.p7_holdout_control as control  # noqa: E402
import scripts.p7_holdout_readiness as readiness  # noqa: E402
import scripts.p7_holdout_runner as runner  # noqa: E402
from repopilot.enums import (  # noqa: E402
    EvaluationRunnerErrorCode,
    ModelProvider,
    ReproductionStatus,
    RunStatus,
    TerminationReason,
    ValidationStatus,
)
from repopilot.evals.baselines import BaselineEvaluationExecutor  # noqa: E402
from repopilot.evals.runner import (  # noqa: E402
    EvaluationExecutionEvidence,
    EvaluationExecutionRequest,
    EvaluationRunManifest,
    RepairWorkflowEvaluationExecutor,
)
from repopilot.schemas import (  # noqa: E402
    ArtifactRef,
    FinalResult,
    LocalizationMetrics,
    ReproductionResult,
    TaskMetrics,
    UsageMetrics,
    ValidationResult,
)


def _controls(monkeypatch: pytest.MonkeyPatch) -> runner._Controls:
    monkeypatch.setattr("scripts.p7_holdout_runner.platform.system", lambda: "Linux")
    monkeypatch.setattr("scripts.p7_holdout_runner.platform.machine", lambda: "x86_64")
    monkeypatch.setattr(
        runner,
        "_git",
        lambda root, *args: "f" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(runner, "_verify_frozen_runtime", lambda **kwargs: object())
    monkeypatch.setattr(runner, "_verify_runtime_assets", lambda root, controls: None)
    for name in tuple(os.environ):
        if name.startswith("REPOPILOT_"):
            monkeypatch.delenv(name, raising=False)
    return runner._load_and_verify_controls(PROJECT_ROOT, require_credential=False)


def test_committed_runtime_inputs_are_exact_and_gold_free() -> None:
    readiness_raw = PROJECT_ROOT.joinpath(readiness.READINESS_PATH).read_bytes()
    inputs_raw = PROJECT_ROOT.joinpath(agent_inputs.AGENT_INPUTS_PATH).read_bytes()
    inputs = agent_inputs.HoldoutAgentInputs.model_validate_json(inputs_raw)

    assert hashlib.sha256(readiness_raw).hexdigest() == runner.READINESS_RAW_SHA256
    assert hashlib.sha256(inputs_raw).hexdigest() == runner.AGENT_INPUTS_RAW_SHA256
    assert len(inputs.tasks) == 8
    assert not inputs.source_rows_loaded_by_formal_runner
    assert not inputs.evaluator_gold_serialized
    assert inputs.provider_requests_made == 0
    assert inputs.paid_requests_made == 0
    for forbidden in (
        b'"expected_patch"',
        b'"gold_files"',
        b'"gold_patch"',
        b'"test_patch"',
        b'"FAIL_TO_PASS"',
        b'"PASS_TO_PASS"',
    ):
        assert forbidden not in inputs_raw


def test_formal_loader_never_opens_source_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        if path.name == "source-rows.json":
            raise AssertionError("formal runner attempted to load evaluator source rows")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    controls = _controls(monkeypatch)

    assert controls.agent_inputs.task_ids == control.EXPECTED_TASK_IDS
    assert "source-rows.json" not in inspect.getsource(runner)


def test_runner_wrapper_uses_an_explicit_withheld_sentinel() -> None:
    inputs = agent_inputs.HoldoutAgentInputs.model_validate_json(
        PROJECT_ROOT.joinpath(agent_inputs.AGENT_INPUTS_PATH).read_bytes()
    )
    wrapper = runner._runner_task(inputs.tasks[0])

    assert wrapper.task == inputs.tasks[0]
    assert wrapper.expected_patch == runner.WITHHELD_EVALUATOR_PATCH
    assert wrapper.gold_files == (runner.WITHHELD_EVALUATOR_FILE,)
    serialized_agent_task = wrapper.task.model_dump_json()
    assert "WITHHELD" not in serialized_agent_task
    assert "gold" not in serialized_agent_task.casefold()


def test_executor_uses_task_specific_qualified_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = _controls(monkeypatch)
    factory = runner._real_executor_factory(
        root=PROJECT_ROOT,
        controls=controls,
        openai_api_key="unit-test-key",
        database_url="postgresql://unit-test",
    )
    task_id = controls.plan.task_ids[0]
    executor = factory("dense-only-v1", task_id)
    binding = next(item for item in controls.environment.bindings if item.task_id == task_id)

    assert isinstance(executor, RepairWorkflowEvaluationExecutor)
    assert executor.sandbox_image_reference == binding.sandbox_image_reference
    assert executor.sandbox_image_reference != controls.plan.frozen_sandbox_image_reference
    assert executor.sandbox_factory is not None


@pytest.mark.parametrize("arm_id", control.EXPECTED_ARM_IDS)
def test_all_formal_arms_use_isolated_tests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arm_id: str,
) -> None:
    controls = _controls(monkeypatch)
    task = controls.agent_inputs.tasks[0].model_copy(
        update={"repository_path": str(tmp_path), "repository_url": None}
    )
    monkeypatch.setattr(runner, "_runtime_task", lambda *args: task)
    received: list[dict[str, Any]] = []
    sentinel = object()

    def isolated_factory(**kwargs: Any) -> Any:
        received.append(kwargs)
        return lambda output_root: sentinel

    monkeypatch.setattr(runner, "build_isolated_test_sandbox_factory", isolated_factory)
    executor = runner._real_executor_factory(
        root=PROJECT_ROOT,
        controls=controls,
        openai_api_key="unit-test-key",
        database_url="postgresql://unit-test",
    )(arm_id, task.task_id)
    assert isinstance(executor, (RepairWorkflowEvaluationExecutor, BaselineEvaluationExecutor))
    assert executor.sandbox_factory is not None
    assert executor.sandbox_factory(tmp_path) is sentinel
    assert received[0]["task"] == task
    assert received[0]["regression_patch"].startswith("diff --git ")
    assert "gold_patch" not in received[0]
    assert "expected_patch" not in received[0]


def test_manifest_binds_exact_slot_without_serializing_gold_or_host_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = _controls(monkeypatch)
    slot = controls.plan.run_slots[0]
    source = next(item for item in controls.agent_inputs.tasks if item.task_id == slot.task_id)
    local_task = source.model_copy(
        update={"repository_path": str(PROJECT_ROOT), "repository_url": None}
    )
    monkeypatch.setattr(runner, "_runtime_task", lambda *args: local_task)

    manifest = runner._expected_manifest(PROJECT_ROOT, controls, slot)
    binding = next(item for item in controls.environment.bindings if item.task_id == slot.task_id)
    serialized = manifest.model_dump_json()

    assert manifest.run_id == slot.run_id
    assert manifest.source_commit == controls.plan.frozen_source_commit
    assert manifest.configuration_sha256 == slot.configuration_sha256
    assert manifest.sandbox_image_reference == binding.sandbox_image_reference
    assert manifest.task_ids == (slot.task_id,)
    assert manifest.max_execution_attempts == runner.MAX_EXECUTION_ATTEMPTS
    assert runner.WITHHELD_EVALUATOR_PATCH not in serialized
    assert runner.WITHHELD_EVALUATOR_FILE not in serialized
    assert str(PROJECT_ROOT) not in serialized
    assert source.issue_body not in serialized


def test_wrong_plan_confirmation_stops_before_executor_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = _controls(monkeypatch)
    progress = runner._Progress(
        outcomes=(),
        pending_slots=controls.plan.run_slots,
        interrupted_slots=(),
        known_cost_usd=Decimal("0"),
        reserved_unknown_cost_usd=Decimal("0"),
    )
    monkeypatch.setattr(runner, "_load_and_verify_controls", lambda *args, **kwargs: controls)
    monkeypatch.setattr(runner, "_inspect_progress", lambda *args, **kwargs: progress)
    monkeypatch.setattr(
        runner,
        "_real_executor_factory",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("executor constructed")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["p7_holdout_runner", "--execute", "--confirm-plan-sha256", "0" * 64],
    )

    with pytest.raises(runner.HoldoutRunnerError) as error:
        runner.main()

    assert error.value.code is runner.HoldoutRunnerErrorCode.INVALID_CONFIGURATION


def _artifact(path: Path) -> ArtifactRef:
    content = path.read_bytes()
    return ArtifactRef(
        relative_path=path.name,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        media_type="application/json",
    )


class _JsonExecutor:
    def __init__(self, manifest: EvaluationRunManifest, *, interrupt_once: bool = False) -> None:
        self.manifest = manifest
        self.interrupt_once = interrupt_once
        self.requests: list[EvaluationExecutionRequest] = []

    def execute(self, request: EvaluationExecutionRequest) -> EvaluationExecutionEvidence:
        self.requests.append(request)
        if self.interrupt_once:
            self.interrupt_once = False
            request.output_dir.joinpath("partial.txt").write_text(
                "retained interrupted attempt",
                encoding="utf-8",
            )
            raise KeyboardInterrupt
        usage = UsageMetrics(
            input_tokens=5,
            output_tokens=1,
            total_tokens=6,
            estimated_cost_usd=Decimal("0.25"),
            latency_ms=1,
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
        artifacts_path = request.output_dir / "artifacts.json"
        provider_name = self.manifest.model
        if self.manifest.model_provider is ModelProvider.OPENAI:
            provider_name = (
                f"openai-responses:{self.manifest.model}:{self.manifest.prompt_versions[0]}"
            )
        result_path.write_text(result.model_dump_json(), encoding="utf-8")
        trace_path.write_text(
            json.dumps({"run_id": str(request.run_id), "task_id": request.task.task_id}),
            encoding="utf-8",
        )
        artifacts_path.write_text(
            json.dumps(
                {
                    "provider_name": provider_name,
                    "run_id": str(request.run_id),
                    "sandbox_image_reference": self.manifest.sandbox_image_reference,
                    "settings_sha256": self.manifest.configuration_sha256,
                    "task_id": request.task.task_id,
                    "prompt_versions": list(self.manifest.prompt_versions),
                    "result_path": "result.json",
                    "trace_path": "trace.json",
                }
            ),
            encoding="utf-8",
        )
        return EvaluationExecutionEvidence(
            task_id=request.task.task_id,
            run_id=request.run_id,
            status=RunStatus.STOPPED,
            termination_reason=TerminationReason.AGENT_STOPPED,
            task_success=False,
            usage=usage,
            result_ref=_artifact(result_path),
            trace_ref=_artifact(trace_path),
            artifact_manifest_ref=_artifact(artifacts_path),
        )


def test_execute_slot_persists_only_agent_task_and_budget_guard_precedes_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controls = _controls(monkeypatch)
    slot = controls.plan.run_slots[0].model_copy(
        update={"output_path": "results/p7-holdout/raw/unit/slot"}
    )
    source = next(item for item in controls.agent_inputs.tasks if item.task_id == slot.task_id)
    task = source.model_copy(update={"repository_path": str(PROJECT_ROOT), "repository_url": None})
    monkeypatch.setattr(runner, "_runtime_task", lambda *args: task)
    manifest = runner._expected_manifest(PROJECT_ROOT, controls, slot)
    monkeypatch.setattr(runner, "_expected_manifest", lambda *args: manifest)

    blocked = _JsonExecutor(manifest)
    with pytest.raises(runner.HoldoutRunnerError) as error:
        runner._execute_slot(
            root=tmp_path,
            controls=controls,
            slot=slot,
            executor=blocked,
            accounted_cost_before=Decimal("32.00"),
        )
    assert error.value.code is runner.HoldoutRunnerErrorCode.BUDGET_EXHAUSTED
    assert blocked.requests == []

    allowed = _JsonExecutor(manifest)
    report = runner._execute_slot(
        root=tmp_path,
        controls=controls,
        slot=slot,
        executor=allowed,
        accounted_cost_before=Decimal("31.00"),
    )
    assert report.completed_attempts == 1
    assert len(allowed.requests) == 1
    request_payload: dict[str, Any] = allowed.requests[0].task.model_dump(mode="json")
    assert "expected_patch" not in request_payload
    assert "gold_files" not in request_payload
    persisted = tmp_path.joinpath(slot.output_path, "manifest.json").read_text(encoding="utf-8")
    assert runner.WITHHELD_EVALUATOR_PATCH not in persisted
    assert source.issue_body not in persisted


def test_output_hierarchy_rejects_nested_symlink(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "results").symlink_to(outside, target_is_directory=True)

    with pytest.raises(runner.HoldoutRunnerError) as create_error:
        runner._prepare_output_parent(tmp_path, "results/p7-holdout/raw/slot")
    with pytest.raises(runner.HoldoutRunnerError) as existing_error:
        runner._existing_output_directory(tmp_path, "results")

    assert create_error.value.code is runner.HoldoutRunnerErrorCode.OUTPUT_CONFLICT
    assert existing_error.value.code is runner.HoldoutRunnerErrorCode.OUTPUT_CONFLICT


def test_interrupted_slot_resumes_with_distinct_execution_and_cost_reserve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = _controls(monkeypatch)
    slot = original.plan.run_slots[0].model_copy(
        update={"output_path": "results/p7-holdout/raw/unit/resume"}
    )
    controls = runner._Controls(
        plan=original.plan.model_copy(update={"run_slots": (slot, *original.plan.run_slots[1:])}),
        authorization=original.authorization,
        environment=original.environment,
        readiness=original.readiness,
        agent_inputs=original.agent_inputs,
        readiness_raw=original.readiness_raw,
        agent_inputs_raw=original.agent_inputs_raw,
        controller_commit=original.controller_commit,
    )
    source = next(item for item in controls.agent_inputs.tasks if item.task_id == slot.task_id)
    task = source.model_copy(update={"repository_path": str(PROJECT_ROOT), "repository_url": None})
    monkeypatch.setattr(runner, "_runtime_task", lambda *args: task)
    manifest = runner._expected_manifest(PROJECT_ROOT, controls, slot)
    monkeypatch.setattr(runner, "_expected_manifest", lambda *args: manifest)

    with pytest.raises(KeyboardInterrupt):
        runner._execute_slot(
            root=tmp_path,
            controls=controls,
            slot=slot,
            executor=_JsonExecutor(manifest, interrupt_once=True),
            accounted_cost_before=Decimal("0"),
        )
    interrupted = runner._inspect_progress(tmp_path, controls)
    assert interrupted.interrupted_slots == (slot,)
    assert interrupted.reserved_unknown_cost_usd == Decimal("1")

    completed = runner._execute_slot(
        root=tmp_path,
        controls=controls,
        slot=slot,
        executor=_JsonExecutor(manifest),
        accounted_cost_before=interrupted.accounted_cost_usd,
    )
    assert completed.attempts[0].execution_count == 2
    resumed = runner._inspect_progress(tmp_path, controls)
    assert resumed.interrupted_slots == ()
    assert resumed.known_cost_usd == Decimal("0.25")
    assert resumed.reserved_unknown_cost_usd == Decimal("1")


def test_final_raw_index_is_create_only_idempotent_and_defers_aggregation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controls = _controls(monkeypatch)
    usage = UsageMetrics(
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        estimated_cost_usd=Decimal("0.10"),
        latency_ms=1,
    )
    outcomes = tuple(
        runner.HoldoutSlotOutcome(
            arm_id=slot.arm_id,
            task_id=slot.task_id,
            run_id=slot.run_id,
            output_path=slot.output_path,
            manifest_ref=ArtifactRef(
                relative_path=f"{slot.output_path}/manifest.json",
                sha256="a" * 64,
                size_bytes=1,
                media_type="application/json",
            ),
            attempt_status="completed",
            execution_count=1,
            run_status=RunStatus.STOPPED,
            termination_reason=TerminationReason.AGENT_STOPPED,
            task_success=False,
            usage=usage,
            reserved_unknown_cost_usd=Decimal("0"),
        )
        for slot in controls.plan.run_slots
    )
    progress = runner._Progress(
        outcomes=outcomes,
        pending_slots=(),
        interrupted_slots=(),
        known_cost_usd=Decimal("3.20"),
        reserved_unknown_cost_usd=Decimal("0"),
    )

    first = runner._write_final_report_if_complete(tmp_path, controls, progress)
    second = runner._write_final_report_if_complete(tmp_path, controls, progress)

    assert first == second
    assert first.slot_count == 32
    assert first.completed_slot_count == 32
    assert first.technical_failure_slot_count == 0
    assert first.aggregation_deferred_to == "P7-S7"
    assert not first.gold_visible_to_agent
    assert tmp_path.joinpath(runner.EXECUTION_REPORT_PATH).stat().st_mode & 0o777 == 0o600


def test_unknown_paid_attempts_are_reserved_conservatively() -> None:
    with pytest.raises(ValidationError):
        runner.HoldoutSlotOutcome(
            arm_id="dense-only-v1",
            task_id="django__django-15902",
            run_id=UUID("8010b0ee-77ab-5966-a26a-6496e1ddd19a"),
            output_path="results/p7-holdout/raw/example",
            manifest_ref=ArtifactRef(
                relative_path="results/p7-holdout/raw/example/manifest.json",
                sha256="a" * 64,
                size_bytes=1,
                media_type="application/json",
            ),
            attempt_status="technical_failure",
            execution_count=2,
            technical_failure_code=EvaluationRunnerErrorCode.EXECUTION_FAILURE,
            reserved_unknown_cost_usd=Decimal("1.00"),
        )

    valid = runner.HoldoutSlotOutcome(
        arm_id="dense-only-v1",
        task_id="django__django-15902",
        run_id=UUID("8010b0ee-77ab-5966-a26a-6496e1ddd19a"),
        output_path="results/p7-holdout/raw/example",
        manifest_ref=ArtifactRef(
            relative_path="results/p7-holdout/raw/example/manifest.json",
            sha256="a" * 64,
            size_bytes=1,
            media_type="application/json",
        ),
        attempt_status="technical_failure",
        execution_count=2,
        technical_failure_code=EvaluationRunnerErrorCode.EXECUTION_FAILURE,
        reserved_unknown_cost_usd=Decimal("2.00"),
    )
    assert valid.reserved_unknown_cost_usd == Decimal("2.00")


def test_default_cli_is_read_only_and_does_not_build_provider_executor(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    controls = _controls(monkeypatch)
    progress = runner._Progress(
        outcomes=(),
        pending_slots=controls.plan.run_slots,
        interrupted_slots=(),
        known_cost_usd=Decimal("0"),
        reserved_unknown_cost_usd=Decimal("0"),
    )
    monkeypatch.setattr(runner, "_load_and_verify_controls", lambda *args, **kwargs: controls)
    monkeypatch.setattr(runner, "_inspect_progress", lambda *args, **kwargs: progress)
    monkeypatch.setattr(
        runner,
        "_real_executor_factory",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("paid executor constructed")),
    )
    monkeypatch.setattr(sys, "argv", ["p7_holdout_runner"])

    assert runner.main() == 0
    output = capsys.readouterr().out
    assert "execute=false" in output
    assert "provider_requests=0" in output
