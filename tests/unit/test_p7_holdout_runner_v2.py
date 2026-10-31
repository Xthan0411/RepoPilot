"""V2 runner authority, no-request defaults, plan locking, and report isolation."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from repopilot.enums import EvaluationRunnerErrorCode, RunStatus, TerminationReason  # noqa: E402
from repopilot.evals.runner import (  # noqa: E402
    EvaluationExecutionEvidence,
    EvaluationExecutionRequest,
)
from repopilot.schemas import ArtifactRef, UsageMetrics  # noqa: E402
from scripts import p7_holdout_readiness as old_readiness  # noqa: E402
from scripts import p7_holdout_readiness_v2 as probes  # noqa: E402
from scripts import p7_holdout_runner as shared  # noqa: E402
from scripts import p7_holdout_runner_v2 as runner  # noqa: E402
from scripts import p7_holdout_v2 as control  # noqa: E402
from scripts.p7_holdout_agent_inputs import AGENT_INPUTS_PATH, HoldoutAgentInputs  # noqa: E402
from scripts.p7_holdout_control import ENVIRONMENT_PATH, HoldoutEnvironmentManifest  # noqa: E402
from scripts.p7_holdout_requalification import (  # noqa: E402
    PLAN_V2_PATH,
    POST_REPAIR_FREEZE_PATH,
    HoldoutApplicationPlanV2,
    PostRepairFreeze,
    _sealed,
)


def _fixture_controls(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> tuple[control.V2Controls, control.HoldoutRuntimeReadinessV2]:
    """Build actual v2 models around deterministic fake readiness observations."""

    plan_raw = (PROJECT_ROOT / PLAN_V2_PATH).read_bytes()
    freeze_raw = (PROJECT_ROOT / POST_REPAIR_FREEZE_PATH).read_bytes()
    environment_raw = (PROJECT_ROOT / ENVIRONMENT_PATH).read_bytes()
    agent_raw = (PROJECT_ROOT / AGENT_INPUTS_PATH).read_bytes()
    plan = HoldoutApplicationPlanV2.model_validate_json(plan_raw)
    freeze = PostRepairFreeze.model_validate_json(freeze_raw)
    approved_at = datetime.now(UTC)
    authorization = _sealed(
        control.PaidHoldoutAuthorizationV2,
        {
            "approved_at": approved_at,
            "experiment_id": plan.experiment_id,
            "plan_id": plan.plan_id,
            "plan_sha256": plan.content_sha256,
            "plan_raw_sha256": hashlib.sha256(plan_raw).hexdigest(),
            "freeze_marker_sha256": freeze.content_sha256,
            "freeze_raw_sha256": hashlib.sha256(freeze_raw).hexdigest(),
            "matrix_sha256": plan.matrix_sha256,
            "max_total_cost_usd": Decimal("32.00"),
            "paid_execution_approved": True,
            "final_in_flight_response_overshoot_acknowledged": True,
            "readiness_required_before_paid": True,
        },
    )
    inputs = control.V2Controls(
        freeze=freeze,
        plan=plan,
        authorization=authorization,
        environment=HoldoutEnvironmentManifest.model_validate_json(environment_raw),
        agent_inputs=HoldoutAgentInputs.model_validate_json(agent_raw),
        controller_commit="f" * 40,
        freeze_raw=freeze_raw,
        plan_raw=plan_raw,
        authorization_raw=authorization.model_dump_json(indent=2).encode() + b"\n",
        environment_raw=environment_raw,
        agent_inputs_raw=agent_raw,
    )
    historical = old_readiness.HoldoutRuntimeReadiness.model_validate_json(
        (PROJECT_ROOT / old_readiness.READINESS_PATH).read_bytes()
    )
    readiness = control.build_readiness(
        inputs,
        checked_at=approved_at,
        execution_controller_objects={
            path: "a" * 40 for path in sorted(control.REQUIRED_EXECUTION_PATHS)
        },
        embedding=historical.embedding,
        vector_backend=historical.vector_backend,
        images=historical.images,
    )
    target = root / control.READINESS_V2_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(readiness.model_dump_json(indent=2).encode() + b"\n")
    monkeypatch.setattr(runner, "load_controls", lambda root: inputs)
    monkeypatch.setattr(runner, "validate_readiness_bindings", lambda *args: None)
    return inputs, readiness


def _controls(monkeypatch: pytest.MonkeyPatch, root: Path) -> shared._Controls:
    _fixture_controls(monkeypatch, root)
    return runner._load_bound_readiness(root)


def _progress(controls: shared._Controls, *, cost: str = "0") -> shared._Progress:
    return shared._Progress(
        outcomes=(),
        pending_slots=controls.plan.run_slots,
        interrupted_slots=(),
        known_cost_usd=Decimal(cost),
        reserved_unknown_cost_usd=Decimal("0"),
    )


def test_loader_returns_actual_v2_models_without_legacy_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    assert isinstance(controls.plan, HoldoutApplicationPlanV2)
    assert isinstance(controls.authorization, control.PaidHoldoutAuthorizationV2)
    assert isinstance(controls.readiness, control.HoldoutRuntimeReadinessV2)
    assert hashlib.sha256(controls.readiness_raw).hexdigest() != shared.READINESS_RAW_SHA256
    assert all(controls.plan.plan_id in slot.output_path for slot in controls.plan.run_slots)


@pytest.mark.parametrize("case", ["missing", "legacy", "malformed", "symlink"])
def test_loader_rejects_missing_old_or_invalid_readiness_without_secret_echo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    _fixture_controls(monkeypatch, tmp_path)
    path = tmp_path / control.READINESS_V2_PATH
    path.unlink()
    if case == "legacy":
        path.write_bytes((PROJECT_ROOT / old_readiness.READINESS_PATH).read_bytes())
    elif case == "malformed":
        path.write_text('{"secret":"should-never-appear"}', encoding="utf-8")
    elif case == "symlink":
        path.symlink_to(PROJECT_ROOT / old_readiness.READINESS_PATH)
    with pytest.raises(shared.HoldoutRunnerError) as error:
        runner._load_bound_readiness(tmp_path)
    assert error.value.code is shared.HoldoutRunnerErrorCode.SOURCE_ARTIFACT_INVALID
    assert "should-never-appear" not in str(error.value)


def test_loader_rejects_binding_or_controller_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fixture_controls(monkeypatch, tmp_path)

    def invalid(*args: object) -> None:
        raise ValueError("private stale or modified readiness details")

    monkeypatch.setattr(runner, "validate_readiness_bindings", invalid)
    with pytest.raises(shared.HoldoutRunnerError) as error:
        runner._load_bound_readiness(tmp_path)
    assert str(error.value) == "source_identity_mismatch"


def test_loader_rechecks_dependencies_and_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fixture_controls(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "verify_runtime_environment", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(shared.HoldoutRunnerError) as credential:
        runner._load_and_verify_controls(tmp_path, require_credential=True)
    assert credential.value.code is shared.HoldoutRunnerErrorCode.PROVIDER_CREDENTIAL_REQUIRED
    monkeypatch.setattr(probes, "_missing_dependencies", lambda: ("sentence_transformers",))
    with pytest.raises(shared.HoldoutRunnerError) as dependency:
        runner._load_and_verify_controls(tmp_path, require_credential=False)
    assert dependency.value.code is shared.HoldoutRunnerErrorCode.LOCAL_DEPENDENCY_UNAVAILABLE


@pytest.mark.parametrize("case", ["image", "checkout", "changed_image"])
def test_runtime_assets_must_still_match_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    images = controls.readiness.images
    if case == "changed_image":
        images = (
            images[0].model_copy(update={"local_image_id": "sha256:" + "e" * 64}),
            *images[1:],
        )
    monkeypatch.setattr(
        probes,
        "_resolve_images",
        lambda *args: (images, ("django__django-15902",) if case == "image" else ()),
    )
    monkeypatch.setattr(
        probes,
        "_checkouts",
        lambda *args: ("django__django-15902",) if case == "checkout" else (),
    )
    with pytest.raises(shared.HoldoutRunnerError) as error:
        runner._verify_runtime_assets(tmp_path, controls)
    expected = {
        "image": shared.HoldoutRunnerErrorCode.SANDBOX_IMAGE_UNAVAILABLE,
        "checkout": shared.HoldoutRunnerErrorCode.CHECKOUT_UNAVAILABLE,
        "changed_image": shared.HoldoutRunnerErrorCode.SOURCE_IDENTITY_MISMATCH,
    }
    assert error.value.code is expected[case]


def test_default_dry_run_never_locks_writes_or_constructs_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_load_and_verify_controls", lambda *args, **kwargs: controls)
    monkeypatch.setattr(shared, "_inspect_progress", lambda *args: _progress(controls))

    def forbidden(**kwargs: object) -> Any:
        raise AssertionError("executor constructed in dry-run")

    monkeypatch.setattr(shared, "_real_executor_factory", forbidden)
    monkeypatch.setattr(sys, "argv", ["runner_v2", "--repository-root", str(tmp_path)])
    assert runner.main() == 0
    assert "provider_requests=0" in capsys.readouterr().out
    assert not (tmp_path / "results/p7-holdout/locks").exists()
    assert not (tmp_path / runner.EXECUTION_REPORT_V2_PATH).exists()


def test_wrong_confirmation_precedes_lock_and_progress_inspection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_load_and_verify_controls", lambda *args, **kwargs: controls)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner_v2",
            "--repository-root",
            str(tmp_path),
            "--execute",
            "--confirm-plan-sha256",
            "0" * 64,
        ],
    )
    with pytest.raises(shared.HoldoutRunnerError) as error:
        runner.main()
    assert error.value.code is shared.HoldoutRunnerErrorCode.INVALID_CONFIGURATION
    assert not (tmp_path / "results/p7-holdout/locks").exists()


def test_paid_default_is_one_slot_and_lock_covers_shared_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_load_and_verify_controls", lambda *args, **kwargs: controls)
    received: list[dict[str, Any]] = []

    def invoke(**kwargs: Any) -> int:
        received.append(kwargs)
        with (
            pytest.raises(shared.HoldoutRunnerError) as error,
            runner._execution_lock(tmp_path, controls.plan.plan_id),
        ):
            raise AssertionError("second invocation acquired the lock")
        assert error.value.code is shared.HoldoutRunnerErrorCode.CONCURRENT_EXECUTION
        return 0

    monkeypatch.setattr(shared, "_run_loaded_controls", invoke)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner_v2",
            "--repository-root",
            str(tmp_path),
            "--execute",
            "--confirm-plan-sha256",
            controls.plan.content_sha256,
        ],
    )
    assert runner.main() == 0
    assert received[0]["max_new_slots"] == 1
    assert received[0]["continue_after_technical_failure"] is False
    with runner._execution_lock(tmp_path, controls.plan.plan_id):
        pass  # released even when the previous invocation ended


def test_execution_lock_released_on_error_and_rejects_symlinks(tmp_path: Path) -> None:
    with (
        pytest.raises(RuntimeError, match="interrupted"),
        runner._execution_lock(tmp_path, "test-plan"),
    ):
        raise RuntimeError("interrupted")
    with runner._execution_lock(tmp_path, "test-plan"):
        pass
    lock = tmp_path / "results/p7-holdout/locks/test-plan.lock"
    assert lock.stat().st_mode & 0o777 == 0o600
    lock.unlink()
    outside = tmp_path / "outside"
    outside.write_text("unchanged", encoding="utf-8")
    lock.symlink_to(outside)
    with (
        pytest.raises(shared.HoldoutRunnerError) as error,
        runner._execution_lock(tmp_path, "test-plan"),
    ):
        pass
    assert error.value.code is shared.HoldoutRunnerErrorCode.OUTPUT_CONFLICT
    assert outside.read_text(encoding="utf-8") == "unchanged"


def test_execution_lock_blocks_a_separate_process(tmp_path: Path) -> None:
    code = """import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(42)
finally:
    os.close(fd)
"""
    lock = tmp_path / "results/p7-holdout/locks/test-plan.lock"
    with runner._execution_lock(tmp_path, "test-plan"):
        blocked = subprocess.run(
            (sys.executable, "-c", code, str(lock)), capture_output=True, check=False, timeout=10
        )
    released = subprocess.run(
        (sys.executable, "-c", code, str(lock)), capture_output=True, check=False, timeout=10
    )
    assert blocked.returncode == 42
    assert released.returncode == 0


def test_lock_rejects_path_traversal(tmp_path: Path) -> None:
    with (
        pytest.raises(shared.HoldoutRunnerError) as error,
        runner._execution_lock(tmp_path, "../../escape"),
    ):
        pass
    assert error.value.code is shared.HoldoutRunnerErrorCode.INVALID_CONFIGURATION
    assert not (tmp_path / "results").exists()


def test_budget_guard_precedes_slot_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test")
    monkeypatch.setattr(
        shared, "_inspect_progress", lambda *args: _progress(controls, cost="31.01")
    )

    def slot_factory(arm: str, task: str) -> Any:
        raise AssertionError("over-budget slot constructed")

    monkeypatch.setattr(shared, "_real_executor_factory", lambda **kwargs: slot_factory)
    with pytest.raises(shared.HoldoutRunnerError) as error:
        shared._run_loaded_controls(
            root=tmp_path,
            controls=controls,
            execute=True,
            confirm_plan_sha256=controls.plan.content_sha256,
            max_new_slots=1,
            continue_after_technical_failure=False,
            report_writer=runner._write_final_report_if_complete,
        )
    assert error.value.code is shared.HoldoutRunnerErrorCode.BUDGET_EXHAUSTED


def _outcomes(controls: shared._Controls) -> tuple[shared.HoldoutSlotOutcome, ...]:
    return tuple(
        shared.HoldoutSlotOutcome(
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
            usage=UsageMetrics(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                estimated_cost_usd=Decimal("0.10"),
                latency_ms=1,
            ),
            reserved_unknown_cost_usd=Decimal("0"),
        )
        for slot in controls.plan.run_slots
    )


def test_technical_failure_stops_before_executor_and_preserves_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    slot = controls.plan.run_slots[0]
    failure = shared.HoldoutSlotOutcome(
        arm_id=slot.arm_id,
        task_id=slot.task_id,
        run_id=slot.run_id,
        output_path=slot.output_path,
        manifest_ref=ArtifactRef(
            relative_path="failure.json",
            sha256="a" * 64,
            size_bytes=1,
            media_type="application/json",
        ),
        attempt_status="technical_failure",
        execution_count=1,
        technical_failure_code=EvaluationRunnerErrorCode.EXECUTION_FAILURE,
        reserved_unknown_cost_usd=Decimal("1"),
    )
    progress = shared._Progress(
        (failure,), controls.plan.run_slots[1:], (), Decimal("0"), Decimal("1")
    )
    monkeypatch.setattr(shared, "_inspect_progress", lambda *args: progress)
    with pytest.raises(shared.HoldoutRunnerError) as error:
        shared._run_loaded_controls(
            root=tmp_path,
            controls=controls,
            execute=True,
            confirm_plan_sha256=controls.plan.content_sha256,
            max_new_slots=1,
            continue_after_technical_failure=False,
            report_writer=runner._write_final_report_if_complete,
        )
    assert error.value.code is shared.HoldoutRunnerErrorCode.TECHNICAL_FAILURE
    assert progress.reserved_unknown_cost_usd == Decimal("1")


@pytest.mark.parametrize("interrupt_first", [False, True])
def test_v2_slots_reuse_real_append_only_failure_and_resume_protocol(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, interrupt_first: bool
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    slot = controls.plan.run_slots[0]
    source = controls.agent_inputs.tasks[0]
    local_task = source.model_copy(
        update={"repository_path": str(PROJECT_ROOT), "repository_url": None}
    )
    monkeypatch.setattr(shared, "_runtime_task", lambda *args: local_task)
    manifest = shared._expected_manifest(PROJECT_ROOT, controls, slot)
    monkeypatch.setattr(shared, "_expected_manifest", lambda *args: manifest)

    class FailedExecutor:
        def __init__(self, interrupt: bool = False) -> None:
            self.interrupt = interrupt

        def execute(self, request: EvaluationExecutionRequest) -> EvaluationExecutionEvidence:
            request.output_dir.joinpath("retained-debug.txt").write_text(
                "simulated technical failure", encoding="utf-8"
            )
            if self.interrupt:
                raise KeyboardInterrupt
            raise RuntimeError("simulated execution failure")

    if interrupt_first:
        with pytest.raises(KeyboardInterrupt):
            shared._execute_slot(
                root=tmp_path,
                controls=controls,
                slot=slot,
                executor=FailedExecutor(True),
                accounted_cost_before=Decimal("0"),
            )
        interrupted = shared._inspect_progress(tmp_path, controls)
        assert interrupted.interrupted_slots == (slot,)
        assert interrupted.reserved_unknown_cost_usd == Decimal("1")
    shared._execute_slot(
        root=tmp_path,
        controls=controls,
        slot=slot,
        executor=FailedExecutor(),
        accounted_cost_before=Decimal(int(interrupt_first)),
    )
    progress = shared._inspect_progress(tmp_path, controls)
    assert len(progress.outcomes) == 1
    assert progress.technical_failures == 1
    assert progress.interrupted_slots == ()
    assert progress.known_cost_usd == 0
    assert progress.reserved_unknown_cost_usd == Decimal(1 + int(interrupt_first))
    assert progress.outcomes[0].execution_count == 1 + int(interrupt_first)
    assert (tmp_path / slot.output_path / "manifest.json").is_file()
    assert len(tuple((tmp_path / slot.output_path).rglob("retained-debug.txt"))) == 1 + int(
        interrupt_first
    )
    assert manifest.source_commit == controls.plan.frozen_source_commit
    assert source.issue_body not in manifest.model_dump_json()


def test_final_report_is_v2_create_only_and_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    progress = shared._Progress(_outcomes(controls), (), (), Decimal("3.20"), Decimal("0"))
    first = runner._write_final_report_if_complete(tmp_path, controls, progress)
    second = runner._write_final_report_if_complete(tmp_path, controls, progress)
    assert first == second
    assert first.execution_version == "holdout-execution-v2"
    assert first.plan_content_sha256 == control.APPROVED_PLAN_SHA256
    assert first.test_protocol_version == "isolated-regression-tests-v1"
    assert first.aggregation_deferred_to == "P7-S7"
    assert not (tmp_path / shared.EXECUTION_REPORT_PATH).exists()
    assert (tmp_path / runner.EXECUTION_REPORT_V2_PATH).stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValidationError):
        shared.HoldoutExecutionReport.model_validate_json(first.model_dump_json())
    invalid = first.model_dump(mode="python")
    invalid["known_estimated_cost_usd"] = "999"
    with pytest.raises(ValidationError):
        runner.HoldoutExecutionReportV2.model_validate(invalid)


def test_partial_run_never_claims_execution_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    with pytest.raises(shared.HoldoutRunnerError) as error:
        runner._write_final_report_if_complete(tmp_path, controls, _progress(controls))
    assert error.value.code is shared.HoldoutRunnerErrorCode.OUTPUT_CONFLICT
    assert not (tmp_path / runner.EXECUTION_REPORT_V2_PATH).exists()


@pytest.mark.parametrize("completed_before", [30, 31])
def test_overshoot_stops_but_writes_final_report_when_last_slot_completed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, completed_before: int
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    base = _outcomes(controls)
    completed = tuple(
        outcome.model_copy(
            update={"usage": outcome.usage.model_copy(update={"estimated_cost_usd": Decimal("1")})}
        )
        for outcome in base[:completed_before]
        if outcome.usage is not None
    )
    next_outcome = base[completed_before]
    assert next_outcome.usage is not None
    final_outcome = next_outcome.model_copy(
        update={
            "usage": next_outcome.usage.model_copy(
                update={"estimated_cost_usd": Decimal(33 - completed_before)}
            )
        }
    )
    before = shared._Progress(
        completed,
        controls.plan.run_slots[completed_before:],
        (),
        Decimal(completed_before),
        Decimal("0"),
    )
    after = shared._Progress(
        (*completed, final_outcome),
        controls.plan.run_slots[completed_before + 1 :],
        (),
        Decimal("33"),
        Decimal("0"),
    )
    executions: list[str] = []
    constructions: list[tuple[str, str]] = []

    def factory(arm: str, task: str) -> Any:
        constructions.append((arm, task))
        return object()

    def execute(**kwargs: Any) -> None:
        executions.append(str(kwargs["slot"].run_id))

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test")
    monkeypatch.setattr(shared, "_inspect_progress", lambda *args: after if executions else before)
    monkeypatch.setattr(shared, "_real_executor_factory", lambda **kwargs: factory)
    monkeypatch.setattr(shared, "_execute_slot", execute)
    with pytest.raises(shared.HoldoutRunnerError) as error:
        shared._run_loaded_controls(
            root=tmp_path,
            controls=controls,
            execute=True,
            confirm_plan_sha256=controls.plan.content_sha256,
            max_new_slots=32,
            continue_after_technical_failure=False,
            report_writer=runner._write_final_report_if_complete,
        )
    assert error.value.code is shared.HoldoutRunnerErrorCode.BUDGET_EXHAUSTED
    assert len(executions) == len(constructions) == 1
    report_path = tmp_path / runner.EXECUTION_REPORT_V2_PATH
    if completed_before == 31:
        report = runner.HoldoutExecutionReportV2.model_validate_json(report_path.read_bytes())
        assert report.final_in_flight_overshoot_observed
        assert report.application_accounted_cost_usd == Decimal("33")
        assert report.completed_slot_count == 32
    else:
        assert not report_path.exists()
