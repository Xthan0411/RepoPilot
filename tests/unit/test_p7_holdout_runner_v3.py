"""Zero-network v3 authorization, GPU-cache injection, and carried-budget checks."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from repopilot.embeddings import EmbeddingRequest, EmbeddingResponse  # noqa: E402
from repopilot.enums import EvaluationRunnerErrorCode, RunStatus, TerminationReason  # noqa: E402
from repopilot.evals.runner import (  # noqa: E402
    EvaluationExecutionEvidence,
    EvaluationExecutionRequest,
    RepairWorkflowEvaluationExecutor,
)
from repopilot.schemas import ArtifactRef, UsageMetrics  # noqa: E402
from scripts import p7_gpu_execution_profile as profile  # noqa: E402
from scripts import p7_holdout_gpu as control  # noqa: E402
from scripts import p7_holdout_readiness as historical  # noqa: E402
from scripts import p7_holdout_readiness_v2 as probes  # noqa: E402
from scripts import p7_holdout_runner as shared  # noqa: E402
from scripts import p7_holdout_runner_v2 as previous  # noqa: E402
from scripts import p7_holdout_runner_v3 as runner  # noqa: E402
from scripts import p7_repository_workload_preflight as workload  # noqa: E402
from scripts.p7_gpu_cache_acceptance import CacheAcceptanceError, ReadOnlyGpuCache  # noqa: E402
from scripts.p7_gpu_cpu_probe import PINNED_IDENTITY  # noqa: E402
from scripts.p7_holdout_agent_inputs import AGENT_INPUTS_PATH, HoldoutAgentInputs  # noqa: E402
from scripts.p7_holdout_control import ENVIRONMENT_PATH, HoldoutEnvironmentManifest  # noqa: E402
from scripts.p7_holdout_requalification import _sealed  # noqa: E402


class NeverCalledEmbeddingProvider:
    identity = PINNED_IDENTITY

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise AssertionError("test must not perform any actual embedding")


def _fixture_controls(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> tuple[control.GpuControls, control.GpuReadiness]:
    """Real sealed v3 models; only source probes are replaced with offline fixtures."""
    report = profile.load_qualification_report(ROOT)
    objects = {path: "a" * 40 for path in sorted(control.REQUIRED_EXECUTION_PATHS)}
    monkeypatch.setattr(control, "verify_qualification", lambda root: report)
    monkeypatch.setattr(workload, "verify_source", lambda root: ("f" * 40, objects))
    monkeypatch.setattr(control, "_objects_at", lambda *args: objects)
    monkeypatch.setattr(control, "_git", lambda *args: "f" * 40)
    now = datetime(2026, 11, 1, tzinfo=UTC)
    freeze, plan = control.build_plan(ROOT, frozen_at=now)
    freeze_raw, plan_raw = freeze.model_dump_json().encode(), plan.model_dump_json().encode()
    auth = _sealed(
        control.GpuAuthorization,
        {
            "authorization_id": f"{plan.plan_id}-usd32-ethan",
            "approved_at": now + timedelta(seconds=1),
            "experiment_id": plan.experiment_id,
            "plan_id": plan.plan_id,
            "plan_sha256": plan.content_sha256,
            "plan_raw_sha256": hashlib.sha256(plan_raw).hexdigest(),
            "freeze_marker_sha256": freeze.content_sha256,
            "freeze_raw_sha256": hashlib.sha256(freeze_raw).hexdigest(),
            "matrix_sha256": plan.matrix_sha256,
            "runtime_profile_sha256": plan.runtime_profile_sha256,
            "carry_forward_sha256": plan.carry_forward_sha256,
            "carried_unknown_cost_usd": Decimal("1.00"),
            "max_total_cost_usd": Decimal("32.00"),
            "new_execution_budget_usd": Decimal("31.00"),
            "paid_execution_approved": True,
            "final_in_flight_response_overshoot_acknowledged": True,
            "readiness_required_before_paid": True,
        },
    )
    environment_raw = (ROOT / ENVIRONMENT_PATH).read_bytes()
    agent_raw = (ROOT / AGENT_INPUTS_PATH).read_bytes()
    inputs = control.GpuControls(
        freeze=freeze,
        plan=plan,
        authorization=auth,
        environment=HoldoutEnvironmentManifest.model_validate_json(environment_raw),
        agent_inputs=HoldoutAgentInputs.model_validate_json(agent_raw),
        controller_commit="f" * 40,
        freeze_raw=freeze_raw,
        plan_raw=plan_raw,
        authorization_raw=auth.model_dump_json().encode(),
        environment_raw=environment_raw,
        agent_inputs_raw=agent_raw,
    )
    old = historical.HoldoutRuntimeReadiness.model_validate_json(
        (ROOT / historical.READINESS_PATH).read_bytes()
    )
    readiness = control.build_readiness(
        inputs,
        checked_at=now + timedelta(seconds=2),
        execution_controller_objects=objects,
        embedding=old.embedding,
        vector_backend=old.vector_backend,
        images=old.images,
        cpu_profile=freeze.cpu_runtime,
        gpu_cache=freeze.cache_evidence,
    )
    path = root / control.READINESS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(readiness.model_dump_json().encode())
    monkeypatch.setattr(runner, "load_controls", lambda root: inputs)
    monkeypatch.setattr(runner, "validate_readiness_bindings", lambda *args: None)
    return inputs, readiness


def _controls(monkeypatch: pytest.MonkeyPatch, root: Path) -> shared._Controls:
    _fixture_controls(monkeypatch, root)
    return runner._load_bound_readiness(root)


def _progress(controls: shared._Controls, *, cost: str = "0") -> shared._Progress:
    return shared._Progress((), controls.plan.run_slots, (), Decimal(cost), Decimal("0"))


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


def test_loader_returns_real_v3_models_and_new_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    assert isinstance(controls.plan, control.GpuPlan)
    assert isinstance(controls.authorization, control.GpuAuthorization)
    assert isinstance(controls.readiness, control.GpuReadiness)
    assert all("/p7-holdout-v3-" in slot.output_path for slot in controls.plan.run_slots)
    assert all(
        str(slot.run_id) != control.CarriedUnknownAttempt().run_id
        for slot in controls.plan.run_slots
    )


@pytest.mark.parametrize("case", ["missing", "legacy", "malformed", "symlink"])
def test_loader_rejects_missing_or_invalid_readiness_without_echo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    _fixture_controls(monkeypatch, tmp_path)
    target = tmp_path / control.READINESS_PATH
    target.unlink()
    if case == "legacy":
        target.write_bytes((ROOT / historical.READINESS_PATH).read_bytes())
    elif case == "malformed":
        target.write_text('{"secret":"must-never-be-printed"}', encoding="utf-8")
    elif case == "symlink":
        target.symlink_to(ROOT / historical.READINESS_PATH)
    with pytest.raises(shared.HoldoutRunnerError) as error:
        runner._load_bound_readiness(tmp_path)
    assert str(error.value) == "source_artifact_invalid"


def test_loader_rejects_stale_or_modified_bindings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fixture_controls(monkeypatch, tmp_path)

    def invalid(*args: object) -> None:
        raise ValueError("private diagnostics must not be printed")

    monkeypatch.setattr(runner, "validate_readiness_bindings", invalid)
    with pytest.raises(shared.HoldoutRunnerError, match=r"^source_identity_mismatch$"):
        runner._load_bound_readiness(tmp_path)


def test_dry_run_never_constructs_gpu_resources_or_locks_and_counts_old_reserve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_load_and_verify_controls", lambda *args, **kwargs: controls)
    monkeypatch.setattr(shared, "_inspect_progress", lambda *args: _progress(controls))
    monkeypatch.setattr(runner, "build_gpu_resources", lambda *args: pytest.fail("GPU loaded"))
    monkeypatch.setattr(sys, "argv", ["v3", "--repository-root", str(tmp_path)])
    assert runner.main() == 0
    output = capsys.readouterr().out
    assert '"accounted_cost_usd":"1.00"' in output
    assert '"carried_cost_usd":"1.00"' in output
    assert "provider_requests=0" in output
    assert not (tmp_path / "results/p7-holdout/locks").exists()
    assert not (tmp_path / runner.EXECUTION_REPORT_V3_PATH).exists()


def test_wrong_confirmation_precedes_lock_and_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_load_and_verify_controls", lambda *args, **kwargs: controls)
    monkeypatch.setattr(shared, "_run_loaded_controls", lambda **kwargs: pytest.fail("executed"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["v3", "--repository-root", str(tmp_path), "--execute", "--confirm-plan-sha256", "0" * 64],
    )
    with pytest.raises(shared.HoldoutRunnerError, match=r"^invalid_configuration$"):
        runner.main()
    assert not (tmp_path / "results/p7-holdout/locks").exists()


def test_execute_is_plan_locked_defaults_to_one_slot_and_uses_v3_factory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_load_and_verify_controls", lambda *args, **kwargs: controls)

    def invoke(**kwargs: Any) -> int:
        assert kwargs["max_new_slots"] == 1
        assert kwargs["carried_cost_usd"] == Decimal("1.00")
        assert kwargs["executor_factory_builder"] is runner._gpu_executor_factory
        with (
            pytest.raises(shared.HoldoutRunnerError, match=r"^concurrent_execution$"),
            previous._execution_lock(tmp_path, controls.plan.plan_id),
        ):
            pass
        return 0

    monkeypatch.setattr(shared, "_run_loaded_controls", invoke)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v3",
            "--repository-root",
            str(tmp_path),
            "--execute",
            "--confirm-plan-sha256",
            controls.plan.content_sha256,
        ],
    )
    assert runner.main() == 0


def test_runtime_loader_checks_credentials_dependencies_and_actual_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fixture_controls(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "verify_runtime_environment", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(shared.HoldoutRunnerError, match=r"^provider_credential_required$"):
        runner._load_and_verify_controls(tmp_path, require_credential=True)
    monkeypatch.setattr(probes, "_missing_dependencies", lambda: ("torch",))
    with pytest.raises(shared.HoldoutRunnerError, match=r"^local_dependency_unavailable$"):
        runner._load_and_verify_controls(tmp_path, require_credential=False)
    monkeypatch.setattr(probes, "_missing_dependencies", lambda: ())
    monkeypatch.setattr(previous, "_verify_runtime_assets", lambda *args: None)
    calls: list[Path] = []
    monkeypatch.setattr(runner, "verify_pinned_cache", lambda path: calls.append(path))
    runner._load_and_verify_controls(tmp_path, require_credential=False)
    assert calls == [tmp_path / profile.CACHE_PATH]

    def corrupt(path: Path) -> None:
        raise CacheAcceptanceError("secret must never be printed")

    monkeypatch.setattr(runner, "verify_pinned_cache", corrupt)
    with pytest.raises(shared.HoldoutRunnerError, match=r"^source_artifact_invalid$"):
        runner._load_and_verify_controls(tmp_path, require_credential=False)


def test_gpu_factory_injects_immutable_cache_without_legacy_resource_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    provider = NeverCalledEmbeddingProvider()
    cache = ReadOnlyGpuCache(tmp_path, PINNED_IDENTITY, {})
    monkeypatch.setattr(runner, "build_gpu_resources", lambda root: (provider, cache))
    monkeypatch.setattr(
        shared, "embedding_provider_from_settings", lambda *args: pytest.fail("legacy provider")
    )
    monkeypatch.setattr(
        shared, "embedding_cache_from_settings", lambda *args, **kwargs: pytest.fail("legacy cache")
    )
    factory = runner._gpu_executor_factory(
        root=ROOT, controls=controls, openai_api_key="unit-test", database_url="postgresql://unit"
    )
    for arm in ("dense-only-v1", "hybrid-rrf-v1"):
        executor = factory(arm, controls.plan.task_ids[0])
        assert isinstance(executor, RepairWorkflowEvaluationExecutor)
        assert executor.embedding_provider is provider
        assert executor.embedding_cache is cache
    with pytest.raises(CacheAcceptanceError, match="corpus_cache_miss"):
        cache.get(provider.identity, "0" * 64)


@pytest.mark.parametrize("only", ["provider", "cache"])
def test_shared_factory_rejects_partial_resource_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, only: str
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    kwargs: dict[str, Any] = {
        "embedding_provider": NeverCalledEmbeddingProvider(),
        "embedding_cache": ReadOnlyGpuCache(tmp_path, PINNED_IDENTITY, {}),
    }
    kwargs.pop("embedding_cache" if only == "provider" else "embedding_provider")
    with pytest.raises(shared.HoldoutRunnerError, match=r"^invalid_configuration$"):
        shared._real_executor_factory(
            root=ROOT,
            controls=controls,
            openai_api_key="unit",
            database_url="postgresql://unit",
            **kwargs,
        )


def test_carried_budget_stops_new_slot_before_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test")
    monkeypatch.setattr(
        shared, "_inspect_progress", lambda *args: _progress(controls, cost="30.01")
    )

    def builder(**kwargs: Any) -> shared.ExecutorFactory:
        return lambda arm, task: pytest.fail("over-budget slot constructed")

    with pytest.raises(shared.HoldoutRunnerError, match=r"^budget_exhausted$"):
        shared._run_loaded_controls(
            root=tmp_path,
            controls=controls,
            execute=True,
            confirm_plan_sha256=controls.plan.content_sha256,
            max_new_slots=1,
            continue_after_technical_failure=False,
            report_writer=runner._write_final_report_if_complete,
            executor_factory_builder=builder,
            carried_cost_usd=Decimal("1.00"),
        )


@pytest.mark.parametrize("value", [Decimal("-1"), Decimal("NaN"), Decimal("Infinity")])
def test_invalid_carried_budget_is_rejected_before_inspection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: Decimal
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    monkeypatch.setattr(
        shared, "_inspect_progress", lambda *args: pytest.fail("progress inspected")
    )
    with pytest.raises(shared.HoldoutRunnerError, match=r"^invalid_configuration$"):
        shared._run_loaded_controls(
            root=tmp_path,
            controls=controls,
            execute=False,
            confirm_plan_sha256=None,
            max_new_slots=1,
            continue_after_technical_failure=False,
            report_writer=runner._write_final_report_if_complete,
            carried_cost_usd=value,
        )


def test_new_technical_failure_stops_without_forgiving_old_reserve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    first = _outcomes(controls)[0]
    failure = shared.HoldoutSlotOutcome(
        **first.model_dump(
            exclude={
                "attempt_status",
                "run_status",
                "termination_reason",
                "task_success",
                "usage",
                "technical_failure_code",
                "reserved_unknown_cost_usd",
            }
        ),
        attempt_status="technical_failure",
        technical_failure_code=EvaluationRunnerErrorCode.EXECUTION_FAILURE,
        reserved_unknown_cost_usd=Decimal("1"),
    )
    progress = shared._Progress(
        (failure,), controls.plan.run_slots[1:], (), Decimal("0"), Decimal("1")
    )
    monkeypatch.setattr(shared, "_inspect_progress", lambda *args: progress)
    with pytest.raises(shared.HoldoutRunnerError, match=r"^technical_failure$"):
        shared._run_loaded_controls(
            root=tmp_path,
            controls=controls,
            execute=True,
            confirm_plan_sha256=controls.plan.content_sha256,
            max_new_slots=1,
            continue_after_technical_failure=False,
            report_writer=runner._write_final_report_if_complete,
            carried_cost_usd=Decimal("1.00"),
        )
    assert replace(progress, carried_cost_usd=Decimal("1")).accounted_cost_usd == Decimal("2")


def test_final_report_preserves_historical_reserve_separately_and_is_create_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    progress = shared._Progress(
        _outcomes(controls), (), (), Decimal("3.20"), Decimal("0"), Decimal("1.00")
    )
    first = runner._write_final_report_if_complete(tmp_path, controls, progress)
    assert runner._write_final_report_if_complete(tmp_path, controls, progress) == first
    assert first.application_accounted_cost_usd == Decimal("4.20")
    assert first.new_execution_accounted_cost_usd == Decimal("3.20")
    assert first.carried_unknown_cost_usd == Decimal("1.00")
    assert first.completed_slot_count == 32
    assert first.technical_failure_slot_count == 0
    assert not (tmp_path / shared.EXECUTION_REPORT_PATH).exists()
    assert not (tmp_path / previous.EXECUTION_REPORT_V2_PATH).exists()
    assert (tmp_path / runner.EXECUTION_REPORT_V3_PATH).stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValidationError):
        previous.HoldoutExecutionReportV2.model_validate_json(first.model_dump_json())
    payload = first.model_dump()
    payload["carried_unknown_cost_usd"] = Decimal("0")
    with pytest.raises(ValidationError):
        runner.HoldoutExecutionReportV3.model_validate(payload)


def test_incomplete_v3_never_creates_final_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    with pytest.raises(shared.HoldoutRunnerError, match=r"^output_conflict$"):
        runner._write_final_report_if_complete(tmp_path, controls, _progress(controls))
    assert not (tmp_path / runner.EXECUTION_REPORT_V3_PATH).exists()


@pytest.mark.parametrize("interrupt_first", [False, True])
def test_v3_append_only_failures_and_interruptions_retain_both_reserves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, interrupt_first: bool
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    slot = controls.plan.run_slots[0]
    task = controls.agent_inputs.tasks[0].model_copy(
        update={"repository_path": str(ROOT), "repository_url": None}
    )
    monkeypatch.setattr(shared, "_runtime_task", lambda *args: task)
    manifest = shared._expected_manifest(ROOT, controls, slot)
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
            raise RuntimeError("simulated technical failure")

    if interrupt_first:
        with pytest.raises(KeyboardInterrupt):
            shared._execute_slot(
                root=tmp_path,
                controls=controls,
                slot=slot,
                executor=FailedExecutor(True),
                accounted_cost_before=Decimal("1.00"),
            )
        interrupted = replace(
            shared._inspect_progress(tmp_path, controls), carried_cost_usd=Decimal("1.00")
        )
        assert interrupted.interrupted_slots == (slot,)
        assert interrupted.accounted_cost_usd == Decimal("2.00")
    shared._execute_slot(
        root=tmp_path,
        controls=controls,
        slot=slot,
        executor=FailedExecutor(),
        accounted_cost_before=Decimal(1 + int(interrupt_first)),
    )
    progress = replace(
        shared._inspect_progress(tmp_path, controls), carried_cost_usd=Decimal("1.00")
    )
    assert progress.technical_failures == 1
    assert progress.interrupted_slots == ()
    assert progress.reserved_unknown_cost_usd == Decimal(1 + int(interrupt_first))
    assert progress.accounted_cost_usd == Decimal(2 + int(interrupt_first))
    assert progress.outcomes[0].execution_count == 1 + int(interrupt_first)
    assert manifest.source_commit == controls.plan.frozen_source_commit
    assert len(tuple((tmp_path / slot.output_path).rglob("retained-debug.txt"))) == 1 + int(
        interrupt_first
    )


def test_completed_slot_passes_carried_total_to_inner_attempt_guard_and_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    outcomes = _outcomes(controls)
    before = _progress(controls)
    after = shared._Progress(
        (outcomes[0],), controls.plan.run_slots[1:], (), Decimal("0.10"), Decimal("0")
    )
    executions: list[Any] = []
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test")
    monkeypatch.setattr(shared, "_inspect_progress", lambda *args: after if executions else before)
    monkeypatch.setattr(shared, "_execute_slot", lambda **kwargs: executions.append(kwargs))

    def builder(**kwargs: Any) -> shared.ExecutorFactory:
        return lambda arm, task: NeverCalledTaskExecutor()

    shared._run_loaded_controls(
        root=tmp_path,
        controls=controls,
        execute=True,
        confirm_plan_sha256=controls.plan.content_sha256,
        max_new_slots=1,
        continue_after_technical_failure=False,
        report_writer=runner._write_final_report_if_complete,
        executor_factory_builder=builder,
        carried_cost_usd=Decimal("1.00"),
    )
    assert len(executions) == 1
    assert executions[0]["accounted_cost_before"] == Decimal("1.00")
    assert '"accounted_cost_usd":"1.10"' in capsys.readouterr().out
    assert not (tmp_path / runner.EXECUTION_REPORT_V3_PATH).exists()


class NeverCalledTaskExecutor:
    def execute(self, request: EvaluationExecutionRequest) -> EvaluationExecutionEvidence:
        raise AssertionError("test must not execute a real task")


def test_last_slot_inflight_overshoot_preserves_final_report_including_old_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controls = _controls(monkeypatch, tmp_path)
    outcomes = _outcomes(controls)
    changed: list[shared.HoldoutSlotOutcome] = []
    for index, outcome in enumerate(outcomes):
        assert outcome.usage is not None
        cost = Decimal("1") if index < 30 else Decimal("0") if index == 30 else Decimal("1.10")
        changed.append(
            outcome.model_copy(
                update={"usage": outcome.usage.model_copy(update={"estimated_cost_usd": cost})}
            )
        )
    before = shared._Progress(
        tuple(changed[:31]), controls.plan.run_slots[31:], (), Decimal("30"), Decimal("0")
    )
    after = shared._Progress(tuple(changed), (), (), Decimal("31.10"), Decimal("0"))
    executions: list[Any] = []
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test")
    monkeypatch.setattr(shared, "_inspect_progress", lambda *args: after if executions else before)
    monkeypatch.setattr(shared, "_execute_slot", lambda **kwargs: executions.append(kwargs))

    def builder(**kwargs: Any) -> shared.ExecutorFactory:
        return lambda arm, task: NeverCalledTaskExecutor()

    with pytest.raises(shared.HoldoutRunnerError, match=r"^budget_exhausted$"):
        shared._run_loaded_controls(
            root=tmp_path,
            controls=controls,
            execute=True,
            confirm_plan_sha256=controls.plan.content_sha256,
            max_new_slots=32,
            continue_after_technical_failure=False,
            report_writer=runner._write_final_report_if_complete,
            executor_factory_builder=builder,
            carried_cost_usd=Decimal("1.00"),
        )
    assert len(executions) == 1
    assert executions[0]["accounted_cost_before"] == Decimal("31.00")
    report = runner.HoldoutExecutionReportV3.model_validate_json(
        (tmp_path / runner.EXECUTION_REPORT_V3_PATH).read_bytes()
    )
    assert report.final_in_flight_overshoot_observed
    assert report.application_accounted_cost_usd == Decimal("32.10")
    assert report.new_execution_accounted_cost_usd == Decimal("31.10")
    assert report.completed_slot_count == 32
