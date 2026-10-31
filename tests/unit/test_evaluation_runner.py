"""P7-S1 append-only evaluation runner, resume, and provenance contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from repopilot import repair_workflow as workflow
from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    DatasetSplit,
    EvaluationAttemptStatus,
    EvaluationRunnerErrorCode,
    MemoryAccessMode,
    ModelProvider,
    RepairWorkflowErrorCode,
    RepositoryIngestErrorCode,
    ReproductionStatus,
    RunStatus,
    TerminationReason,
    ValidationStatus,
)
from repopilot.evals import (
    EVALUATION_MANIFEST_FILENAME,
    EvaluationExecutionEvidence,
    EvaluationExecutionRequest,
    EvaluationRunManifest,
    EvaluationRunnerError,
    EvaluationRunReport,
    EvaluationTaskExecutionError,
    FileEvaluationRunStore,
    build_evaluation_run_manifest,
    evaluation_attempt_id,
    evaluation_task_set_sha256,
    inspect_evaluation_run,
    load_evaluation_run_manifest,
    run_evaluation,
)
from repopilot.evals import runner as runner_module
from repopilot.retrieval import RepositoryIngestError
from repopilot.schemas import (
    ArtifactRef,
    EvaluationTaskSpec,
    FinalResult,
    LocalizationMetrics,
    MemoryNamespace,
    ReproductionResult,
    TaskMetrics,
    UsageMetrics,
    ValidationResult,
)
from repopilot.tools import ReadOnlyRepository

FIXTURES = Path(__file__).parents[1] / "fixtures"
RUN_ID = UUID("622bb729-f4bf-4dd8-b6f3-a591ee3f45ea")
CREATED_AT = datetime(2026, 8, 7, 20, tzinfo=UTC)
IMAGE = f"python@sha256:{'a' * 64}"
PROMPT_VERSIONS = ("eval-prompt-v1",)


def _evaluation_task(
    tmp_path: Path,
    task_id: str,
    *,
    split: DatasetSplit = DatasetSplit.DEV,
) -> EvaluationTaskSpec:
    payload = cast(
        dict[str, Any],
        json.loads((FIXTURES / "valid_evaluation_task.json").read_text(encoding="utf-8")),
    )
    task = cast(dict[str, Any], payload["task"])
    task["task_id"] = task_id
    task["repository_path"] = str(tmp_path / f"{task_id}-repository")
    payload["expected_patch"] = f"EVALUATOR_GOLD_{task_id}"
    payload["split"] = split.value
    return EvaluationTaskSpec.model_validate(payload)


def _tasks(tmp_path: Path, count: int = 2) -> tuple[EvaluationTaskSpec, ...]:
    return tuple(_evaluation_task(tmp_path, f"eval-task-{index:02d}") for index in range(count))


def _configuration() -> dict[str, object]:
    return {
        "profile": "eval",
        "model_provider": "fake",
        "model": "fake-deterministic",
        "max_steps": 30,
        "max_tokens": 100_000,
        "database_url": "<redacted>",
        "openai_api_key": None,
    }


def _manifest(
    tasks: tuple[EvaluationTaskSpec, ...],
    **updates: object,
) -> EvaluationRunManifest:
    values: dict[str, object] = {
        "experiment_id": "eval-runner-contract",
        "run_id": RUN_ID,
        "created_at": CREATED_AT,
        "source_commit": "b" * 40,
        "dependency_lock_sha256": "c" * 64,
        "dataset_manifest_sha256": "d" * 64,
        "configuration": _configuration(),
        "split": DatasetSplit.DEV,
        "arm_id": "repopilot-bm25",
        "tasks": tasks,
        "memory_namespace": None,
        "model_provider": ModelProvider.FAKE,
        "model": "fake-deterministic",
        "prompt_versions": PROMPT_VERSIONS,
        "sandbox_image_reference": IMAGE,
        "repetitions": 1,
        "max_execution_attempts": 3,
    }
    values.update(updates)
    return build_evaluation_run_manifest(**cast(Any, values))


def _artifact_ref(path: Path) -> ArtifactRef:
    content = path.read_bytes()
    return ArtifactRef(
        relative_path=path.name,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        media_type="application/json",
    )


def _failed_diagnostic(
    root: Path, manifest: EvaluationRunManifest
) -> tuple[Path, runner_module.EvaluationFailureDiagnostic]:
    store = FileEvaluationRunStore(root)
    event = store.read_events(manifest)[-1]
    diagnostic = store.read_failure_diagnostic(manifest, event)
    assert diagnostic is not None
    return (
        root / event.output_path / runner_module.EVALUATION_FAILURE_DIAGNOSTIC_FILENAME,
        diagnostic,
    )


class JsonFakeExecutor:
    """Trusted test adapter that emits structurally valid result/evidence bytes."""

    def __init__(
        self,
        manifest: EvaluationRunManifest,
        *,
        interrupt_once: bool = False,
        fail_task_id: str | None = None,
        corrupt_reference: bool = False,
        provider_name: str | None = None,
    ) -> None:
        self.manifest = manifest
        self.interrupt_once = interrupt_once
        self.fail_task_id = fail_task_id
        self.corrupt_reference = corrupt_reference
        self.provider_name = provider_name or manifest.model
        self.requests: list[EvaluationExecutionRequest] = []

    def execute(self, request: EvaluationExecutionRequest) -> EvaluationExecutionEvidence:
        self.requests.append(request)
        if self.interrupt_once:
            self.interrupt_once = False
            request.output_dir.joinpath("partial.txt").write_text(
                "interrupted execution remains immutable",
                encoding="utf-8",
            )
            raise KeyboardInterrupt
        if request.task.task_id == self.fail_task_id:
            raise EvaluationTaskExecutionError
        usage = UsageMetrics(
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            estimated_cost_usd=Decimal("0"),
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
        artifact_manifest_path = request.output_dir / "artifacts.json"
        result_path.write_text(f"{result.model_dump_json(indent=2)}\n", encoding="utf-8")
        trace_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": str(request.run_id),
                    "task_id": request.task.task_id,
                    "content_sha256": "e" * 64,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        artifact_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task_id": request.task.task_id,
                    "run_id": str(request.run_id),
                    "thread_id": request.thread_id,
                    "provider_name": self.provider_name,
                    "sandbox_image_reference": self.manifest.sandbox_image_reference,
                    "settings_sha256": self.manifest.configuration_sha256,
                    "prompt_versions": list(self.manifest.prompt_versions),
                    "result_path": "result.json",
                    "trace_path": "trace.json",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        result_ref = _artifact_ref(result_path)
        if self.corrupt_reference:
            result_ref = result_ref.model_copy(update={"sha256": "f" * 64})
        return EvaluationExecutionEvidence(
            task_id=request.task.task_id,
            run_id=request.run_id,
            status=result.status,
            termination_reason=result.termination_reason,
            task_success=result.task_metrics.task_success,
            usage=result.usage,
            result_ref=result_ref,
            trace_ref=_artifact_ref(trace_path),
            artifact_manifest_ref=_artifact_ref(artifact_manifest_path),
        )


def test_manifest_freezes_sanitized_provenance_without_gold_or_host_paths(
    tmp_path: Path,
) -> None:
    tasks = _tasks(tmp_path)
    manifest = _manifest(tasks)
    moved = tuple(
        item.model_copy(
            update={
                "task": item.task.model_copy(
                    update={"repository_path": f"/another/host/{item.task.task_id}"}
                )
            }
        )
        for item in reversed(tasks)
    )

    assert manifest.task_ids == ("eval-task-00", "eval-task-01")
    assert manifest.task_set_sha256 == evaluation_task_set_sha256(moved)
    assert manifest.configuration["database_url"] == "<redacted>"
    assert (
        manifest.configuration_sha256
        == hashlib.sha256(
            json.dumps(
                manifest.configuration,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
    )
    serialized = manifest.model_dump_json()
    assert "EVALUATOR_GOLD" not in serialized
    assert str(tmp_path) not in serialized
    assert (
        _manifest(
            tasks,
            configuration=RepoPilotSettings().redacted_snapshot(),
        ).configuration["max_tokens"]
        == 100_000
    )


def test_manifest_rejects_secret_split_drift_mutable_image_and_boolean_limits(
    tmp_path: Path,
) -> None:
    tasks = _tasks(tmp_path, 1)
    with pytest.raises(ValidationError, match="unredacted secret"):
        _manifest(tasks, configuration={"openai_api_key": "sk-secret-canary"})
    with pytest.raises(ValidationError, match="immutable digest"):
        _manifest(tasks, sandbox_image_reference="python:latest")
    with pytest.raises(ValidationError):
        _manifest(tasks, repetitions=True)

    holdout_memory = MemoryNamespace(
        experiment_id="holdout-eval",
        snapshot_id="snapshot-v1",
        split=DatasetSplit.HOLDOUT,
        access_mode=MemoryAccessMode.READ_ONLY,
    )
    with pytest.raises(ValidationError, match="split"):
        _manifest(tasks, memory_namespace=holdout_memory)

    copied_holdout = tasks[0].model_copy(update={"split": DatasetSplit.HOLDOUT})
    with pytest.raises(EvaluationRunnerError) as captured:
        _manifest((copied_holdout,), split=DatasetSplit.DEV)
    assert captured.value.code is EvaluationRunnerErrorCode.IDENTITY_MISMATCH


def test_runner_executes_each_slot_once_and_idempotently_rebuilds_report(
    tmp_path: Path,
) -> None:
    tasks = _tasks(tmp_path)
    manifest = _manifest(tasks, repetitions=2)
    output = tmp_path / "run"
    executor = JsonFakeExecutor(manifest)

    first = run_evaluation(
        manifest=manifest,
        tasks=tasks,
        output_dir=output,
        executor=executor,
        clock=lambda: CREATED_AT,
    )
    duplicate_executor = JsonFakeExecutor(manifest)
    persisted_manifest = load_evaluation_run_manifest(output)
    second = run_evaluation(
        manifest=persisted_manifest,
        tasks=tuple(reversed(tasks)),
        output_dir=output,
        executor=duplicate_executor,
        clock=lambda: CREATED_AT,
    )

    assert first == second
    assert inspect_evaluation_run(output) == first
    assert first.total_attempts == 4
    assert first.completed_attempts == 4
    assert first.failed_attempts == 0
    assert first.pending_attempts == 0
    assert len(executor.requests) == 4
    assert duplicate_executor.requests == []
    assert all("expected_patch" not in request.task.model_dump() for request in executor.requests)
    assert output.joinpath(EVALUATION_MANIFEST_FILENAME).stat().st_mode & 0o777 == 0o600
    assert output.stat().st_mode & 0o777 == 0o700
    assert len(tuple(output.joinpath("events").glob("*.json"))) == 8


def test_runner_accepts_exact_openai_adapter_identity(tmp_path: Path) -> None:
    tasks = _tasks(tmp_path, 1)
    model = "gpt-5.6-terra"
    prompt_version = "repopilot-agent-v1"
    manifest = _manifest(
        tasks,
        model_provider=ModelProvider.OPENAI,
        model=model,
        prompt_versions=(prompt_version,),
    )
    provider_name = f"openai-responses:{model}:{prompt_version}"

    report = run_evaluation(
        manifest=manifest,
        tasks=tasks,
        output_dir=tmp_path / "openai-run",
        executor=JsonFakeExecutor(manifest, provider_name=provider_name),
        clock=lambda: CREATED_AT,
    )

    assert report.completed_attempts == 1
    assert report.failed_attempts == 0


def test_runner_can_revalidate_retained_bytes_after_metadata_verifier_failure(
    tmp_path: Path,
) -> None:
    tasks = _tasks(tmp_path, 1)
    manifest = _manifest(tasks)
    output = tmp_path / "recoverable-run"
    report = run_evaluation(
        manifest=manifest,
        tasks=tasks,
        output_dir=output,
        executor=JsonFakeExecutor(manifest, corrupt_reference=True),
        clock=lambda: CREATED_AT,
    )
    store = FileEvaluationRunStore(output)
    events = store.read_events(manifest)

    evidence = store.recover_execution_evidence(manifest=manifest, event=events[-1])

    assert report.failed_attempts == 1
    assert events[-1].failure_code is EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY
    assert evidence.task_id == tasks[0].task.task_id
    assert evidence.usage.estimated_cost_usd == Decimal("0")


def test_before_attempt_guard_stops_without_creating_the_next_slot(tmp_path: Path) -> None:
    tasks = _tasks(tmp_path, 2)
    manifest = _manifest(tasks)
    output = tmp_path / "guarded-run"
    executor = JsonFakeExecutor(manifest)

    def stop_after_one(report: EvaluationRunReport) -> None:
        if report.completed_attempts == 1:
            raise RuntimeError("application budget reached")

    with pytest.raises(RuntimeError, match="application budget reached"):
        run_evaluation(
            manifest=manifest,
            tasks=tasks,
            output_dir=output,
            executor=executor,
            clock=lambda: CREATED_AT,
            before_attempt=stop_after_one,
        )

    assert len(executor.requests) == 1
    assert len(tuple(output.joinpath("events").glob("*.json"))) == 2


def test_interrupted_execution_resumes_in_new_directory_without_overwrite(
    tmp_path: Path,
) -> None:
    tasks = _tasks(tmp_path, 1)
    manifest = _manifest(tasks)
    output = tmp_path / "run"
    interrupted = JsonFakeExecutor(manifest, interrupt_once=True)

    with pytest.raises(KeyboardInterrupt):
        run_evaluation(
            manifest=manifest,
            tasks=tasks,
            output_dir=output,
            executor=interrupted,
            clock=lambda: CREATED_AT,
        )

    attempt_id = evaluation_attempt_id(RUN_ID, tasks[0].task.task_id, 0)
    first_output = output / "executions" / attempt_id / "0000"
    interrupted_report = inspect_evaluation_run(output)
    assert interrupted_report.pending_attempts == 1
    assert interrupted_report.attempts[0].status is EvaluationAttemptStatus.STARTED
    assert (
        first_output.joinpath("partial.txt").read_text(encoding="utf-8").startswith("interrupted")
    )
    resumed = JsonFakeExecutor(manifest)
    report = run_evaluation(
        manifest=manifest,
        tasks=tasks,
        output_dir=output,
        executor=resumed,
        clock=lambda: CREATED_AT,
    )
    events = FileEvaluationRunStore(output).read_events(manifest)

    assert tuple(event.status for event in events) == (
        EvaluationAttemptStatus.STARTED,
        EvaluationAttemptStatus.INTERRUPTED,
        EvaluationAttemptStatus.STARTED,
        EvaluationAttemptStatus.COMPLETED,
    )
    assert report.completed_attempts == 1
    assert report.attempts[0].execution_count == 2
    assert resumed.requests[0].execution_index == 1
    assert first_output.joinpath("partial.txt").is_file()
    assert output.joinpath("executions", attempt_id, "0001", "result.json").is_file()


def test_executor_and_artifact_failures_are_retained_and_not_retried(
    tmp_path: Path,
) -> None:
    tasks = _tasks(tmp_path)
    manifest = _manifest(tasks)
    output = tmp_path / "run"
    executor = JsonFakeExecutor(manifest, fail_task_id=tasks[0].task.task_id)

    report = run_evaluation(
        manifest=manifest,
        tasks=tasks,
        output_dir=output,
        executor=executor,
        clock=lambda: CREATED_AT,
    )

    assert report.completed_attempts == 1
    assert report.failed_attempts == 1
    assert report.attempts[0].failure_code is EvaluationRunnerErrorCode.EXECUTION_FAILURE
    duplicate = JsonFakeExecutor(manifest)
    assert (
        run_evaluation(
            manifest=manifest,
            tasks=tasks,
            output_dir=output,
            executor=duplicate,
            clock=lambda: CREATED_AT,
        )
        == report
    )
    assert duplicate.requests == []

    corrupt_tasks = _tasks(tmp_path / "corrupt", 1)
    corrupt_manifest = _manifest(
        corrupt_tasks,
        run_id=UUID("f74bfb87-0916-4435-8958-e0ddf4f87eb4"),
    )
    corrupt_report = run_evaluation(
        manifest=corrupt_manifest,
        tasks=corrupt_tasks,
        output_dir=tmp_path / "corrupt-run",
        executor=JsonFakeExecutor(corrupt_manifest, corrupt_reference=True),
        clock=lambda: CREATED_AT,
    )
    assert corrupt_report.failed_attempts == 1
    assert corrupt_report.attempts[0].failure_code is EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY


def test_resume_rejects_configuration_drift_and_retained_artifact_tampering(
    tmp_path: Path,
) -> None:
    tasks = _tasks(tmp_path, 1)
    manifest = _manifest(tasks)
    output = tmp_path / "run"
    report = run_evaluation(
        manifest=manifest,
        tasks=tasks,
        output_dir=output,
        executor=JsonFakeExecutor(manifest),
        clock=lambda: CREATED_AT,
    )
    drifted = _manifest(tasks, arm_id="repopilot-dense")

    with pytest.raises(EvaluationRunnerError) as drift:
        run_evaluation(
            manifest=drifted,
            tasks=tasks,
            output_dir=output,
            executor=JsonFakeExecutor(drifted),
            clock=lambda: CREATED_AT,
        )
    assert drift.value.code is EvaluationRunnerErrorCode.CONFIGURATION_DRIFT

    attempt = report.attempts[0]
    result = output / "executions" / attempt.attempt_id / "0000" / "result.json"
    result.write_bytes(result.read_bytes() + b" ")
    with pytest.raises(EvaluationRunnerError) as tampered:
        run_evaluation(
            manifest=manifest,
            tasks=tasks,
            output_dir=output,
            executor=JsonFakeExecutor(manifest),
            clock=lambda: CREATED_AT,
        )
    assert tampered.value.code is EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY


def test_runner_rejects_unsafe_roots_conflicts_concurrency_and_resume_overflow(
    tmp_path: Path,
) -> None:
    tasks = _tasks(tmp_path, 1)
    manifest = _manifest(tasks, max_execution_attempts=1)
    with pytest.raises(EvaluationRunnerError) as relative:
        run_evaluation(
            manifest=manifest,
            tasks=tasks,
            output_dir=Path("relative-run"),
            executor=JsonFakeExecutor(manifest),
            clock=lambda: CREATED_AT,
        )
    assert relative.value.code is EvaluationRunnerErrorCode.PATH_UNSAFE

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    occupied.joinpath("foreign.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(EvaluationRunnerError) as conflict:
        run_evaluation(
            manifest=manifest,
            tasks=tasks,
            output_dir=occupied,
            executor=JsonFakeExecutor(manifest),
            clock=lambda: CREATED_AT,
        )
    assert conflict.value.code is EvaluationRunnerErrorCode.OUTPUT_CONFLICT
    assert occupied.joinpath("foreign.txt").read_text(encoding="utf-8") == "preserve"

    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run_evaluation(
            manifest=manifest,
            tasks=tasks,
            output_dir=output,
            executor=JsonFakeExecutor(manifest, interrupt_once=True),
            clock=lambda: CREATED_AT,
        )
    store = FileEvaluationRunStore(output)
    event = store.read_events(manifest)[0]
    with pytest.raises(EvaluationRunnerError) as concurrent:
        store.append_event(event)
    assert concurrent.value.code is EvaluationRunnerErrorCode.CONCURRENT_WRITE

    with pytest.raises(EvaluationRunnerError) as resume_limit:
        run_evaluation(
            manifest=manifest,
            tasks=tasks,
            output_dir=output,
            executor=JsonFakeExecutor(manifest),
            clock=lambda: CREATED_AT,
        )
    assert resume_limit.value.code is EvaluationRunnerErrorCode.RESUME_LIMIT

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "symlink-run"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(EvaluationRunnerError) as unsafe:
        run_evaluation(
            manifest=manifest,
            tasks=tasks,
            output_dir=symlink,
            executor=JsonFakeExecutor(manifest),
            clock=lambda: CREATED_AT,
        )
    assert unsafe.value.code is EvaluationRunnerErrorCode.PATH_UNSAFE


@pytest.mark.parametrize("unexpected", [False, True])
def test_retrieval_failure_preserves_safe_ingest_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unexpected: bool
) -> None:
    task = _tasks(tmp_path, count=1)[0].task.model_copy(update={"repository_path": str(tmp_path)})
    secret = "sk-private-secret postgresql://user:password@host/db hidden prompt"

    class UnprintableError(RuntimeError):
        def __str__(self) -> str:
            raise AssertionError("exception message must never be rendered")

    def fail_ingest(*args: object, **kwargs: object) -> Any:
        if unexpected:
            raise UnprintableError(secret)
        error = RepositoryIngestError(RepositoryIngestErrorCode.LISTING_TRUNCATED)
        error.args = (secret,)
        raise error

    monkeypatch.setattr(workflow, "ingest_repository", fail_ingest)
    with pytest.raises(workflow.RepairWorkflowError) as error:
        workflow.build_repair_retrieval_context(
            ReadOnlyRepository(tmp_path), task, RepoPilotSettings()
        )
    assert error.value.code is RepairWorkflowErrorCode.RETRIEVAL_FAILURE
    assert error.value.detail.stage == "ingest"
    assert error.value.detail.cause_code == (
        "unexpected_exception" if unexpected else "listing_truncated"
    )
    assert secret not in error.value.detail.model_dump_json()


def test_workflow_detail_survives_executor_and_binds_failed_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = _tasks(tmp_path, count=1)
    manifest = _manifest(tasks)
    detail = workflow.RepairFailureDetail(stage="ingest", cause_code="listing_truncated")

    def fail_workflow(*args: object, **kwargs: object) -> Any:
        raise workflow.RepairWorkflowError(RepairWorkflowErrorCode.RETRIEVAL_FAILURE, detail=detail)

    monkeypatch.setattr(runner_module, "run_repair_request", fail_workflow)
    root = tmp_path / "run"
    report = run_evaluation(
        manifest=manifest,
        tasks=tasks,
        output_dir=root,
        executor=runner_module.RepairWorkflowEvaluationExecutor(RepoPilotSettings(), IMAGE),
        clock=lambda: CREATED_AT,
    )
    path, diagnostic = _failed_diagnostic(root, manifest)
    event = FileEvaluationRunStore(root).read_events(manifest)[-1]
    assert diagnostic.stage == detail.stage
    assert diagnostic.cause_code == detail.cause_code
    assert diagnostic.failed_event_content_sha256 == event.content_sha256
    assert diagnostic.manifest_content_sha256 == manifest.content_sha256
    assert diagnostic.execution_run_id == event.execution_run_id
    assert diagnostic.run_id == manifest.run_id
    assert diagnostic.task_id == tasks[0].task.task_id
    assert diagnostic.provider_requests_made == "unknown"
    assert diagnostic.budget_reserve_unchanged
    assert report.failed_attempts == 1
    assert report.attempts[0].execution_count == 1
    assert report.attempts[0].failure_code is EvaluationRunnerErrorCode.EXECUTION_FAILURE
    assert path.stat().st_mode & 0o777 == 0o600
    assert not (path.parent / "trace.json").exists()
    assert "diagnostic" not in event.model_dump_json()


def test_unexpected_error_never_serializes_arbitrary_message_code_or_class(tmp_path: Path) -> None:
    tasks = _tasks(tmp_path, count=1)
    manifest = _manifest(tasks)
    secret = "sk-private-key postgresql://private:secret@host/db PROMPT_PRIVATE"

    class LeakyError(RuntimeError):
        code: str

    class FailedExecutor:
        def execute(self, request: EvaluationExecutionRequest) -> EvaluationExecutionEvidence:
            error = LeakyError(secret)
            error.code = secret
            raise error

    root = tmp_path / "run"
    run_evaluation(manifest=manifest, tasks=tasks, output_dir=root, executor=FailedExecutor())
    path, diagnostic = _failed_diagnostic(root, manifest)
    assert diagnostic.stage == "executor"
    assert diagnostic.cause_code == "unexpected_exception"
    for forbidden in (secret, "LeakyError", "Traceback", "PROMPT_PRIVATE", "postgresql://"):
        assert forbidden not in path.read_text(encoding="utf-8")
    with pytest.raises(ValidationError):
        workflow.RepairFailureDetail(stage="executor", cause_code=secret)
    with pytest.raises(ValidationError):
        workflow.RepairFailureDetail(stage=secret, cause_code="unexpected_exception")  # type: ignore[arg-type]


def test_diagnostic_create_only_and_legacy_event_compatibility(tmp_path: Path) -> None:
    tasks = _tasks(tmp_path, count=1)
    manifest = _manifest(tasks)
    root = tmp_path / "run"
    run_evaluation(
        manifest=manifest,
        tasks=tasks,
        output_dir=root,
        executor=JsonFakeExecutor(manifest, fail_task_id=tasks[0].task.task_id),
    )
    path, diagnostic = _failed_diagnostic(root, manifest)
    original = path.read_bytes()
    store = FileEvaluationRunStore(root)
    failed = store.read_events(manifest)[-1]
    with pytest.raises(EvaluationRunnerError) as conflict:
        store.write_failure_diagnostic(manifest, failed, diagnostic)
    assert conflict.value.code is EvaluationRunnerErrorCode.OUTPUT_CONFLICT
    assert path.read_bytes() == original
    events_before = tuple(p.read_bytes() for p in sorted((root / "events").glob("*.json")))
    path.unlink()  # Simulate a retained historical FAILED event without the new optional sidecar.
    old_report = inspect_evaluation_run(root)
    assert old_report.failed_attempts == 1
    assert store.read_failure_diagnostic(manifest, failed) is None
    assert events_before == tuple(p.read_bytes() for p in sorted((root / "events").glob("*.json")))


@pytest.mark.parametrize("tamper", ["digest", "binding", "symlink"])
def test_existing_failure_diagnostic_is_validated(tmp_path: Path, tamper: str) -> None:
    tasks = _tasks(tmp_path, count=1)
    manifest = _manifest(tasks)
    root = tmp_path / "run"
    run_evaluation(
        manifest=manifest,
        tasks=tasks,
        output_dir=root,
        executor=JsonFakeExecutor(manifest, fail_task_id=tasks[0].task.task_id),
    )
    path, diagnostic = _failed_diagnostic(root, manifest)
    if tamper == "symlink":
        outside = tmp_path / "outside.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        payload = diagnostic.model_dump(mode="python", exclude={"content_sha256"})
        payload["task_id"] = "different-task"
        payload["content_sha256"] = (
            runner_module._canonical_sha256(payload) if tamper == "binding" else "a" * 64
        )
        path.write_text(runner_module._canonical_json(payload), encoding="utf-8")
    with pytest.raises(EvaluationRunnerError) as invalid:
        inspect_evaluation_run(root)
    assert invalid.value.code in {
        EvaluationRunnerErrorCode.ARTIFACT_INTEGRITY,
        EvaluationRunnerErrorCode.PATH_UNSAFE,
    }


@pytest.mark.parametrize("symlink", [False, True])
def test_failure_diagnostic_collision_retains_original_failed_event(
    tmp_path: Path, symlink: bool
) -> None:
    tasks = _tasks(tmp_path, count=1)
    manifest = _manifest(tasks)
    outside = tmp_path / "outside"
    outside.write_text("untouched", encoding="utf-8")

    class CollisionExecutor:
        def execute(self, request: EvaluationExecutionRequest) -> EvaluationExecutionEvidence:
            target = request.output_dir / runner_module.EVALUATION_FAILURE_DIAGNOSTIC_FILENAME
            if symlink:
                target.symlink_to(outside)
            else:
                target.write_text("untouched", encoding="utf-8")
            raise RuntimeError("not serialized")

    root = tmp_path / "run"
    with pytest.raises(EvaluationRunnerError) as conflict:
        run_evaluation(
            manifest=manifest,
            tasks=tasks,
            output_dir=root,
            executor=CollisionExecutor(),
        )
    assert conflict.value.code is EvaluationRunnerErrorCode.OUTPUT_CONFLICT
    events = tuple(
        runner_module.EvaluationAttemptEvent.model_validate_json(path.read_bytes())
        for path in sorted((root / "events").glob("*.json"))
    )
    assert events[-1].status is EvaluationAttemptStatus.FAILED
    assert events[-1].failure_code is EvaluationRunnerErrorCode.EXECUTION_FAILURE
    assert outside.read_text(encoding="utf-8") == "untouched"


def test_artifact_integrity_failure_has_safe_evidence_stage(tmp_path: Path) -> None:
    tasks = _tasks(tmp_path, count=1)
    manifest = _manifest(tasks)
    root = tmp_path / "run"
    run_evaluation(
        manifest=manifest,
        tasks=tasks,
        output_dir=root,
        executor=JsonFakeExecutor(manifest, corrupt_reference=True),
    )
    _, diagnostic = _failed_diagnostic(root, manifest)
    assert diagnostic.stage == "evidence_verification"
    assert diagnostic.cause_code == "artifact_integrity"


def test_diagnostic_refuses_a_symlinked_execution_parent(tmp_path: Path) -> None:
    tasks = _tasks(tmp_path, count=1)
    manifest = _manifest(tasks)
    root = tmp_path / "run"
    run_evaluation(
        manifest=manifest,
        tasks=tasks,
        output_dir=root,
        executor=JsonFakeExecutor(manifest, fail_task_id=tasks[0].task.task_id),
    )
    path, diagnostic = _failed_diagnostic(root, manifest)
    store = FileEvaluationRunStore(root)
    event = store.read_events(manifest)[-1]
    original_bytes = path.read_bytes()
    original_parent = path.parent
    saved = tmp_path / "saved-execution"
    original_parent.rename(saved)
    original_parent.symlink_to(saved, target_is_directory=True)
    with pytest.raises(EvaluationRunnerError) as invalid:
        store.write_failure_diagnostic(manifest, event, diagnostic)
    assert invalid.value.code is EvaluationRunnerErrorCode.PATH_UNSAFE
    assert saved.joinpath(path.name).read_bytes() == original_bytes
    with pytest.raises(EvaluationRunnerError) as read_invalid:
        inspect_evaluation_run(root)
    assert read_invalid.value.code is EvaluationRunnerErrorCode.PATH_UNSAFE
