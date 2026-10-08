"""Real-Docker gate for the complete localization-to-verified-repair workflow."""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast
from uuid import UUID, uuid5

import pytest
from pydantic import JsonValue
from typer.testing import CliRunner

from repopilot.agent import repair_decision_request_id
from repopilot.cli import app
from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    AgentFailureKind,
    ContextBudgetKind,
    ContextSourceKind,
    ContextTrustLevel,
    DatasetSplit,
    MemoryAccessMode,
    PatchStatus,
    ProviderErrorCode,
    RepairWorkflowErrorCode,
    RepositoryMemoryLookupStatus,
    ReproductionStatus,
    RunStatus,
    SandboxLifecycle,
    SandboxNetworkMode,
    TerminationReason,
    ValidationStatus,
    VerifiedEpisodeLookupStatus,
)
from repopilot.enums import TestStatus as RepoTestStatus
from repopilot.evals import (
    BaselineEvaluationExecutor,
    RepairFixtureCase,
    RepairWorkflowEvaluationExecutor,
    baseline_configuration_snapshot,
    build_evaluation_run_manifest,
    inspect_evaluation_run,
    load_baseline_definition,
    load_repair_fixture_manifest,
    require_comparable_baselines,
    run_evaluation,
)
from repopilot.memory import FileMemoryNamespaceManager
from repopilot.providers import (
    FinishAction,
    ProviderResponse,
    ProviderUsageDelta,
    ToolAction,
)
from repopilot.repair_workflow import (
    RepairArtifactManifest,
    RepairRunRequest,
    RepairSandbox,
    RepairWorkflowError,
    ScriptedProviderManifest,
    run_repair_request,
)
from repopilot.replay import replay_trace
from repopilot.sandbox import (
    DockerCliClient,
    DockerClientError,
    DockerSandbox,
    SandboxPolicy,
)
from repopilot.schemas import EvaluationTaskSpec, FinalResult, MemoryNamespace, TaskSpec, ToolCall
from repopilot.trace import AgentTrace

pytestmark = pytest.mark.integration

FIXTURE_MANIFEST = Path(__file__).parents[1] / "fixtures" / "p5_repair_tasks" / "manifest.json"
CONFIG_PATH = Path(__file__).parents[2] / "configs" / "default.yaml"
RUN_NAMESPACE = UUID("17275b3e-f6d4-46e8-b093-38776a58a4b5")
STARTED_AT = datetime(2026, 8, 6, 16, tzinfo=UTC)


def _image() -> str:
    configured = os.environ.get("REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE")
    if configured is None:
        pytest.skip("set REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE to a local Python image digest")
    return configured


def _settings() -> RepoPilotSettings:
    return RepoPilotSettings(
        sandbox_network=SandboxNetworkMode.NONE,
        sandbox_cpu_limit=0.25,
        sandbox_memory_limit_mb=128,
        sandbox_pids_limit=32,
        sandbox_tmpfs_limit_mb=16,
        sandbox_command_timeout_seconds=30,
        sandbox_max_output_bytes=16_384,
        sandbox_max_artifact_bytes=65_536,
        patch_max_bytes=65_536,
        max_steps=20,
        max_tokens=20_000,
        timeout_seconds=120,
    )


def _dev_memory_namespace(snapshot_id: str) -> MemoryNamespace:
    return MemoryNamespace(
        experiment_id="repair-e2e",
        snapshot_id=snapshot_id,
        split=DatasetSplit.DEV,
        access_mode=MemoryAccessMode.READ_WRITE,
    )


def _usage() -> ProviderUsageDelta:
    return ProviderUsageDelta(
        input_tokens=8,
        output_tokens=2,
        total_tokens=10,
        estimated_cost_usd=Decimal("0"),
        latency_ms=1,
    )


def _tool_response(
    run_id: UUID,
    decision: int,
    tool_name: str,
    arguments: dict[str, JsonValue],
    considered_tools: tuple[str, ...],
) -> ProviderResponse:
    return ProviderResponse(
        request_id=repair_decision_request_id(run_id, decision),
        action=ToolAction(
            tool_call=ToolCall(
                call_id=uuid5(run_id, f"tool:{decision}:{tool_name}"),
                tool_name=tool_name,
                validated_arguments=arguments,
                requested_at=STARTED_AT,
            ),
            considered_tools=considered_tools,
            selection_reason=f"Use {tool_name} for the current evidence step.",
        ),
        usage=_usage(),
        model="fake-repair-model-v1",
        prompt_version="repair-e2e-v1",
    )


def _success_responses(
    case: RepairFixtureCase,
    run_id: UUID,
) -> tuple[ProviderResponse, ...]:
    task_id = case.evaluation_task.task.task_id
    if task_id == "p5-config-parser-blank":
        query = "parse_config"
        path = "config_parser/parser.py"
    elif task_id == "p5-parser-service-async":
        query = "parse_async"
        path = "parser_service/service.py"
    else:
        raise AssertionError(f"unexpected repair fixture: {task_id}")
    responses = (
        _tool_response(run_id, 0, "run_tests", {"test_index": 0}, ("run_tests",)),
        _tool_response(
            run_id,
            1,
            "search_code",
            {
                "query": query,
                "path": ".",
                "case_sensitive": True,
                "max_matches": 20,
            },
            ("search_code",),
        ),
        _tool_response(
            run_id,
            2,
            "read_file",
            {"path": path, "start_line": 1, "end_line": None},
            ("inspect_symbol", "read_file"),
        ),
        _tool_response(
            run_id,
            3,
            "apply_patch",
            {"diff": case.evaluation_task.expected_patch},
            ("apply_patch",),
        ),
        _tool_response(run_id, 4, "run_tests", {"test_index": 0}, ("run_tests",)),
        ProviderResponse(
            request_id=repair_decision_request_id(run_id, 5),
            action=FinishAction(
                summary="Collected evidence is ready for deterministic verification."
            ),
            usage=_usage(),
            model="fake-repair-model-v1",
            prompt_version="repair-e2e-v1",
        ),
    )
    return responses


def _baseline_success_responses(
    task: TaskSpec,
    run_id: UUID,
) -> tuple[ProviderResponse, ...]:
    if task.task_id != "p5-config-parser-blank":
        raise AssertionError(f"unexpected baseline fixture: {task.task_id}")
    independently_authored_patch = (
        "diff --git a/config_parser/parser.py b/config_parser/parser.py\n"
        "--- a/config_parser/parser.py\n"
        "+++ b/config_parser/parser.py\n"
        "@@ -5,7 +5,7 @@ def parse_config(raw: str) -> dict[str, str]:\n"
        '     """Return stripped key/value pairs."""\n'
        " \n"
        "     result: dict[str, str] = {}\n"
        "-    for line in raw.splitlines():\n"
        "+    for line in filter(str.strip, raw.splitlines()):\n"
        '         key, value = line.split("=", maxsplit=1)\n'
        "         result[key.strip()] = value.strip()\n"
        "     return result\n"
    )
    responses = (
        _tool_response(run_id, 0, "run_tests", {"test_index": 0}, ("run_tests",)),
        _tool_response(
            run_id,
            1,
            "search_code",
            {
                "query": "parse_config",
                "path": ".",
                "case_sensitive": True,
                "max_matches": 20,
            },
            ("search_code",),
        ),
        _tool_response(
            run_id,
            2,
            "read_file",
            {
                "path": "config_parser/parser.py",
                "start_line": 1,
                "end_line": None,
            },
            ("inspect_symbol", "read_file"),
        ),
        _tool_response(
            run_id,
            3,
            "apply_patch",
            {"diff": independently_authored_patch},
            ("apply_patch",),
        ),
        _tool_response(run_id, 4, "run_tests", {"test_index": 0}, ("run_tests",)),
        ProviderResponse(
            request_id=repair_decision_request_id(run_id, 5),
            action=FinishAction(
                summary="Collected baseline evidence is ready for deterministic verification."
            ),
            usage=_usage(),
            model="fake-repair-model-v1",
            prompt_version="baseline-smoke-v1",
        ),
    )
    return tuple(
        response.model_copy(update={"prompt_version": "baseline-smoke-v1"})
        for response in responses
    )


def _sandbox_factory(
    settings: RepoPilotSettings,
    image: str,
    created: list[DockerSandbox],
    client: DockerCliClient,
) -> Callable[[Path], RepairSandbox]:
    def build(root: Path) -> RepairSandbox:
        sandbox = DockerSandbox(
            policy=SandboxPolicy.from_settings(settings),
            image_reference=image,
            artifact_root=root / "artifacts",
            client=client,
        )
        created.append(sandbox)
        return cast(RepairSandbox, sandbox)

    return build


def _assert_artifacts(output: Path, manifest: RepairArtifactManifest) -> None:
    assert manifest.artifacts
    assert manifest.artifact_root == "artifacts"
    for artifact in manifest.artifacts:
        path = output / artifact.relative_path
        content = path.read_bytes()
        assert len(content) == artifact.size_bytes
        assert hashlib.sha256(content).hexdigest() == artifact.sha256


def _assert_destroyed(sandbox: DockerSandbox, client: DockerCliClient) -> None:
    assert sandbox.lifecycle is SandboxLifecycle.DESTROYED
    assert sandbox.preparation is not None
    with pytest.raises(DockerClientError):
        client.inspect_container(sandbox.preparation.container_id)


def _managed_container_ids() -> bytes:
    completed = subprocess.run(
        (
            "docker",
            "ps",
            "-aq",
            "--filter",
            "label=com.repopilot.managed=true",
        ),
        check=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def test_repair_cli_runs_one_complete_real_docker_task(tmp_path: Path) -> None:
    case = load_repair_fixture_manifest(FIXTURE_MANIFEST).cases[0]
    task = case.evaluation_task.task
    run_id = uuid5(RUN_NAMESPACE, f"{task.task_id}:cli")
    task_path = tmp_path / "task.json"
    provider_path = tmp_path / "provider.json"
    output = tmp_path / "cli-output"
    task_path.write_text(task.model_dump_json(indent=2), encoding="utf-8")
    provider_path.write_text(
        ScriptedProviderManifest(
            thread_id="repair-e2e-cli",
            run_id=run_id,
            started_at=STARTED_AT,
            responses=_success_responses(case, run_id),
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    invoked = CliRunner().invoke(
        app,
        [
            "repair",
            "--task",
            str(task_path),
            "--sandbox-image",
            _image(),
            "--output-dir",
            str(output),
            "--provider-script",
            str(provider_path),
            "--config",
            str(CONFIG_PATH),
        ],
    )

    assert invoked.exit_code == 0
    assert FinalResult.model_validate_json(invoked.stdout).status is RunStatus.SUCCESS
    trace = AgentTrace.model_validate_json(output.joinpath("trace.json").read_bytes())
    assert replay_trace(trace).final_state == trace.final_state
    artifact_manifest = RepairArtifactManifest.model_validate_json(
        output.joinpath("artifacts.json").read_bytes()
    )
    _assert_artifacts(output, artifact_manifest)
    assert _managed_container_ids() == b""


def test_eval_runner_real_docker_fake_smoke_is_idempotent(
    tmp_path: Path,
) -> None:
    case = load_repair_fixture_manifest(FIXTURE_MANIFEST).cases[0]
    settings = _settings()
    client = DockerCliClient()
    memory_stores = FileMemoryNamespaceManager(tmp_path / "memory").open(
        _dev_memory_namespace("eval-runner-smoke-v1"),
        created_at=STARTED_AT,
    )
    manifest = build_evaluation_run_manifest(
        experiment_id="eval-runner-real-docker",
        run_id=uuid5(RUN_NAMESPACE, "eval-runner-real-docker"),
        created_at=STARTED_AT,
        source_commit="e" * 40,
        dependency_lock_sha256=hashlib.sha256(
            Path(__file__).parents[2].joinpath("uv.lock").read_bytes()
        ).hexdigest(),
        dataset_manifest_sha256=hashlib.sha256(FIXTURE_MANIFEST.read_bytes()).hexdigest(),
        configuration=settings.redacted_snapshot(),
        split=DatasetSplit.DEV,
        arm_id="repopilot-bm25",
        tasks=(case.evaluation_task,),
        memory_namespace=memory_stores.namespace,
        model_provider=settings.model_provider,
        model=settings.model,
        prompt_versions=("repair-e2e-v1",),
        sandbox_image_reference=_image(),
    )
    output = tmp_path / "eval-run"
    created: list[DockerSandbox] = []
    executor = RepairWorkflowEvaluationExecutor(
        settings=settings,
        sandbox_image_reference=_image(),
        provider_responses_factory=(lambda _task, run_id: _success_responses(case, run_id)),
        sandbox_factory=_sandbox_factory(settings, _image(), created, client),
        memory_stores=memory_stores,
    )

    first = run_evaluation(
        manifest=manifest,
        tasks=(case.evaluation_task,),
        output_dir=output,
        executor=executor,
        clock=lambda: STARTED_AT,
    )
    duplicate_created: list[DockerSandbox] = []
    second = run_evaluation(
        manifest=manifest,
        tasks=(case.evaluation_task,),
        output_dir=output,
        executor=RepairWorkflowEvaluationExecutor(
            settings=settings,
            sandbox_image_reference=_image(),
            provider_responses_factory=(lambda _task, run_id: _success_responses(case, run_id)),
            sandbox_factory=_sandbox_factory(
                settings,
                _image(),
                duplicate_created,
                client,
            ),
            memory_stores=memory_stores,
        ),
        clock=lambda: STARTED_AT,
    )

    assert first == second == inspect_evaluation_run(output)
    assert first.completed_attempts == 1
    assert first.failed_attempts == 0
    assert first.attempts[0].evidence is not None
    assert first.attempts[0].evidence.status is RunStatus.SUCCESS
    assert first.attempts[0].evidence.task_success is True
    assert len(created) == 1
    _assert_destroyed(created[0], client)
    assert duplicate_created == []
    run_manifest_text = output.joinpath("manifest.json").read_text(encoding="utf-8")
    assert case.evaluation_task.expected_patch not in run_manifest_text
    assert str(case.evaluation_task.task.repository_path) not in run_manifest_text


def test_baseline_adapters_share_fairness_and_keep_evaluator_gold_out_of_docker(
    tmp_path: Path,
) -> None:
    case = load_repair_fixture_manifest(FIXTURE_MANIFEST).cases[0]
    evaluator_gold_canary = "EVALUATOR_GOLD_PATCH_CANARY_MUST_NOT_ENTER_BASELINE"
    evaluator_gold_file = "evaluator/private_gold_file.py"
    evaluation_task = EvaluationTaskSpec(
        task=case.evaluation_task.task,
        expected_patch=evaluator_gold_canary,
        gold_files=(evaluator_gold_file,),
        split=DatasetSplit.DEV,
    )
    settings = _settings()
    image = _image()
    definitions = (
        load_baseline_definition(
            Path(__file__).parents[2] / "configs" / "baselines" / "mini-swe-safe-v1.yaml"
        ),
        load_baseline_definition(
            Path(__file__).parents[2] / "configs" / "baselines" / "no-semantic-rag-v1.yaml"
        ),
    )
    executors = tuple(
        BaselineEvaluationExecutor(
            definition=definition,
            settings=settings,
            sandbox_image_reference=image,
            provider_responses_factory=_baseline_success_responses,
            sandbox_factory=_sandbox_factory(settings, image, [], DockerCliClient()),
        )
        for definition in definitions
    )
    assert require_comparable_baselines(tuple(executor.runtime_contract for executor in executors))

    observed_results: list[FinalResult] = []
    for index, executor in enumerate(executors):
        contract = executor.runtime_contract
        manifest = build_evaluation_run_manifest(
            experiment_id="baseline-adapter-real-docker",
            run_id=uuid5(RUN_NAMESPACE, f"baseline-adapter:{contract.baseline_id}"),
            created_at=STARTED_AT,
            source_commit="e" * 40,
            dependency_lock_sha256=hashlib.sha256(
                Path(__file__).parents[2].joinpath("uv.lock").read_bytes()
            ).hexdigest(),
            dataset_manifest_sha256=hashlib.sha256(FIXTURE_MANIFEST.read_bytes()).hexdigest(),
            configuration=baseline_configuration_snapshot(contract),
            split=DatasetSplit.DEV,
            arm_id=contract.baseline_id,
            tasks=(evaluation_task,),
            memory_namespace=None,
            model_provider=settings.model_provider,
            model=settings.model,
            prompt_versions=("baseline-smoke-v1",),
            sandbox_image_reference=image,
        )
        output = tmp_path / f"baseline-{index}"

        report = run_evaluation(
            manifest=manifest,
            tasks=(evaluation_task,),
            output_dir=output,
            executor=executor,
            clock=lambda: STARTED_AT,
        )

        assert report.completed_attempts == 1
        assert report.failed_attempts == 0
        evidence = report.attempts[0].evidence
        assert evidence is not None
        assert evidence.status is RunStatus.SUCCESS
        assert evidence.task_success is True
        execution_root = output / "executions" / report.attempts[0].attempt_id / "0000"
        trace = AgentTrace.model_validate_json(execution_root.joinpath("trace.json").read_bytes())
        artifact_manifest = RepairArtifactManifest.model_validate_json(
            execution_root.joinpath("artifacts.json").read_bytes()
        )
        result = FinalResult.model_validate_json(
            execution_root.joinpath("result.json").read_bytes()
        )
        observed_results.append(result)
        assert artifact_manifest.context_mode is contract.context_mode
        assert artifact_manifest.evaluation_configuration_sha256 == (manifest.configuration_sha256)
        assert artifact_manifest.settings_sha256 != manifest.configuration_sha256
        assert trace.final_state.memory_namespace is None
        assert trace.final_state.memory_load_audit is None
        assert trace.final_state.repository_memory == ()
        assert trace.final_state.episodic_memory == ()
        if contract.context_mode.value == "none":
            assert trace.final_state.retrieved_context == ()
        else:
            assert trace.final_state.retrieved_context
        retained_bytes = b"".join(path.read_bytes() for path in output.rglob("*") if path.is_file())
        assert evaluator_gold_canary.encode() not in retained_bytes
        assert evaluator_gold_file.encode() not in retained_bytes
        assert case.evaluation_task.expected_patch.encode() not in retained_bytes

    assert observed_results[0].usage == observed_results[1].usage
    assert observed_results[0].task_metrics.tool_steps == (
        observed_results[1].task_metrics.tool_steps
    )
    assert _managed_container_ids() == b""


def test_real_docker_completes_both_verified_repairs_and_replays(
    tmp_path: Path,
) -> None:
    manifest = load_repair_fixture_manifest(FIXTURE_MANIFEST)
    settings = _settings()
    client = DockerCliClient()
    memory_stores = FileMemoryNamespaceManager(tmp_path / "memory").open(
        _dev_memory_namespace("verified-repairs-v1"),
        created_at=STARTED_AT,
    )

    for case in manifest.cases:
        task = case.evaluation_task.task
        run_id = uuid5(RUN_NAMESPACE, task.task_id)
        request = RepairRunRequest(
            thread_id=f"repair-e2e-{task.task_id}",
            run_id=run_id,
            started_at=STARTED_AT,
            task=task,
            sandbox_image_reference=_image(),
            provider_responses=_success_responses(case, run_id),
        )
        output = tmp_path / task.task_id
        created: list[DockerSandbox] = []

        artifacts = run_repair_request(
            request,
            output_dir=output,
            settings=settings,
            sandbox_factory=_sandbox_factory(settings, _image(), created, client),
            memory_stores=memory_stores,
        )

        assert artifacts.result.status is RunStatus.SUCCESS
        assert artifacts.result.termination_reason is TerminationReason.SUCCESS
        assert artifacts.result.reproduction_result.status is ReproductionStatus.REPRODUCED
        assert artifacts.result.validation_result.status is ValidationStatus.PASSED
        assert artifacts.result.task_metrics.task_success is True
        assert artifacts.verified_episode is not None
        episode = artifacts.verified_episode
        episode_lookup = memory_stores.episodic.lookup(
            repository_id=episode.scope.repository_id,
            repository_commit=episode.scope.repository_commit,
            repository_content_sha256=episode.scope.repository_content_sha256,
            as_of=STARTED_AT,
        )
        assert episode_lookup.status is VerifiedEpisodeLookupStatus.CURRENT
        assert episode_lookup.records == (episode,)
        assert episode.source_task_id == task.task_id
        assert episode.evidence.patch_artifact.relative_path == artifacts.result.patch_path
        assert episode.provenance.model_versions == ("fake-repair-model-v1",)
        assert episode.provenance.prompt_versions == ("repair-e2e-v1",)
        assert artifacts.final_state.baseline_test_result is not None
        assert artifacts.final_state.baseline_test_result.status is RepoTestStatus.FAILED
        assert artifacts.final_state.test_state.status is RepoTestStatus.PASSED
        assert artifacts.final_state.test_state.commands_run == 2
        assert artifacts.final_state.patch_state.status is PatchStatus.APPLIED
        assert artifacts.final_state.patch_state.files_changed == case.evaluation_task.gold_files
        assert artifacts.final_state.memory_namespace == memory_stores.namespace
        assert artifacts.final_state.memory_load_audit is not None
        assert (
            artifacts.final_state.memory_load_audit.repository_status
            is RepositoryMemoryLookupStatus.CURRENT
        )
        assert artifacts.final_state.memory_load_audit.episodic_status in {
            VerifiedEpisodeLookupStatus.EMPTY,
            VerifiedEpisodeLookupStatus.CURRENT,
        }
        assert any(
            case.evaluation_task.gold_files[0] in hit.retrieval_reason
            for hit in artifacts.final_state.retrieved_context
        )
        assert replay_trace(artifacts.trace).final_state == artifacts.final_state
        persisted_result = FinalResult.model_validate_json(artifacts.result_path.read_bytes())
        assert persisted_result == artifacts.result
        assert AgentTrace.model_validate_json(artifacts.trace_path.read_bytes()) == artifacts.trace
        persisted_manifest = RepairArtifactManifest.model_validate_json(
            artifacts.artifact_manifest_path.read_bytes()
        )
        assert persisted_manifest == artifacts.artifact_manifest
        assert persisted_manifest.prompt_versions == ("repair-e2e-v1",)
        assert persisted_manifest.sandbox_image_reference == _image()
        assert len(persisted_manifest.settings_sha256) == 64
        _assert_artifacts(output, persisted_manifest)
        assert len(created) == 1
        _assert_destroyed(created[0], client)

        trace_text = artifacts.trace_path.read_text(encoding="utf-8")
        assert '"expected_patch"' not in trace_text
        assert '"gold_files"' not in trace_text

    reused_case = manifest.cases[0]
    reused_task = reused_case.evaluation_task.task
    reused_run_id = uuid5(RUN_NAMESPACE, f"{reused_task.task_id}:memory-reuse")
    reused_request = RepairRunRequest(
        thread_id=f"repair-e2e-{reused_task.task_id}-memory-reuse",
        run_id=reused_run_id,
        started_at=STARTED_AT,
        task=reused_task,
        sandbox_image_reference=_image(),
        provider_responses=_success_responses(reused_case, reused_run_id),
    )
    reused_created: list[DockerSandbox] = []
    reused_artifacts = run_repair_request(
        reused_request,
        output_dir=tmp_path / f"{reused_task.task_id}-memory-reuse",
        settings=settings,
        sandbox_factory=_sandbox_factory(settings, _image(), reused_created, client),
        memory_stores=memory_stores,
    )

    assert len(reused_artifacts.final_state.episodic_memory) == 1
    assert reused_artifacts.final_state.memory_load_audit is not None
    assert (
        reused_artifacts.final_state.memory_load_audit.episodic_status
        is VerifiedEpisodeLookupStatus.CURRENT
    )
    assert reused_artifacts.final_state.memory_load_audit.episodic_records_loaded == 1
    first_request = next(
        event.provider_request
        for event in reused_artifacts.trace.events
        if event.provider_request is not None
    )
    assert first_request.context_window is not None
    historical_item = next(
        item
        for item in first_request.context_window.items
        if item.source_kind is ContextSourceKind.EPISODIC_MEMORY
    )
    memory_usage = next(
        usage
        for usage in first_request.context_window.budget_usages
        if usage.budget_kind is ContextBudgetKind.MEMORY
    )
    assert historical_item.trust_level is ContextTrustLevel.HISTORICAL_EVIDENCE
    assert historical_item.item_id in first_request.context_window.included_item_ids
    assert memory_usage.selected_tokens >= historical_item.estimated_tokens
    assert replay_trace(reused_artifacts.trace).final_state == reused_artifacts.final_state
    assert len(reused_created) == 1
    _assert_destroyed(reused_created[0], client)


def test_real_docker_provider_failure_is_persisted_replayable_and_cleaned(
    tmp_path: Path,
) -> None:
    case = load_repair_fixture_manifest(FIXTURE_MANIFEST).cases[0]
    task = case.evaluation_task.task
    run_id = uuid5(RUN_NAMESPACE, f"{task.task_id}:provider-failure")
    baseline_only = _success_responses(case, run_id)[:1]
    request = RepairRunRequest(
        thread_id="repair-e2e-provider-failure",
        run_id=run_id,
        started_at=STARTED_AT,
        task=task,
        sandbox_image_reference=_image(),
        provider_responses=baseline_only,
    )
    settings = _settings()
    client = DockerCliClient()
    created: list[DockerSandbox] = []
    output = tmp_path / "provider-failure"
    memory_stores = FileMemoryNamespaceManager(tmp_path / "memory").open(
        _dev_memory_namespace("provider-failure-v1"),
        created_at=STARTED_AT,
    )

    artifacts = run_repair_request(
        request,
        output_dir=output,
        settings=settings,
        sandbox_factory=_sandbox_factory(settings, _image(), created, client),
        memory_stores=memory_stores,
    )

    assert artifacts.result.status is RunStatus.STOPPED
    assert artifacts.result.termination_reason is TerminationReason.PROVIDER_ERROR
    assert artifacts.result.task_metrics.task_success is False
    assert artifacts.verified_episode is None
    repository_memory = artifacts.final_state.repository_memory
    assert repository_memory
    assert (
        memory_stores.episodic.lookup(
            repository_id=repository_memory[0].repository_id,
            repository_commit=task.base_commit,
            repository_content_sha256=repository_memory[0].repository_content_sha256,
            as_of=STARTED_AT,
        ).status
        is VerifiedEpisodeLookupStatus.EMPTY
    )
    assert artifacts.final_state.failure_history[-1].kind is AgentFailureKind.PROVIDER_FAILURE
    assert (
        artifacts.final_state.failure_history[-1].provider_error_code
        is ProviderErrorCode.SCRIPT_EXHAUSTED
    )
    assert replay_trace(artifacts.trace).final_state == artifacts.final_state
    assert artifacts.result_path.is_file()
    assert artifacts.trace_path.is_file()
    assert artifacts.artifact_manifest_path.is_file()
    _assert_artifacts(output, artifacts.artifact_manifest)
    assert len(created) == 1
    _assert_destroyed(created[0], client)


def test_verified_episode_store_failure_is_a_typed_workflow_failure(
    tmp_path: Path,
) -> None:
    case = load_repair_fixture_manifest(FIXTURE_MANIFEST).cases[0]
    task = case.evaluation_task.task
    run_id = uuid5(RUN_NAMESPACE, f"{task.task_id}:episode-store-failure")
    request = RepairRunRequest(
        thread_id="repair-e2e-episode-store-failure",
        run_id=run_id,
        started_at=STARTED_AT,
        task=task,
        sandbox_image_reference=_image(),
        provider_responses=_success_responses(case, run_id),
    )
    settings = _settings()
    client = DockerCliClient()
    created: list[DockerSandbox] = []
    output = tmp_path / "episode-store-failure"
    memory_stores = FileMemoryNamespaceManager(tmp_path / "memory").open(
        _dev_memory_namespace("undersized-store-v1"),
        created_at=STARTED_AT,
        episodic_max_record_bytes=1_024,
    )

    with pytest.raises(RepairWorkflowError) as captured:
        run_repair_request(
            request,
            output_dir=output,
            settings=settings,
            sandbox_factory=_sandbox_factory(settings, _image(), created, client),
            memory_stores=memory_stores,
        )

    assert captured.value.code is RepairWorkflowErrorCode.MEMORY_FAILURE
    assert output.joinpath("result.json").is_file()
    assert output.joinpath("trace.json").is_file()
    assert output.joinpath("artifacts.json").is_file()
    assert len(created) == 1
    _assert_destroyed(created[0], client)
