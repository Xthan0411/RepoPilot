"""Real-Docker proof for baseline-fail, patch, test-pass, verify success."""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from repopilot.config import RepoPilotSettings
from repopilot.enums import RunStatus, SandboxNetworkMode, TerminationReason
from repopilot.evals import load_repair_fixture_manifest
from repopilot.sandbox import (
    DockerCliClient,
    DockerSandbox,
    RunRequest,
    SandboxPolicy,
)
from repopilot.schemas import (
    LocalizationMetrics,
    ToolCall,
    UsageMetrics,
)
from repopilot.tools import (
    PatchToolSession,
    build_reproduction_result,
    interpret_test_observation,
)
from repopilot.verifier import (
    FinalResultContext,
    RuleBasedVerifier,
    VerificationEvidence,
    build_final_result,
)

pytestmark = pytest.mark.integration

FIXTURE_MANIFEST = Path(__file__).parents[1] / "fixtures" / "p5_repair_tasks" / "manifest.json"
REQUESTED_AT = datetime(2026, 8, 6, 9, tzinfo=UTC)


def test_real_docker_evidence_authorizes_both_fixture_repairs(
    tmp_path: Path,
) -> None:
    image = os.environ.get("REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE")
    if image is None:
        pytest.skip("set REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE to a local Python image digest")

    manifest = load_repair_fixture_manifest(FIXTURE_MANIFEST)
    settings = RepoPilotSettings(
        sandbox_network=SandboxNetworkMode.NONE,
        sandbox_cpu_limit=0.25,
        sandbox_memory_limit_mb=128,
        sandbox_pids_limit=32,
        sandbox_tmpfs_limit_mb=16,
        sandbox_command_timeout_seconds=30,
        sandbox_max_output_bytes=16_384,
        sandbox_max_artifact_bytes=65_536,
        patch_max_bytes=65_536,
    )
    client = DockerCliClient()
    verifier = RuleBasedVerifier()

    for case in manifest.cases:
        task = case.evaluation_task.task
        artifact_root = tmp_path / f"{task.task_id}-artifacts"
        sandbox = DockerSandbox(
            policy=SandboxPolicy.from_settings(settings),
            image_reference=image,
            artifact_root=artifact_root,
            client=client,
        )
        handle = sandbox.prepare(task)
        session = PatchToolSession.from_settings(
            sandbox=sandbox,
            handle=handle,
            task=task,
            settings=settings,
        )

        try:
            baseline_observation = sandbox.run(
                handle,
                RunRequest(call_id=uuid4(), command=task.test_commands[0]),
            )
            baseline_test_run = interpret_test_observation(
                baseline_observation,
                task.test_commands[0],
                command_index=0,
            )
            reproduction = build_reproduction_result(baseline_test_run)

            patch_call = ToolCall(
                call_id=uuid4(),
                tool_name="apply_patch",
                validated_arguments={"diff": case.evaluation_task.expected_patch},
                requested_at=REQUESTED_AT,
            )
            patch_result = session.apply(
                case.evaluation_task.expected_patch,
                call=patch_call,
            )

            validation_observation = sandbox.run(
                handle,
                RunRequest(call_id=uuid4(), command=task.test_commands[0]),
            )
            validation_test_run = interpret_test_observation(
                validation_observation,
                task.test_commands[0],
                command_index=0,
            )
            decision = verifier.verify(
                VerificationEvidence(
                    reproduction_result=reproduction,
                    patch_state=patch_result.patch_state,
                    validation_test_run=validation_test_run,
                    tool_steps=3,
                )
            )
            final_result = build_final_result(
                decision,
                FinalResultContext(
                    task_id=task.task_id,
                    run_id=uuid4(),
                    localization_metrics=LocalizationMetrics(
                        file_recall_at_1=0,
                        file_recall_at_3=0,
                        file_recall_at_5=0,
                        mean_reciprocal_rank=0,
                        indexing_latency_ms=0,
                        retrieval_latency_ms=0,
                    ),
                    usage=UsageMetrics(
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                        estimated_cost_usd=Decimal("0"),
                        latency_ms=0,
                    ),
                    trace_path=f"runs/{task.task_id}/trace.json",
                ),
            )

            assert decision.status is RunStatus.SUCCESS
            assert decision.termination_reason is TerminationReason.SUCCESS
            assert final_result.status is RunStatus.SUCCESS
            assert final_result.task_metrics.issue_reproduced is True
            assert final_result.task_metrics.patch_applied is True
            assert final_result.task_metrics.tests_passed is True
            assert final_result.task_metrics.task_success is True
            assert final_result.reproduction_result.test_run == baseline_test_run
            assert final_result.validation_result.test_run == validation_test_run
            assert patch_result.patch_state.diff_ref is not None
            assert final_result.patch_path == patch_result.patch_state.diff_ref.relative_path
            assert "expected_patch" not in final_result.model_dump_json()
        finally:
            sandbox.destroy(handle)

        artifacts = sandbox.collect_artifacts(handle)
        assert len(artifacts) == 7
        artifact_directory = artifact_root / handle.sandbox_id
        for artifact in artifacts:
            content = (artifact_directory / Path(artifact.relative_path).name).read_bytes()
            assert len(content) == artifact.size_bytes
            assert hashlib.sha256(content).hexdigest() == artifact.sha256
