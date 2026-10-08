"""Host-free provenance leakage and tamper proofs for the evaluation runner."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from repopilot.enums import DatasetSplit, EvaluationRunnerErrorCode, ModelProvider
from repopilot.evals import (
    EvaluationRunnerError,
    FileEvaluationRunStore,
    build_evaluation_run_manifest,
    load_evaluation_run_manifest,
)
from repopilot.schemas import EvaluationTaskSpec

pytestmark = pytest.mark.security

FIXTURE = Path(__file__).parents[1] / "fixtures" / "valid_evaluation_task.json"
GOLD_CANARY = "EVALUATOR_GOLD_MUST_NOT_ENTER_RUN_MANIFEST"


def _task(tmp_path: Path) -> EvaluationTaskSpec:
    payload = cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))
    task = cast(dict[str, Any], payload["task"])
    task["task_id"] = "eval-security-task"
    task["repository_path"] = str(tmp_path / "private-host-repository")
    payload["expected_patch"] = GOLD_CANARY
    return EvaluationTaskSpec.model_validate(payload)


def test_run_manifest_hides_gold_and_host_path_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    evaluation_task = _task(tmp_path)
    manifest = build_evaluation_run_manifest(
        experiment_id="eval-security",
        run_id=UUID("786b94dc-e286-4e3f-bfe2-5fc29732079e"),
        created_at=datetime(2026, 8, 7, 20, tzinfo=UTC),
        source_commit="a" * 40,
        dependency_lock_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
        configuration={
            "profile": "eval",
            "model": "fake-deterministic",
            "database_url": "<redacted>",
            "openai_api_key": None,
        },
        split=DatasetSplit.DEV,
        arm_id="security-smoke",
        tasks=(evaluation_task,),
        memory_namespace=None,
        model_provider=ModelProvider.FAKE,
        model="fake-deterministic",
        prompt_versions=("eval-prompt-v1",),
        sandbox_image_reference=f"python@sha256:{'d' * 64}",
    )
    output = tmp_path / "run"
    FileEvaluationRunStore(output).open(manifest)
    manifest_path = output / "manifest.json"
    serialized = manifest_path.read_text(encoding="utf-8")

    assert GOLD_CANARY not in serialized
    assert str(tmp_path) not in serialized
    assert "private-host-repository" not in serialized

    manifest_path.write_text(
        serialized.replace('"source_commit":"', '"source_commit":"f'),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationRunnerError) as tampered:
        load_evaluation_run_manifest(output)
    assert tampered.value.code is EvaluationRunnerErrorCode.INVALID_MANIFEST

    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked-run"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(EvaluationRunnerError) as symlink:
        load_evaluation_run_manifest(linked)
    assert symlink.value.code is EvaluationRunnerErrorCode.PATH_UNSAFE
