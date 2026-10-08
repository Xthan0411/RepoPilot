from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from repopilot.enums import (
    DatasetExclusionReason,
    DatasetManifestErrorCode,
    DatasetPreflightStatus,
    DatasetSelectionDecision,
)
from repopilot.evals import (
    CandidateDecision,
    DatasetManifestError,
    DatasetPreflightEvidence,
    FrozenDatasetManifest,
    FrozenExternalCase,
    build_dataset_artifact_ref,
    load_candidate_decision_log,
    load_dataset_preflight_report,
    load_frozen_dataset,
)

REPOSITORY_ROOT = Path(__file__).parents[2].resolve()
MANIFEST = REPOSITORY_ROOT / "data/manifests/p7-v1/manifest.json"
DECISIONS = REPOSITORY_ROOT / "data/manifests/p7-v1/candidate-decisions.json"
PREFLIGHT = REPOSITORY_ROOT / "results/p7-dataset-v1/preflight-report.json"


def test_tracked_frozen_dataset_loads_with_exact_counts_and_provenance() -> None:
    bundle = load_frozen_dataset(MANIFEST, repository_root=REPOSITORY_ROOT)
    decisions = load_candidate_decision_log(
        DECISIONS,
        repository_root=REPOSITORY_ROOT,
        manifest=bundle.manifest,
    )
    report = load_dataset_preflight_report(
        PREFLIGHT,
        repository_root=REPOSITORY_ROOT,
        manifest=bundle.manifest,
    )

    assert bundle.manifest.dataset_id == "repopilot-p7-v1"
    assert bundle.manifest.source.dataset_revision == ("69611d31007e1c6731db8bd5b5c3f2d33f5bab6e")
    assert len(bundle.fixture_manifest.cases) == 2
    assert len(bundle.external_evaluation_tasks) == 8
    assert len(bundle.evaluation_tasks()) == 10
    assert len(decisions.decisions) == 11
    assert (
        sum(item.decision is DatasetSelectionDecision.EXCLUDED for item in decisions.decisions) == 3
    )
    assert [
        item.case_id
        for item in report.entries
        if item.status is DatasetPreflightStatus.BASELINE_FAILED_GOLD_PASSED
    ] == ["pallets__flask-4992"]
    assert {item.repository_license.spdx_identifier for item in bundle.manifest.external_cases} == {
        "Apache-2.0",
        "BSD-3-Clause",
        "MIT",
    }


def test_agent_projection_contains_no_evaluator_gold() -> None:
    bundle = load_frozen_dataset(MANIFEST, repository_root=REPOSITORY_ROOT)
    agent_payload = json.dumps(
        [item.model_dump(mode="json") for item in bundle.agent_tasks()],
        sort_keys=True,
    )
    evaluator_payload = json.dumps(
        [item.model_dump(mode="json") for item in bundle.external_evaluation_tasks],
        sort_keys=True,
    )

    assert "expected_patch" not in agent_payload
    assert "gold_files" not in agent_payload
    assert "+        text: bool = True" not in agent_payload
    assert "+        text: bool = True" in evaluator_payload
    flask_task = next(
        item for item in bundle.agent_tasks() if item.task_id == "pallets__flask-4992"
    )
    assert str(flask_task.repository_url) == "https://github.com/pallets/flask.git"
    assert flask_task.repository_path is None


def test_artifact_builder_uses_actual_contained_bytes() -> None:
    reference = build_dataset_artifact_ref(
        REPOSITORY_ROOT,
        "data/manifests/p7-v1/source-rows.json",
        "application/json",
    )

    assert reference.sha256 == ("66235d8dc551ae5aa7049705f6ff726a04310b4090e0624292260199f0dfa41b")
    assert reference.size_bytes == 146_362


def test_source_snapshot_and_preflight_log_tampering_are_rejected(tmp_path: Path) -> None:
    copied_root = _copy_dataset_inputs(tmp_path)
    copied_manifest = copied_root / "data/manifests/p7-v1/manifest.json"
    source_rows = copied_root / "data/manifests/p7-v1/source-rows.json"
    source_rows.write_bytes(source_rows.read_bytes() + b" ")

    with pytest.raises(DatasetManifestError) as source_error:
        load_frozen_dataset(copied_manifest, repository_root=copied_root)
    assert source_error.value.code is DatasetManifestErrorCode.INTEGRITY_MISMATCH

    copied_root = _copy_dataset_inputs(tmp_path / "logs")
    bundle = load_frozen_dataset(
        copied_root / "data/manifests/p7-v1/manifest.json",
        repository_root=copied_root,
    )
    gold_log = copied_root / "results/p7-dataset-v1/preflight/flask-4992-gold.log"
    gold_log.write_bytes(gold_log.read_bytes() + b"tampered")
    with pytest.raises(DatasetManifestError) as log_error:
        load_dataset_preflight_report(
            copied_root / "results/p7-dataset-v1/preflight-report.json",
            repository_root=copied_root,
            manifest=bundle.manifest,
        )
    assert log_error.value.code is DatasetManifestErrorCode.INTEGRITY_MISMATCH


def test_unknown_version_and_candidate_identity_drift_are_rejected(
    tmp_path: Path,
) -> None:
    copied_root = _copy_dataset_inputs(tmp_path)
    manifest_path = copied_root / "data/manifests/p7-v1/manifest.json"
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(raw_manifest), encoding="utf-8")
    with pytest.raises(DatasetManifestError) as version_error:
        load_frozen_dataset(manifest_path, repository_root=copied_root)
    assert version_error.value.code is DatasetManifestErrorCode.UNSUPPORTED_VERSION

    bundle = load_frozen_dataset(MANIFEST, repository_root=REPOSITORY_ROOT)
    copied_log = tmp_path / "invalid-decisions.json"
    raw_log = json.loads(DECISIONS.read_text(encoding="utf-8"))
    included = next(item for item in raw_log["decisions"] if item["decision"] == "included")
    included["decision"] = "excluded"
    copied_log.write_text(json.dumps(raw_log), encoding="utf-8")
    with pytest.raises(DatasetManifestError) as decision_error:
        load_candidate_decision_log(
            copied_log,
            repository_root=tmp_path.resolve(),
            manifest=bundle.manifest,
        )
    assert decision_error.value.code is DatasetManifestErrorCode.MANIFEST_INVALID


def test_schema_rejects_remote_setup_sources_and_false_preflight_claims() -> None:
    raw_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw_case = raw_manifest["external_cases"][0]
    raw_case["setup_commands"] = [
        {
            "argv": [
                "python",
                "-m",
                "pip",
                "install",
                "https://untrusted.invalid/package.whl",
            ]
        }
    ]
    with pytest.raises(ValidationError):
        FrozenExternalCase.model_validate(raw_case)

    with pytest.raises(ValidationError):
        DatasetPreflightEvidence.model_validate(
            {
                "case_id": "case",
                "kind": "github_issue",
                "status": "metadata_verified",
                "checked_at": "2026-08-07T00:00:00Z",
                "python_version": "3.12",
                "baseline_exit_code": 1,
            }
        )
    with pytest.raises(ValidationError):
        DatasetPreflightEvidence.model_validate(
            {
                "case_id": "case",
                "kind": "github_issue",
                "status": "baseline_failed_gold_passed",
                "checked_at": "2026-08-07T00:00:00Z",
                "python_version": "3.12",
                "sandbox_image_reference": "python:3.12",
                "baseline_exit_code": 0,
                "gold_exit_code": 0,
            }
        )


def test_candidate_decision_requires_reason_only_when_excluded() -> None:
    common = {
        "source_instance_id": "owner__repo-1",
        "repository": "owner/repo",
        "pull_request_url": "https://github.com/owner/repo/pull/1",
        "base_commit": "a" * 40,
        "rationale": "Reviewed candidate.",
    }
    with pytest.raises(ValidationError):
        CandidateDecision.model_validate(
            {
                **common,
                "decision": DatasetSelectionDecision.EXCLUDED,
                "reasons": (),
            }
        )
    with pytest.raises(ValidationError):
        CandidateDecision.model_validate(
            {
                **common,
                "decision": DatasetSelectionDecision.INCLUDED,
                "reasons": (DatasetExclusionReason.RESOURCE_LIMIT,),
            }
        )


def test_manifest_rejects_boolean_counts_and_mutable_repository_url() -> None:
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["dev_task_count"] = True
    with pytest.raises(ValidationError):
        FrozenDatasetManifest.model_validate(raw)

    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["external_cases"][0]["repository_url"] = "https://github.com/django/django/tree/main"
    with pytest.raises(ValidationError):
        FrozenDatasetManifest.model_validate(raw)


def _copy_dataset_inputs(destination: Path) -> Path:
    root = destination.resolve()
    (root / "data/manifests").mkdir(parents=True)
    (root / "tests/fixtures").mkdir(parents=True)
    (root / "results").mkdir(parents=True)
    shutil.copytree(
        REPOSITORY_ROOT / "data/manifests/p7-v1",
        root / "data/manifests/p7-v1",
    )
    shutil.copytree(
        REPOSITORY_ROOT / "tests/fixtures/p5_repair_tasks",
        root / "tests/fixtures/p5_repair_tasks",
    )
    shutil.copytree(
        REPOSITORY_ROOT / "results/p7-dataset-v1",
        root / "results/p7-dataset-v1",
    )
    return root
