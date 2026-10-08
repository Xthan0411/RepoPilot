"""P7-S6 pinned upstream evaluator evidence tests."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from repopilot.evals.dataset import FrozenDatasetManifest
from repopilot.schemas import ArtifactRef

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p7_swebench_environment import (  # noqa: E402
    EXPECTED_TASK_IDS,
    EnvironmentQualificationError,
    EnvironmentQualificationErrorCode,
    SweBenchAggregateReport,
    UpstreamTaskImageEvidence,
    _validate_report_outcomes,
    _write_private_create_only,
    build_upstream_qualification,
    python_version_matches,
)

NOW = datetime(2026, 8, 8, 14, tzinfo=UTC)


def _report(*, resolved: bool) -> SweBenchAggregateReport:
    resolved_ids = EXPECTED_TASK_IDS if resolved else ()
    unresolved_ids = () if resolved else EXPECTED_TASK_IDS
    return SweBenchAggregateReport.model_validate(
        {
            "total_instances": 8,
            "submitted_instances": 8,
            "completed_instances": 8,
            "resolved_instances": 8 if resolved else 0,
            "unresolved_instances": 0 if resolved else 8,
            "empty_patch_instances": 0,
            "error_instances": 0,
            "completed_ids": EXPECTED_TASK_IDS,
            "incomplete_ids": (),
            "empty_patch_ids": (),
            "submitted_ids": EXPECTED_TASK_IDS,
            "resolved_ids": resolved_ids,
            "unresolved_ids": unresolved_ids,
            "error_ids": (),
            "schema_version": 2,
        }
    )


def _manifest() -> FrozenDatasetManifest:
    return FrozenDatasetManifest.model_validate_json(
        PROJECT_ROOT.joinpath("data/manifests/p7-v1/manifest.json").read_bytes()
    )


def _images() -> tuple[UpstreamTaskImageEvidence, ...]:
    cases = {item.case_id: item for item in _manifest().external_cases}
    return tuple(
        UpstreamTaskImageEvidence(
            task_id=task_id,
            base_commit=cases[task_id].base_commit,
            declared_python_version=cases[task_id].python_version,
            observed_python_version=(
                "3.11.10"
                if task_id == "pallets__flask-4992"
                else ("3.10.15" if task_id.startswith("pydata__") else "3.9.20")
            ),
            python_matches_declaration=task_id != "pallets__flask-4992",
            local_image_tag=(
                f"swebench/sweb.eval.x86_64.{task_id.replace('__', '_1776_')}:repopilot-p7-s6-v1"
            ),
            source_image_reference=f"swebench/task@sha256:{'a' * 64}",
            platform="linux/amd64",
            size_bytes=1_000_000,
            base_commit_reachable=True,
            baseline_failed=True,
            gold_passed=True,
        )
        for task_id in EXPECTED_TASK_IDS
    )


def _ref(path: str) -> ArtifactRef:
    return ArtifactRef(
        relative_path=path,
        sha256=hashlib.sha256(path.encode()).hexdigest(),
        size_bytes=len(path),
        media_type="application/json",
    )


@pytest.mark.parametrize(
    ("declared", "observed", "matches"),
    (
        ("3.9", "3.9.20", True),
        ("3.10", "3.10.15", True),
        ("3.12.13", "3.12.13", True),
        ("3.12.13", "3.12.12", False),
        ("3.12.13", "3.11.10", False),
    ),
)
def test_python_version_match_is_explicit(
    declared: str,
    observed: str,
    matches: bool,
) -> None:
    assert python_version_matches(declared, observed) is matches


def test_upstream_reports_require_baseline_fail_and_gold_pass() -> None:
    _validate_report_outcomes(_report(resolved=False), _report(resolved=True))
    with pytest.raises(EnvironmentQualificationError) as captured:
        _validate_report_outcomes(_report(resolved=True), _report(resolved=True))
    assert captured.value.code is EnvironmentQualificationErrorCode.REPORT_OUTCOME_MISMATCH


def test_qualification_is_gold_free_and_adapter_blocked_by_flask_python() -> None:
    qualification = build_upstream_qualification(
        collector_commit="b" * 40,
        collected_at=NOW,
        dataset_manifest=_manifest(),
        baseline_ref=_ref("results/p7-holdout/upstream-evaluator/baseline-v1.json"),
        gold_ref=_ref("results/p7-holdout/upstream-evaluator/gold-v1.json"),
        task_images=_images(),
    )

    assert qualification.upstream_environment_qualified
    assert not qualification.repopilot_sandbox_adapter_ready
    assert qualification.paid_requests_made == 0
    assert qualification.adapter_blockers == (
        EnvironmentQualificationErrorCode.SANDBOX_ADAPTER_UNVERIFIED,
        EnvironmentQualificationErrorCode.TASK_PYTHON_MISMATCH,
    )
    assert "patch" not in qualification.model_dump_json()


def test_private_evidence_write_is_create_only(tmp_path: Path) -> None:
    path = tmp_path / "qualification.json"
    _write_private_create_only(path, json.dumps({"ready": False}).encode())

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(EnvironmentQualificationError) as captured:
        _write_private_create_only(path, b"replacement")
    assert captured.value.code is EnvironmentQualificationErrorCode.OUTPUT_CONFLICT
