"""Offline S8 inventory checks; no Provider or sandbox capabilities are used."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from repopilot.evals.reports import ReportIntegrityError, canonical_sha256

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from scripts import p7_error_analysis as analysis  # noqa: E402


def _single_fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    slot = {
        "arm_id": analysis.SINGLE_SHOT,
        "task_id": "fixture",
        "run_id": "planned-run",
        "trace_ref": {"relative_path": "trace.json"},
        "result_ref": {"relative_path": "result.json"},
        "execution": {"task_success": False, "termination_reason": "invalid_patch"},
    }
    result = {
        "task_id": "fixture",
        "run_id": "execution-run",
        "status": "failed",
        "termination_reason": "invalid_patch",
        "error_category": "invalid_patch",
        "verifier_error_code": "patch_not_applied",
        "task_metrics": {"task_success": False},
        "usage": {"estimated_cost_usd": "0.1"},
    }
    trace = {
        "trace_version": "single-shot-trace-v1",
        "task_id": "fixture",
        "run_id": "execution-run",
        "candidate_diff": "--- a/file.py\n+++ b/file.py\n",
        "patch": {
            "status": "rejected",
            "error_code": "invalid_diff",
            "summary": None,
            "patch_state": {"status": "none"},
        },
        "provider_requests": 1,
        "provider_retries": 0,
        "provider_status": "completed",
        "provider_failure": None,
        "provider_usage": {"total_tokens": 4},
        "request_sha256": "a" * 64,
        "context": {"initial_context_sha256": "b" * 64},
        "limits": {"max_tokens": 100},
        "elapsed_before_evaluator_ms": 10,
        "elapsed_total_ms": 30,
        "evaluation_stop": None,
        "baseline": {"test_run": {"status": "failed"}},
        "validation": None,
        "execution_order": ["provider_response", "sandbox_setup", "baseline", "patch"],
    }
    return slot, trace, result


def test_single_failure_facts_preserve_format_caveat_without_asserting_root_cause() -> None:
    case = analysis.extract_case(*_single_fixture())
    assert case["root_cause"] == "not_inferred"
    facts = case["trace_facts"]
    assert facts["patch_error_code"] == "invalid_diff"
    assert facts["candidate_has_git_header"] is False
    assert facts["provider_failure"] is None
    assert facts["provider_requests"] == 1
    assert facts["provider_retries"] == 0
    assert facts["validation_test_run"] is None
    assert "weaker" in facts["format_contract_caveat"]
    assert "candidate_diff" not in facts  # No large untrusted source body is copied.


@pytest.mark.parametrize("change", ["success", "task", "execution", "termination", "type"])
def test_case_extraction_rejects_wrong_or_successful_slot(change: str) -> None:
    slot, trace, result = _single_fixture()
    if change == "success":
        slot["execution"]["task_success"] = True
    elif change == "task":
        trace["task_id"] = "other"
    elif change == "execution":
        result["run_id"] = "other"
    elif change == "termination":
        result["termination_reason"] = "tests_failed"
    else:
        trace["trace_version"] = "wrong"
    with pytest.raises(ReportIntegrityError):
        analysis.extract_case(slot, trace, result)


def test_writer_is_idempotent_and_check_is_read_only(tmp_path: Path) -> None:
    path = tmp_path / "analysis.json"
    value = {"observed": 1}
    analysis.write_or_check(path, value, check=False)
    before = path.read_bytes()
    analysis.write_or_check(path, value, check=False)
    analysis.write_or_check(path, value, check=True)
    assert path.read_bytes() == before
    for check in (False, True):
        with pytest.raises(ReportIntegrityError, match="analysis_output_"):
            analysis.write_or_check(path, {"observed": 2}, check=check)
        assert path.read_bytes() == before


def test_check_missing_output_never_creates_it(tmp_path: Path) -> None:
    path = tmp_path / "not-created" / "result.json"
    with pytest.raises(ReportIntegrityError, match="analysis_output_mismatch"):
        analysis.write_or_check(path, {}, check=True)
    assert not path.parent.exists()


def test_output_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("original")
    path = tmp_path / "link.json"
    path.symlink_to(target)
    with pytest.raises(ReportIntegrityError, match="analysis_output_symlink"):
        analysis.write_or_check(path, {}, check=False)
    assert target.read_text() == "original"


def test_saved_s7_must_match_fresh_verified_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / analysis.AGGREGATE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(analysis.serialize({"planned_slots": 39}))
    monkeypatch.setattr(analysis, "build_v6_report", lambda **_: {"planned_slots": 40})
    with pytest.raises(ReportIntegrityError, match="saved_aggregate_does_not_match_raw"):
        analysis.build_analysis(repository_root=tmp_path, evidence_root=tmp_path)


@pytest.fixture(scope="module")
def actual_inventory() -> dict[str, Any]:
    evidence = ROOT / "artifacts/p7-postdebug-v6-evidence-v1"
    if not (evidence / "results/p7-holdout/postdebug-report-v6.json").is_file():
        pytest.skip("optional retained v6 archive is absent")
    return analysis.build_analysis(repository_root=ROOT, evidence_root=evidence)


def test_optional_inventory_covers_every_failed_slot_exactly_once(
    actual_inventory: dict[str, Any],
) -> None:
    aggregate = json.loads((ROOT / analysis.AGGREGATE_PATH).read_bytes())
    expected = {
        (s["arm_id"], s["task_id"])
        for s in aggregate["slots"]
        if s["execution"]["task_success"] is False
    }
    cases = actual_inventory["cases"]
    observed = [(s["arm_id"], s["task_id"]) for s in cases]
    assert len(observed) == len(set(observed)) == 20
    assert set(observed) == expected
    assert actual_inventory["missing_failed_slots"] == []
    assert actual_inventory["termination_counts"] == {
        "invalid_patch": 7,
        "repeated_action": 1,
        "tests_failed": 3,
        "token_limit": 9,
    }
    assert all(case["root_cause"] == "not_inferred" for case in cases)
    assert actual_inventory["provider_requests_made_by_analysis"] == 0
    assert actual_inventory["sandbox_test_runs_by_analysis"] == 0
    unsealed = copy.deepcopy(actual_inventory)
    digest = unsealed.pop("content_sha256")
    assert digest == canonical_sha256(unsealed)


def test_optional_single_shot_patch_rejections_are_not_reported_as_failed_validation(
    actual_inventory: dict[str, Any],
) -> None:
    cases = [c for c in actual_inventory["cases"] if c["arm_id"] == analysis.SINGLE_SHOT]
    expected = {
        "django__django-15902": "invalid_diff",
        "mwaskom__seaborn-3190": "invalid_diff",
        "pallets__flask-4992": "invalid_diff",
        "pydata__xarray-4094": "apply_conflict",
        "pydata__xarray-5131": "apply_conflict",
        "pytest-dev__pytest-11148": "apply_conflict",
        "sympy__sympy-20590": "apply_conflict",
    }
    assert {c["task_id"]: c["trace_facts"]["patch_error_code"] for c in cases} == expected
    for case in cases:
        facts = case["trace_facts"]
        assert facts["provider_requests"] == 1 and facts["provider_retries"] == 0
        assert facts["provider_status"] == "completed" and facts["provider_failure"] is None
        assert facts["patch_state"]["revision"] == 0
        assert facts["validation_test_run"] is None


def test_optional_budget_termination_is_observed_with_limits_not_assumed_root_cause(
    actual_inventory: dict[str, Any],
) -> None:
    cases = [
        c
        for c in actual_inventory["cases"]
        if c["observed_result"]["termination_reason"] == "token_limit"
    ]
    assert len(cases) == 9
    for case in cases:
        facts = case["trace_facts"]
        assert facts["budget"]["tokens_used"] > facts["budget"]["max_tokens"] == 100000
        assert case["root_cause"] == "not_inferred"
        assert facts["provider_events"][-1]["sequence"] < facts["terminal_event_sequence"]
        assert all(p["context_usage"] is not None for p in facts["provider_events"])


def test_optional_failed_validation_is_not_conflated_with_patch_format_rejection(
    actual_inventory: dict[str, Any],
) -> None:
    cases = [
        c
        for c in actual_inventory["cases"]
        if c["observed_result"]["termination_reason"] == "tests_failed"
    ]
    assert len(cases) == 3
    for case in cases:
        assert case["task_id"] == "pallets__flask-4992"
        assert case["observed_result"]["error_category"] == "invalid_patch"
        facts = case["trace_facts"]
        assert [p["patch_state"]["revision"] for p in facts["patch_events"]] == [1, 2]
        assert all(p["patch_error_code"] is None for p in facts["patch_events"])
        assert [t["test_run"]["status"] for t in facts["test_events"]] == ["failed"] * 3
        assert len(facts["replan_events"]) == 1


def test_optional_proposed_action_is_not_counted_as_executed_after_stop(
    actual_inventory: dict[str, Any],
) -> None:
    expected = {
        ("dense-only-v1", "pytest-dev__pytest-11148"): "apply_patch",
        ("hybrid-rrf-v1", "sympy__sympy-20590"): "run_tests",
        ("mini-swe-safe-v1", "sympy__sympy-20590"): "inspect_symbol",
    }
    for case in actual_inventory["cases"]:
        key = (case["arm_id"], case["task_id"])
        if key in expected:
            proposed = case["trace_facts"]["provider_events"][-1]["proposed_tool"]
            assert proposed["tool_name"] == expected[key]
            assert proposed["execution_event_sequence"] is None
