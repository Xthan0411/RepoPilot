"""Offline report integrity and optional retained-v6 consistency checks."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from repopilot.enums import (
    ErrorCategory,
    ReproductionStatus,
    RunStatus,
    TerminationReason,
    TraceNode,
    ValidationStatus,
)
from repopilot.evals import reports
from repopilot.evals.report_formats import render_markdown, render_slots_csv
from repopilot.repair_workflow import RepairArtifactManifest
from repopilot.schemas import (
    AgentState,
    ArtifactRef,
    BudgetState,
    FinalResult,
    LocalizationMetrics,
    ReproductionResult,
    TaskMetrics,
    TaskSpec,
    UsageMetrics,
    ValidationResult,
)
from repopilot.trace import AgentTrace, TraceEvent

ROOT = Path(__file__).parents[2]
V6_EVIDENCE = ROOT / "artifacts/p7-postdebug-v6-evidence-v1"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_contained_file_accepts_regular_nested_file(tmp_path: Path) -> None:
    target = tmp_path / "artifacts" / "result.txt"
    target.parent.mkdir()
    target.write_text("retained output")
    assert reports.contained_file(tmp_path, "artifacts/result.txt") == target


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../../outside", "a\\outside"])
def test_contained_file_rejects_unsafe_paths(tmp_path: Path, name: str) -> None:
    with pytest.raises(reports.ReportIntegrityError, match="unsafe_artifact_path"):
        reports.contained_file(tmp_path, name)


@pytest.mark.parametrize("parent_link", [False, True])
def test_contained_file_rejects_symlinks_even_to_in_root_files(
    tmp_path: Path, parent_link: bool
) -> None:
    target = tmp_path / "real"
    target.mkdir()
    (target / "value.txt").write_text("not a capability through a symlink")
    if parent_link:
        (tmp_path / "linked").symlink_to(target, target_is_directory=True)
        name = "linked/value.txt"
    else:
        (tmp_path / "linked.txt").symlink_to(target / "value.txt")
        name = "linked.txt"
    with pytest.raises(reports.ReportIntegrityError, match="symlink_artifact"):
        reports.contained_file(tmp_path, name)


def test_missing_file_is_explicit_not_a_zero_metric(tmp_path: Path) -> None:
    with pytest.raises(reports.ReportIntegrityError, match=r"missing_artifact:absent\.json"):
        reports.contained_file(tmp_path, "absent.json")


def test_sealed_controls_are_recomputed_not_trusted(tmp_path: Path) -> None:
    content = {"version": "fixture-v1", "successes": 2}
    content["content_sha256"] = reports.canonical_sha256(content)
    _write_json(tmp_path / "control.json", content)
    assert reports._sealed(tmp_path, "control.json") == content
    content["successes"] = 3
    _write_json(tmp_path / "control.json", content)
    with pytest.raises(reports.ReportIntegrityError, match="control_digest_mismatch"):
        reports._sealed(tmp_path, "control.json")


def _slots() -> list[dict[str, Any]]:
    return [
        {"arm_id": "arm-a", "task_id": "task-1", "run_id": "run-1", "output_path": "raw/1"},
        {"arm_id": "arm-a", "task_id": "task-2", "run_id": "run-2", "output_path": "raw/2"},
        {"arm_id": "arm-b", "task_id": "task-1", "run_id": "run-3", "output_path": "raw/3"},
        {"arm_id": "arm-b", "task_id": "task-2", "run_id": "run-4", "output_path": "raw/4"},
    ]


def test_slot_inventory_is_order_independent_but_preserves_arm_task_denominators() -> None:
    slots = _slots()
    reports.validate_slot_inventory(slots, list(reversed(slots)))


@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("missing", "missing_or_extra_reported_slot"),
        ("extra", "missing_or_extra_reported_slot"),
        ("duplicate_reported", "duplicate_reported_slot"),
        ("duplicate_planned", "duplicate_planned_slot"),
        ("duplicate_run", "duplicate_run_id"),
        ("duplicate_path", "duplicate_output_path"),
    ],
)
def test_slot_inventory_rejects_missing_duplicate_and_unplanned_rows(case: str, code: str) -> None:
    slots = _slots()
    outcomes = copy.deepcopy(slots)
    if case == "missing":
        outcomes.pop()
    elif case == "extra":
        outcomes.append({"arm_id": "arm-extra", "task_id": "task-1"})
    elif case == "duplicate_reported":
        outcomes.append(copy.deepcopy(outcomes[0]))
    elif case == "duplicate_planned":
        slots.append(copy.deepcopy(slots[0]))
    elif case == "duplicate_run":
        slots[1]["run_id"] = slots[0]["run_id"]
    elif case == "duplicate_path":
        slots[1]["output_path"] = slots[0]["output_path"]
    with pytest.raises(reports.ReportIntegrityError, match=code):
        reports.validate_slot_inventory(slots, outcomes)


@pytest.fixture
def nested_evidence(tmp_path: Path) -> tuple[Path, ArtifactRef]:
    content = b"retained evaluator output\n"
    target = tmp_path / "artifacts" / "output.txt"
    target.parent.mkdir()
    target.write_bytes(content)
    ref = ArtifactRef(
        relative_path="artifacts/output.txt",
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        media_type="text/plain",
    )
    manifest = RepairArtifactManifest(
        task_id="fixture-task",
        run_id=UUID("00000000-0000-0000-0000-000000000001"),
        thread_id="fixture-thread",
        provider_name="no-provider-constructed",
        sandbox_image_reference="python@sha256:" + "1" * 64,
        settings_sha256="2" * 64,
        prompt_versions=("fixture-v1",),
        artifacts=(ref,),
    )
    _write_json(tmp_path / "artifacts.json", manifest.model_dump(mode="json"))
    _write_json(tmp_path / "result.json", {"nested": {"stdout_ref": ref.model_dump(mode="json")}})
    _write_json(tmp_path / "trace.json", {"deep": [{"stdout_ref": ref.model_dump(mode="json")}]})
    return tmp_path, ref


def test_nested_artifact_references_are_deduplicated_and_all_bytes_checked(
    nested_evidence: tuple[Path, ArtifactRef],
) -> None:
    directory, _ = nested_evidence
    assert reports.verify_execution_artifacts(directory) == 1


@pytest.mark.parametrize("same_length", [False, True])
def test_nested_artifact_corruption_is_rejected_even_with_same_size(
    nested_evidence: tuple[Path, ArtifactRef], same_length: bool
) -> None:
    directory, ref = nested_evidence
    target = directory / ref.relative_path
    target.write_bytes(b"X" * (ref.size_bytes if same_length else ref.size_bytes + 1))
    with pytest.raises(reports.ReportIntegrityError, match="artifact_digest_mismatch"):
        reports.verify_execution_artifacts(directory)


def test_conflicting_reference_for_same_path_is_rejected(
    nested_evidence: tuple[Path, ArtifactRef],
) -> None:
    directory, ref = nested_evidence
    changed = ref.model_dump(mode="json")
    changed["sha256"] = "0" * 64
    _write_json(directory / "trace.json", {"nested": changed})
    with pytest.raises(reports.ReportIntegrityError, match="conflicting_artifact_reference"):
        reports.verify_execution_artifacts(directory)


def test_unlisted_execution_file_is_rejected(nested_evidence: tuple[Path, ArtifactRef]) -> None:
    directory, _ = nested_evidence
    (directory / "unlisted.txt").write_text("not silently ignored")
    with pytest.raises(reports.ReportIntegrityError, match="unlisted_execution_artifact"):
        reports.verify_execution_artifacts(directory)


def test_missing_referenced_artifact_is_rejected(nested_evidence: tuple[Path, ArtifactRef]) -> None:
    directory, ref = nested_evidence
    (directory / ref.relative_path).unlink()
    with pytest.raises(reports.ReportIntegrityError, match="missing_artifact"):
        reports.verify_execution_artifacts(directory)


def test_extra_reference_not_in_artifact_manifest_is_rejected(
    nested_evidence: tuple[Path, ArtifactRef],
) -> None:
    directory, ref = nested_evidence
    extra = ref.model_dump(mode="json")
    extra["relative_path"] = "artifacts/not-listed.txt"
    _write_json(directory / "trace.json", {"extra": extra})
    with pytest.raises(reports.ReportIntegrityError, match="artifact_set_mismatch"):
        reports.verify_execution_artifacts(directory)


@pytest.fixture
def stopped_trace_result() -> tuple[AgentTrace, FinalResult]:
    run_id = UUID("00000000-0000-0000-0000-000000000001")
    started = datetime(2026, 10, 31, tzinfo=UTC)
    state = AgentState(
        task=TaskSpec.model_validate_json((ROOT / "tests/fixtures/valid_task.json").read_bytes()),
        run_id=run_id,
        budget=BudgetState(
            max_steps=10,
            max_tokens=100,
            max_cost_usd=Decimal(1),
            started_at=started,
            deadline_at=started + timedelta(seconds=60),
        ),
    )
    final = state.model_copy(update={"termination_reason": TerminationReason.TOKEN_LIMIT})
    trace = AgentTrace.build(
        run_id=run_id,
        task_id=state.task.task_id,
        thread_id="offline-result-fixture",
        provider_name="no-provider",
        settings={},
        events=(
            TraceEvent(
                sequence=0, node=TraceNode.INITIALIZE_PLAN, state_before=state, state_after=state
            ),
            TraceEvent(sequence=1, node=TraceNode.DECIDE, state_before=state, state_after=final),
            TraceEvent(sequence=2, node=TraceNode.TERMINATE, state_before=final, state_after=final),
        ),
        final_state=final,
    )
    result = FinalResult(
        task_id=state.task.task_id,
        run_id=run_id,
        status=RunStatus.STOPPED,
        termination_reason=TerminationReason.TOKEN_LIMIT,
        reproduction_result=ReproductionResult(status=ReproductionStatus.NOT_ATTEMPTED),
        validation_result=ValidationResult(
            status=ValidationStatus.NOT_RUN, summary="Post-patch validation was not run."
        ),
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
        usage=UsageMetrics(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            estimated_cost_usd=Decimal(0),
            latency_ms=0,
        ),
        trace_path="trace.json",
        error_category=ErrorCategory.BUDGET_EXHAUSTED,
    )
    return trace, result


def test_agent_result_reconstruction_preserves_runtime_budget_stop(
    stopped_trace_result: tuple[AgentTrace, FinalResult],
) -> None:
    reports.verify_agent_result(*stopped_trace_result)


@pytest.mark.parametrize("change", ["usage", "steps", "termination", "patch", "task", "trace"])
def test_agent_result_cannot_change_verified_trace_outcome(
    stopped_trace_result: tuple[AgentTrace, FinalResult], change: str
) -> None:
    trace, result = stopped_trace_result
    updates: dict[str, Any]
    if change == "usage":
        updates = {"usage": result.usage.model_copy(update={"estimated_cost_usd": Decimal("0.1")})}
    elif change == "steps":
        updates = {"task_metrics": result.task_metrics.model_copy(update={"tool_steps": 9})}
    elif change == "termination":
        updates = {"termination_reason": TerminationReason.TIMEOUT}
    elif change == "patch":
        updates = {"patch_path": "artifacts/invented.patch"}
    elif change == "task":
        updates = {"task_id": "other-task"}
    else:
        updates = {"trace_path": "another-trace.json"}
    with pytest.raises(reports.ReportIntegrityError, match="agent_result_"):
        reports.verify_agent_result(trace, result.model_copy(update=updates))


@pytest.fixture(scope="module")
def retained_v6_report() -> dict[str, Any]:
    if not (V6_EVIDENCE / "results/p7-holdout/postdebug-report-v6.json").is_file():
        pytest.skip("optional local retained-v6 archive is not installed")
    return reports.build_v6_report(repository_root=ROOT, evidence_root=V6_EVIDENCE)


def test_optional_retained_v6_numbers_are_rebuilt_with_all_40_slots(
    retained_v6_report: dict[str, Any],
) -> None:
    report = retained_v6_report
    assert report["planned_slots"] == 40
    assert report["distinct_task_count"] == 8
    assert report["missing_slots"] == []
    assert report["complete"] is True
    assert report["provider_requests_made_by_report"] == 0
    assert len(report["slots"]) == 40
    assert len({(r["arm_id"], r["task_id"]) for r in report["slots"]}) == 40
    assert report["overall"]["successful_slots"] == 20
    assert report["overall"]["completed_slots"] == 40
    assert report["overall"]["technical_failure_slots"] == 0
    expected = {
        "dense-only-v1": (4, "1.926355625"),
        "hybrid-rrf-v1": (6, "1.810553750"),
        "mini-swe-safe-v1": (5, "1.618186875"),
        "no-semantic-rag-v1": (4, "2.056925000"),
        "single-shot-hybrid-context-v1": (1, "0.303288750"),
    }
    for arm in report["arms"]:
        success, cost = expected[arm["arm_id"]]
        metrics = arm["execution"]
        assert metrics["planned_slots"] == metrics["completed_slots"] == 8
        assert metrics["successful_slots"] == success
        assert Decimal(metrics["known_cost_usd"]) == Decimal(cost)
    assert Decimal(report["accounting"]["known_cost_usd"]) == Decimal("7.715310000")
    assert Decimal(report["accounting"]["total_accounted_cost_usd"]) == Decimal("30.340901875")
    assert Decimal(report["historical_unknown_reserve_usd"]) == Decimal(11)
    assert report["content_sha256"] == reports.canonical_sha256(
        {k: v for k, v in report.items() if k != "content_sha256"}
    )


def test_optional_retained_report_does_not_invent_safety_attack_denominators(
    retained_v6_report: dict[str, Any],
) -> None:
    metrics = retained_v6_report["overall"]
    assert metrics["safety_evaluation"] == "not_measured_by_this_dataset"
    assert metrics["adversarial_interception_rate"] is None
    assert metrics["unsafe_host_execution_count"] is None
    assert metrics["secret_exposure_count"] is None
    assert any("diff-format" in text for text in retained_v6_report["limitations"])


def test_optional_rendered_outputs_keep_missing_metrics_and_model_steps_separate(
    retained_v6_report: dict[str, Any],
) -> None:
    report = retained_v6_report
    rendered = render_slots_csv(report)
    rows = list(csv.DictReader(io.StringIO(rendered)))
    assert len(rows) == 40
    assert all(row["trace_sha256"] and row["result_sha256"] for row in rows)
    mini = [r for r in rows if r["arm_id"] == "mini-swe-safe-v1"]
    assert len(mini) == 8 and all(r["file_recall_at_1"] == "" for r in mini)
    single = [r for r in rows if r["arm_id"] == reports.SINGLE_SHOT]
    assert len(single) == 8 and all(r["tool_steps"] == "0" for r in single)
    markdown = render_markdown(report)
    assert "0.875 (n=4)" not in markdown
    assert "6/8" in markdown and "4/8" in markdown and "N/A" in markdown
    assert render_slots_csv(report) == rendered
    assert render_markdown(report) == markdown


def test_csv_escapes_formula_like_text_but_preserves_numeric_zero() -> None:
    minimal = {
        "slots": [
            {
                "execution": {"arm_id": "=HYPERLINK(1)", "tool_steps": 0},
                "localization": {"metrics": {}, "basis": "raw", "status": "unavailable"},
                "trace_ref": {"relative_path": "trace.json", "sha256": "a" * 64},
                "result_ref": {"sha256": "b" * 64},
            }
        ],
    }
    row = next(csv.DictReader(io.StringIO(render_slots_csv(minimal))))
    assert row["arm_id"] == "'=HYPERLINK(1)"
    assert row["tool_steps"] == "0"
    assert row["file_recall_at_1"] == ""


@pytest.mark.parametrize("field", ["task_success", "cost"])
def test_optional_report_outcome_cannot_override_verified_raw(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    if not (V6_EVIDENCE / "results/p7-holdout/postdebug-report-v6.json").is_file():
        pytest.skip("optional local retained-v6 archive is not installed")
    original = reports._sealed

    def changed_after_control_validation(root: Path, name: str) -> dict[str, Any]:
        document = original(root, name)
        if name.endswith("postdebug-report-v6.json"):
            if field == "task_success":
                document["outcomes"][0]["task_success"] = False
            else:
                document["outcomes"][0]["usage"]["estimated_cost_usd"] = "0"
        return document

    # Deliberately bypass the already-tested control seal in memory to exercise
    # the independent raw-vs-summary comparison. No retained file is modified.
    monkeypatch.setattr(reports, "_sealed", changed_after_control_validation)
    with pytest.raises(reports.ReportIntegrityError, match="raw_and_reported_outcome_differ"):
        reports.build_v6_report(repository_root=ROOT, evidence_root=V6_EVIDENCE)
