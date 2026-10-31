"""Offline, fail-closed aggregation of retained evaluation evidence.

No executor, model, embedding backend, database, or sandbox is constructed here.
Gold file names are joined only after execution, never injected into Agent state.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

from repopilot.enums import ReproductionStatus, RunStatus, TerminationReason
from repopilot.evals.dataset import load_frozen_dataset
from repopilot.evals.report_execution import (
    SlotExecutionMetrics,
    aggregate_execution_metrics,
    extract_execution_metrics,
)
from repopilot.evals.report_localization import (
    METRIC_DEFINITIONS,
    reconstruct_initial_localization,
    retained_chunk_paths,
    summarize_localization,
)
from repopilot.evals.runner import (
    EvaluationAttemptEvent,
    inspect_evaluation_run,
    load_evaluation_run_manifest,
)
from repopilot.evals.single_shot import (
    SingleShotTrace,
    load_hybrid_initial_context,
    verify_single_shot_trace,
)
from repopilot.repair_workflow import (
    RepairArtifactManifest,
    _fresh_validation_test,
    _runtime_error_category,
)
from repopilot.replay import replay_trace
from repopilot.schemas import FinalResult, ReproductionResult, UsageMetrics
from repopilot.tools import build_reproduction_result
from repopilot.trace import AgentTrace
from repopilot.verifier import (
    FinalResultContext,
    RuleBasedVerifier,
    VerificationEvidence,
    build_final_result,
)

CONTROL_DIR = "results/p7-holdout"
SINGLE_SHOT = "single-shot-hybrid-context-v1"
LIMITATIONS = (
    "8 previously inspected issues, one repetition per arm; "
    "40 slots are not 40 independent issues.",
    "Post-debug seen-task rerun, not fresh blind holdout or the official full SWE-bench score.",
    "Single-shot uses the first actually rendered Hybrid context, but its diff-format instruction "
    "is weaker. Interaction, compute, and output-contract differences confound causal attribution.",
    "The 75% versus 12.5% comparison is descriptive, not a clean architecture-only uplift claim.",
    "Historical unknown-cost reserves remain reserved; estimates exclude cloud charges and "
    "are not a reconciled provider invoice.",
    "No new paid experiment, patch repair, sandbox test, indexing, "
    "or embedding is run by this report.",
    "Safety telemetry is not an adversarial security evaluation or proof of no secret exposure.",
)


class ReportIntegrityError(ValueError):
    """Evidence is incomplete or inconsistent; no report should be published."""


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ReportIntegrityError(code)


def contained_file(root: Path, name: str) -> Path:
    """Reject traversal and links before reading artifact bytes."""
    relative = PurePosixPath(name)
    _require(
        not relative.is_absolute()
        and bool(relative.parts)
        and all(part not in {".", ".."} for part in relative.parts)
        and "\\" not in name,
        "unsafe_artifact_path",
    )
    current = root
    for part in relative.parts:
        current = current / part
        _require(not current.is_symlink(), "symlink_artifact")
    _require(current.is_file(), f"missing_artifact:{name}")
    return current


def _read_json(root: Path, name: str) -> dict[str, Any]:
    value = json.loads(contained_file(root, name).read_bytes())
    _require(isinstance(value, dict), "expected_json_object")
    return dict(value)


def _sealed(root: Path, name: str) -> dict[str, Any]:
    value = _read_json(root, name)
    digest = value.get("content_sha256")
    _require(
        digest == canonical_sha256({k: v for k, v in value.items() if k != "content_sha256"}),
        f"control_digest_mismatch:{name}",
    )
    return value


def _file_ref(root: Path, name: str) -> dict[str, Any]:
    raw = contained_file(root, name).read_bytes()
    return {
        "relative_path": name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "media_type": "application/json",
    }


def _references(value: object) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {"relative_path", "sha256", "size_bytes", "media_type"} <= value.keys():
            refs.append(value)
        for child in value.values():
            refs.extend(_references(child))
    elif isinstance(value, list):
        for child in value:
            refs.extend(_references(child))
    return refs


def verify_execution_artifacts(execution: Path) -> int:
    """Verify every nested source/output reference, not just the three root JSONs."""
    docs = [_read_json(execution, name) for name in ("result.json", "trace.json", "artifacts.json")]
    manifest = RepairArtifactManifest.model_validate(docs[2])
    refs: dict[str, dict[str, Any]] = {}
    for ref in _references(docs):
        name = ref["relative_path"]
        _require(name not in refs or refs[name] == ref, "conflicting_artifact_reference")
        refs[name] = ref
    _require(set(refs) == {r.relative_path for r in manifest.artifacts}, "artifact_set_mismatch")
    for name, ref in refs.items():
        raw = contained_file(execution, name).read_bytes()
        _require(
            len(raw) == ref["size_bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"],
            f"artifact_digest_mismatch:{name}",
        )
    actual = set()
    for path in execution.rglob("*"):
        _require(not path.is_symlink(), "symlink_artifact")
        if path.is_file():
            actual.add(path.relative_to(execution).as_posix())
    _require(
        actual == set(refs) | {"result.json", "trace.json", "artifacts.json"},
        "unlisted_execution_artifact",
    )
    return len(refs)


def validate_slot_inventory(slots: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> None:
    """Check identities and coverage before aggregating, preserving planned denominators."""
    expected = [(s["arm_id"], s["task_id"]) for s in slots]
    observed = [(s["arm_id"], s["task_id"]) for s in outcomes]
    _require(len(expected) == len(set(expected)), "duplicate_planned_slot")
    _require(len(observed) == len(set(observed)), "duplicate_reported_slot")
    _require(set(expected) == set(observed), "missing_or_extra_reported_slot")
    _require(len({s["run_id"] for s in slots}) == len(slots), "duplicate_run_id")
    _require(len({s["output_path"] for s in slots}) == len(slots), "duplicate_output_path")


def verify_agent_result(trace: AgentTrace, result: FinalResult) -> None:
    """Re-derive Agent outcome/usage from retained state without any execution.

    Localization fields are not evidence of runtime quality and are deliberately
    passed through here: S7 reconstructs their metrics separately from frozen Gold.
    No retrieval timing or metadata is invented to call a production executor.
    """
    state = replay_trace(trace).final_state
    _require(
        trace.task_id == result.task_id and trace.run_id == result.run_id,
        "agent_result_identity_mismatch",
    )
    responses = [event.provider_response for event in trace.events if event.provider_response]
    usage = UsageMetrics(
        input_tokens=sum(response.usage.input_tokens for response in responses),
        output_tokens=sum(response.usage.output_tokens for response in responses),
        total_tokens=sum(response.usage.total_tokens for response in responses),
        estimated_cost_usd=sum(
            (response.usage.estimated_cost_usd for response in responses), Decimal(0)
        ),
        latency_ms=sum(response.usage.latency_ms for response in responses),
    )
    reproduction = (
        ReproductionResult(status=ReproductionStatus.NOT_ATTEMPTED)
        if state.baseline_test_result is None
        else build_reproduction_result(state.baseline_test_result)
    )
    decision = RuleBasedVerifier().verify(
        VerificationEvidence(
            reproduction_result=reproduction,
            patch_state=state.patch_state,
            validation_test_run=_fresh_validation_test(state),
            tool_steps=len(state.tool_results),
        )
    )
    context = FinalResultContext(
        task_id=trace.task_id,
        run_id=trace.run_id,
        localization_metrics=result.localization_metrics,
        usage=usage,
        trace_path="trace.json",
    )
    if state.termination_reason in {
        TerminationReason.AGENT_STOPPED,
        TerminationReason.TESTS_FAILED,
        TerminationReason.INVALID_PATCH,
    }:
        expected = build_final_result(decision, context)
    else:
        _require(state.termination_reason is not None, "agent_missing_terminal_reason")
        assert state.termination_reason is not None
        expected = FinalResult(
            task_id=trace.task_id,
            run_id=trace.run_id,
            status=RunStatus.STOPPED,
            termination_reason=state.termination_reason,
            patch_path=decision.patch_path,
            reproduction_result=decision.reproduction_result,
            validation_result=decision.validation_result,
            localization_metrics=result.localization_metrics,
            task_metrics=decision.task_metrics.model_copy(update={"task_success": False}),
            usage=usage,
            trace_path="trace.json",
            error_category=_runtime_error_category(state.termination_reason),
        )
    _require(expected == result, "agent_result_does_not_match_trace")


def build_v6_report(*, repository_root: Path, evidence_root: Path) -> dict[str, Any]:
    """Rebuild the accepted v6 report from immutable raw files; fail on missing evidence.

    Historical spend is the sealed plan's carried ledger (audited in the S6 receipt),
    not a second replay of prior rounds. This report does not alter that ledger.
    """
    root, evidence_root = repository_root.resolve(), evidence_root.resolve()
    names = {
        kind: f"{CONTROL_DIR}/postdebug-{kind}-v6.json"
        for kind in ("plan", "authorization", "report")
    }
    plan, auth, reported = (
        _sealed(evidence_root, names[k]) for k in ("plan", "authorization", "report")
    )
    for name in names.values():
        _require(
            contained_file(root, name).read_bytes()
            == contained_file(evidence_root, name).read_bytes(),
            "tracked_and_received_controls_differ",
        )
    _require(
        auth["plan_sha256"] == reported["plan_sha256"] == plan["content_sha256"],
        "plan_binding_mismatch",
    )
    _require(
        auth["plan_raw_sha256"] == _file_ref(evidence_root, names["plan"])["sha256"],
        "plan_raw_binding_mismatch",
    )
    _require(reported["authorization_sha256"] == auth["content_sha256"], "authorization_mismatch")
    _require(
        reported["source_commit"] == auth["frozen_source_commit"] == plan["frozen_source_commit"],
        "source_binding_mismatch",
    )
    _require(
        reported["report_version"] == "post-debug-rerun-report-v6"
        and reported["fresh_blind_holdout"] is False,
        "unsupported_report",
    )
    for name, digest in plan["input_files"].items():
        _require(_file_ref(root, name)["sha256"] == digest, "frozen_input_mismatch")
    dataset_name = "data/manifests/p7-v1/manifest.json"
    dataset = load_frozen_dataset(root / dataset_name, repository_root=root)
    _require(
        dataset.manifest.content_sha256 == plan["dataset_manifest_sha256"],
        "dataset_binding_mismatch",
    )
    gold = {task.task.task_id: task.gold_files for task in dataset.external_evaluation_tasks}
    slots = plan["run_slots"]
    validate_slot_inventory(slots, reported["outcomes"])
    _require(set(gold) == {s["task_id"] for s in slots}, "task_set_mismatch")
    by_key = {(r["arm_id"], r["task_id"]): r for r in reported["outcomes"]}
    rows: list[dict[str, Any]] = []
    metrics: list[SlotExecutionMetrics] = []
    traces: list[dict[str, Any]] = []
    task_traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    verified_artifacts = 0
    hybrid_traces: dict[str, AgentTrace] = {}
    for slot in slots:
        run_path = slot["output_path"]
        # Parent link checks must happen before the runner opens its own files.
        contained_file(evidence_root, f"{run_path}/manifest.json")
        directory = evidence_root / run_path
        _require(not (directory / "events").is_symlink(), "symlink_artifact")
        manifest = load_evaluation_run_manifest(directory)
        inspection = inspect_evaluation_run(directory)
        _require(
            manifest.arm_id == slot["arm_id"]
            and str(manifest.run_id) == slot["run_id"]
            and manifest.task_ids == (slot["task_id"],)
            and manifest.source_commit == plan["frozen_source_commit"]
            and manifest.configuration_sha256 == slot["configuration_sha256"]
            and manifest.dataset_manifest_sha256 == plan["dataset_manifest_sha256"]
            and manifest.dependency_lock_sha256 == plan["dependency_lock_sha256"],
            "run_manifest_binding_mismatch",
        )
        _require(len(inspection.attempts) == 1, "unexpected_repetition_count")
        attempt = inspection.attempts[0]
        # Accepted v6 has exactly one completed execution per slot. Do not silently
        # select the latest success or hide failed/retried executions.
        _require(
            attempt.status == "completed"
            and attempt.execution_count == 1
            and attempt.evidence is not None,
            "v6_execution_not_complete_once",
        )
        evidence = attempt.evidence
        assert evidence is not None
        expected_outcome = {
            "arm_id": slot["arm_id"],
            "task_id": slot["task_id"],
            "run_id": slot["run_id"],
            "output_path": run_path,
            "manifest_ref": _file_ref(evidence_root, f"{run_path}/manifest.json"),
            "attempt_status": "completed",
            "execution_count": 1,
            "run_status": evidence.status.value,
            "termination_reason": evidence.termination_reason.value,
            "task_success": evidence.task_success,
            "usage": evidence.usage.model_dump(mode="json"),
            "technical_failure_code": None,
            "reserved_unknown_cost_usd": "0",
        }
        _require(
            by_key[(slot["arm_id"], slot["task_id"])] == expected_outcome,
            "raw_and_reported_outcome_differ",
        )
        events = tuple(
            EvaluationAttemptEvent.model_validate_json(
                contained_file(directory, p.relative_to(directory).as_posix()).read_bytes()
            )
            for p in sorted((directory / "events").glob("*.json"))
        )
        _require([e.status.value for e in events] == ["started", "completed"], "unexpected_history")
        execution = contained_file(directory, f"{events[-1].output_path}/result.json").parent
        verified_artifacts += verify_execution_artifacts(execution)
        result = FinalResult.model_validate_json((execution / "result.json").read_bytes())
        trace_dict = _read_json(execution, "trace.json")
        trace: AgentTrace | SingleShotTrace
        if slot["arm_id"] == SINGLE_SHOT:
            trace = SingleShotTrace.model_validate(trace_dict)
            verify_single_shot_trace(trace, result)
            hybrid_slot = next(
                s
                for s in slots
                if s["arm_id"] == "hybrid-rrf-v1" and s["task_id"] == slot["task_id"]
            )
            hybrid_trace = hybrid_traces[slot["task_id"]]
            context = load_hybrid_initial_context(
                evidence_root / hybrid_slot["output_path"],
                hybrid_trace.events[0].state_before.task,
                expected_source_commit=plan["frozen_source_commit"],
                expected_configuration_sha256=hybrid_slot["configuration_sha256"],
            )
            _require(trace.context == context, "single_shot_context_mismatch")
        else:
            trace = AgentTrace.model_validate(trace_dict)
            verify_agent_result(trace, result)
            if slot["arm_id"] == "hybrid-rrf-v1":
                hybrid_traces[slot["task_id"]] = trace
        metric = extract_execution_metrics(
            arm_id=slot["arm_id"],
            task_id=slot["task_id"],
            attempt_status="completed",
            result=result,
            trace=trace,
            events=events,
        )
        metrics.append(metric)
        traces.append(trace_dict)
        task_traces[slot["task_id"]].append(trace_dict)
        rows.append(
            {
                "arm_id": slot["arm_id"],
                "task_id": slot["task_id"],
                "run_id": slot["run_id"],
                "execution": metric.model_dump(mode="json"),
                "gold_files": list(gold[slot["task_id"]]),
                "retained_initial_timing": {
                    "indexing_latency_ms": result.localization_metrics.indexing_latency_ms,
                    "retrieval_latency_ms": result.localization_metrics.retrieval_latency_ms,
                },
                "trace_ref": _file_ref(
                    evidence_root, f"{run_path}/{events[-1].output_path}/trace.json"
                ),
                "result_ref": _file_ref(
                    evidence_root, f"{run_path}/{events[-1].output_path}/result.json"
                ),
            }
        )
    maps = {task: retained_chunk_paths(items) for task, items in task_traces.items()}
    for row, trace_dict in zip(rows, traces, strict=True):
        row["localization"] = reconstruct_initial_localization(
            trace_dict,
            gold[row["task_id"]],
            chunk_paths=maps[row["task_id"]],
        )
        measured = row["arm_id"] not in {SINGLE_SHOT, "mini-swe-safe-v1"}
        row["localization"]["indexing_latency_ms"] = (
            row["retained_initial_timing"]["indexing_latency_ms"] if measured else None
        )
        row["localization"]["retrieval_latency_ms"] = (
            row["retained_initial_timing"]["retrieval_latency_ms"] if measured else None
        )
        row["localization"]["latency_availability"] = (
            "retained_runtime_timers" if measured else "no_independent_initial_retrieval"
        )
    arms = []
    for arm_id in dict.fromkeys(s["arm_id"] for s in slots):
        selected = [r for r in rows if r["arm_id"] == arm_id]
        arms.append(
            {
                "arm_id": arm_id,
                "execution": aggregate_execution_metrics(
                    [m for m in metrics if m.arm_id == arm_id]
                ).model_dump(mode="json"),
                "localization": summarize_localization([r["localization"] for r in selected]),
            }
        )
    overall = aggregate_execution_metrics(metrics).model_dump(mode="json")
    # Cross-check original aggregate counters/cost, independently of serialized rows.
    cost = sum((Decimal(r["usage"]["estimated_cost_usd"]) for r in by_key.values()), Decimal(0))
    _require(
        reported["slot_count"] == reported["completed_slots"] == len(slots)
        and reported["technical_failure_slots"] == 0
        and reported["task_success_count"] == sum(r["task_success"] for r in by_key.values())
        and Decimal(reported["known_cost_usd"]) == cost
        and Decimal(reported["reserved_unknown_cost_usd"]) == 0
        and Decimal(reported["carried_cost_usd"])
        == Decimal(plan["carried_cost_usd"])
        == Decimal(auth["carried_cost_usd"])
        and Decimal(reported["total_accounted_cost_usd"])
        == cost + Decimal(plan["carried_cost_usd"])
        and Decimal(reported["budget_usd"])
        == Decimal(plan["application_budget_usd"])
        == Decimal(auth["max_total_cost_usd"]),
        "aggregate_and_raw_mismatch",
    )
    result_report: dict[str, Any] = {
        "report_version": "evaluation-aggregate-v1",
        "experiment_id": plan["experiment_id"],
        "evaluation_label": reported["evaluation_label"],
        "fresh_blind_holdout": False,
        "source_commit": plan["frozen_source_commit"],
        "controller_commit": reported["controller_commit"],
        "distinct_task_count": len(gold),
        "planned_slots": len(slots),
        "missing_slots": [],
        "complete": True,
        "provider_requests_made_by_report": 0,
        "verified_artifact_references": verified_artifacts,
        "sources": {
            **{k: _file_ref(evidence_root, n) for k, n in names.items()},
            "dataset": _file_ref(root, dataset_name),
        },
        "overall": overall,
        "arms": arms,
        "slots": rows,
        "accounting": {
            k: reported[k]
            for k in (
                "known_cost_usd",
                "reserved_unknown_cost_usd",
                "carried_cost_usd",
                "total_accounted_cost_usd",
                "budget_usd",
            )
        },
        "historical_carry_basis": "sealed_plan_previously_audited_in_s6_not_replayed_here",
        "historical_unknown_reserve_usd": plan["prior_unknown_cost_usd"],
        "localization_definitions": METRIC_DEFINITIONS,
        "localization_aggregation_warning": (
            "Available-only means with n<8 are diagnostic subsets, not full-arm comparative scores."
        ),
        "initial_timing_scope": (
            "Indexing timer includes ingest, chunking, BM25 construction and/or cached embedding "
            "lookup and vector indexing. Retrieval timer covers the initial query. These retained "
            "runtime timers exclude earlier GPU cache construction; not cold embedding benchmarks."
        ),
        "limitations": list(LIMITATIONS),
    }
    result_report["content_sha256"] = canonical_sha256(result_report)
    return result_report
