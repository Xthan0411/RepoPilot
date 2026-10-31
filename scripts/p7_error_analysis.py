"""Build the complete S8 failure inventory offline; no model or sandbox execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from repopilot.evals.reports import (
    ReportIntegrityError,
    build_v6_report,
    canonical_sha256,
    contained_file,
)

AGGREGATE_PATH = "results/p7-holdout/aggregate-v6.json"
OUTPUT_NAME = "error-analysis-v6.json"
SINGLE_SHOT = "single-shot-hybrid-context-v1"


def serialize(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportIntegrityError(message)


def _ref(root: Path, name: str) -> dict[str, Any]:
    raw = contained_file(root, name).read_bytes()
    return {
        "relative_path": name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "media_type": "application/json",
    }


def _agent_facts(trace: dict[str, Any]) -> dict[str, Any]:
    providers, patches, tests, replans, failures = [], [], [], [], []
    executed = {
        event["tool_result"]["call_id"]: event["sequence"]
        for event in trace["events"]
        if event.get("tool_result") is not None
    }
    for event in trace["events"]:
        sequence = event["sequence"]
        before, after = event["state_before"], event["state_after"]
        request, response = event.get("provider_request"), event.get("provider_response")
        if request is not None:
            window = request.get("context_window")
            action = response["action"] if response else None
            proposed_tool = action.get("tool_call") if action else None
            providers.append(
                {
                    "sequence": sequence,
                    "request_id": request["request_id"],
                    "response_present": response is not None,
                    "usage": response["usage"] if response else None,
                    "attempts": response["attempts"] if response else None,
                    "action_kind": action["kind"] if action else None,
                    "active_step_before": before["active_step_id"],
                    "patch_revision_before": before["patch_state"]["revision"],
                    "proposed_tool": None
                    if proposed_tool is None
                    else {
                        "call_id": proposed_tool["call_id"],
                        "tool_name": proposed_tool["tool_name"],
                        "arguments_sha256": canonical_sha256(proposed_tool["validated_arguments"]),
                        "execution_event_sequence": executed.get(proposed_tool["call_id"]),
                    },
                    "error_type": event.get("provider_error_type"),
                    "error_code": event.get("provider_error_code"),
                    "context_usage": event.get("context_usage"),
                    "context_window": None
                    if window is None
                    else {
                        key: window[key]
                        for key in (
                            "max_prompt_chars",
                            "context_chars_before",
                            "context_chars_after",
                            "estimated_tokens_before",
                            "estimated_tokens_after",
                            "rendered_prompt_chars",
                            "rendered_prompt_estimated_tokens",
                            "rendered_prompt_sha256",
                            "fits_character_budget",
                        )
                    },
                    "context_transform_count": len(window["transforms"]) if window else None,
                    "context_exclusion_count": len(window["exclusions"]) if window else None,
                }
            )
        tool = event.get("tool_result")
        if tool is not None:
            common = {
                "sequence": sequence,
                "call_id": tool["call_id"],
                "tool_name": tool["tool_name"],
                "status": tool["status"],
                "error_code": tool.get("error_code"),
                "repository_error_code": tool.get("repository_error_code"),
            }
            payload = tool.get("payload") or {}
            if tool["tool_name"] == "apply_patch":
                patches.append(
                    {
                        **common,
                        "patch_status": payload.get("status"),
                        "patch_error_code": payload.get("error_code"),
                        "patch_state": payload.get("patch_state"),
                        "summary": payload.get("summary"),
                    }
                )
            elif tool["tool_name"] == "run_tests":
                tests.append({**common, "test_run": payload.get("test_run")})
        if after["replan_count"] > before["replan_count"]:
            replans.append(
                {
                    "sequence": sequence,
                    "before_count": before["replan_count"],
                    "after_count": after["replan_count"],
                    "before_step": before["active_step_id"],
                    "after_step": after["active_step_id"],
                }
            )
        previous = before["failure_history"]
        current = after["failure_history"]
        for failure in current[len(previous) :] if current[: len(previous)] == previous else []:
            failures.append({"sequence": sequence, "recorded_failure": failure})
    final = trace["final_state"]
    return {
        "trace_type": "agent",
        "terminal_event_sequence": trace["events"][-1]["sequence"],
        "state_termination_reason": final["termination_reason"],
        "budget": final["budget"],
        "provider_limits": {
            key: trace["settings"][key]
            for key in (
                "model",
                "openai_reasoning_effort",
                "openai_max_output_tokens",
                "openai_prompt_max_chars",
                "max_tokens",
                "max_steps",
                "max_replans",
                "timeout_seconds",
                "max_cost_usd",
            )
        },
        "provider_events": providers,
        "patch_events": patches,
        "test_events": tests,
        "replan_events": replans,
        "failure_events": failures,
        "final_failure_history": final["failure_history"],
        "decision_count": final["decision_count"],
        "tool_result_count": len(final["tool_results"]),
        "final_patch_state": final["patch_state"],
        "baseline_test_result": final["baseline_test_result"],
        "final_test_state": final["test_state"],
    }


def _single_shot_facts(trace: dict[str, Any]) -> dict[str, Any]:
    candidate = trace["candidate_diff"]
    patch = trace["patch"]
    return {
        "trace_type": "single_shot",
        "provider_requests": trace["provider_requests"],
        "provider_retries": trace["provider_retries"],
        "provider_status": trace["provider_status"],
        "provider_failure": trace["provider_failure"],
        "provider_usage": trace["provider_usage"],
        "request_sha256": trace["request_sha256"],
        "context_sha256": trace["context"]["initial_context_sha256"],
        "limits": trace["limits"],
        "elapsed_before_evaluator_ms": trace["elapsed_before_evaluator_ms"],
        "elapsed_total_ms": trace["elapsed_total_ms"],
        "evaluation_stop": trace["evaluation_stop"],
        "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest()
        if candidate is not None
        else None,
        "candidate_size_bytes": len(candidate.encode()) if candidate is not None else None,
        "candidate_has_git_header": candidate.startswith("diff --git ")
        if candidate is not None
        else None,
        "patch_status": patch["status"] if patch else None,
        "patch_error_code": patch["error_code"] if patch else None,
        "patch_summary": patch["summary"] if patch else None,
        "patch_state": patch["patch_state"] if patch else None,
        "baseline_test_run": trace["baseline"]["test_run"] if trace["baseline"] else None,
        "validation_test_run": trace["validation"]["test_run"] if trace["validation"] else None,
        "execution_order": trace["execution_order"],
        "format_contract_caveat": (
            "Single-shot ordinary-unified-diff wording is weaker than Agent canonical-git-style "
            "tool schema; do not attribute this comparison to architecture alone."
        ),
    }


def extract_case(
    slot: dict[str, Any], trace: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Extract observable facts only from an already verified unsuccessful slot."""
    _require(slot["execution"]["task_success"] is False, "case_is_not_failed_slot")
    _require(result["task_metrics"]["task_success"] is False, "case_result_not_failed")
    _require(slot["task_id"] == trace["task_id"] == result["task_id"], "case_task_mismatch")
    _require(trace["run_id"] == result["run_id"], "case_execution_identity_mismatch")
    _require(
        slot["execution"]["termination_reason"] == result["termination_reason"],
        "case_termination_mismatch",
    )
    single = slot["arm_id"] == SINGLE_SHOT
    _require(
        single == (trace.get("trace_version") == "single-shot-trace-v1"), "case_trace_type_mismatch"
    )
    return {
        "arm_id": slot["arm_id"],
        "task_id": slot["task_id"],
        "run_id": slot["run_id"],
        "execution_run_id": result["run_id"],
        "trace_ref": slot["trace_ref"],
        "result_ref": slot["result_ref"],
        "observed_result": {
            key: result[key]
            for key in (
                "status",
                "termination_reason",
                "error_category",
                "verifier_error_code",
                "task_metrics",
                "usage",
            )
        },
        "root_cause": "not_inferred",
        "trace_facts": _single_shot_facts(trace) if single else _agent_facts(trace),
    }


def build_analysis(*, repository_root: Path, evidence_root: Path) -> dict[str, Any]:
    root, evidence = repository_root.resolve(), evidence_root.resolve()
    aggregate = build_v6_report(repository_root=root, evidence_root=evidence)
    _require(
        contained_file(root, AGGREGATE_PATH).read_bytes() == serialize(aggregate).encode(),
        "saved_aggregate_does_not_match_raw",
    )
    failures = [row for row in aggregate["slots"] if row["execution"]["task_success"] is False]
    cases = []
    for slot in failures:
        documents = []
        for key in ("trace_ref", "result_ref"):
            ref = slot[key]
            _require(_ref(evidence, ref["relative_path"]) == ref, "case_artifact_changed")
            documents.append(
                json.loads(contained_file(evidence, ref["relative_path"]).read_bytes())
            )
        cases.append(extract_case(slot, documents[0], documents[1]))
    _require(len({(c["arm_id"], c["task_id"]) for c in cases}) == len(cases), "duplicate_case")
    _require(
        len(cases) == aggregate["overall"]["completed_unsuccessful_slots"],
        "failure_inventory_incomplete",
    )
    result: dict[str, Any] = {
        "analysis_version": "evaluation-error-analysis-v1",
        "experiment_id": aggregate["experiment_id"],
        "source_commit": aggregate["source_commit"],
        "aggregate_ref": _ref(root, AGGREGATE_PATH),
        "aggregate_content_sha256": aggregate["content_sha256"],
        "planned_slot_count": aggregate["planned_slots"],
        "completed_slot_count": aggregate["overall"]["completed_slots"],
        "failed_slot_count": len(cases),
        "missing_failed_slots": [],
        "termination_counts": dict(
            sorted(Counter(case["observed_result"]["termination_reason"] for case in cases).items())
        ),
        "recorded_error_category_counts": dict(
            sorted(
                Counter(
                    case["observed_result"]["error_category"] or "null" for case in cases
                ).items()
            )
        ),
        "root_cause_policy": (
            "not_inferred: terminal reasons, schema error categories, and failure-history labels "
            "are recorded outcomes, not proof of causal attribution."
        ),
        "event_sequence_indexing": "zero_based_trace_event_sequence",
        "cases": cases,
        "provider_requests_made_by_analysis": 0,
        "sandbox_test_runs_by_analysis": 0,
        "canonical_patch_reconstruction_performed": False,
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def write_or_check(path: Path, analysis: dict[str, Any], *, check: bool) -> None:
    """Retain prior derived files; differing content requires an explicit new path."""
    content = serialize(analysis).encode()
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "analysis_output_symlink")
    if check:
        _require(path.is_file() and path.read_bytes() == content, "analysis_output_mismatch")
        return
    if path.exists():
        _require(
            path.is_file() and path.read_bytes() == content,
            "analysis_output_exists_with_different_content",
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.repository_root.resolve()
    analysis = build_analysis(repository_root=root, evidence_root=args.evidence_root)
    output = args.output or root / "results/p7-holdout" / OUTPUT_NAME
    write_or_check(output, analysis, check=args.check)
    print(
        json.dumps(
            {
                "analysis_verified": True,
                "failed_slots": analysis["failed_slot_count"],
                "content_sha256": analysis["content_sha256"],
                "check_only": args.check,
                "provider_requests": 0,
                "sandbox_test_runs": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
