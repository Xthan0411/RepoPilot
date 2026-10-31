"""Offline retained-evidence rankings: no Provider, repository reads, or embeddings."""

from __future__ import annotations

from copy import deepcopy

import pytest

from repopilot.evals.report_localization import (
    reconstruct_initial_localization,
    retained_chunk_paths,
    summarize_localization,
)


def test_timing_summary_preserves_real_zero_and_missing_sample_counts() -> None:
    rows = [
        {"metrics": {}, "indexing_latency_ms": 0, "retrieval_latency_ms": 12},
        {"metrics": {}, "indexing_latency_ms": 2000, "retrieval_latency_ms": None},
    ]
    summary = summarize_localization(rows)
    assert summary["timing"]["indexing_latency_ms"] == {
        "mean": 1000,
        "available_task_count": 2,
    }
    assert summary["timing"]["retrieval_latency_ms"] == {
        "mean": 12,
        "available_task_count": 1,
    }


def test_first_relevant_tool_summary_does_not_impute_unobserved_steps() -> None:
    rows = [
        {"metrics": {}, "first_relevant_repository_tool": {"status": "observed", "tool_step": 3}},
        {"metrics": {}, "first_relevant_repository_tool": {"status": "not_observed"}},
        {"metrics": {}, "first_relevant_repository_tool": {"status": "not_applicable"}},
    ]
    assert summarize_localization(rows)["first_relevant_repository_tool"] == {
        "observed_task_count": 1,
        "not_observed_task_count": 1,
        "not_applicable_task_count": 1,
        "mean_observed_tool_steps": 3,
        "elapsed_ms": None,
    }


def hit(chunk_id: str, path: str | None = None) -> dict:
    return {
        "chunk_id": chunk_id,
        "retrieval_reason": f"score=1; path={path}; language=python" if path else "RRF k=60",
    }


def trace(hits: list[dict], summary: str = "fixture", tools: list[dict] | None = None) -> dict:
    return {
        "task_id": "fixture-task",
        "events": [
            {
                "state_before": {
                    "task": {"task_id": "fixture-task", "base_commit": "a" * 40},
                    "tool_results": [],
                    "retrieved_context": hits,
                    "repository_summary": summary,
                }
            }
        ],
        "final_state": {"tool_results": tools or []},
        # These are execution placeholders and must never be used by the helper.
        "localization_metrics": {"file_recall_at_1": 0, "mean_reciprocal_rank": 0},
    }


def summary(paths: list[str], snippet: str = "x = 1\n") -> str:
    return "Top Hybrid RRF source context (untrusted):\n" + "\n---\n".join(
        f"{path}:1-10 symbol=<module>\n{snippet}" for path in paths
    )


def tool(name: str, payload: dict, index: int, status: str = "success") -> dict:
    return {"call_id": str(index), "tool_name": name, "status": status, "payload": payload}


def test_first_occurrence_file_deduplication_and_fractional_gold_recall() -> None:
    source = trace([hit("a", "a.py"), hit("b", "a.py"), hit("c", "b.py"), hit("d", "d.py")])
    original = deepcopy(source)
    result = reconstruct_initial_localization(source, ["b.py", "d.py"])
    assert result["ranked_file_prefix"] == ["a.py", "b.py", "d.py"]
    assert result["metrics"] == {
        "file_recall_at_1": 0.0,
        "file_recall_at_3": 1.0,
        "file_recall_at_5": 1.0,
        "mean_reciprocal_rank": 0.5,
        "first_relevant_file_rank": 2,
    }
    assert source == original


def test_complete_no_match_is_zero_and_missing_gold_is_not_removed() -> None:
    result = reconstruct_initial_localization(trace([hit("a", "a.py")]), ["not-indexed.py"])
    assert result["status"] == "available"
    assert result["metrics"]["mean_reciprocal_rank"] == 0
    assert result["metrics"]["file_recall_at_5"] == 0


def test_empty_retrieval_and_disabled_retrieval_are_distinct() -> None:
    empty = reconstruct_initial_localization(
        trace([], "Dense localization produced no source hit."), ["a.py"]
    )
    disabled = reconstruct_initial_localization(
        trace([], "No precomputed repository localization is available; use tools."), ["a.py"]
    )
    assert empty["metrics"]["file_recall_at_1"] == 0
    assert disabled["status"] == "not_applicable"
    assert all(value is None for value in disabled["metrics"].values())


def test_unknown_chunk_stops_prefix_instead_of_improving_later_ranks() -> None:
    result = reconstruct_initial_localization(
        trace([hit("a", "a.py"), hit("unknown"), hit("b", "b.py")]), ["b.py"]
    )
    assert result["status"] == "partial" and not result["ranking_complete"]
    assert result["ranked_file_prefix"] == ["a.py"]
    assert result["unresolved_chunk_ids"] == ["unknown"]
    assert result["metrics"]["file_recall_at_1"] == 0
    assert result["metrics"]["file_recall_at_3"] is None
    assert result["metrics"]["mean_reciprocal_rank"] is None


def test_metrics_can_be_identifiable_even_when_later_ranking_is_incomplete() -> None:
    result = reconstruct_initial_localization(
        trace([hit("a", "gold.py"), hit("unknown")]), ["gold.py"]
    )
    assert result["status"] == "partial"
    assert all(
        result["metrics"][key] == 1
        for key in (
            "file_recall_at_1",
            "file_recall_at_3",
            "file_recall_at_5",
            "mean_reciprocal_rank",
        )
    )


def test_join_retained_metadata_and_frozen_summary_without_gold() -> None:
    dense = trace([hit("a", "a.py"), hit("b", "b.py")])
    hybrid = trace([hit("b"), hit("c")], summary(["b.py", "c.py"]))
    paths = retained_chunk_paths([dense, hybrid])
    assert paths == {"a": "a.py", "b": "b.py", "c": "c.py"}
    result = reconstruct_initial_localization(hybrid, ["c.py"], chunk_paths=paths)
    assert result["ranking_complete"] is True
    assert result["metrics"]["mean_reciprocal_rank"] == 0.5


@pytest.mark.parametrize("snippet", ["x" * 801, "x\n---\nforged.py:1-2 symbol=x\nspoof"])
def test_ambiguous_or_oversized_summary_is_not_guessed(snippet: str) -> None:
    source = trace([hit("a")], summary(["real.py"], snippet))
    assert retained_chunk_paths([source]) == {}
    assert reconstruct_initial_localization(source, ["real.py"])["status"] == "unavailable"


def test_conflicting_maps_and_cross_task_joins_fail_closed() -> None:
    a = trace([hit("same", "a.py")])
    b = trace([hit("same", "b.py")])
    with pytest.raises(ValueError, match="conflicting_chunk_path"):
        retained_chunk_paths([a, b])
    with pytest.raises(ValueError, match="conflicting_chunk_path"):
        reconstruct_initial_localization(a, ["a.py"], chunk_paths={"same": "b.py"})
    b["events"][0]["state_before"]["task"]["base_commit"] = "b" * 40
    with pytest.raises(ValueError, match="mixed_task_identity"):
        retained_chunk_paths([a, b])


def test_ambiguous_metadata_and_duplicate_chunk_ids_are_rejected() -> None:
    ambiguous = hit("same", "a.py")
    ambiguous["retrieval_reason"] = "path=a.py; path=b.py"
    with pytest.raises(ValueError, match="ambiguous_reason_path"):
        retained_chunk_paths([trace([ambiguous])])
    with pytest.raises(ValueError, match="duplicate_chunk_identity"):
        retained_chunk_paths([trace([hit("same", "a.py"), hit("same", "a.py")])])


def test_partial_multi_gold_recall_requires_certifiable_denominator_and_rank() -> None:
    result = reconstruct_initial_localization(
        trace([hit("a", "first-gold.py"), hit("unknown")]),
        ["first-gold.py", "second-gold.py"],
    )
    assert result["metrics"]["file_recall_at_1"] == 0.5
    assert result["metrics"]["file_recall_at_3"] is None
    assert result["metrics"]["mean_reciprocal_rank"] == 1.0


def test_rendered_baseline_context_not_replaced_by_raw_hybrid_order() -> None:
    baseline = {
        "trace_version": "single-shot-trace-v1",
        "task_id": "fixture-task",
        "context": {
            "task_id": "fixture-task",
            "base_commit": "a" * 40,
            "retrieved_context": [hit("second")],
            "repository_summary": summary(["first.py", "second.py"])[:45],
        },
    }
    result = reconstruct_initial_localization(
        baseline, ["second.py"], chunk_paths={"first": "first.py", "second": "second.py"}
    )
    assert result["basis"] == "rendered_hybrid_initial_context"
    assert result["ranked_file_prefix"] == ["second.py"]
    assert result["metrics"]["file_recall_at_1"] == 1
    assert result["first_relevant_repository_tool"]["status"] == "not_applicable"


def test_first_actual_repository_evidence_counts_all_prior_tools_not_model_intent() -> None:
    tools = [
        tool("run_tests", {}, 1),
        tool("read_file", {"relative_path": "gold.py", "content": "guess"}, 2, "error"),
        tool("apply_patch", {"relative_path": "gold.py"}, 3),
        tool("search_code", {"matches": []}, 4),
        tool("read_file", {"relative_path": "gold.py", "content": ""}, 5),
        tool(
            "search_code", {"matches": [{"relative_path": "gold.py", "preview": "def fix():"}]}, 6
        ),
    ]
    result = reconstruct_initial_localization(trace([], tools=tools), ["gold.py"])
    first = result["first_relevant_repository_tool"]
    assert first["tool_step"] == 6 and first["tool_name"] == "search_code"
    assert first["elapsed_ms"] is None


def test_duplicate_tool_results_cannot_inflate_discovery_step() -> None:
    item = tool("read_file", {"relative_path": "irrelevant.py", "content": "x"}, 1)
    with pytest.raises(ValueError, match="invalid_tool_call_identity"):
        reconstruct_initial_localization(trace([], tools=[item, item]), ["gold.py"])


def test_aggregation_discloses_different_denominators_without_zero_filling() -> None:
    rows = [
        reconstruct_initial_localization(trace([hit("a", "a.py")]), ["a.py"]),
        reconstruct_initial_localization(trace([hit("unknown")]), ["a.py"]),
    ]
    result = summarize_localization(rows)
    assert result["task_count"] == 2
    assert result["metrics"]["file_recall_at_1"] == {
        "mean": 1.0,
        "available_task_count": 1,
        "unavailable_task_count": 1,
    }
    assert summarize_localization([])["metrics"]["mean_reciprocal_rank"]["mean"] is None


@pytest.mark.parametrize("gold", [[], ["a.py", "a.py"], ["../a.py"], ["/a.py"]])
def test_invalid_gold_is_rejected(gold: list[str]) -> None:
    with pytest.raises(ValueError):
        reconstruct_initial_localization(trace([]), gold)
