"""Offline localization reconstruction from retained traces, never runtime zero fields.

Inputs must already have passed the report's trace/integrity and frozen-Gold checks.
This module does no I/O, retrieval, model inference, or evaluation execution. Chunk
paths are joined before Gold; unresolved chunks stop the certifiable file prefix.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import cast

METRIC_DEFINITIONS = {
    "ranking": "retained chunk order, deduplicated by first occurrence of relative_path",
    "recall": "unique Gold files in first K deduplicated files / all frozen Gold files",
    "mrr": "reciprocal first relevant file rank; zero only for a fully resolved no-match list",
    "depth": "retained initial retrieval only, not an exhaustive repository ranking",
    "unresolved": "unresolved chunk stops certified prefix; non-identifiable metrics are null",
    "single_shot": "selected rendered Hybrid context, not a fresh raw retrieval ranking",
    "mini": "no precomputed retrieval: initial localization is not applicable, not zero",
    "tool_step": (
        "one-based position in all completed tool results; first successful nonempty "
        "read_file/search_code response for a Gold file, excluding initial context"
    ),
    "time": "read/search completion timestamps are not retained: elapsed_ms is unavailable",
    "aggregate": "macro mean over available task metrics; each denominator is explicit",
}
_METRICS = (
    "file_recall_at_1",
    "file_recall_at_3",
    "file_recall_at_5",
    "mean_reciprocal_rank",
)
_HEADERS = {
    "Top Dense source context (untrusted):\n",
    "Top BM25 source context (untrusted):\n",
    "Top Hybrid RRF source context (untrusted):\n",
}
_SUMMARY_HEADER = re.compile(r"(?P<path>[^\n:]+):[1-9][0-9]*-[1-9][0-9]* symbol=[^\n]+")
_REASON_PATH = re.compile(r"(?:^|; )path=([^;]+)(?=;|$)")


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("localization_expected_mapping")
    return cast(Mapping[str, object], value)


def _items(value: object) -> list[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("localization_expected_sequence")
    return list(value)


def _path(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or "\\" in value
        or any(ord(char) < 32 for char in value)
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("localization_invalid_relative_path")
    return value


def _initial(trace: Mapping[str, object]) -> tuple[Mapping[str, object], bool]:
    single_shot = trace.get("trace_version") == "single-shot-trace-v1"
    if single_shot:
        return _mapping(trace.get("context")), True
    events = _items(trace.get("events"))
    if not events:
        raise ValueError("localization_missing_initial_state")
    initial = _mapping(_mapping(events[0]).get("state_before"))
    if initial.get("tool_results") not in ([], ()):
        raise ValueError("localization_initial_state_has_tool_results")
    return initial, False


def _identity(trace: Mapping[str, object]) -> tuple[object, object]:
    initial, single_shot = _initial(trace)
    task = initial if single_shot else _mapping(initial.get("task"))
    identity = task.get("task_id"), task.get("base_commit")
    if not all(isinstance(value, str) and value for value in identity):
        raise ValueError("localization_missing_task_identity")
    if trace.get("task_id") != identity[0]:
        raise ValueError("localization_task_identity_mismatch")
    return identity


def _hits(initial: Mapping[str, object]) -> list[Mapping[str, object]]:
    hits = [_mapping(item) for item in _items(initial.get("retrieved_context"))]
    identifiers = [hit.get("chunk_id") for hit in hits]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("localization_invalid_chunk_identity")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("localization_duplicate_chunk_identity")
    return hits


def _add_path(paths: dict[str, str], chunk_id: str, path: str) -> None:
    path = _path(path)
    if chunk_id in paths and paths[chunk_id] != path:
        raise ValueError("localization_conflicting_chunk_path")
    paths[chunk_id] = path


def _summary_paths(initial: Mapping[str, object], hit_count: int) -> list[str]:
    """Decode only complete fixed-format Agent summaries, never clipped baseline text.

    Exactly min(5, hits) blocks and the frozen 800-character snippet bound are
    required. Ambiguous source-contained separator/header sequences fail closed.
    """
    summary = initial.get("repository_summary")
    if not isinstance(summary, str):
        raise ValueError("localization_invalid_repository_summary")
    prefix = next((item for item in _HEADERS if summary.startswith(item)), None)
    if prefix is None or not hit_count:
        return []
    blocks = summary[len(prefix) :].split("\n---\n")
    if len(blocks) != min(5, hit_count):
        return []
    paths: list[str] = []
    for block in blocks:
        header, separator, snippet = block.partition("\n")
        match = _SUMMARY_HEADER.fullmatch(header)
        if not separator or match is None or len(snippet) > 800:
            return []
        paths.append(_path(match.group("path")))
    return paths


def retained_chunk_paths(traces: Sequence[Mapping[str, object]]) -> dict[str, str]:
    """Join same-task/base-commit retained chunk IDs to paths without consulting Gold.

    Dense/BM25 metadata and complete Agent summary headers are independent retained
    evidence. A rendered single-shot summary is never interpreted as raw chunk order.
    Conflicting mappings or mixed task identities are errors, not silently resolved.
    """
    paths: dict[str, str] = {}
    identities = {_identity(trace) for trace in traces}
    if len(identities) > 1:
        raise ValueError("localization_mixed_task_identity")
    for trace in traces:
        initial, single_shot = _initial(trace)
        hits = _hits(initial)
        for hit in hits:
            reason = hit.get("retrieval_reason")
            matches = _REASON_PATH.findall(reason) if isinstance(reason, str) else []
            if len(matches) > 1:
                raise ValueError("localization_ambiguous_reason_path")
            if matches:
                _add_path(paths, cast(str, hit["chunk_id"]), matches[0])
        if not single_shot:
            for hit, path in zip(hits, _summary_paths(initial, len(hits)), strict=False):
                _add_path(paths, cast(str, hit["chunk_id"]), path)
    return paths


def _first_relevant_tool(
    trace: Mapping[str, object], gold: set[str], *, single_shot: bool
) -> dict[str, object]:
    result: dict[str, object] = {
        "status": "not_applicable" if single_shot else "not_observed",
        "tool_step": None,
        "tool_name": None,
        "relative_path": None,
        "elapsed_ms": None,
        "time_availability": "completion_timestamp_not_retained",
    }
    if single_shot:
        return result
    final = _mapping(trace.get("final_state"))
    seen_calls: set[str] = set()
    for index, item in enumerate(_items(final.get("tool_results")), 1):
        tool = _mapping(item)
        call_id = tool.get("call_id")
        if not isinstance(call_id, str) or not call_id or call_id in seen_calls:
            raise ValueError("localization_invalid_tool_call_identity")
        seen_calls.add(call_id)
        if tool.get("status") != "success":
            continue
        name = tool.get("tool_name")
        payload = _mapping(tool.get("payload"))
        candidates: list[str] = []
        if name == "read_file" and isinstance(payload.get("content"), str) and payload["content"]:
            candidates.append(_path(payload.get("relative_path")))
        elif name == "search_code":
            for match in _items(payload.get("matches", [])):
                match = _mapping(match)
                if isinstance(match.get("preview"), str) and match["preview"]:
                    candidates.append(_path(match.get("relative_path")))
        found = next((path for path in candidates if path in gold), None)
        if found is not None:
            result.update(status="observed", tool_step=index, tool_name=name, relative_path=found)
            return result
    return result


def reconstruct_initial_localization(
    trace: Mapping[str, object],
    gold_files: Sequence[str],
    *,
    chunk_paths: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Reconstruct identifiable metrics, distinguishing no hits from no retrieval.

    Gold is supplied by the offline evaluator and is never placed into a trace.
    A missing chunk cannot be dropped: its unknown file might shift all later ranks.
    """
    _identity(trace)
    if isinstance(gold_files, (str, bytes)):
        raise ValueError("localization_gold_must_be_nonempty_and_unique")
    gold = {_path(path) for path in gold_files}
    if not gold or len(gold) != len(gold_files):
        raise ValueError("localization_gold_must_be_nonempty_and_unique")
    initial, single_shot = _initial(trace)
    hits = _hits(initial)
    paths = retained_chunk_paths((trace,))
    for chunk_id, path in (chunk_paths or {}).items():
        _add_path(paths, chunk_id, path)
    summary = initial.get("repository_summary")
    no_retrieval = (
        not single_shot
        and isinstance(summary, str)
        and summary.startswith("No precomputed repository localization is available;")
    )
    if no_retrieval and hits:
        raise ValueError("localization_disabled_retrieval_has_hits")
    unresolved = [cast(str, hit["chunk_id"]) for hit in hits if hit["chunk_id"] not in paths]
    prefix: list[str] = []
    for hit in hits:
        resolved_path = paths.get(cast(str, hit["chunk_id"]))
        if resolved_path is None:
            break
        if resolved_path not in prefix:
            prefix.append(resolved_path)
    complete = not unresolved and not no_retrieval
    first_rank = next((index for index, path in enumerate(prefix, 1) if path in gold), None)
    metrics: dict[str, object] = {}
    for limit in (1, 3, 5):
        identifiable = complete or len(prefix) >= limit or gold.issubset(prefix)
        metrics[f"file_recall_at_{limit}"] = (
            len(set(prefix[:limit]) & gold) / len(gold)
            if identifiable and not no_retrieval
            else None
        )
    metrics["mean_reciprocal_rank"] = (
        1.0 / first_rank if first_rank else (0.0 if complete else None)
    )
    metrics["first_relevant_file_rank"] = first_rank
    status = "available" if complete else "partial"
    reason = None if complete else "unresolved_chunk_paths"
    if no_retrieval:
        status, reason = "not_applicable", "no_precomputed_retrieval"
    elif unresolved and not prefix:
        status = "unavailable"
    return {
        "status": status,
        "basis": "rendered_hybrid_initial_context" if single_shot else "raw_initial_retrieval",
        "reason": reason,
        "retrieved_chunk_count": len(hits),
        "resolved_chunk_count": len(hits) - len(unresolved),
        "unresolved_chunk_ids": unresolved,
        "ranked_file_prefix": prefix,
        "ranking_complete": complete,
        "gold_file_count": len(gold),
        "metrics": metrics,
        "first_relevant_repository_tool": _first_relevant_tool(
            trace, gold, single_shot=single_shot
        ),
    }


def summarize_localization(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Macro-average available metrics and expose each denominator and N/A count."""
    summaries: dict[str, object] = {}
    for key in _METRICS:
        values: list[float] = []
        for row in rows:
            value = _mapping(row.get("metrics")).get(key)
            if value is None:
                continue
            if type(value) not in (int, float) or not 0 <= cast(float, value) <= 1:
                raise ValueError("localization_invalid_metric_value")
            values.append(float(cast(float, value)))
        summaries[key] = {
            "mean": sum(values) / len(values) if values else None,
            "available_task_count": len(values),
            "unavailable_task_count": len(rows) - len(values),
        }
    timings: dict[str, object] = {}
    for key in ("indexing_latency_ms", "retrieval_latency_ms"):
        samples = [row.get(key) for row in rows if row.get(key) is not None]
        if any(type(value) is not int or value < 0 for value in samples):
            raise ValueError("localization_invalid_timing")
        timings[key] = {
            "mean": sum(cast(int, value) for value in samples) / len(samples) if samples else None,
            "available_task_count": len(samples),
        }
    tools = [
        _mapping(row.get("first_relevant_repository_tool", {"status": "unavailable"}))
        for row in rows
    ]
    observed = [cast(int, item["tool_step"]) for item in tools if item["status"] == "observed"]
    return {
        "task_count": len(rows),
        "metrics": summaries,
        "timing": timings,
        "first_relevant_repository_tool": {
            "observed_task_count": len(observed),
            "not_observed_task_count": sum(item["status"] == "not_observed" for item in tools),
            "not_applicable_task_count": sum(item["status"] == "not_applicable" for item in tools),
            "mean_observed_tool_steps": sum(observed) / len(observed) if observed else None,
            "elapsed_ms": None,
        },
    }
