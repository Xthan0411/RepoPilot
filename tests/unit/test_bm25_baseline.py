"""Gold isolation, baseline workflow, artifact, and hostile manifest tests."""

from __future__ import annotations

import json
import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from repopilot.config import RepoPilotSettings
from repopilot.enums import LocalizationEvalErrorCode
from repopilot.evals import (
    BM25BaselineTaskResult,
    LocalizationEvalError,
    build_localization_query,
    load_bm25_baseline_manifest,
    run_bm25_baseline,
    run_bm25_baseline_manifest,
)
from repopilot.providers import FakeProvider
from repopilot.sandbox import FakeSandbox

FIXTURES = Path(__file__).parents[1] / "fixtures"
MANIFEST_PATH = FIXTURES / "p2_bm25_baseline_manifest.json"
REPOSITORY_ROOT = FIXTURES / "repositories" / "read_only"
PATCH_CANARY = "EVALUATOR_ONLY_PATCH_CANARY"
ARTIFACT_NAMES = (
    "chunks.jsonl",
    "bm25-index.json",
    "tasks.jsonl",
    "tasks.csv",
    "summary.md",
    "baseline.json",
)


def clock(*values: int) -> Any:
    ticks: Iterator[int] = iter(values)
    return lambda: next(ticks)


def load_payload() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
    )


def write_payload(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def assert_eval_error(
    captured: pytest.ExceptionInfo[LocalizationEvalError],
    code: LocalizationEvalErrorCode,
) -> None:
    assert captured.value.code is code


def test_fixture_baseline_builds_shared_index_metrics_and_all_artifacts(
    tmp_path: Path,
) -> None:
    artifacts = run_bm25_baseline(
        MANIFEST_PATH,
        tmp_path / "baseline",
        RepoPilotSettings(),
        clock_ns=clock(
            0,
            5_000_000,
            10_000_000,
            11_000_000,
            20_000_000,
            22_000_000,
            30_000_000,
            33_000_000,
        ),
    )
    report = artifacts.report

    assert report.benchmark_id == "p2.synthetic-smoke"
    assert report.source_count == 5
    assert report.chunk_count == report.indexed_document_count == 10
    assert tuple(item.task_id for item in report.task_results) == (
        "baseline-001",
        "baseline-002",
        "baseline-003",
    )
    assert report.summary.model_dump() == {
        "task_count": 3,
        "tasks_with_relevant_file": 3,
        "mean_file_recall_at_1": 1.0,
        "mean_file_recall_at_3": 1.0,
        "mean_file_recall_at_5": 1.0,
        "mean_reciprocal_rank": 1.0,
        "indexing_latency_ms": 5,
        "total_retrieval_latency_ms": 6,
        "mean_retrieval_latency_ms": 2.0,
    }
    assert tuple((item.split.value, item.task_count) for item in report.split_counts) == (
        ("dev", 2),
        ("holdout", 1),
    )
    assert all(item.metrics.first_relevant_file_rank == 1 for item in report.task_results)
    assert all(item.missing_gold_files == () for item in report.task_results)
    assert not any(
        chunk.relative_path.startswith("_repopilot_issues/") for chunk in artifacts.corpus.chunks
    )
    assert (
        tuple(
            path.name
            for path in (
                artifacts.corpus_path,
                artifacts.index_path,
                artifacts.task_jsonl_path,
                artifacts.task_csv_path,
                artifacts.summary_path,
                artifacts.report_path,
            )
        )
        == ARTIFACT_NAMES
    )
    assert all((tmp_path / "baseline" / name).is_file() for name in ARTIFACT_NAMES)

    task_lines = artifacts.task_jsonl_path.read_text(encoding="utf-8").splitlines()
    assert tuple(
        BM25BaselineTaskResult.model_validate_json(line).task_id for line in task_lines
    ) == ("baseline-001", "baseline-002", "baseline-003")
    assert artifacts.task_csv_path.read_text(encoding="utf-8").startswith(
        "task_id,split,query,gold_files"
    )
    assert "| 1.000000 | 1.000000 | 1.000000 | 1.000000 | 3/3 |" in (
        artifacts.summary_path.read_text(encoding="utf-8")
    )


def test_query_and_artifacts_never_contain_expected_patch_canary(tmp_path: Path) -> None:
    manifest = load_bm25_baseline_manifest(MANIFEST_PATH)
    query = build_localization_query(manifest.tasks[0].task)

    assert query == (
        "Handle empty key value configuration\n"
        "The parse function in parser.py should return an empty mapping for empty "
        "configuration text."
    )
    assert PATCH_CANARY not in query
    assert "pkg/parser.py" not in query
    assert "parser.py" in query

    artifacts = run_bm25_baseline(
        MANIFEST_PATH,
        tmp_path,
        RepoPilotSettings(),
        clock_ns=clock(0, 1, 2, 3, 4, 5, 6, 7),
    )
    for name in ARTIFACT_NAMES:
        assert PATCH_CANARY not in tmp_path.joinpath(name).read_text(encoding="utf-8")
    assert PATCH_CANARY not in artifacts.report.model_dump_json()
    assert PATCH_CANARY not in repr(manifest)


def test_same_inputs_and_clock_produce_byte_identical_artifacts(tmp_path: Path) -> None:
    timings = (0, 5_000_000, 10, 1_000_010, 20, 2_000_020, 30, 3_000_030)

    run_bm25_baseline(
        MANIFEST_PATH,
        tmp_path / "first",
        RepoPilotSettings(),
        clock_ns=clock(*timings),
    )
    run_bm25_baseline(
        MANIFEST_PATH,
        tmp_path / "second",
        RepoPilotSettings(),
        clock_ns=clock(*timings),
    )

    for name in ARTIFACT_NAMES:
        assert (
            tmp_path.joinpath("first", name).read_bytes()
            == tmp_path.joinpath("second", name).read_bytes()
        )


def test_missing_gold_file_is_reported_and_scores_zero(tmp_path: Path) -> None:
    manifest = load_bm25_baseline_manifest(MANIFEST_PATH)
    evaluation = manifest.tasks[0].model_copy(update={"gold_files": ("not-indexed.py",)})
    single_task = manifest.model_copy(update={"tasks": (evaluation,)})

    artifacts = run_bm25_baseline_manifest(
        single_task,
        repository_root=REPOSITORY_ROOT,
        output_dir=tmp_path,
        settings=RepoPilotSettings(),
        clock_ns=clock(0, 1, 2, 3),
    )
    result = artifacts.report.task_results[0]

    assert result.missing_gold_files == ("not-indexed.py",)
    assert result.metrics.file_recall_at_1 == 0
    assert result.metrics.file_recall_at_3 == 0
    assert result.metrics.file_recall_at_5 == 0
    assert result.metrics.mean_reciprocal_rank == 0
    assert result.metrics.first_relevant_file_rank is None


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            lambda payload: payload.update({"schema_version": 2}),
            LocalizationEvalErrorCode.UNSUPPORTED_VERSION,
        ),
        (
            lambda payload: payload.update({"retrieval_depth": 4}),
            LocalizationEvalErrorCode.INVALID_MANIFEST,
        ),
        (
            lambda payload: payload["tasks"][1]["task"].update({"task_id": "baseline-001"}),
            LocalizationEvalErrorCode.INVALID_MANIFEST,
        ),
        (
            lambda payload: payload["tasks"].reverse(),
            LocalizationEvalErrorCode.INVALID_MANIFEST,
        ),
        (
            lambda payload: payload["tasks"][0]["task"].update({"base_commit": "b" * 40}),
            LocalizationEvalErrorCode.IDENTITY_MISMATCH,
        ),
        (
            lambda payload: payload["tasks"][0]["task"].update({"repository_path": "other"}),
            LocalizationEvalErrorCode.IDENTITY_MISMATCH,
        ),
        (
            lambda payload: payload["tasks"][0].update({"split": "train"}),
            LocalizationEvalErrorCode.INVALID_MANIFEST,
        ),
        (
            lambda payload: payload["tasks"][0].update({"gold_files": []}),
            LocalizationEvalErrorCode.INVALID_MANIFEST,
        ),
        (
            lambda payload: payload.update({"index_filename": "chunks.jsonl"}),
            LocalizationEvalErrorCode.INVALID_MANIFEST,
        ),
        (
            lambda payload: payload.update(
                {
                    "corpus_filename": "nested",
                    "index_filename": "nested/index.json",
                }
            ),
            LocalizationEvalErrorCode.INVALID_MANIFEST,
        ),
    ],
)
def test_loader_rejects_invalid_version_identity_order_and_artifacts(
    tmp_path: Path,
    mutation: Any,
    expected_code: LocalizationEvalErrorCode,
) -> None:
    payload = load_payload()
    mutation(payload)
    path = tmp_path / "manifest.json"
    write_payload(path, payload)

    with pytest.raises(LocalizationEvalError) as captured:
        load_bm25_baseline_manifest(path)

    assert_eval_error(captured, expected_code)


def test_loader_distinguishes_missing_malformed_and_non_utf8_input(
    tmp_path: Path,
) -> None:
    with pytest.raises(LocalizationEvalError) as captured:
        load_bm25_baseline_manifest(tmp_path / "missing.json")
    assert_eval_error(captured, LocalizationEvalErrorCode.INPUT_IO_ERROR)

    path = tmp_path / "manifest.json"
    for payload in (b"{", b"[]", b'{"tasks":[]}', b"\xff"):
        path.write_bytes(payload)
        with pytest.raises(LocalizationEvalError) as captured:
            load_bm25_baseline_manifest(path)
        assert_eval_error(captured, LocalizationEvalErrorCode.INVALID_MANIFEST)


def test_repository_symlink_escape_is_rejected_before_indexing(tmp_path: Path) -> None:
    payload = load_payload()
    payload["repository_root"] = "repo"
    for item in payload["tasks"]:
        item["task"]["repository_path"] = "repo"
    manifest_path = tmp_path / "manifest.json"
    write_payload(manifest_path, payload)
    tmp_path.joinpath("repo").symlink_to(REPOSITORY_ROOT, target_is_directory=True)

    with pytest.raises(LocalizationEvalError) as captured:
        run_bm25_baseline(
            manifest_path,
            tmp_path / "output",
            RepoPilotSettings(),
        )

    assert_eval_error(captured, LocalizationEvalErrorCode.PATH_VIOLATION)


def test_output_symlink_escape_and_occupied_directory_are_typed(
    tmp_path: Path,
) -> None:
    manifest = load_bm25_baseline_manifest(MANIFEST_PATH).model_copy(
        update={"summary_filename": "escape/summary.md"}
    )
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    output.joinpath("escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(LocalizationEvalError) as captured:
        run_bm25_baseline_manifest(
            manifest,
            repository_root=REPOSITORY_ROOT,
            output_dir=output,
            settings=RepoPilotSettings(),
            clock_ns=clock(0, 1, 2, 3, 4, 5, 6, 7),
        )
    assert_eval_error(captured, LocalizationEvalErrorCode.PATH_VIOLATION)

    occupied = tmp_path / "occupied"
    occupied.write_text("file", encoding="utf-8")
    with pytest.raises(LocalizationEvalError) as captured:
        run_bm25_baseline_manifest(
            load_bm25_baseline_manifest(MANIFEST_PATH),
            repository_root=REPOSITORY_ROOT,
            output_dir=occupied,
            settings=RepoPilotSettings(),
            clock_ns=clock(0, 1, 2, 3, 4, 5, 6, 7),
        )
    assert_eval_error(captured, LocalizationEvalErrorCode.OUTPUT_IO_ERROR)


def test_indexing_and_tokenless_query_fail_with_retrieval_error(
    tmp_path: Path,
) -> None:
    manifest = load_bm25_baseline_manifest(MANIFEST_PATH)
    with pytest.raises(LocalizationEvalError) as captured:
        run_bm25_baseline_manifest(
            manifest,
            repository_root=REPOSITORY_ROOT,
            output_dir=tmp_path / "truncated",
            settings=RepoPilotSettings(repository_max_list_entries=1),
            clock_ns=clock(0, 1),
        )
    assert_eval_error(captured, LocalizationEvalErrorCode.RETRIEVAL_FAILURE)

    task = manifest.tasks[0].task.model_copy(update={"issue_title": "!!!", "issue_body": "..."})
    evaluation = manifest.tasks[0].model_copy(update={"task": task})
    tokenless = manifest.model_copy(update={"tasks": (evaluation,)})
    with pytest.raises(LocalizationEvalError) as captured:
        run_bm25_baseline_manifest(
            tokenless,
            repository_root=REPOSITORY_ROOT,
            output_dir=tmp_path / "tokenless",
            settings=RepoPilotSettings(),
            clock_ns=clock(0, 1, 2),
        )
    assert_eval_error(captured, LocalizationEvalErrorCode.RETRIEVAL_FAILURE)


def test_baseline_never_calls_provider_sandbox_network_or_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline baseline attempted an external or execution capability")

    monkeypatch.setattr(FakeProvider, "generate", forbidden)
    monkeypatch.setattr(FakeSandbox, "prepare", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)

    artifacts = run_bm25_baseline(
        MANIFEST_PATH,
        tmp_path,
        RepoPilotSettings(),
        clock_ns=clock(0, 1, 2, 3, 4, 5, 6, 7),
    )

    assert artifacts.report.summary.task_count == 3
