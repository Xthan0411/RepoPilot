"""Fair three-mode comparison, rewrite parity, failures, and artifact tests."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest

from repopilot.enums import (
    ComparisonCaseOutcome,
    HybridRetrievalErrorCode,
    QueryRewriteDecision,
    RetrievalComparisonErrorCode,
    RetrievalMode,
    SymbolType,
)
from repopilot.evals import (
    RETRIEVAL_COMPARISON_MODES,
    RetrievalComparisonError,
    build_localization_query,
    evaluate_retrieval_comparison,
    load_bm25_baseline_manifest,
    write_retrieval_comparison,
)
from repopilot.retrieval import (
    ChunkCorpus,
    ChunkSourceSnapshot,
    HybridPolicy,
    HybridRetrievalError,
    HybridSearchResult,
    QueryRewriteOutcome,
    RetrievalFilter,
)
from repopilot.schemas import CodeChunk, RetrievalHit

FIXTURES = Path(__file__).parents[1] / "fixtures"
MANIFEST_PATH = FIXTURES / "p2_bm25_baseline_manifest.json"
PATCH_CANARY = "EVALUATOR_ONLY_PATCH_CANARY"
SOURCE_HASH = "a" * 64
BUILD_FINGERPRINT = "b" * 64


def clock(*values: int) -> Callable[[], int]:
    ticks: Iterator[int] = iter(values)
    return lambda: next(ticks)


def comparison_corpus() -> ChunkCorpus:
    manifest = load_bm25_baseline_manifest(MANIFEST_PATH)
    chunks = tuple(
        _chunk(chunk_id, path)
        for chunk_id, path in (
            ("chunk_init", "pkg/__init__.py"),
            ("chunk_parser", "pkg/parser.py"),
            ("chunk_service", "pkg/service.py"),
        )
    )
    return ChunkCorpus(
        repository_id=manifest.repository_id,
        commit=manifest.commit,
        sources=tuple(
            ChunkSourceSnapshot(
                source_id=str(chunk.metadata["source_id"]),
                relative_path=chunk.relative_path,
                source_content_hash=SOURCE_HASH,
                processor="fixture",
                build_fingerprint=BUILD_FINGERPRINT,
                chunk_ids=(chunk.chunk_id,),
            )
            for chunk in chunks
        ),
        chunks=chunks,
    )


def _chunk(chunk_id: str, path: str) -> CodeChunk:
    content = f"{path} implementation"
    return CodeChunk(
        chunk_id=chunk_id,
        repository_id="fixture.baseline",
        commit="a" * 40,
        relative_path=path,
        symbol_type=SymbolType.TEXT,
        start_line=1,
        end_line=1,
        language="python",
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        metadata={
            "source_id": f"source_{chunk_id}",
            "source_content_hash": SOURCE_HASH,
            "processor": "fixture",
        },
    )


class ScriptedComparisonRetriever:
    def __init__(
        self,
        mode: RetrievalMode,
        result_orders: Sequence[Sequence[str]],
        *,
        fail_at: frozenset[int] = frozenset(),
    ) -> None:
        self._policy = HybridPolicy(
            mode=mode,
            default_top_k=100,
            candidate_top_k=100,
        )
        self._orders = tuple(tuple(order) for order in result_orders)
        self._fail_at = fail_at
        self.calls: list[tuple[str, int | None]] = []

    @property
    def policy(self) -> HybridPolicy:
        return self._policy

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata_filter: RetrievalFilter | None = None,
    ) -> HybridSearchResult:
        call_index = len(self.calls)
        self.calls.append((query, top_k))
        if call_index in self._fail_at:
            raise HybridRetrievalError(HybridRetrievalErrorCode.DENSE_FAILURE)
        assert top_k is not None
        assert metadata_filter is None
        order = self._orders[call_index]
        hits = tuple(
            _hit(self.policy.mode, chunk_id, rank, query)
            for rank, chunk_id in enumerate(order, start=1)
        )
        dense = self.policy.mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID)
        sparse = self.policy.mode in (RetrievalMode.BM25, RetrievalMode.HYBRID)
        return HybridSearchResult(
            repository_id="fixture.baseline",
            commit="a" * 40,
            mode=self.policy.mode,
            query=query,
            top_k=top_k,
            candidate_top_k=top_k,
            rrf_k=60 if self.policy.mode is RetrievalMode.HYBRID else None,
            metadata_filter=RetrievalFilter(),
            eligible_chunk_count=3,
            sparse_candidate_count=len(hits) if sparse else 0,
            dense_candidate_count=len(hits) if dense else 0,
            provider_call_count=1 if dense else 0,
            retry_count=0,
            hits=hits,
        )


def _hit(
    mode: RetrievalMode,
    chunk_id: str,
    rank: int,
    query: str,
) -> RetrievalHit:
    sparse = mode in (RetrievalMode.BM25, RetrievalMode.HYBRID)
    dense = mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID)
    return RetrievalHit(
        chunk_id=chunk_id,
        sparse_rank=rank if sparse else None,
        sparse_score=float(10 - rank) if sparse else None,
        dense_rank=rank if dense else None,
        dense_score=1.0 - rank / 10 if dense else None,
        fused_score=1.0 - rank / 10 if mode is RetrievalMode.HYBRID else None,
        query=query,
        retrieval_reason=f"scripted {mode.value}",
    )


def retrievers(
    *,
    dense_failures: frozenset[int] = frozenset(),
) -> dict[RetrievalMode, ScriptedComparisonRetriever]:
    return {
        RetrievalMode.BM25: ScriptedComparisonRetriever(
            RetrievalMode.BM25,
            (
                ("chunk_service", "chunk_parser"),
                ("chunk_service",),
                ("chunk_init",),
            ),
        ),
        RetrievalMode.DENSE: ScriptedComparisonRetriever(
            RetrievalMode.DENSE,
            (
                ("chunk_parser",),
                ("chunk_parser", "chunk_service"),
                ("chunk_init",),
            ),
            fail_at=dense_failures,
        ),
        RetrievalMode.HYBRID: ScriptedComparisonRetriever(
            RetrievalMode.HYBRID,
            (
                ("chunk_parser",),
                ("chunk_parser", "chunk_service"),
                ("chunk_init",),
            ),
        ),
    }


def rewrite_for_first_task() -> QueryRewriteOutcome:
    manifest = load_bm25_baseline_manifest(MANIFEST_PATH)
    original = build_localization_query(manifest.tasks[0].task)
    implementation = "parse empty configuration mapping implementation"
    test = "test parse empty configuration mapping"
    return QueryRewriteOutcome(
        repository_id=manifest.repository_id,
        commit=manifest.commit,
        decision=QueryRewriteDecision.REWRITTEN,
        original_query=original,
        effective_query=f"{implementation}\n{test}",
        implementation_query=implementation,
        test_query=test,
        rationale="Separate implementation and test retrieval intent.",
        rewrite_count=1,
        provider_call_count=1,
        untrusted_context_count=1,
        request_id="c" * 64,
    )


def test_comparison_freezes_inputs_and_classifies_improvements_and_regressions(
    tmp_path: Path,
) -> None:
    manifest = load_bm25_baseline_manifest(MANIFEST_PATH)
    chunk_corpus = comparison_corpus()
    scripted = retrievers()
    rewrite = rewrite_for_first_task()
    report = evaluate_retrieval_comparison(
        manifest,
        corpus=chunk_corpus,
        retrievers=scripted,
        indexing_latency_ms={
            RetrievalMode.BM25: 2,
            RetrievalMode.DENSE: 7,
            RetrievalMode.HYBRID: 9,
        },
        rewrites={manifest.tasks[0].task.task_id: rewrite},
        clock_ns=clock(
            0,
            1_000_000,
            2_000_000,
            4_000_000,
            5_000_000,
            8_000_000,
            10_000_000,
            14_000_000,
            20_000_000,
            25_000_000,
            30_000_000,
            36_000_000,
            40_000_000,
            47_000_000,
            50_000_000,
            58_000_000,
            60_000_000,
            69_000_000,
        ),
    )

    assert tuple(mode.mode for mode in report.modes) == RETRIEVAL_COMPARISON_MODES
    assert report.rewrite_task_count == 1
    assert tuple(mode.summary.indexing_latency_ms for mode in report.modes) == (
        2,
        7,
        9,
    )
    assert tuple(mode.summary.total_retrieval_latency_ms for mode in report.modes) == (
        6,
        15,
        24,
    )
    assert report.modes[0].summary.provider_call_count == 0
    assert report.modes[1].summary.provider_call_count == 3
    assert report.modes[2].summary.provider_call_count == 3
    assert {case.outcome for case in report.cases} >= {
        ComparisonCaseOutcome.IMPROVED,
        ComparisonCaseOutcome.REGRESSED,
    }
    expected_queries = tuple(
        (rewrite.effective_query if index == 0 else build_localization_query(evaluation.task))
        for index, evaluation in enumerate(manifest.tasks)
    )
    for retriever in scripted.values():
        assert tuple(query for query, _ in retriever.calls) == expected_queries
        assert all(top_k == manifest.retrieval_depth for _, top_k in retriever.calls)

    artifacts = write_retrieval_comparison(report, tmp_path / "comparison")
    first_json = artifacts.json_path.read_text(encoding="utf-8")
    first_markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    write_retrieval_comparison(report, tmp_path / "comparison")
    assert artifacts.json_path.read_text(encoding="utf-8") == first_json
    assert artifacts.markdown_path.read_text(encoding="utf-8") == first_markdown
    assert "improved" in first_markdown
    assert "regressed" in first_markdown
    assert PATCH_CANARY not in first_json
    assert PATCH_CANARY not in first_markdown


def test_comparison_records_one_mode_failure_without_dropping_the_sample() -> None:
    manifest = load_bm25_baseline_manifest(MANIFEST_PATH)
    report = evaluate_retrieval_comparison(
        manifest,
        corpus=comparison_corpus(),
        retrievers=retrievers(dense_failures=frozenset({1})),
        indexing_latency_ms={mode: 0 for mode in RETRIEVAL_COMPARISON_MODES},
        clock_ns=clock(*range(18)),
    )

    dense = report.modes[1]
    assert dense.summary.task_count == 3
    assert dense.summary.successful_task_count == 2
    assert dense.summary.failed_task_count == 1
    assert dense.task_results[1].failure_code is (RetrievalComparisonErrorCode.RETRIEVAL_FAILURE)
    assert dense.task_results[1].ranked_files == ()
    assert dense.task_results[1].metrics.file_recall_at_5 == 0.0


def test_comparison_rejects_unknown_or_mismatched_rewrite_before_retrieval() -> None:
    manifest = load_bm25_baseline_manifest(MANIFEST_PATH)
    scripted = retrievers()
    invalid = rewrite_for_first_task().model_copy(update={"repository_id": "fixture.other"})

    with pytest.raises(RetrievalComparisonError) as captured:
        evaluate_retrieval_comparison(
            manifest,
            corpus=comparison_corpus(),
            retrievers=scripted,
            indexing_latency_ms={mode: 0 for mode in RETRIEVAL_COMPARISON_MODES},
            rewrites={manifest.tasks[0].task.task_id: invalid},
        )

    assert captured.value.code is RetrievalComparisonErrorCode.INVALID_REWRITE
    assert all(retriever.calls == [] for retriever in scripted.values())


def test_comparison_rejects_missing_mode_and_backwards_clock() -> None:
    manifest = load_bm25_baseline_manifest(MANIFEST_PATH)
    scripted = retrievers()
    scripted.pop(RetrievalMode.HYBRID)

    with pytest.raises(RetrievalComparisonError) as captured:
        evaluate_retrieval_comparison(
            manifest,
            corpus=comparison_corpus(),
            retrievers=scripted,
            indexing_latency_ms={mode: 0 for mode in RETRIEVAL_COMPARISON_MODES},
        )
    assert captured.value.code is RetrievalComparisonErrorCode.IDENTITY_MISMATCH

    with pytest.raises(RetrievalComparisonError) as captured:
        evaluate_retrieval_comparison(
            manifest,
            corpus=comparison_corpus(),
            retrievers=retrievers(),
            indexing_latency_ms={mode: 0 for mode in RETRIEVAL_COMPARISON_MODES},
            clock_ns=clock(2, 1),
        )
    assert captured.value.code is RetrievalComparisonErrorCode.INVALID_RESULT
