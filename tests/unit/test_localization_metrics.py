"""File deduplication, Recall@K, MRR, and aggregate metric tests."""

from __future__ import annotations

import hashlib

import pytest

from repopilot.embeddings import EmbeddingIdentity
from repopilot.enums import EmbeddingProviderKind, SymbolType
from repopilot.evals.metrics import (
    DenseRankedFile,
    RankedFile,
    aggregate_localization_metrics,
    rank_dense_retrieved_files,
    rank_retrieved_files,
    score_file_localization,
)
from repopilot.retrieval import BM25SearchResult, ChunkCorpus, DenseSearchResult
from repopilot.schemas import CodeChunk, RetrievalHit
from repopilot.storage import VectorNamespace

REPOSITORY_ID = "fixture.metrics"
COMMIT = "c" * 40


def dense_identity() -> EmbeddingIdentity:
    return EmbeddingIdentity(
        provider=EmbeddingProviderKind.FAKE,
        model="fixture-dense",
        revision="v1",
        dimension=3,
        normalize=True,
        max_length=128,
    )


def chunk(chunk_id: str, path: str) -> CodeChunk:
    content = f"{path}\n"
    return CodeChunk(
        chunk_id=chunk_id,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        relative_path=path,
        symbol_type=SymbolType.TEXT,
        start_line=1,
        end_line=1,
        language="text",
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def corpus(*chunks: CodeChunk) -> ChunkCorpus:
    return ChunkCorpus.model_construct(
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        sources=(),
        chunks=chunks,
    )


def result(*chunk_ids: str) -> BM25SearchResult:
    return BM25SearchResult(
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        query="parser",
        top_k=max(1, len(chunk_ids)),
        hits=tuple(
            RetrievalHit(
                chunk_id=chunk_id,
                sparse_rank=rank,
                sparse_score=float(len(chunk_ids) - rank + 1),
                query="parser",
                retrieval_reason="fixture",
            )
            for rank, chunk_id in enumerate(chunk_ids, start=1)
        ),
    )


def dense_result(*chunk_ids: str, repository_id: str = REPOSITORY_ID) -> DenseSearchResult:
    identity = dense_identity()
    namespace = VectorNamespace(
        repository_id=repository_id,
        commit=COMMIT,
        embedding_model=identity.model,
        embedding_version=identity.fingerprint,
        dimension=identity.dimension,
    )
    return DenseSearchResult(
        repository_id=repository_id,
        commit=COMMIT,
        identity=identity,
        namespace=namespace,
        query="parser",
        top_k=max(1, len(chunk_ids)),
        hits=tuple(
            RetrievalHit(
                chunk_id=chunk_id,
                dense_rank=rank,
                dense_score=1.0 - (rank - 1) * 0.1,
                query="parser",
                retrieval_reason="fixture",
            )
            for rank, chunk_id in enumerate(chunk_ids, start=1)
        ),
        provider_call_count=1,
        retry_count=0,
    )


def ranked_file(path: str, rank: int) -> RankedFile:
    return RankedFile(
        relative_path=path,
        rank=rank,
        best_chunk_id=f"chunk_{rank}",
        sparse_score=10.0 - rank,
    )


def test_chunk_hits_are_deduplicated_to_first_file_evidence() -> None:
    chunks = corpus(
        chunk("chunk_a1", "a.py"),
        chunk("chunk_a2", "a.py"),
        chunk("chunk_b", "b.py"),
    )

    ranked = rank_retrieved_files(
        result("chunk_a2", "chunk_a1", "chunk_b"),
        chunks,
    )

    assert ranked == (
        RankedFile(
            relative_path="a.py",
            rank=1,
            best_chunk_id="chunk_a2",
            sparse_score=3.0,
        ),
        RankedFile(
            relative_path="b.py",
            rank=2,
            best_chunk_id="chunk_b",
            sparse_score=1.0,
        ),
    )


def test_dense_chunk_hits_are_deduplicated_to_first_file_evidence() -> None:
    chunks = corpus(
        chunk("chunk_a1", "a.py"),
        chunk("chunk_a2", "a.py"),
        chunk("chunk_b", "b.py"),
    )

    ranked = rank_dense_retrieved_files(
        dense_result("chunk_a2", "chunk_a1", "chunk_b"),
        chunks,
    )

    assert ranked == (
        DenseRankedFile(
            relative_path="a.py",
            rank=1,
            best_chunk_id="chunk_a2",
            dense_score=1.0,
        ),
        DenseRankedFile(
            relative_path="b.py",
            rank=2,
            best_chunk_id="chunk_b",
            dense_score=0.8,
        ),
    )


def test_file_recall_is_fractional_and_mrr_uses_first_relevant_rank() -> None:
    ranked = tuple(
        ranked_file(path, rank)
        for rank, path in enumerate(("a.py", "b.py", "c.py", "d.py"), start=1)
    )

    metrics = score_file_localization(
        gold_files=("b.py", "d.py"),
        ranked_files=ranked,
        indexing_latency_ms=12,
        retrieval_latency_ms=3,
    )

    assert metrics.file_recall_at_1 == 0.0
    assert metrics.file_recall_at_3 == 0.5
    assert metrics.file_recall_at_5 == 1.0
    assert metrics.first_relevant_file_rank == 2
    assert metrics.mean_reciprocal_rank == 0.5
    assert metrics.indexing_latency_ms == 12
    assert metrics.retrieval_latency_ms == 3


def test_no_relevant_file_produces_zero_metrics_without_error() -> None:
    metrics = score_file_localization(
        gold_files=("missing.py",),
        ranked_files=(ranked_file("other.py", 1),),
        indexing_latency_ms=0,
        retrieval_latency_ms=0,
    )

    assert metrics.file_recall_at_1 == 0
    assert metrics.file_recall_at_3 == 0
    assert metrics.file_recall_at_5 == 0
    assert metrics.mean_reciprocal_rank == 0
    assert metrics.first_relevant_file_rank is None


def test_aggregate_uses_macro_average_and_counts_indexing_once() -> None:
    first = score_file_localization(
        gold_files=("a.py",),
        ranked_files=(ranked_file("a.py", 1),),
        indexing_latency_ms=8,
        retrieval_latency_ms=2,
    )
    second = score_file_localization(
        gold_files=("missing.py",),
        ranked_files=(ranked_file("a.py", 1),),
        indexing_latency_ms=8,
        retrieval_latency_ms=4,
    )

    aggregate = aggregate_localization_metrics((first, second))

    assert aggregate.task_count == 2
    assert aggregate.tasks_with_relevant_file == 1
    assert aggregate.mean_file_recall_at_1 == 0.5
    assert aggregate.mean_file_recall_at_3 == 0.5
    assert aggregate.mean_file_recall_at_5 == 0.5
    assert aggregate.mean_reciprocal_rank == 0.5
    assert aggregate.indexing_latency_ms == 8
    assert aggregate.total_retrieval_latency_ms == 6
    assert aggregate.mean_retrieval_latency_ms == 3.0


def test_metric_functions_reject_inconsistent_inputs() -> None:
    with pytest.raises(ValueError, match="identity"):
        rank_retrieved_files(
            BM25SearchResult(
                repository_id="other.repo",
                commit=COMMIT,
                query="parser",
                top_k=1,
                hits=(),
            ),
            corpus(),
        )
    with pytest.raises(ValueError, match="unknown chunk"):
        rank_retrieved_files(result("chunk_missing"), corpus())
    with pytest.raises(ValueError, match="identity"):
        rank_dense_retrieved_files(
            dense_result(repository_id="other.repo"),
            corpus(),
        )
    with pytest.raises(ValueError, match="unknown chunk"):
        rank_dense_retrieved_files(dense_result("chunk_missing"), corpus())
    for gold_files in ((), ("same.py", "same.py")):
        with pytest.raises(ValueError, match="gold files"):
            score_file_localization(
                gold_files=gold_files,
                ranked_files=(),
                indexing_latency_ms=0,
                retrieval_latency_ms=0,
            )
    with pytest.raises(ValueError, match="contiguous"):
        score_file_localization(
            gold_files=("a.py",),
            ranked_files=(ranked_file("a.py", 2),),
            indexing_latency_ms=0,
            retrieval_latency_ms=0,
        )
    with pytest.raises(ValueError, match="indexing latency"):
        aggregate_localization_metrics(
            (
                score_file_localization(
                    gold_files=("a.py",),
                    ranked_files=(),
                    indexing_latency_ms=1,
                    retrieval_latency_ms=0,
                ),
                score_file_localization(
                    gold_files=("a.py",),
                    ranked_files=(),
                    indexing_latency_ms=2,
                    retrieval_latency_ms=0,
                ),
            )
        )
    with pytest.raises(ValueError, match="at least one"):
        aggregate_localization_metrics(())
