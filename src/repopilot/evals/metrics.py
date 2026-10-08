"""Pure file-ranking and gold-aware localization metric functions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import Field, model_validator

from repopilot.retrieval import BM25SearchResult, ChunkCorpus, DenseSearchResult
from repopilot.schemas import (
    FiniteScore,
    Identifier,
    LocalizationMetrics,
    RelativePath,
    StrictModel,
)


class RankedFile(StrictModel):
    """Best BM25 chunk evidence retained for one deduplicated repository path."""

    relative_path: RelativePath
    rank: int = Field(gt=0)
    best_chunk_id: Identifier
    sparse_score: FiniteScore


class DenseRankedFile(StrictModel):
    """Best Dense chunk evidence retained for one deduplicated repository path."""

    relative_path: RelativePath
    rank: int = Field(gt=0)
    best_chunk_id: Identifier
    dense_score: FiniteScore = Field(ge=-1.0, le=1.0)


class RankedFileLike(Protocol):
    """Minimum file-ranking surface accepted by localization metrics."""

    relative_path: str
    rank: int


def rank_retrieved_files(
    result: BM25SearchResult,
    corpus: ChunkCorpus,
) -> tuple[RankedFile, ...]:
    """Convert chunk hits to first-occurrence file ranking without gold labels."""

    if result.repository_id != corpus.repository_id or result.commit != corpus.commit:
        raise ValueError("retrieval result identity must match the chunk corpus")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    ranked: list[RankedFile] = []
    seen_paths: set[str] = set()
    for hit in result.hits:
        chunk = chunks_by_id.get(hit.chunk_id)
        if chunk is None:
            raise ValueError("retrieval result references an unknown chunk")
        if chunk.relative_path in seen_paths:
            continue
        if hit.sparse_score is None:
            raise ValueError("BM25 file ranking requires a sparse score")
        seen_paths.add(chunk.relative_path)
        ranked.append(
            RankedFile(
                relative_path=chunk.relative_path,
                rank=len(ranked) + 1,
                best_chunk_id=chunk.chunk_id,
                sparse_score=hit.sparse_score,
            )
        )
    return tuple(ranked)


def rank_dense_retrieved_files(
    result: DenseSearchResult,
    corpus: ChunkCorpus,
) -> tuple[DenseRankedFile, ...]:
    """Convert Dense chunk hits to first-occurrence files without gold labels."""

    if result.repository_id != corpus.repository_id or result.commit != corpus.commit:
        raise ValueError("retrieval result identity must match the chunk corpus")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    ranked: list[DenseRankedFile] = []
    seen_paths: set[str] = set()
    for hit in result.hits:
        chunk = chunks_by_id.get(hit.chunk_id)
        if chunk is None:
            raise ValueError("retrieval result references an unknown chunk")
        if chunk.relative_path in seen_paths:
            continue
        if hit.dense_score is None:
            raise ValueError("Dense file ranking requires a Dense score")
        seen_paths.add(chunk.relative_path)
        ranked.append(
            DenseRankedFile(
                relative_path=chunk.relative_path,
                rank=len(ranked) + 1,
                best_chunk_id=chunk.chunk_id,
                dense_score=hit.dense_score,
            )
        )
    return tuple(ranked)


def score_file_localization(
    *,
    gold_files: Sequence[str],
    ranked_files: Sequence[RankedFileLike],
    indexing_latency_ms: int,
    retrieval_latency_ms: int,
) -> LocalizationMetrics:
    """Score file-level Recall@1/3/5 and reciprocal first-relevant rank."""

    if not gold_files or len(set(gold_files)) != len(gold_files):
        raise ValueError("gold files must be a non-empty unique sequence")
    ranks = tuple(item.rank for item in ranked_files)
    if ranks != tuple(range(1, len(ranked_files) + 1)):
        raise ValueError("ranked file positions must be contiguous")
    paths = tuple(item.relative_path for item in ranked_files)
    if len(set(paths)) != len(paths):
        raise ValueError("ranked file paths must be unique")

    gold_set = set(gold_files)

    def recall_at(limit: int) -> float:
        retrieved = {item.relative_path for item in ranked_files[:limit]}
        return len(retrieved & gold_set) / len(gold_set)

    first_relevant_rank = next(
        (item.rank for item in ranked_files if item.relative_path in gold_set),
        None,
    )
    return LocalizationMetrics(
        file_recall_at_1=recall_at(1),
        file_recall_at_3=recall_at(3),
        file_recall_at_5=recall_at(5),
        mean_reciprocal_rank=(
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        ),
        first_relevant_file_rank=first_relevant_rank,
        indexing_latency_ms=indexing_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
    )


class AggregateLocalizationMetrics(StrictModel):
    """Macro averages and latency totals across a fixed task set."""

    task_count: int = Field(gt=0)
    tasks_with_relevant_file: int = Field(ge=0)
    mean_file_recall_at_1: FiniteScore = Field(ge=0.0, le=1.0)
    mean_file_recall_at_3: FiniteScore = Field(ge=0.0, le=1.0)
    mean_file_recall_at_5: FiniteScore = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: FiniteScore = Field(ge=0.0, le=1.0)
    indexing_latency_ms: int = Field(ge=0)
    total_retrieval_latency_ms: int = Field(ge=0)
    mean_retrieval_latency_ms: FiniteScore = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_counts_and_recall(self) -> AggregateLocalizationMetrics:
        if self.tasks_with_relevant_file > self.task_count:
            raise ValueError("relevant-file task count cannot exceed task count")
        if not (
            self.mean_file_recall_at_1 <= self.mean_file_recall_at_3 <= self.mean_file_recall_at_5
        ):
            raise ValueError("mean file recall must be monotonic")
        return self


def aggregate_localization_metrics(
    metrics: Sequence[LocalizationMetrics],
) -> AggregateLocalizationMetrics:
    """Macro-average task metrics while counting shared indexing latency once."""

    if not metrics:
        raise ValueError("at least one task metric is required")
    indexing_latencies = {item.indexing_latency_ms for item in metrics}
    if len(indexing_latencies) != 1:
        raise ValueError("task metrics must share one indexing latency")
    count = len(metrics)
    total_retrieval_latency_ms = sum(item.retrieval_latency_ms for item in metrics)
    return AggregateLocalizationMetrics(
        task_count=count,
        tasks_with_relevant_file=sum(item.first_relevant_file_rank is not None for item in metrics),
        mean_file_recall_at_1=sum(item.file_recall_at_1 for item in metrics) / count,
        mean_file_recall_at_3=sum(item.file_recall_at_3 for item in metrics) / count,
        mean_file_recall_at_5=sum(item.file_recall_at_5 for item in metrics) / count,
        mean_reciprocal_rank=sum(item.mean_reciprocal_rank for item in metrics) / count,
        indexing_latency_ms=indexing_latencies.pop(),
        total_retrieval_latency_ms=total_retrieval_latency_ms,
        mean_retrieval_latency_ms=total_retrieval_latency_ms / count,
    )
