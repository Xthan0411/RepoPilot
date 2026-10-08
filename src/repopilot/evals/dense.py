"""Gold-isolated Dense localization evaluation using the frozen P2 task manifest."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Literal, Self

from pydantic import Field, model_validator

from repopilot.embeddings import EmbeddingIdentity, EmbeddingIndexResult
from repopilot.enums import DatasetSplit, DenseEvalErrorCode
from repopilot.evals.baseline import (
    BaselineSplitCount,
    BaselineSummary,
    BM25BaselineManifest,
    build_localization_query,
)
from repopilot.evals.metrics import (
    DenseRankedFile,
    aggregate_localization_metrics,
    rank_dense_retrieved_files,
    score_file_localization,
)
from repopilot.retrieval import ChunkCorpus, DenseRetrievalError, DenseRetriever
from repopilot.schemas import (
    GitSha,
    Identifier,
    LocalizationMetrics,
    NonEmptyText,
    RelativePath,
    StrictModel,
    TaskId,
)

DENSE_EVALUATION_SCHEMA_VERSION: Literal[1] = 1
DENSE_EVAL_ERROR_MESSAGES = {
    DenseEvalErrorCode.IDENTITY_MISMATCH: "Dense evaluation identity is inconsistent",
    DenseEvalErrorCode.RETRIEVAL_FAILURE: "Dense evaluation retrieval failed",
    DenseEvalErrorCode.INVALID_RESULT: "Dense evaluation result is invalid",
}


class DenseEvaluationError(RuntimeError):
    """Safe evaluation-layer failure without query, gold, or source content."""

    def __init__(self, code: DenseEvalErrorCode) -> None:
        super().__init__(DENSE_EVAL_ERROR_MESSAGES[code])
        self.code = code


class DenseEvaluationTaskResult(StrictModel):
    """One Dense file-localization result joined with gold only after retrieval."""

    task_id: TaskId
    split: DatasetSplit
    query: NonEmptyText
    gold_files: tuple[RelativePath, ...] = Field(min_length=1)
    missing_gold_files: tuple[RelativePath, ...] = ()
    retrieved_chunk_count: int = Field(ge=0)
    ranked_files: tuple[DenseRankedFile, ...]
    metrics: LocalizationMetrics

    @model_validator(mode="after")
    def validate_task_result(self) -> Self:
        if self.gold_files != tuple(sorted(self.gold_files)) or len(set(self.gold_files)) != len(
            self.gold_files
        ):
            raise ValueError("Dense gold files must be unique and sorted")
        if self.missing_gold_files != tuple(sorted(self.missing_gold_files)) or len(
            set(self.missing_gold_files)
        ) != len(self.missing_gold_files):
            raise ValueError("Dense missing gold files must be unique and sorted")
        if not set(self.missing_gold_files).issubset(self.gold_files):
            raise ValueError("Dense missing gold files must be a subset of gold")
        if self.retrieved_chunk_count < len(self.ranked_files):
            raise ValueError("Dense chunk count cannot be smaller than file count")
        ranks = tuple(item.rank for item in self.ranked_files)
        if ranks != tuple(range(1, len(self.ranked_files) + 1)):
            raise ValueError("Dense ranked files must have contiguous ranks")
        paths = tuple(item.relative_path for item in self.ranked_files)
        if len(set(paths)) != len(paths):
            raise ValueError("Dense ranked file paths must be unique")
        return self


class DenseEvaluationReport(StrictModel):
    """JSON-serializable metrics for the exact P2 benchmark task set."""

    schema_version: Literal[1] = DENSE_EVALUATION_SCHEMA_VERSION
    run_mode: Literal["p3_dense_evaluation"] = "p3_dense_evaluation"
    benchmark_id: Identifier
    repository_id: Identifier
    commit: GitSha
    identity: EmbeddingIdentity
    retrieval_depth: int = Field(ge=5, le=100)
    source_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    unique_content_count: int = Field(ge=0)
    cache_hit_count: int = Field(ge=0)
    cache_miss_count: int = Field(ge=0)
    index_provider_call_count: int = Field(ge=0)
    query_provider_call_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    index_retry_count: int = Field(ge=0)
    query_retry_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    split_counts: tuple[BaselineSplitCount, ...]
    task_results: tuple[DenseEvaluationTaskResult, ...] = Field(min_length=1)
    summary: BaselineSummary

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        task_ids = tuple(item.task_id for item in self.task_results)
        if task_ids != tuple(sorted(task_ids)) or len(set(task_ids)) != len(task_ids):
            raise ValueError("Dense task results must have unique sorted task IDs")
        if self.summary.task_count != len(self.task_results):
            raise ValueError("Dense summary task count does not match results")
        expected_splits = Counter(item.split for item in self.task_results)
        actual_splits = {item.split: item.task_count for item in self.split_counts}
        if actual_splits != dict(expected_splits):
            raise ValueError("Dense split counts do not match task results")
        split_order = tuple(item.split.value for item in self.split_counts)
        if split_order != tuple(sorted(split_order)):
            raise ValueError("Dense split counts must be sorted")
        if self.provider_call_count != (
            self.index_provider_call_count + self.query_provider_call_count
        ):
            raise ValueError("Dense provider call total does not match its phases")
        if self.retry_count != self.index_retry_count + self.query_retry_count:
            raise ValueError("Dense retry total does not match its phases")
        if self.unique_content_count != self.cache_hit_count + self.cache_miss_count:
            raise ValueError("Dense unique content does not match cache outcomes")
        return self


def evaluate_dense_localization(
    manifest: BM25BaselineManifest,
    *,
    corpus: ChunkCorpus,
    index_result: EmbeddingIndexResult,
    retriever: DenseRetriever,
    indexing_latency_ms: int,
    clock_ns: Callable[[], int],
) -> DenseEvaluationReport:
    """Reuse P2 tasks/query/gold/Top-K exactly, changing only the Retriever."""

    if (
        indexing_latency_ms < 0
        or manifest.repository_id != corpus.repository_id
        or manifest.commit != corpus.commit
        or index_result.namespace.repository_id != corpus.repository_id
        or index_result.namespace.commit != corpus.commit
        or retriever.namespace != index_result.namespace
        or index_result.chunk_count != len(corpus.chunks)
        or index_result.identity.model != index_result.namespace.embedding_model
        or index_result.identity.fingerprint != index_result.namespace.embedding_version
        or index_result.identity.dimension != index_result.namespace.dimension
    ):
        raise DenseEvaluationError(DenseEvalErrorCode.IDENTITY_MISMATCH)
    indexed_paths = {chunk.relative_path for chunk in corpus.chunks}
    task_results: list[DenseEvaluationTaskResult] = []
    query_provider_call_count = 0
    query_retry_count = 0
    for evaluation in manifest.tasks:
        query = build_localization_query(evaluation.task)
        started = clock_ns()
        try:
            retrieval = retriever.search(query, top_k=manifest.retrieval_depth)
            retrieval_latency_ms = _elapsed_ms(started, clock_ns())
            ranked_files = rank_dense_retrieved_files(retrieval, corpus)
        except (DenseRetrievalError, ValueError):
            raise DenseEvaluationError(DenseEvalErrorCode.RETRIEVAL_FAILURE) from None
        query_provider_call_count += retrieval.provider_call_count
        query_retry_count += retrieval.retry_count
        try:
            gold_files = tuple(sorted(evaluation.gold_files))
            metrics = score_file_localization(
                gold_files=gold_files,
                ranked_files=ranked_files,
                indexing_latency_ms=indexing_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
            )
            task_results.append(
                DenseEvaluationTaskResult(
                    task_id=evaluation.task.task_id,
                    split=evaluation.split,
                    query=query,
                    gold_files=gold_files,
                    missing_gold_files=tuple(
                        path for path in gold_files if path not in indexed_paths
                    ),
                    retrieved_chunk_count=len(retrieval.hits),
                    ranked_files=ranked_files,
                    metrics=metrics,
                )
            )
        except (TypeError, ValueError):
            raise DenseEvaluationError(DenseEvalErrorCode.INVALID_RESULT) from None
    try:
        aggregate = aggregate_localization_metrics(tuple(result.metrics for result in task_results))
        return DenseEvaluationReport(
            benchmark_id=manifest.benchmark_id,
            repository_id=manifest.repository_id,
            commit=manifest.commit,
            identity=index_result.identity,
            retrieval_depth=manifest.retrieval_depth,
            source_count=len(corpus.sources),
            chunk_count=len(corpus.chunks),
            unique_content_count=index_result.unique_content_count,
            cache_hit_count=index_result.cache_hit_count,
            cache_miss_count=index_result.cache_miss_count,
            index_provider_call_count=index_result.provider_call_count,
            query_provider_call_count=query_provider_call_count,
            provider_call_count=index_result.provider_call_count + query_provider_call_count,
            index_retry_count=index_result.retry_count,
            query_retry_count=query_retry_count,
            retry_count=index_result.retry_count + query_retry_count,
            split_counts=tuple(
                BaselineSplitCount(split=split, task_count=count)
                for split, count in sorted(
                    Counter(item.split for item in task_results).items(),
                    key=lambda item: item[0].value,
                )
            ),
            task_results=tuple(task_results),
            summary=BaselineSummary.model_validate(aggregate.model_dump()),
        )
    except (TypeError, ValueError):
        raise DenseEvaluationError(DenseEvalErrorCode.INVALID_RESULT) from None


def _elapsed_ms(started_ns: int, finished_ns: int) -> int:
    if finished_ns < started_ns:
        raise DenseEvaluationError(DenseEvalErrorCode.INVALID_RESULT)
    return (finished_ns - started_ns) // 1_000_000
