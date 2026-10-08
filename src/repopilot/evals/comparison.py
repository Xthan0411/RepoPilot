"""Fair, deterministic BM25/Dense/Hybrid file-localization comparison."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from repopilot.enums import (
    ComparisonCaseOutcome,
    DatasetSplit,
    RetrievalComparisonErrorCode,
    RetrievalMode,
)
from repopilot.evals.baseline import (
    BaselineSplitCount,
    BM25BaselineManifest,
    build_localization_query,
)
from repopilot.evals.metrics import (
    AggregateLocalizationMetrics,
    aggregate_localization_metrics,
    score_file_localization,
)
from repopilot.retrieval import (
    ChunkCorpus,
    HybridPolicy,
    HybridRetrievalError,
    HybridSearchResult,
    QueryRewriteOutcome,
    RetrievalFilter,
)
from repopilot.schemas import (
    FiniteScore,
    GitSha,
    Identifier,
    LocalizationMetrics,
    NonEmptyText,
    RelativePath,
    StrictModel,
    TaskId,
)

RETRIEVAL_COMPARISON_SCHEMA_VERSION: Literal[1] = 1
RETRIEVAL_COMPARISON_MODES = (
    RetrievalMode.BM25,
    RetrievalMode.DENSE,
    RetrievalMode.HYBRID,
)
ComparisonEvidenceKind = Literal["deterministic_fake_smoke", "real_model_eval"]
COMPARISON_ERROR_MESSAGES = {
    RetrievalComparisonErrorCode.IDENTITY_MISMATCH: (
        "retrieval comparison identity is inconsistent"
    ),
    RetrievalComparisonErrorCode.INVALID_REWRITE: ("retrieval comparison rewrite is invalid"),
    RetrievalComparisonErrorCode.RETRIEVAL_FAILURE: (
        "retrieval comparison has no successful retrieval results"
    ),
    RetrievalComparisonErrorCode.OUTPUT_IO_ERROR: (
        "retrieval comparison artifact could not be written"
    ),
    RetrievalComparisonErrorCode.INVALID_RESULT: ("retrieval comparison result is invalid"),
}


class RetrievalComparisonError(RuntimeError):
    """Safe comparison failure without query, source, gold, or Provider payloads."""

    def __init__(self, code: RetrievalComparisonErrorCode) -> None:
        super().__init__(COMPARISON_ERROR_MESSAGES[code])
        self.code = code


class ComparisonRetriever(Protocol):
    """Stable retrieval surface used by each comparison arm."""

    @property
    def policy(self) -> HybridPolicy: ...

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata_filter: RetrievalFilter | None = None,
    ) -> HybridSearchResult: ...


class ComparisonRankedFile(StrictModel):
    """First chunk evidence retained for one deduplicated repository path."""

    relative_path: RelativePath
    rank: int = Field(gt=0)
    best_chunk_id: Identifier
    sparse_score: FiniteScore | None = None
    dense_score: FiniteScore | None = None
    fused_score: FiniteScore | None = None

    @model_validator(mode="after")
    def require_score(self) -> Self:
        if all(
            score is None
            for score in (
                self.sparse_score,
                self.dense_score,
                self.fused_score,
            )
        ):
            raise ValueError("a ranked file requires retrieval score evidence")
        return self


class ComparisonTaskResult(StrictModel):
    """One mode's auditable result for the fixed evaluator-owned task."""

    task_id: TaskId
    split: DatasetSplit
    original_query: NonEmptyText
    effective_query: NonEmptyText
    rewritten: bool
    gold_files: tuple[RelativePath, ...] = Field(min_length=1)
    missing_gold_files: tuple[RelativePath, ...] = ()
    retrieved_chunk_count: int = Field(ge=0)
    ranked_files: tuple[ComparisonRankedFile, ...]
    provider_call_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    failure_code: RetrievalComparisonErrorCode | None = None
    metrics: LocalizationMetrics

    @model_validator(mode="after")
    def validate_task_result(self) -> Self:
        if self.rewritten != (self.original_query != self.effective_query):
            raise ValueError("rewrite flag does not match the compared queries")
        if self.gold_files != tuple(sorted(set(self.gold_files))):
            raise ValueError("comparison gold files must be unique and sorted")
        if self.missing_gold_files != tuple(sorted(set(self.missing_gold_files))):
            raise ValueError("comparison missing gold files must be unique and sorted")
        if not set(self.missing_gold_files).issubset(self.gold_files):
            raise ValueError("comparison missing gold files must be a subset of gold")
        if self.retrieved_chunk_count < len(self.ranked_files):
            raise ValueError("comparison chunk count cannot be smaller than file count")
        if tuple(item.rank for item in self.ranked_files) != tuple(
            range(1, len(self.ranked_files) + 1)
        ):
            raise ValueError("comparison ranked files must have contiguous ranks")
        if len({item.relative_path for item in self.ranked_files}) != len(self.ranked_files):
            raise ValueError("comparison ranked file paths must be unique")
        if self.failure_code is not None and (
            self.ranked_files
            or self.retrieved_chunk_count
            or self.provider_call_count
            or self.retry_count
            or self.metrics.first_relevant_file_rank is not None
            or self.metrics.mean_reciprocal_rank != 0.0
            or self.metrics.file_recall_at_5 != 0.0
        ):
            raise ValueError("failed comparison tasks cannot claim retrieval evidence")
        return self


class ComparisonModeSummary(AggregateLocalizationMetrics):
    """Mode aggregate plus success/failure and Provider accounting."""

    successful_task_count: int = Field(ge=0)
    failed_task_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_mode_counts(self) -> Self:
        if self.successful_task_count + self.failed_task_count != self.task_count:
            raise ValueError("comparison mode task counts do not add up")
        return self


class ComparisonModeReport(StrictModel):
    """All task results and metrics for exactly one retrieval strategy."""

    mode: RetrievalMode
    split_counts: tuple[BaselineSplitCount, ...]
    task_results: tuple[ComparisonTaskResult, ...] = Field(min_length=1)
    summary: ComparisonModeSummary

    @model_validator(mode="after")
    def validate_mode_report(self) -> Self:
        task_ids = tuple(item.task_id for item in self.task_results)
        if task_ids != tuple(sorted(task_ids)) or len(set(task_ids)) != len(task_ids):
            raise ValueError("comparison task IDs must be unique and sorted")
        expected_splits = Counter(item.split for item in self.task_results)
        if {item.split: item.task_count for item in self.split_counts} != dict(expected_splits):
            raise ValueError("comparison split counts do not match task results")
        if tuple(item.split.value for item in self.split_counts) != tuple(
            sorted(item.split.value for item in self.split_counts)
        ):
            raise ValueError("comparison split counts must be sorted")
        if self.summary.task_count != len(self.task_results):
            raise ValueError("comparison summary task count does not match results")
        if self.summary.failed_task_count != sum(
            item.failure_code is not None for item in self.task_results
        ):
            raise ValueError("comparison failed task count does not match results")
        if self.summary.provider_call_count != sum(
            item.provider_call_count for item in self.task_results
        ):
            raise ValueError("comparison Provider calls do not match task results")
        if self.summary.retry_count != sum(item.retry_count for item in self.task_results):
            raise ValueError("comparison retries do not match task results")
        return self


class ComparisonCase(StrictModel):
    """Hybrid first-relevant-rank change relative to one standalone baseline."""

    task_id: TaskId
    baseline_mode: Literal[RetrievalMode.BM25, RetrievalMode.DENSE]
    baseline_first_relevant_rank: int | None = Field(default=None, gt=0)
    hybrid_first_relevant_rank: int | None = Field(default=None, gt=0)
    outcome: ComparisonCaseOutcome


class RetrievalComparisonReport(StrictModel):
    """Versioned result for the same tasks, queries, gold labels, and Top-K."""

    schema_version: Literal[1] = RETRIEVAL_COMPARISON_SCHEMA_VERSION
    run_mode: Literal["p3_retrieval_comparison"] = "p3_retrieval_comparison"
    evidence_kind: ComparisonEvidenceKind
    benchmark_id: Identifier
    repository_id: Identifier
    commit: GitSha
    retrieval_depth: int = Field(ge=5, le=100)
    source_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    rewrite_task_count: int = Field(ge=0)
    modes: tuple[ComparisonModeReport, ComparisonModeReport, ComparisonModeReport]
    cases: tuple[ComparisonCase, ...]

    @model_validator(mode="after")
    def validate_fair_comparison(self) -> Self:
        if tuple(item.mode for item in self.modes) != RETRIEVAL_COMPARISON_MODES:
            raise ValueError("comparison modes must be BM25, Dense, then Hybrid")
        reference = self.modes[0].task_results
        reference_inputs = tuple(_task_input(item) for item in reference)
        if any(
            tuple(_task_input(item) for item in mode.task_results) != reference_inputs
            for mode in self.modes[1:]
        ):
            raise ValueError("comparison modes do not share identical task inputs")
        if self.rewrite_task_count != sum(item.rewritten for item in reference):
            raise ValueError("comparison rewrite count does not match task inputs")
        expected_case_keys = tuple(
            (task.task_id, baseline)
            for task in reference
            for baseline in (RetrievalMode.BM25, RetrievalMode.DENSE)
        )
        if tuple((item.task_id, item.baseline_mode) for item in self.cases) != (expected_case_keys):
            raise ValueError("comparison cases are missing or out of order")
        return self


@dataclass(frozen=True)
class RetrievalComparisonArtifacts:
    """Written JSON and human-readable Markdown comparison artifacts."""

    report: RetrievalComparisonReport
    json_path: Path
    markdown_path: Path


def rank_comparison_files(
    result: HybridSearchResult,
    corpus: ChunkCorpus,
) -> tuple[ComparisonRankedFile, ...]:
    """Deduplicate chunk hits to file rank without access to evaluator gold."""

    if result.repository_id != corpus.repository_id or result.commit != corpus.commit:
        raise RetrievalComparisonError(RetrievalComparisonErrorCode.IDENTITY_MISMATCH)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    ranked: list[ComparisonRankedFile] = []
    seen_paths: set[str] = set()
    for hit in result.hits:
        chunk = chunks_by_id.get(hit.chunk_id)
        if chunk is None:
            raise RetrievalComparisonError(RetrievalComparisonErrorCode.INVALID_RESULT)
        if chunk.relative_path in seen_paths:
            continue
        seen_paths.add(chunk.relative_path)
        try:
            ranked.append(
                ComparisonRankedFile(
                    relative_path=chunk.relative_path,
                    rank=len(ranked) + 1,
                    best_chunk_id=chunk.chunk_id,
                    sparse_score=hit.sparse_score,
                    dense_score=hit.dense_score,
                    fused_score=hit.fused_score,
                )
            )
        except (TypeError, ValueError):
            raise RetrievalComparisonError(RetrievalComparisonErrorCode.INVALID_RESULT) from None
    return tuple(ranked)


def evaluate_retrieval_comparison(
    manifest: BM25BaselineManifest,
    *,
    corpus: ChunkCorpus,
    retrievers: Mapping[RetrievalMode, ComparisonRetriever],
    indexing_latency_ms: Mapping[RetrievalMode, int],
    rewrites: Mapping[str, QueryRewriteOutcome] | None = None,
    evidence_kind: ComparisonEvidenceKind = "deterministic_fake_smoke",
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> RetrievalComparisonReport:
    """Compare all three modes while freezing evaluator-controlled inputs."""

    _validate_comparison_inputs(
        manifest,
        corpus=corpus,
        retrievers=retrievers,
        indexing_latency_ms=indexing_latency_ms,
        rewrites=rewrites,
    )
    indexed_paths = {chunk.relative_path for chunk in corpus.chunks}
    effective_rewrites = rewrites or {}
    mode_reports: list[ComparisonModeReport] = []
    for mode in RETRIEVAL_COMPARISON_MODES:
        task_results: list[ComparisonTaskResult] = []
        for evaluation in manifest.tasks:
            original_query = build_localization_query(evaluation.task)
            rewrite = effective_rewrites.get(evaluation.task.task_id)
            effective_query = rewrite.effective_query if rewrite is not None else original_query
            started = clock_ns()
            failure_code: RetrievalComparisonErrorCode | None = None
            try:
                retrieval = retrievers[mode].search(
                    effective_query,
                    top_k=manifest.retrieval_depth,
                )
                retrieval_latency_ms = _elapsed_ms(started, clock_ns())
                _validate_retrieval(
                    retrieval,
                    mode=mode,
                    corpus=corpus,
                    query=effective_query,
                    top_k=manifest.retrieval_depth,
                )
                ranked_files = rank_comparison_files(retrieval, corpus)
                retrieved_chunk_count = len(retrieval.hits)
                provider_call_count = retrieval.provider_call_count
                retry_count = retrieval.retry_count
            except HybridRetrievalError:
                retrieval_latency_ms = _elapsed_ms(started, clock_ns())
                failure_code = RetrievalComparisonErrorCode.RETRIEVAL_FAILURE
                ranked_files = ()
                retrieved_chunk_count = 0
                provider_call_count = 0
                retry_count = 0
            gold_files = tuple(sorted(evaluation.gold_files))
            try:
                metrics = score_file_localization(
                    gold_files=gold_files,
                    ranked_files=ranked_files,
                    indexing_latency_ms=indexing_latency_ms[mode],
                    retrieval_latency_ms=retrieval_latency_ms,
                )
                task_results.append(
                    ComparisonTaskResult(
                        task_id=evaluation.task.task_id,
                        split=evaluation.split,
                        original_query=original_query,
                        effective_query=effective_query,
                        rewritten=effective_query != original_query,
                        gold_files=gold_files,
                        missing_gold_files=tuple(
                            path for path in gold_files if path not in indexed_paths
                        ),
                        retrieved_chunk_count=retrieved_chunk_count,
                        ranked_files=ranked_files,
                        provider_call_count=provider_call_count,
                        retry_count=retry_count,
                        failure_code=failure_code,
                        metrics=metrics,
                    )
                )
            except (TypeError, ValueError):
                raise RetrievalComparisonError(
                    RetrievalComparisonErrorCode.INVALID_RESULT
                ) from None
        mode_reports.append(_build_mode_report(mode, tuple(task_results)))
    if all(mode.summary.successful_task_count == 0 for mode in mode_reports):
        raise RetrievalComparisonError(RetrievalComparisonErrorCode.RETRIEVAL_FAILURE)
    try:
        modes = (mode_reports[0], mode_reports[1], mode_reports[2])
        return RetrievalComparisonReport(
            benchmark_id=manifest.benchmark_id,
            repository_id=manifest.repository_id,
            commit=manifest.commit,
            evidence_kind=evidence_kind,
            retrieval_depth=manifest.retrieval_depth,
            source_count=len(corpus.sources),
            chunk_count=len(corpus.chunks),
            rewrite_task_count=sum(
                rewrite.effective_query != rewrite.original_query
                for rewrite in effective_rewrites.values()
            ),
            modes=modes,
            cases=_build_cases(modes, manifest.retrieval_depth),
        )
    except (TypeError, ValueError):
        raise RetrievalComparisonError(RetrievalComparisonErrorCode.INVALID_RESULT) from None


def write_retrieval_comparison(
    report: RetrievalComparisonReport,
    output_dir: Path,
) -> RetrievalComparisonArtifacts:
    """Atomically write deterministic JSON and Markdown comparison reports."""

    json_path = output_dir / "retrieval-comparison.json"
    markdown_path = output_dir / "retrieval-comparison.md"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_text(json_path, f"{report.model_dump_json(indent=2)}\n")
        _write_text(markdown_path, render_retrieval_comparison_markdown(report))
    except OSError:
        raise RetrievalComparisonError(RetrievalComparisonErrorCode.OUTPUT_IO_ERROR) from None
    return RetrievalComparisonArtifacts(
        report=report,
        json_path=json_path,
        markdown_path=markdown_path,
    )


def render_retrieval_comparison_markdown(
    report: RetrievalComparisonReport,
) -> str:
    """Render a reviewer-facing summary without source or expected patch text."""

    lines = [
        "# P3 Retrieval Comparison",
        "",
        f"- Benchmark: `{report.benchmark_id}`",
        f"- Repository: `{report.repository_id}`",
        f"- Commit: `{report.commit}`",
        f"- Fixed Top-K: `{report.retrieval_depth}`",
        f"- Tasks: `{report.modes[0].summary.task_count}`",
        f"- Rewritten tasks: `{report.rewrite_task_count}`",
        f"- Evidence: `{report.evidence_kind}`",
        "",
        (
            "> This synthetic Fake run validates mechanics and comparison accounting. "
            "It does not measure real-model retrieval quality."
            if report.evidence_kind == "deterministic_fake_smoke"
            else "> This report was generated by an explicitly selected real-model evaluation."
        ),
        "",
        "## Mode summary",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Index ms | Query ms | "
        "Provider calls | Retries | Failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in report.modes:
        summary = mode.summary
        lines.append(
            f"| {mode.mode.value} | {summary.mean_file_recall_at_1:.3f} | "
            f"{summary.mean_file_recall_at_3:.3f} | "
            f"{summary.mean_file_recall_at_5:.3f} | "
            f"{summary.mean_reciprocal_rank:.3f} | "
            f"{summary.indexing_latency_ms} | "
            f"{summary.total_retrieval_latency_ms} | "
            f"{summary.provider_call_count} | {summary.retry_count} | "
            f"{summary.failed_task_count} |"
        )
    lines.extend(
        [
            "",
            "## Hybrid rank changes",
            "",
            "| Task | Baseline | Baseline rank | Hybrid rank | Outcome |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for case in report.cases:
        baseline_rank = (
            str(case.baseline_first_relevant_rank)
            if case.baseline_first_relevant_rank is not None
            else "not found"
        )
        hybrid_rank = (
            str(case.hybrid_first_relevant_rank)
            if case.hybrid_first_relevant_rank is not None
            else "not found"
        )
        lines.append(
            f"| {case.task_id} | {case.baseline_mode.value} | {baseline_rank} | "
            f"{hybrid_rank} | {case.outcome.value} |"
        )
    return "\n".join(lines) + "\n"


def _validate_comparison_inputs(
    manifest: BM25BaselineManifest,
    *,
    corpus: ChunkCorpus,
    retrievers: Mapping[RetrievalMode, ComparisonRetriever],
    indexing_latency_ms: Mapping[RetrievalMode, int],
    rewrites: Mapping[str, QueryRewriteOutcome] | None,
) -> None:
    required_modes = set(RETRIEVAL_COMPARISON_MODES)
    if (
        manifest.repository_id != corpus.repository_id
        or manifest.commit != corpus.commit
        or set(retrievers) != required_modes
        or set(indexing_latency_ms) != required_modes
        or any(
            isinstance(indexing_latency_ms[mode], bool)
            or not isinstance(indexing_latency_ms[mode], int)
            or indexing_latency_ms[mode] < 0
            for mode in RETRIEVAL_COMPARISON_MODES
        )
        or any(retrievers[mode].policy.mode is not mode for mode in RETRIEVAL_COMPARISON_MODES)
    ):
        raise RetrievalComparisonError(RetrievalComparisonErrorCode.IDENTITY_MISMATCH)
    if rewrites is None:
        return
    task_queries = {
        evaluation.task.task_id: build_localization_query(evaluation.task)
        for evaluation in manifest.tasks
    }
    if set(rewrites) - set(task_queries):
        raise RetrievalComparisonError(RetrievalComparisonErrorCode.INVALID_REWRITE)
    for task_id, rewrite in rewrites.items():
        if (
            rewrite.repository_id != corpus.repository_id
            or rewrite.commit != corpus.commit
            or rewrite.original_query != task_queries[task_id]
        ):
            raise RetrievalComparisonError(RetrievalComparisonErrorCode.INVALID_REWRITE)


def _validate_retrieval(
    retrieval: HybridSearchResult,
    *,
    mode: RetrievalMode,
    corpus: ChunkCorpus,
    query: str,
    top_k: int,
) -> None:
    if (
        retrieval.mode is not mode
        or retrieval.repository_id != corpus.repository_id
        or retrieval.commit != corpus.commit
        or retrieval.query != query
        or retrieval.top_k != top_k
    ):
        raise RetrievalComparisonError(RetrievalComparisonErrorCode.INVALID_RESULT)


def _build_mode_report(
    mode: RetrievalMode,
    task_results: tuple[ComparisonTaskResult, ...],
) -> ComparisonModeReport:
    try:
        aggregate = aggregate_localization_metrics(tuple(result.metrics for result in task_results))
        failed = sum(item.failure_code is not None for item in task_results)
        return ComparisonModeReport(
            mode=mode,
            split_counts=tuple(
                BaselineSplitCount(split=split, task_count=count)
                for split, count in sorted(
                    Counter(item.split for item in task_results).items(),
                    key=lambda item: item[0].value,
                )
            ),
            task_results=task_results,
            summary=ComparisonModeSummary(
                **aggregate.model_dump(),
                successful_task_count=len(task_results) - failed,
                failed_task_count=failed,
                provider_call_count=sum(item.provider_call_count for item in task_results),
                retry_count=sum(item.retry_count for item in task_results),
            ),
        )
    except (TypeError, ValueError):
        raise RetrievalComparisonError(RetrievalComparisonErrorCode.INVALID_RESULT) from None


def _build_cases(
    modes: tuple[ComparisonModeReport, ComparisonModeReport, ComparisonModeReport],
    retrieval_depth: int,
) -> tuple[ComparisonCase, ...]:
    by_mode = {mode.mode: {item.task_id: item for item in mode.task_results} for mode in modes}
    cases: list[ComparisonCase] = []
    for hybrid in modes[2].task_results:
        hybrid_rank = hybrid.metrics.first_relevant_file_rank
        for baseline_mode in (RetrievalMode.BM25, RetrievalMode.DENSE):
            baseline_rank = by_mode[baseline_mode][hybrid.task_id].metrics.first_relevant_file_rank
            baseline_value = baseline_rank or retrieval_depth + 1
            hybrid_value = hybrid_rank or retrieval_depth + 1
            if hybrid_value < baseline_value:
                outcome = ComparisonCaseOutcome.IMPROVED
            elif hybrid_value > baseline_value:
                outcome = ComparisonCaseOutcome.REGRESSED
            else:
                outcome = ComparisonCaseOutcome.UNCHANGED
            cases.append(
                ComparisonCase(
                    task_id=hybrid.task_id,
                    baseline_mode=baseline_mode,
                    baseline_first_relevant_rank=baseline_rank,
                    hybrid_first_relevant_rank=hybrid_rank,
                    outcome=outcome,
                )
            )
    return tuple(cases)


def _task_input(result: ComparisonTaskResult) -> tuple[object, ...]:
    return (
        result.task_id,
        result.split,
        result.original_query,
        result.effective_query,
        result.rewritten,
        result.gold_files,
        result.missing_gold_files,
    )


def _elapsed_ms(started_ns: int, finished_ns: int) -> int:
    if finished_ns < started_ns:
        raise RetrievalComparisonError(RetrievalComparisonErrorCode.INVALID_RESULT)
    return (finished_ns - started_ns) // 1_000_000


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    except OSError:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise
