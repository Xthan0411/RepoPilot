"""One-command, gold-isolated BM25 file-localization baseline workflow."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, ValidationError, field_validator, model_validator

from repopilot.config import RepoPilotSettings
from repopilot.enums import DatasetSplit, LocalizationEvalErrorCode
from repopilot.evals.metrics import (
    AggregateLocalizationMetrics,
    RankedFile,
    aggregate_localization_metrics,
    rank_retrieved_files,
    score_file_localization,
)
from repopilot.retrieval import (
    BM25Error,
    BM25Policy,
    BM25Retriever,
    ChunkCorpus,
    ChunkCorpusError,
    RepositoryChunkCorpusBuilder,
    RepositoryIngestError,
    build_bm25_index,
    ingest_repository,
    verify_bm25_index_corpus,
    write_bm25_index,
    write_chunk_corpus,
)
from repopilot.schemas import (
    EvaluationTaskSpec,
    FiniteScore,
    GitSha,
    Identifier,
    LocalizationMetrics,
    NonEmptyText,
    RelativePath,
    Sha256,
    StrictModel,
    TaskId,
    TaskSpec,
)
from repopilot.tools import ReadOnlyRepository, RepositoryAccessError

BM25_BASELINE_SCHEMA_VERSION: Literal[1] = 1
BASELINE_ERROR_MESSAGES = {
    LocalizationEvalErrorCode.INPUT_IO_ERROR: "unable to read BM25 baseline input",
    LocalizationEvalErrorCode.OUTPUT_IO_ERROR: "unable to write BM25 baseline artifacts",
    LocalizationEvalErrorCode.INVALID_MANIFEST: "BM25 baseline manifest is invalid",
    LocalizationEvalErrorCode.UNSUPPORTED_VERSION: (
        "BM25 baseline manifest version is unsupported"
    ),
    LocalizationEvalErrorCode.PATH_VIOLATION: ("BM25 baseline path escapes its allowed directory"),
    LocalizationEvalErrorCode.IDENTITY_MISMATCH: (
        "BM25 baseline repository or commit identity is inconsistent"
    ),
    LocalizationEvalErrorCode.RETRIEVAL_FAILURE: ("BM25 baseline indexing or retrieval failed"),
}


class LocalizationEvalError(RuntimeError):
    """Safe typed failure from baseline input, retrieval, or artifact output."""

    def __init__(
        self,
        code: LocalizationEvalErrorCode,
        *,
        detail: str | None = None,
    ) -> None:
        message = BASELINE_ERROR_MESSAGES[code]
        if detail is not None:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.code = code


class BM25BaselineManifest(StrictModel):
    """Evaluator-owned task set for one fixed local repository checkout."""

    schema_version: Literal[1] = BM25_BASELINE_SCHEMA_VERSION
    benchmark_id: Identifier
    repository_id: Identifier
    repository_root: RelativePath
    commit: GitSha
    tasks: tuple[EvaluationTaskSpec, ...] = Field(min_length=1)
    retrieval_depth: int = Field(default=100, ge=5, le=100)
    corpus_filename: RelativePath = "chunks.jsonl"
    index_filename: RelativePath = "bm25-index.json"
    task_jsonl_filename: RelativePath = "tasks.jsonl"
    task_csv_filename: RelativePath = "tasks.csv"
    summary_filename: RelativePath = "summary.md"
    report_filename: RelativePath = "baseline.json"

    @field_validator("retrieval_depth", mode="before")
    @classmethod
    def reject_boolean_depth(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("retrieval_depth must be an integer")
        return value

    @model_validator(mode="after")
    def validate_tasks_and_artifacts(self) -> Self:
        task_ids = tuple(item.task.task_id for item in self.tasks)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("baseline task IDs must be unique")
        if task_ids != tuple(sorted(task_ids)):
            raise ValueError("baseline tasks must be sorted by task ID")
        for item in self.tasks:
            if item.task.base_commit != self.commit:
                raise ValueError("baseline task commits must match the repository commit")
            if (
                item.task.repository_url is not None
                or item.task.repository_path != self.repository_root
            ):
                raise ValueError("baseline tasks must use the manifest repository root")
        filenames = (
            self.corpus_filename,
            self.index_filename,
            self.task_jsonl_filename,
            self.task_csv_filename,
            self.summary_filename,
            self.report_filename,
        )
        if len(set(filenames)) != len(filenames):
            raise ValueError("baseline artifact filenames must be unique")
        artifact_parts = tuple(PurePosixPath(name).parts for name in filenames)
        if any(
            left == right[: len(left)] or right == left[: len(right)]
            for index, left in enumerate(artifact_parts)
            for right in artifact_parts[index + 1 :]
        ):
            raise ValueError("baseline artifact paths must not contain each other")
        return self


class BaselineConfiguration(StrictModel):
    """Effective deterministic settings that affect this baseline."""

    bm25_k1: FiniteScore = Field(gt=0.0, le=5.0)
    bm25_b: FiniteScore = Field(ge=0.0, le=1.0)
    bm25_default_top_k: int = Field(gt=0, le=100)
    retrieval_depth: int = Field(ge=5, le=100)
    chunk_max_lines: int = Field(gt=0, le=2_000)
    chunk_overlap_lines: int = Field(ge=0, le=500)
    repository_max_file_bytes: int = Field(gt=0, le=10_000_000)
    repository_max_list_entries: int = Field(gt=0, le=20_000)


class BM25BaselineTaskResult(StrictModel):
    """One evaluator-owned file localization result without expected patch text."""

    task_id: TaskId
    split: DatasetSplit
    query: NonEmptyText
    gold_files: tuple[RelativePath, ...] = Field(min_length=1)
    missing_gold_files: tuple[RelativePath, ...] = ()
    retrieved_chunk_count: int = Field(ge=0)
    ranked_files: tuple[RankedFile, ...]
    metrics: LocalizationMetrics

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.gold_files != tuple(sorted(self.gold_files)) or len(set(self.gold_files)) != len(
            self.gold_files
        ):
            raise ValueError("baseline gold files must be unique and sorted")
        if self.missing_gold_files != tuple(sorted(self.missing_gold_files)):
            raise ValueError("missing gold files must be sorted")
        if not set(self.missing_gold_files).issubset(self.gold_files):
            raise ValueError("missing gold files must be a subset of gold files")
        if self.retrieved_chunk_count < len(self.ranked_files):
            raise ValueError("retrieved chunk count cannot be smaller than file count")
        ranks = tuple(item.rank for item in self.ranked_files)
        if ranks != tuple(range(1, len(self.ranked_files) + 1)):
            raise ValueError("baseline ranked files must have contiguous ranks")
        return self


class BaselineSummary(AggregateLocalizationMetrics):
    """Named aggregate contract embedded in the baseline report."""


class BaselineSplitCount(StrictModel):
    split: DatasetSplit
    task_count: int = Field(gt=0)


class BM25BaselineReport(StrictModel):
    """Complete auditable summary for one fixed sparse localization run."""

    schema_version: Literal[1] = BM25_BASELINE_SCHEMA_VERSION
    run_mode: Literal["p2_bm25_baseline"] = "p2_bm25_baseline"
    benchmark_id: Identifier
    repository_id: Identifier
    commit: GitSha
    configuration: BaselineConfiguration
    source_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    indexed_document_count: int = Field(ge=0)
    corpus_sha256: Sha256
    index_sha256: Sha256
    split_counts: tuple[BaselineSplitCount, ...]
    task_results: tuple[BM25BaselineTaskResult, ...] = Field(min_length=1)
    summary: BaselineSummary
    artifact_paths: tuple[RelativePath, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.chunk_count != self.indexed_document_count:
            raise ValueError("BM25 index document count must match corpus chunks")
        task_ids = tuple(item.task_id for item in self.task_results)
        if task_ids != tuple(sorted(task_ids)) or len(set(task_ids)) != len(task_ids):
            raise ValueError("baseline task results must have unique sorted task IDs")
        if self.summary.task_count != len(self.task_results):
            raise ValueError("baseline summary task count does not match results")
        expected_splits = Counter(item.split for item in self.task_results)
        actual_splits = {item.split: item.task_count for item in self.split_counts}
        if actual_splits != dict(expected_splits):
            raise ValueError("baseline split counts do not match task results")
        split_order = tuple(item.split.value for item in self.split_counts)
        if split_order != tuple(sorted(split_order)):
            raise ValueError("baseline split counts must be sorted")
        if len(set(self.artifact_paths)) != len(self.artifact_paths):
            raise ValueError("baseline artifact paths must be unique")
        return self


@dataclass(frozen=True)
class BM25BaselineArtifacts:
    """In-memory baseline output and every persisted artifact path."""

    report: BM25BaselineReport
    corpus: ChunkCorpus
    output_dir: Path
    corpus_path: Path
    index_path: Path
    task_jsonl_path: Path
    task_csv_path: Path
    summary_path: Path
    report_path: Path


def load_bm25_baseline_manifest(path: Path) -> BM25BaselineManifest:
    """Read a versioned evaluator manifest without echoing rejected values."""

    try:
        serialized = path.read_text(encoding="utf-8")
    except UnicodeError:
        raise LocalizationEvalError(LocalizationEvalErrorCode.INVALID_MANIFEST) from None
    except OSError:
        raise LocalizationEvalError(LocalizationEvalErrorCode.INPUT_IO_ERROR) from None
    try:
        raw = json.loads(serialized)
    except json.JSONDecodeError:
        raise LocalizationEvalError(LocalizationEvalErrorCode.INVALID_MANIFEST) from None
    if not isinstance(raw, dict) or "schema_version" not in raw:
        raise LocalizationEvalError(LocalizationEvalErrorCode.INVALID_MANIFEST)
    if raw["schema_version"] != BM25_BASELINE_SCHEMA_VERSION:
        raise LocalizationEvalError(LocalizationEvalErrorCode.UNSUPPORTED_VERSION)
    try:
        return BM25BaselineManifest.model_validate(raw)
    except ValidationError as error:
        identity_messages = {
            "Value error, baseline task commits must match the repository commit",
            "Value error, baseline tasks must use the manifest repository root",
        }
        error_code = (
            LocalizationEvalErrorCode.IDENTITY_MISMATCH
            if any(item["msg"] in identity_messages for item in error.errors())
            else LocalizationEvalErrorCode.INVALID_MANIFEST
        )
        raise LocalizationEvalError(
            error_code,
            detail=_safe_validation_detail(error),
        ) from None


def build_localization_query(task: TaskSpec) -> str:
    """Construct the v1 query from Agent-visible Issue fields only."""

    return f"{task.issue_title}\n{task.issue_body}"


def run_bm25_baseline(
    manifest_path: Path,
    output_dir: Path,
    settings: RepoPilotSettings,
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> BM25BaselineArtifacts:
    """Load one evaluator manifest and execute its fixed local baseline."""

    manifest = load_bm25_baseline_manifest(manifest_path)
    repository_root = _resolve_repository_root(
        manifest_path,
        manifest.repository_root,
    )
    return run_bm25_baseline_manifest(
        manifest,
        repository_root=repository_root,
        output_dir=output_dir,
        settings=settings,
        clock_ns=clock_ns,
    )


def run_bm25_baseline_manifest(
    manifest: BM25BaselineManifest,
    *,
    repository_root: Path,
    output_dir: Path,
    settings: RepoPilotSettings,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> BM25BaselineArtifacts:
    """Build one shared index, retrieve each Agent-only query, then score with gold."""

    try:
        repository = ReadOnlyRepository.from_settings(repository_root, settings)
        indexing_started = clock_ns()
        repository_manifest = ingest_repository(
            repository,
            repository_id=manifest.repository_id,
            commit=manifest.commit,
        )
        corpus = (
            RepositoryChunkCorpusBuilder.from_settings(
                repository,
                settings,
            )
            .build(repository_manifest)
            .corpus
        )
        index = build_bm25_index(
            corpus,
            policy=BM25Policy.from_settings(settings),
        )
        verify_bm25_index_corpus(index, corpus)
        indexing_latency_ms = _elapsed_ms(indexing_started, clock_ns())
        retriever = BM25Retriever(index)
    except (RepositoryAccessError, RepositoryIngestError, ChunkCorpusError, BM25Error):
        raise LocalizationEvalError(LocalizationEvalErrorCode.RETRIEVAL_FAILURE) from None

    indexed_paths = {chunk.relative_path for chunk in corpus.chunks}
    task_results: list[BM25BaselineTaskResult] = []
    for evaluation in manifest.tasks:
        task = evaluation.task
        query = build_localization_query(task)
        retrieval_started = clock_ns()
        try:
            retrieval = retriever.search(
                repository_id=manifest.repository_id,
                commit=manifest.commit,
                query=query,
                top_k=manifest.retrieval_depth,
            )
            retrieval_latency_ms = _elapsed_ms(retrieval_started, clock_ns())
            ranked_files = rank_retrieved_files(retrieval, corpus)
        except (BM25Error, ValueError):
            raise LocalizationEvalError(LocalizationEvalErrorCode.RETRIEVAL_FAILURE) from None

        gold_files = tuple(sorted(evaluation.gold_files))
        metrics = score_file_localization(
            gold_files=gold_files,
            ranked_files=ranked_files,
            indexing_latency_ms=indexing_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
        )
        task_results.append(
            BM25BaselineTaskResult(
                task_id=task.task_id,
                split=evaluation.split,
                query=query,
                gold_files=gold_files,
                missing_gold_files=tuple(path for path in gold_files if path not in indexed_paths),
                retrieved_chunk_count=len(retrieval.hits),
                ranked_files=ranked_files,
                metrics=metrics,
            )
        )

    aggregate = aggregate_localization_metrics(tuple(item.metrics for item in task_results))
    artifact_names = (
        manifest.corpus_filename,
        manifest.index_filename,
        manifest.task_jsonl_filename,
        manifest.task_csv_filename,
        manifest.summary_filename,
        manifest.report_filename,
    )
    report = BM25BaselineReport(
        benchmark_id=manifest.benchmark_id,
        repository_id=manifest.repository_id,
        commit=manifest.commit,
        configuration=_configuration(settings, manifest),
        source_count=len(corpus.sources),
        chunk_count=len(corpus.chunks),
        indexed_document_count=index.document_count,
        corpus_sha256=index.corpus_sha256,
        index_sha256=index.content_sha256,
        split_counts=tuple(
            BaselineSplitCount(split=split, task_count=count)
            for split, count in sorted(
                Counter(item.split for item in task_results).items(),
                key=lambda item: item[0].value,
            )
        ),
        task_results=tuple(task_results),
        summary=BaselineSummary.model_validate(aggregate.model_dump()),
        artifact_paths=artifact_names,
    )
    resolved_output_dir, paths = _prepare_output_paths(output_dir, artifact_names)
    try:
        write_chunk_corpus(paths[0], corpus)
        write_bm25_index(paths[1], index)
        _write_text(paths[2], _render_task_jsonl(report.task_results))
        _write_text(paths[3], _render_task_csv(report.task_results))
        _write_text(paths[4], _render_summary_markdown(report))
        _write_text(paths[5], f"{report.model_dump_json(indent=2)}\n")
    except (ChunkCorpusError, BM25Error, OSError):
        raise LocalizationEvalError(LocalizationEvalErrorCode.OUTPUT_IO_ERROR) from None
    return BM25BaselineArtifacts(
        report=report,
        corpus=corpus,
        output_dir=resolved_output_dir,
        corpus_path=paths[0],
        index_path=paths[1],
        task_jsonl_path=paths[2],
        task_csv_path=paths[3],
        summary_path=paths[4],
        report_path=paths[5],
    )


def _resolve_repository_root(manifest_path: Path, relative_root: str) -> Path:
    try:
        fixture_root = manifest_path.resolve(strict=True).parent
        repository_root = (fixture_root / relative_root).resolve(strict=True)
    except OSError:
        raise LocalizationEvalError(LocalizationEvalErrorCode.INPUT_IO_ERROR) from None
    if not repository_root.is_relative_to(fixture_root):
        raise LocalizationEvalError(LocalizationEvalErrorCode.PATH_VIOLATION)
    return repository_root


def _prepare_output_paths(
    output_dir: Path,
    artifact_names: tuple[str, ...],
) -> tuple[Path, tuple[Path, ...]]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        root = output_dir.resolve(strict=True)
        paths: list[Path] = []
        for name in artifact_names:
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise LocalizationEvalError(LocalizationEvalErrorCode.PATH_VIOLATION)
            target.parent.mkdir(parents=True, exist_ok=True)
            paths.append(target)
        return root, tuple(paths)
    except LocalizationEvalError:
        raise
    except OSError:
        raise LocalizationEvalError(LocalizationEvalErrorCode.OUTPUT_IO_ERROR) from None


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    except OSError:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


def _render_task_jsonl(
    task_results: tuple[BM25BaselineTaskResult, ...],
) -> str:
    return "".join(f"{item.model_dump_json()}\n" for item in task_results)


def _render_task_csv(
    task_results: tuple[BM25BaselineTaskResult, ...],
) -> str:
    output = StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=(
            "task_id",
            "split",
            "query",
            "gold_files",
            "missing_gold_files",
            "ranked_files",
            "file_recall_at_1",
            "file_recall_at_3",
            "file_recall_at_5",
            "mean_reciprocal_rank",
            "first_relevant_file_rank",
            "indexing_latency_ms",
            "retrieval_latency_ms",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    for item in task_results:
        writer.writerow(
            {
                "task_id": item.task_id,
                "split": item.split.value,
                "query": item.query,
                "gold_files": json.dumps(
                    item.gold_files,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "missing_gold_files": json.dumps(
                    item.missing_gold_files,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "ranked_files": json.dumps(
                    [ranked.relative_path for ranked in item.ranked_files],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "file_recall_at_1": item.metrics.file_recall_at_1,
                "file_recall_at_3": item.metrics.file_recall_at_3,
                "file_recall_at_5": item.metrics.file_recall_at_5,
                "mean_reciprocal_rank": item.metrics.mean_reciprocal_rank,
                "first_relevant_file_rank": (item.metrics.first_relevant_file_rank or ""),
                "indexing_latency_ms": item.metrics.indexing_latency_ms,
                "retrieval_latency_ms": item.metrics.retrieval_latency_ms,
            }
        )
    return output.getvalue()


def _render_summary_markdown(report: BM25BaselineReport) -> str:
    summary = report.summary
    lines = [
        f"# RepoPilot BM25 Baseline: {report.benchmark_id}",
        "",
        "## Run",
        "",
        f"- Repository: `{report.repository_id}`",
        f"- Commit: `{report.commit}`",
        f"- Tasks: {summary.task_count}",
        f"- Sources / chunks: {report.source_count} / {report.chunk_count}",
        f"- Retrieval depth: {report.configuration.retrieval_depth} chunks",
        f"- BM25 k1 / b: {report.configuration.bm25_k1} / {report.configuration.bm25_b}",
        f"- Indexing latency: {summary.indexing_latency_ms} ms",
        f"- Mean retrieval latency: {summary.mean_retrieval_latency_ms:.3f} ms",
        "",
        "## Aggregate localization",
        "",
        "| Recall@1 | Recall@3 | Recall@5 | MRR | Relevant tasks |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {summary.mean_file_recall_at_1:.6f} "
            f"| {summary.mean_file_recall_at_3:.6f} "
            f"| {summary.mean_file_recall_at_5:.6f} "
            f"| {summary.mean_reciprocal_rank:.6f} "
            f"| {summary.tasks_with_relevant_file}/{summary.task_count} |"
        ),
        "",
        "## Tasks",
        "",
        "| Task | Split | First relevant rank | R@1 | R@3 | R@5 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in report.task_results:
        first_rank = item.metrics.first_relevant_file_rank
        lines.append(
            f"| `{item.task_id}` | {item.split.value} "
            f"| {first_rank if first_rank is not None else '-'} "
            f"| {item.metrics.file_recall_at_1:.6f} "
            f"| {item.metrics.file_recall_at_3:.6f} "
            f"| {item.metrics.file_recall_at_5:.6f} |"
        )
    lines.extend(
        (
            "",
            "> This report is evaluator-owned. Gold files are used only after retrieval;",
            "> expected patch content is never included in the query, corpus, index, or report.",
            "",
        )
    )
    return "\n".join(lines)


def _configuration(
    settings: RepoPilotSettings,
    manifest: BM25BaselineManifest,
) -> BaselineConfiguration:
    return BaselineConfiguration(
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        bm25_default_top_k=settings.bm25_default_top_k,
        retrieval_depth=manifest.retrieval_depth,
        chunk_max_lines=settings.chunk_max_lines,
        chunk_overlap_lines=settings.chunk_overlap_lines,
        repository_max_file_bytes=settings.repository_max_file_bytes,
        repository_max_list_entries=settings.repository_max_list_entries,
    )


def _elapsed_ms(start_ns: int, end_ns: int) -> int:
    if end_ns < start_ns:
        raise LocalizationEvalError(LocalizationEvalErrorCode.RETRIEVAL_FAILURE)
    return (end_ns - start_ns) // 1_000_000


def _safe_validation_detail(error: ValidationError) -> str:
    messages: list[str] = []
    for item in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        location = ".".join(str(part) for part in item["loc"]) or "root"
        messages.append(f"{location}: {item['msg']}")
    return "; ".join(messages)
