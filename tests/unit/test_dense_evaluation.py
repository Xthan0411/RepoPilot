"""Dense evaluation parity, call accounting, isolation, and error tests."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from repopilot.config import RepoPilotSettings
from repopilot.embeddings import (
    EmbeddingBatchExecutor,
    EmbeddingIdentity,
    EmbeddingIndexer,
    FakeEmbeddingProvider,
    LocalEmbeddingCache,
)
from repopilot.enums import DenseEvalErrorCode, EmbeddingProviderKind
from repopilot.evals import (
    DenseEvaluationError,
    build_localization_query,
    evaluate_dense_localization,
    load_bm25_baseline_manifest,
    run_bm25_baseline,
)
from repopilot.retrieval import DenseRetriever
from repopilot.storage import InMemoryVectorStore

FIXTURES = Path(__file__).parents[1] / "fixtures"
MANIFEST_PATH = FIXTURES / "p2_bm25_baseline_manifest.json"
PATCH_CANARY = "EVALUATOR_ONLY_PATCH_CANARY"


def clock(*values: int) -> Callable[[], int]:
    ticks: Iterator[int] = iter(values)
    return lambda: next(ticks)


def test_dense_evaluation_reuses_p2_tasks_and_accounts_for_every_provider_call(
    tmp_path: Path,
) -> None:
    manifest = load_bm25_baseline_manifest(MANIFEST_PATH)
    baseline = run_bm25_baseline(
        MANIFEST_PATH,
        tmp_path / "baseline",
        RepoPilotSettings(),
        clock_ns=clock(0, 1, 2, 3, 4, 5, 6, 7),
    )
    identity = EmbeddingIdentity(
        provider=EmbeddingProviderKind.FAKE,
        model="fake-dense-evaluation",
        revision="v1",
        dimension=16,
        normalize=True,
        max_length=256,
    )
    provider = FakeEmbeddingProvider(identity)
    executor = EmbeddingBatchExecutor(
        provider,
        batch_size=4,
        max_retries=1,
        retry_backoff_ms=0,
        sleep=lambda _: None,
    )
    store = InMemoryVectorStore(dimension=identity.dimension)
    index_result = EmbeddingIndexer(
        executor=executor,
        cache=LocalEmbeddingCache(tmp_path / "embedding-cache"),
        vector_store=store,
    ).index(baseline.corpus)
    retriever = DenseRetriever(
        repository_id=manifest.repository_id,
        commit=manifest.commit,
        executor=executor,
        vector_store=store,
    )

    report = evaluate_dense_localization(
        manifest,
        corpus=baseline.corpus,
        index_result=index_result,
        retriever=retriever,
        indexing_latency_ms=7,
        clock_ns=clock(
            0,
            1_000_000,
            10_000_000,
            12_000_000,
            20_000_000,
            23_000_000,
        ),
    )

    assert report.run_mode == "p3_dense_evaluation"
    assert report.identity == identity
    assert report.retrieval_depth == manifest.retrieval_depth
    assert tuple(item.task_id for item in report.task_results) == tuple(
        item.task.task_id for item in manifest.tasks
    )
    assert tuple(item.query for item in report.task_results) == tuple(
        build_localization_query(item.task) for item in manifest.tasks
    )
    assert tuple((item.split.value, item.task_count) for item in report.split_counts) == (
        ("dev", 2),
        ("holdout", 1),
    )
    assert report.summary.task_count == 3
    assert report.summary.indexing_latency_ms == 7
    assert report.summary.total_retrieval_latency_ms == 6
    assert report.summary.mean_retrieval_latency_ms == 2.0
    assert report.index_provider_call_count == index_result.provider_call_count
    assert report.query_provider_call_count == len(manifest.tasks)
    assert report.provider_call_count == provider.calls_made
    assert report.index_retry_count == 0
    assert report.query_retry_count == 0
    assert report.retry_count == 0
    assert all(item.missing_gold_files == () for item in report.task_results)
    assert PATCH_CANARY not in report.model_dump_json()


def test_dense_evaluation_rejects_identity_mismatch_before_retrieval(
    tmp_path: Path,
) -> None:
    manifest = load_bm25_baseline_manifest(MANIFEST_PATH)
    baseline = run_bm25_baseline(
        MANIFEST_PATH,
        tmp_path / "baseline",
        RepoPilotSettings(),
        clock_ns=clock(0, 1, 2, 3, 4, 5, 6, 7),
    )
    identity = EmbeddingIdentity(
        provider=EmbeddingProviderKind.FAKE,
        model="fake-dense-evaluation",
        revision="v1",
        dimension=8,
        normalize=True,
        max_length=256,
    )
    provider = FakeEmbeddingProvider(identity)
    executor = EmbeddingBatchExecutor(
        provider,
        batch_size=16,
        max_retries=0,
        retry_backoff_ms=0,
    )
    store = InMemoryVectorStore(dimension=identity.dimension)
    index_result = EmbeddingIndexer(
        executor=executor,
        cache=LocalEmbeddingCache(tmp_path / "embedding-cache"),
        vector_store=store,
    ).index(baseline.corpus)
    retriever = DenseRetriever(
        repository_id=manifest.repository_id,
        commit=manifest.commit,
        executor=executor,
        vector_store=store,
    )
    mismatched_corpus = baseline.corpus.model_copy(update={"repository_id": "fixture.other"})

    with pytest.raises(DenseEvaluationError) as captured:
        evaluate_dense_localization(
            manifest,
            corpus=mismatched_corpus,
            index_result=index_result,
            retriever=retriever,
            indexing_latency_ms=0,
            clock_ns=clock(),
        )

    assert captured.value.code is DenseEvalErrorCode.IDENTITY_MISMATCH
    assert provider.calls_made == index_result.provider_call_count


def test_dense_evaluation_rejects_backwards_clock_as_invalid_result(
    tmp_path: Path,
) -> None:
    manifest = load_bm25_baseline_manifest(MANIFEST_PATH)
    baseline = run_bm25_baseline(
        MANIFEST_PATH,
        tmp_path / "baseline",
        RepoPilotSettings(),
        clock_ns=clock(0, 1, 2, 3, 4, 5, 6, 7),
    )
    identity = EmbeddingIdentity(
        provider=EmbeddingProviderKind.FAKE,
        model="fake-dense-evaluation",
        revision="v1",
        dimension=8,
        normalize=True,
        max_length=256,
    )
    provider = FakeEmbeddingProvider(identity)
    executor = EmbeddingBatchExecutor(
        provider,
        batch_size=16,
        max_retries=0,
        retry_backoff_ms=0,
    )
    store = InMemoryVectorStore(dimension=identity.dimension)
    index_result = EmbeddingIndexer(
        executor=executor,
        cache=LocalEmbeddingCache(tmp_path / "embedding-cache"),
        vector_store=store,
    ).index(baseline.corpus)
    retriever = DenseRetriever(
        repository_id=manifest.repository_id,
        commit=manifest.commit,
        executor=executor,
        vector_store=store,
    )

    with pytest.raises(DenseEvaluationError) as captured:
        evaluate_dense_localization(
            manifest,
            corpus=baseline.corpus,
            index_result=index_result,
            retriever=retriever,
            indexing_latency_ms=0,
            clock_ns=clock(2, 1),
        )

    assert captured.value.code is DenseEvalErrorCode.INVALID_RESULT
