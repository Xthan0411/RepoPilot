"""Generate the deterministic P3 retrieval-mechanics comparison artifact."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from repopilot.embeddings import (
    EmbeddingBatchExecutor,
    EmbeddingIdentity,
    EmbeddingIndexer,
    FakeEmbeddingProvider,
    LocalEmbeddingCache,
)
from repopilot.enums import EmbeddingProviderKind, RetrievalMode
from repopilot.evals import (
    build_localization_query,
    evaluate_retrieval_comparison,
    load_bm25_baseline_manifest,
    write_retrieval_comparison,
)
from repopilot.retrieval import (
    BM25Retriever,
    DenseRetriever,
    FakeQueryRewriteProvider,
    HybridPolicy,
    HybridRetriever,
    QueryRewriteController,
    QueryRewritePolicy,
    QueryRewriteProposal,
    read_bm25_index,
    read_chunk_corpus,
)
from repopilot.storage import InMemoryVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "tests/fixtures/p2_bm25_baseline_manifest.json"
CORPUS_PATH = PROJECT_ROOT / "results/p2-baseline/chunks.jsonl"
BM25_INDEX_PATH = PROJECT_ROOT / "results/p2-baseline/bm25-index.json"
OUTPUT_DIR = PROJECT_ROOT / "results/p3-retrieval-comparison"


def main() -> None:
    """Run only deterministic Fakes; this command makes no network/model call."""

    manifest = load_bm25_baseline_manifest(MANIFEST_PATH)
    corpus = read_chunk_corpus(CORPUS_PATH)
    sparse = BM25Retriever(read_bm25_index(BM25_INDEX_PATH))
    identity = EmbeddingIdentity(
        provider=EmbeddingProviderKind.FAKE,
        model="fake-p3-comparison",
        revision="v1",
        dimension=16,
        normalize=True,
        max_length=512,
    )
    provider = FakeEmbeddingProvider(identity)
    executor = EmbeddingBatchExecutor(
        provider,
        batch_size=16,
        max_retries=0,
        retry_backoff_ms=0,
    )
    store = InMemoryVectorStore(dimension=identity.dimension)
    with TemporaryDirectory(prefix="repopilot-p3-smoke-") as cache_directory:
        EmbeddingIndexer(
            executor=executor,
            cache=LocalEmbeddingCache(Path(cache_directory)),
            vector_store=store,
        ).index(corpus)
        dense = DenseRetriever(
            repository_id=corpus.repository_id,
            commit=corpus.commit,
            executor=executor,
            vector_store=store,
        )
        retrievers = {
            mode: HybridRetriever(
                corpus=corpus,
                sparse_retriever=(
                    sparse if mode in (RetrievalMode.BM25, RetrievalMode.HYBRID) else None
                ),
                dense_retriever=(
                    dense if mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID) else None
                ),
                policy=HybridPolicy(
                    mode=mode,
                    default_top_k=manifest.retrieval_depth,
                    candidate_top_k=manifest.retrieval_depth,
                    max_query_length=512,
                ),
            )
            for mode in (
                RetrievalMode.BM25,
                RetrievalMode.DENSE,
                RetrievalMode.HYBRID,
            )
        }
        first_task = manifest.tasks[0].task
        original_query = build_localization_query(first_task)
        initial = retrievers[RetrievalMode.BM25].search(
            original_query,
            top_k=manifest.retrieval_depth,
        )
        rewrite = QueryRewriteController(
            policy=QueryRewritePolicy(
                enabled=True,
                minimum_initial_hits=100,
                max_context_chunks=5,
                max_query_length=512,
            ),
            provider=FakeQueryRewriteProvider(
                QueryRewriteProposal(
                    implementation_query="parse empty configuration mapping implementation",
                    test_query="test parse empty configuration mapping",
                    symbols=("parse",),
                    error_terms=("empty configuration",),
                    behavior_constraints=("return empty mapping",),
                    rationale="Separate implementation and regression-test retrieval intent.",
                )
            ),
        ).rewrite_if_needed(
            original_query,
            initial_result=initial,
            corpus=corpus,
        )
        report = evaluate_retrieval_comparison(
            manifest,
            corpus=corpus,
            retrievers=retrievers,
            indexing_latency_ms={
                RetrievalMode.BM25: 0,
                RetrievalMode.DENSE: 0,
                RetrievalMode.HYBRID: 0,
            },
            rewrites={first_task.task_id: rewrite},
            evidence_kind="deterministic_fake_smoke",
            clock_ns=lambda: 0,
        )
        artifacts = write_retrieval_comparison(report, OUTPUT_DIR)
    print(artifacts.markdown_path)


if __name__ == "__main__":
    main()
