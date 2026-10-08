"""Dense query Embedding, cosine normalization, isolation, and error tests."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import pytest

from repopilot.config import RepoPilotSettings
from repopilot.embeddings import (
    EmbeddingBatchExecutor,
    EmbeddingIdentity,
    EmbeddingInput,
    FakeEmbeddingProvider,
)
from repopilot.enums import (
    DenseRetrievalErrorCode,
    EmbeddingErrorCode,
    EmbeddingProviderKind,
)
from repopilot.retrieval import (
    DensePolicy,
    DenseRetrievalError,
    DenseRetriever,
)
from repopilot.storage import (
    InMemoryVectorStore,
    VectorNamespace,
    VectorRecord,
    VectorSearchHit,
)

COMMIT = "a" * 40
QUERY = "find parser validation"


def identity() -> EmbeddingIdentity:
    return EmbeddingIdentity(
        provider=EmbeddingProviderKind.FAKE,
        model="fake-code-embedding",
        revision="v1",
        dimension=3,
        normalize=True,
        max_length=128,
    )


def executor(
    provider: FakeEmbeddingProvider,
    *,
    max_retries: int = 0,
) -> EmbeddingBatchExecutor:
    return EmbeddingBatchExecutor(
        provider,
        batch_size=8,
        max_retries=max_retries,
        retry_backoff_ms=0,
        sleep=lambda _: None,
    )


def query_vector(model_identity: EmbeddingIdentity, query: str) -> tuple[float, ...]:
    item = EmbeddingInput(
        content_hash=hashlib.sha256(query.encode("utf-8")).hexdigest(),
        content=query,
    )
    provider = FakeEmbeddingProvider(model_identity)
    result = executor(provider).embed((item,))
    return result.vectors[0].values


def orthogonal(vector: tuple[float, ...]) -> tuple[float, ...]:
    candidate = (-vector[1], vector[0], 0.0)
    if candidate == (0.0, 0.0, 0.0):
        candidate = (0.0, -vector[2], vector[1])
    norm = math.sqrt(math.fsum(value * value for value in candidate))
    return tuple(value / norm for value in candidate)


def vector_record(
    namespace: VectorNamespace,
    chunk_id: str,
    embedding: tuple[float, ...],
) -> VectorRecord:
    return VectorRecord(
        namespace=namespace,
        chunk_id=chunk_id,
        content_hash=hashlib.sha256(chunk_id.encode("utf-8")).hexdigest(),
        relative_path=f"pkg/{chunk_id}.py",
        language="python",
        symbol_name=chunk_id,
        embedding=embedding,
        metadata={"fixture": True},
    )


def prepared_store(
    model_identity: EmbeddingIdentity,
) -> tuple[InMemoryVectorStore, VectorNamespace]:
    namespace = VectorNamespace(
        repository_id="fixture.dense",
        commit=COMMIT,
        embedding_model=model_identity.model,
        embedding_version=model_identity.fingerprint,
        dimension=model_identity.dimension,
    )
    vector = query_vector(model_identity, QUERY)
    perpendicular = orthogonal(vector)
    store = InMemoryVectorStore(dimension=3)
    store.initialize()
    store.upsert(
        namespace,
        (
            vector_record(namespace, "chunk_z", vector),
            vector_record(namespace, "chunk_a", vector),
            vector_record(namespace, "chunk_b", perpendicular),
            vector_record(
                namespace,
                "chunk_c",
                tuple(-value for value in vector),
            ),
        ),
    )
    other = namespace.model_copy(update={"repository_id": "fixture.other"})
    store.upsert(other, (vector_record(other, "chunk_other", vector),))
    return store, namespace


def assert_dense_error(
    captured: pytest.ExceptionInfo[DenseRetrievalError],
    code: DenseRetrievalErrorCode,
) -> None:
    assert captured.value.code is code


def test_dense_policy_uses_validated_settings() -> None:
    policy = DensePolicy.from_settings(
        RepoPilotSettings(
            dense_default_top_k=25,
            search_max_query_length=512,
        )
    )

    assert policy.default_top_k == 25
    assert policy.max_query_length == 512


def test_dense_retriever_embeds_once_ranks_ties_and_normalizes_hits() -> None:
    model_identity = identity()
    store, namespace = prepared_store(model_identity)
    provider = FakeEmbeddingProvider(model_identity)
    retriever = DenseRetriever(
        repository_id=namespace.repository_id,
        commit=namespace.commit,
        executor=executor(provider),
        vector_store=store,
        policy=DensePolicy(default_top_k=3, max_query_length=256),
    )

    result = retriever.search(f"  {QUERY}  ")

    assert result.namespace == namespace
    assert result.query == QUERY
    assert result.top_k == 3
    assert tuple(hit.chunk_id for hit in result.hits) == (
        "chunk_a",
        "chunk_z",
        "chunk_b",
    )
    assert tuple(hit.dense_rank for hit in result.hits) == (1, 2, 3)
    assert tuple(hit.dense_score for hit in result.hits) == pytest.approx(
        (1.0, 1.0, 0.0),
        abs=1e-12,
    )
    assert all(hit.sparse_rank is None for hit in result.hits)
    assert all("gold" not in hit.retrieval_reason for hit in result.hits)
    assert result.provider_call_count == 1
    assert result.retry_count == 0
    assert provider.calls_made == 1


@pytest.mark.parametrize(
    ("query", "top_k", "code"),
    [
        ("  ", None, DenseRetrievalErrorCode.EMPTY_QUERY),
        ("x" * 257, None, DenseRetrievalErrorCode.INVALID_QUERY),
        ("valid", 0, DenseRetrievalErrorCode.INVALID_QUERY),
        ("valid", 101, DenseRetrievalErrorCode.INVALID_QUERY),
        ("valid", True, DenseRetrievalErrorCode.INVALID_QUERY),
    ],
)
def test_dense_retriever_rejects_invalid_query_before_provider(
    query: str,
    top_k: int | None,
    code: DenseRetrievalErrorCode,
) -> None:
    model_identity = identity()
    store, namespace = prepared_store(model_identity)
    provider = FakeEmbeddingProvider(model_identity)
    retriever = DenseRetriever(
        repository_id=namespace.repository_id,
        commit=namespace.commit,
        executor=executor(provider),
        vector_store=store,
        policy=DensePolicy(max_query_length=256),
    )

    with pytest.raises(DenseRetrievalError) as captured:
        retriever.search(query, top_k=top_k)

    assert_dense_error(captured, code)
    assert provider.calls_made == 0


def test_dense_retriever_rejects_non_string_query_before_provider() -> None:
    model_identity = identity()
    store, namespace = prepared_store(model_identity)
    provider = FakeEmbeddingProvider(model_identity)
    retriever = DenseRetriever(
        repository_id=namespace.repository_id,
        commit=namespace.commit,
        executor=executor(provider),
        vector_store=store,
    )

    with pytest.raises(DenseRetrievalError) as captured:
        retriever.search(None)  # type: ignore[arg-type]

    assert_dense_error(captured, DenseRetrievalErrorCode.INVALID_QUERY)
    assert provider.calls_made == 0


def test_dense_retriever_reports_successful_query_retry() -> None:
    model_identity = identity()
    store, namespace = prepared_store(model_identity)
    provider = FakeEmbeddingProvider(model_identity, fail_first_calls=1)
    retriever = DenseRetriever(
        repository_id=namespace.repository_id,
        commit=namespace.commit,
        executor=executor(provider, max_retries=1),
        vector_store=store,
    )

    result = retriever.search(QUERY, top_k=1)

    assert result.provider_call_count == 2
    assert result.retry_count == 1
    assert provider.calls_made == 2


def test_dense_retriever_applies_allowlist_before_ranking() -> None:
    model_identity = identity()
    store, namespace = prepared_store(model_identity)
    retriever = DenseRetriever(
        repository_id=namespace.repository_id,
        commit=namespace.commit,
        executor=executor(FakeEmbeddingProvider(model_identity)),
        vector_store=store,
    )

    result = retriever.search(
        QUERY,
        top_k=3,
        allowed_chunk_ids=("chunk_c", "chunk_b"),
    )

    assert tuple(hit.chunk_id for hit in result.hits) == ("chunk_b", "chunk_c")
    assert tuple(hit.dense_rank for hit in result.hits) == (1, 2)


def test_dense_retriever_preserves_embedding_failure_code() -> None:
    model_identity = identity()
    store, namespace = prepared_store(model_identity)
    provider = FakeEmbeddingProvider(model_identity, fail_first_calls=100)
    retriever = DenseRetriever(
        repository_id=namespace.repository_id,
        commit=namespace.commit,
        executor=executor(provider),
        vector_store=store,
    )

    with pytest.raises(DenseRetrievalError) as captured:
        retriever.search(QUERY)

    assert_dense_error(captured, DenseRetrievalErrorCode.EMBEDDING_FAILURE)
    assert captured.value.embedding_code is EmbeddingErrorCode.RETRY_EXHAUSTED
    assert captured.value.storage_code is None


def test_dense_retriever_rejects_store_dimension_mismatch_before_embedding() -> None:
    model_identity = identity()
    provider = FakeEmbeddingProvider(model_identity)
    retriever = DenseRetriever(
        repository_id="fixture.dense",
        commit=COMMIT,
        executor=executor(provider),
        vector_store=InMemoryVectorStore(dimension=4),
    )

    with pytest.raises(DenseRetrievalError) as captured:
        retriever.search(QUERY)

    assert_dense_error(captured, DenseRetrievalErrorCode.IDENTITY_MISMATCH)
    assert provider.calls_made == 0


class ReversedSearchStore(InMemoryVectorStore):
    def search(
        self,
        namespace: VectorNamespace,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        allowed_chunk_ids: Sequence[str] | None = None,
    ) -> tuple[VectorSearchHit, ...]:
        hits = super().search(
            namespace,
            query_embedding,
            top_k=top_k,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        return tuple(reversed(hits))


def test_dense_retriever_rejects_backend_order_violation() -> None:
    model_identity = identity()
    vector = query_vector(model_identity, QUERY)
    namespace = VectorNamespace(
        repository_id="fixture.dense",
        commit=COMMIT,
        embedding_model=model_identity.model,
        embedding_version=model_identity.fingerprint,
        dimension=3,
    )
    store = ReversedSearchStore(dimension=3)
    store.initialize()
    store.upsert(
        namespace,
        (
            vector_record(namespace, "chunk_a", vector),
            vector_record(namespace, "chunk_b", orthogonal(vector)),
        ),
    )
    retriever = DenseRetriever(
        repository_id=namespace.repository_id,
        commit=namespace.commit,
        executor=executor(FakeEmbeddingProvider(model_identity)),
        vector_store=store,
    )

    with pytest.raises(DenseRetrievalError) as captured:
        retriever.search(QUERY)

    assert_dense_error(captured, DenseRetrievalErrorCode.INVALID_RESULT)
