"""Namespace-isolated cosine Dense Retriever with query Embedding."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Self

from pydantic import Field, model_validator

from repopilot.config import RepoPilotSettings
from repopilot.embeddings import (
    EmbeddingBatchExecutor,
    EmbeddingError,
    EmbeddingIdentity,
    EmbeddingInput,
)
from repopilot.enums import (
    DenseRetrievalErrorCode,
    EmbeddingErrorCode,
    VectorStoreErrorCode,
)
from repopilot.schemas import GitSha, Identifier, NonEmptyText, RetrievalHit, StrictModel
from repopilot.storage import (
    VectorNamespace,
    VectorSearchHit,
    VectorStore,
    VectorStoreError,
    validate_chunk_allowlist,
)

DENSE_ERROR_MESSAGES = {
    DenseRetrievalErrorCode.EMPTY_QUERY: "Dense query must not be empty",
    DenseRetrievalErrorCode.INVALID_QUERY: "Dense query or Top-K is invalid",
    DenseRetrievalErrorCode.IDENTITY_MISMATCH: (
        "Dense query identity does not match the vector index"
    ),
    DenseRetrievalErrorCode.EMBEDDING_FAILURE: "Dense query embedding failed",
    DenseRetrievalErrorCode.STORAGE_FAILURE: "Dense vector search failed",
    DenseRetrievalErrorCode.INVALID_RESULT: "Dense backend returned an invalid result",
}


class DenseRetrievalError(RuntimeError):
    """Safe error retaining the failing pipeline layer and typed cause."""

    def __init__(
        self,
        code: DenseRetrievalErrorCode,
        *,
        embedding_code: EmbeddingErrorCode | None = None,
        storage_code: VectorStoreErrorCode | None = None,
    ) -> None:
        super().__init__(DENSE_ERROR_MESSAGES[code])
        self.code = code
        self.embedding_code = embedding_code
        self.storage_code = storage_code


class DensePolicy(StrictModel):
    """Validated query limits independent of a concrete vector backend."""

    default_top_k: int = Field(default=10, gt=0, le=100)
    max_query_length: int = Field(default=256, gt=0, le=2_000)

    @classmethod
    def from_settings(cls, settings: RepoPilotSettings) -> DensePolicy:
        return cls(
            default_top_k=settings.dense_default_top_k,
            max_query_length=settings.search_max_query_length,
        )


class DenseSearchResult(StrictModel):
    """Normalized Dense hits for one exact query and vector namespace."""

    repository_id: Identifier
    commit: GitSha
    identity: EmbeddingIdentity
    namespace: VectorNamespace
    query: NonEmptyText
    top_k: int = Field(gt=0, le=100)
    hits: tuple[RetrievalHit, ...]
    provider_call_count: int = Field(gt=0)
    retry_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if (
            self.namespace.repository_id != self.repository_id
            or self.namespace.commit != self.commit
            or self.namespace.embedding_model != self.identity.model
            or self.namespace.embedding_version != self.identity.fingerprint
            or self.namespace.dimension != self.identity.dimension
        ):
            raise ValueError("Dense result identity is inconsistent")
        chunk_ids = tuple(hit.chunk_id for hit in self.hits)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Dense result chunk IDs must be unique")
        if len(self.hits) > self.top_k:
            raise ValueError("Dense hits cannot exceed requested Top-K")
        for expected_rank, hit in enumerate(self.hits, start=1):
            if (
                hit.dense_rank != expected_rank
                or hit.dense_score is None
                or hit.sparse_rank is not None
                or hit.sparse_score is not None
                or hit.fused_score is not None
                or hit.query != self.query
            ):
                raise ValueError("Dense hits must use contiguous Dense-only ranks")
        order = tuple(
            (-float(hit.dense_score), hit.chunk_id)
            for hit in self.hits
            if hit.dense_score is not None
        )
        if order != tuple(sorted(order)):
            raise ValueError("Dense hits must use deterministic score order")
        if self.provider_call_count != 1 + self.retry_count:
            raise ValueError("Dense query calls must equal one batch plus retries")
        return self


class DenseRetriever:
    """Embed a query once and normalize namespace-scoped cosine results."""

    def __init__(
        self,
        *,
        repository_id: str,
        commit: str,
        executor: EmbeddingBatchExecutor,
        vector_store: VectorStore,
        policy: DensePolicy | None = None,
    ) -> None:
        self._executor = executor
        self._vector_store = vector_store
        self._policy = policy or DensePolicy()
        self._store_ready = False
        try:
            self._namespace = VectorNamespace(
                repository_id=repository_id,
                commit=commit,
                embedding_model=executor.identity.model,
                embedding_version=executor.identity.fingerprint,
                dimension=executor.identity.dimension,
            )
        except (TypeError, ValueError):
            raise DenseRetrievalError(DenseRetrievalErrorCode.IDENTITY_MISMATCH) from None

    @property
    def namespace(self) -> VectorNamespace:
        return self._namespace

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        allowed_chunk_ids: Sequence[str] | None = None,
    ) -> DenseSearchResult:
        if not isinstance(query, str):
            raise DenseRetrievalError(DenseRetrievalErrorCode.INVALID_QUERY)
        normalized_query = query.strip()
        if not normalized_query:
            raise DenseRetrievalError(DenseRetrievalErrorCode.EMPTY_QUERY)
        requested_top_k = self._policy.default_top_k if top_k is None else top_k
        if (
            len(normalized_query) > self._policy.max_query_length
            or isinstance(requested_top_k, bool)
            or not isinstance(requested_top_k, int)
            or not 1 <= requested_top_k <= 100
        ):
            raise DenseRetrievalError(DenseRetrievalErrorCode.INVALID_QUERY)
        try:
            allowed = validate_chunk_allowlist(allowed_chunk_ids)
        except VectorStoreError:
            raise DenseRetrievalError(DenseRetrievalErrorCode.INVALID_QUERY) from None
        self._ensure_store_ready()

        query_input = EmbeddingInput(
            content_hash=hashlib.sha256(normalized_query.encode("utf-8")).hexdigest(),
            content=normalized_query,
        )
        try:
            execution = self._executor.embed((query_input,))
        except EmbeddingError as error:
            raise DenseRetrievalError(
                DenseRetrievalErrorCode.EMBEDDING_FAILURE,
                embedding_code=error.code,
            ) from None
        if (
            execution.identity != self._executor.identity
            or len(execution.vectors) != 1
            or execution.vectors[0].content_hash != query_input.content_hash
        ):
            raise DenseRetrievalError(DenseRetrievalErrorCode.INVALID_RESULT)

        try:
            vector_hits = self._vector_store.search(
                self._namespace,
                execution.vectors[0].values,
                top_k=requested_top_k,
                allowed_chunk_ids=allowed,
            )
        except DenseRetrievalError:
            raise
        except VectorStoreError as error:
            raise DenseRetrievalError(
                DenseRetrievalErrorCode.STORAGE_FAILURE,
                storage_code=error.code,
            ) from None
        _validate_vector_hits(self._namespace, vector_hits, requested_top_k)
        try:
            hits = tuple(
                RetrievalHit(
                    chunk_id=hit.chunk_id,
                    dense_rank=rank,
                    dense_score=hit.score,
                    query=normalized_query,
                    retrieval_reason=(
                        f"cosine_similarity={hit.score:.9g}; "
                        f"path={hit.relative_path}; language={hit.language}; "
                        f"model={self._executor.identity.model}; "
                        f"revision={self._executor.identity.revision}"
                    ),
                )
                for rank, hit in enumerate(vector_hits, start=1)
            )
            return DenseSearchResult(
                repository_id=self._namespace.repository_id,
                commit=self._namespace.commit,
                identity=self._executor.identity,
                namespace=self._namespace,
                query=normalized_query,
                top_k=requested_top_k,
                hits=hits,
                provider_call_count=execution.provider_call_count,
                retry_count=execution.retry_count,
            )
        except (TypeError, ValueError):
            raise DenseRetrievalError(DenseRetrievalErrorCode.INVALID_RESULT) from None

    def _ensure_store_ready(self) -> None:
        if self._store_ready:
            return
        try:
            store_info = self._vector_store.initialize()
        except VectorStoreError as error:
            raise DenseRetrievalError(
                DenseRetrievalErrorCode.STORAGE_FAILURE,
                storage_code=error.code,
            ) from None
        if store_info.dimension != self._namespace.dimension:
            raise DenseRetrievalError(DenseRetrievalErrorCode.IDENTITY_MISMATCH)
        self._store_ready = True


def _validate_vector_hits(
    namespace: VectorNamespace,
    hits: tuple[VectorSearchHit, ...],
    top_k: int,
) -> None:
    if len(hits) > top_k or any(hit.namespace != namespace for hit in hits):
        raise DenseRetrievalError(DenseRetrievalErrorCode.INVALID_RESULT)
    chunk_ids = tuple(hit.chunk_id for hit in hits)
    if len(set(chunk_ids)) != len(chunk_ids):
        raise DenseRetrievalError(DenseRetrievalErrorCode.INVALID_RESULT)
    expected_order = tuple(sorted(hits, key=lambda hit: (-hit.score, hit.chunk_id)))
    if hits != expected_order:
        raise DenseRetrievalError(DenseRetrievalErrorCode.INVALID_RESULT)
