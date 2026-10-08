"""Configurable BM25/Dense routing, metadata filters, and deterministic RRF."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Protocol, Self

from pydantic import Field, field_validator, model_validator

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    BM25ErrorCode,
    DenseRetrievalErrorCode,
    HybridRetrievalErrorCode,
    RetrievalMode,
    RetrievalTestScope,
)
from repopilot.retrieval.bm25 import (
    BM25Error,
    BM25Index,
    BM25SearchResult,
    verify_bm25_index_corpus,
)
from repopilot.retrieval.corpus import ChunkCorpus
from repopilot.retrieval.dense import (
    DenseRetrievalError,
    DenseSearchResult,
)
from repopilot.schemas import (
    GitSha,
    Identifier,
    NonEmptyText,
    RelativePath,
    RetrievalHit,
    StrictModel,
)
from repopilot.storage import VectorNamespace

HYBRID_ERROR_MESSAGES = {
    HybridRetrievalErrorCode.INVALID_CONFIGURATION: (
        "retrieval mode is missing a required backend"
    ),
    HybridRetrievalErrorCode.INVALID_QUERY: "retrieval query or Top-K is invalid",
    HybridRetrievalErrorCode.IDENTITY_MISMATCH: (
        "retrieval backend identity does not match the corpus"
    ),
    HybridRetrievalErrorCode.SPARSE_FAILURE: "BM25 retrieval failed",
    HybridRetrievalErrorCode.DENSE_FAILURE: "Dense retrieval failed",
    HybridRetrievalErrorCode.INVALID_RESULT: "retrieval backend or fusion result is invalid",
}


class HybridRetrievalError(RuntimeError):
    """Safe routing/fusion failure with an optional typed branch cause."""

    def __init__(
        self,
        code: HybridRetrievalErrorCode,
        *,
        sparse_code: BM25ErrorCode | None = None,
        dense_code: DenseRetrievalErrorCode | None = None,
    ) -> None:
        super().__init__(HYBRID_ERROR_MESSAGES[code])
        self.code = code
        self.sparse_code = sparse_code
        self.dense_code = dense_code


class RetrievalFilter(StrictModel):
    """Canonical metadata predicates compiled into a trusted chunk allowlist."""

    path_prefixes: tuple[RelativePath, ...] = ()
    languages: tuple[Identifier, ...] = ()
    test_scope: RetrievalTestScope = RetrievalTestScope.ALL

    @field_validator("path_prefixes")
    @classmethod
    def canonicalize_path_prefixes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(set(value)))

    @field_validator("languages")
    @classmethod
    def canonicalize_languages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({language.casefold() for language in value}))


class HybridPolicy(StrictModel):
    """Validated mode, output depth, candidate depth, and RRF constant."""

    mode: RetrievalMode = RetrievalMode.BM25
    default_top_k: int = Field(default=10, gt=0, le=100)
    candidate_top_k: int = Field(default=100, gt=0, le=100)
    rrf_k: int = Field(default=60, gt=0, le=1_000)
    max_query_length: int = Field(default=256, gt=0, le=2_000)

    @model_validator(mode="after")
    def validate_depths(self) -> Self:
        if self.candidate_top_k < self.default_top_k:
            raise ValueError("Hybrid candidate depth must cover default Top-K")
        return self

    @classmethod
    def from_settings(cls, settings: RepoPilotSettings) -> HybridPolicy:
        return cls(
            mode=settings.retrieval_mode,
            default_top_k=settings.retrieval_default_top_k,
            candidate_top_k=settings.hybrid_candidate_top_k,
            rrf_k=settings.hybrid_rrf_k,
            max_query_length=settings.search_max_query_length,
        )


class SparseSearchBackend(Protocol):
    """BM25 capability consumed by the mode router."""

    @property
    def index(self) -> BM25Index: ...

    def search(
        self,
        *,
        repository_id: str,
        commit: str,
        query: str,
        top_k: int | None = None,
        allowed_chunk_ids: Sequence[str] | None = None,
    ) -> BM25SearchResult: ...


class DenseSearchBackend(Protocol):
    """Dense capability consumed by the mode router."""

    @property
    def namespace(self) -> VectorNamespace: ...

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        allowed_chunk_ids: Sequence[str] | None = None,
    ) -> DenseSearchResult: ...


class HybridSearchResult(StrictModel):
    """One normalized result independent of the selected retrieval mode."""

    repository_id: Identifier
    commit: GitSha
    mode: RetrievalMode
    query: NonEmptyText
    top_k: int = Field(gt=0, le=100)
    candidate_top_k: int = Field(gt=0, le=100)
    rrf_k: int | None = Field(default=None, gt=0, le=1_000)
    metadata_filter: RetrievalFilter
    eligible_chunk_count: int = Field(ge=0)
    sparse_candidate_count: int = Field(ge=0)
    dense_candidate_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    hits: tuple[RetrievalHit, ...]

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.candidate_top_k < self.top_k or len(self.hits) > self.top_k:
            raise ValueError("retrieval depths are inconsistent")
        if (
            self.sparse_candidate_count > self.candidate_top_k
            or self.dense_candidate_count > self.candidate_top_k
            or self.sparse_candidate_count > self.eligible_chunk_count
            or self.dense_candidate_count > self.eligible_chunk_count
        ):
            raise ValueError("branch candidates exceed requested depth")
        chunk_ids = tuple(hit.chunk_id for hit in self.hits)
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("retrieval result chunk IDs must be unique")
        if any(hit.query != self.query for hit in self.hits):
            raise ValueError("retrieval hits must preserve the query")

        if self.mode is RetrievalMode.BM25:
            if (
                self.rrf_k is not None
                or self.dense_candidate_count != 0
                or self.provider_call_count != 0
                or self.retry_count != 0
                or any(
                    hit.sparse_rank != rank
                    or hit.dense_rank is not None
                    or hit.fused_score is not None
                    for rank, hit in enumerate(self.hits, start=1)
                )
            ):
                raise ValueError("BM25 mode must contain sparse-only evidence")
        elif self.mode is RetrievalMode.DENSE:
            if (
                self.rrf_k is not None
                or self.sparse_candidate_count != 0
                or self.provider_call_count != 1 + self.retry_count
                or any(
                    hit.dense_rank != rank
                    or hit.sparse_rank is not None
                    or hit.fused_score is not None
                    for rank, hit in enumerate(self.hits, start=1)
                )
            ):
                raise ValueError("Dense mode must contain Dense-only evidence")
        elif (
            self.rrf_k is None
            or self.provider_call_count != 1 + self.retry_count
            or any(hit.fused_score is None for hit in self.hits)
        ):
            raise ValueError("Hybrid mode must contain fused evidence")

        if self.mode is RetrievalMode.HYBRID:
            order = tuple(
                (-float(hit.fused_score), hit.chunk_id)
                for hit in self.hits
                if hit.fused_score is not None
            )
            if order != tuple(sorted(order)):
                raise ValueError("RRF hits must use deterministic score order")
        return self


class HybridRetriever:
    """Route one stable call shape to BM25, Dense, or both with RRF."""

    def __init__(
        self,
        *,
        corpus: ChunkCorpus,
        sparse_retriever: SparseSearchBackend | None = None,
        dense_retriever: DenseSearchBackend | None = None,
        policy: HybridPolicy | None = None,
    ) -> None:
        self._corpus = corpus
        self._sparse = sparse_retriever
        self._dense = dense_retriever
        self._policy = policy or HybridPolicy()
        needs_sparse = self._policy.mode in (RetrievalMode.BM25, RetrievalMode.HYBRID)
        needs_dense = self._policy.mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID)
        if (needs_sparse and self._sparse is None) or (needs_dense and self._dense is None):
            raise HybridRetrievalError(HybridRetrievalErrorCode.INVALID_CONFIGURATION)
        if self._sparse is not None and needs_sparse:
            try:
                verify_bm25_index_corpus(self._sparse.index, corpus)
            except BM25Error:
                raise HybridRetrievalError(HybridRetrievalErrorCode.IDENTITY_MISMATCH) from None
        if (
            self._dense is not None
            and needs_dense
            and (
                self._dense.namespace.repository_id != corpus.repository_id
                or self._dense.namespace.commit != corpus.commit
            )
        ):
            raise HybridRetrievalError(HybridRetrievalErrorCode.IDENTITY_MISMATCH)

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
        normalized_query, requested_top_k = self._validate_query(query, top_k)
        effective_filter = metadata_filter or RetrievalFilter()
        allowed = eligible_chunk_ids(self._corpus, effective_filter)
        backend_allowlist = None if effective_filter == RetrievalFilter() else allowed
        candidate_top_k = (
            max(requested_top_k, self._policy.candidate_top_k)
            if self._policy.mode is RetrievalMode.HYBRID
            else requested_top_k
        )
        sparse_result = self._search_sparse(
            normalized_query,
            candidate_top_k,
            backend_allowlist,
        )
        dense_result = self._search_dense(
            normalized_query,
            candidate_top_k,
            backend_allowlist,
        )
        self._validate_branch_results(
            query=normalized_query,
            candidate_top_k=candidate_top_k,
            eligible_chunk_ids=allowed,
            sparse_result=sparse_result,
            dense_result=dense_result,
        )
        if self._policy.mode is RetrievalMode.HYBRID:
            assert sparse_result is not None
            assert dense_result is not None
            hits = reciprocal_rank_fusion(
                sparse_result,
                dense_result,
                rrf_k=self._policy.rrf_k,
                top_k=requested_top_k,
            )
        elif sparse_result is not None:
            hits = sparse_result.hits[:requested_top_k]
        else:
            assert dense_result is not None
            hits = dense_result.hits[:requested_top_k]

        try:
            return HybridSearchResult(
                repository_id=self._corpus.repository_id,
                commit=self._corpus.commit,
                mode=self._policy.mode,
                query=normalized_query,
                top_k=requested_top_k,
                candidate_top_k=candidate_top_k,
                rrf_k=(self._policy.rrf_k if self._policy.mode is RetrievalMode.HYBRID else None),
                metadata_filter=effective_filter,
                eligible_chunk_count=len(allowed),
                sparse_candidate_count=(
                    len(sparse_result.hits) if sparse_result is not None else 0
                ),
                dense_candidate_count=(len(dense_result.hits) if dense_result is not None else 0),
                provider_call_count=(
                    dense_result.provider_call_count if dense_result is not None else 0
                ),
                retry_count=(dense_result.retry_count if dense_result is not None else 0),
                hits=hits,
            )
        except (TypeError, ValueError):
            raise HybridRetrievalError(HybridRetrievalErrorCode.INVALID_RESULT) from None

    def _validate_query(
        self,
        query: str,
        top_k: int | None,
    ) -> tuple[str, int]:
        if not isinstance(query, str):
            raise HybridRetrievalError(HybridRetrievalErrorCode.INVALID_QUERY)
        normalized = query.strip()
        requested = self._policy.default_top_k if top_k is None else top_k
        if (
            not normalized
            or "\x00" in normalized
            or len(normalized) > self._policy.max_query_length
            or isinstance(requested, bool)
            or not isinstance(requested, int)
            or not 1 <= requested <= 100
        ):
            raise HybridRetrievalError(HybridRetrievalErrorCode.INVALID_QUERY)
        return normalized, requested

    def _search_sparse(
        self,
        query: str,
        top_k: int,
        allowed: Sequence[str] | None,
    ) -> BM25SearchResult | None:
        if self._policy.mode is RetrievalMode.DENSE:
            return None
        assert self._sparse is not None
        try:
            return self._sparse.search(
                repository_id=self._corpus.repository_id,
                commit=self._corpus.commit,
                query=query,
                top_k=top_k,
                allowed_chunk_ids=allowed,
            )
        except BM25Error as error:
            raise HybridRetrievalError(
                HybridRetrievalErrorCode.SPARSE_FAILURE,
                sparse_code=error.code,
            ) from None

    def _search_dense(
        self,
        query: str,
        top_k: int,
        allowed: Sequence[str] | None,
    ) -> DenseSearchResult | None:
        if self._policy.mode is RetrievalMode.BM25:
            return None
        assert self._dense is not None
        try:
            return self._dense.search(
                query,
                top_k=top_k,
                allowed_chunk_ids=allowed,
            )
        except DenseRetrievalError as error:
            raise HybridRetrievalError(
                HybridRetrievalErrorCode.DENSE_FAILURE,
                dense_code=error.code,
            ) from None

    def _validate_branch_results(
        self,
        *,
        query: str,
        candidate_top_k: int,
        eligible_chunk_ids: Sequence[str],
        sparse_result: BM25SearchResult | None,
        dense_result: DenseSearchResult | None,
    ) -> None:
        eligible = set(eligible_chunk_ids)
        for result in (sparse_result, dense_result):
            if result is None:
                continue
            if (
                result.repository_id != self._corpus.repository_id
                or result.commit != self._corpus.commit
                or result.query != query
                or result.top_k != candidate_top_k
                or any(hit.chunk_id not in eligible for hit in result.hits)
            ):
                raise HybridRetrievalError(HybridRetrievalErrorCode.INVALID_RESULT)


def eligible_chunk_ids(
    corpus: ChunkCorpus,
    metadata_filter: RetrievalFilter,
) -> tuple[str, ...]:
    """Compile auditable corpus metadata predicates into sorted chunk IDs."""

    return tuple(
        sorted(
            chunk.chunk_id
            for chunk in corpus.chunks
            if _matches_filter(
                relative_path=chunk.relative_path,
                language=chunk.language,
                metadata_filter=metadata_filter,
            )
        )
    )


def is_test_path(relative_path: str) -> bool:
    """Classify common test paths without executing or importing repository code."""

    path = PurePosixPath(relative_path)
    directories = {part.casefold() for part in path.parts[:-1]}
    name = path.name.casefold()
    stem = path.stem.casefold()
    return (
        bool(directories & {"test", "tests", "__tests__"})
        or name == "conftest.py"
        or name.startswith("test_")
        or stem.endswith("_test")
    )


def reciprocal_rank_fusion(
    sparse: BM25SearchResult,
    dense: DenseSearchResult,
    *,
    rrf_k: int,
    top_k: int,
) -> tuple[RetrievalHit, ...]:
    """Fuse branch ranks without mixing incomparable raw score scales."""

    if (
        isinstance(rrf_k, bool)
        or not isinstance(rrf_k, int)
        or not 1 <= rrf_k <= 1_000
        or isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= 100
        or sparse.repository_id != dense.repository_id
        or sparse.commit != dense.commit
        or sparse.query != dense.query
    ):
        raise HybridRetrievalError(HybridRetrievalErrorCode.INVALID_RESULT)

    sparse_by_id = {hit.chunk_id: hit for hit in sparse.hits}
    dense_by_id = {hit.chunk_id: hit for hit in dense.hits}
    fused: list[RetrievalHit] = []
    for chunk_id in sorted(set(sparse_by_id) | set(dense_by_id)):
        sparse_hit = sparse_by_id.get(chunk_id)
        dense_hit = dense_by_id.get(chunk_id)
        fused_score = 0.0
        if sparse_hit is not None:
            assert sparse_hit.sparse_rank is not None
            fused_score += 1.0 / (rrf_k + sparse_hit.sparse_rank)
        if dense_hit is not None:
            assert dense_hit.dense_rank is not None
            fused_score += 1.0 / (rrf_k + dense_hit.dense_rank)
        fused.append(
            RetrievalHit(
                chunk_id=chunk_id,
                sparse_rank=(sparse_hit.sparse_rank if sparse_hit is not None else None),
                sparse_score=(sparse_hit.sparse_score if sparse_hit is not None else None),
                dense_rank=(dense_hit.dense_rank if dense_hit is not None else None),
                dense_score=(dense_hit.dense_score if dense_hit is not None else None),
                fused_score=fused_score,
                query=sparse.query,
                retrieval_reason=(
                    f"RRF k={rrf_k}; "
                    f"sparse_rank={_rank_text(sparse_hit, sparse=True)}; "
                    f"dense_rank={_rank_text(dense_hit, sparse=False)}"
                ),
            )
        )
    fused.sort(
        key=lambda hit: (
            -float(hit.fused_score) if hit.fused_score is not None else 0.0,
            hit.chunk_id,
        )
    )
    return tuple(fused[:top_k])


def _matches_filter(
    *,
    relative_path: str,
    language: str,
    metadata_filter: RetrievalFilter,
) -> bool:
    path = PurePosixPath(relative_path)
    if metadata_filter.path_prefixes and not any(
        path.parts[: len(PurePosixPath(prefix).parts)] == PurePosixPath(prefix).parts
        for prefix in metadata_filter.path_prefixes
    ):
        return False
    if metadata_filter.languages and language.casefold() not in metadata_filter.languages:
        return False
    path_is_test = is_test_path(relative_path)
    return (
        metadata_filter.test_scope is RetrievalTestScope.ALL
        or (metadata_filter.test_scope is RetrievalTestScope.TESTS_ONLY and path_is_test)
        or (metadata_filter.test_scope is RetrievalTestScope.NON_TESTS_ONLY and not path_is_test)
    )


def _rank_text(hit: RetrievalHit | None, *, sparse: bool) -> str:
    if hit is None:
        return "missing"
    rank = hit.sparse_rank if sparse else hit.dense_rank
    return str(rank)
