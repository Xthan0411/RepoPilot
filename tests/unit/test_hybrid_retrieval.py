"""RRF math, mode routing, metadata filters, and typed failure tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any

import pytest
from pydantic import ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.embeddings import EmbeddingIdentity
from repopilot.enums import (
    BM25ErrorCode,
    EmbeddingProviderKind,
    HybridRetrievalErrorCode,
    RetrievalMode,
    RetrievalTestScope,
    SymbolType,
)
from repopilot.retrieval import (
    BM25Index,
    BM25Retriever,
    BM25SearchResult,
    ChunkCorpus,
    ChunkSourceSnapshot,
    DenseSearchResult,
    HybridPolicy,
    HybridRetrievalError,
    HybridRetriever,
    RetrievalFilter,
    eligible_chunk_ids,
    is_test_path,
    reciprocal_rank_fusion,
)
from repopilot.schemas import CodeChunk, RetrievalHit
from repopilot.storage import VectorNamespace

REPOSITORY_ID = "fixture.hybrid"
COMMIT = "d" * 40
SOURCE_HASH = "a" * 64
BUILD_FINGERPRINT = "b" * 64
QUERY = "parser shared"


def code_chunk(
    chunk_id: str,
    path: str,
    *,
    language: str,
    content: str,
) -> CodeChunk:
    source_id = f"source_{chunk_id}"
    return CodeChunk(
        chunk_id=chunk_id,
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        relative_path=path,
        symbol_type=SymbolType.TEXT,
        start_line=1,
        end_line=max(1, content.count("\n")),
        language=language,
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        metadata={
            "source_id": source_id,
            "source_content_hash": SOURCE_HASH,
            "processor": "fixture",
        },
    )


def corpus() -> ChunkCorpus:
    chunks = tuple(
        sorted(
            (
                code_chunk(
                    "chunk_docs",
                    "docs/parser.md",
                    language="markdown",
                    content="parser shared documentation\n",
                ),
                code_chunk(
                    "chunk_parser",
                    "pkg/parser.py",
                    language="python",
                    content="parser shared implementation\n",
                ),
                code_chunk(
                    "chunk_service",
                    "pkg/service.py",
                    language="python",
                    content="service implementation\n",
                ),
                code_chunk(
                    "chunk_test",
                    "tests/test_parser.py",
                    language="python",
                    content="parser shared test\n",
                ),
                code_chunk(
                    "chunk_other",
                    "pkg_extra/parser.py",
                    language="python",
                    content="parser shared other\n",
                ),
            ),
            key=lambda item: item.relative_path,
        )
    )
    return ChunkCorpus(
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        sources=tuple(
            ChunkSourceSnapshot(
                source_id=str(chunk.metadata["source_id"]),
                relative_path=chunk.relative_path,
                source_content_hash=SOURCE_HASH,
                processor="fixture",
                build_fingerprint=BUILD_FINGERPRINT,
                chunk_ids=(chunk.chunk_id,),
            )
            for chunk in chunks
        ),
        chunks=chunks,
    )


def embedding_identity() -> EmbeddingIdentity:
    return EmbeddingIdentity(
        provider=EmbeddingProviderKind.FAKE,
        model="fake-hybrid",
        revision="v1",
        dimension=3,
        normalize=True,
        max_length=128,
    )


def sparse_result(*chunk_ids: str, query: str = QUERY) -> BM25SearchResult:
    return BM25SearchResult(
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        query=query,
        top_k=max(1, len(chunk_ids)),
        hits=tuple(
            RetrievalHit(
                chunk_id=chunk_id,
                sparse_rank=rank,
                sparse_score=float(len(chunk_ids) - rank + 1),
                query=query,
                retrieval_reason="fixture sparse",
            )
            for rank, chunk_id in enumerate(chunk_ids, start=1)
        ),
    )


def dense_result(*chunk_ids: str, query: str = QUERY) -> DenseSearchResult:
    identity = embedding_identity()
    namespace = VectorNamespace(
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        embedding_model=identity.model,
        embedding_version=identity.fingerprint,
        dimension=identity.dimension,
    )
    return DenseSearchResult(
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        identity=identity,
        namespace=namespace,
        query=query,
        top_k=max(1, len(chunk_ids)),
        hits=tuple(
            RetrievalHit(
                chunk_id=chunk_id,
                dense_rank=rank,
                dense_score=1.0 - rank * 0.1,
                query=query,
                retrieval_reason="fixture dense",
            )
            for rank, chunk_id in enumerate(chunk_ids, start=1)
        ),
        provider_call_count=1,
        retry_count=0,
    )


class RecordingSparseBackend:
    def __init__(self, chunk_corpus: ChunkCorpus) -> None:
        self._retriever = BM25Retriever.from_corpus(chunk_corpus)
        self.calls: list[dict[str, object]] = []

    @property
    def index(self) -> BM25Index:
        return self._retriever.index

    def search(
        self,
        *,
        repository_id: str,
        commit: str,
        query: str,
        top_k: int | None = None,
        allowed_chunk_ids: Sequence[str] | None = None,
    ) -> BM25SearchResult:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "allowed_chunk_ids": (
                    tuple(allowed_chunk_ids) if allowed_chunk_ids is not None else None
                ),
            }
        )
        return self._retriever.search(
            repository_id=repository_id,
            commit=commit,
            query=query,
            top_k=top_k,
            allowed_chunk_ids=allowed_chunk_ids,
        )


class RecordingDenseBackend:
    def __init__(
        self,
        ordered_chunk_ids: Sequence[str],
        *,
        repository_id: str = REPOSITORY_ID,
    ) -> None:
        self._identity = embedding_identity()
        self._namespace = VectorNamespace(
            repository_id=repository_id,
            commit=COMMIT,
            embedding_model=self._identity.model,
            embedding_version=self._identity.fingerprint,
            dimension=self._identity.dimension,
        )
        self._ordered_chunk_ids = tuple(ordered_chunk_ids)
        self.calls: list[dict[str, object]] = []

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
        assert top_k is not None
        allowed = set(allowed_chunk_ids) if allowed_chunk_ids is not None else None
        selected = tuple(
            chunk_id
            for chunk_id in self._ordered_chunk_ids
            if allowed is None or chunk_id in allowed
        )[:top_k]
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "allowed_chunk_ids": (
                    tuple(allowed_chunk_ids) if allowed_chunk_ids is not None else None
                ),
            }
        )
        return DenseSearchResult(
            repository_id=self._namespace.repository_id,
            commit=self._namespace.commit,
            identity=self._identity,
            namespace=self._namespace,
            query=query,
            top_k=top_k,
            hits=tuple(
                RetrievalHit(
                    chunk_id=chunk_id,
                    dense_rank=rank,
                    dense_score=1.0 - rank * 0.1,
                    query=query,
                    retrieval_reason="recording dense",
                )
                for rank, chunk_id in enumerate(selected, start=1)
            ),
            provider_call_count=1,
            retry_count=0,
        )


class IgnoringFilterDenseBackend(RecordingDenseBackend):
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        allowed_chunk_ids: Sequence[str] | None = None,
    ) -> DenseSearchResult:
        return super().search(
            query,
            top_k=top_k,
            allowed_chunk_ids=None,
        )


def assert_hybrid_error(
    captured: pytest.ExceptionInfo[HybridRetrievalError],
    code: HybridRetrievalErrorCode,
) -> None:
    assert captured.value.code is code


def test_rrf_uses_ranks_preserves_branch_evidence_and_breaks_ties_by_chunk_id() -> None:
    fused = reciprocal_rank_fusion(
        sparse_result("chunk_overlap", "chunk_a"),
        dense_result("chunk_overlap", "chunk_z"),
        rrf_k=60,
        top_k=3,
    )

    assert tuple(hit.chunk_id for hit in fused) == (
        "chunk_overlap",
        "chunk_a",
        "chunk_z",
    )
    assert fused[0].fused_score == pytest.approx(2 / 61)
    assert fused[0].sparse_rank == fused[0].dense_rank == 1
    assert fused[1].fused_score == fused[2].fused_score == pytest.approx(1 / 62)
    assert fused[1].dense_rank is None
    assert fused[2].sparse_rank is None
    assert "dense_rank=missing" in fused[1].retrieval_reason


def test_rrf_keeps_a_nonempty_branch_when_the_other_branch_has_no_hits() -> None:
    fused = reciprocal_rank_fusion(
        sparse_result(),
        dense_result("chunk_z"),
        rrf_k=60,
        top_k=5,
    )

    assert tuple(hit.chunk_id for hit in fused) == ("chunk_z",)
    assert fused[0].sparse_rank is None
    assert fused[0].dense_rank == 1
    assert fused[0].fused_score == pytest.approx(1 / 61)


def test_metadata_filter_is_canonical_and_uses_segment_language_and_test_rules() -> None:
    chunk_corpus = corpus()
    metadata_filter = RetrievalFilter(
        path_prefixes=("pkg", "pkg"),
        languages=("PYTHON", "python"),
        test_scope=RetrievalTestScope.NON_TESTS_ONLY,
    )

    assert metadata_filter.path_prefixes == ("pkg",)
    assert metadata_filter.languages == ("python",)
    assert eligible_chunk_ids(chunk_corpus, metadata_filter) == (
        "chunk_parser",
        "chunk_service",
    )
    assert is_test_path("tests/unit/test_parser.py")
    assert is_test_path("pkg/parser_test.py")
    assert is_test_path("pkg/conftest.py")
    assert not is_test_path("pkg/testing.py")
    assert not is_test_path("contest/parser.py")


@pytest.mark.parametrize(
    "path_prefix",
    ("../pkg", "/pkg", "pkg/../tests", "pkg\\tests"),
)
def test_metadata_filter_rejects_noncanonical_path_prefix(path_prefix: str) -> None:
    with pytest.raises(ValidationError):
        RetrievalFilter(path_prefixes=(path_prefix,))


def test_bm25_mode_uses_common_call_shape_without_dense_backend() -> None:
    chunk_corpus = corpus()
    sparse = RecordingSparseBackend(chunk_corpus)
    retriever = HybridRetriever(
        corpus=chunk_corpus,
        sparse_retriever=sparse,
        policy=HybridPolicy(
            mode=RetrievalMode.BM25,
            default_top_k=2,
            candidate_top_k=4,
        ),
    )

    result = retriever.search(
        QUERY,
        metadata_filter=RetrievalFilter(
            path_prefixes=("docs",),
            languages=("markdown",),
        ),
    )

    assert result.mode is RetrievalMode.BM25
    assert tuple(hit.chunk_id for hit in result.hits) == ("chunk_docs",)
    assert result.candidate_top_k == result.top_k == 2
    assert result.eligible_chunk_count == 1
    assert result.provider_call_count == 0
    assert sparse.calls[0]["allowed_chunk_ids"] == ("chunk_docs",)


def test_dense_mode_uses_filter_allowlist_without_sparse_backend() -> None:
    chunk_corpus = corpus()
    dense = RecordingDenseBackend(("chunk_parser", "chunk_test", "chunk_docs", "chunk_service"))
    retriever = HybridRetriever(
        corpus=chunk_corpus,
        dense_retriever=dense,
        policy=HybridPolicy(
            mode=RetrievalMode.DENSE,
            default_top_k=3,
            candidate_top_k=4,
        ),
    )

    result = retriever.search(
        QUERY,
        metadata_filter=RetrievalFilter(
            languages=("python",),
            test_scope=RetrievalTestScope.TESTS_ONLY,
        ),
    )

    assert result.mode is RetrievalMode.DENSE
    assert tuple(hit.chunk_id for hit in result.hits) == ("chunk_test",)
    assert result.sparse_candidate_count == 0
    assert result.provider_call_count == 1
    assert dense.calls[0]["allowed_chunk_ids"] == ("chunk_test",)


def test_hybrid_mode_pushes_same_candidates_to_both_branches_and_fuses() -> None:
    chunk_corpus = corpus()
    sparse = RecordingSparseBackend(chunk_corpus)
    dense = RecordingDenseBackend(("chunk_service", "chunk_parser", "chunk_test", "chunk_docs"))
    retriever = HybridRetriever(
        corpus=chunk_corpus,
        sparse_retriever=sparse,
        dense_retriever=dense,
        policy=HybridPolicy(
            mode=RetrievalMode.HYBRID,
            default_top_k=2,
            candidate_top_k=4,
            rrf_k=30,
        ),
    )
    metadata_filter = RetrievalFilter(
        path_prefixes=("pkg",),
        languages=("python",),
        test_scope=RetrievalTestScope.NON_TESTS_ONLY,
    )

    result = retriever.search(QUERY, metadata_filter=metadata_filter)

    expected_allowed = ("chunk_parser", "chunk_service")
    assert result.mode is RetrievalMode.HYBRID
    assert result.top_k == 2
    assert result.candidate_top_k == 4
    assert result.rrf_k == 30
    assert result.eligible_chunk_count == 2
    assert sparse.calls[0]["allowed_chunk_ids"] == expected_allowed
    assert dense.calls[0]["allowed_chunk_ids"] == expected_allowed
    assert all(hit.fused_score is not None for hit in result.hits)
    assert result.provider_call_count == 1


def test_empty_filter_does_not_materialize_an_all_chunk_allowlist() -> None:
    chunk_corpus = corpus()
    sparse = RecordingSparseBackend(chunk_corpus)
    retriever = HybridRetriever(
        corpus=chunk_corpus,
        sparse_retriever=sparse,
        policy=HybridPolicy(mode=RetrievalMode.BM25),
    )

    retriever.search(QUERY)

    assert sparse.calls[0]["allowed_chunk_ids"] is None


@pytest.mark.parametrize(
    ("query", "top_k"),
    [
        ("", None),
        ("query\x00hidden", None),
        ("x" * 257, None),
        ("query", 0),
        ("query", 101),
        ("query", True),
    ],
)
def test_router_rejects_invalid_query_before_either_backend(
    query: str,
    top_k: Any,
) -> None:
    chunk_corpus = corpus()
    sparse = RecordingSparseBackend(chunk_corpus)
    dense = RecordingDenseBackend(("chunk_parser",))
    retriever = HybridRetriever(
        corpus=chunk_corpus,
        sparse_retriever=sparse,
        dense_retriever=dense,
        policy=HybridPolicy(mode=RetrievalMode.HYBRID),
    )

    with pytest.raises(HybridRetrievalError) as captured:
        retriever.search(query, top_k=top_k)

    assert_hybrid_error(captured, HybridRetrievalErrorCode.INVALID_QUERY)
    assert sparse.calls == []
    assert dense.calls == []


def test_router_rejects_missing_backend_and_identity_mismatch() -> None:
    chunk_corpus = corpus()
    with pytest.raises(HybridRetrievalError) as missing:
        HybridRetriever(
            corpus=chunk_corpus,
            sparse_retriever=RecordingSparseBackend(chunk_corpus),
            policy=HybridPolicy(mode=RetrievalMode.HYBRID),
        )
    assert_hybrid_error(missing, HybridRetrievalErrorCode.INVALID_CONFIGURATION)

    with pytest.raises(HybridRetrievalError) as mismatch:
        HybridRetriever(
            corpus=chunk_corpus,
            dense_retriever=RecordingDenseBackend(
                ("chunk_parser",),
                repository_id="fixture.other",
            ),
            policy=HybridPolicy(mode=RetrievalMode.DENSE),
        )
    assert_hybrid_error(mismatch, HybridRetrievalErrorCode.IDENTITY_MISMATCH)


def test_router_preserves_sparse_failure_code() -> None:
    chunk_corpus = corpus()
    retriever = HybridRetriever(
        corpus=chunk_corpus,
        sparse_retriever=RecordingSparseBackend(chunk_corpus),
        policy=HybridPolicy(mode=RetrievalMode.BM25),
    )

    with pytest.raises(HybridRetrievalError) as captured:
        retriever.search("!!!")

    assert_hybrid_error(captured, HybridRetrievalErrorCode.SPARSE_FAILURE)
    assert captured.value.sparse_code is BM25ErrorCode.EMPTY_QUERY
    assert captured.value.dense_code is None


def test_router_rejects_backend_hit_outside_compiled_filter() -> None:
    chunk_corpus = corpus()
    retriever = HybridRetriever(
        corpus=chunk_corpus,
        dense_retriever=IgnoringFilterDenseBackend(("chunk_parser", "chunk_test")),
        policy=HybridPolicy(mode=RetrievalMode.DENSE),
    )

    with pytest.raises(HybridRetrievalError) as captured:
        retriever.search(
            QUERY,
            metadata_filter=RetrievalFilter(test_scope=RetrievalTestScope.TESTS_ONLY),
        )

    assert_hybrid_error(captured, HybridRetrievalErrorCode.INVALID_RESULT)


def test_policy_maps_settings_and_rejects_candidate_depth_below_default() -> None:
    settings = RepoPilotSettings(
        retrieval_mode=RetrievalMode.HYBRID,
        retrieval_default_top_k=20,
        hybrid_candidate_top_k=80,
        hybrid_rrf_k=45,
        search_max_query_length=512,
    )

    assert HybridPolicy.from_settings(settings) == HybridPolicy(
        mode=RetrievalMode.HYBRID,
        default_top_k=20,
        candidate_top_k=80,
        rrf_k=45,
        max_query_length=512,
    )
    with pytest.raises(ValidationError):
        HybridPolicy(default_top_k=20, candidate_top_k=10)


def test_rrf_rejects_mismatched_query_or_illegal_parameters() -> None:
    for sparse, dense, rrf_k, top_k in (
        (sparse_result(query="one"), dense_result(query="two"), 60, 5),
        (sparse_result(), dense_result(), 0, 5),
        (sparse_result(), dense_result(), 60, 0),
    ):
        with pytest.raises(HybridRetrievalError) as captured:
            reciprocal_rank_fusion(
                sparse,
                dense,
                rrf_k=rrf_k,
                top_k=top_k,
            )
        assert_hybrid_error(captured, HybridRetrievalErrorCode.INVALID_RESULT)


def test_filter_prefix_matches_path_segments_not_raw_string_prefixes() -> None:
    chunk_corpus = corpus()

    selected = eligible_chunk_ids(
        chunk_corpus,
        RetrievalFilter(path_prefixes=(PurePosixPath("pkg").as_posix(),)),
    )

    assert "chunk_parser" in selected
    assert "chunk_other" not in selected
