"""Version-pinned deterministic Okapi BM25 indexing, ranking, and artifact I/O."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, ValidationError, field_validator, model_validator
from pydantic_core import to_jsonable_python

from repopilot.config import RepoPilotSettings
from repopilot.enums import BM25ErrorCode
from repopilot.retrieval.corpus import ChunkCorpus
from repopilot.retrieval.tokenizer import (
    BM25_PATH_BOOST,
    BM25_SYMBOL_BOOST,
    tokenize_chunk,
    tokenize_code_text,
)
from repopilot.schemas import (
    FiniteScore,
    GitSha,
    Identifier,
    NonEmptyText,
    RelativePath,
    RetrievalHit,
    Sha256,
    StrictModel,
)
from repopilot.storage import VectorStoreError, validate_chunk_allowlist

BM25_INDEX_SCHEMA_VERSION: Literal[1] = 1
BM25_MAX_TOP_K = 100
BM25_MAX_QUERY_LENGTH = 2_000
BM25_ERROR_MESSAGES = {
    BM25ErrorCode.EMPTY_QUERY: "BM25 query must contain at least one searchable token",
    BM25ErrorCode.INVALID_QUERY: "BM25 query parameters are invalid",
    BM25ErrorCode.IDENTITY_MISMATCH: ("BM25 query repository or commit does not match the index"),
    BM25ErrorCode.IO_ERROR: "BM25 index artifact could not be read or written",
    BM25ErrorCode.INVALID_DOCUMENT: "BM25 index document is invalid",
    BM25ErrorCode.UNSUPPORTED_VERSION: "BM25 index schema version is unsupported",
    BM25ErrorCode.INTEGRITY_MISMATCH: "BM25 index content digest does not match",
}

BM25Term = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
]


class BM25Error(RuntimeError):
    """Safe typed failure from BM25 query validation or artifact I/O."""

    def __init__(self, code: BM25ErrorCode) -> None:
        super().__init__(BM25_ERROR_MESSAGES[code])
        self.code = code


class BM25Policy(StrictModel):
    """Frozen scoring and result-limit policy persisted into each index."""

    k1: float = Field(default=1.5, gt=0.0, le=5.0, allow_inf_nan=False)
    b: float = Field(default=0.75, ge=0.0, le=1.0, allow_inf_nan=False)
    default_top_k: int = Field(default=10, gt=0, le=BM25_MAX_TOP_K)

    @field_validator("k1", "b", "default_top_k", mode="before")
    @classmethod
    def reject_boolean_values(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("BM25 policy values must not be booleans")
        return value

    @classmethod
    def from_settings(cls, settings: RepoPilotSettings) -> Self:
        return cls(
            k1=settings.bm25_k1,
            b=settings.bm25_b,
            default_top_k=settings.bm25_default_top_k,
        )


class BM25TermFrequency(StrictModel):
    term: BM25Term
    count: int = Field(gt=0)

    @field_validator("term")
    @classmethod
    def validate_term(cls, value: str) -> str:
        if value != value.lower() or any(character.isspace() for character in value):
            raise ValueError("BM25 terms must be normalized lowercase tokens")
        return value

    @field_validator("count", mode="before")
    @classmethod
    def reject_boolean_count(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("BM25 term count must be an integer")
        return value


class BM25Document(StrictModel):
    """Sparse statistics and explainability metadata for one chunk."""

    chunk_id: Identifier
    relative_path: RelativePath
    symbol_name: str | None = Field(default=None, min_length=1, max_length=512)
    length: int = Field(ge=0)
    term_frequencies: tuple[BM25TermFrequency, ...]

    @field_validator("length", mode="before")
    @classmethod
    def reject_boolean_length(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("BM25 document length must be an integer")
        return value

    @model_validator(mode="after")
    def validate_statistics(self) -> Self:
        terms = tuple(item.term for item in self.term_frequencies)
        if terms != tuple(sorted(terms)) or len(set(terms)) != len(terms):
            raise ValueError("BM25 document terms must be unique and sorted")
        if sum(item.count for item in self.term_frequencies) != self.length:
            raise ValueError("BM25 document length must equal summed term frequencies")
        return self


class BM25DocumentFrequency(StrictModel):
    term: BM25Term
    document_count: int = Field(gt=0)

    @field_validator("term")
    @classmethod
    def validate_term(cls, value: str) -> str:
        if value != value.lower() or any(character.isspace() for character in value):
            raise ValueError("BM25 terms must be normalized lowercase tokens")
        return value

    @field_validator("document_count", mode="before")
    @classmethod
    def reject_boolean_count(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("BM25 document count must be an integer")
        return value


class BM25Index(StrictModel):
    """Complete deterministic sparse index for exactly one repository commit."""

    schema_version: Literal[1] = BM25_INDEX_SCHEMA_VERSION
    repository_id: Identifier
    commit: GitSha
    k1: float = Field(gt=0.0, le=5.0, allow_inf_nan=False)
    b: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    default_top_k: int = Field(gt=0, le=BM25_MAX_TOP_K)
    path_boost: Literal[2] = BM25_PATH_BOOST
    symbol_boost: Literal[3] = BM25_SYMBOL_BOOST
    corpus_sha256: Sha256
    document_count: int = Field(ge=0)
    total_document_length: int = Field(ge=0)
    average_document_length: FiniteScore = Field(ge=0.0)
    documents: tuple[BM25Document, ...]
    document_frequencies: tuple[BM25DocumentFrequency, ...]
    content_sha256: Sha256

    @field_validator(
        "k1",
        "b",
        "default_top_k",
        "document_count",
        "total_document_length",
        "average_document_length",
        mode="before",
    )
    @classmethod
    def reject_boolean_statistics(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("BM25 numeric statistics must not be booleans")
        return value

    @model_validator(mode="after")
    def validate_statistics(self) -> Self:
        chunk_ids = tuple(document.chunk_id for document in self.documents)
        if chunk_ids != tuple(sorted(chunk_ids)) or len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("BM25 documents must have unique sorted chunk IDs")
        if self.document_count != len(self.documents):
            raise ValueError("BM25 document count does not match documents")
        expected_total = sum(document.length for document in self.documents)
        if self.total_document_length != expected_total:
            raise ValueError("BM25 total document length does not match documents")
        expected_average = expected_total / len(self.documents) if self.documents else 0.0
        if not math.isclose(
            self.average_document_length,
            expected_average,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("BM25 average document length does not match documents")

        frequency_terms = tuple(item.term for item in self.document_frequencies)
        if frequency_terms != tuple(sorted(frequency_terms)) or len(set(frequency_terms)) != len(
            frequency_terms
        ):
            raise ValueError("BM25 document frequencies must have unique sorted terms")
        expected_frequencies: Counter[str] = Counter()
        for document in self.documents:
            expected_frequencies.update(item.term for item in document.term_frequencies)
        actual_frequencies = {item.term: item.document_count for item in self.document_frequencies}
        if actual_frequencies != dict(expected_frequencies):
            raise ValueError("BM25 document frequencies do not match documents")
        if any(item.document_count > self.document_count for item in self.document_frequencies):
            raise ValueError("BM25 term document count exceeds corpus size")
        return self


class BM25SearchResult(StrictModel):
    """Repository-pinned sparse retrieval response."""

    repository_id: Identifier
    commit: GitSha
    query: NonEmptyText
    top_k: int = Field(gt=0, le=BM25_MAX_TOP_K)
    hits: tuple[RetrievalHit, ...]

    @model_validator(mode="after")
    def validate_hits(self) -> Self:
        if len(self.hits) > self.top_k:
            raise ValueError("BM25 result contains more than top_k hits")
        expected_ranks = tuple(range(1, len(self.hits) + 1))
        if tuple(hit.sparse_rank for hit in self.hits) != expected_ranks:
            raise ValueError("BM25 sparse ranks must be contiguous")
        if any(hit.query != self.query for hit in self.hits):
            raise ValueError("BM25 hits must preserve the normalized query")
        scores: list[float] = []
        for hit in self.hits:
            if hit.sparse_score is None:
                raise ValueError("BM25 hits must contain sparse scores")
            scores.append(hit.sparse_score)
        if scores != sorted(scores, reverse=True):
            raise ValueError("BM25 hit scores must be descending")
        return self


class BM25Retriever:
    """Query one immutable :class:`BM25Index` using standard Okapi BM25."""

    def __init__(self, index: BM25Index) -> None:
        verify_bm25_index_integrity(index)
        self._index = index
        self._document_frequencies = {
            item.term: item.document_count for item in index.document_frequencies
        }

    @property
    def index(self) -> BM25Index:
        return self._index

    @classmethod
    def from_corpus(
        cls,
        corpus: ChunkCorpus,
        *,
        policy: BM25Policy | None = None,
    ) -> Self:
        return cls(build_bm25_index(corpus, policy=policy))

    def search(
        self,
        *,
        repository_id: str,
        commit: str,
        query: str,
        top_k: int | None = None,
        allowed_chunk_ids: Sequence[str] | None = None,
    ) -> BM25SearchResult:
        """Rank matching chunks, rejecting blank/tokenless and cross-version queries."""

        if repository_id != self._index.repository_id or commit != self._index.commit:
            raise BM25Error(BM25ErrorCode.IDENTITY_MISMATCH)
        if not isinstance(query, str):
            raise BM25Error(BM25ErrorCode.INVALID_QUERY)
        normalized_query = query.strip()
        if not normalized_query:
            raise BM25Error(BM25ErrorCode.EMPTY_QUERY)
        if "\x00" in normalized_query or len(normalized_query) > BM25_MAX_QUERY_LENGTH:
            raise BM25Error(BM25ErrorCode.INVALID_QUERY)
        query_tokens = tokenize_code_text(normalized_query)
        if not query_tokens:
            raise BM25Error(BM25ErrorCode.EMPTY_QUERY)
        requested_top_k = self._index.default_top_k if top_k is None else top_k
        if (
            isinstance(requested_top_k, bool)
            or not isinstance(requested_top_k, int)
            or not 1 <= requested_top_k <= BM25_MAX_TOP_K
        ):
            raise BM25Error(BM25ErrorCode.INVALID_QUERY)
        try:
            allowed = validate_chunk_allowlist(allowed_chunk_ids)
        except VectorStoreError:
            raise BM25Error(BM25ErrorCode.INVALID_QUERY) from None
        allowed_set = set(allowed) if allowed is not None else None

        query_frequency = Counter(query_tokens)
        ranked: list[tuple[float, str, BM25Document, tuple[str, ...]]] = []
        for document in self._index.documents:
            if allowed_set is not None and document.chunk_id not in allowed_set:
                continue
            score, matched_terms = self._score_document(document, query_frequency)
            if score > 0.0:
                ranked.append((score, document.chunk_id, document, matched_terms))
        ranked.sort(key=lambda item: (-item[0], item[1]))

        hits = tuple(
            RetrievalHit(
                chunk_id=document.chunk_id,
                sparse_rank=rank,
                sparse_score=score,
                query=normalized_query,
                retrieval_reason=_retrieval_reason(document, matched_terms),
            )
            for rank, (score, _, document, matched_terms) in enumerate(
                ranked[:requested_top_k],
                start=1,
            )
        )
        return BM25SearchResult(
            repository_id=self._index.repository_id,
            commit=self._index.commit,
            query=normalized_query,
            top_k=requested_top_k,
            hits=hits,
        )

    def _score_document(
        self,
        document: BM25Document,
        query_frequency: Counter[str],
    ) -> tuple[float, tuple[str, ...]]:
        term_frequencies = {item.term: item.count for item in document.term_frequencies}
        score = 0.0
        matched_terms: list[str] = []
        for term, query_count in query_frequency.items():
            term_frequency = term_frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = self._document_frequencies[term]
            inverse_document_frequency = math.log(
                1.0
                + (self._index.document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            length_ratio = (
                document.length / self._index.average_document_length
                if self._index.average_document_length > 0
                else 0.0
            )
            denominator = term_frequency + self._index.k1 * (
                1.0 - self._index.b + self._index.b * length_ratio
            )
            score += (
                query_count
                * inverse_document_frequency
                * term_frequency
                * (self._index.k1 + 1.0)
                / denominator
            )
            matched_terms.append(term)
        return score, tuple(matched_terms)


def build_bm25_index(
    corpus: ChunkCorpus,
    *,
    policy: BM25Policy | None = None,
) -> BM25Index:
    """Build deterministic BM25 statistics from one validated chunk corpus."""

    effective_policy = policy or BM25Policy()
    documents: list[BM25Document] = []
    document_frequencies: Counter[str] = Counter()
    for chunk in sorted(corpus.chunks, key=lambda item: item.chunk_id):
        tokens = tokenize_chunk(chunk)
        term_frequencies = Counter(tokens)
        document_frequencies.update(term_frequencies.keys())
        documents.append(
            BM25Document(
                chunk_id=chunk.chunk_id,
                relative_path=chunk.relative_path,
                symbol_name=chunk.symbol_name,
                length=len(tokens),
                term_frequencies=tuple(
                    BM25TermFrequency(term=term, count=count)
                    for term, count in sorted(term_frequencies.items())
                ),
            )
        )
    total_document_length = sum(document.length for document in documents)
    payload: dict[str, object] = {
        "schema_version": BM25_INDEX_SCHEMA_VERSION,
        "repository_id": corpus.repository_id,
        "commit": corpus.commit,
        "k1": effective_policy.k1,
        "b": effective_policy.b,
        "default_top_k": effective_policy.default_top_k,
        "path_boost": BM25_PATH_BOOST,
        "symbol_boost": BM25_SYMBOL_BOOST,
        "corpus_sha256": chunk_corpus_content_sha256(corpus),
        "document_count": len(documents),
        "total_document_length": total_document_length,
        "average_document_length": (total_document_length / len(documents) if documents else 0.0),
        "documents": tuple(documents),
        "document_frequencies": tuple(
            BM25DocumentFrequency(term=term, document_count=count)
            for term, count in sorted(document_frequencies.items())
        ),
    }
    index = BM25Index.model_validate({**payload, "content_sha256": "0" * 64})
    return index.model_copy(update={"content_sha256": bm25_index_content_sha256(payload)})


def chunk_corpus_content_sha256(corpus: ChunkCorpus) -> str:
    """Hash the exact chunk identities/content consumed by a BM25 index."""

    payload = {
        "repository_id": corpus.repository_id,
        "commit": corpus.commit,
        "chunks": tuple(
            {
                "chunk_id": chunk.chunk_id,
                "content_hash": chunk.content_hash,
                "relative_path": chunk.relative_path,
                "symbol_name": chunk.symbol_name,
            }
            for chunk in corpus.chunks
        ),
    }
    return _canonical_sha256(payload)


def bm25_index_content_sha256(payload: dict[str, object]) -> str:
    """Hash canonical index content excluding its digest field."""

    return _canonical_sha256(payload)


def verify_bm25_index_integrity(index: BM25Index) -> None:
    """Reject an in-memory or loaded index whose fields no longer match its digest."""

    payload = index.model_dump(mode="python", exclude={"content_sha256"})
    if bm25_index_content_sha256(payload) != index.content_sha256:
        raise BM25Error(BM25ErrorCode.INTEGRITY_MISMATCH)


def verify_bm25_index_corpus(index: BM25Index, corpus: ChunkCorpus) -> None:
    """Require an index to describe the exact repository, commit, and chunks."""

    verify_bm25_index_integrity(index)
    if (
        index.repository_id != corpus.repository_id
        or index.commit != corpus.commit
        or index.corpus_sha256 != chunk_corpus_content_sha256(corpus)
    ):
        raise BM25Error(BM25ErrorCode.IDENTITY_MISMATCH)


def write_bm25_index(path: Path, index: BM25Index) -> None:
    """Integrity-check and atomically write one deterministic JSON index."""

    verify_bm25_index_integrity(index)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(f"{index.model_dump_json(indent=2)}\n", encoding="utf-8")
        temporary.replace(path)
    except OSError:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise BM25Error(BM25ErrorCode.IO_ERROR) from None


def read_bm25_index(path: Path) -> BM25Index:
    """Read, validate, and integrity-check one supported JSON index."""

    try:
        serialized = path.read_text(encoding="utf-8")
    except UnicodeError:
        raise BM25Error(BM25ErrorCode.INVALID_DOCUMENT) from None
    except OSError:
        raise BM25Error(BM25ErrorCode.IO_ERROR) from None
    try:
        raw = json.loads(serialized)
    except json.JSONDecodeError:
        raise BM25Error(BM25ErrorCode.INVALID_DOCUMENT) from None
    if not isinstance(raw, dict) or "schema_version" not in raw:
        raise BM25Error(BM25ErrorCode.INVALID_DOCUMENT)
    if raw["schema_version"] != BM25_INDEX_SCHEMA_VERSION:
        raise BM25Error(BM25ErrorCode.UNSUPPORTED_VERSION)
    try:
        index = BM25Index.model_validate(raw)
    except ValidationError:
        raise BM25Error(BM25ErrorCode.INVALID_DOCUMENT) from None
    verify_bm25_index_integrity(index)
    return index


def _retrieval_reason(
    document: BM25Document,
    matched_terms: tuple[str, ...],
) -> str:
    terms = ", ".join(dict.fromkeys(matched_terms[:8]))
    symbol = f"; symbol={document.symbol_name}" if document.symbol_name is not None else ""
    return f"BM25 matched [{terms}]; path={document.relative_path}{symbol}"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        to_jsonable_python(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
