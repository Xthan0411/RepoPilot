"""Vector storage data contracts independent of PostgreSQL and pgvector."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import Field, JsonValue, field_validator, model_validator

from repopilot.enums import VectorStoreErrorCode
from repopilot.schemas import (
    GitSha,
    Identifier,
    ModelIdentifier,
    RelativePath,
    Sha256,
    StrictModel,
    VersionIdentifier,
)

VECTOR_STORE_SCHEMA_VERSION: Literal[1] = 1
MAX_VECTOR_DIMENSION = 2_000
MAX_VECTOR_COORDINATE = 3.4e38
VECTOR_STORE_ERROR_MESSAGES = {
    VectorStoreErrorCode.NOT_INITIALIZED: "vector store has not been initialized",
    VectorStoreErrorCode.INVALID_RECORD: "vector record is invalid",
    VectorStoreErrorCode.DUPLICATE_RECORD: "vector batch contains duplicate chunk IDs",
    VectorStoreErrorCode.NAMESPACE_MISMATCH: (
        "vector records do not belong to the requested namespace"
    ),
    VectorStoreErrorCode.DIMENSION_MISMATCH: (
        "vector dimension does not match the configured store"
    ),
    VectorStoreErrorCode.CONNECTION_ERROR: "vector database connection failed",
    VectorStoreErrorCode.EXTENSION_UNAVAILABLE: "pgvector extension is unavailable",
    VectorStoreErrorCode.MIGRATION_ERROR: "vector database migration failed",
    VectorStoreErrorCode.SCHEMA_MISMATCH: "vector database schema is incompatible",
    VectorStoreErrorCode.QUERY_ERROR: "vector database operation failed",
}


class VectorStoreError(RuntimeError):
    """Safe typed storage failure that never includes DSNs, SQL, or vectors."""

    def __init__(self, code: VectorStoreErrorCode) -> None:
        super().__init__(VECTOR_STORE_ERROR_MESSAGES[code])
        self.code = code


class VectorNamespace(StrictModel):
    """Isolation key shared by writes, reads, replacement, and deletion."""

    repository_id: Identifier
    commit: GitSha
    embedding_model: ModelIdentifier
    embedding_version: VersionIdentifier
    dimension: int = Field(gt=0, le=MAX_VECTOR_DIMENSION)

    @field_validator("dimension", mode="before")
    @classmethod
    def reject_boolean_dimension(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("vector dimension must be an integer")
        return value


class VectorRecord(StrictModel):
    """One model-versioned embedding bound to one immutable code chunk."""

    namespace: VectorNamespace
    chunk_id: Identifier
    content_hash: Sha256
    relative_path: RelativePath
    language: Identifier
    symbol_name: str | None = Field(default=None, min_length=1, max_length=512)
    embedding: tuple[float, ...] = Field(min_length=1, max_length=MAX_VECTOR_DIMENSION)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("embedding", mode="before")
    @classmethod
    def validate_embedding_values(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            raise ValueError("embedding must be a list or tuple")
        for coordinate in value:
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(float(coordinate))
                or abs(float(coordinate)) > MAX_VECTOR_COORDINATE
            ):
                raise ValueError("embedding coordinates must be finite float32 values")
        return value

    @model_validator(mode="after")
    def validate_dimension(self) -> Self:
        if len(self.embedding) != self.namespace.dimension:
            raise ValueError("embedding length must match namespace dimension")
        if not any(coordinate != 0.0 for coordinate in self.embedding):
            raise ValueError("cosine-search embeddings must not be all zero")
        return self


class VectorSearchHit(StrictModel):
    """One namespace-scoped cosine result before Retriever rank normalization."""

    namespace: VectorNamespace
    chunk_id: Identifier
    content_hash: Sha256
    relative_path: RelativePath
    language: Identifier
    symbol_name: str | None = Field(default=None, min_length=1, max_length=512)
    score: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class VectorWriteResult(StrictModel):
    """Observable outcome from upsert, replacement, or namespace deletion."""

    namespace: VectorNamespace
    written_count: int = Field(ge=0)
    deleted_count: int = Field(ge=0)
    total_count: int = Field(ge=0)


class VectorStoreInfo(StrictModel):
    """Initialized backend identity without connection secrets."""

    schema_version: Literal[1] = VECTOR_STORE_SCHEMA_VERSION
    backend: Identifier
    dimension: int = Field(gt=0, le=MAX_VECTOR_DIMENSION)
    schema_name: Identifier | None = None
    table_name: Identifier | None = None
    extension_version: str | None = Field(default=None, min_length=1, max_length=64)


@runtime_checkable
class VectorStore(Protocol):
    """Storage capability consumed by future Embedding and Dense layers."""

    def initialize(self) -> VectorStoreInfo:
        """Create/validate storage and make subsequent operations available."""

    def upsert(
        self,
        namespace: VectorNamespace,
        records: Sequence[VectorRecord],
    ) -> VectorWriteResult:
        """Idempotently insert or refresh records without deleting absent chunks."""

    def replace_namespace(
        self,
        namespace: VectorNamespace,
        records: Sequence[VectorRecord],
    ) -> VectorWriteResult:
        """Upsert the exact namespace snapshot and delete records absent from it."""

    def list_records(
        self,
        namespace: VectorNamespace,
    ) -> tuple[VectorRecord, ...]:
        """Return one isolated namespace in deterministic chunk-ID order."""

    def search(
        self,
        namespace: VectorNamespace,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        allowed_chunk_ids: Sequence[str] | None = None,
    ) -> tuple[VectorSearchHit, ...]:
        """Return allowed cosine hits by score descending, then chunk ID."""

    def delete_namespace(self, namespace: VectorNamespace) -> VectorWriteResult:
        """Delete only the requested repository/commit/model namespace."""


def validate_record_batch(
    namespace: VectorNamespace,
    records: Sequence[VectorRecord],
) -> tuple[VectorRecord, ...]:
    """Validate one deterministic single-namespace batch before any mutation."""

    normalized = tuple(records)
    if namespace.dimension <= 0:
        raise VectorStoreError(VectorStoreErrorCode.DIMENSION_MISMATCH)
    if any(record.namespace != namespace for record in normalized):
        raise VectorStoreError(VectorStoreErrorCode.NAMESPACE_MISMATCH)
    chunk_ids = tuple(record.chunk_id for record in normalized)
    if len(set(chunk_ids)) != len(chunk_ids):
        raise VectorStoreError(VectorStoreErrorCode.DUPLICATE_RECORD)
    return tuple(sorted(normalized, key=lambda record: record.chunk_id))


def validate_vector_query(
    namespace: VectorNamespace,
    query_embedding: Sequence[float],
    *,
    top_k: int,
) -> tuple[float, ...]:
    """Reject invalid cosine queries before any backend operation."""

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= 100
        or len(query_embedding) != namespace.dimension
    ):
        raise VectorStoreError(VectorStoreErrorCode.INVALID_RECORD)
    normalized: list[float] = []
    for coordinate in query_embedding:
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(float(coordinate))
            or abs(float(coordinate)) > MAX_VECTOR_COORDINATE
        ):
            raise VectorStoreError(VectorStoreErrorCode.INVALID_RECORD)
        normalized.append(float(coordinate))
    if not any(coordinate != 0.0 for coordinate in normalized):
        raise VectorStoreError(VectorStoreErrorCode.INVALID_RECORD)
    return tuple(normalized)


def validate_chunk_allowlist(
    allowed_chunk_ids: Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Canonicalize one parameter-safe optional chunk allowlist."""

    if allowed_chunk_ids is None:
        return None
    normalized = tuple(allowed_chunk_ids)
    if (
        len(normalized) > 100_000
        or len(set(normalized)) != len(normalized)
        or any(
            not isinstance(chunk_id, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", chunk_id) is None
            for chunk_id in normalized
        )
    ):
        raise VectorStoreError(VectorStoreErrorCode.INVALID_RECORD)
    return tuple(sorted(normalized))
