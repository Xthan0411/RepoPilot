"""Deterministic in-memory VectorStore used by ordinary tests and Fakes."""

from __future__ import annotations

import math
from collections.abc import Sequence

from repopilot.enums import VectorStoreErrorCode
from repopilot.storage.base import (
    VECTOR_STORE_SCHEMA_VERSION,
    VectorNamespace,
    VectorRecord,
    VectorSearchHit,
    VectorStoreError,
    VectorStoreInfo,
    VectorWriteResult,
    validate_chunk_allowlist,
    validate_record_batch,
    validate_vector_query,
)


class InMemoryVectorStore:
    """Exact lifecycle semantics with deterministic cosine search."""

    def __init__(self, *, dimension: int) -> None:
        self._dimension = dimension
        self._initialized = False
        self._records: dict[
            tuple[str, str, str, str, int],
            dict[str, VectorRecord],
        ] = {}

    def initialize(self) -> VectorStoreInfo:
        if not 1 <= self._dimension <= 2_000:
            raise VectorStoreError(VectorStoreErrorCode.DIMENSION_MISMATCH)
        self._initialized = True
        return VectorStoreInfo(
            schema_version=VECTOR_STORE_SCHEMA_VERSION,
            backend="memory",
            dimension=self._dimension,
        )

    def upsert(
        self,
        namespace: VectorNamespace,
        records: Sequence[VectorRecord],
    ) -> VectorWriteResult:
        self._require_ready(namespace)
        batch = validate_record_batch(namespace, records)
        stored = self._records.setdefault(_namespace_key(namespace), {})
        for record in batch:
            previous = stored.get(record.chunk_id)
            if previous is not None and previous.content_hash != record.content_hash:
                raise VectorStoreError(VectorStoreErrorCode.INVALID_RECORD)
        for record in batch:
            stored[record.chunk_id] = record
        return VectorWriteResult(
            namespace=namespace,
            written_count=len(batch),
            deleted_count=0,
            total_count=len(stored),
        )

    def replace_namespace(
        self,
        namespace: VectorNamespace,
        records: Sequence[VectorRecord],
    ) -> VectorWriteResult:
        self._require_ready(namespace)
        batch = validate_record_batch(namespace, records)
        stored = self._records.setdefault(_namespace_key(namespace), {})
        for record in batch:
            previous = stored.get(record.chunk_id)
            if previous is not None and previous.content_hash != record.content_hash:
                raise VectorStoreError(VectorStoreErrorCode.INVALID_RECORD)
        incoming_ids = {record.chunk_id for record in batch}
        removed = len(set(stored) - incoming_ids)
        self._records[_namespace_key(namespace)] = {record.chunk_id: record for record in batch}
        return VectorWriteResult(
            namespace=namespace,
            written_count=len(batch),
            deleted_count=removed,
            total_count=len(batch),
        )

    def list_records(
        self,
        namespace: VectorNamespace,
    ) -> tuple[VectorRecord, ...]:
        self._require_ready(namespace)
        stored = self._records.get(_namespace_key(namespace), {})
        return tuple(stored[chunk_id] for chunk_id in sorted(stored))

    def search(
        self,
        namespace: VectorNamespace,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        allowed_chunk_ids: Sequence[str] | None = None,
    ) -> tuple[VectorSearchHit, ...]:
        self._require_ready(namespace)
        query = validate_vector_query(namespace, query_embedding, top_k=top_k)
        allowed = validate_chunk_allowlist(allowed_chunk_ids)
        allowed_set = set(allowed) if allowed is not None else None
        stored = self._records.get(_namespace_key(namespace), {})
        query_norm = math.sqrt(math.fsum(value * value for value in query))
        scored: list[VectorSearchHit] = []
        for record in stored.values():
            if allowed_set is not None and record.chunk_id not in allowed_set:
                continue
            record_norm = math.sqrt(math.fsum(value * value for value in record.embedding))
            score = math.fsum(
                left * right for left, right in zip(query, record.embedding, strict=True)
            ) / (query_norm * record_norm)
            score = min(1.0, max(-1.0, score))
            scored.append(
                VectorSearchHit(
                    namespace=namespace,
                    chunk_id=record.chunk_id,
                    content_hash=record.content_hash,
                    relative_path=record.relative_path,
                    language=record.language,
                    symbol_name=record.symbol_name,
                    score=score,
                    metadata=record.metadata,
                )
            )
        scored.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return tuple(scored[:top_k])

    def delete_namespace(self, namespace: VectorNamespace) -> VectorWriteResult:
        self._require_ready(namespace)
        removed = len(self._records.pop(_namespace_key(namespace), {}))
        return VectorWriteResult(
            namespace=namespace,
            written_count=0,
            deleted_count=removed,
            total_count=0,
        )

    def _require_ready(self, namespace: VectorNamespace) -> None:
        if not self._initialized:
            raise VectorStoreError(VectorStoreErrorCode.NOT_INITIALIZED)
        if namespace.dimension != self._dimension:
            raise VectorStoreError(VectorStoreErrorCode.DIMENSION_MISMATCH)


def _namespace_key(
    namespace: VectorNamespace,
) -> tuple[str, str, str, str, int]:
    return (
        namespace.repository_id,
        namespace.commit,
        namespace.embedding_model,
        namespace.embedding_version,
        namespace.dimension,
    )
