"""Replaceable vector-storage contracts and adapters."""

from repopilot.storage.base import (
    VECTOR_STORE_SCHEMA_VERSION,
    VectorNamespace,
    VectorRecord,
    VectorSearchHit,
    VectorStore,
    VectorStoreError,
    VectorStoreInfo,
    VectorWriteResult,
    validate_chunk_allowlist,
    validate_record_batch,
    validate_vector_query,
)
from repopilot.storage.memory import InMemoryVectorStore
from repopilot.storage.postgres import PostgresVectorStore

__all__ = [
    "VECTOR_STORE_SCHEMA_VERSION",
    "InMemoryVectorStore",
    "PostgresVectorStore",
    "VectorNamespace",
    "VectorRecord",
    "VectorSearchHit",
    "VectorStore",
    "VectorStoreError",
    "VectorStoreInfo",
    "VectorWriteResult",
    "validate_chunk_allowlist",
    "validate_record_batch",
    "validate_vector_query",
]
