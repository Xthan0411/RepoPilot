"""Replaceable embedding Providers, local cache, and vector indexing pipeline."""

from repopilot.embeddings.base import (
    EMBEDDING_SCHEMA_VERSION,
    EmbeddingBatchExecutor,
    EmbeddingError,
    EmbeddingExecutionResult,
    EmbeddingIdentity,
    EmbeddingInput,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    embedding_request_id,
    identity_fingerprint,
)
from repopilot.embeddings.cache import (
    EMBEDDING_CACHE_SCHEMA_VERSION,
    EmbeddingCacheRecord,
    LocalEmbeddingCache,
    embedding_cache_key,
)
from repopilot.embeddings.fake import FakeEmbeddingProvider
from repopilot.embeddings.indexer import (
    EmbeddingIndexer,
    EmbeddingIndexResult,
    embedding_cache_from_settings,
    embedding_executor_from_settings,
    embedding_identity_from_settings,
    embedding_provider_from_settings,
)
from repopilot.embeddings.jina import JinaEmbeddingProvider

__all__ = [
    "EMBEDDING_CACHE_SCHEMA_VERSION",
    "EMBEDDING_SCHEMA_VERSION",
    "EmbeddingBatchExecutor",
    "EmbeddingCacheRecord",
    "EmbeddingError",
    "EmbeddingExecutionResult",
    "EmbeddingIdentity",
    "EmbeddingIndexResult",
    "EmbeddingIndexer",
    "EmbeddingInput",
    "EmbeddingProvider",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingVector",
    "FakeEmbeddingProvider",
    "JinaEmbeddingProvider",
    "LocalEmbeddingCache",
    "embedding_cache_from_settings",
    "embedding_cache_key",
    "embedding_executor_from_settings",
    "embedding_identity_from_settings",
    "embedding_provider_from_settings",
    "embedding_request_id",
    "identity_fingerprint",
]
