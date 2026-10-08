"""Cache-aware chunk embedding and exact VectorStore replacement pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from pydantic import Field, model_validator

from repopilot.config import PROJECT_ROOT, RepoPilotSettings
from repopilot.embeddings.base import (
    EmbeddingBatchExecutor,
    EmbeddingError,
    EmbeddingExecutionResult,
    EmbeddingIdentity,
    EmbeddingInput,
    EmbeddingProvider,
    EmbeddingVector,
)
from repopilot.embeddings.cache import LocalEmbeddingCache
from repopilot.embeddings.fake import FakeEmbeddingProvider
from repopilot.embeddings.jina import JinaEmbeddingProvider
from repopilot.enums import EmbeddingErrorCode, EmbeddingProviderKind
from repopilot.retrieval.corpus import ChunkCorpus
from repopilot.schemas import CodeChunk, StrictModel
from repopilot.storage import (
    VectorNamespace,
    VectorRecord,
    VectorStore,
    VectorWriteResult,
)


class EmbeddingIndexResult(StrictModel):
    """Auditable outcome of one corpus-to-vector snapshot build."""

    identity: EmbeddingIdentity
    namespace: VectorNamespace
    chunk_count: int = Field(ge=0)
    unique_content_count: int = Field(ge=0)
    cache_hit_count: int = Field(ge=0)
    cache_miss_count: int = Field(ge=0)
    provider_batch_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    write_result: VectorWriteResult

    @model_validator(mode="after")
    def validate_counts_and_namespace(self) -> EmbeddingIndexResult:
        if self.unique_content_count != self.cache_hit_count + self.cache_miss_count:
            raise ValueError("unique content must equal cache hits plus misses")
        if self.provider_call_count != self.provider_batch_count + self.retry_count:
            raise ValueError("provider calls must equal batches plus retries")
        if self.write_result.namespace != self.namespace:
            raise ValueError("vector write result namespace is inconsistent")
        if self.write_result.written_count != self.chunk_count:
            raise ValueError("vector writes must cover every chunk")
        return self


class EmbeddingIndexer:
    """Reuse content-addressed vectors and replace one exact corpus namespace."""

    def __init__(
        self,
        *,
        executor: EmbeddingBatchExecutor,
        cache: LocalEmbeddingCache,
        vector_store: VectorStore,
    ) -> None:
        self._executor = executor
        self._cache = cache
        self._vector_store = vector_store

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._executor.identity

    def index(self, corpus: ChunkCorpus) -> EmbeddingIndexResult:
        """Embed only unique cache misses, then replace the corpus namespace."""

        store_info = self._vector_store.initialize()
        if store_info.dimension != self.identity.dimension:
            raise EmbeddingError(EmbeddingErrorCode.IDENTITY_MISMATCH)
        namespace = VectorNamespace(
            repository_id=corpus.repository_id,
            commit=corpus.commit,
            embedding_model=self.identity.model,
            embedding_version=self.identity.fingerprint,
            dimension=self.identity.dimension,
        )
        unique_inputs = _unique_embedding_inputs(corpus.chunks)
        vectors_by_hash: dict[str, EmbeddingVector] = {}
        misses: list[EmbeddingInput] = []
        for item in unique_inputs:
            cached = self._cache.get(self.identity, item.content_hash)
            if cached is None:
                misses.append(item)
            else:
                vectors_by_hash[item.content_hash] = cached

        execution = self._executor.embed(misses)
        _validate_execution_identity(self.identity, execution)
        for vector in execution.vectors:
            self._cache.put(self.identity, vector)
            vectors_by_hash[vector.content_hash] = vector
        if set(vectors_by_hash) != {item.content_hash for item in unique_inputs}:
            raise EmbeddingError(EmbeddingErrorCode.INVALID_RESPONSE)

        records = tuple(
            _vector_record(
                namespace,
                self.identity,
                chunk,
                vectors_by_hash[chunk.content_hash],
            )
            for chunk in corpus.chunks
        )
        write_result = self._vector_store.replace_namespace(namespace, records)
        return EmbeddingIndexResult(
            identity=self.identity,
            namespace=namespace,
            chunk_count=len(corpus.chunks),
            unique_content_count=len(unique_inputs),
            cache_hit_count=len(unique_inputs) - len(misses),
            cache_miss_count=len(misses),
            provider_batch_count=execution.batch_count,
            provider_call_count=execution.provider_call_count,
            retry_count=execution.retry_count,
            write_result=write_result,
        )


def embedding_identity_from_settings(
    settings: RepoPilotSettings,
) -> EmbeddingIdentity:
    """Build the canonical cache and namespace identity from validated settings."""

    return EmbeddingIdentity(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        revision=settings.embedding_revision,
        code_revision=settings.embedding_code_revision,
        dimension=settings.vector_dimension,
        normalize=settings.embedding_normalize,
        max_length=settings.embedding_max_length,
    )


def embedding_provider_from_settings(
    settings: RepoPilotSettings,
) -> EmbeddingProvider:
    """Construct a lazy Provider; Jina weights are not loaded here."""

    identity = embedding_identity_from_settings(settings)
    if settings.embedding_provider is EmbeddingProviderKind.FAKE:
        return FakeEmbeddingProvider(identity)
    return JinaEmbeddingProvider(
        identity,
        trust_remote_code=settings.embedding_trust_remote_code,
    )


def embedding_executor_from_settings(
    settings: RepoPilotSettings,
    provider: EmbeddingProvider,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> EmbeddingBatchExecutor:
    """Apply the configured batch and retry policy to one Provider."""

    expected_identity = embedding_identity_from_settings(settings)
    if provider.identity != expected_identity:
        raise EmbeddingError(EmbeddingErrorCode.IDENTITY_MISMATCH)
    return EmbeddingBatchExecutor(
        provider,
        batch_size=settings.embedding_batch_size,
        max_retries=settings.embedding_max_retries,
        retry_backoff_ms=settings.embedding_retry_backoff_ms,
        sleep=sleep,
    )


def embedding_cache_from_settings(
    settings: RepoPilotSettings,
    *,
    project_root: Path = PROJECT_ROOT,
) -> LocalEmbeddingCache:
    """Resolve a contained project-local cache directory."""

    try:
        root = project_root.resolve(strict=True)
        target = (root / settings.embedding_cache_dir).resolve()
    except OSError:
        raise EmbeddingError(EmbeddingErrorCode.CACHE_IO_ERROR) from None
    if not target.is_relative_to(root):
        raise EmbeddingError(EmbeddingErrorCode.CACHE_IO_ERROR)
    return LocalEmbeddingCache(target)


def _unique_embedding_inputs(
    chunks: tuple[CodeChunk, ...],
) -> tuple[EmbeddingInput, ...]:
    inputs_by_hash: dict[str, EmbeddingInput] = {}
    for chunk in chunks:
        candidate = EmbeddingInput(
            content_hash=chunk.content_hash,
            content=chunk.content,
        )
        previous = inputs_by_hash.get(candidate.content_hash)
        if previous is not None and previous.content != candidate.content:
            raise EmbeddingError(EmbeddingErrorCode.CORPUS_MISMATCH)
        inputs_by_hash.setdefault(candidate.content_hash, candidate)
    return tuple(inputs_by_hash.values())


def _validate_execution_identity(
    identity: EmbeddingIdentity,
    execution: EmbeddingExecutionResult,
) -> None:
    if execution.identity != identity:
        raise EmbeddingError(EmbeddingErrorCode.IDENTITY_MISMATCH)


def _vector_record(
    namespace: VectorNamespace,
    identity: EmbeddingIdentity,
    chunk: CodeChunk,
    vector: EmbeddingVector,
) -> VectorRecord:
    if vector.content_hash != chunk.content_hash:
        raise EmbeddingError(EmbeddingErrorCode.INVALID_RESPONSE)
    return VectorRecord(
        namespace=namespace,
        chunk_id=chunk.chunk_id,
        content_hash=chunk.content_hash,
        relative_path=chunk.relative_path,
        language=chunk.language,
        symbol_name=chunk.symbol_name,
        embedding=vector.values,
        metadata={
            "chunk": chunk.metadata,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbol_type": chunk.symbol_type.value,
            "embedding": identity.model_dump(mode="json"),
            "embedding_identity_fingerprint": identity.fingerprint,
        },
    )
