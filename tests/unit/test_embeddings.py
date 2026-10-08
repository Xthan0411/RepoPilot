"""Embedding identity, Fake batching, Jina boundary, cache, and indexing tests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.embeddings import (
    EmbeddingBatchExecutor,
    EmbeddingError,
    EmbeddingIdentity,
    EmbeddingIndexer,
    EmbeddingInput,
    EmbeddingProvider,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    FakeEmbeddingProvider,
    JinaEmbeddingProvider,
    LocalEmbeddingCache,
    embedding_cache_from_settings,
    embedding_cache_key,
    embedding_request_id,
)
from repopilot.enums import (
    EmbeddingErrorCode,
    EmbeddingProviderKind,
    SymbolType,
)
from repopilot.retrieval.corpus import ChunkCorpus, ChunkSourceSnapshot
from repopilot.schemas import CodeChunk
from repopilot.storage import InMemoryVectorStore

COMMIT = "a" * 40
JINA_REVISION = "516f4baf13dec4ddddda8631e019b5737c8bc250"
JINA_CODE_REVISION = "3baf9e3ac750e76e8edd3019170176884695fb94"
SHARED_CONTENT = "def answer() -> int:\n    return 42\n"
SHARED_CONTENT_HASH = hashlib.sha256(SHARED_CONTENT.encode("utf-8")).hexdigest()


def fake_identity(
    *,
    revision: str = "v1",
    dimension: int = 3,
    normalize: bool = True,
    max_length: int = 128,
) -> EmbeddingIdentity:
    return EmbeddingIdentity(
        provider=EmbeddingProviderKind.FAKE,
        model="fake-code-embedding",
        revision=revision,
        dimension=dimension,
        normalize=normalize,
        max_length=max_length,
    )


def jina_identity(*, dimension: int = 3) -> EmbeddingIdentity:
    return EmbeddingIdentity(
        provider=EmbeddingProviderKind.JINA_LOCAL,
        model="jinaai/jina-embeddings-v2-base-code",
        revision=JINA_REVISION,
        code_revision=JINA_CODE_REVISION,
        dimension=dimension,
        normalize=True,
        max_length=2_048,
    )


def embedding_input(content: str) -> EmbeddingInput:
    return EmbeddingInput(
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        content=content,
    )


def request_for(
    identity: EmbeddingIdentity,
    inputs: tuple[EmbeddingInput, ...],
    *,
    attempt: int = 1,
) -> EmbeddingRequest:
    return EmbeddingRequest(
        request_id=embedding_request_id(identity, inputs, attempt),
        identity=identity,
        inputs=inputs,
        attempt=attempt,
    )


def assert_embedding_error(
    captured: pytest.ExceptionInfo[EmbeddingError],
    code: EmbeddingErrorCode,
) -> None:
    assert captured.value.code is code


def fixture_corpus(*, repository_id: str = "fixture.embedding") -> ChunkCorpus:
    chunks: list[CodeChunk] = []
    sources: list[ChunkSourceSnapshot] = []
    for index, path in enumerate(("a.py", "b.py")):
        source_id = f"source_{index}"
        source_hash = str(index + 1) * 64
        chunk_id = f"chunk_{index}"
        chunk = CodeChunk(
            chunk_id=chunk_id,
            repository_id=repository_id,
            commit=COMMIT,
            relative_path=path,
            symbol_name=f"answer_{index}",
            symbol_type=SymbolType.FUNCTION,
            start_line=1,
            end_line=2,
            language="python",
            content=SHARED_CONTENT,
            content_hash=SHARED_CONTENT_HASH,
            metadata={
                "source_id": source_id,
                "source_content_hash": source_hash,
                "processor": "fixture",
                "part_index": 0,
            },
        )
        chunks.append(chunk)
        sources.append(
            ChunkSourceSnapshot(
                source_id=source_id,
                relative_path=path,
                source_content_hash=source_hash,
                processor="fixture",
                build_fingerprint="f" * 64,
                chunk_ids=(chunk_id,),
            )
        )
    return ChunkCorpus(
        repository_id=repository_id,
        commit=COMMIT,
        sources=tuple(sources),
        chunks=tuple(chunks),
    )


def test_identity_fingerprint_and_cache_key_cover_encoding_policy() -> None:
    original = fake_identity()
    changed_revision = fake_identity(revision="v2")
    changed_code_revision = original.model_copy(update={"code_revision": "v2"})
    changed_policy = fake_identity(max_length=256)

    assert original.fingerprint == fake_identity().fingerprint
    assert len(original.fingerprint) == 64
    assert (
        len(
            {
                original.fingerprint,
                changed_revision.fingerprint,
                changed_code_revision.fingerprint,
                changed_policy.fingerprint,
            }
        )
        == 4
    )
    assert embedding_cache_key(original, SHARED_CONTENT_HASH) != embedding_cache_key(
        changed_revision,
        SHARED_CONTENT_HASH,
    )


def test_embedding_input_and_request_reject_tampering_or_duplicate_content() -> None:
    with pytest.raises(ValidationError):
        EmbeddingInput(content_hash="0" * 64, content="different")

    identity = fake_identity()
    item = embedding_input("same")
    with pytest.raises(ValidationError, match="unique"):
        request_for(identity, (item, item))
    with pytest.raises(ValidationError, match="request_id"):
        EmbeddingRequest(
            request_id="0" * 64,
            identity=identity,
            inputs=(item,),
            attempt=1,
        )


def test_fake_provider_is_deterministic_normalized_and_matches_protocol() -> None:
    identity = fake_identity(dimension=17)
    first = FakeEmbeddingProvider(identity)
    second = FakeEmbeddingProvider(identity)
    protocol_provider: EmbeddingProvider = first
    inputs = (embedding_input("alpha"), embedding_input("beta"))
    request = request_for(identity, inputs)

    first_response = protocol_provider.embed(request)
    second_response = second.embed(request)

    assert first_response == second_response
    assert tuple(vector.content_hash for vector in first_response.vectors) == tuple(
        item.content_hash for item in inputs
    )
    for vector in first_response.vectors:
        assert len(vector.values) == 17
        assert math.sqrt(math.fsum(value * value for value in vector.values)) == pytest.approx(1.0)


def test_batch_executor_batches_retries_and_preserves_input_order() -> None:
    identity = fake_identity()
    provider = FakeEmbeddingProvider(identity, fail_first_calls=1)
    sleeps: list[float] = []
    executor = EmbeddingBatchExecutor(
        provider,
        batch_size=2,
        max_retries=2,
        retry_backoff_ms=100,
        sleep=sleeps.append,
    )
    inputs = tuple(embedding_input(value) for value in ("one", "two", "three"))

    result = executor.embed(inputs)

    assert tuple(vector.content_hash for vector in result.vectors) == tuple(
        item.content_hash for item in inputs
    )
    assert result.batch_count == 2
    assert result.provider_call_count == 3
    assert result.retry_count == 1
    assert sleeps == [0.1]
    assert tuple(request.attempt for request in provider.requests) == (1, 2, 1)


def test_batch_executor_reports_retry_exhaustion_without_leaking_content() -> None:
    provider = FakeEmbeddingProvider(fake_identity(), fail_first_calls=100)
    executor = EmbeddingBatchExecutor(
        provider,
        batch_size=1,
        max_retries=1,
        retry_backoff_ms=0,
        sleep=lambda _: None,
    )
    secret = "secret source content"

    with pytest.raises(EmbeddingError) as captured:
        executor.embed((embedding_input(secret),))

    assert_embedding_error(captured, EmbeddingErrorCode.RETRY_EXHAUSTED)
    assert provider.calls_made == 2
    assert secret not in str(captured.value)


class InvalidResponseProvider:
    def __init__(self, identity: EmbeddingIdentity) -> None:
        self._identity = identity

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        return cast(EmbeddingResponse, object())


def test_batch_executor_rejects_uncorrelated_provider_response_without_retry() -> None:
    executor = EmbeddingBatchExecutor(
        InvalidResponseProvider(fake_identity()),
        batch_size=1,
        max_retries=5,
        retry_backoff_ms=100,
        sleep=lambda _: pytest.fail("invalid responses must not be retried"),
    )

    with pytest.raises(EmbeddingError) as captured:
        executor.embed((embedding_input("input"),))

    assert_embedding_error(captured, EmbeddingErrorCode.INVALID_RESPONSE)


def test_local_cache_round_trip_identity_isolation_and_conflict(tmp_path: Path) -> None:
    cache = LocalEmbeddingCache(tmp_path / "cache")
    identity = fake_identity()
    other_identity = fake_identity(revision="v2")
    vector = EmbeddingVector(
        content_hash=SHARED_CONTENT_HASH,
        values=(1.0, 0.0, 0.0),
    )

    assert cache.get(identity, SHARED_CONTENT_HASH) is None
    cache.put(identity, vector)

    assert cache.get(identity, SHARED_CONTENT_HASH) == vector
    assert cache.get(other_identity, SHARED_CONTENT_HASH) is None
    cache.put(identity, vector)
    with pytest.raises(EmbeddingError) as captured:
        cache.put(
            identity,
            vector.model_copy(update={"values": (0.0, 1.0, 0.0)}),
        )
    assert_embedding_error(captured, EmbeddingErrorCode.CACHE_CORRUPT)


def test_local_cache_detects_valid_json_tampering(tmp_path: Path) -> None:
    cache = LocalEmbeddingCache(tmp_path / "cache")
    identity = fake_identity()
    vector = EmbeddingVector(
        content_hash=SHARED_CONTENT_HASH,
        values=(1.0, 0.0, 0.0),
    )
    cache.put(identity, vector)
    cache_path = next(cache.root.rglob("*.json"))
    document = json.loads(cache_path.read_text(encoding="utf-8"))
    document["vector"]["values"][0] = 0.5
    cache_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(EmbeddingError) as captured:
        cache.get(identity, SHARED_CONTENT_HASH)

    assert_embedding_error(captured, EmbeddingErrorCode.CACHE_CORRUPT)


def test_cache_factory_rejects_symlink_escape(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    outside = tmp_path / "outside"
    project_root.joinpath("artifacts").mkdir(parents=True)
    outside.mkdir()
    project_root.joinpath("artifacts", "embedding-cache").symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(EmbeddingError) as captured:
        embedding_cache_from_settings(RepoPilotSettings(), project_root=project_root)

    assert_embedding_error(captured, EmbeddingErrorCode.CACHE_IO_ERROR)


class StubSentenceEncoder:
    def __init__(self, *, dimension: int) -> None:
        self.dimension = dimension
        self.max_seq_length = 0
        self.calls: list[dict[str, object]] = []

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> object:
        self.calls.append(
            {
                "sentences": tuple(sentences),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
            }
        )
        return [[1.0, 0.0, 0.0] for _ in sentences]


def test_jina_adapter_is_lazy_pinned_and_bounded_with_injected_encoder() -> None:
    identity = jina_identity()
    encoder = StubSentenceEncoder(dimension=3)
    factory_calls: list[tuple[str, str, str, bool]] = []

    def factory(
        model: str,
        revision: str,
        code_revision: str,
        trust_remote_code: bool,
    ) -> StubSentenceEncoder:
        factory_calls.append((model, revision, code_revision, trust_remote_code))
        return encoder

    provider = JinaEmbeddingProvider(
        identity,
        trust_remote_code=True,
        model_factory=factory,
    )
    inputs = (embedding_input("query"), embedding_input("code"))

    assert provider.loaded is False
    response = provider.embed(request_for(identity, inputs))

    assert provider.loaded is True
    assert factory_calls == [(identity.model, JINA_REVISION, JINA_CODE_REVISION, True)]
    assert encoder.max_seq_length == 2_048
    assert encoder.calls == [
        {
            "sentences": ("query", "code"),
            "batch_size": 2,
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
    ]
    assert len(response.vectors) == 2


def test_jina_adapter_requires_explicit_remote_code_for_default_loader() -> None:
    identity = jina_identity()
    provider = JinaEmbeddingProvider(identity, trust_remote_code=False)

    with pytest.raises(EmbeddingError) as captured:
        provider.embed(request_for(identity, (embedding_input("code"),)))

    assert_embedding_error(captured, EmbeddingErrorCode.REMOTE_CODE_DISABLED)
    assert provider.loaded is False


def test_jina_adapter_rejects_loaded_model_dimension_mismatch() -> None:
    identity = jina_identity()
    provider = JinaEmbeddingProvider(
        identity,
        trust_remote_code=False,
        model_factory=lambda *_: StubSentenceEncoder(dimension=4),
    )

    with pytest.raises(EmbeddingError) as captured:
        provider.embed(request_for(identity, (embedding_input("code"),)))

    assert_embedding_error(captured, EmbeddingErrorCode.INVALID_RESPONSE)


def test_indexer_embeds_unique_misses_reuses_cache_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    corpus = fixture_corpus()
    identity = fake_identity()
    provider = FakeEmbeddingProvider(identity)
    executor = EmbeddingBatchExecutor(
        provider,
        batch_size=1,
        max_retries=0,
        retry_backoff_ms=0,
        sleep=lambda _: None,
    )
    store = InMemoryVectorStore(dimension=3)
    indexer = EmbeddingIndexer(
        executor=executor,
        cache=LocalEmbeddingCache(tmp_path / "cache"),
        vector_store=store,
    )

    first = indexer.index(corpus)
    second = indexer.index(corpus)
    records = store.list_records(first.namespace)

    assert first.chunk_count == 2
    assert first.unique_content_count == 1
    assert first.cache_hit_count == 0
    assert first.cache_miss_count == 1
    assert first.provider_batch_count == first.provider_call_count == 1
    assert second.cache_hit_count == 1
    assert second.cache_miss_count == 0
    assert second.provider_call_count == 0
    assert provider.calls_made == 1
    assert len(records) == 2
    assert records[0].embedding == records[1].embedding
    assert records[0].metadata["start_line"] == 1
    assert records[0].metadata["embedding_identity_fingerprint"] == identity.fingerprint
    assert records[0].metadata["embedding"] == identity.model_dump(mode="json")


def test_indexer_identity_change_forces_cache_miss_and_new_namespace(
    tmp_path: Path,
) -> None:
    corpus = fixture_corpus()
    cache = LocalEmbeddingCache(tmp_path / "cache")
    store = InMemoryVectorStore(dimension=3)
    first_identity = fake_identity(revision="v1")
    second_identity = fake_identity(revision="v2")

    def build_indexer(
        identity: EmbeddingIdentity,
    ) -> tuple[EmbeddingIndexer, FakeEmbeddingProvider]:
        provider = FakeEmbeddingProvider(identity)
        return (
            EmbeddingIndexer(
                executor=EmbeddingBatchExecutor(
                    provider,
                    batch_size=8,
                    max_retries=0,
                    retry_backoff_ms=0,
                    sleep=lambda _: None,
                ),
                cache=cache,
                vector_store=store,
            ),
            provider,
        )

    first_indexer, first_provider = build_indexer(first_identity)
    second_indexer, second_provider = build_indexer(second_identity)
    first = first_indexer.index(corpus)
    second = second_indexer.index(corpus)

    assert first.cache_miss_count == second.cache_miss_count == 1
    assert first_provider.calls_made == second_provider.calls_made == 1
    assert first.namespace.embedding_version == first_identity.fingerprint
    assert second.namespace.embedding_version == second_identity.fingerprint
    assert first.namespace != second.namespace
    assert len(store.list_records(first.namespace)) == 2
    assert len(store.list_records(second.namespace)) == 2


def test_indexer_handles_empty_corpus_without_provider_call(tmp_path: Path) -> None:
    corpus = ChunkCorpus(
        repository_id="fixture.empty",
        commit=COMMIT,
        sources=(),
        chunks=(),
    )
    identity = fake_identity()
    provider = FakeEmbeddingProvider(identity)
    indexer = EmbeddingIndexer(
        executor=EmbeddingBatchExecutor(
            provider,
            batch_size=8,
            max_retries=0,
            retry_backoff_ms=0,
            sleep=lambda _: None,
        ),
        cache=LocalEmbeddingCache(tmp_path / "cache"),
        vector_store=InMemoryVectorStore(dimension=3),
    )

    result = indexer.index(corpus)

    assert result.chunk_count == result.unique_content_count == 0
    assert result.provider_batch_count == result.provider_call_count == 0
    assert result.write_result.total_count == 0
    assert provider.calls_made == 0
