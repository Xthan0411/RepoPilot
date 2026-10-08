"""Vector contracts, in-memory lifecycle, and safe PostgreSQL adapter tests."""

from __future__ import annotations

import math
from typing import Self, cast

import pytest
from pydantic import ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.enums import VectorStoreErrorCode
from repopilot.storage import (
    InMemoryVectorStore,
    PostgresVectorStore,
    VectorNamespace,
    VectorRecord,
    VectorStore,
    VectorStoreError,
    validate_chunk_allowlist,
    validate_record_batch,
)
from repopilot.storage.postgres import (
    _parse_vector,
    _psycopg_dsn,
    _quoted_identifier,
    _search_hit_from_row,
    _vector_literal,
)

COMMIT = "a" * 40
CONTENT_HASH = "b" * 64


class FakeCursor:
    def __init__(self, *, dimension: int = 3) -> None:
        self.dimension = dimension
        self.executions: list[tuple[str, object | None]] = []
        self.rowcount = 0
        self._last_query = ""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, query: str, parameters: object | None = None) -> None:
        self._last_query = query
        self.executions.append((query, parameters))

    def fetchone(self) -> tuple[object, ...] | None:
        if "SELECT schema_version" in self._last_query:
            return (1, self.dimension)
        if "SELECT extversion" in self._last_query:
            return ("0.8.2",)
        if "RETURNING chunk_id" in self._last_query:
            return ("chunk_a",)
        if "SELECT COUNT(*)" in self._last_query:
            return (1,)
        return None

    def fetchall(self) -> list[tuple[object, ...]]:
        if "WITH query AS" in self._last_query:
            return [
                (
                    "chunk_a",
                    CONTENT_HASH,
                    "pkg/chunk_a.py",
                    "python",
                    "symbol_chunk_a",
                    {"fixture": True},
                    0.75,
                )
            ]
        return []


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self._cursor


def namespace(
    *,
    repository_id: str = "fixture.storage",
    commit: str = COMMIT,
    model: str = "fake-embedding",
    version: str = "v1",
    dimension: int = 3,
) -> VectorNamespace:
    return VectorNamespace(
        repository_id=repository_id,
        commit=commit,
        embedding_model=model,
        embedding_version=version,
        dimension=dimension,
    )


def record(
    chunk_id: str,
    *,
    vector_namespace: VectorNamespace | None = None,
    embedding: tuple[float, ...] = (1.0, 0.0, 0.0),
    content_hash: str = CONTENT_HASH,
) -> VectorRecord:
    return VectorRecord(
        namespace=vector_namespace or namespace(),
        chunk_id=chunk_id,
        content_hash=content_hash,
        relative_path=f"pkg/{chunk_id}.py",
        language="python",
        symbol_name=f"symbol_{chunk_id}",
        embedding=embedding,
        metadata={"fixture": True},
    )


def assert_store_error(
    captured: pytest.ExceptionInfo[VectorStoreError],
    code: VectorStoreErrorCode,
) -> None:
    assert captured.value.code is code


def test_vector_namespace_and_record_reject_invalid_dimensions_and_values() -> None:
    jina_namespace = namespace(
        model="jinaai/jina-embeddings-v2-base-code",
        version="516f4baf13dec4ddddda8631e019b5737c8bc250",
    )
    assert jina_namespace.embedding_model == "jinaai/jina-embeddings-v2-base-code"
    assert jina_namespace.embedding_version.startswith("516f4ba")

    for invalid_dimension in (0, 2_001, True):
        with pytest.raises(ValidationError):
            namespace(dimension=invalid_dimension)

    for invalid_embedding in (
        (1.0, 2.0),
        (1.0, math.nan, 0.0),
        (1.0, math.inf, 0.0),
        (1.0, True, 0.0),
        (1.0, "2", 0.0),
        (1.0, 3.41e38, 0.0),
        (0.0, 0.0, 0.0),
    ):
        with pytest.raises(ValidationError):
            record(
                "chunk_invalid",
                embedding=cast(tuple[float, ...], invalid_embedding),
            )


def test_batch_validation_sorts_and_rejects_duplicates_or_mixed_namespace() -> None:
    requested = namespace()
    assert tuple(
        item.chunk_id
        for item in validate_record_batch(
            requested,
            (record("chunk_z"), record("chunk_a")),
        )
    ) == ("chunk_a", "chunk_z")

    with pytest.raises(VectorStoreError) as captured:
        validate_record_batch(
            requested,
            (record("chunk_a"), record("chunk_a")),
        )
    assert_store_error(captured, VectorStoreErrorCode.DUPLICATE_RECORD)

    with pytest.raises(VectorStoreError) as captured:
        validate_record_batch(
            requested,
            (record("chunk_a", vector_namespace=namespace(commit="c" * 40)),),
        )
    assert_store_error(captured, VectorStoreErrorCode.NAMESPACE_MISMATCH)


def test_in_memory_store_requires_initialize_and_matches_protocol() -> None:
    store = InMemoryVectorStore(dimension=3)
    protocol_store: VectorStore = store

    with pytest.raises(VectorStoreError) as captured:
        protocol_store.list_records(namespace())
    assert_store_error(captured, VectorStoreErrorCode.NOT_INITIALIZED)

    info = protocol_store.initialize()

    assert info.backend == "memory"
    assert info.dimension == 3


def test_in_memory_upsert_is_idempotent_and_rejects_chunk_hash_conflict() -> None:
    store = InMemoryVectorStore(dimension=3)
    store.initialize()
    requested = namespace()
    original = record("chunk_a")
    refreshed = original.model_copy(update={"embedding": (0.0, 1.0, 0.0)})

    first = store.upsert(requested, (original,))
    second = store.upsert(requested, (refreshed,))

    assert first.written_count == second.written_count == 1
    assert first.total_count == second.total_count == 1
    assert store.list_records(requested) == (refreshed,)

    conflict = original.model_copy(update={"content_hash": "c" * 64})
    with pytest.raises(VectorStoreError) as captured:
        store.upsert(requested, (conflict,))
    assert_store_error(captured, VectorStoreErrorCode.INVALID_RECORD)
    assert store.list_records(requested) == (refreshed,)


def test_replace_delete_and_namespaces_are_exactly_isolated() -> None:
    store = InMemoryVectorStore(dimension=3)
    store.initialize()
    first_namespace = namespace(repository_id="repo.first")
    second_namespace = namespace(repository_id="repo.second")
    first_records = (
        record("chunk_a", vector_namespace=first_namespace),
        record("chunk_b", vector_namespace=first_namespace),
    )
    second_record = record("chunk_a", vector_namespace=second_namespace)
    store.upsert(first_namespace, first_records)
    store.upsert(second_namespace, (second_record,))

    replacement = store.replace_namespace(first_namespace, (first_records[1],))

    assert replacement.written_count == 1
    assert replacement.deleted_count == 1
    assert replacement.total_count == 1
    assert store.list_records(first_namespace) == (first_records[1],)
    assert store.list_records(second_namespace) == (second_record,)

    deleted = store.delete_namespace(first_namespace)
    repeated = store.delete_namespace(first_namespace)

    assert deleted.deleted_count == 1
    assert repeated.deleted_count == 0
    assert store.list_records(first_namespace) == ()
    assert store.list_records(second_namespace) == (second_record,)


def test_in_memory_cosine_search_is_ranked_tied_and_namespace_isolated() -> None:
    store = InMemoryVectorStore(dimension=3)
    store.initialize()
    requested = namespace(repository_id="repo.search")
    other = namespace(repository_id="repo.other")
    records = (
        record("chunk_z", vector_namespace=requested, embedding=(1.0, 0.0, 0.0)),
        record("chunk_a", vector_namespace=requested, embedding=(1.0, 0.0, 0.0)),
        record("chunk_b", vector_namespace=requested, embedding=(0.8, 0.6, 0.0)),
        record("chunk_c", vector_namespace=requested, embedding=(-1.0, 0.0, 0.0)),
    )
    store.upsert(requested, records)
    store.upsert(
        other,
        (record("chunk_other", vector_namespace=other, embedding=(1.0, 0.0, 0.0)),),
    )

    hits = store.search(requested, (1.0, 0.0, 0.0), top_k=3)

    assert tuple(hit.chunk_id for hit in hits) == ("chunk_a", "chunk_z", "chunk_b")
    assert tuple(hit.score for hit in hits) == pytest.approx((1.0, 1.0, 0.8))
    assert all(hit.namespace == requested for hit in hits)


def test_vector_search_allowlist_filters_before_ranking() -> None:
    store = InMemoryVectorStore(dimension=3)
    store.initialize()
    requested = namespace(repository_id="repo.filtered")
    store.upsert(
        requested,
        (
            record(
                "chunk_a",
                vector_namespace=requested,
                embedding=(1.0, 0.0, 0.0),
            ),
            record(
                "chunk_b",
                vector_namespace=requested,
                embedding=(0.8, 0.6, 0.0),
            ),
            record(
                "chunk_c",
                vector_namespace=requested,
                embedding=(-1.0, 0.0, 0.0),
            ),
        ),
    )

    hits = store.search(
        requested,
        (1.0, 0.0, 0.0),
        top_k=3,
        allowed_chunk_ids=("chunk_c", "chunk_b"),
    )
    empty = store.search(
        requested,
        (1.0, 0.0, 0.0),
        top_k=3,
        allowed_chunk_ids=(),
    )

    assert tuple(hit.chunk_id for hit in hits) == ("chunk_b", "chunk_c")
    assert empty == ()
    assert validate_chunk_allowlist(("chunk_c", "chunk_b")) == (
        "chunk_b",
        "chunk_c",
    )


@pytest.mark.parametrize(
    "allowed_chunk_ids",
    [
        ("chunk_a", "chunk_a"),
        ("invalid/chunk",),
        ("1invalid",),
    ],
)
def test_vector_search_rejects_invalid_allowlist(
    allowed_chunk_ids: tuple[str, ...],
) -> None:
    store = InMemoryVectorStore(dimension=3)
    store.initialize()

    with pytest.raises(VectorStoreError) as captured:
        store.search(
            namespace(),
            (1.0, 0.0, 0.0),
            top_k=1,
            allowed_chunk_ids=allowed_chunk_ids,
        )

    assert_store_error(captured, VectorStoreErrorCode.INVALID_RECORD)


@pytest.mark.parametrize(
    ("query", "top_k"),
    [
        ((0.0, 0.0, 0.0), 1),
        ((1.0, 0.0), 1),
        ((math.nan, 0.0, 1.0), 1),
        ((1.0, 0.0, 0.0), 0),
        ((1.0, 0.0, 0.0), 101),
        ((1.0, 0.0, 0.0), True),
    ],
)
def test_vector_search_rejects_invalid_cosine_queries(
    query: tuple[float, ...],
    top_k: int,
) -> None:
    store = InMemoryVectorStore(dimension=3)
    store.initialize()

    with pytest.raises(VectorStoreError) as captured:
        store.search(namespace(), query, top_k=top_k)

    assert_store_error(captured, VectorStoreErrorCode.INVALID_RECORD)


def test_store_rejects_namespace_dimension_different_from_backend() -> None:
    store = InMemoryVectorStore(dimension=3)
    store.initialize()

    with pytest.raises(VectorStoreError) as captured:
        store.list_records(namespace(dimension=2))

    assert_store_error(captured, VectorStoreErrorCode.DIMENSION_MISMATCH)


def test_postgres_adapter_does_not_connect_before_initialize_and_hides_failure() -> None:
    secret = "database-adapter-secret"

    def fail_connect(*args: object, **kwargs: object) -> object:
        raise RuntimeError(secret)

    store = PostgresVectorStore(
        RepoPilotSettings(vector_dimension=3),
        connect_factory=fail_connect,
    )
    with pytest.raises(VectorStoreError) as captured:
        store.list_records(namespace())
    assert_store_error(captured, VectorStoreErrorCode.NOT_INITIALIZED)

    with pytest.raises(VectorStoreError) as captured:
        store.initialize()

    assert_store_error(captured, VectorStoreErrorCode.CONNECTION_ERROR)
    assert secret not in str(captured.value)
    assert "postgresql://" not in str(captured.value)


def test_postgres_adapter_renders_ddl_and_binds_vector_values_as_parameters() -> None:
    cursor = FakeCursor()
    connection = FakeConnection(cursor)
    captured_connection: dict[str, object] = {}

    def connect(dsn: str, **kwargs: object) -> FakeConnection:
        captured_connection["dsn"] = dsn
        captured_connection.update(kwargs)
        return connection

    store = PostgresVectorStore(
        RepoPilotSettings(
            vector_schema="custom_schema",
            vector_table="custom_vectors",
            vector_dimension=3,
            database_connect_timeout_seconds=8,
            database_statement_timeout_ms=9_000,
        ),
        connect_factory=connect,
    )

    info = store.initialize()
    result = store.upsert(namespace(), (record("chunk_a"),))
    search_hits = store.search(
        namespace(),
        (1.0, 0.0, 0.0),
        top_k=5,
        allowed_chunk_ids=("chunk_a",),
    )

    assert info.model_dump() == {
        "schema_version": 1,
        "backend": "pgvector",
        "dimension": 3,
        "schema_name": "custom_schema",
        "table_name": "custom_vectors",
        "extension_version": "0.8.2",
    }
    assert result.written_count == result.total_count == 1
    assert captured_connection["connect_timeout"] == 8
    migration = next(query for query, _ in cursor.executions if "CREATE EXTENSION" in query)
    assert '"custom_schema"."custom_vectors"' in migration
    assert "{schema}" not in migration
    upsert_query, upsert_parameters = next(
        (query, parameters)
        for query, parameters in cursor.executions
        if "RETURNING chunk_id" in query
    )
    assert "%s::vector" in upsert_query
    assert "[1,0,0]" not in upsert_query
    assert isinstance(upsert_parameters, tuple)
    assert "[1,0,0]" in upsert_parameters
    migration = next(query for query, _ in cursor.executions if "CREATE EXTENSION" in query)
    assert "USING hnsw" in migration
    assert "vector(3)" in migration
    search_query, search_parameters = next(
        (query, parameters) for query, parameters in cursor.executions if "WITH query AS" in query
    )
    assert "1 - (" in search_query
    assert "<=>" in search_query
    assert "target.chunk_id = ANY(%s)" in search_query
    assert "ORDER BY" in search_query
    assert isinstance(search_parameters, tuple)
    assert search_parameters[0] == "[1,0,0]"
    assert search_parameters[-2] == ["chunk_a"]
    assert search_parameters[-1] == 5
    assert tuple(hit.chunk_id for hit in search_hits) == ("chunk_a",)
    assert search_hits[0].score == 0.75


def test_postgres_initialize_rejects_existing_dimension_mismatch() -> None:
    cursor = FakeCursor(dimension=4)
    store = PostgresVectorStore(
        RepoPilotSettings(vector_dimension=3),
        connect_factory=lambda *args, **kwargs: FakeConnection(cursor),
    )

    with pytest.raises(VectorStoreError) as captured:
        store.initialize()

    assert_store_error(captured, VectorStoreErrorCode.SCHEMA_MISMATCH)


def test_postgres_helpers_are_deterministic_and_reject_unsafe_identifier() -> None:
    assert (
        _psycopg_dsn("postgresql+psycopg://user:password@localhost/database")
        == "postgresql://user:password@localhost/database"
    )
    assert _quoted_identifier("vector_embeddings") == '"vector_embeddings"'
    assert _vector_literal((1.0, -0.25, 0.000_001)) == "[1,-0.25,1e-06]"
    assert _parse_vector("[1,-0.25,1e-06]") == (1.0, -0.25, 0.000_001)

    for value in ("", "Bad", "bad-name", "bad name", "1table", "a" * 64):
        with pytest.raises(VectorStoreError) as captured:
            _quoted_identifier(value)
        assert_store_error(captured, VectorStoreErrorCode.SCHEMA_MISMATCH)


def test_postgres_search_row_rejects_invalid_score_or_metadata() -> None:
    requested = namespace()
    valid_prefix: tuple[object, ...] = (
        "chunk_a",
        CONTENT_HASH,
        "pkg/chunk_a.py",
        "python",
        None,
    )
    for row in (
        (*valid_prefix, [], 0.5),
        (*valid_prefix, {}, math.nan),
        (*valid_prefix, {}, 1.1),
        (*valid_prefix, {}, True),
    ):
        with pytest.raises(VectorStoreError) as captured:
            _search_hit_from_row(requested, row)
        assert_store_error(captured, VectorStoreErrorCode.SCHEMA_MISMATCH)
