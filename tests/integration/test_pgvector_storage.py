"""Docker-gated PostgreSQL 16 + pgvector lifecycle contract test."""

from __future__ import annotations

import importlib
import importlib.util
import os
from uuid import uuid4

import pytest
from pydantic import SecretStr

from repopilot.config import RepoPilotSettings
from repopilot.storage import (
    PostgresVectorStore,
    VectorNamespace,
    VectorRecord,
)

pytestmark = pytest.mark.integration


def test_pgvector_store_is_idempotent_isolated_and_cleanable() -> None:
    dsn = os.environ.get("REPOPILOT_TEST_DATABASE_URL")
    if dsn is None:
        pytest.skip("set REPOPILOT_TEST_DATABASE_URL to run pgvector integration")
    if importlib.util.find_spec("psycopg") is None:
        pytest.skip("install the locked Psycopg dependency to run pgvector integration")

    suffix = uuid4().hex[:12]
    schema = f"repopilot_it_{suffix}"
    settings = RepoPilotSettings(
        database_url=SecretStr(dsn),
        vector_schema=schema,
        vector_table="vectors",
        vector_dimension=3,
    )
    store = PostgresVectorStore(settings)
    first = VectorNamespace(
        repository_id="integration.first",
        commit="a" * 40,
        embedding_model="fake-embedding",
        embedding_version="v1",
        dimension=3,
    )
    isolated_namespaces = (
        first.model_copy(update={"repository_id": "integration.second"}),
        first.model_copy(update={"commit": "c" * 40}),
        first.model_copy(update={"embedding_model": "fake-embedding-next"}),
        first.model_copy(update={"embedding_version": "v2"}),
    )
    first_records = (
        _record(first, "chunk_a", (1.0, 0.0, 0.0)),
        _record(first, "chunk_z", (1.0, 0.0, 0.0)),
        _record(first, "chunk_b", (0.8, 0.6, 0.0)),
        _record(first, "chunk_c", (-1.0, 0.0, 0.0)),
    )
    isolated_records = tuple(
        _record(namespace, "chunk_a", (0.0, 1.0, 0.0)) for namespace in isolated_namespaces
    )

    failure: BaseException | None = None
    try:
        info = store.initialize()
        repeated_info = store.initialize()
        assert info == repeated_info
        assert info.backend == "pgvector"
        assert info.extension_version is not None

        assert store.upsert(first, first_records).total_count == 4
        assert store.upsert(first, first_records).total_count == 4
        for isolated_namespace, isolated_record in zip(
            isolated_namespaces,
            isolated_records,
            strict=True,
        ):
            assert store.upsert(isolated_namespace, (isolated_record,)).total_count == 1
        assert store.list_records(first) == tuple(
            sorted(first_records, key=lambda record: record.chunk_id)
        )
        search_hits = store.search(first, (1.0, 0.0, 0.0), top_k=3)
        assert tuple(hit.chunk_id for hit in search_hits) == (
            "chunk_a",
            "chunk_z",
            "chunk_b",
        )
        assert tuple(hit.score for hit in search_hits) == pytest.approx((1.0, 1.0, 0.8))
        filtered_hits = store.search(
            first,
            (1.0, 0.0, 0.0),
            top_k=3,
            allowed_chunk_ids=("chunk_c", "chunk_b"),
        )
        assert tuple(hit.chunk_id for hit in filtered_hits) == (
            "chunk_b",
            "chunk_c",
        )
        assert tuple(hit.score for hit in filtered_hits) == pytest.approx((0.8, -1.0))
        for isolated_namespace, isolated_record in zip(
            isolated_namespaces,
            isolated_records,
            strict=True,
        ):
            assert store.list_records(isolated_namespace) == (isolated_record,)

        replacement = store.replace_namespace(first, ())
        assert replacement.deleted_count == 4
        assert store.list_records(first) == ()
        for isolated_namespace, isolated_record in zip(
            isolated_namespaces,
            isolated_records,
            strict=True,
        ):
            assert store.list_records(isolated_namespace) == (isolated_record,)
            assert store.delete_namespace(isolated_namespace).deleted_count == 1
            assert store.delete_namespace(isolated_namespace).deleted_count == 0
    except BaseException as error:
        failure = error
        raise
    finally:
        _drop_schema(dsn, schema, suppress_error=failure is not None)


def _record(
    namespace: VectorNamespace,
    chunk_id: str,
    embedding: tuple[float, ...],
) -> VectorRecord:
    return VectorRecord(
        namespace=namespace,
        chunk_id=chunk_id,
        content_hash="b" * 64,
        relative_path=f"pkg/{chunk_id}.py",
        language="python",
        embedding=embedding,
        metadata={"integration": True},
    )


def _drop_schema(dsn: str, schema: str, *, suppress_error: bool = False) -> None:
    if importlib.util.find_spec("psycopg") is None:
        return
    try:
        psycopg = importlib.import_module("psycopg")
        normalized_dsn = dsn.replace("postgresql+psycopg://", "postgresql://", 1)
        with psycopg.connect(normalized_dsn) as connection, connection.cursor() as cursor:
            cursor.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    except Exception:
        if not suppress_error:
            raise
