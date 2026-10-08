"""Synchronous PostgreSQL/pgvector VectorStore adapter with isolated namespaces."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from repopilot.config import RepoPilotSettings
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

MIGRATION_PATH = Path(__file__).resolve().parents[3] / "migrations" / "0001_pgvector.sql"
ConnectFactory = Callable[..., Any]


class PostgresVectorStore:
    """Persist exact vector snapshots without exposing SQL to Retriever callers."""

    def __init__(
        self,
        settings: RepoPilotSettings,
        *,
        connect_factory: ConnectFactory | None = None,
    ) -> None:
        self._dsn = _psycopg_dsn(settings.database_url.get_secret_value())
        self._schema = settings.vector_schema
        self._table = settings.vector_table
        self._metadata_table = f"{self._table}_schema"
        self._namespace_index = f"{self._table}_namespace_idx"
        self._cosine_index = f"{self._table}_cosine_idx"
        self._dimension = settings.vector_dimension
        self._connect_timeout = settings.database_connect_timeout_seconds
        self._statement_timeout_ms = settings.database_statement_timeout_ms
        self._connect_factory = connect_factory or _default_connect
        self._initialized = False
        self._info: VectorStoreInfo | None = None

    def initialize(self) -> VectorStoreInfo:
        """Apply the idempotent migration and verify extension/schema/dimension."""

        try:
            migration = MIGRATION_PATH.read_text(encoding="utf-8").format(
                schema=_quoted_identifier(self._schema),
                table=_quoted_identifier(self._table),
                metadata_table=_quoted_identifier(self._metadata_table),
                namespace_index=_quoted_identifier(self._namespace_index),
                cosine_index=_quoted_identifier(self._cosine_index),
                dimension=str(self._dimension),
            )
        except (OSError, KeyError, ValueError):
            raise VectorStoreError(VectorStoreErrorCode.MIGRATION_ERROR) from None

        connection = self._open_connection()
        try:
            with connection, connection.cursor() as cursor:
                self._configure_cursor(cursor)
                cursor.execute(migration)
                cursor.execute(
                    (
                        f"INSERT INTO {self._metadata_relation} "
                        "(singleton, schema_version, embedding_dimension) "
                        "VALUES (TRUE, %s, %s) "
                        "ON CONFLICT (singleton) DO NOTHING"
                    ),
                    (VECTOR_STORE_SCHEMA_VERSION, self._dimension),
                )
                cursor.execute(
                    "SELECT schema_version, embedding_dimension "
                    f"FROM {self._metadata_relation} WHERE singleton = TRUE"
                )
                metadata_row = cursor.fetchone()
                cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                extension_row = cursor.fetchone()
        except Exception as error:
            code = (
                VectorStoreErrorCode.EXTENSION_UNAVAILABLE
                if _is_extension_error(error)
                else VectorStoreErrorCode.MIGRATION_ERROR
            )
            raise VectorStoreError(code) from None

        if (
            metadata_row is None
            or len(metadata_row) != 2
            or metadata_row[0] != VECTOR_STORE_SCHEMA_VERSION
            or metadata_row[1] != self._dimension
        ):
            raise VectorStoreError(VectorStoreErrorCode.SCHEMA_MISMATCH)
        if extension_row is None or not extension_row[0]:
            raise VectorStoreError(VectorStoreErrorCode.EXTENSION_UNAVAILABLE)
        self._info = VectorStoreInfo(
            backend="pgvector",
            dimension=self._dimension,
            schema_name=self._schema,
            table_name=self._table,
            extension_version=str(extension_row[0]),
        )
        self._initialized = True
        return self._info

    def upsert(
        self,
        namespace: VectorNamespace,
        records: Sequence[VectorRecord],
    ) -> VectorWriteResult:
        self._require_ready(namespace)
        batch = validate_record_batch(namespace, records)
        connection = self._open_connection()
        try:
            with connection, connection.cursor() as cursor:
                self._configure_cursor(cursor)
                self._upsert_batch(cursor, batch)
                total = self._count_records(cursor, namespace)
        except VectorStoreError:
            raise
        except Exception:
            raise VectorStoreError(VectorStoreErrorCode.QUERY_ERROR) from None
        return VectorWriteResult(
            namespace=namespace,
            written_count=len(batch),
            deleted_count=0,
            total_count=total,
        )

    def replace_namespace(
        self,
        namespace: VectorNamespace,
        records: Sequence[VectorRecord],
    ) -> VectorWriteResult:
        self._require_ready(namespace)
        batch = validate_record_batch(namespace, records)
        incoming_ids = tuple(record.chunk_id for record in batch)
        connection = self._open_connection()
        try:
            with connection, connection.cursor() as cursor:
                self._configure_cursor(cursor)
                self._upsert_batch(cursor, batch)
                if incoming_ids:
                    cursor.execute(
                        (
                            f"DELETE FROM {self._relation} "
                            f"WHERE {self._namespace_predicate} "
                            "AND NOT (chunk_id = ANY(%s))"
                        ),
                        (*_namespace_values(namespace), list(incoming_ids)),
                    )
                else:
                    cursor.execute(
                        (f"DELETE FROM {self._relation} WHERE {self._namespace_predicate}"),
                        _namespace_values(namespace),
                    )
                deleted = max(cursor.rowcount, 0)
                total = self._count_records(cursor, namespace)
        except VectorStoreError:
            raise
        except Exception:
            raise VectorStoreError(VectorStoreErrorCode.QUERY_ERROR) from None
        return VectorWriteResult(
            namespace=namespace,
            written_count=len(batch),
            deleted_count=deleted,
            total_count=total,
        )

    def list_records(
        self,
        namespace: VectorNamespace,
    ) -> tuple[VectorRecord, ...]:
        self._require_ready(namespace)
        connection = self._open_connection()
        try:
            with connection, connection.cursor() as cursor:
                self._configure_cursor(cursor)
                cursor.execute(
                    (
                        "SELECT chunk_id, content_hash, relative_path, language, "
                        "symbol_name, embedding::text, metadata "
                        f"FROM {self._relation} "
                        f"WHERE {self._namespace_predicate} "
                        "ORDER BY chunk_id"
                    ),
                    _namespace_values(namespace),
                )
                rows = cursor.fetchall()
            return tuple(_record_from_row(namespace, row) for row in rows)
        except Exception:
            raise VectorStoreError(VectorStoreErrorCode.QUERY_ERROR) from None

    def delete_namespace(self, namespace: VectorNamespace) -> VectorWriteResult:
        self._require_ready(namespace)
        connection = self._open_connection()
        try:
            with connection, connection.cursor() as cursor:
                self._configure_cursor(cursor)
                cursor.execute(
                    (f"DELETE FROM {self._relation} WHERE {self._namespace_predicate}"),
                    _namespace_values(namespace),
                )
                deleted = max(cursor.rowcount, 0)
        except Exception:
            raise VectorStoreError(VectorStoreErrorCode.QUERY_ERROR) from None
        return VectorWriteResult(
            namespace=namespace,
            written_count=0,
            deleted_count=deleted,
            total_count=0,
        )

    def search(
        self,
        namespace: VectorNamespace,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        allowed_chunk_ids: Sequence[str] | None = None,
    ) -> tuple[VectorSearchHit, ...]:
        self._require_ready(namespace)
        query_vector = validate_vector_query(
            namespace,
            query_embedding,
            top_k=top_k,
        )
        allowed = validate_chunk_allowlist(allowed_chunk_ids)
        if allowed == ():
            return ()
        distance = f"target.embedding::vector({self._dimension}) <=> query.embedding"
        allowlist_predicate = " AND target.chunk_id = ANY(%s)" if allowed is not None else ""
        parameters: tuple[object, ...] = (
            _vector_literal(query_vector),
            *_namespace_values(namespace),
            *((list(allowed),) if allowed is not None else ()),
            top_k,
        )
        connection = self._open_connection()
        try:
            with connection, connection.cursor() as cursor:
                self._configure_cursor(cursor)
                cursor.execute(
                    (
                        f"WITH query AS (SELECT %s::vector({self._dimension}) AS embedding) "
                        "SELECT target.chunk_id, target.content_hash, "
                        "target.relative_path, target.language, target.symbol_name, "
                        f"target.metadata, 1 - ({distance}) AS score "
                        f"FROM {self._relation} AS target CROSS JOIN query "
                        f"WHERE {self._namespace_predicate}{allowlist_predicate} "
                        f"ORDER BY {distance} ASC, target.chunk_id ASC LIMIT %s"
                    ),
                    parameters,
                )
                rows = cursor.fetchall()
            return tuple(_search_hit_from_row(namespace, row) for row in rows)
        except VectorStoreError:
            raise
        except Exception:
            raise VectorStoreError(VectorStoreErrorCode.QUERY_ERROR) from None

    @property
    def info(self) -> VectorStoreInfo | None:
        return self._info

    @property
    def _relation(self) -> str:
        return f"{_quoted_identifier(self._schema)}.{_quoted_identifier(self._table)}"

    @property
    def _metadata_relation(self) -> str:
        return f"{_quoted_identifier(self._schema)}.{_quoted_identifier(self._metadata_table)}"

    @property
    def _namespace_predicate(self) -> str:
        return (
            "repository_id = %s AND commit = %s AND embedding_model = %s "
            "AND embedding_version = %s AND embedding_dimension = %s"
        )

    def _open_connection(self) -> Any:
        try:
            return self._connect_factory(
                self._dsn,
                connect_timeout=self._connect_timeout,
                application_name="repopilot-vector-store",
            )
        except Exception:
            raise VectorStoreError(VectorStoreErrorCode.CONNECTION_ERROR) from None

    def _configure_cursor(self, cursor: Any) -> None:
        cursor.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (str(self._statement_timeout_ms),),
        )

    def _require_ready(self, namespace: VectorNamespace) -> None:
        if not self._initialized:
            raise VectorStoreError(VectorStoreErrorCode.NOT_INITIALIZED)
        if namespace.dimension != self._dimension:
            raise VectorStoreError(VectorStoreErrorCode.DIMENSION_MISMATCH)

    def _upsert_batch(
        self,
        cursor: Any,
        records: Sequence[VectorRecord],
    ) -> None:
        query = (
            f"INSERT INTO {self._relation} AS target ("
            "repository_id, commit, embedding_model, embedding_version, "
            "embedding_dimension, chunk_id, content_hash, relative_path, language, "
            "symbol_name, embedding, metadata"
            ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb) "
            "ON CONFLICT (repository_id, commit, embedding_model, "
            "embedding_version, chunk_id) DO UPDATE SET "
            "relative_path = EXCLUDED.relative_path, "
            "language = EXCLUDED.language, "
            "symbol_name = EXCLUDED.symbol_name, "
            "embedding = EXCLUDED.embedding, "
            "metadata = EXCLUDED.metadata, "
            "updated_at = CURRENT_TIMESTAMP "
            "WHERE target.content_hash = EXCLUDED.content_hash "
            "RETURNING chunk_id"
        )
        for record in records:
            cursor.execute(
                query,
                (
                    *_namespace_values(record.namespace),
                    record.chunk_id,
                    record.content_hash,
                    record.relative_path,
                    record.language,
                    record.symbol_name,
                    _vector_literal(record.embedding),
                    json.dumps(
                        record.metadata,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                ),
            )
            if cursor.fetchone() is None:
                raise VectorStoreError(VectorStoreErrorCode.INVALID_RECORD)

    def _count_records(self, cursor: Any, namespace: VectorNamespace) -> int:
        cursor.execute(
            (f"SELECT COUNT(*) FROM {self._relation} WHERE {self._namespace_predicate}"),
            _namespace_values(namespace),
        )
        row = cursor.fetchone()
        if row is None or len(row) != 1 or not isinstance(row[0], int):
            raise VectorStoreError(VectorStoreErrorCode.SCHEMA_MISMATCH)
        return row[0]


def _default_connect(dsn: str, **kwargs: object) -> Any:
    try:
        psycopg = importlib.import_module("psycopg")
    except ImportError:
        raise VectorStoreError(VectorStoreErrorCode.CONNECTION_ERROR) from None
    return psycopg.connect(dsn, **kwargs)


def _psycopg_dsn(value: str) -> str:
    prefix = "postgresql+psycopg://"
    return f"postgresql://{value[len(prefix) :]}" if value.startswith(prefix) else value


def _quoted_identifier(value: str) -> str:
    if (
        not value
        or len(value) > 63
        or not value[0].islower()
        or not value.replace("_", "").isalnum()
        or value.lower() != value
    ):
        raise VectorStoreError(VectorStoreErrorCode.SCHEMA_MISMATCH)
    return f'"{value}"'


def _namespace_values(
    namespace: VectorNamespace,
) -> tuple[str, str, str, str, int]:
    return (
        namespace.repository_id,
        namespace.commit,
        namespace.embedding_model,
        namespace.embedding_version,
        namespace.dimension,
    )


def _vector_literal(embedding: Sequence[float]) -> str:
    return f"[{','.join(format(value, '.9g') for value in embedding)}]"


def _parse_vector(value: object) -> tuple[float, ...]:
    if not isinstance(value, str) or not value.startswith("[") or not value.endswith("]"):
        raise VectorStoreError(VectorStoreErrorCode.SCHEMA_MISMATCH)
    body = value[1:-1]
    if not body:
        return ()
    try:
        return tuple(float(part) for part in body.split(","))
    except ValueError:
        raise VectorStoreError(VectorStoreErrorCode.SCHEMA_MISMATCH) from None


def _record_from_row(
    namespace: VectorNamespace,
    row: Sequence[object],
) -> VectorRecord:
    if len(row) != 7 or not isinstance(row[6], dict):
        raise VectorStoreError(VectorStoreErrorCode.SCHEMA_MISMATCH)
    try:
        return VectorRecord.model_validate(
            {
                "namespace": namespace,
                "chunk_id": row[0],
                "content_hash": row[1],
                "relative_path": row[2],
                "language": row[3],
                "symbol_name": row[4],
                "embedding": _parse_vector(row[5]),
                "metadata": row[6],
            }
        )
    except (TypeError, ValueError):
        raise VectorStoreError(VectorStoreErrorCode.SCHEMA_MISMATCH) from None


def _search_hit_from_row(
    namespace: VectorNamespace,
    row: Sequence[object],
) -> VectorSearchHit:
    if (
        len(row) != 7
        or not isinstance(row[5], dict)
        or isinstance(row[6], bool)
        or not isinstance(row[6], (int, float))
    ):
        raise VectorStoreError(VectorStoreErrorCode.SCHEMA_MISMATCH)
    score = float(row[6])
    if not -1.000_000_001 <= score <= 1.000_000_001:
        raise VectorStoreError(VectorStoreErrorCode.SCHEMA_MISMATCH)
    try:
        return VectorSearchHit.model_validate(
            {
                "namespace": namespace,
                "chunk_id": row[0],
                "content_hash": row[1],
                "relative_path": row[2],
                "language": row[3],
                "symbol_name": row[4],
                "metadata": row[5],
                "score": min(1.0, max(-1.0, score)),
            }
        )
    except (TypeError, ValueError):
        raise VectorStoreError(VectorStoreErrorCode.SCHEMA_MISMATCH) from None


def _is_extension_error(error: Exception) -> bool:
    return getattr(error, "sqlstate", None) in {"0A000", "42704", "58P01"}
