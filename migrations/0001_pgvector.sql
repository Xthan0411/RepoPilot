CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.{metadata_table} (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    schema_version INTEGER NOT NULL,
    embedding_dimension INTEGER NOT NULL UNIQUE CHECK (
        embedding_dimension BETWEEN 1 AND 2000
    )
);

CREATE TABLE IF NOT EXISTS {schema}.{table} (
    repository_id VARCHAR(128) NOT NULL,
    commit CHAR(40) NOT NULL,
    embedding_model VARCHAR(128) NOT NULL,
    embedding_version VARCHAR(128) NOT NULL,
    embedding_dimension INTEGER NOT NULL REFERENCES {schema}.{metadata_table} (
        embedding_dimension
    ),
    chunk_id VARCHAR(128) NOT NULL,
    content_hash CHAR(64) NOT NULL,
    relative_path TEXT NOT NULL,
    language VARCHAR(128) NOT NULL,
    symbol_name VARCHAR(512),
    embedding vector NOT NULL,
    metadata JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (
        repository_id,
        commit,
        embedding_model,
        embedding_version,
        chunk_id
    ),
    CHECK (vector_dims(embedding) = embedding_dimension)
);

CREATE INDEX IF NOT EXISTS {namespace_index}
ON {schema}.{table} (
    repository_id,
    commit,
    embedding_model,
    embedding_version
);

CREATE INDEX IF NOT EXISTS {cosine_index}
ON {schema}.{table}
USING hnsw ((embedding::vector({dimension})) vector_cosine_ops);
