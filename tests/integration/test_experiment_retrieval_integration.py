"""Docker-gated proof that experiment Dense/Hybrid context uses pgvector."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import SecretStr

from repopilot.config import RepoPilotSettings
from repopilot.embeddings import LocalEmbeddingCache, embedding_provider_from_settings
from repopilot.enums import RetrievalMode
from repopilot.repair_workflow import build_repair_retrieval_context
from repopilot.schemas import CommandSpec, TaskSpec
from repopilot.storage import PostgresVectorStore
from repopilot.tools import ReadOnlyRepository

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("mode", (RetrievalMode.DENSE, RetrievalMode.HYBRID))
def test_experiment_context_routes_dense_and_hybrid_through_pgvector(
    tmp_path: Path,
    mode: RetrievalMode,
) -> None:
    dsn = os.environ.get("REPOPILOT_TEST_DATABASE_URL")
    if dsn is None:
        pytest.skip("set REPOPILOT_TEST_DATABASE_URL to run pgvector integration")
    if importlib.util.find_spec("psycopg") is None:
        pytest.skip("install the locked Psycopg dependency to run pgvector integration")

    repository_path = tmp_path / "repository"
    package = repository_path / "parser"
    package.mkdir(parents=True)
    package.joinpath("core.py").write_text(
        "def parse_config(raw: str) -> dict[str, str]:\n"
        "    return dict(line.split('=') for line in raw.splitlines())\n",
        encoding="utf-8",
    )
    package.joinpath("format.py").write_text(
        "def format_config(value: dict[str, str]) -> str:\n"
        "    return '\\n'.join(f'{key}={item}' for key, item in value.items())\n",
        encoding="utf-8",
    )
    schema = f"repopilot_exp_{uuid4().hex[:12]}"
    settings = RepoPilotSettings(
        database_url=SecretStr(dsn),
        vector_schema=schema,
        vector_table="vectors",
        retrieval_mode=mode,
        retrieval_default_top_k=5,
        dense_default_top_k=5,
        hybrid_candidate_top_k=5,
    )
    task = TaskSpec(
        task_id=f"experiment-{mode.value}",
        repository_path=str(repository_path),
        base_commit="a" * 40,
        issue_title="Blank parser input raises an exception",
        issue_body="The parse_config function should ignore blank lines.",
        test_commands=(
            CommandSpec(
                argv=("python", "-m", "unittest", "-q"),
                timeout_seconds=30,
            ),
        ),
        timeout_seconds=60,
    )
    failure: BaseException | None = None
    try:
        context = build_repair_retrieval_context(
            ReadOnlyRepository.from_settings(repository_path, settings),
            task,
            settings,
            embedding_provider=embedding_provider_from_settings(settings),
            embedding_cache=LocalEmbeddingCache(tmp_path / "embedding-cache"),
            vector_store=PostgresVectorStore(settings),
        )

        assert context.hits
        assert all(hit.dense_score is not None for hit in context.hits)
        if mode is RetrievalMode.DENSE:
            assert all(hit.sparse_score is None for hit in context.hits)
        else:
            assert all(hit.sparse_score is not None for hit in context.hits)
            assert all(hit.fused_score is not None for hit in context.hits)
    except BaseException as error:
        failure = error
        raise
    finally:
        _drop_schema(dsn, schema, suppress_error=failure is not None)


def _drop_schema(dsn: str, schema: str, *, suppress_error: bool) -> None:
    try:
        psycopg = importlib.import_module("psycopg")
        normalized_dsn = dsn.replace("postgresql+psycopg://", "postgresql://", 1)
        with psycopg.connect(normalized_dsn) as connection, connection.cursor() as cursor:
            cursor.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    except Exception:
        if not suppress_error:
            raise
