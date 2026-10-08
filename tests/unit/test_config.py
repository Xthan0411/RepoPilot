"""Tests for configuration precedence, validation, and secret redaction."""

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import SecretStr

from repopilot.config import (
    DEFAULT_JINA_CODE_MODEL,
    DEFAULT_JINA_CODE_REVISION,
    ConfigurationError,
    load_settings,
)
from repopilot.enums import (
    ConfigProfile,
    EmbeddingProviderKind,
    LogLevel,
    ModelProvider,
    ReasoningEffort,
    RetrievalMode,
    SandboxNetworkMode,
)

CONFIG_ENVIRONMENT_VARIABLES = (
    "REPOPILOT_PROFILE",
    "REPOPILOT_MODEL_PROVIDER",
    "REPOPILOT_MODEL",
    "REPOPILOT_TEMPERATURE",
    "REPOPILOT_OPENAI_REASONING_EFFORT",
    "REPOPILOT_OPENAI_MAX_OUTPUT_TOKENS",
    "REPOPILOT_OPENAI_REQUEST_TIMEOUT_SECONDS",
    "REPOPILOT_OPENAI_MAX_RETRIES",
    "REPOPILOT_OPENAI_RETRY_BACKOFF_MS",
    "REPOPILOT_OPENAI_PROMPT_MAX_CHARS",
    "REPOPILOT_OPENAI_INPUT_COST_PER_MILLION_USD",
    "REPOPILOT_OPENAI_CACHED_INPUT_COST_PER_MILLION_USD",
    "REPOPILOT_OPENAI_CACHE_WRITE_COST_PER_MILLION_USD",
    "REPOPILOT_OPENAI_OUTPUT_COST_PER_MILLION_USD",
    "REPOPILOT_MAX_STEPS",
    "REPOPILOT_MAX_REPLANS",
    "REPOPILOT_MAX_TOKENS",
    "REPOPILOT_TIMEOUT_SECONDS",
    "REPOPILOT_MAX_COST_USD",
    "REPOPILOT_REPOSITORY_MAX_FILE_BYTES",
    "REPOPILOT_REPOSITORY_MAX_READ_BYTES",
    "REPOPILOT_REPOSITORY_MAX_READ_LINES",
    "REPOPILOT_REPOSITORY_MAX_LIST_ENTRIES",
    "REPOPILOT_TOOL_MAX_OUTPUT_BYTES",
    "REPOPILOT_SEARCH_MAX_MATCHES",
    "REPOPILOT_SEARCH_MAX_QUERY_LENGTH",
    "REPOPILOT_CHUNK_MAX_LINES",
    "REPOPILOT_CHUNK_OVERLAP_LINES",
    "REPOPILOT_BM25_K1",
    "REPOPILOT_BM25_B",
    "REPOPILOT_BM25_DEFAULT_TOP_K",
    "REPOPILOT_RETRIEVAL_MODE",
    "REPOPILOT_RETRIEVAL_DEFAULT_TOP_K",
    "REPOPILOT_HYBRID_CANDIDATE_TOP_K",
    "REPOPILOT_HYBRID_RRF_K",
    "REPOPILOT_QUERY_REWRITE_ENABLED",
    "REPOPILOT_QUERY_REWRITE_MIN_HITS",
    "REPOPILOT_QUERY_REWRITE_MAX_CONTEXT_CHUNKS",
    "REPOPILOT_EMBEDDING_PROVIDER",
    "REPOPILOT_EMBEDDING_MODEL",
    "REPOPILOT_EMBEDDING_REVISION",
    "REPOPILOT_EMBEDDING_CODE_REVISION",
    "REPOPILOT_EMBEDDING_BATCH_SIZE",
    "REPOPILOT_EMBEDDING_MAX_RETRIES",
    "REPOPILOT_EMBEDDING_RETRY_BACKOFF_MS",
    "REPOPILOT_EMBEDDING_MAX_LENGTH",
    "REPOPILOT_EMBEDDING_NORMALIZE",
    "REPOPILOT_EMBEDDING_CACHE_DIR",
    "REPOPILOT_EMBEDDING_TRUST_REMOTE_CODE",
    "REPOPILOT_DENSE_DEFAULT_TOP_K",
    "REPOPILOT_VECTOR_SCHEMA",
    "REPOPILOT_VECTOR_TABLE",
    "REPOPILOT_VECTOR_DIMENSION",
    "REPOPILOT_DATABASE_CONNECT_TIMEOUT_SECONDS",
    "REPOPILOT_DATABASE_STATEMENT_TIMEOUT_MS",
    "DATABASE_URL",
    "REPOPILOT_SANDBOX_NETWORK",
    "REPOPILOT_SANDBOX_CPU_LIMIT",
    "REPOPILOT_SANDBOX_MEMORY_LIMIT_MB",
    "REPOPILOT_SANDBOX_PIDS_LIMIT",
    "REPOPILOT_SANDBOX_TMPFS_LIMIT_MB",
    "REPOPILOT_SANDBOX_COMMAND_TIMEOUT_SECONDS",
    "REPOPILOT_SANDBOX_MAX_OUTPUT_BYTES",
    "REPOPILOT_SANDBOX_MAX_ARTIFACT_BYTES",
    "REPOPILOT_PATCH_MAX_BYTES",
    "REPOPILOT_PATCH_MAX_FILES",
    "REPOPILOT_PATCH_MAX_HUNKS",
    "REPOPILOT_PATCH_MAX_CHANGED_LINES",
    "REPOPILOT_LOG_LEVEL",
    "OPENAI_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
)


@pytest.fixture(autouse=True)
def clear_config_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep local developer credentials and settings out of unit tests."""

    for variable_name in CONFIG_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable_name, raising=False)


def test_loads_default_profile() -> None:
    settings = load_settings()

    assert settings.profile is ConfigProfile.DEFAULT
    assert settings.model_provider is ModelProvider.FAKE
    assert settings.model == "fake-deterministic"
    assert settings.temperature == 0.0
    assert settings.openai_reasoning_effort is ReasoningEffort.MEDIUM
    assert settings.openai_max_output_tokens == 4_096
    assert settings.openai_request_timeout_seconds == 120
    assert settings.openai_max_retries == 2
    assert settings.openai_retry_backoff_ms == 500
    assert settings.openai_prompt_max_chars == 50_000
    assert settings.openai_input_cost_per_million_usd == Decimal("2.50")
    assert settings.openai_cached_input_cost_per_million_usd == Decimal("0.25")
    assert settings.openai_cache_write_cost_per_million_usd == Decimal("3.125")
    assert settings.openai_output_cost_per_million_usd == Decimal("15.00")
    assert settings.max_steps == 30
    assert settings.max_replans == 1
    assert settings.max_tokens == 100_000
    assert settings.timeout_seconds == 900
    assert settings.max_cost_usd == Decimal("1.00")
    assert settings.repository_max_file_bytes == 1_000_000
    assert settings.repository_max_read_bytes == 65_536
    assert settings.repository_max_read_lines == 400
    assert settings.repository_max_list_entries == 2_000
    assert settings.tool_max_output_bytes == 65_536
    assert settings.search_max_matches == 100
    assert settings.search_max_query_length == 256
    assert settings.chunk_max_lines == 200
    assert settings.chunk_overlap_lines == 20
    assert settings.bm25_k1 == 1.5
    assert settings.bm25_b == 0.75
    assert settings.bm25_default_top_k == 10
    assert settings.retrieval_mode is RetrievalMode.BM25
    assert settings.retrieval_default_top_k == 10
    assert settings.hybrid_candidate_top_k == 100
    assert settings.hybrid_rrf_k == 60
    assert settings.query_rewrite_enabled is False
    assert settings.query_rewrite_min_hits == 3
    assert settings.query_rewrite_max_context_chunks == 5
    assert settings.embedding_provider is EmbeddingProviderKind.FAKE
    assert settings.embedding_model == "fake-code-embedding"
    assert settings.embedding_revision == "v1"
    assert settings.embedding_code_revision == "v1"
    assert settings.embedding_batch_size == 32
    assert settings.embedding_max_retries == 2
    assert settings.embedding_retry_backoff_ms == 100
    assert settings.embedding_max_length == 2_048
    assert settings.embedding_normalize is True
    assert settings.embedding_cache_dir == "artifacts/embedding-cache"
    assert settings.embedding_trust_remote_code is False
    assert settings.dense_default_top_k == 10
    assert settings.vector_schema == "repopilot"
    assert settings.vector_table == "vector_embeddings"
    assert settings.vector_dimension == 768
    assert settings.database_connect_timeout_seconds == 5
    assert settings.database_statement_timeout_ms == 5_000
    assert settings.sandbox_network is SandboxNetworkMode.NONE
    assert settings.sandbox_cpu_limit == 1.0
    assert settings.sandbox_memory_limit_mb == 1_024
    assert settings.sandbox_pids_limit == 256
    assert settings.sandbox_tmpfs_limit_mb == 256
    assert settings.sandbox_command_timeout_seconds == 300
    assert settings.sandbox_max_output_bytes == 1_000_000
    assert settings.sandbox_max_artifact_bytes == 20_000_000
    assert settings.patch_max_bytes == 262_144
    assert settings.patch_max_files == 20
    assert settings.patch_max_hunks == 100
    assert settings.patch_max_changed_lines == 2_000
    assert settings.log_level is LogLevel.INFO
    assert settings.openai_api_key is None
    assert settings.langsmith_api_key is None
    assert settings.langsmith_tracing is False


def test_loads_eval_profile() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "eval.yaml"

    settings = load_settings(config_path)

    assert settings.profile is ConfigProfile.EVAL
    assert settings.model_provider is ModelProvider.FAKE
    assert settings.sandbox_network is SandboxNetworkMode.NONE


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPOPILOT_MODEL", "environment-model")
    monkeypatch.setenv("REPOPILOT_OPENAI_REASONING_EFFORT", "high")
    monkeypatch.setenv("REPOPILOT_OPENAI_MAX_OUTPUT_TOKENS", "8192")
    monkeypatch.setenv("REPOPILOT_OPENAI_REQUEST_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("REPOPILOT_OPENAI_MAX_RETRIES", "4")
    monkeypatch.setenv("REPOPILOT_OPENAI_RETRY_BACKOFF_MS", "750")
    monkeypatch.setenv("REPOPILOT_OPENAI_PROMPT_MAX_CHARS", "60000")
    monkeypatch.setenv("REPOPILOT_OPENAI_INPUT_COST_PER_MILLION_USD", "3.00")
    monkeypatch.setenv("REPOPILOT_OPENAI_CACHED_INPUT_COST_PER_MILLION_USD", "0.30")
    monkeypatch.setenv("REPOPILOT_OPENAI_CACHE_WRITE_COST_PER_MILLION_USD", "3.75")
    monkeypatch.setenv("REPOPILOT_OPENAI_OUTPUT_COST_PER_MILLION_USD", "18.00")
    monkeypatch.setenv("REPOPILOT_MAX_STEPS", "17")
    monkeypatch.setenv("REPOPILOT_MAX_REPLANS", "3")
    monkeypatch.setenv("REPOPILOT_MAX_TOKENS", "4321")
    monkeypatch.setenv("REPOPILOT_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("REPOPILOT_MAX_COST_USD", "0.25")
    monkeypatch.setenv("REPOPILOT_REPOSITORY_MAX_FILE_BYTES", "200000")
    monkeypatch.setenv("REPOPILOT_REPOSITORY_MAX_READ_BYTES", "32000")
    monkeypatch.setenv("REPOPILOT_REPOSITORY_MAX_READ_LINES", "250")
    monkeypatch.setenv("REPOPILOT_REPOSITORY_MAX_LIST_ENTRIES", "900")
    monkeypatch.setenv("REPOPILOT_TOOL_MAX_OUTPUT_BYTES", "48000")
    monkeypatch.setenv("REPOPILOT_SEARCH_MAX_MATCHES", "75")
    monkeypatch.setenv("REPOPILOT_SEARCH_MAX_QUERY_LENGTH", "180")
    monkeypatch.setenv("REPOPILOT_CHUNK_MAX_LINES", "120")
    monkeypatch.setenv("REPOPILOT_CHUNK_OVERLAP_LINES", "12")
    monkeypatch.setenv("REPOPILOT_BM25_K1", "1.2")
    monkeypatch.setenv("REPOPILOT_BM25_B", "0.6")
    monkeypatch.setenv("REPOPILOT_BM25_DEFAULT_TOP_K", "25")
    monkeypatch.setenv("REPOPILOT_RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("REPOPILOT_RETRIEVAL_DEFAULT_TOP_K", "20")
    monkeypatch.setenv("REPOPILOT_HYBRID_CANDIDATE_TOP_K", "80")
    monkeypatch.setenv("REPOPILOT_HYBRID_RRF_K", "45")
    monkeypatch.setenv("REPOPILOT_QUERY_REWRITE_ENABLED", "true")
    monkeypatch.setenv("REPOPILOT_QUERY_REWRITE_MIN_HITS", "4")
    monkeypatch.setenv("REPOPILOT_QUERY_REWRITE_MAX_CONTEXT_CHUNKS", "2")
    monkeypatch.setenv("REPOPILOT_EMBEDDING_MODEL", "fake-environment")
    monkeypatch.setenv("REPOPILOT_EMBEDDING_REVISION", "v2")
    monkeypatch.setenv("REPOPILOT_EMBEDDING_CODE_REVISION", "v3")
    monkeypatch.setenv("REPOPILOT_EMBEDDING_BATCH_SIZE", "16")
    monkeypatch.setenv("REPOPILOT_EMBEDDING_MAX_RETRIES", "3")
    monkeypatch.setenv("REPOPILOT_EMBEDDING_RETRY_BACKOFF_MS", "250")
    monkeypatch.setenv("REPOPILOT_EMBEDDING_MAX_LENGTH", "1024")
    monkeypatch.setenv("REPOPILOT_EMBEDDING_NORMALIZE", "false")
    monkeypatch.setenv("REPOPILOT_EMBEDDING_CACHE_DIR", "artifacts/custom-cache")
    monkeypatch.setenv("REPOPILOT_DENSE_DEFAULT_TOP_K", "30")
    monkeypatch.setenv("REPOPILOT_VECTOR_SCHEMA", "custom_schema")
    monkeypatch.setenv("REPOPILOT_VECTOR_TABLE", "custom_vectors")
    monkeypatch.setenv("REPOPILOT_VECTOR_DIMENSION", "384")
    monkeypatch.setenv("REPOPILOT_DATABASE_CONNECT_TIMEOUT_SECONDS", "8")
    monkeypatch.setenv("REPOPILOT_DATABASE_STATEMENT_TIMEOUT_MS", "9000")
    monkeypatch.setenv("REPOPILOT_SANDBOX_NETWORK", "setup_only")
    monkeypatch.setenv("REPOPILOT_SANDBOX_CPU_LIMIT", "2.5")
    monkeypatch.setenv("REPOPILOT_SANDBOX_MEMORY_LIMIT_MB", "2048")
    monkeypatch.setenv("REPOPILOT_SANDBOX_PIDS_LIMIT", "128")
    monkeypatch.setenv("REPOPILOT_SANDBOX_TMPFS_LIMIT_MB", "512")
    monkeypatch.setenv("REPOPILOT_SANDBOX_COMMAND_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("REPOPILOT_SANDBOX_MAX_OUTPUT_BYTES", "2000000")
    monkeypatch.setenv("REPOPILOT_PATCH_MAX_BYTES", "131072")
    monkeypatch.setenv("REPOPILOT_PATCH_MAX_FILES", "12")
    monkeypatch.setenv("REPOPILOT_PATCH_MAX_HUNKS", "40")
    monkeypatch.setenv("REPOPILOT_PATCH_MAX_CHANGED_LINES", "800")
    monkeypatch.setenv("REPOPILOT_SANDBOX_MAX_ARTIFACT_BYTES", "30000000")
    monkeypatch.setenv("REPOPILOT_LOG_LEVEL", "debug")

    settings = load_settings()

    assert settings.model == "environment-model"
    assert settings.openai_reasoning_effort is ReasoningEffort.HIGH
    assert settings.openai_max_output_tokens == 8_192
    assert settings.openai_request_timeout_seconds == 90
    assert settings.openai_max_retries == 4
    assert settings.openai_retry_backoff_ms == 750
    assert settings.openai_prompt_max_chars == 60_000
    assert settings.openai_input_cost_per_million_usd == Decimal("3.00")
    assert settings.openai_cached_input_cost_per_million_usd == Decimal("0.30")
    assert settings.openai_cache_write_cost_per_million_usd == Decimal("3.75")
    assert settings.openai_output_cost_per_million_usd == Decimal("18.00")
    assert settings.max_steps == 17
    assert settings.max_replans == 3
    assert settings.max_tokens == 4321
    assert settings.timeout_seconds == 45
    assert settings.max_cost_usd == Decimal("0.25")
    assert settings.repository_max_file_bytes == 200_000
    assert settings.repository_max_read_bytes == 32_000
    assert settings.repository_max_read_lines == 250
    assert settings.repository_max_list_entries == 900
    assert settings.tool_max_output_bytes == 48_000
    assert settings.search_max_matches == 75
    assert settings.search_max_query_length == 180
    assert settings.chunk_max_lines == 120
    assert settings.chunk_overlap_lines == 12
    assert settings.bm25_k1 == 1.2
    assert settings.bm25_b == 0.6
    assert settings.bm25_default_top_k == 25
    assert settings.retrieval_mode is RetrievalMode.HYBRID
    assert settings.retrieval_default_top_k == 20
    assert settings.hybrid_candidate_top_k == 80
    assert settings.hybrid_rrf_k == 45
    assert settings.query_rewrite_enabled is True
    assert settings.query_rewrite_min_hits == 4
    assert settings.query_rewrite_max_context_chunks == 2
    assert settings.embedding_model == "fake-environment"
    assert settings.embedding_revision == "v2"
    assert settings.embedding_code_revision == "v3"
    assert settings.embedding_batch_size == 16
    assert settings.embedding_max_retries == 3
    assert settings.embedding_retry_backoff_ms == 250
    assert settings.embedding_max_length == 1_024
    assert settings.embedding_normalize is False
    assert settings.embedding_cache_dir == "artifacts/custom-cache"
    assert settings.dense_default_top_k == 30
    assert settings.vector_schema == "custom_schema"
    assert settings.vector_table == "custom_vectors"
    assert settings.vector_dimension == 384
    assert settings.database_connect_timeout_seconds == 8
    assert settings.database_statement_timeout_ms == 9_000
    assert settings.sandbox_network is SandboxNetworkMode.SETUP_ONLY
    assert settings.sandbox_cpu_limit == 2.5
    assert settings.sandbox_memory_limit_mb == 2_048
    assert settings.sandbox_pids_limit == 128
    assert settings.sandbox_tmpfs_limit_mb == 512
    assert settings.sandbox_command_timeout_seconds == 600
    assert settings.sandbox_max_output_bytes == 2_000_000
    assert settings.sandbox_max_artifact_bytes == 30_000_000
    assert settings.patch_max_bytes == 131_072
    assert settings.patch_max_files == 12
    assert settings.patch_max_hunks == 40
    assert settings.patch_max_changed_lines == 800
    assert settings.log_level is LogLevel.DEBUG


def test_explicit_overrides_win_over_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPOPILOT_MAX_STEPS", "17")

    settings = load_settings(overrides={"max_steps": 9})

    assert settings.max_steps == 9


def test_openai_settings_and_snapshot_redact_secrets() -> None:
    openai_secret = "openai-secret-value"
    langsmith_secret = "langsmith-secret-value"
    database_secret = "database-secret-value"

    settings = load_settings(
        overrides={
            "model_provider": "openai",
            "model": "gpt-test-snapshot",
            "openai_api_key": openai_secret,
            "langsmith_api_key": langsmith_secret,
            "langsmith_tracing": True,
            "database_url": (f"postgresql://repopilot:{database_secret}@localhost:5432/repopilot"),
        }
    )
    rendered = f"{settings!r} {settings.redacted_snapshot()}"

    assert isinstance(settings.openai_api_key, SecretStr)
    assert openai_secret not in rendered
    assert langsmith_secret not in rendered
    assert database_secret not in rendered
    assert settings.redacted_snapshot()["openai_api_key"] == "<redacted>"
    assert settings.redacted_snapshot()["langsmith_api_key"] == "<redacted>"
    assert settings.redacted_snapshot()["database_url"] == "<redacted>"


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("max_steps", 0),
        ("max_replans", -1),
        ("max_replans", 11),
        ("max_tokens", -1),
        ("timeout_seconds", 0),
        ("max_cost_usd", "0"),
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("openai_max_output_tokens", 0),
        ("openai_max_output_tokens", 128_001),
        ("openai_max_output_tokens", True),
        ("openai_request_timeout_seconds", 0),
        ("openai_request_timeout_seconds", 601),
        ("openai_request_timeout_seconds", False),
        ("openai_max_retries", -1),
        ("openai_max_retries", 6),
        ("openai_max_retries", True),
        ("openai_retry_backoff_ms", -1),
        ("openai_retry_backoff_ms", 10_001),
        ("openai_retry_backoff_ms", False),
        ("openai_prompt_max_chars", 4_095),
        ("openai_prompt_max_chars", 200_001),
        ("openai_prompt_max_chars", True),
        ("openai_input_cost_per_million_usd", -1),
        ("openai_input_cost_per_million_usd", 1_001),
        ("openai_input_cost_per_million_usd", False),
        ("openai_cached_input_cost_per_million_usd", -1),
        ("openai_cached_input_cost_per_million_usd", 1_001),
        ("openai_cached_input_cost_per_million_usd", True),
        ("openai_cache_write_cost_per_million_usd", -1),
        ("openai_cache_write_cost_per_million_usd", 1_001),
        ("openai_cache_write_cost_per_million_usd", False),
        ("openai_output_cost_per_million_usd", -1),
        ("openai_output_cost_per_million_usd", 1_001),
        ("openai_output_cost_per_million_usd", True),
        ("repository_max_file_bytes", 10_000_001),
        ("repository_max_file_bytes", 0),
        ("repository_max_read_bytes", 1_000_001),
        ("repository_max_read_bytes", -1),
        ("repository_max_read_lines", 5_001),
        ("repository_max_read_lines", 0),
        ("repository_max_list_entries", 20_001),
        ("repository_max_list_entries", 0),
        ("tool_max_output_bytes", 1_000_001),
        ("tool_max_output_bytes", 0),
        ("search_max_matches", 1_001),
        ("search_max_matches", 0),
        ("search_max_query_length", 2_001),
        ("search_max_query_length", 0),
        ("chunk_max_lines", 2_001),
        ("chunk_max_lines", 0),
        ("chunk_overlap_lines", 501),
        ("chunk_overlap_lines", -1),
        ("chunk_max_lines", True),
        ("chunk_overlap_lines", False),
        ("bm25_k1", 0),
        ("bm25_k1", 5.1),
        ("bm25_k1", float("nan")),
        ("bm25_k1", float("inf")),
        ("bm25_k1", True),
        ("bm25_b", -0.1),
        ("bm25_b", 1.1),
        ("bm25_b", float("nan")),
        ("bm25_b", float("inf")),
        ("bm25_b", False),
        ("bm25_default_top_k", 0),
        ("bm25_default_top_k", 101),
        ("bm25_default_top_k", True),
        ("retrieval_default_top_k", 0),
        ("retrieval_default_top_k", 101),
        ("retrieval_default_top_k", True),
        ("hybrid_candidate_top_k", 0),
        ("hybrid_candidate_top_k", 101),
        ("hybrid_candidate_top_k", False),
        ("hybrid_rrf_k", 0),
        ("hybrid_rrf_k", 1_001),
        ("hybrid_rrf_k", True),
        ("query_rewrite_min_hits", 0),
        ("query_rewrite_min_hits", 101),
        ("query_rewrite_min_hits", True),
        ("query_rewrite_max_context_chunks", -1),
        ("query_rewrite_max_context_chunks", 11),
        ("query_rewrite_max_context_chunks", False),
        ("embedding_batch_size", 0),
        ("embedding_batch_size", 257),
        ("embedding_batch_size", True),
        ("embedding_max_retries", -1),
        ("embedding_max_retries", 6),
        ("embedding_max_retries", False),
        ("embedding_retry_backoff_ms", -1),
        ("embedding_retry_backoff_ms", 10_001),
        ("embedding_retry_backoff_ms", True),
        ("embedding_max_length", 0),
        ("embedding_max_length", 8_193),
        ("embedding_max_length", False),
        ("dense_default_top_k", 0),
        ("dense_default_top_k", 101),
        ("dense_default_top_k", True),
        ("vector_dimension", 0),
        ("vector_dimension", 2_001),
        ("vector_dimension", True),
        ("database_connect_timeout_seconds", 0),
        ("database_connect_timeout_seconds", 61),
        ("database_connect_timeout_seconds", True),
        ("database_statement_timeout_ms", 99),
        ("database_statement_timeout_ms", 120_001),
        ("database_statement_timeout_ms", False),
        ("sandbox_cpu_limit", 0),
        ("sandbox_cpu_limit", 8.1),
        ("sandbox_cpu_limit", float("nan")),
        ("sandbox_cpu_limit", True),
        ("sandbox_memory_limit_mb", 127),
        ("sandbox_memory_limit_mb", 8_193),
        ("sandbox_memory_limit_mb", False),
        ("sandbox_pids_limit", 15),
        ("sandbox_pids_limit", 1_025),
        ("sandbox_pids_limit", True),
        ("sandbox_tmpfs_limit_mb", 15),
        ("sandbox_tmpfs_limit_mb", 2_049),
        ("sandbox_tmpfs_limit_mb", False),
        ("sandbox_command_timeout_seconds", 0),
        ("sandbox_command_timeout_seconds", 3_601),
        ("sandbox_command_timeout_seconds", True),
        ("sandbox_max_output_bytes", 0),
        ("sandbox_max_output_bytes", 10_000_001),
        ("sandbox_max_output_bytes", False),
        ("sandbox_max_artifact_bytes", 0),
        ("sandbox_max_artifact_bytes", 100_000_001),
        ("sandbox_max_artifact_bytes", True),
        ("patch_max_bytes", 0),
        ("patch_max_bytes", 1_000_001),
        ("patch_max_bytes", True),
        ("patch_max_files", 0),
        ("patch_max_files", 101),
        ("patch_max_files", False),
        ("patch_max_hunks", 0),
        ("patch_max_hunks", 1_001),
        ("patch_max_hunks", True),
        ("patch_max_changed_lines", 0),
        ("patch_max_changed_lines", 10_001),
        ("patch_max_changed_lines", False),
    ],
)
def test_rejects_invalid_numeric_boundaries(field_name: str, invalid_value: object) -> None:
    with pytest.raises(ConfigurationError, match=field_name):
        load_settings(overrides={field_name: invalid_value})


@pytest.mark.parametrize(
    ("environment_name", "invalid_value"),
    [
        ("REPOPILOT_PROFILE", "production"),
        ("REPOPILOT_MODEL_PROVIDER", "unknown-provider"),
        ("REPOPILOT_OPENAI_REASONING_EFFORT", "extreme"),
        ("REPOPILOT_EMBEDDING_PROVIDER", "unknown-provider"),
        ("REPOPILOT_RETRIEVAL_MODE", "unknown-mode"),
        ("REPOPILOT_SANDBOX_NETWORK", "host"),
        ("REPOPILOT_LOG_LEVEL", "verbose"),
    ],
)
def test_rejects_unknown_enum_environment_values(
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    invalid_value: str,
) -> None:
    monkeypatch.setenv(environment_name, invalid_value)

    with pytest.raises(ConfigurationError, match=environment_name):
        load_settings()


def test_rejects_blank_model_name() -> None:
    with pytest.raises(ConfigurationError, match="model must not be empty"):
        load_settings(overrides={"model": "  "})


def test_rejects_hybrid_candidate_depth_below_default_output_depth() -> None:
    with pytest.raises(ConfigurationError, match="hybrid_candidate_top_k"):
        load_settings(
            overrides={
                "retrieval_default_top_k": 20,
                "hybrid_candidate_top_k": 10,
            }
        )


def test_rejects_patch_limit_larger_than_artifact_limit() -> None:
    with pytest.raises(ConfigurationError, match="patch_max_bytes"):
        load_settings(
            overrides={
                "patch_max_bytes": 1_000,
                "sandbox_max_artifact_bytes": 999,
                "sandbox_max_output_bytes": 999,
            }
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("embedding_model", ""),
        ("embedding_model", "/absolute"),
        ("embedding_model", "bad model"),
        ("embedding_model", "owner//model"),
        ("embedding_revision", ""),
        ("embedding_revision", "bad revision"),
        ("embedding_revision", "owner/revision"),
        ("embedding_code_revision", ""),
        ("embedding_code_revision", "bad revision"),
        ("embedding_code_revision", "owner/revision"),
        ("embedding_cache_dir", "/absolute/cache"),
        ("embedding_cache_dir", "../cache"),
        ("embedding_cache_dir", "artifacts/../cache"),
        ("embedding_cache_dir", "artifacts\\cache"),
    ],
)
def test_rejects_invalid_embedding_identifiers_and_cache_paths(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ConfigurationError, match=field_name):
        load_settings(overrides={field_name: invalid_value})


def test_accepts_explicit_revision_pinned_jina_configuration() -> None:
    settings = load_settings(
        overrides={
            "embedding_provider": "jina_local",
            "embedding_model": DEFAULT_JINA_CODE_MODEL,
            "embedding_revision": "516f4baf13dec4ddddda8631e019b5737c8bc250",
            "embedding_code_revision": DEFAULT_JINA_CODE_REVISION,
            "embedding_trust_remote_code": True,
        }
    )

    assert settings.embedding_provider is EmbeddingProviderKind.JINA_LOCAL
    assert settings.vector_dimension == 768


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {
                "embedding_provider": "jina_local",
                "embedding_model": DEFAULT_JINA_CODE_MODEL,
                "embedding_revision": "516f4baf13dec4ddddda8631e019b5737c8bc250",
                "embedding_code_revision": DEFAULT_JINA_CODE_REVISION,
            },
            "trust_remote_code",
        ),
        (
            {
                "embedding_provider": "jina_local",
                "embedding_model": DEFAULT_JINA_CODE_MODEL,
                "embedding_revision": "main",
                "embedding_code_revision": DEFAULT_JINA_CODE_REVISION,
                "embedding_trust_remote_code": True,
            },
            "immutable",
        ),
        (
            {
                "embedding_provider": "jina_local",
                "embedding_model": DEFAULT_JINA_CODE_MODEL,
                "embedding_revision": "516f4baf13dec4ddddda8631e019b5737c8bc250",
                "embedding_code_revision": DEFAULT_JINA_CODE_REVISION,
                "embedding_trust_remote_code": True,
                "vector_dimension": 384,
            },
            "vector_dimension",
        ),
        (
            {
                "embedding_provider": "jina_local",
                "embedding_model": DEFAULT_JINA_CODE_MODEL,
                "embedding_revision": "516f4baf13dec4ddddda8631e019b5737c8bc250",
                "embedding_code_revision": "main",
                "embedding_trust_remote_code": True,
            },
            "code revision",
        ),
        (
            {"embedding_trust_remote_code": True},
            "only valid",
        ),
    ],
)
def test_rejects_unsafe_or_incompatible_jina_configuration(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_settings(overrides=override)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("vector_schema", "RepoPilot"),
        ("vector_schema", "bad-schema"),
        ("vector_schema", "1schema"),
        ("vector_schema", "schema.name"),
        ("vector_table", "VectorTable"),
        ("vector_table", "bad table"),
        ("vector_table", "table;drop"),
        ("vector_table", "a" * 49),
    ],
)
def test_rejects_unsafe_vector_database_identifiers(
    field_name: str,
    invalid_value: str,
) -> None:
    with pytest.raises(ConfigurationError, match=field_name):
        load_settings(overrides={field_name: invalid_value})


def test_rejects_repository_read_limit_larger_than_file_limit() -> None:
    with pytest.raises(ConfigurationError, match="must not exceed"):
        load_settings(
            overrides={
                "repository_max_file_bytes": 100,
                "repository_max_read_bytes": 101,
            }
        )


def test_rejects_chunk_overlap_not_smaller_than_window() -> None:
    with pytest.raises(ConfigurationError, match="must be less"):
        load_settings(
            overrides={
                "chunk_max_lines": 20,
                "chunk_overlap_lines": 20,
            }
        )


def test_rejects_sandbox_tmpfs_or_output_above_parent_limit() -> None:
    with pytest.raises(ConfigurationError, match="sandbox_tmpfs_limit_mb"):
        load_settings(
            overrides={
                "sandbox_memory_limit_mb": 512,
                "sandbox_tmpfs_limit_mb": 513,
            }
        )
    with pytest.raises(ConfigurationError, match="sandbox_max_output_bytes"):
        load_settings(
            overrides={
                "sandbox_max_output_bytes": 2_000_000,
                "sandbox_max_artifact_bytes": 1_000_000,
            }
        )


def test_rejects_openai_provider_without_key() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        load_settings(overrides={"model_provider": "openai", "model": "gpt-test-snapshot"})


def test_rejects_openai_provider_without_explicit_model() -> None:
    with pytest.raises(ConfigurationError, match="explicit model"):
        load_settings(overrides={"model_provider": "openai", "openai_api_key": "secret"})


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ("claude-example", "canonical gpt"),
        ("gpt-5.6", "pinned model ID"),
        ("gpt-5-latest", "pinned model ID"),
        ("gpt-custom-latest", "pinned model ID"),
    ],
)
def test_rejects_noncanonical_or_mutable_openai_models(
    model: str,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_settings(
            overrides={
                "model_provider": "openai",
                "model": model,
                "openai_api_key": "secret",
            }
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "openai_input_cost_per_million_usd",
        "openai_output_cost_per_million_usd",
    ],
)
def test_openai_primary_token_prices_must_be_positive(field_name: str) -> None:
    with pytest.raises(ConfigurationError, match="pricing must be positive"):
        load_settings(
            overrides={
                "model_provider": "openai",
                "model": "gpt-test-snapshot",
                "openai_api_key": "secret",
                field_name: 0,
            }
        )


def test_rejects_tracing_without_key() -> None:
    with pytest.raises(ConfigurationError, match="LANGSMITH_API_KEY"):
        load_settings(overrides={"langsmith_tracing": True})


def test_blank_secret_is_treated_as_absent() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        load_settings(
            overrides={
                "model_provider": "openai",
                "model": "gpt-test-snapshot",
                "openai_api_key": "  ",
            }
        )


def test_invalid_database_url_does_not_leak_password() -> None:
    secret = "do-not-leak-this-password"

    with pytest.raises(ConfigurationError) as captured_error:
        load_settings(overrides={"database_url": f"mysql://user:{secret}@localhost/database"})

    assert secret not in str(captured_error.value)
    assert "PostgreSQL scheme" in str(captured_error.value)


def test_invalid_environment_value_does_not_echo_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_value = "not-an-integer-secret-shaped-value"
    monkeypatch.setenv("REPOPILOT_MAX_STEPS", invalid_value)

    with pytest.raises(ConfigurationError) as captured_error:
        load_settings()

    assert invalid_value not in str(captured_error.value)
    assert "REPOPILOT_MAX_STEPS" in str(captured_error.value)


def test_invalid_repository_limit_environment_value_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_value = "not-a-repository-limit-secret-shaped-value"
    monkeypatch.setenv("REPOPILOT_REPOSITORY_MAX_READ_LINES", invalid_value)

    with pytest.raises(ConfigurationError) as captured_error:
        load_settings()

    assert invalid_value not in str(captured_error.value)
    assert "REPOPILOT_REPOSITORY_MAX_READ_LINES" in str(captured_error.value)


def test_rejects_unknown_yaml_field(tmp_path: Path) -> None:
    config_path = tmp_path / "unknown-field.yaml"
    config_path.write_text("profile: test\nunknown_setting: true\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="unknown_setting"):
        load_settings(config_path)


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    [
        ("malformed.yaml", "profile: [\n", "not valid YAML"),
        ("sequence.yaml", "- default\n- eval\n", "root must be a mapping"),
        ("empty.yaml", "", "root must be a mapping"),
        ("non-string-key.yaml", "1: value\n", "keys must be strings"),
    ],
)
def test_rejects_invalid_yaml_root(
    tmp_path: Path,
    filename: str,
    contents: str,
    message: str,
) -> None:
    config_path = tmp_path / filename
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        load_settings(config_path)


def test_rejects_missing_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigurationError, match="not readable"):
        load_settings(config_path)
