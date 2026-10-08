"""Validated RepoPilot configuration with explicit source precedence."""

import re
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Self, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from repopilot.enums import (
    ConfigProfile,
    EmbeddingProviderKind,
    LogLevel,
    ModelProvider,
    ReasoningEffort,
    RetrievalMode,
    SandboxNetworkMode,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
EVAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval.yaml"
DEFAULT_DATABASE_URL = "postgresql://repopilot:repopilot@localhost:55433/repopilot"
DEFAULT_JINA_CODE_MODEL = "jinaai/jina-embeddings-v2-base-code"
DEFAULT_JINA_CODE_REVISION = "3baf9e3ac750e76e8edd3019170176884695fb94"


class ConfigurationError(ValueError):
    """A safe configuration error that omits raw input and secrets."""


class _EnvironmentSettings(BaseSettings):
    """Optional environment-only values used by :func:`load_settings`."""

    profile: ConfigProfile | None = Field(default=None, validation_alias="REPOPILOT_PROFILE")
    model_provider: ModelProvider | None = Field(
        default=None,
        validation_alias="REPOPILOT_MODEL_PROVIDER",
    )
    model: str | None = Field(default=None, validation_alias="REPOPILOT_MODEL")
    temperature: float | None = Field(
        default=None,
        validation_alias="REPOPILOT_TEMPERATURE",
    )
    openai_reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        validation_alias="REPOPILOT_OPENAI_REASONING_EFFORT",
    )
    openai_max_output_tokens: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_OPENAI_MAX_OUTPUT_TOKENS",
    )
    openai_request_timeout_seconds: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_OPENAI_REQUEST_TIMEOUT_SECONDS",
    )
    openai_max_retries: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_OPENAI_MAX_RETRIES",
    )
    openai_retry_backoff_ms: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_OPENAI_RETRY_BACKOFF_MS",
    )
    openai_prompt_max_chars: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_OPENAI_PROMPT_MAX_CHARS",
    )
    openai_input_cost_per_million_usd: Decimal | None = Field(
        default=None,
        validation_alias="REPOPILOT_OPENAI_INPUT_COST_PER_MILLION_USD",
    )
    openai_cached_input_cost_per_million_usd: Decimal | None = Field(
        default=None,
        validation_alias="REPOPILOT_OPENAI_CACHED_INPUT_COST_PER_MILLION_USD",
    )
    openai_cache_write_cost_per_million_usd: Decimal | None = Field(
        default=None,
        validation_alias="REPOPILOT_OPENAI_CACHE_WRITE_COST_PER_MILLION_USD",
    )
    openai_output_cost_per_million_usd: Decimal | None = Field(
        default=None,
        validation_alias="REPOPILOT_OPENAI_OUTPUT_COST_PER_MILLION_USD",
    )
    max_steps: int | None = Field(default=None, validation_alias="REPOPILOT_MAX_STEPS")
    max_replans: int | None = Field(default=None, validation_alias="REPOPILOT_MAX_REPLANS")
    max_tokens: int | None = Field(default=None, validation_alias="REPOPILOT_MAX_TOKENS")
    timeout_seconds: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_TIMEOUT_SECONDS",
    )
    max_cost_usd: Decimal | None = Field(
        default=None,
        validation_alias="REPOPILOT_MAX_COST_USD",
    )
    repository_max_file_bytes: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_REPOSITORY_MAX_FILE_BYTES",
    )
    repository_max_read_bytes: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_REPOSITORY_MAX_READ_BYTES",
    )
    repository_max_read_lines: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_REPOSITORY_MAX_READ_LINES",
    )
    repository_max_list_entries: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_REPOSITORY_MAX_LIST_ENTRIES",
    )
    tool_max_output_bytes: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_TOOL_MAX_OUTPUT_BYTES",
    )
    search_max_matches: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_SEARCH_MAX_MATCHES",
    )
    search_max_query_length: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_SEARCH_MAX_QUERY_LENGTH",
    )
    chunk_max_lines: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_CHUNK_MAX_LINES",
    )
    chunk_overlap_lines: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_CHUNK_OVERLAP_LINES",
    )
    bm25_k1: float | None = Field(
        default=None,
        validation_alias="REPOPILOT_BM25_K1",
    )
    bm25_b: float | None = Field(
        default=None,
        validation_alias="REPOPILOT_BM25_B",
    )
    bm25_default_top_k: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_BM25_DEFAULT_TOP_K",
    )
    retrieval_mode: RetrievalMode | None = Field(
        default=None,
        validation_alias="REPOPILOT_RETRIEVAL_MODE",
    )
    retrieval_default_top_k: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_RETRIEVAL_DEFAULT_TOP_K",
    )
    hybrid_candidate_top_k: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_HYBRID_CANDIDATE_TOP_K",
    )
    hybrid_rrf_k: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_HYBRID_RRF_K",
    )
    query_rewrite_enabled: bool | None = Field(
        default=None,
        validation_alias="REPOPILOT_QUERY_REWRITE_ENABLED",
    )
    query_rewrite_min_hits: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_QUERY_REWRITE_MIN_HITS",
    )
    query_rewrite_max_context_chunks: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_QUERY_REWRITE_MAX_CONTEXT_CHUNKS",
    )
    embedding_provider: EmbeddingProviderKind | None = Field(
        default=None,
        validation_alias="REPOPILOT_EMBEDDING_PROVIDER",
    )
    embedding_model: str | None = Field(
        default=None,
        validation_alias="REPOPILOT_EMBEDDING_MODEL",
    )
    embedding_revision: str | None = Field(
        default=None,
        validation_alias="REPOPILOT_EMBEDDING_REVISION",
    )
    embedding_code_revision: str | None = Field(
        default=None,
        validation_alias="REPOPILOT_EMBEDDING_CODE_REVISION",
    )
    embedding_batch_size: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_EMBEDDING_BATCH_SIZE",
    )
    embedding_max_retries: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_EMBEDDING_MAX_RETRIES",
    )
    embedding_retry_backoff_ms: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_EMBEDDING_RETRY_BACKOFF_MS",
    )
    embedding_max_length: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_EMBEDDING_MAX_LENGTH",
    )
    embedding_normalize: bool | None = Field(
        default=None,
        validation_alias="REPOPILOT_EMBEDDING_NORMALIZE",
    )
    embedding_cache_dir: str | None = Field(
        default=None,
        validation_alias="REPOPILOT_EMBEDDING_CACHE_DIR",
    )
    embedding_trust_remote_code: bool | None = Field(
        default=None,
        validation_alias="REPOPILOT_EMBEDDING_TRUST_REMOTE_CODE",
    )
    dense_default_top_k: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_DENSE_DEFAULT_TOP_K",
    )
    vector_schema: str | None = Field(
        default=None,
        validation_alias="REPOPILOT_VECTOR_SCHEMA",
    )
    vector_table: str | None = Field(
        default=None,
        validation_alias="REPOPILOT_VECTOR_TABLE",
    )
    vector_dimension: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_VECTOR_DIMENSION",
    )
    database_connect_timeout_seconds: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_DATABASE_CONNECT_TIMEOUT_SECONDS",
    )
    database_statement_timeout_ms: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_DATABASE_STATEMENT_TIMEOUT_MS",
    )
    database_url: SecretStr | None = Field(default=None, validation_alias="DATABASE_URL")
    sandbox_network: SandboxNetworkMode | None = Field(
        default=None,
        validation_alias="REPOPILOT_SANDBOX_NETWORK",
    )
    sandbox_cpu_limit: float | None = Field(
        default=None,
        validation_alias="REPOPILOT_SANDBOX_CPU_LIMIT",
    )
    sandbox_memory_limit_mb: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_SANDBOX_MEMORY_LIMIT_MB",
    )
    sandbox_pids_limit: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_SANDBOX_PIDS_LIMIT",
    )
    sandbox_tmpfs_limit_mb: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_SANDBOX_TMPFS_LIMIT_MB",
    )
    sandbox_command_timeout_seconds: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_SANDBOX_COMMAND_TIMEOUT_SECONDS",
    )
    sandbox_max_output_bytes: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_SANDBOX_MAX_OUTPUT_BYTES",
    )
    sandbox_max_artifact_bytes: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_SANDBOX_MAX_ARTIFACT_BYTES",
    )
    patch_max_bytes: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_PATCH_MAX_BYTES",
    )
    patch_max_files: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_PATCH_MAX_FILES",
    )
    patch_max_hunks: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_PATCH_MAX_HUNKS",
    )
    patch_max_changed_lines: int | None = Field(
        default=None,
        validation_alias="REPOPILOT_PATCH_MAX_CHANGED_LINES",
    )
    log_level: LogLevel | None = Field(default=None, validation_alias="REPOPILOT_LOG_LEVEL")
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="LANGSMITH_API_KEY",
    )
    langsmith_tracing: bool | None = Field(
        default=None,
        validation_alias="LANGSMITH_TRACING",
    )

    model_config = SettingsConfigDict(extra="ignore", env_file=None)


class RepoPilotSettings(BaseModel):
    """Immutable runtime settings after all configuration sources are merged."""

    profile: ConfigProfile = ConfigProfile.DEFAULT
    model_provider: ModelProvider = ModelProvider.FAKE
    model: str = "fake-deterministic"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    openai_reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM
    openai_max_output_tokens: int = Field(default=4_096, gt=0, le=128_000)
    openai_request_timeout_seconds: int = Field(default=120, gt=0, le=600)
    openai_max_retries: int = Field(default=2, ge=0, le=5)
    openai_retry_backoff_ms: int = Field(default=500, ge=0, le=10_000)
    openai_prompt_max_chars: int = Field(default=50_000, ge=4_096, le=200_000)
    openai_input_cost_per_million_usd: Decimal = Field(
        default=Decimal("2.50"),
        ge=0,
        le=1_000,
    )
    openai_cached_input_cost_per_million_usd: Decimal = Field(
        default=Decimal("0.25"),
        ge=0,
        le=1_000,
    )
    openai_cache_write_cost_per_million_usd: Decimal = Field(
        default=Decimal("3.125"),
        ge=0,
        le=1_000,
    )
    openai_output_cost_per_million_usd: Decimal = Field(
        default=Decimal("15.00"),
        ge=0,
        le=1_000,
    )
    max_steps: int = Field(default=30, gt=0)
    max_replans: int = Field(default=1, ge=0, le=10)
    max_tokens: int = Field(default=100_000, gt=0)
    timeout_seconds: int = Field(default=900, gt=0)
    max_cost_usd: Decimal = Field(default=Decimal("1.00"), gt=0)
    repository_max_file_bytes: int = Field(default=1_000_000, gt=0, le=10_000_000)
    repository_max_read_bytes: int = Field(default=65_536, gt=0, le=1_000_000)
    repository_max_read_lines: int = Field(default=400, gt=0, le=5_000)
    repository_max_list_entries: int = Field(default=2_000, gt=0, le=20_000)
    tool_max_output_bytes: int = Field(default=65_536, gt=0, le=1_000_000)
    search_max_matches: int = Field(default=100, gt=0, le=1_000)
    search_max_query_length: int = Field(default=256, gt=0, le=2_000)
    chunk_max_lines: int = Field(default=200, gt=0, le=2_000)
    chunk_overlap_lines: int = Field(default=20, ge=0, le=500)
    bm25_k1: float = Field(default=1.5, gt=0.0, le=5.0, allow_inf_nan=False)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0, allow_inf_nan=False)
    bm25_default_top_k: int = Field(default=10, gt=0, le=100)
    retrieval_mode: RetrievalMode = RetrievalMode.BM25
    retrieval_default_top_k: int = Field(default=10, gt=0, le=100)
    hybrid_candidate_top_k: int = Field(default=100, gt=0, le=100)
    hybrid_rrf_k: int = Field(default=60, gt=0, le=1_000)
    query_rewrite_enabled: bool = False
    query_rewrite_min_hits: int = Field(default=3, gt=0, le=100)
    query_rewrite_max_context_chunks: int = Field(default=5, ge=0, le=10)
    embedding_provider: EmbeddingProviderKind = EmbeddingProviderKind.FAKE
    embedding_model: str = Field(default="fake-code-embedding", min_length=1, max_length=128)
    embedding_revision: str = Field(default="v1", min_length=1, max_length=128)
    embedding_code_revision: str = Field(default="v1", min_length=1, max_length=128)
    embedding_batch_size: int = Field(default=32, gt=0, le=256)
    embedding_max_retries: int = Field(default=2, ge=0, le=5)
    embedding_retry_backoff_ms: int = Field(default=100, ge=0, le=10_000)
    embedding_max_length: int = Field(default=2_048, gt=0, le=8_192)
    embedding_normalize: bool = True
    embedding_cache_dir: str = Field(
        default="artifacts/embedding-cache",
        min_length=1,
        max_length=512,
    )
    embedding_trust_remote_code: bool = False
    dense_default_top_k: int = Field(default=10, gt=0, le=100)
    vector_schema: str = Field(default="repopilot", min_length=1, max_length=63)
    vector_table: str = Field(default="vector_embeddings", min_length=1, max_length=48)
    vector_dimension: int = Field(default=768, gt=0, le=2_000)
    database_connect_timeout_seconds: int = Field(default=5, gt=0, le=60)
    database_statement_timeout_ms: int = Field(default=5_000, ge=100, le=120_000)
    database_url: SecretStr = SecretStr(DEFAULT_DATABASE_URL)
    sandbox_network: SandboxNetworkMode = SandboxNetworkMode.NONE
    sandbox_cpu_limit: float = Field(default=1.0, gt=0.0, le=8.0, allow_inf_nan=False)
    sandbox_memory_limit_mb: int = Field(default=1_024, ge=128, le=8_192)
    sandbox_pids_limit: int = Field(default=256, ge=16, le=1_024)
    sandbox_tmpfs_limit_mb: int = Field(default=256, ge=16, le=2_048)
    sandbox_command_timeout_seconds: int = Field(default=300, gt=0, le=3_600)
    sandbox_max_output_bytes: int = Field(default=1_000_000, gt=0, le=10_000_000)
    sandbox_max_artifact_bytes: int = Field(
        default=20_000_000,
        gt=0,
        le=100_000_000,
    )
    patch_max_bytes: int = Field(default=262_144, gt=0, le=1_000_000)
    patch_max_files: int = Field(default=20, gt=0, le=100)
    patch_max_hunks: int = Field(default=100, gt=0, le=1_000)
    patch_max_changed_lines: int = Field(default=2_000, gt=0, le=10_000)
    log_level: LogLevel = LogLevel.INFO
    openai_api_key: SecretStr | None = None
    langsmith_api_key: SecretStr | None = None
    langsmith_tracing: bool = False

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("model")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        """Reject empty or whitespace-only model identifiers."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("model must not be empty")
        return normalized

    @field_validator("chunk_max_lines", "chunk_overlap_lines", mode="before")
    @classmethod
    def reject_boolean_chunk_limits(cls, value: object) -> object:
        """Do not silently treat booleans as integer chunk limits."""

        if isinstance(value, bool):
            raise ValueError("chunk line limits must be integers, not booleans")
        return value

    @field_validator("bm25_k1", "bm25_b", "bm25_default_top_k", mode="before")
    @classmethod
    def reject_boolean_bm25_values(cls, value: object) -> object:
        """Do not silently coerce booleans into BM25 numeric settings."""

        if isinstance(value, bool):
            raise ValueError("BM25 numeric settings must not be booleans")
        return value

    @field_validator(
        "embedding_batch_size",
        "embedding_max_retries",
        "embedding_retry_backoff_ms",
        "embedding_max_length",
        "dense_default_top_k",
        "retrieval_default_top_k",
        "hybrid_candidate_top_k",
        "hybrid_rrf_k",
        "query_rewrite_min_hits",
        "query_rewrite_max_context_chunks",
        "vector_dimension",
        "database_connect_timeout_seconds",
        "database_statement_timeout_ms",
        "sandbox_cpu_limit",
        "sandbox_memory_limit_mb",
        "sandbox_pids_limit",
        "sandbox_tmpfs_limit_mb",
        "sandbox_command_timeout_seconds",
        "sandbox_max_output_bytes",
        "sandbox_max_artifact_bytes",
        "patch_max_bytes",
        "patch_max_files",
        "patch_max_hunks",
        "patch_max_changed_lines",
        "openai_max_output_tokens",
        "openai_request_timeout_seconds",
        "openai_max_retries",
        "openai_retry_backoff_ms",
        "openai_prompt_max_chars",
        "openai_input_cost_per_million_usd",
        "openai_cached_input_cost_per_million_usd",
        "openai_cache_write_cost_per_million_usd",
        "openai_output_cost_per_million_usd",
        mode="before",
    )
    @classmethod
    def reject_boolean_runtime_numeric_values(cls, value: object) -> object:
        """Do not silently coerce booleans into runtime numeric settings."""

        if isinstance(value, bool):
            raise ValueError("runtime numeric settings must not be booleans")
        return value

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, value: str) -> str:
        """Accept canonical local/Hugging Face model names without control characters."""

        normalized = value.strip()
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", normalized) is None
            or "//" in normalized
        ):
            raise ValueError("embedding_model must be a canonical model identifier")
        return normalized

    @field_validator("embedding_revision", "embedding_code_revision")
    @classmethod
    def validate_embedding_revision(cls, value: str) -> str:
        """Require one traceable tag or immutable hexadecimal revision."""

        normalized = value.strip()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", normalized) is None:
            raise ValueError("embedding_revision must be a canonical revision")
        return normalized

    @field_validator("embedding_cache_dir")
    @classmethod
    def validate_embedding_cache_dir(cls, value: str) -> str:
        """Keep the local cache at one canonical project-relative path."""

        path = PurePosixPath(value)
        if (
            not value
            or "\x00" in value
            or "\\" in value
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != value
            or value in {".", ".."}
        ):
            raise ValueError("embedding_cache_dir must be a canonical relative path")
        return value

    @field_validator("vector_schema", "vector_table")
    @classmethod
    def validate_database_identifier(cls, value: str) -> str:
        """Restrict dynamic SQL identifiers before quoting them."""

        if re.fullmatch(r"[a-z][a-z0-9_]*", value) is None:
            raise ValueError(
                "database identifiers must use lowercase letters, digits, and underscores"
            )
        return value

    @field_validator("openai_api_key", "langsmith_api_key", mode="before")
    @classmethod
    def normalize_optional_secret(cls, value: object) -> object:
        """Treat blank optional secrets as absent."""

        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, SecretStr) and not value.get_secret_value().strip():
            return None
        return value

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: SecretStr) -> SecretStr:
        """Allow only PostgreSQL DSNs without exposing them in errors."""

        raw_value = value.get_secret_value().strip()
        allowed_prefixes = ("postgresql://", "postgresql+psycopg://")
        if not raw_value.startswith(allowed_prefixes):
            raise ValueError("database_url must use a PostgreSQL scheme")
        return SecretStr(raw_value)

    @model_validator(mode="after")
    def validate_cross_field_settings(self) -> Self:
        """Enforce integration credentials and related numeric limits."""

        if self.model_provider is ModelProvider.OPENAI:
            if self.openai_api_key is None:
                raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
            if self.model == "fake-deterministic":
                raise ValueError("an explicit model is required for the OpenAI provider")
            if re.fullmatch(r"gpt-[A-Za-z0-9][A-Za-z0-9._-]{0,123}", self.model) is None:
                raise ValueError("OpenAI model must be a canonical gpt-* identifier")
            if self.model in {"gpt-5.6", "gpt-5-latest"} or self.model.endswith("-latest"):
                raise ValueError("OpenAI model must be a pinned model ID, not a mutable alias")
            if (
                self.openai_input_cost_per_million_usd == 0
                or self.openai_output_cost_per_million_usd == 0
            ):
                raise ValueError("OpenAI input and output pricing must be positive")
        if self.langsmith_tracing and self.langsmith_api_key is None:
            raise ValueError("LANGSMITH_API_KEY is required when tracing is enabled")
        if self.repository_max_read_bytes > self.repository_max_file_bytes:
            raise ValueError("repository_max_read_bytes must not exceed repository_max_file_bytes")
        if self.chunk_overlap_lines >= self.chunk_max_lines:
            raise ValueError("chunk_overlap_lines must be less than chunk_max_lines")
        if self.hybrid_candidate_top_k < self.retrieval_default_top_k:
            raise ValueError("hybrid_candidate_top_k must not be less than retrieval_default_top_k")
        if self.sandbox_tmpfs_limit_mb > self.sandbox_memory_limit_mb:
            raise ValueError("sandbox_tmpfs_limit_mb must not exceed sandbox_memory_limit_mb")
        if self.sandbox_max_output_bytes > self.sandbox_max_artifact_bytes:
            raise ValueError("sandbox_max_output_bytes must not exceed sandbox_max_artifact_bytes")
        if self.patch_max_bytes > self.sandbox_max_artifact_bytes:
            raise ValueError("patch_max_bytes must not exceed sandbox_max_artifact_bytes")
        if self.embedding_provider is EmbeddingProviderKind.JINA_LOCAL:
            if self.embedding_model != DEFAULT_JINA_CODE_MODEL:
                raise ValueError(f"jina_local requires embedding_model={DEFAULT_JINA_CODE_MODEL}")
            if re.fullmatch(r"[0-9a-f]{40}", self.embedding_revision) is None:
                raise ValueError("jina_local requires an immutable 40-character revision")
            if re.fullmatch(r"[0-9a-f]{40}", self.embedding_code_revision) is None:
                raise ValueError("jina_local requires an immutable 40-character code revision")
            if self.vector_dimension != 768:
                raise ValueError("jina_local requires vector_dimension=768")
            if not self.embedding_trust_remote_code:
                raise ValueError("jina_local requires explicit embedding_trust_remote_code=true")
        elif self.embedding_trust_remote_code:
            raise ValueError(
                "embedding_trust_remote_code is only valid for the jina_local provider"
            )
        return self

    def redacted_snapshot(self) -> dict[str, object]:
        """Return a JSON-compatible configuration snapshot safe for traces."""

        snapshot = cast(dict[str, object], self.model_dump(mode="json"))
        snapshot["database_url"] = "<redacted>"
        snapshot["openai_api_key"] = "<redacted>" if self.openai_api_key is not None else None
        snapshot["langsmith_api_key"] = "<redacted>" if self.langsmith_api_key is not None else None
        return snapshot


def load_settings(
    config_path: str | Path | None = None,
    *,
    overrides: Mapping[str, object] | None = None,
) -> RepoPilotSettings:
    """Load settings with defaults < YAML < environment < explicit overrides."""

    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    yaml_values = _load_yaml(path)

    try:
        environment_values = _EnvironmentSettings().model_dump(exclude_none=True)
    except ValidationError as error:
        raise _safe_validation_error("environment", error) from None

    merged_values: dict[str, object] = dict(yaml_values)
    merged_values.update(environment_values)
    if overrides is not None:
        merged_values.update(overrides)

    try:
        return RepoPilotSettings.model_validate(merged_values)
    except ValidationError as error:
        raise _safe_validation_error("merged configuration", error) from None


def _load_yaml(path: Path) -> dict[str, object]:
    """Read a YAML mapping without echoing file contents on failure."""

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        raise ConfigurationError(f"configuration file is not readable: {path}") from None

    try:
        loaded = yaml.safe_load(raw_text)
    except yaml.YAMLError:
        raise ConfigurationError(f"configuration file is not valid YAML: {path}") from None

    if not isinstance(loaded, dict):
        raise ConfigurationError(f"configuration root must be a mapping: {path}")
    if not all(isinstance(key, str) for key in loaded):
        raise ConfigurationError(f"configuration keys must be strings: {path}")
    return cast(dict[str, object], loaded)


def _safe_validation_error(source: str, error: ValidationError) -> ConfigurationError:
    """Build a useful validation message without including rejected input values."""

    messages: list[str] = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in item["loc"]) or "root"
        messages.append(f"{location}: {item['msg']}")
    return ConfigurationError(f"invalid {source}: {'; '.join(messages)}")
