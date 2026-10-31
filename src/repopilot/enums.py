"""Stable string enums shared by configuration, runtime state, and evaluation."""

from enum import IntEnum, StrEnum, unique


@unique
class ConfigProfile(StrEnum):
    """Named configuration profiles shipped with RepoPilot."""

    DEFAULT = "default"
    EVAL = "eval"
    TEST = "test"


@unique
class ModelProvider(StrEnum):
    """Model-provider adapters supported by the MVP."""

    FAKE = "fake"
    OPENAI = "openai"


@unique
class ReasoningEffort(StrEnum):
    """Reasoning effort values supported by the pinned OpenAI model family."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


@unique
class EmbeddingProviderKind(StrEnum):
    """Embedding implementations kept separate from Agent decision providers."""

    FAKE = "fake"
    JINA_LOCAL = "jina_local"


@unique
class RetrievalMode(StrEnum):
    """Selectable standalone retrieval strategies."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


@unique
class RepairContextMode(StrEnum):
    """Precomputed repository context admitted to one repair run."""

    NONE = "none"
    BM25 = "bm25"
    CONFIGURED = "configured"


@unique
class BaselineKind(StrEnum):
    """Frozen P7 comparison arms implemented by the baseline adapters."""

    MINI_SWE_SAFE = "mini_swe_safe"
    NO_SEMANTIC_RAG = "no_semantic_rag"


@unique
class ExperimentArmKind(StrEnum):
    """Frozen P7 comparison and optional Memory arms."""

    MINI_SWE_SAFE = "mini_swe_safe"
    NO_SEMANTIC_RAG = "no_semantic_rag"
    DENSE_ONLY = "dense_only"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_MEMORY = "hybrid_memory"


@unique
class ExperimentPreflightCheckStatus(StrEnum):
    """Result of one fail-closed experiment readiness check."""

    PASSED = "passed"
    BLOCKED = "blocked"
    NOT_RUN = "not_run"


@unique
class ExperimentPreflightStatus(StrEnum):
    """Whether a frozen matrix may enter a formal evaluation run."""

    READY = "ready"
    BLOCKED = "blocked"


@unique
class ExperimentErrorCode(StrEnum):
    """Stable failures while loading, selecting, or preflighting an experiment."""

    INPUT_IO_ERROR = "input_io_error"
    INVALID_CONFIGURATION = "invalid_configuration"
    UNSUPPORTED_VERSION = "unsupported_version"
    INTEGRITY_MISMATCH = "integrity_mismatch"
    DATASET_MISMATCH = "dataset_mismatch"
    FAIRNESS_VIOLATION = "fairness_violation"
    ARM_UNAVAILABLE = "arm_unavailable"
    MODEL_SNAPSHOT_INVALID = "model_snapshot_invalid"
    EMBEDDING_SNAPSHOT_INVALID = "embedding_snapshot_invalid"
    MEMORY_SNAPSHOT_INVALID = "memory_snapshot_invalid"
    DEPENDENCY_LOCK_MISMATCH = "dependency_lock_mismatch"
    SOURCE_SNAPSHOT_UNCLEAN = "source_snapshot_unclean"
    SANDBOX_IMAGE_UNAVAILABLE = "sandbox_image_unavailable"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    VECTOR_BACKEND_UNAVAILABLE = "vector_backend_unavailable"
    GOLD_BOUNDARY_VIOLATION = "gold_boundary_violation"
    OUTPUT_CONFLICT = "output_conflict"
    SECRET_EXPOSURE = "secret_exposure"


@unique
class CalibrationErrorCode(StrEnum):
    """Stable failures for dev-only calibration and formal freezing."""

    INVALID_CONFIGURATION = "invalid_configuration"
    UNSUPPORTED_VERSION = "unsupported_version"
    INTEGRITY_MISMATCH = "integrity_mismatch"
    DEV_TASK_SET_MISMATCH = "dev_task_set_mismatch"
    HOLDOUT_ACCESS_DENIED = "holdout_access_denied"
    BUDGET_APPROVAL_REQUIRED = "budget_approval_required"
    BUDGET_APPROVAL_MISMATCH = "budget_approval_mismatch"
    BUDGET_LIMIT_EXCEEDED = "budget_limit_exceeded"
    PROVIDER_CREDENTIAL_REQUIRED = "provider_credential_required"
    PREFLIGHT_BLOCKED = "preflight_blocked"
    SOURCE_SNAPSHOT_UNCLEAN = "source_snapshot_unclean"
    RUN_INCOMPLETE = "run_incomplete"
    OUTPUT_CONFLICT = "output_conflict"
    PATH_UNSAFE = "path_unsafe"
    INPUT_IO_ERROR = "input_io_error"


@unique
class RetrievalTestScope(StrEnum):
    """Whether metadata filtering includes source files, tests, or both."""

    ALL = "all"
    TESTS_ONLY = "tests_only"
    NON_TESTS_ONLY = "non_tests_only"


@unique
class QueryRewriteDecision(StrEnum):
    """Why a query was preserved, rewritten, or prevented from another rewrite."""

    DISABLED = "disabled"
    SUFFICIENT_EVIDENCE = "sufficient_evidence"
    REWRITTEN = "rewritten"
    LIMIT_REACHED = "limit_reached"


@unique
class ComparisonCaseOutcome(StrEnum):
    """Hybrid first-relevant-rank change relative to one baseline."""

    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"


@unique
class SandboxNetworkMode(StrEnum):
    """Network modes allowed by configuration.

    Policy enforcement remains the responsibility of the Sandbox phase.
    """

    NONE = "none"
    SETUP_ONLY = "setup_only"


@unique
class SandboxPolicyOperation(StrEnum):
    """Requests decided before a Docker adapter may act."""

    PREPARE = "prepare"
    COMMAND = "command"
    PATH = "path"


@unique
class SandboxCommandPhase(StrEnum):
    """Task-owned command group and its network eligibility."""

    SETUP = "setup"
    TEST = "test"


@unique
class SandboxPathOperation(StrEnum):
    """Contained path capability requested from the Sandbox."""

    READ_REPOSITORY = "read_repository"
    WRITE_REPOSITORY = "write_repository"
    WRITE_ARTIFACT = "write_artifact"


@unique
class SandboxPolicyErrorCode(StrEnum):
    """Stable pre-execution Sandbox policy denials."""

    INVALID_REQUEST = "invalid_request"
    IMAGE_DENIED = "image_denied"
    TASK_MISMATCH = "task_mismatch"
    COMMAND_NOT_CONFIGURED = "command_not_configured"
    EXECUTABLE_DENIED = "executable_denied"
    PATH_DENIED = "path_denied"
    NETWORK_DENIED = "network_denied"
    HOST_MOUNT_DENIED = "host_mount_denied"
    DOCKER_SOCKET_DENIED = "docker_socket_denied"
    PRIVILEGED_DENIED = "privileged_denied"
    ROOT_USER_DENIED = "root_user_denied"
    CAPABILITY_DENIED = "capability_denied"
    ROOTFS_WRITE_DENIED = "rootfs_write_denied"
    NO_NEW_PRIVILEGES_REQUIRED = "no_new_privileges_required"
    ENVIRONMENT_DENIED = "environment_denied"
    RESOURCE_LIMIT_DENIED = "resource_limit_denied"
    TIMEOUT_DENIED = "timeout_denied"


@unique
class DockerSandboxErrorCode(StrEnum):
    """Stable Docker lifecycle failures safe to expose in traces."""

    LIFECYCLE_ERROR = "lifecycle_error"
    POLICY_DENIED = "policy_denied"
    SOURCE_UNAVAILABLE = "source_unavailable"
    DAEMON_UNAVAILABLE = "daemon_unavailable"
    IMAGE_UNAVAILABLE = "image_unavailable"
    CREATE_FAILED = "create_failed"
    START_FAILED = "start_failed"
    WORKSPACE_COPY_FAILED = "workspace_copy_failed"
    INSPECT_FAILED = "inspect_failed"
    POSTURE_MISMATCH = "posture_mismatch"
    DUPLICATE_CALL = "duplicate_call"
    EXECUTION_FAILED = "execution_failed"
    ARTIFACT_IO_FAILED = "artifact_io_failed"
    NETWORK_TRANSITION_FAILED = "network_transition_failed"
    OPERATION_UNAVAILABLE = "operation_unavailable"
    CLEANUP_FAILED = "cleanup_failed"


@unique
class LogLevel(StrEnum):
    """Supported application log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@unique
class TerminationReason(StrEnum):
    """Why an Agent run reached a terminal state."""

    SUCCESS = "success"
    MAX_STEPS = "max_steps"
    TOKEN_LIMIT = "token_limit"
    TIMEOUT = "timeout"
    COST_LIMIT = "cost_limit"
    SANDBOX_ERROR = "sandbox_error"
    INVALID_PATCH = "invalid_patch"
    TESTS_FAILED = "tests_failed"
    TOOL_ERROR = "tool_error"
    PROVIDER_ERROR = "provider_error"
    RETRIEVAL_EXHAUSTED = "retrieval_exhausted"
    REPEATED_ACTION = "repeated_action"
    AGENT_STOPPED = "agent_stopped"


@unique
class ErrorType(StrEnum):
    """Structured runtime error attached to an observation or final result."""

    CONFIG_ERROR = "config_error"
    VALIDATION_ERROR = "validation_error"
    PROVIDER_ERROR = "provider_error"
    MODEL_FORMAT_ERROR = "model_format_error"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    DUPLICATE_TOOL_CALL = "duplicate_tool_call"
    POLICY_DENIED = "policy_denied"
    PATH_VIOLATION = "path_violation"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    TOOL_TIMEOUT = "tool_timeout"
    SANDBOX_ERROR = "sandbox_error"
    RETRIEVAL_ERROR = "retrieval_error"
    PATCH_ERROR = "patch_error"
    TEST_ERROR = "test_error"
    INTERNAL_ERROR = "internal_error"


@unique
class ProviderErrorCode(StrEnum):
    """Stable Provider failure causes safe to retain in Agent traces."""

    INVALID_CONFIGURATION = "invalid_configuration"
    SDK_UNAVAILABLE = "sdk_unavailable"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMITED = "rate_limited"
    BAD_REQUEST = "bad_request"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    SERVER_ERROR = "server_error"
    RESPONSE_INCOMPLETE = "response_incomplete"
    RESPONSE_MALFORMED = "response_malformed"
    MODEL_MISMATCH = "model_mismatch"
    TOOL_COUNT_INVALID = "tool_count_invalid"
    TOOL_NOT_ALLOWED = "tool_not_allowed"
    TOOL_ARGUMENTS_INVALID = "tool_arguments_invalid"
    USAGE_INVALID = "usage_invalid"
    PROMPT_TOO_LARGE = "prompt_too_large"
    SCRIPT_EXHAUSTED = "script_exhausted"


@unique
class ContextSourceKind(StrEnum):
    """Logical source of one model-visible or candidate context item."""

    SYSTEM = "system"
    CAPABILITIES = "capabilities"
    REQUEST = "request"
    TASK = "task"
    REPOSITORY = "repository"
    REPOSITORY_MEMORY = "repository_memory"
    EPISODIC_MEMORY = "episodic_memory"
    PLAN = "plan"
    RETRIEVAL = "retrieval"
    PATCH = "patch"
    TEST = "test"
    BUDGET = "budget"
    TOOL_RESULT = "tool_result"
    FAILURE = "failure"
    WORKING_MEMORY = "working_memory"


@unique
class ContextTrustLevel(StrEnum):
    """Security/provenance boundary carried with projected context."""

    TRUSTED_CONTROL = "trusted_control"
    UNTRUSTED_INPUT = "untrusted_input"
    UNTRUSTED_REPOSITORY = "untrusted_repository"
    RUNTIME_STATE = "runtime_state"
    OBSERVED_EVIDENCE = "observed_evidence"
    HISTORICAL_EVIDENCE = "historical_evidence"
    UNVERIFIED_MEMORY = "unverified_memory"


@unique
class ContextExclusionReason(StrEnum):
    """Why a context candidate did not enter the final Prompt."""

    RECENCY_LIMIT = "recency_limit"
    CHARACTER_LIMIT = "character_limit"
    DUPLICATE = "duplicate"
    SOURCE_BUDGET = "source_budget"
    GLOBAL_BUDGET = "global_budget"
    SUPERSEDED = "superseded"
    SUMMARY_REPLACEMENT = "summary_replacement"
    ARTIFACT_OFFLOAD = "artifact_offload"


@unique
class ContextBudgetKind(StrEnum):
    """Independent budget pools used by the deterministic Context Manager."""

    CONTROL = "control"
    TASK = "task"
    PLAN = "plan"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    MEMORY = "memory"


@unique
class ContextTransformKind(StrEnum):
    """Auditable, deterministic context transformations."""

    DEDUPLICATE = "deduplicate"
    EXTRACTIVE_SUMMARY = "extractive_summary"
    ARTIFACT_PROJECTION = "artifact_projection"


@unique
class ContextErrorCode(StrEnum):
    """Stable failures while constructing or reconciling context accounting."""

    INVALID_LIMIT = "invalid_limit"
    INVALID_NAMESPACE = "invalid_namespace"
    INVALID_ITEM = "invalid_item"
    LEDGER_MISMATCH = "ledger_mismatch"
    PROMPT_MISMATCH = "prompt_mismatch"
    INVALID_POLICY = "invalid_policy"
    ESSENTIAL_OVERFLOW = "essential_overflow"
    ARTIFACT_REQUIRED = "artifact_required"
    TRANSFORM_MISMATCH = "transform_mismatch"


@unique
class RepairWorkflowErrorCode(StrEnum):
    """Stable orchestration failures outside model-controlled Agent transitions."""

    INPUT_IO_ERROR = "input_io_error"
    INVALID_TASK = "invalid_task"
    INVALID_PROVIDER_SCRIPT = "invalid_provider_script"
    INVALID_OUTPUT = "invalid_output"
    PROVIDER_CONFIGURATION = "provider_configuration"
    RETRIEVAL_FAILURE = "retrieval_failure"
    MEMORY_FAILURE = "memory_failure"
    SANDBOX_PREPARATION = "sandbox_preparation"
    SETUP_FAILURE = "setup_failure"
    AGENT_FAILURE = "agent_failure"
    TRACE_FAILURE = "trace_failure"
    CLEANUP_FAILURE = "cleanup_failure"
    OUTPUT_IO_ERROR = "output_io_error"


@unique
class ErrorCategory(StrEnum):
    """Failure taxonomy used for aggregate evaluation and bad-case analysis."""

    RETRIEVAL_MISS = "retrieval_miss"
    WRONG_PLAN = "wrong_plan"
    TOOL_MISUSE = "tool_misuse"
    CONTEXT_LOSS = "context_loss"
    INVALID_PATCH = "invalid_patch"
    TEST_MISREAD = "test_misread"
    SANDBOX_FAILURE = "sandbox_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MODEL_FORMAT_ERROR = "model_format_error"


@unique
class DatasetSplit(StrEnum):
    """Dataset partitions with different gold-data visibility rules."""

    DEV = "dev"
    HOLDOUT = "holdout"


@unique
class EvaluationAttemptStatus(StrEnum):
    """Append-only lifecycle states for one task/repetition evaluation slot."""

    STARTED = "started"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


@unique
class EvaluationRunnerErrorCode(StrEnum):
    """Stable failures from the evaluation run/result persistence boundary."""

    INVALID_MANIFEST = "invalid_manifest"
    UNSUPPORTED_VERSION = "unsupported_version"
    IDENTITY_MISMATCH = "identity_mismatch"
    CONFIGURATION_DRIFT = "configuration_drift"
    DUPLICATE_ATTEMPT = "duplicate_attempt"
    RESUME_LIMIT = "resume_limit"
    OUTPUT_CONFLICT = "output_conflict"
    ARTIFACT_INTEGRITY = "artifact_integrity"
    EXECUTION_FAILURE = "execution_failure"
    PATH_UNSAFE = "path_unsafe"
    CONCURRENT_WRITE = "concurrent_write"
    INPUT_IO_ERROR = "input_io_error"
    OUTPUT_IO_ERROR = "output_io_error"


@unique
class BaselineErrorCode(StrEnum):
    """Stable failures while loading or binding one baseline execution arm."""

    INPUT_IO_ERROR = "input_io_error"
    INVALID_CONFIGURATION = "invalid_configuration"
    UNSUPPORTED_VERSION = "unsupported_version"
    CAPABILITY_MISMATCH = "capability_mismatch"
    FAIRNESS_VIOLATION = "fairness_violation"
    INFORMATION_BOUNDARY = "information_boundary"


@unique
class DatasetCaseKind(StrEnum):
    """Origin class for one frozen evaluation case."""

    FIXTURE = "fixture"
    GITHUB_ISSUE = "github_issue"


@unique
class DatasetSelectionDecision(StrEnum):
    """Whether a reviewed public candidate entered the frozen dataset."""

    INCLUDED = "included"
    EXCLUDED = "excluded"


@unique
class DatasetExclusionReason(StrEnum):
    """Stable reasons for retaining but rejecting a candidate task."""

    DUPLICATE = "duplicate"
    MUTABLE_REVISION = "mutable_revision"
    REPOSITORY_UNAVAILABLE = "repository_unavailable"
    ATTRIBUTION_MISSING = "attribution_missing"
    LICENSE_UNVERIFIED = "license_unverified"
    SETUP_UNSUPPORTED = "setup_unsupported"
    BASELINE_NOT_REPRODUCIBLE = "baseline_not_reproducible"
    GOLD_PATCH_INVALID = "gold_patch_invalid"
    RESOURCE_LIMIT = "resource_limit"
    SECRET_OR_MANUAL_STATE = "secret_or_manual_state"
    UNCONTROLLED_NETWORK = "uncontrolled_network"
    SPLIT_POLICY = "split_policy"
    ENVIRONMENT_INCOMPATIBLE = "environment_incompatible"


@unique
class DatasetPreflightStatus(StrEnum):
    """Evidence depth reached for one frozen dataset case."""

    METADATA_VERIFIED = "metadata_verified"
    BASELINE_FAILED_GOLD_PASSED = "baseline_failed_gold_passed"


@unique
class DatasetManifestErrorCode(StrEnum):
    """Stable failures at the frozen dataset/evaluator boundary."""

    MANIFEST_UNAVAILABLE = "manifest_unavailable"
    MANIFEST_INVALID = "manifest_invalid"
    UNSUPPORTED_VERSION = "unsupported_version"
    INTEGRITY_MISMATCH = "integrity_mismatch"
    IDENTITY_MISMATCH = "identity_mismatch"
    DUPLICATE_CASE = "duplicate_case"
    SOURCE_UNVERIFIED = "source_unverified"
    ATTRIBUTION_MISSING = "attribution_missing"
    LICENSE_MISSING = "license_missing"
    SPLIT_VIOLATION = "split_violation"
    GOLD_UNAVAILABLE = "gold_unavailable"
    PREFLIGHT_INVALID = "preflight_invalid"
    PATH_UNSAFE = "path_unsafe"
    INPUT_IO_ERROR = "input_io_error"


@unique
class MemoryAccessMode(StrEnum):
    """Whether one declared experiment snapshot may accept runtime writes."""

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


@unique
class SymbolType(StrEnum):
    """Coarse symbol kinds emitted by repository indexing."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    TEXT = "text"


@unique
class PlanStepStatus(StrEnum):
    """Lifecycle states for a single plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@unique
class PolicyDecision(StrEnum):
    """Whether the sandbox policy allowed a requested operation."""

    ALLOWED = "allowed"
    DENIED = "denied"


@unique
class RunStatus(StrEnum):
    """Coarse outcome of a completed Agent run."""

    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


@unique
class PatchStatus(StrEnum):
    """Lifecycle state of the candidate patch."""

    NONE = "none"
    PROPOSED = "proposed"
    APPLIED = "applied"
    REJECTED = "rejected"


@unique
class PatchErrorCode(StrEnum):
    """Stable reasons a candidate diff was rejected or could not be applied."""

    INVALID_DIFF = "invalid_diff"
    PATCH_TOO_LARGE = "patch_too_large"
    TOO_MANY_FILES = "too_many_files"
    TOO_MANY_HUNKS = "too_many_hunks"
    TOO_MANY_CHANGED_LINES = "too_many_changed_lines"
    PATH_DENIED = "path_denied"
    UNSUPPORTED_PATCH = "unsupported_patch"
    ARTIFACT_MISMATCH = "artifact_mismatch"
    APPLY_CONFLICT = "apply_conflict"
    ROLLBACK_FAILED = "rollback_failed"
    INSPECT_UNAVAILABLE = "inspect_unavailable"
    SANDBOX_ERROR = "sandbox_error"
    PROTECTED_TEST_PATH = "protected_test_path"


@unique
class TestStatus(StrEnum):
    """Most recent validation-test state."""

    NOT_RUN = "not_run"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@unique
class TestRunCategory(StrEnum):
    """Semantic outcome of one configured test command."""

    PASSED = "passed"
    ASSERTION_FAILURE = "assertion_failure"
    TEST_ERROR = "test_error"
    IMPORT_ERROR = "import_error"
    NO_TESTS_COLLECTED = "no_tests_collected"
    OUTPUT_TRUNCATED = "output_truncated"
    TIMEOUT = "timeout"
    POLICY_DENIED = "policy_denied"
    SANDBOX_ERROR = "sandbox_error"
    EXECUTION_ERROR = "execution_error"
    UNKNOWN_NONZERO = "unknown_nonzero"


@unique
class ReproductionStatus(StrEnum):
    """Whether the original issue was reproduced before editing."""

    NOT_ATTEMPTED = "not_attempted"
    REPRODUCED = "reproduced"
    NOT_REPRODUCED = "not_reproduced"
    ERROR = "error"


@unique
class ValidationStatus(StrEnum):
    """Verifier outcome for the candidate patch."""

    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@unique
class VerifierErrorCode(StrEnum):
    """Stable reasons the rule-based Verifier refused task success."""

    REPRODUCTION_NOT_ATTEMPTED = "reproduction_not_attempted"
    ISSUE_NOT_REPRODUCED = "issue_not_reproduced"
    REPRODUCTION_ERROR = "reproduction_error"
    REPRODUCTION_EVIDENCE_INVALID = "reproduction_evidence_invalid"
    PATCH_NOT_APPLIED = "patch_not_applied"
    PATCH_EVIDENCE_INVALID = "patch_evidence_invalid"
    VALIDATION_NOT_RUN = "validation_not_run"
    TESTS_FAILED = "tests_failed"
    VALIDATION_EVIDENCE_INVALID = "validation_evidence_invalid"
    TEST_COMMAND_MISMATCH = "test_command_mismatch"
    OBSERVATION_REUSED = "observation_reused"
    TEST_INFRASTRUCTURE_ERROR = "test_infrastructure_error"


@unique
class ProviderActionKind(StrEnum):
    """Structured action kinds a model provider may request."""

    TOOL_CALL = "tool_call"
    FINISH = "finish"


@unique
class SandboxOperation(StrEnum):
    """Operations exposed by the minimal Sandbox adapter."""

    RUN = "run"
    READ_FILE = "read_file"
    APPLY_PATCH = "apply_patch"


@unique
class SandboxLifecycle(StrEnum):
    """Lifecycle states maintained by a Sandbox implementation."""

    NEW = "new"
    PREPARED = "prepared"
    DESTROYED = "destroyed"


@unique
class CliExitCode(IntEnum):
    """Stable process exit codes for RepoPilot command-line workflows."""

    SUCCESS = 0
    INTERNAL_ERROR = 1
    USAGE_ERROR = 2
    INVALID_INPUT = 3
    IO_ERROR = 4
    ADAPTER_ERROR = 5


@unique
class RepositoryEntryType(StrEnum):
    """Filesystem entry kinds exposed by read-only repository listings."""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"


@unique
class RepositoryErrorCode(StrEnum):
    """Stable reasons a read-only repository operation can fail."""

    INVALID_PATH = "invalid_path"
    OUTSIDE_ROOT = "outside_root"
    SYMLINK_FORBIDDEN = "symlink_forbidden"
    DENIED_PATH = "denied_path"
    NOT_FOUND = "not_found"
    NOT_FILE = "not_file"
    NOT_DIRECTORY = "not_directory"
    BINARY_FILE = "binary_file"
    FILE_TOO_LARGE = "file_too_large"
    DECODE_ERROR = "decode_error"
    IO_ERROR = "io_error"


@unique
class RepositoryIngestErrorCode(StrEnum):
    """Stable failures for deterministic repository-manifest construction."""

    LISTING_TRUNCATED = "listing_truncated"
    SOURCE_CHANGED = "source_changed"
    IO_ERROR = "io_error"
    INVALID_DOCUMENT = "invalid_document"
    UNSUPPORTED_VERSION = "unsupported_version"
    INTEGRITY_MISMATCH = "integrity_mismatch"


@unique
class ToolResultStatus(StrEnum):
    """Whether a Tool Registry invocation returned a payload or an error."""

    SUCCESS = "success"
    ERROR = "error"


@unique
class ToolErrorCode(StrEnum):
    """Stable Registry-level tool invocation failure reasons."""

    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    DUPLICATE_CALL = "duplicate_call"
    OUTPUT_TOO_LARGE = "output_too_large"
    REPOSITORY_ERROR = "repository_error"
    TOOL_EXECUTION_ERROR = "tool_execution_error"


@unique
class SymbolKind(StrEnum):
    """Python definition kinds returned by the first symbol inspector."""

    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    CLASS = "class"
    METHOD = "method"
    ASYNC_METHOD = "async_method"


@unique
class SourceInspectionErrorCode(StrEnum):
    """Non-filesystem reasons a source file could not be fully inspected."""

    PARTIAL_LINE = "partial_line"
    SYNTAX_ERROR = "syntax_error"


@unique
class PythonChunkMode(StrEnum):
    """How one Python source was converted into retrieval chunks."""

    AST = "ast"
    SYNTAX_FALLBACK = "syntax_fallback"
    EMPTY = "empty"


@unique
class PythonChunkErrorCode(StrEnum):
    """Stable fatal failures from Python source chunk construction."""

    SOURCE_UNAVAILABLE = "source_unavailable"
    SOURCE_CHANGED = "source_changed"


@unique
class TextSourceKind(StrEnum):
    """Agent-visible text source families handled by the sparse corpus."""

    MARKDOWN = "markdown"
    TEXT = "text"
    CONFIG = "config"
    ISSUE = "issue"


@unique
class TextChunkErrorCode(StrEnum):
    """Stable fatal failures from repository text chunk construction."""

    SOURCE_UNAVAILABLE = "source_unavailable"
    SOURCE_CHANGED = "source_changed"


@unique
class ChunkCorpusErrorCode(StrEnum):
    """Stable failures from combined chunk-corpus construction and JSONL I/O."""

    IO_ERROR = "io_error"
    INVALID_DOCUMENT = "invalid_document"
    UNSUPPORTED_VERSION = "unsupported_version"
    MIXED_CORPUS = "mixed_corpus"
    DUPLICATE_CHUNK = "duplicate_chunk"
    UNSORTED_CORPUS = "unsorted_corpus"


@unique
class BM25ErrorCode(StrEnum):
    """Stable failures from BM25 index persistence and querying."""

    EMPTY_QUERY = "empty_query"
    INVALID_QUERY = "invalid_query"
    IDENTITY_MISMATCH = "identity_mismatch"
    IO_ERROR = "io_error"
    INVALID_DOCUMENT = "invalid_document"
    UNSUPPORTED_VERSION = "unsupported_version"
    INTEGRITY_MISMATCH = "integrity_mismatch"


@unique
class LocalizationEvalErrorCode(StrEnum):
    """Stable failures from the offline BM25 localization baseline."""

    INPUT_IO_ERROR = "input_io_error"
    OUTPUT_IO_ERROR = "output_io_error"
    INVALID_MANIFEST = "invalid_manifest"
    UNSUPPORTED_VERSION = "unsupported_version"
    PATH_VIOLATION = "path_violation"
    IDENTITY_MISMATCH = "identity_mismatch"
    RETRIEVAL_FAILURE = "retrieval_failure"


@unique
class VectorStoreErrorCode(StrEnum):
    """Stable failures from vector storage contracts and adapters."""

    NOT_INITIALIZED = "not_initialized"
    INVALID_RECORD = "invalid_record"
    DUPLICATE_RECORD = "duplicate_record"
    NAMESPACE_MISMATCH = "namespace_mismatch"
    DIMENSION_MISMATCH = "dimension_mismatch"
    CONNECTION_ERROR = "connection_error"
    EXTENSION_UNAVAILABLE = "extension_unavailable"
    MIGRATION_ERROR = "migration_error"
    SCHEMA_MISMATCH = "schema_mismatch"
    QUERY_ERROR = "query_error"


@unique
class EmbeddingErrorCode(StrEnum):
    """Stable failures from embedding, caching, and batch orchestration."""

    INVALID_INPUT = "invalid_input"
    IDENTITY_MISMATCH = "identity_mismatch"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    REMOTE_CODE_DISABLED = "remote_code_disabled"
    PROVIDER_FAILURE = "provider_failure"
    RETRY_EXHAUSTED = "retry_exhausted"
    INVALID_RESPONSE = "invalid_response"
    CACHE_IO_ERROR = "cache_io_error"
    CACHE_CORRUPT = "cache_corrupt"
    CORPUS_MISMATCH = "corpus_mismatch"


@unique
class DenseRetrievalErrorCode(StrEnum):
    """Stable failures from query embedding and Dense result normalization."""

    EMPTY_QUERY = "empty_query"
    INVALID_QUERY = "invalid_query"
    IDENTITY_MISMATCH = "identity_mismatch"
    EMBEDDING_FAILURE = "embedding_failure"
    STORAGE_FAILURE = "storage_failure"
    INVALID_RESULT = "invalid_result"


@unique
class DenseEvalErrorCode(StrEnum):
    """Stable failures from gold-isolated Dense localization evaluation."""

    IDENTITY_MISMATCH = "identity_mismatch"
    RETRIEVAL_FAILURE = "retrieval_failure"
    INVALID_RESULT = "invalid_result"


@unique
class HybridRetrievalErrorCode(StrEnum):
    """Stable failures from routing, filtering, and RRF fusion."""

    INVALID_CONFIGURATION = "invalid_configuration"
    INVALID_QUERY = "invalid_query"
    IDENTITY_MISMATCH = "identity_mismatch"
    SPARSE_FAILURE = "sparse_failure"
    DENSE_FAILURE = "dense_failure"
    INVALID_RESULT = "invalid_result"


@unique
class QueryRewriteErrorCode(StrEnum):
    """Stable failures from one-shot structured query rewriting."""

    INVALID_INPUT = "invalid_input"
    IDENTITY_MISMATCH = "identity_mismatch"
    CONTEXT_MISMATCH = "context_mismatch"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_FAILURE = "provider_failure"
    INVALID_RESPONSE = "invalid_response"


@unique
class RetrievalComparisonErrorCode(StrEnum):
    """Stable failures from three-mode localization comparison."""

    IDENTITY_MISMATCH = "identity_mismatch"
    INVALID_REWRITE = "invalid_rewrite"
    RETRIEVAL_FAILURE = "retrieval_failure"
    OUTPUT_IO_ERROR = "output_io_error"
    INVALID_RESULT = "invalid_result"


@unique
class AgentStateErrorCode(StrEnum):
    """Stable failures for thread-scoped Agent state and checkpoints."""

    INVALID_THREAD_ID = "invalid_thread_id"
    THREAD_NOT_FOUND = "thread_not_found"
    THREAD_ALREADY_EXISTS = "thread_already_exists"
    TASK_MISMATCH = "task_mismatch"
    STATE_CONFLICT = "state_conflict"
    CHECKPOINT_ERROR = "checkpoint_error"


@unique
class AgentLoopErrorCode(StrEnum):
    """Stable failures for deterministic Agent-loop node execution."""

    INVALID_STATE_TRANSITION = "invalid_state_transition"
    PROVIDER_FAILURE = "provider_failure"
    TOOL_FAILURE = "tool_failure"
    UNFINISHED_PENDING_ACTION = "unfinished_pending_action"


@unique
class AgentFailureKind(StrEnum):
    """Recoverable or terminal failure observations recorded by the Agent loop."""

    EMPTY_SEARCH = "empty_search"
    TOOL_ERROR = "tool_error"
    PATCH_FAILURE = "patch_failure"
    TEST_FAILURE = "test_failure"
    REPEATED_ACTION = "repeated_action"
    PROVIDER_FAILURE = "provider_failure"


@unique
class TraceNode(StrEnum):
    """Agent graph transitions persisted in a versioned execution Trace."""

    INITIALIZE_PLAN = "initialize_plan"
    DECIDE = "decide"
    EXECUTE_TOOL = "execute_tool"
    OBSERVE = "observe"
    FINISH = "finish"
    TERMINATE = "terminate"


@unique
class TraceErrorCode(StrEnum):
    """Stable failures produced while loading or replaying a Trace artifact."""

    IO_ERROR = "io_error"
    INVALID_DOCUMENT = "invalid_document"
    UNSUPPORTED_VERSION = "unsupported_version"
    INTEGRITY_MISMATCH = "integrity_mismatch"
    EVENT_ORDER = "event_order"
    STATE_CHAIN_MISMATCH = "state_chain_mismatch"
    CORRELATION_MISMATCH = "correlation_mismatch"
    FINAL_STATE_MISMATCH = "final_state_mismatch"


@unique
class WorkingMemoryErrorCode(StrEnum):
    """Stable failures for explicit Working Memory save and recovery."""

    INVALID_SNAPSHOT = "invalid_snapshot"
    NAMESPACE_MISMATCH = "namespace_mismatch"
    COMMIT_STALE = "commit_stale"
    INTEGRITY_MISMATCH = "integrity_mismatch"
    SEQUENCE_ROLLBACK = "sequence_rollback"
    SNAPSHOT_LIMIT = "snapshot_limit"
    TERMINAL_SNAPSHOT = "terminal_snapshot"
    RECOVERY_CONFLICT = "recovery_conflict"
    PATH_UNSAFE = "path_unsafe"
    UNSUPPORTED_VERSION = "unsupported_version"
    IO_ERROR = "io_error"


@unique
class RepositoryMemoryKind(StrEnum):
    """Kinds of repository-scoped knowledge that may cross task boundaries."""

    REPOSITORY_MAP = "repository_map"
    TEST_COMMAND = "test_command"
    CONFIRMED_FACT = "confirmed_fact"


@unique
class RepositoryMemoryEvidenceKind(StrEnum):
    """Trusted origin used to admit one Repository Memory record."""

    MANIFEST = "manifest"
    SOURCE = "source"
    TASK_CONFIGURATION = "task_configuration"


@unique
class RepositoryMemoryConfirmation(StrEnum):
    """How a Repository Memory record became eligible for persistence."""

    MANIFEST_DERIVED = "manifest_derived"
    TASK_CONFIGURED = "task_configured"
    SOURCE_CONFIRMED = "source_confirmed"


@unique
class RepositoryMemoryLookupStatus(StrEnum):
    """Whether a repository query found current or invalidated memory."""

    EMPTY = "empty"
    CURRENT = "current"
    COMMIT_STALE = "commit_stale"
    CONTENT_STALE = "content_stale"


@unique
class RepositoryMemoryErrorCode(StrEnum):
    """Stable failures for Repository Memory admission, storage, and lookup."""

    INVALID_RECORD = "invalid_record"
    NAMESPACE_MISMATCH = "namespace_mismatch"
    COMMIT_STALE = "commit_stale"
    CONTENT_STALE = "content_stale"
    INTEGRITY_MISMATCH = "integrity_mismatch"
    UNCONFIRMED_RECORD = "unconfirmed_record"
    DUPLICATE_CONFLICT = "duplicate_conflict"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    PATH_UNSAFE = "path_unsafe"
    UNSUPPORTED_VERSION = "unsupported_version"
    IO_ERROR = "io_error"


@unique
class VerifiedEpisodeLookupStatus(StrEnum):
    """Whether an exact episodic namespace has usable or expired records."""

    EMPTY = "empty"
    CURRENT = "current"
    EXPIRED = "expired"


@unique
class VerifiedEpisodeErrorCode(StrEnum):
    """Stable failures for verified-episode admission, storage, and lookup."""

    INVALID_WRITE = "invalid_write"
    VERIFIER_REJECTED = "verifier_rejected"
    TRACE_INVALID = "trace_invalid"
    RESULT_MISMATCH = "result_mismatch"
    EVIDENCE_INCOMPLETE = "evidence_incomplete"
    ARTIFACT_MISMATCH = "artifact_mismatch"
    NAMESPACE_MISMATCH = "namespace_mismatch"
    EXPIRED = "expired"
    INTEGRITY_MISMATCH = "integrity_mismatch"
    DUPLICATE_CONFLICT = "duplicate_conflict"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    PATH_UNSAFE = "path_unsafe"
    UNSUPPORTED_VERSION = "unsupported_version"
    IO_ERROR = "io_error"


@unique
class MemoryNamespaceErrorCode(StrEnum):
    """Stable failures for experiment/snapshot/split Memory isolation."""

    INVALID_NAMESPACE = "invalid_namespace"
    SPLIT_MISMATCH = "split_mismatch"
    SNAPSHOT_MISMATCH = "snapshot_mismatch"
    ACCESS_DENIED = "access_denied"
    INTEGRITY_MISMATCH = "integrity_mismatch"
    PATH_UNSAFE = "path_unsafe"
    UNSUPPORTED_VERSION = "unsupported_version"
    IO_ERROR = "io_error"
