"""Tests that protect serialized enum values used by traces and results."""

from enum import StrEnum

from repopilot.enums import (
    AgentFailureKind,
    AgentLoopErrorCode,
    AgentStateErrorCode,
    BM25ErrorCode,
    ChunkCorpusErrorCode,
    CliExitCode,
    ComparisonCaseOutcome,
    ConfigProfile,
    ContextBudgetKind,
    ContextErrorCode,
    ContextExclusionReason,
    ContextSourceKind,
    ContextTransformKind,
    ContextTrustLevel,
    DatasetCaseKind,
    DatasetExclusionReason,
    DatasetManifestErrorCode,
    DatasetPreflightStatus,
    DatasetSelectionDecision,
    DatasetSplit,
    DenseEvalErrorCode,
    DenseRetrievalErrorCode,
    DockerSandboxErrorCode,
    EmbeddingErrorCode,
    EmbeddingProviderKind,
    ErrorCategory,
    ErrorType,
    EvaluationAttemptStatus,
    EvaluationRunnerErrorCode,
    ExperimentArmKind,
    ExperimentErrorCode,
    ExperimentPreflightCheckStatus,
    ExperimentPreflightStatus,
    HybridRetrievalErrorCode,
    LocalizationEvalErrorCode,
    LogLevel,
    MemoryAccessMode,
    MemoryNamespaceErrorCode,
    ModelProvider,
    PatchErrorCode,
    PatchStatus,
    PlanStepStatus,
    PolicyDecision,
    ProviderActionKind,
    ProviderErrorCode,
    PythonChunkErrorCode,
    PythonChunkMode,
    QueryRewriteDecision,
    QueryRewriteErrorCode,
    ReasoningEffort,
    RepairWorkflowErrorCode,
    RepositoryEntryType,
    RepositoryErrorCode,
    RepositoryIngestErrorCode,
    RepositoryMemoryConfirmation,
    RepositoryMemoryErrorCode,
    RepositoryMemoryEvidenceKind,
    RepositoryMemoryKind,
    RepositoryMemoryLookupStatus,
    ReproductionStatus,
    RetrievalComparisonErrorCode,
    RetrievalMode,
    RetrievalTestScope,
    RunStatus,
    SandboxCommandPhase,
    SandboxLifecycle,
    SandboxNetworkMode,
    SandboxOperation,
    SandboxPathOperation,
    SandboxPolicyErrorCode,
    SandboxPolicyOperation,
    SourceInspectionErrorCode,
    SymbolKind,
    SymbolType,
    TerminationReason,
    TextChunkErrorCode,
    TextSourceKind,
    ToolErrorCode,
    ToolResultStatus,
    TraceErrorCode,
    TraceNode,
    ValidationStatus,
    VectorStoreErrorCode,
    VerifiedEpisodeErrorCode,
    VerifiedEpisodeLookupStatus,
    VerifierErrorCode,
    WorkingMemoryErrorCode,
)
from repopilot.enums import (
    TestRunCategory as RepoTestRunCategory,
)
from repopilot.enums import (
    TestStatus as RepoTestStatus,
)


def enum_values(enum_type: type[StrEnum]) -> tuple[str, ...]:
    return tuple(member.value for member in enum_type)


def test_config_profile_values_are_stable() -> None:
    assert enum_values(ConfigProfile) == ("default", "eval", "test")


def test_model_provider_values_are_stable() -> None:
    assert enum_values(ModelProvider) == ("fake", "openai")
    assert enum_values(ReasoningEffort) == (
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )
    assert enum_values(ProviderErrorCode) == (
        "invalid_configuration",
        "sdk_unavailable",
        "connection_error",
        "timeout",
        "authentication_error",
        "permission_denied",
        "rate_limited",
        "bad_request",
        "not_found",
        "conflict",
        "server_error",
        "response_incomplete",
        "response_malformed",
        "model_mismatch",
        "tool_count_invalid",
        "tool_not_allowed",
        "tool_arguments_invalid",
        "usage_invalid",
        "prompt_too_large",
        "script_exhausted",
    )
    assert enum_values(RepairWorkflowErrorCode) == (
        "input_io_error",
        "invalid_task",
        "invalid_provider_script",
        "invalid_output",
        "provider_configuration",
        "retrieval_failure",
        "memory_failure",
        "sandbox_preparation",
        "setup_failure",
        "agent_failure",
        "trace_failure",
        "cleanup_failure",
        "output_io_error",
    )


def test_context_accounting_values_are_stable() -> None:
    assert enum_values(ContextSourceKind) == (
        "system",
        "capabilities",
        "request",
        "task",
        "repository",
        "repository_memory",
        "episodic_memory",
        "plan",
        "retrieval",
        "patch",
        "test",
        "budget",
        "tool_result",
        "failure",
        "working_memory",
    )
    assert enum_values(ContextTrustLevel) == (
        "trusted_control",
        "untrusted_input",
        "untrusted_repository",
        "runtime_state",
        "observed_evidence",
        "historical_evidence",
        "unverified_memory",
    )
    assert enum_values(ContextExclusionReason) == (
        "recency_limit",
        "character_limit",
        "duplicate",
        "source_budget",
        "global_budget",
        "superseded",
        "summary_replacement",
        "artifact_offload",
    )
    assert enum_values(ContextBudgetKind) == (
        "control",
        "task",
        "plan",
        "retrieval",
        "tool",
        "memory",
    )
    assert enum_values(ContextTransformKind) == (
        "deduplicate",
        "extractive_summary",
        "artifact_projection",
    )
    assert enum_values(ContextErrorCode) == (
        "invalid_limit",
        "invalid_namespace",
        "invalid_item",
        "ledger_mismatch",
        "prompt_mismatch",
        "invalid_policy",
        "essential_overflow",
        "artifact_required",
        "transform_mismatch",
    )


def test_embedding_provider_values_are_stable() -> None:
    assert enum_values(EmbeddingProviderKind) == ("fake", "jina_local")


def test_evaluation_runner_values_are_stable() -> None:
    assert enum_values(EvaluationAttemptStatus) == (
        "started",
        "interrupted",
        "completed",
        "failed",
    )
    assert enum_values(EvaluationRunnerErrorCode) == (
        "invalid_manifest",
        "unsupported_version",
        "identity_mismatch",
        "configuration_drift",
        "duplicate_attempt",
        "resume_limit",
        "output_conflict",
        "artifact_integrity",
        "execution_failure",
        "path_unsafe",
        "concurrent_write",
        "input_io_error",
        "output_io_error",
    )
    assert enum_values(ExperimentArmKind) == (
        "mini_swe_safe",
        "no_semantic_rag",
        "dense_only",
        "hybrid_rrf",
        "hybrid_memory",
    )
    assert enum_values(ExperimentPreflightCheckStatus) == (
        "passed",
        "blocked",
        "not_run",
    )
    assert enum_values(ExperimentPreflightStatus) == ("ready", "blocked")
    assert enum_values(ExperimentErrorCode) == (
        "input_io_error",
        "invalid_configuration",
        "unsupported_version",
        "integrity_mismatch",
        "dataset_mismatch",
        "fairness_violation",
        "arm_unavailable",
        "model_snapshot_invalid",
        "embedding_snapshot_invalid",
        "memory_snapshot_invalid",
        "dependency_lock_mismatch",
        "source_snapshot_unclean",
        "sandbox_image_unavailable",
        "provider_unavailable",
        "vector_backend_unavailable",
        "gold_boundary_violation",
        "output_conflict",
        "secret_exposure",
    )


def test_repository_memory_values_are_stable() -> None:
    assert enum_values(RepositoryMemoryKind) == (
        "repository_map",
        "test_command",
        "confirmed_fact",
    )
    assert enum_values(RepositoryMemoryEvidenceKind) == (
        "manifest",
        "source",
        "task_configuration",
    )
    assert enum_values(RepositoryMemoryConfirmation) == (
        "manifest_derived",
        "task_configured",
        "source_confirmed",
    )
    assert enum_values(RepositoryMemoryLookupStatus) == (
        "empty",
        "current",
        "commit_stale",
        "content_stale",
    )
    assert enum_values(RepositoryMemoryErrorCode) == (
        "invalid_record",
        "namespace_mismatch",
        "commit_stale",
        "content_stale",
        "integrity_mismatch",
        "unconfirmed_record",
        "duplicate_conflict",
        "capacity_exceeded",
        "path_unsafe",
        "unsupported_version",
        "io_error",
    )
    assert enum_values(VerifiedEpisodeLookupStatus) == (
        "empty",
        "current",
        "expired",
    )
    assert enum_values(VerifiedEpisodeErrorCode) == (
        "invalid_write",
        "verifier_rejected",
        "trace_invalid",
        "result_mismatch",
        "evidence_incomplete",
        "artifact_mismatch",
        "namespace_mismatch",
        "expired",
        "integrity_mismatch",
        "duplicate_conflict",
        "capacity_exceeded",
        "path_unsafe",
        "unsupported_version",
        "io_error",
    )
    assert enum_values(MemoryAccessMode) == ("read_only", "read_write")
    assert enum_values(MemoryNamespaceErrorCode) == (
        "invalid_namespace",
        "split_mismatch",
        "snapshot_mismatch",
        "access_denied",
        "integrity_mismatch",
        "path_unsafe",
        "unsupported_version",
        "io_error",
    )


def test_retrieval_mode_and_test_scope_values_are_stable() -> None:
    assert enum_values(RetrievalMode) == ("bm25", "dense", "hybrid")
    assert enum_values(RetrievalTestScope) == (
        "all",
        "tests_only",
        "non_tests_only",
    )
    assert enum_values(QueryRewriteDecision) == (
        "disabled",
        "sufficient_evidence",
        "rewritten",
        "limit_reached",
    )
    assert enum_values(ComparisonCaseOutcome) == (
        "improved",
        "regressed",
        "unchanged",
    )


def test_sandbox_network_values_are_stable() -> None:
    assert enum_values(SandboxNetworkMode) == ("none", "setup_only")
    assert enum_values(SandboxPolicyOperation) == ("prepare", "command", "path")
    assert enum_values(SandboxCommandPhase) == ("setup", "test")
    assert enum_values(SandboxPathOperation) == (
        "read_repository",
        "write_repository",
        "write_artifact",
    )
    assert enum_values(SandboxPolicyErrorCode) == (
        "invalid_request",
        "image_denied",
        "task_mismatch",
        "command_not_configured",
        "executable_denied",
        "path_denied",
        "network_denied",
        "host_mount_denied",
        "docker_socket_denied",
        "privileged_denied",
        "root_user_denied",
        "capability_denied",
        "rootfs_write_denied",
        "no_new_privileges_required",
        "environment_denied",
        "resource_limit_denied",
        "timeout_denied",
    )
    assert enum_values(DockerSandboxErrorCode) == (
        "lifecycle_error",
        "policy_denied",
        "source_unavailable",
        "daemon_unavailable",
        "image_unavailable",
        "create_failed",
        "start_failed",
        "workspace_copy_failed",
        "inspect_failed",
        "posture_mismatch",
        "duplicate_call",
        "execution_failed",
        "artifact_io_failed",
        "network_transition_failed",
        "operation_unavailable",
        "cleanup_failed",
    )


def test_log_level_values_are_stable() -> None:
    assert enum_values(LogLevel) == ("debug", "info", "warning", "error")


def test_termination_reason_values_are_stable() -> None:
    assert enum_values(TerminationReason) == (
        "success",
        "max_steps",
        "token_limit",
        "timeout",
        "cost_limit",
        "sandbox_error",
        "invalid_patch",
        "tests_failed",
        "tool_error",
        "provider_error",
        "retrieval_exhausted",
        "repeated_action",
        "agent_stopped",
    )


def test_test_run_category_values_are_stable() -> None:
    assert enum_values(RepoTestRunCategory) == (
        "passed",
        "assertion_failure",
        "test_error",
        "import_error",
        "no_tests_collected",
        "output_truncated",
        "timeout",
        "policy_denied",
        "sandbox_error",
        "execution_error",
        "unknown_nonzero",
    )


def test_verifier_error_code_values_are_stable() -> None:
    assert enum_values(VerifierErrorCode) == (
        "reproduction_not_attempted",
        "issue_not_reproduced",
        "reproduction_error",
        "reproduction_evidence_invalid",
        "patch_not_applied",
        "patch_evidence_invalid",
        "validation_not_run",
        "tests_failed",
        "validation_evidence_invalid",
        "test_command_mismatch",
        "observation_reused",
        "test_infrastructure_error",
    )


def test_runtime_error_values_are_stable() -> None:
    assert enum_values(ErrorType) == (
        "config_error",
        "validation_error",
        "provider_error",
        "model_format_error",
        "unknown_tool",
        "invalid_tool_arguments",
        "duplicate_tool_call",
        "policy_denied",
        "path_violation",
        "tool_execution_error",
        "tool_timeout",
        "sandbox_error",
        "retrieval_error",
        "patch_error",
        "test_error",
        "internal_error",
    )


def test_evaluation_error_categories_are_stable() -> None:
    assert enum_values(ErrorCategory) == (
        "retrieval_miss",
        "wrong_plan",
        "tool_misuse",
        "context_loss",
        "invalid_patch",
        "test_misread",
        "sandbox_failure",
        "budget_exhausted",
        "model_format_error",
    )


def test_dataset_split_values_are_stable() -> None:
    assert enum_values(DatasetSplit) == ("dev", "holdout")


def test_symbol_type_values_are_stable() -> None:
    assert enum_values(SymbolType) == ("module", "class", "function", "text")


def test_plan_step_status_values_are_stable() -> None:
    assert enum_values(PlanStepStatus) == (
        "pending",
        "in_progress",
        "completed",
        "blocked",
        "cancelled",
    )


def test_policy_decision_values_are_stable() -> None:
    assert enum_values(PolicyDecision) == ("allowed", "denied")


def test_run_status_values_are_stable() -> None:
    assert enum_values(RunStatus) == ("success", "failed", "stopped")


def test_patch_status_values_are_stable() -> None:
    assert enum_values(PatchStatus) == ("none", "proposed", "applied", "rejected")


def test_patch_error_code_values_are_stable() -> None:
    assert enum_values(PatchErrorCode) == (
        "invalid_diff",
        "patch_too_large",
        "too_many_files",
        "too_many_hunks",
        "too_many_changed_lines",
        "path_denied",
        "unsupported_patch",
        "artifact_mismatch",
        "apply_conflict",
        "rollback_failed",
        "inspect_unavailable",
        "sandbox_error",
    )


def test_test_status_values_are_stable() -> None:
    assert enum_values(RepoTestStatus) == (
        "not_run",
        "running",
        "passed",
        "failed",
        "error",
    )


def test_reproduction_status_values_are_stable() -> None:
    assert enum_values(ReproductionStatus) == (
        "not_attempted",
        "reproduced",
        "not_reproduced",
        "error",
    )


def test_validation_status_values_are_stable() -> None:
    assert enum_values(ValidationStatus) == ("not_run", "passed", "failed", "error")


def test_provider_action_kind_values_are_stable() -> None:
    assert enum_values(ProviderActionKind) == ("tool_call", "finish")


def test_sandbox_operation_values_are_stable() -> None:
    assert enum_values(SandboxOperation) == ("run", "read_file", "apply_patch")


def test_sandbox_lifecycle_values_are_stable() -> None:
    assert enum_values(SandboxLifecycle) == ("new", "prepared", "destroyed")


def test_cli_exit_code_values_are_stable() -> None:
    assert tuple(int(member) for member in CliExitCode) == (0, 1, 2, 3, 4, 5)


def test_repository_entry_type_values_are_stable() -> None:
    assert enum_values(RepositoryEntryType) == ("file", "directory", "symlink")


def test_repository_error_code_values_are_stable() -> None:
    assert enum_values(RepositoryErrorCode) == (
        "invalid_path",
        "outside_root",
        "symlink_forbidden",
        "denied_path",
        "not_found",
        "not_file",
        "not_directory",
        "binary_file",
        "file_too_large",
        "decode_error",
        "io_error",
    )


def test_repository_ingest_error_code_values_are_stable() -> None:
    assert enum_values(RepositoryIngestErrorCode) == (
        "listing_truncated",
        "source_changed",
        "io_error",
        "invalid_document",
        "unsupported_version",
        "integrity_mismatch",
    )


def test_tool_result_status_values_are_stable() -> None:
    assert enum_values(ToolResultStatus) == ("success", "error")


def test_tool_error_code_values_are_stable() -> None:
    assert enum_values(ToolErrorCode) == (
        "unknown_tool",
        "invalid_arguments",
        "duplicate_call",
        "output_too_large",
        "repository_error",
        "tool_execution_error",
    )


def test_symbol_kind_values_are_stable() -> None:
    assert enum_values(SymbolKind) == (
        "function",
        "async_function",
        "class",
        "method",
        "async_method",
    )


def test_source_inspection_error_code_values_are_stable() -> None:
    assert enum_values(SourceInspectionErrorCode) == ("partial_line", "syntax_error")


def test_python_chunk_mode_values_are_stable() -> None:
    assert enum_values(PythonChunkMode) == ("ast", "syntax_fallback", "empty")


def test_python_chunk_error_code_values_are_stable() -> None:
    assert enum_values(PythonChunkErrorCode) == (
        "source_unavailable",
        "source_changed",
    )


def test_text_source_kind_values_are_stable() -> None:
    assert enum_values(TextSourceKind) == ("markdown", "text", "config", "issue")


def test_text_chunk_error_code_values_are_stable() -> None:
    assert enum_values(TextChunkErrorCode) == (
        "source_unavailable",
        "source_changed",
    )


def test_chunk_corpus_error_code_values_are_stable() -> None:
    assert enum_values(ChunkCorpusErrorCode) == (
        "io_error",
        "invalid_document",
        "unsupported_version",
        "mixed_corpus",
        "duplicate_chunk",
        "unsorted_corpus",
    )


def test_bm25_error_code_values_are_stable() -> None:
    assert enum_values(BM25ErrorCode) == (
        "empty_query",
        "invalid_query",
        "identity_mismatch",
        "io_error",
        "invalid_document",
        "unsupported_version",
        "integrity_mismatch",
    )


def test_localization_eval_error_code_values_are_stable() -> None:
    assert enum_values(LocalizationEvalErrorCode) == (
        "input_io_error",
        "output_io_error",
        "invalid_manifest",
        "unsupported_version",
        "path_violation",
        "identity_mismatch",
        "retrieval_failure",
    )


def test_vector_store_error_code_values_are_stable() -> None:
    assert enum_values(VectorStoreErrorCode) == (
        "not_initialized",
        "invalid_record",
        "duplicate_record",
        "namespace_mismatch",
        "dimension_mismatch",
        "connection_error",
        "extension_unavailable",
        "migration_error",
        "schema_mismatch",
        "query_error",
    )


def test_embedding_error_code_values_are_stable() -> None:
    assert enum_values(EmbeddingErrorCode) == (
        "invalid_input",
        "identity_mismatch",
        "provider_unavailable",
        "remote_code_disabled",
        "provider_failure",
        "retry_exhausted",
        "invalid_response",
        "cache_io_error",
        "cache_corrupt",
        "corpus_mismatch",
    )


def test_dense_retrieval_error_code_values_are_stable() -> None:
    assert enum_values(DenseRetrievalErrorCode) == (
        "empty_query",
        "invalid_query",
        "identity_mismatch",
        "embedding_failure",
        "storage_failure",
        "invalid_result",
    )


def test_dense_evaluation_error_code_values_are_stable() -> None:
    assert enum_values(DenseEvalErrorCode) == (
        "identity_mismatch",
        "retrieval_failure",
        "invalid_result",
    )


def test_hybrid_retrieval_error_code_values_are_stable() -> None:
    assert enum_values(HybridRetrievalErrorCode) == (
        "invalid_configuration",
        "invalid_query",
        "identity_mismatch",
        "sparse_failure",
        "dense_failure",
        "invalid_result",
    )


def test_query_rewrite_and_comparison_error_code_values_are_stable() -> None:
    assert enum_values(QueryRewriteErrorCode) == (
        "invalid_input",
        "identity_mismatch",
        "context_mismatch",
        "provider_unavailable",
        "provider_failure",
        "invalid_response",
    )
    assert enum_values(RetrievalComparisonErrorCode) == (
        "identity_mismatch",
        "invalid_rewrite",
        "retrieval_failure",
        "output_io_error",
        "invalid_result",
    )


def test_agent_state_error_code_values_are_stable() -> None:
    assert enum_values(AgentStateErrorCode) == (
        "invalid_thread_id",
        "thread_not_found",
        "thread_already_exists",
        "task_mismatch",
        "state_conflict",
        "checkpoint_error",
    )


def test_agent_loop_error_code_values_are_stable() -> None:
    assert enum_values(AgentLoopErrorCode) == (
        "invalid_state_transition",
        "provider_failure",
        "tool_failure",
        "unfinished_pending_action",
    )


def test_frozen_dataset_enum_values_are_stable() -> None:
    assert enum_values(DatasetCaseKind) == ("fixture", "github_issue")
    assert enum_values(DatasetSelectionDecision) == ("included", "excluded")
    assert enum_values(DatasetPreflightStatus) == (
        "metadata_verified",
        "baseline_failed_gold_passed",
    )
    assert enum_values(DatasetExclusionReason) == (
        "duplicate",
        "mutable_revision",
        "repository_unavailable",
        "attribution_missing",
        "license_unverified",
        "setup_unsupported",
        "baseline_not_reproducible",
        "gold_patch_invalid",
        "resource_limit",
        "secret_or_manual_state",
        "uncontrolled_network",
        "split_policy",
        "environment_incompatible",
    )
    assert enum_values(DatasetManifestErrorCode) == (
        "manifest_unavailable",
        "manifest_invalid",
        "unsupported_version",
        "integrity_mismatch",
        "identity_mismatch",
        "duplicate_case",
        "source_unverified",
        "attribution_missing",
        "license_missing",
        "split_violation",
        "gold_unavailable",
        "preflight_invalid",
        "path_unsafe",
        "input_io_error",
    )


def test_agent_failure_kind_values_are_stable() -> None:
    assert enum_values(AgentFailureKind) == (
        "empty_search",
        "tool_error",
        "patch_failure",
        "test_failure",
        "repeated_action",
        "provider_failure",
    )


def test_trace_node_and_error_values_are_stable() -> None:
    assert enum_values(TraceNode) == (
        "initialize_plan",
        "decide",
        "execute_tool",
        "observe",
        "finish",
        "terminate",
    )
    assert enum_values(TraceErrorCode) == (
        "io_error",
        "invalid_document",
        "unsupported_version",
        "integrity_mismatch",
        "event_order",
        "state_chain_mismatch",
        "correlation_mismatch",
        "final_state_mismatch",
    )
    assert enum_values(WorkingMemoryErrorCode) == (
        "invalid_snapshot",
        "namespace_mismatch",
        "commit_stale",
        "integrity_mismatch",
        "sequence_rollback",
        "snapshot_limit",
        "terminal_snapshot",
        "recovery_conflict",
        "path_unsafe",
        "unsupported_version",
        "io_error",
    )
