"""Frozen, gold-isolated execution adapters for the P7 baseline arms."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self, cast

import yaml
from pydantic import Field, JsonValue, ValidationError, model_validator
from pydantic_core import to_jsonable_python

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    BaselineErrorCode,
    BaselineKind,
    ModelProvider,
    ReasoningEffort,
    RepairContextMode,
    RetrievalMode,
    SandboxNetworkMode,
)
from repopilot.evals.runner import (
    EvaluationExecutionEvidence,
    EvaluationExecutionRequest,
    ProviderResponsesFactory,
    RepairWorkflowEvaluationExecutor,
)
from repopilot.repair_workflow import RepairSandboxFactory
from repopilot.schemas import (
    Identifier,
    ModelIdentifier,
    Sha256,
    StrictModel,
)

BASELINE_CONFIGURATION_SCHEMA_VERSION: Literal[1] = 1
BASELINE_ADAPTER_VERSION: Literal["baseline-adapter-v1"] = "baseline-adapter-v1"
BASELINE_SUCCESS_CRITERION: Literal["rule-based-verifier-v1"] = "rule-based-verifier-v1"
BASELINE_TOOL_NAMES = (
    "apply_patch",
    "inspect_diff",
    "inspect_symbol",
    "list_files",
    "read_file",
    "run_tests",
    "search_code",
)
_MAX_BASELINE_CONFIGURATION_BYTES = 64 * 1024
_IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:~-]{0,439}@sha256:[0-9a-f]{64}$")
_EVALUATOR_ONLY_KEYS = frozenset(
    {
        "expected_patch",
        "gold",
        "gold_files",
        "gold_patch",
        "oracle",
        "test_patch",
    }
)


BASELINE_ERROR_MESSAGES: Mapping[BaselineErrorCode, str] = {
    BaselineErrorCode.INPUT_IO_ERROR: "baseline configuration is unavailable",
    BaselineErrorCode.INVALID_CONFIGURATION: "baseline configuration is invalid",
    BaselineErrorCode.UNSUPPORTED_VERSION: "baseline configuration version is unsupported",
    BaselineErrorCode.CAPABILITY_MISMATCH: "baseline capabilities do not match the frozen arm",
    BaselineErrorCode.FAIRNESS_VIOLATION: "baseline runtime violates the fairness contract",
    BaselineErrorCode.INFORMATION_BOUNDARY: (
        "baseline configuration crosses the evaluator information boundary"
    ),
}


class BaselineError(ValueError):
    """Safe typed baseline failure that never echoes configuration values."""

    def __init__(self, code: BaselineErrorCode) -> None:
        super().__init__(BASELINE_ERROR_MESSAGES[code])
        self.code = code


class BaselineDefinition(StrictModel):
    """Mechanism-only configuration; model and budgets come from shared settings."""

    schema_version: Literal[1] = BASELINE_CONFIGURATION_SCHEMA_VERSION
    adapter_version: Literal["baseline-adapter-v1"] = BASELINE_ADAPTER_VERSION
    baseline_id: Identifier
    kind: BaselineKind
    context_mode: RepairContextMode
    retrieval_mode: RetrievalMode | None
    query_rewrite_enabled: Literal[False]
    memory_enabled: Literal[False]
    arbitrary_shell_enabled: Literal[False]
    evaluator_patch_access: Literal["forbidden"]
    evaluator_gold_file_access: Literal["forbidden"]
    test_patch_access: Literal["evaluator_only"]
    tool_names: tuple[str, ...] = Field(min_length=1, max_length=32)
    success_criterion: Literal["rule-based-verifier-v1"] = BASELINE_SUCCESS_CRITERION

    @model_validator(mode="after")
    def validate_frozen_capabilities(self) -> Self:
        if self.tool_names != BASELINE_TOOL_NAMES:
            raise ValueError("baseline tool capabilities do not match")
        expected = {
            BaselineKind.MINI_SWE_SAFE: (
                "mini-swe-safe-v1",
                RepairContextMode.NONE,
                None,
            ),
            BaselineKind.NO_SEMANTIC_RAG: (
                "no-semantic-rag-v1",
                RepairContextMode.BM25,
                RetrievalMode.BM25,
            ),
        }[self.kind]
        if (
            self.baseline_id,
            self.context_mode,
            self.retrieval_mode,
        ) != expected:
            raise ValueError("baseline identity and retrieval capabilities do not match")
        return self


class BaselineFairnessContract(StrictModel):
    """All runtime fields that must stay identical between comparison arms."""

    model_provider: ModelProvider
    model: ModelIdentifier
    temperature: float = Field(ge=0.0, le=2.0)
    reasoning_effort: ReasoningEffort
    max_output_tokens: int = Field(gt=0, strict=True)
    max_steps: int = Field(gt=0, strict=True)
    max_replans: int = Field(ge=0, strict=True)
    max_tokens: int = Field(gt=0, strict=True)
    timeout_seconds: int = Field(gt=0, strict=True)
    max_cost_usd: Decimal = Field(gt=0)
    sandbox_image_reference: str = Field(min_length=1, max_length=512)
    sandbox_network: SandboxNetworkMode
    sandbox_cpu_limit: float = Field(gt=0)
    sandbox_memory_limit_mb: int = Field(gt=0, strict=True)
    sandbox_pids_limit: int = Field(gt=0, strict=True)
    sandbox_tmpfs_limit_mb: int = Field(gt=0, strict=True)
    sandbox_command_timeout_seconds: int = Field(gt=0, strict=True)
    sandbox_max_output_bytes: int = Field(gt=0, strict=True)
    sandbox_max_artifact_bytes: int = Field(gt=0, strict=True)
    repository_max_file_bytes: int = Field(gt=0, strict=True)
    repository_max_read_bytes: int = Field(gt=0, strict=True)
    repository_max_read_lines: int = Field(gt=0, strict=True)
    repository_max_list_entries: int = Field(gt=0, strict=True)
    tool_max_output_bytes: int = Field(gt=0, strict=True)
    search_max_matches: int = Field(gt=0, strict=True)
    search_max_query_length: int = Field(gt=0, strict=True)
    patch_max_bytes: int = Field(gt=0, strict=True)
    patch_max_files: int = Field(gt=0, strict=True)
    patch_max_hunks: int = Field(gt=0, strict=True)
    patch_max_changed_lines: int = Field(gt=0, strict=True)
    tool_names: tuple[str, ...]
    success_criterion: Literal["rule-based-verifier-v1"] = BASELINE_SUCCESS_CRITERION
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_fairness_identity(self) -> Self:
        if _IMAGE_PATTERN.fullmatch(self.sandbox_image_reference) is None:
            raise ValueError("baseline Sandbox image must use an immutable digest")
        if self.tool_names != BASELINE_TOOL_NAMES:
            raise ValueError("baseline fairness tool set does not match")
        payload = self.model_dump(mode="python", exclude={"content_sha256"})
        if _canonical_sha256(payload) != self.content_sha256:
            raise ValueError("baseline fairness digest does not match")
        return self


class BaselineRuntimeContract(StrictModel):
    """One fully bound arm, including its comparable and arm-specific identities."""

    schema_version: Literal[1] = BASELINE_CONFIGURATION_SCHEMA_VERSION
    adapter_version: Literal["baseline-adapter-v1"] = BASELINE_ADAPTER_VERSION
    baseline_id: Identifier
    kind: BaselineKind
    definition_sha256: Sha256
    context_mode: RepairContextMode
    retrieval_mode: RetrievalMode | None
    query_rewrite_enabled: Literal[False]
    memory_enabled: Literal[False]
    arbitrary_shell_enabled: Literal[False]
    fairness: BaselineFairnessContract
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_runtime_identity(self) -> Self:
        payload = self.model_dump(mode="python", exclude={"content_sha256"})
        if _canonical_sha256(payload) != self.content_sha256:
            raise ValueError("baseline runtime digest does not match")
        return self


@dataclass(frozen=True)
class BaselineEvaluationExecutor:
    """Bind one frozen baseline to the Agent-only P7 evaluation executor input."""

    definition: BaselineDefinition
    settings: RepoPilotSettings
    sandbox_image_reference: str
    evaluation_configuration_sha256: Sha256 | None = None
    provider_responses_factory: ProviderResponsesFactory | None = None
    sandbox_factory: RepairSandboxFactory | None = None

    def __post_init__(self) -> None:
        build_baseline_runtime_contract(
            self.definition,
            settings=self.settings,
            sandbox_image_reference=self.sandbox_image_reference,
        )

    @property
    def runtime_contract(self) -> BaselineRuntimeContract:
        return build_baseline_runtime_contract(
            self.definition,
            settings=self.settings,
            sandbox_image_reference=self.sandbox_image_reference,
        )

    def execute(self, request: EvaluationExecutionRequest) -> EvaluationExecutionEvidence:
        """Execute without accepting Memory stores or evaluator-owned task fields."""

        contract = self.runtime_contract
        return RepairWorkflowEvaluationExecutor(
            settings=self.settings,
            sandbox_image_reference=self.sandbox_image_reference,
            context_mode=self.definition.context_mode,
            evaluation_configuration_sha256=(
                self.evaluation_configuration_sha256 or baseline_configuration_sha256(contract)
            ),
            provider_responses_factory=self.provider_responses_factory,
            sandbox_factory=self.sandbox_factory,
            memory_stores=None,
        ).execute(request)


def load_baseline_definition(path: Path) -> BaselineDefinition:
    """Load one bounded strict YAML definition without exposing rejected content."""

    try:
        if path.is_symlink() or not path.is_file():
            raise OSError
        content = path.read_bytes()
    except OSError:
        raise BaselineError(BaselineErrorCode.INPUT_IO_ERROR) from None
    if not content or len(content) > _MAX_BASELINE_CONFIGURATION_BYTES:
        raise BaselineError(BaselineErrorCode.INPUT_IO_ERROR)
    try:
        loaded = yaml.safe_load(content)
    except (UnicodeError, yaml.YAMLError):
        raise BaselineError(BaselineErrorCode.INVALID_CONFIGURATION) from None
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise BaselineError(BaselineErrorCode.INVALID_CONFIGURATION)
    if _contains_evaluator_only_key(cast(Mapping[str, object], loaded)):
        raise BaselineError(BaselineErrorCode.INFORMATION_BOUNDARY)
    if loaded.get("schema_version") != BASELINE_CONFIGURATION_SCHEMA_VERSION:
        raise BaselineError(BaselineErrorCode.UNSUPPORTED_VERSION)
    try:
        return BaselineDefinition.model_validate(loaded)
    except ValidationError:
        raise BaselineError(BaselineErrorCode.CAPABILITY_MISMATCH) from None


def build_baseline_runtime_contract(
    definition: BaselineDefinition,
    *,
    settings: RepoPilotSettings,
    sandbox_image_reference: str,
) -> BaselineRuntimeContract:
    """Bind mechanism settings to the shared model, budget, Sandbox, and Verifier."""

    try:
        validated_definition = BaselineDefinition.model_validate(
            definition.model_dump(mode="python")
        )
    except ValidationError:
        raise BaselineError(BaselineErrorCode.INVALID_CONFIGURATION) from None
    if settings.query_rewrite_enabled:
        raise BaselineError(BaselineErrorCode.CAPABILITY_MISMATCH)
    if (
        validated_definition.kind is BaselineKind.NO_SEMANTIC_RAG
        and settings.retrieval_mode is not RetrievalMode.BM25
    ):
        raise BaselineError(BaselineErrorCode.CAPABILITY_MISMATCH)
    if _IMAGE_PATTERN.fullmatch(sandbox_image_reference) is None:
        raise BaselineError(BaselineErrorCode.FAIRNESS_VIOLATION)
    fairness_payload: dict[str, object] = {
        "model_provider": settings.model_provider,
        "model": settings.model,
        "temperature": settings.temperature,
        "reasoning_effort": settings.openai_reasoning_effort,
        "max_output_tokens": settings.openai_max_output_tokens,
        "max_steps": settings.max_steps,
        "max_replans": settings.max_replans,
        "max_tokens": settings.max_tokens,
        "timeout_seconds": settings.timeout_seconds,
        "max_cost_usd": settings.max_cost_usd,
        "sandbox_image_reference": sandbox_image_reference,
        "sandbox_network": settings.sandbox_network,
        "sandbox_cpu_limit": settings.sandbox_cpu_limit,
        "sandbox_memory_limit_mb": settings.sandbox_memory_limit_mb,
        "sandbox_pids_limit": settings.sandbox_pids_limit,
        "sandbox_tmpfs_limit_mb": settings.sandbox_tmpfs_limit_mb,
        "sandbox_command_timeout_seconds": settings.sandbox_command_timeout_seconds,
        "sandbox_max_output_bytes": settings.sandbox_max_output_bytes,
        "sandbox_max_artifact_bytes": settings.sandbox_max_artifact_bytes,
        "repository_max_file_bytes": settings.repository_max_file_bytes,
        "repository_max_read_bytes": settings.repository_max_read_bytes,
        "repository_max_read_lines": settings.repository_max_read_lines,
        "repository_max_list_entries": settings.repository_max_list_entries,
        "tool_max_output_bytes": settings.tool_max_output_bytes,
        "search_max_matches": settings.search_max_matches,
        "search_max_query_length": settings.search_max_query_length,
        "patch_max_bytes": settings.patch_max_bytes,
        "patch_max_files": settings.patch_max_files,
        "patch_max_hunks": settings.patch_max_hunks,
        "patch_max_changed_lines": settings.patch_max_changed_lines,
        "tool_names": BASELINE_TOOL_NAMES,
        "success_criterion": BASELINE_SUCCESS_CRITERION,
    }
    fairness = BaselineFairnessContract.model_validate(
        {
            **fairness_payload,
            "content_sha256": _canonical_sha256(fairness_payload),
        }
    )
    definition_sha256 = _canonical_sha256(validated_definition.model_dump(mode="python"))
    payload: dict[str, object] = {
        "schema_version": BASELINE_CONFIGURATION_SCHEMA_VERSION,
        "adapter_version": BASELINE_ADAPTER_VERSION,
        "baseline_id": validated_definition.baseline_id,
        "kind": validated_definition.kind,
        "definition_sha256": definition_sha256,
        "context_mode": validated_definition.context_mode,
        "retrieval_mode": validated_definition.retrieval_mode,
        "query_rewrite_enabled": validated_definition.query_rewrite_enabled,
        "memory_enabled": validated_definition.memory_enabled,
        "arbitrary_shell_enabled": validated_definition.arbitrary_shell_enabled,
        "fairness": fairness,
    }
    return BaselineRuntimeContract.model_validate(
        {
            **payload,
            "content_sha256": _canonical_sha256(payload),
        }
    )


def baseline_configuration_snapshot(
    contract: BaselineRuntimeContract,
) -> dict[str, JsonValue]:
    """Return a secret-free runner configuration snapshot for this arm."""

    return cast(
        dict[str, JsonValue],
        to_jsonable_python(contract.model_dump(mode="python")),
    )


def baseline_configuration_sha256(contract: BaselineRuntimeContract) -> str:
    """Hash the exact baseline snapshot persisted by the evaluation runner."""

    return _canonical_sha256(baseline_configuration_snapshot(contract))


def require_comparable_baselines(contracts: Sequence[BaselineRuntimeContract]) -> str:
    """Return the shared fairness digest or reject model/budget/Sandbox drift."""

    if len(contracts) < 2:
        raise BaselineError(BaselineErrorCode.FAIRNESS_VIOLATION)
    digests = {contract.fairness.content_sha256 for contract in contracts}
    if len(digests) != 1:
        raise BaselineError(BaselineErrorCode.FAIRNESS_VIOLATION)
    return next(iter(digests))


def _contains_evaluator_only_key(value: Mapping[str, object]) -> bool:
    for key, item in value.items():
        if key.casefold() in _EVALUATOR_ONLY_KEYS:
            return True
        if isinstance(item, Mapping) and _contains_evaluator_only_key(
            cast(Mapping[str, object], item)
        ):
            return True
        if isinstance(item, list):
            for nested in item:
                if isinstance(nested, Mapping) and _contains_evaluator_only_key(
                    cast(Mapping[str, object], nested)
                ):
                    return True
    return False


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
