"""Versioned context provenance and token-accounting contracts."""

from __future__ import annotations

import hashlib
from typing import Annotated, Self
from uuid import UUID

from pydantic import Field, StringConstraints, field_validator, model_validator

from repopilot.enums import (
    ContextBudgetKind,
    ContextErrorCode,
    ContextExclusionReason,
    ContextSourceKind,
    ContextTransformKind,
    ContextTrustLevel,
)
from repopilot.schemas import GitSha, StrictModel, TaskId, VersionIdentifier

CONTEXT_ESTIMATOR_VERSION = "utf8-bytes-ceil-div-3-v1"
CONTEXT_POLICY_VERSION = "repopilot-context-policy-v1"
CONTEXT_SUMMARY_VERSION = "extractive-v1"
MAX_CONTEXT_ITEMS = 1_000
MAX_CONTEXT_ITEM_CHARS = 1_000_000
ContextItemId = Annotated[
    str,
    StringConstraints(pattern=r"^ctx_[0-9a-f]{64}$"),
]
ContextSourceId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_.:/-]{0,255}$"),
]


class ContextAccountingError(ValueError):
    """Safe typed failure while building a context window or usage ledger."""

    def __init__(self, code: ContextErrorCode) -> None:
        super().__init__(
            {
                ContextErrorCode.INVALID_LIMIT: "context character limit is invalid",
                ContextErrorCode.INVALID_NAMESPACE: "context namespace is invalid",
                ContextErrorCode.INVALID_ITEM: "context item is invalid",
                ContextErrorCode.LEDGER_MISMATCH: "context accounting ledger is inconsistent",
                ContextErrorCode.PROMPT_MISMATCH: "context snapshot does not match the prompt",
                ContextErrorCode.INVALID_POLICY: "context budget policy is invalid",
                ContextErrorCode.ESSENTIAL_OVERFLOW: "essential context exceeds its budget",
                ContextErrorCode.ARTIFACT_REQUIRED: "large context requires an artifact reference",
                ContextErrorCode.TRANSFORM_MISMATCH: "context transformation is inconsistent",
            }[code]
        )
        self.code = code


class ContextItem(StrictModel):
    """One bounded candidate with exact projected content and stable provenance."""

    item_id: ContextItemId
    source_id: ContextSourceId
    source_kind: ContextSourceKind
    trust_level: ContextTrustLevel
    task_id: TaskId
    run_id: UUID
    repository_commit: GitSha
    ordinal: int = Field(ge=0)
    priority: int = Field(ge=0, le=100)
    content: str = Field(min_length=1, max_length=MAX_CONTEXT_ITEM_CHARS, repr=False)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    character_count: int = Field(gt=0)
    byte_count: int = Field(gt=0)
    estimated_tokens: int = Field(gt=0)
    truncated: bool = False

    @model_validator(mode="after")
    def validate_content_accounting(self) -> Self:
        encoded = self.content.encode("utf-8")
        if self.character_count != len(self.content):
            raise ValueError("context character count does not match content")
        if self.byte_count != len(encoded):
            raise ValueError("context byte count does not match content")
        if self.content_sha256 != hashlib.sha256(encoded).hexdigest():
            raise ValueError("context content digest does not match content")
        if self.estimated_tokens != estimate_utf8_tokens_from_bytes(self.byte_count):
            raise ValueError("context token estimate does not match estimator")
        expected_id = context_item_id(
            task_id=self.task_id,
            run_id=self.run_id,
            repository_commit=self.repository_commit,
            source_kind=self.source_kind,
            source_id=self.source_id,
            content_sha256=self.content_sha256,
        )
        if self.item_id != expected_id:
            raise ValueError("context item ID does not match provenance")
        return self


class ContextExclusion(StrictModel):
    """One candidate omitted from the final context selection."""

    item_id: ContextItemId
    reason: ContextExclusionReason
    replacement_item_id: ContextItemId | None = None

    @model_validator(mode="after")
    def validate_replacement(self) -> Self:
        replacement_reasons = {
            ContextExclusionReason.DUPLICATE,
            ContextExclusionReason.SUMMARY_REPLACEMENT,
            ContextExclusionReason.ARTIFACT_OFFLOAD,
            ContextExclusionReason.SUPERSEDED,
        }
        if (self.replacement_item_id is not None) != (self.reason in replacement_reasons):
            raise ValueError("context replacement must match its exclusion reason")
        if self.replacement_item_id == self.item_id:
            raise ValueError("context replacement must reference another item")
        return self


class ContextBudgetPolicy(StrictModel):
    """Versioned independent source caps plus one global context-item cap."""

    policy_version: VersionIdentifier = CONTEXT_POLICY_VERSION
    max_estimated_tokens: int = Field(default=18_000, ge=256, le=100_000, strict=True)
    control_tokens: int = Field(default=4_000, ge=64, le=100_000, strict=True)
    task_tokens: int = Field(default=5_000, ge=64, le=100_000, strict=True)
    plan_tokens: int = Field(default=6_000, ge=64, le=100_000, strict=True)
    retrieval_tokens: int = Field(default=6_000, ge=64, le=100_000, strict=True)
    tool_tokens: int = Field(default=6_000, ge=64, le=100_000, strict=True)
    memory_tokens: int = Field(default=3_000, ge=64, le=100_000, strict=True)
    max_inline_tool_tokens: int = Field(default=2_500, ge=32, le=100_000, strict=True)
    summary_target_tokens: int = Field(default=768, ge=32, le=100_000, strict=True)
    max_summary_sources: int = Field(default=16, ge=2, le=64, strict=True)

    @field_validator("policy_version")
    @classmethod
    def require_supported_policy(cls, value: str) -> str:
        if value != CONTEXT_POLICY_VERSION:
            raise ValueError("context policy version is unsupported")
        return value

    @model_validator(mode="after")
    def validate_caps(self) -> Self:
        caps = (
            self.control_tokens,
            self.task_tokens,
            self.plan_tokens,
            self.retrieval_tokens,
            self.tool_tokens,
            self.memory_tokens,
        )
        if any(cap > self.max_estimated_tokens for cap in caps):
            raise ValueError("context source budget cannot exceed global budget")
        if self.max_inline_tool_tokens > self.tool_tokens:
            raise ValueError("inline tool limit cannot exceed tool budget")
        if self.summary_target_tokens > self.memory_tokens:
            raise ValueError("summary target cannot exceed memory budget")
        return self

    def budget_for(self, kind: ContextBudgetKind) -> int:
        """Return the frozen cap for one complete budget category."""

        return {
            ContextBudgetKind.CONTROL: self.control_tokens,
            ContextBudgetKind.TASK: self.task_tokens,
            ContextBudgetKind.PLAN: self.plan_tokens,
            ContextBudgetKind.RETRIEVAL: self.retrieval_tokens,
            ContextBudgetKind.TOOL: self.tool_tokens,
            ContextBudgetKind.MEMORY: self.memory_tokens,
        }[kind]


class ContextBudgetUsage(StrictModel):
    """Before/after token usage and selection counts for one source pool."""

    budget_kind: ContextBudgetKind
    token_limit: int = Field(ge=0, strict=True)
    candidate_tokens: int = Field(ge=0, strict=True)
    selected_tokens: int = Field(ge=0, strict=True)
    candidate_count: int = Field(ge=0, strict=True)
    selected_count: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_usage(self) -> Self:
        if self.selected_tokens > self.candidate_tokens:
            raise ValueError("selected source tokens cannot exceed candidates")
        if self.selected_tokens > self.token_limit:
            raise ValueError("selected source tokens cannot exceed source budget")
        if self.selected_count > self.candidate_count:
            raise ValueError("selected source count cannot exceed candidates")
        return self


class ContextTransformRecord(StrictModel):
    """Provenance for deduplication, summary replacement, or artifact projection."""

    kind: ContextTransformKind
    transform_version: VersionIdentifier
    source_item_ids: tuple[ContextItemId, ...] = Field(min_length=1, max_length=64)
    output_item_id: ContextItemId
    source_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("source_item_ids")
    @classmethod
    def require_unique_sources(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("context transform sources must be unique")
        return value


class ContextWindowSnapshot(StrictModel):
    """Complete candidate partition and before/after accounting for one decision."""

    request_id: UUID
    task_id: TaskId
    run_id: UUID
    repository_commit: GitSha
    prompt_version: VersionIdentifier
    estimator_version: VersionIdentifier = CONTEXT_ESTIMATOR_VERSION
    max_prompt_chars: int = Field(ge=4_096, le=200_000)
    items: tuple[ContextItem, ...] = Field(min_length=1, max_length=MAX_CONTEXT_ITEMS)
    included_item_ids: tuple[ContextItemId, ...] = Field(min_length=1)
    exclusions: tuple[ContextExclusion, ...] = ()
    policy: ContextBudgetPolicy | None = None
    budget_usages: tuple[ContextBudgetUsage, ...] = ()
    transforms: tuple[ContextTransformRecord, ...] = ()
    context_chars_before: int = Field(gt=0)
    context_bytes_before: int = Field(gt=0)
    estimated_tokens_before: int = Field(gt=0)
    context_chars_after: int = Field(gt=0)
    context_bytes_after: int = Field(gt=0)
    estimated_tokens_after: int = Field(gt=0)
    rendered_prompt_chars: int = Field(gt=0)
    rendered_prompt_bytes: int = Field(gt=0)
    rendered_prompt_estimated_tokens: int = Field(gt=0)
    rendered_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fits_character_budget: bool

    @field_validator("estimator_version")
    @classmethod
    def require_supported_estimator(cls, value: str) -> str:
        if value != CONTEXT_ESTIMATOR_VERSION:
            raise ValueError("context estimator version is unsupported")
        return value

    @model_validator(mode="after")
    def validate_window_accounting(self) -> Self:
        ids = tuple(item.item_id for item in self.items)
        if len(set(ids)) != len(ids):
            raise ValueError("context item IDs must be unique")
        if tuple(item.ordinal for item in self.items) != tuple(range(len(self.items))):
            raise ValueError("context item ordinals must be contiguous")
        if any(
            item.task_id != self.task_id
            or item.run_id != self.run_id
            or item.repository_commit != self.repository_commit
            for item in self.items
        ):
            raise ValueError("context items must share the window namespace")

        included = self.included_item_ids
        excluded = tuple(item.item_id for item in self.exclusions)
        if len(set(included)) != len(included) or len(set(excluded)) != len(excluded):
            raise ValueError("context selection IDs must be unique")
        if set(included).intersection(excluded) or set(included).union(excluded) != set(ids):
            raise ValueError("included and excluded items must partition candidates")
        if included != tuple(item_id for item_id in ids if item_id in set(included)):
            raise ValueError("included context IDs must preserve candidate order")
        if excluded != tuple(item_id for item_id in ids if item_id in set(excluded)):
            raise ValueError("excluded context IDs must preserve candidate order")

        by_id = {item.item_id: item for item in self.items}
        exclusion_by_id = {exclusion.item_id: exclusion for exclusion in self.exclusions}
        transform_outputs: set[str] = set()
        transformed_sources: set[str] = set()
        for transform in self.transforms:
            if transform.output_item_id not in by_id or any(
                source not in by_id or source not in excluded
                for source in transform.source_item_ids
            ):
                raise ValueError(
                    "context transform must connect excluded inputs to included output"
                )
            if transform.output_item_id in transform.source_item_ids:
                raise ValueError("context transform output must differ from its inputs")
            expected_digest = transform_source_digest(
                tuple(by_id[source].content_sha256 for source in transform.source_item_ids)
            )
            if transform.source_content_sha256 != expected_digest:
                raise ValueError("context transform source digest is invalid")
            if transform.output_item_id in transform_outputs:
                raise ValueError("context transform outputs must be unique")
            transform_outputs.add(transform.output_item_id)
            for source in transform.source_item_ids:
                if exclusion_by_id[source].replacement_item_id != transform.output_item_id:
                    raise ValueError("context exclusion replacement must match transform output")
                transformed_sources.add(source)
        replaced_sources = {
            exclusion.item_id
            for exclusion in self.exclusions
            if exclusion.replacement_item_id is not None
        }
        if transformed_sources != replaced_sources:
            raise ValueError("context replacements must have complete transform provenance")

        if self.policy is None:
            if self.budget_usages or self.transforms:
                raise ValueError("context policy is required for selection details")
        else:
            expected_kinds = tuple(ContextBudgetKind)
            if tuple(usage.budget_kind for usage in self.budget_usages) != expected_kinds:
                raise ValueError("context budget usage must cover every source pool in order")
            if any(
                usage.token_limit != self.policy.budget_for(usage.budget_kind)
                for usage in self.budget_usages
            ):
                raise ValueError("context budget usage limit does not match policy")
            if (
                sum(usage.selected_tokens for usage in self.budget_usages)
                > self.policy.max_estimated_tokens
            ):
                raise ValueError("selected context items exceed the global token budget")

        if self.context_chars_after > self.context_chars_before:
            raise ValueError("selected context characters cannot exceed candidates")
        if self.context_bytes_after > self.context_bytes_before:
            raise ValueError("selected context bytes cannot exceed candidates")
        if self.estimated_tokens_after > self.estimated_tokens_before:
            raise ValueError("selected context tokens cannot exceed candidates")
        if self.estimated_tokens_before != estimate_utf8_tokens_from_bytes(
            self.context_bytes_before
        ):
            raise ValueError("candidate context token total is invalid")
        if self.estimated_tokens_after != estimate_utf8_tokens_from_bytes(self.context_bytes_after):
            raise ValueError("selected context token total is invalid")
        if self.rendered_prompt_estimated_tokens != estimate_utf8_tokens_from_bytes(
            self.rendered_prompt_bytes
        ):
            raise ValueError("rendered prompt token estimate is invalid")
        if self.context_chars_after < self.rendered_prompt_chars:
            raise ValueError("selected context must contain the rendered prompt")
        if self.context_bytes_after < self.rendered_prompt_bytes:
            raise ValueError("selected context must contain the rendered prompt")
        if self.fits_character_budget != (self.rendered_prompt_chars <= self.max_prompt_chars):
            raise ValueError("context fit flag does not match rendered prompt size")
        return self


class ContextUsageLedger(StrictModel):
    """Reconcile one deterministic estimate with Provider-reported input usage."""

    request_id: UUID
    rendered_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimator_version: VersionIdentifier = CONTEXT_ESTIMATOR_VERSION
    estimated_input_tokens: int = Field(gt=0)
    provider_input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    cache_write_input_tokens: int = Field(ge=0)
    estimate_delta_tokens: int
    attempts: int = Field(ge=1, le=6)

    @field_validator("estimator_version")
    @classmethod
    def require_supported_estimator(cls, value: str) -> str:
        if value != CONTEXT_ESTIMATOR_VERSION:
            raise ValueError("context estimator version is unsupported")
        return value

    @model_validator(mode="after")
    def validate_usage(self) -> Self:
        if self.cached_input_tokens + self.cache_write_input_tokens > (self.provider_input_tokens):
            raise ValueError("context cache token detail exceeds Provider input")
        if self.estimate_delta_tokens != (self.provider_input_tokens - self.estimated_input_tokens):
            raise ValueError("context token estimate delta is inconsistent")
        return self


def estimate_utf8_tokens(value: str) -> int:
    """Return the frozen, dependency-free conservative context estimate."""

    return estimate_utf8_tokens_from_bytes(len(value.encode("utf-8")))


def estimate_utf8_tokens_from_bytes(byte_count: int) -> int:
    """Estimate tokens as ceil(UTF-8 bytes / 3), with zero preserved."""

    if isinstance(byte_count, bool) or byte_count < 0:
        raise ContextAccountingError(ContextErrorCode.INVALID_ITEM)
    return 0 if byte_count == 0 else (byte_count + 2) // 3


def transform_source_digest(content_digests: tuple[str, ...]) -> str:
    """Hash an ordered set of source-content digests for transform provenance."""

    if not content_digests:
        raise ContextAccountingError(ContextErrorCode.TRANSFORM_MISMATCH)
    if any(
        len(digest) != 64
        or digest.casefold() != digest
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in content_digests
    ):
        raise ContextAccountingError(ContextErrorCode.TRANSFORM_MISMATCH)
    return hashlib.sha256("\x00".join(content_digests).encode("ascii")).hexdigest()


def context_item_id(
    *,
    task_id: str,
    run_id: UUID,
    repository_commit: str,
    source_kind: ContextSourceKind,
    source_id: str,
    content_sha256: str,
) -> str:
    """Hash namespace, source identity, and projected content into one stable ID."""

    framed = "\x00".join(
        (
            task_id,
            str(run_id),
            repository_commit,
            source_kind.value,
            source_id,
            content_sha256,
        )
    ).encode("utf-8")
    return f"ctx_{hashlib.sha256(framed).hexdigest()}"


def build_context_item(
    *,
    source_id: str,
    source_kind: ContextSourceKind,
    trust_level: ContextTrustLevel,
    task_id: str,
    run_id: UUID,
    repository_commit: str,
    ordinal: int,
    priority: int,
    content: str,
    truncated: bool = False,
) -> ContextItem:
    """Build a self-validating item from already-redacted projected content."""

    encoded = content.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    try:
        return ContextItem(
            item_id=context_item_id(
                task_id=task_id,
                run_id=run_id,
                repository_commit=repository_commit,
                source_kind=source_kind,
                source_id=source_id,
                content_sha256=digest,
            ),
            source_id=source_id,
            source_kind=source_kind,
            trust_level=trust_level,
            task_id=task_id,
            run_id=run_id,
            repository_commit=repository_commit,
            ordinal=ordinal,
            priority=priority,
            content=content,
            content_sha256=digest,
            character_count=len(content),
            byte_count=len(encoded),
            estimated_tokens=estimate_utf8_tokens_from_bytes(len(encoded)),
            truncated=truncated,
        )
    except ValueError:
        raise ContextAccountingError(ContextErrorCode.INVALID_ITEM) from None


__all__ = [
    "CONTEXT_ESTIMATOR_VERSION",
    "CONTEXT_POLICY_VERSION",
    "CONTEXT_SUMMARY_VERSION",
    "MAX_CONTEXT_ITEMS",
    "ContextAccountingError",
    "ContextBudgetPolicy",
    "ContextBudgetUsage",
    "ContextExclusion",
    "ContextItem",
    "ContextItemId",
    "ContextSourceId",
    "ContextTransformRecord",
    "ContextUsageLedger",
    "ContextWindowSnapshot",
    "build_context_item",
    "context_item_id",
    "estimate_utf8_tokens",
    "estimate_utf8_tokens_from_bytes",
    "transform_source_digest",
]
