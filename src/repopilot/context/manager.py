"""Deterministic source-budgeted context selection and safe transformations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from pydantic import JsonValue, ValidationError

from repopilot.context.models import (
    CONTEXT_SUMMARY_VERSION,
    ContextAccountingError,
    ContextBudgetPolicy,
    ContextBudgetUsage,
    ContextExclusion,
    ContextItem,
    ContextTransformRecord,
    build_context_item,
    estimate_utf8_tokens,
    transform_source_digest,
)
from repopilot.enums import (
    ContextBudgetKind,
    ContextErrorCode,
    ContextExclusionReason,
    ContextSourceKind,
    ContextTransformKind,
    ContextTrustLevel,
)
from repopilot.schemas import ArtifactRef

ARTIFACT_PROJECTION_VERSION = "artifact-projection-v1"
DEDUPLICATION_VERSION = "content-sha256-v1"
_DEDUPLICATED_KINDS = {
    ContextSourceKind.EPISODIC_MEMORY,
    ContextSourceKind.REPOSITORY_MEMORY,
    ContextSourceKind.RETRIEVAL,
    ContextSourceKind.WORKING_MEMORY,
}
_SUMMARIZED_KINDS = {
    ContextSourceKind.FAILURE,
    ContextSourceKind.TOOL_RESULT,
    ContextSourceKind.WORKING_MEMORY,
}
_TOOL_METADATA_KEYS = (
    "call_id",
    "tool_name",
    "status",
    "output_bytes",
    "truncated",
    "error_type",
    "error_code",
    "repository_error_code",
    "message",
)


@dataclass(frozen=True)
class ContextSelection:
    """Manager output consumed by Prompt rendering and the durable snapshot."""

    items: tuple[ContextItem, ...]
    included_item_ids: tuple[str, ...]
    exclusions: tuple[ContextExclusion, ...]
    transforms: tuple[ContextTransformRecord, ...]
    budget_usages: tuple[ContextBudgetUsage, ...]
    mandatory_item_ids: frozenset[str]


class ContextManager:
    """Select a minimum bounded evidence set without model or filesystem access."""

    def __init__(self, policy: ContextBudgetPolicy | None = None) -> None:
        self._policy = policy or ContextBudgetPolicy()

    @property
    def policy(self) -> ContextBudgetPolicy:
        return self._policy

    def select(
        self,
        items: tuple[ContextItem, ...],
        *,
        mandatory_item_ids: frozenset[str],
    ) -> ContextSelection:
        """Transform and select candidates under independent and global token caps."""

        self._validate_candidates(items, mandatory_item_ids)
        working = list(items)
        exclusions: dict[str, ContextExclusion] = {}
        transforms: list[ContextTransformRecord] = []
        mandatory = set(mandatory_item_ids)

        self._project_large_tool_items(
            working,
            mandatory,
            exclusions,
            transforms,
        )
        self._deduplicate(
            working,
            mandatory,
            exclusions,
            transforms,
        )

        included: set[str] = set()
        category_tokens = {kind: 0 for kind in ContextBudgetKind}
        total_tokens = 0
        by_id = {item.item_id: item for item in working}
        for item_id in mandatory:
            if item_id in exclusions:
                continue
            item = by_id[item_id]
            kind = context_budget_kind(item.source_kind)
            category_tokens[kind] += item.estimated_tokens
            total_tokens += item.estimated_tokens
            included.add(item_id)
        if total_tokens > self._policy.max_estimated_tokens or any(
            used > self._policy.budget_for(kind) for kind, used in category_tokens.items()
        ):
            raise ContextAccountingError(ContextErrorCode.ESSENTIAL_OVERFLOW)

        optional = sorted(
            (
                item
                for item in working
                if item.item_id not in mandatory and item.item_id not in exclusions
            ),
            key=lambda item: (-item.priority, -item.ordinal, item.item_id),
        )
        for item in optional:
            kind = context_budget_kind(item.source_kind)
            if category_tokens[kind] + item.estimated_tokens > self._policy.budget_for(kind):
                exclusions[item.item_id] = ContextExclusion(
                    item_id=item.item_id,
                    reason=ContextExclusionReason.SOURCE_BUDGET,
                )
                continue
            if total_tokens + item.estimated_tokens > self._policy.max_estimated_tokens:
                exclusions[item.item_id] = ContextExclusion(
                    item_id=item.item_id,
                    reason=ContextExclusionReason.GLOBAL_BUDGET,
                )
                continue
            included.add(item.item_id)
            category_tokens[kind] += item.estimated_tokens
            total_tokens += item.estimated_tokens

        self._add_summaries(
            working,
            included,
            exclusions,
            transforms,
        )
        return self._build_selection(
            working,
            included,
            exclusions,
            transforms,
            mandatory,
        )

    def remove_for_character_limit(
        self,
        selection: ContextSelection,
        item_ids: tuple[str, ...],
    ) -> ContextSelection:
        """Remove optional, non-transformed outputs when Prompt characters still overflow."""

        remove = set(item_ids)
        if remove.intersection(selection.mandatory_item_ids):
            raise ContextAccountingError(ContextErrorCode.ESSENTIAL_OVERFLOW)
        transform_outputs = {transform.output_item_id for transform in selection.transforms}
        if remove.intersection(transform_outputs):
            raise ContextAccountingError(ContextErrorCode.TRANSFORM_MISMATCH)
        included = set(selection.included_item_ids)
        if not remove.issubset(included):
            raise ContextAccountingError(ContextErrorCode.LEDGER_MISMATCH)
        exclusions = {exclusion.item_id: exclusion for exclusion in selection.exclusions}
        for item_id in remove:
            included.remove(item_id)
            exclusions[item_id] = ContextExclusion(
                item_id=item_id,
                reason=ContextExclusionReason.CHARACTER_LIMIT,
            )
        return self._build_selection(
            list(selection.items),
            included,
            exclusions,
            list(selection.transforms),
            set(selection.mandatory_item_ids),
        )

    def _project_large_tool_items(
        self,
        items: list[ContextItem],
        mandatory: set[str],
        exclusions: dict[str, ContextExclusion],
        transforms: list[ContextTransformRecord],
    ) -> None:
        for item in tuple(items):
            if (
                item.source_kind is not ContextSourceKind.TOOL_RESULT
                or item.estimated_tokens <= self._policy.max_inline_tool_tokens
            ):
                continue
            projection = _artifact_projection(item)
            output = build_context_item(
                source_id=f"artifact_projection:{item.content_sha256[:32]}",
                source_kind=ContextSourceKind.TOOL_RESULT,
                trust_level=ContextTrustLevel.OBSERVED_EVIDENCE,
                task_id=item.task_id,
                run_id=item.run_id,
                repository_commit=item.repository_commit,
                ordinal=len(items),
                priority=item.priority,
                content=_canonical_json(projection),
                truncated=True,
            )
            items.append(output)
            exclusions[item.item_id] = ContextExclusion(
                item_id=item.item_id,
                reason=ContextExclusionReason.ARTIFACT_OFFLOAD,
                replacement_item_id=output.item_id,
            )
            if item.item_id in mandatory:
                mandatory.remove(item.item_id)
                mandatory.add(output.item_id)
            transforms.append(
                ContextTransformRecord(
                    kind=ContextTransformKind.ARTIFACT_PROJECTION,
                    transform_version=ARTIFACT_PROJECTION_VERSION,
                    source_item_ids=(item.item_id,),
                    output_item_id=output.item_id,
                    source_content_sha256=transform_source_digest((item.content_sha256,)),
                )
            )

    def _deduplicate(
        self,
        items: list[ContextItem],
        mandatory: set[str],
        exclusions: dict[str, ContextExclusion],
        transforms: list[ContextTransformRecord],
    ) -> None:
        groups: dict[tuple[ContextSourceKind, str], list[ContextItem]] = {}
        for item in items:
            if item.source_kind in _DEDUPLICATED_KINDS and item.item_id not in exclusions:
                groups.setdefault(
                    (item.source_kind, item.content_sha256),
                    [],
                ).append(item)
        for duplicates in groups.values():
            if len(duplicates) < 2:
                continue
            kept = min(
                duplicates,
                key=lambda item: (
                    item.item_id not in mandatory,
                    -item.priority,
                    -item.ordinal,
                    item.item_id,
                ),
            )
            removed = tuple(item for item in duplicates if item.item_id != kept.item_id)
            for item in removed:
                mandatory.discard(item.item_id)
                exclusions[item.item_id] = ContextExclusion(
                    item_id=item.item_id,
                    reason=ContextExclusionReason.DUPLICATE,
                    replacement_item_id=kept.item_id,
                )
            transforms.append(
                ContextTransformRecord(
                    kind=ContextTransformKind.DEDUPLICATE,
                    transform_version=DEDUPLICATION_VERSION,
                    source_item_ids=tuple(item.item_id for item in removed),
                    output_item_id=kept.item_id,
                    source_content_sha256=transform_source_digest(
                        tuple(item.content_sha256 for item in removed)
                    ),
                )
            )

    def _add_summaries(
        self,
        items: list[ContextItem],
        included: set[str],
        exclusions: dict[str, ContextExclusion],
        transforms: list[ContextTransformRecord],
    ) -> None:
        for source_kind in sorted(_SUMMARIZED_KINDS, key=lambda kind: kind.value):
            candidates = [
                item
                for item in items
                if item.source_kind is source_kind
                and item.item_id in exclusions
                and exclusions[item.item_id].reason
                in {
                    ContextExclusionReason.SOURCE_BUDGET,
                    ContextExclusionReason.GLOBAL_BUDGET,
                }
            ]
            if len(candidates) < 2:
                continue
            sources = tuple(
                sorted(
                    candidates,
                    key=lambda item: (-item.priority, -item.ordinal, item.item_id),
                )[: self._policy.max_summary_sources]
            )
            summary_content = _extractive_summary(
                sources,
                target_tokens=self._policy.summary_target_tokens,
            )
            summary_digest = transform_source_digest(tuple(item.content_sha256 for item in sources))
            summary = build_context_item(
                source_id=f"summary:{source_kind.value}:{summary_digest[:24]}",
                source_kind=ContextSourceKind.WORKING_MEMORY,
                trust_level=ContextTrustLevel.UNVERIFIED_MEMORY,
                task_id=sources[0].task_id,
                run_id=sources[0].run_id,
                repository_commit=sources[0].repository_commit,
                ordinal=len(items),
                priority=max(item.priority for item in sources),
                content=summary_content,
                truncated=True,
            )
            memory_used = sum(
                item.estimated_tokens
                for item in items
                if item.item_id in included
                and context_budget_kind(item.source_kind) is ContextBudgetKind.MEMORY
            )
            total_tokens = sum(item.estimated_tokens for item in items if item.item_id in included)
            if (
                memory_used + summary.estimated_tokens > self._policy.memory_tokens
                or total_tokens + summary.estimated_tokens > self._policy.max_estimated_tokens
            ):
                continue
            items.append(summary)
            included.add(summary.item_id)
            for source in sources:
                exclusions[source.item_id] = ContextExclusion(
                    item_id=source.item_id,
                    reason=ContextExclusionReason.SUMMARY_REPLACEMENT,
                    replacement_item_id=summary.item_id,
                )
            transforms.append(
                ContextTransformRecord(
                    kind=ContextTransformKind.EXTRACTIVE_SUMMARY,
                    transform_version=CONTEXT_SUMMARY_VERSION,
                    source_item_ids=tuple(source.item_id for source in sources),
                    output_item_id=summary.item_id,
                    source_content_sha256=transform_source_digest(
                        tuple(source.content_sha256 for source in sources)
                    ),
                )
            )

    def _build_selection(
        self,
        items: list[ContextItem],
        included: set[str],
        exclusions: dict[str, ContextExclusion],
        transforms: list[ContextTransformRecord],
        mandatory: set[str],
    ) -> ContextSelection:
        ids = tuple(item.item_id for item in items)
        if included.intersection(exclusions) or included.union(exclusions) != set(ids):
            raise ContextAccountingError(ContextErrorCode.LEDGER_MISMATCH)
        budget_usages = tuple(
            ContextBudgetUsage(
                budget_kind=kind,
                token_limit=self._policy.budget_for(kind),
                candidate_tokens=sum(
                    item.estimated_tokens
                    for item in items
                    if context_budget_kind(item.source_kind) is kind
                ),
                selected_tokens=sum(
                    item.estimated_tokens
                    for item in items
                    if item.item_id in included and context_budget_kind(item.source_kind) is kind
                ),
                candidate_count=sum(
                    1 for item in items if context_budget_kind(item.source_kind) is kind
                ),
                selected_count=sum(
                    1
                    for item in items
                    if item.item_id in included and context_budget_kind(item.source_kind) is kind
                ),
            )
            for kind in ContextBudgetKind
        )
        return ContextSelection(
            items=tuple(items),
            included_item_ids=tuple(item_id for item_id in ids if item_id in included),
            exclusions=tuple(exclusions[item_id] for item_id in ids if item_id in exclusions),
            transforms=tuple(transforms),
            budget_usages=budget_usages,
            mandatory_item_ids=frozenset(mandatory),
        )

    @staticmethod
    def _validate_candidates(
        items: tuple[ContextItem, ...],
        mandatory_item_ids: frozenset[str],
    ) -> None:
        if not items:
            raise ContextAccountingError(ContextErrorCode.INVALID_ITEM)
        ids = tuple(item.item_id for item in items)
        first = items[0]
        if (
            len(set(ids)) != len(ids)
            or tuple(item.ordinal for item in items) != tuple(range(len(items)))
            or not mandatory_item_ids.issubset(ids)
            or any(
                item.task_id != first.task_id
                or item.run_id != first.run_id
                or item.repository_commit != first.repository_commit
                for item in items
            )
        ):
            raise ContextAccountingError(ContextErrorCode.INVALID_NAMESPACE)


def context_budget_kind(source_kind: ContextSourceKind) -> ContextBudgetKind:
    """Map every source kind into exactly one independently capped pool."""

    return {
        ContextSourceKind.SYSTEM: ContextBudgetKind.CONTROL,
        ContextSourceKind.CAPABILITIES: ContextBudgetKind.CONTROL,
        ContextSourceKind.REQUEST: ContextBudgetKind.CONTROL,
        ContextSourceKind.BUDGET: ContextBudgetKind.CONTROL,
        ContextSourceKind.TASK: ContextBudgetKind.TASK,
        ContextSourceKind.REPOSITORY: ContextBudgetKind.TASK,
        ContextSourceKind.EPISODIC_MEMORY: ContextBudgetKind.MEMORY,
        ContextSourceKind.REPOSITORY_MEMORY: ContextBudgetKind.MEMORY,
        ContextSourceKind.PLAN: ContextBudgetKind.PLAN,
        ContextSourceKind.PATCH: ContextBudgetKind.PLAN,
        ContextSourceKind.TEST: ContextBudgetKind.PLAN,
        ContextSourceKind.FAILURE: ContextBudgetKind.PLAN,
        ContextSourceKind.RETRIEVAL: ContextBudgetKind.RETRIEVAL,
        ContextSourceKind.TOOL_RESULT: ContextBudgetKind.TOOL,
        ContextSourceKind.WORKING_MEMORY: ContextBudgetKind.MEMORY,
    }[source_kind]


def _artifact_projection(item: ContextItem) -> JsonValue:
    try:
        raw = json.loads(item.content)
    except json.JSONDecodeError:
        raise ContextAccountingError(ContextErrorCode.ARTIFACT_REQUIRED) from None
    if not isinstance(raw, dict):
        raise ContextAccountingError(ContextErrorCode.ARTIFACT_REQUIRED)
    artifacts = _find_artifact_refs(cast(JsonValue, raw))
    if not artifacts:
        raise ContextAccountingError(ContextErrorCode.ARTIFACT_REQUIRED)
    metadata = {key: cast(JsonValue, raw[key]) for key in _TOOL_METADATA_KEYS if key in raw}
    return {
        "context_transform": ARTIFACT_PROJECTION_VERSION,
        "source_content_sha256": item.content_sha256,
        "tool_result_metadata": metadata,
        "artifact_refs": [artifact.model_dump(mode="json") for artifact in artifacts],
    }


def _find_artifact_refs(value: JsonValue) -> tuple[ArtifactRef, ...]:
    found: dict[tuple[str, str], ArtifactRef] = {}

    def visit(current: JsonValue) -> None:
        if isinstance(current, list):
            for child in current:
                visit(child)
            return
        if not isinstance(current, dict):
            return
        if {"relative_path", "sha256", "size_bytes"}.issubset(current):
            try:
                artifact = ArtifactRef.model_validate(current)
            except ValidationError:
                raise ContextAccountingError(ContextErrorCode.ARTIFACT_REQUIRED) from None
            found[(artifact.relative_path, artifact.sha256)] = artifact
            return
        for child in current.values():
            visit(child)

    visit(value)
    return tuple(found[key] for key in sorted(found))


def _extractive_summary(
    sources: tuple[ContextItem, ...],
    *,
    target_tokens: int,
) -> str:
    source_digest = transform_source_digest(tuple(source.content_sha256 for source in sources))
    base: dict[str, JsonValue] = {
        "context_transform": CONTEXT_SUMMARY_VERSION,
        "source_kind": sources[0].source_kind.value,
        "source_count": len(sources),
        "source_content_sha256": source_digest,
        "excerpts": [],
    }
    excerpts = cast(list[JsonValue], base["excerpts"])
    for source in sources:
        excerpt = source.content[:320]
        candidate: JsonValue = {
            "source_id": source.source_id,
            "excerpt": excerpt,
        }
        excerpts.append(candidate)
        if estimate_utf8_tokens(_canonical_json(base)) > target_tokens:
            excerpts.pop()
            break
    if not excerpts:
        base["excerpts"] = [
            {
                "source_id": sources[0].source_id,
                "excerpt": sources[0].content[:64],
            }
        ]
    rendered = _canonical_json(base)
    if estimate_utf8_tokens(rendered) > target_tokens:
        raise ContextAccountingError(ContextErrorCode.ESSENTIAL_OVERFLOW)
    return rendered


def _canonical_json(value: JsonValue) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "ARTIFACT_PROJECTION_VERSION",
    "DEDUPLICATION_VERSION",
    "ContextManager",
    "ContextSelection",
    "context_budget_kind",
]
