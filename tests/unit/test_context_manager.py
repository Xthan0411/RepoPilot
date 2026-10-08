"""Deterministic P6-S2 Context Manager policy and transformation tests."""

from __future__ import annotations

import json
from uuid import UUID

import pytest
from pydantic import ValidationError

from repopilot.context import (
    ARTIFACT_PROJECTION_VERSION,
    CONTEXT_POLICY_VERSION,
    CONTEXT_SUMMARY_VERSION,
    ContextAccountingError,
    ContextBudgetPolicy,
    ContextManager,
    build_context_item,
    context_budget_kind,
)
from repopilot.enums import (
    ContextBudgetKind,
    ContextErrorCode,
    ContextExclusionReason,
    ContextSourceKind,
    ContextTransformKind,
    ContextTrustLevel,
)

RUN_ID = UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")
COMMIT = "a" * 40


def item(
    ordinal: int,
    *,
    source_id: str,
    source_kind: ContextSourceKind,
    content: str,
    priority: int = 50,
    trust_level: ContextTrustLevel = ContextTrustLevel.UNVERIFIED_MEMORY,
):
    return build_context_item(
        source_id=source_id,
        source_kind=source_kind,
        trust_level=trust_level,
        task_id="fixture-001",
        run_id=RUN_ID,
        repository_commit=COMMIT,
        ordinal=ordinal,
        priority=priority,
        content=content,
    )


def policy(**updates: int) -> ContextBudgetPolicy:
    values: dict[str, object] = {
        "max_estimated_tokens": 1_000,
        "control_tokens": 500,
        "task_tokens": 500,
        "plan_tokens": 500,
        "retrieval_tokens": 500,
        "tool_tokens": 500,
        "memory_tokens": 500,
        "max_inline_tool_tokens": 100,
        "summary_target_tokens": 100,
        "max_summary_sources": 8,
    }
    values.update(updates)
    return ContextBudgetPolicy.model_validate(values)


def test_policy_defaults_and_all_source_budget_mappings_are_frozen() -> None:
    assert ContextBudgetPolicy().policy_version == CONTEXT_POLICY_VERSION
    assert tuple(context_budget_kind(kind) for kind in ContextSourceKind) == (
        ContextBudgetKind.CONTROL,
        ContextBudgetKind.CONTROL,
        ContextBudgetKind.CONTROL,
        ContextBudgetKind.TASK,
        ContextBudgetKind.TASK,
        ContextBudgetKind.MEMORY,
        ContextBudgetKind.MEMORY,
        ContextBudgetKind.PLAN,
        ContextBudgetKind.RETRIEVAL,
        ContextBudgetKind.PLAN,
        ContextBudgetKind.PLAN,
        ContextBudgetKind.CONTROL,
        ContextBudgetKind.TOOL,
        ContextBudgetKind.PLAN,
        ContextBudgetKind.MEMORY,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_estimated_tokens", True),
        ("retrieval_tokens", 1_001),
        ("max_inline_tool_tokens", 501),
        ("summary_target_tokens", 501),
    ],
)
def test_policy_rejects_boolean_and_inconsistent_caps(field: str, value: object) -> None:
    values = policy().model_dump(mode="python")
    values[field] = value
    with pytest.raises(ValidationError):
        ContextBudgetPolicy.model_validate(values)


def test_manager_deduplicates_then_uses_priority_and_recency_within_budget() -> None:
    candidates = (
        item(
            0,
            source_id="system:instructions",
            source_kind=ContextSourceKind.SYSTEM,
            content="system",
            priority=100,
            trust_level=ContextTrustLevel.TRUSTED_CONTROL,
        ),
        item(
            1,
            source_id="retrieval:old",
            source_kind=ContextSourceKind.RETRIEVAL,
            content="same chunk",
            priority=70,
            trust_level=ContextTrustLevel.UNTRUSTED_REPOSITORY,
        ),
        item(
            2,
            source_id="retrieval:new",
            source_kind=ContextSourceKind.RETRIEVAL,
            content="same chunk",
            priority=70,
            trust_level=ContextTrustLevel.UNTRUSTED_REPOSITORY,
        ),
        item(
            3,
            source_id="retrieval:high",
            source_kind=ContextSourceKind.RETRIEVAL,
            content="h" * 180,
            priority=90,
            trust_level=ContextTrustLevel.UNTRUSTED_REPOSITORY,
        ),
        item(
            4,
            source_id="retrieval:low",
            source_kind=ContextSourceKind.RETRIEVAL,
            content="l" * 180,
            priority=60,
            trust_level=ContextTrustLevel.UNTRUSTED_REPOSITORY,
        ),
    )
    manager = ContextManager(policy(retrieval_tokens=70))

    selection = manager.select(
        candidates,
        mandatory_item_ids=frozenset({candidates[0].item_id}),
    )
    exclusions = {entry.item_id: entry for entry in selection.exclusions}

    assert candidates[2].item_id in selection.included_item_ids
    assert exclusions[candidates[1].item_id].reason is ContextExclusionReason.DUPLICATE
    assert exclusions[candidates[1].item_id].replacement_item_id == candidates[2].item_id
    assert candidates[3].item_id in selection.included_item_ids
    assert exclusions[candidates[4].item_id].reason is ContextExclusionReason.SOURCE_BUDGET
    assert selection.transforms[0].kind is ContextTransformKind.DEDUPLICATE


def test_manager_summarizes_multiple_budget_exclusions_as_unverified_memory() -> None:
    candidates = tuple(
        item(
            index,
            source_id=f"working_memory:{index}",
            source_kind=ContextSourceKind.WORKING_MEMORY,
            content=f"hypothesis-{index}-" + ("x" * 360),
            priority=50,
        )
        for index in range(4)
    )
    manager = ContextManager(
        policy(
            memory_tokens=350,
            summary_target_tokens=100,
        )
    )

    selection = manager.select(candidates, mandatory_item_ids=frozenset())
    summaries = [
        candidate for candidate in selection.items if candidate.source_id.startswith("summary:")
    ]
    transform = next(
        entry
        for entry in selection.transforms
        if entry.kind is ContextTransformKind.EXTRACTIVE_SUMMARY
    )

    assert len(summaries) == 1
    assert summaries[0].trust_level is ContextTrustLevel.UNVERIFIED_MEMORY
    assert json.loads(summaries[0].content)["context_transform"] == CONTEXT_SUMMARY_VERSION
    assert transform.output_item_id == summaries[0].item_id
    assert all(
        exclusion.reason is ContextExclusionReason.SUMMARY_REPLACEMENT
        for exclusion in selection.exclusions
        if exclusion.replacement_item_id == summaries[0].item_id
    )


def test_global_budget_applies_after_independent_source_caps() -> None:
    candidates = (
        item(
            0,
            source_id="retrieval:one",
            source_kind=ContextSourceKind.RETRIEVAL,
            content="r" * 270,
            priority=90,
            trust_level=ContextTrustLevel.UNTRUSTED_REPOSITORY,
        ),
        item(
            1,
            source_id="working_memory:one",
            source_kind=ContextSourceKind.WORKING_MEMORY,
            content="m" * 270,
            priority=80,
        ),
        item(
            2,
            source_id="plan:one",
            source_kind=ContextSourceKind.PLAN,
            content="p" * 270,
            priority=70,
            trust_level=ContextTrustLevel.RUNTIME_STATE,
        ),
    )
    manager = ContextManager(
        policy(
            max_estimated_tokens=256,
            control_tokens=256,
            task_tokens=256,
            plan_tokens=256,
            retrieval_tokens=256,
            tool_tokens=256,
            memory_tokens=256,
            max_inline_tool_tokens=100,
            summary_target_tokens=100,
        )
    )

    selection = manager.select(candidates, mandatory_item_ids=frozenset())

    assert candidates[0].item_id in selection.included_item_ids
    assert candidates[1].item_id in selection.included_item_ids
    assert (
        next(
            exclusion.reason
            for exclusion in selection.exclusions
            if exclusion.item_id == candidates[2].item_id
        )
        is ContextExclusionReason.GLOBAL_BUDGET
    )


def test_large_tool_result_is_replaced_by_artifact_only_projection() -> None:
    content = json.dumps(
        {
            "call_id": "11111111-1111-4111-8111-111111111111",
            "tool_name": "read_file",
            "status": "success",
            "output_bytes": 9_000,
            "truncated": True,
            "payload": {
                "preview": "x" * 2_000,
                "artifact": {
                    "relative_path": "artifacts/run/read.txt",
                    "sha256": "b" * 64,
                    "size_bytes": 9_000,
                    "media_type": "text/plain",
                },
            },
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    raw = item(
        0,
        source_id="tool_result:read",
        source_kind=ContextSourceKind.TOOL_RESULT,
        content=content,
        priority=75,
        trust_level=ContextTrustLevel.OBSERVED_EVIDENCE,
    )

    selection = ContextManager(policy(max_inline_tool_tokens=32)).select(
        (raw,),
        mandatory_item_ids=frozenset(),
    )
    projection = next(
        candidate
        for candidate in selection.items
        if candidate.item_id in selection.included_item_ids
    )

    assert projection.item_id != raw.item_id
    assert json.loads(projection.content)["context_transform"] == ARTIFACT_PROJECTION_VERSION
    assert "x" * 100 not in projection.content
    assert selection.exclusions[0].reason is ContextExclusionReason.ARTIFACT_OFFLOAD
    assert selection.transforms[0].kind is ContextTransformKind.ARTIFACT_PROJECTION


def test_large_tool_result_without_artifact_fails_closed() -> None:
    raw = item(
        0,
        source_id="tool_result:read",
        source_kind=ContextSourceKind.TOOL_RESULT,
        content=json.dumps({"payload": {"preview": "x" * 2_000}}),
        trust_level=ContextTrustLevel.OBSERVED_EVIDENCE,
    )

    with pytest.raises(ContextAccountingError) as captured:
        ContextManager(policy(max_inline_tool_tokens=32)).select(
            (raw,),
            mandatory_item_ids=frozenset(),
        )

    assert captured.value.code is ContextErrorCode.ARTIFACT_REQUIRED


def test_mandatory_context_overflow_fails_without_silent_eviction() -> None:
    essential = item(
        0,
        source_id="task:issue",
        source_kind=ContextSourceKind.TASK,
        content="x" * 600,
        priority=100,
        trust_level=ContextTrustLevel.UNTRUSTED_INPUT,
    )

    with pytest.raises(ContextAccountingError) as captured:
        ContextManager(policy(task_tokens=100)).select(
            (essential,),
            mandatory_item_ids=frozenset({essential.item_id}),
        )

    assert captured.value.code is ContextErrorCode.ESSENTIAL_OVERFLOW
