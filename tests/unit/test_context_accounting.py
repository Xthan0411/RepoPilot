"""Contracts for P6-S1 context provenance and token accounting."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from repopilot.context import (
    CONTEXT_ESTIMATOR_VERSION,
    ContextAccountingError,
    ContextBudgetPolicy,
    ContextItem,
    ContextUsageLedger,
    ContextWindowSnapshot,
    build_context_item,
    estimate_utf8_tokens,
    estimate_utf8_tokens_from_bytes,
)
from repopilot.enums import (
    ContextBudgetKind,
    ContextErrorCode,
    ContextExclusionReason,
    ContextSourceKind,
    ContextTransformKind,
    ContextTrustLevel,
    DatasetSplit,
    MemoryAccessMode,
    RepositoryMemoryLookupStatus,
    ToolResultStatus,
    VerifiedEpisodeLookupStatus,
)
from repopilot.providers import (
    PromptBuildError,
    ProviderRequest,
    build_agent_prompt,
    build_context_aware_provider_request,
    compose_agent_prompt,
)
from repopilot.schemas import (
    AgentState,
    MemoryLoadAudit,
    MemoryNamespace,
    RepositoryMemoryItem,
    ToolExecutionResult,
    VerifiedEpisodeMemoryItem,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
RUN_ID = UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")
REQUEST_ID = UUID("11111111-2222-4333-8444-555555555555")
REQUESTED_AT = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)


def make_state(**updates: object) -> AgentState:
    payload = cast(
        dict[str, Any],
        json.loads((FIXTURES / "valid_agent_state.json").read_text(encoding="utf-8")),
    )
    payload.update(updates)
    return AgentState.model_validate(payload)


def make_request(
    state: AgentState | None = None,
    *,
    max_chars: int = 50_000,
) -> ProviderRequest:
    return build_context_aware_provider_request(
        request_id=REQUEST_ID,
        run_id=RUN_ID,
        state=state or make_state(),
        allowed_tools=("search_code", "read_file"),
        requested_at=REQUESTED_AT,
        max_chars=max_chars,
    )


def test_utf8_estimator_is_frozen_and_rejects_invalid_byte_counts() -> None:
    assert CONTEXT_ESTIMATOR_VERSION == "utf8-bytes-ceil-div-3-v1"
    assert estimate_utf8_tokens("") == 0
    assert estimate_utf8_tokens("abc") == 1
    assert estimate_utf8_tokens("abcd") == 2
    assert estimate_utf8_tokens("你") == 1
    assert estimate_utf8_tokens("你好") == 2

    for value in (-1, True):
        with pytest.raises(ContextAccountingError) as captured:
            estimate_utf8_tokens_from_bytes(value)
        assert captured.value.code is ContextErrorCode.INVALID_ITEM


def test_context_item_id_and_exact_content_accounting_are_tamper_evident() -> None:
    first = build_context_item(
        source_id="repository:summary",
        source_kind=ContextSourceKind.REPOSITORY,
        trust_level=ContextTrustLevel.UNTRUSTED_REPOSITORY,
        task_id="fixture-001",
        run_id=RUN_ID,
        repository_commit="a" * 40,
        ordinal=0,
        priority=75,
        content="解析器 summary",
    )
    second = build_context_item(
        source_id="repository:summary",
        source_kind=ContextSourceKind.REPOSITORY,
        trust_level=ContextTrustLevel.UNTRUSTED_REPOSITORY,
        task_id="fixture-001",
        run_id=RUN_ID,
        repository_commit="a" * 40,
        ordinal=99,
        priority=1,
        content="解析器 summary",
    )

    assert first.item_id == second.item_id
    assert first.character_count == 11
    assert first.byte_count == len(first.content.encode("utf-8"))
    assert first.estimated_tokens == estimate_utf8_tokens(first.content)

    tampered = first.model_dump(mode="python")
    tampered["content"] = "changed"
    with pytest.raises(ValidationError, match=r"count|digest"):
        ContextItem.model_validate(tampered)


def test_composition_partitions_every_candidate_under_versioned_policy() -> None:
    state = make_state(
        working_memory=tuple(f"memory-{index}" for index in range(12)),
    )
    bare = ProviderRequest(
        request_id=REQUEST_ID,
        run_id=RUN_ID,
        state=state,
        allowed_tools=("search_code", "read_file"),
        requested_at=REQUESTED_AT,
    )
    composition = compose_agent_prompt(bare, max_chars=50_000)
    request = make_request(state)
    window = composition.context_window

    assert build_agent_prompt(request, max_chars=50_000) == composition.prompt
    assert request.context_window == window
    assert window.fits_character_budget
    assert window.context_chars_before >= window.context_chars_after
    assert window.estimated_tokens_before >= window.estimated_tokens_after
    assert len(window.included_item_ids) + len(window.exclusions) == len(window.items)
    assert window.policy is not None
    assert len(window.budget_usages) == 6
    assert window.exclusions == ()
    assert "memory-0" in composition.prompt
    assert "memory-11" in composition.prompt
    assert {item.source_kind for item in window.items}.issuperset(
        {
            ContextSourceKind.SYSTEM,
            ContextSourceKind.CAPABILITIES,
            ContextSourceKind.REQUEST,
            ContextSourceKind.TASK,
            ContextSourceKind.REPOSITORY,
            ContextSourceKind.PLAN,
            ContextSourceKind.RETRIEVAL,
            ContextSourceKind.PATCH,
            ContextSourceKind.TEST,
            ContextSourceKind.BUDGET,
            ContextSourceKind.WORKING_MEMORY,
        }
    )
    assert all(item.task_id == state.task.task_id for item in window.items)
    assert all(item.run_id == RUN_ID for item in window.items)
    assert all(item.repository_commit == state.task.base_commit for item in window.items)


def test_repository_memory_is_separately_accounted_and_remains_untrusted() -> None:
    original = make_state()
    content = json.dumps(
        {
            "memory_version": "source-confirmation-v1",
            "fact": "Parser implementation is in pkg/parser.py.",
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    memory = RepositoryMemoryItem(
        record_id="rmem_" + ("a" * 64),
        repository_id="repo_context_fixture",
        repository_commit=original.task.base_commit,
        repository_content_sha256="b" * 64,
        kind="confirmed_fact",
        confirmation="source_confirmed",
        producer_version="source-confirmation-v1",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        evidence_count=1,
        evidence_sha256="c" * 64,
        source_record_sha256="d" * 64,
    )
    request = make_request(original.model_copy(update={"repository_memory": (memory,)}))
    window = request.context_window

    assert window is not None
    item = next(
        candidate
        for candidate in window.items
        if candidate.source_kind is ContextSourceKind.REPOSITORY_MEMORY
    )
    assert item.trust_level is ContextTrustLevel.UNTRUSTED_REPOSITORY
    assert item.item_id in window.included_item_ids
    assert "pkg/parser.py" in build_agent_prompt(request, max_chars=50_000)


def test_verified_episode_is_historical_memory_with_auditable_token_usage() -> None:
    original = make_state()
    namespace = MemoryNamespace(
        experiment_id="context-memory",
        snapshot_id="snapshot-v1",
        split=DatasetSplit.DEV,
        access_mode=MemoryAccessMode.READ_ONLY,
    )
    episode = VerifiedEpisodeMemoryItem(
        episode_id="episode-context-001",
        source_task_id="source-task-001",
        source_run_id=UUID("99999999-8888-4777-8666-555555555555"),
        repository_id="repo_context_fixture",
        repository_commit=original.task.base_commit,
        repository_content_sha256="b" * 64,
        problem_statement="Blank parser lines previously caused an exception.",
        issue_fingerprint="c" * 64,
        repair_summary="Skipped blank lines before parsing key/value pairs.",
        files_changed=("pkg/parser.py",),
        test_command_sha256="d" * 64,
        patch_artifact_sha256="e" * 64,
        source_result_sha256="f" * 64,
        source_trace_sha256="1" * 64,
        artifacts_sha256="2" * 64,
        verifier_version="rule-verifier-v1",
        model_provider="fake-model-v1",
        validation_tests_passed=1,
        created_at=REQUESTED_AT - timedelta(days=1),
        expires_at=REQUESTED_AT + timedelta(days=29),
        source_record_sha256="3" * 64,
    )
    audit = MemoryLoadAudit(
        namespace=namespace,
        repository_status=RepositoryMemoryLookupStatus.EMPTY,
        episodic_status=VerifiedEpisodeLookupStatus.CURRENT,
        episodic_records_loaded=1,
    )
    state = original.model_copy(
        update={
            "memory_namespace": namespace,
            "memory_load_audit": audit,
            "episodic_memory": (episode,),
        }
    )

    request = make_request(state)
    window = request.context_window

    assert window is not None
    item = next(
        candidate
        for candidate in window.items
        if candidate.source_kind is ContextSourceKind.EPISODIC_MEMORY
    )
    usage = next(
        entry for entry in window.budget_usages if entry.budget_kind is ContextBudgetKind.MEMORY
    )
    assert item.trust_level is ContextTrustLevel.HISTORICAL_EVIDENCE
    assert item.item_id in window.included_item_ids
    assert usage.candidate_tokens >= item.estimated_tokens
    assert usage.selected_tokens >= item.estimated_tokens
    assert usage.candidate_count >= 1
    assert usage.selected_count >= 1
    assert "Blank parser lines" in build_agent_prompt(request, max_chars=50_000)


def test_character_limit_exclusions_record_preexisting_prompt_selection() -> None:
    state = make_state(
        working_memory=tuple(f"memory-{index}-" + ("x" * 1_900) for index in range(10)),
    )
    request = make_request(state, max_chars=4_096)
    window = request.context_window

    assert window is not None
    assert window.fits_character_budget
    assert len(build_agent_prompt(request, max_chars=4_096)) <= 4_096
    prompt = build_agent_prompt(request, max_chars=4_096)
    assert "memory-9-" in prompt
    assert '"active_step_id":"reproduce"' in prompt
    assert '"patch":' in prompt
    assert '"tests":' in prompt
    assert ContextExclusionReason.CHARACTER_LIMIT in {
        exclusion.reason for exclusion in window.exclusions
    }
    assert window.context_chars_before > window.context_chars_after


def test_context_request_rejects_namespace_and_prompt_snapshot_mismatch() -> None:
    request = make_request()
    window = request.context_window
    assert window is not None

    with pytest.raises(ValidationError, match="namespace"):
        ProviderRequest(
            request_id=UUID("ffffffff-ffff-4fff-8fff-ffffffffffff"),
            run_id=RUN_ID,
            state=request.state,
            allowed_tools=request.allowed_tools,
            requested_at=REQUESTED_AT,
            context_window=window,
        )

    changed_state = request.state.model_copy(
        update={"repository_summary": "A changed repository summary."}
    )
    stale_request = request.model_copy(update={"state": changed_state})
    with pytest.raises(PromptBuildError, match="snapshot"):
        build_agent_prompt(stale_request, max_chars=50_000)


def test_invalid_limit_and_usage_ledger_mismatch_fail_closed() -> None:
    bare = ProviderRequest(
        request_id=REQUEST_ID,
        run_id=RUN_ID,
        state=make_state(),
        requested_at=REQUESTED_AT,
    )
    with pytest.raises(PromptBuildError, match="limit"):
        compose_agent_prompt(bare, max_chars=4_095)

    request = make_request()
    window = request.context_window
    assert window is not None
    with pytest.raises(ValidationError, match="delta"):
        ContextUsageLedger(
            request_id=request.request_id,
            rendered_prompt_sha256=window.rendered_prompt_sha256,
            estimated_input_tokens=window.estimated_tokens_after,
            provider_input_tokens=100,
            cached_input_tokens=0,
            cache_write_input_tokens=0,
            estimate_delta_tokens=0,
            attempts=1,
        )


def test_prompt_deduplicates_identical_retrieval_content_with_traceable_replacement() -> None:
    original = make_state()
    hit = original.retrieved_context[0]
    state = original.model_copy(update={"retrieved_context": (hit, hit)})

    composition = compose_agent_prompt(
        ProviderRequest(
            request_id=REQUEST_ID,
            run_id=RUN_ID,
            state=state,
            requested_at=REQUESTED_AT,
        ),
        max_chars=50_000,
    )
    window = composition.context_window

    assert composition.prompt.count(hit.chunk_id) == 1
    assert any(
        transform.kind is ContextTransformKind.DEDUPLICATE for transform in window.transforms
    )
    duplicate = next(
        exclusion
        for exclusion in window.exclusions
        if exclusion.reason is ContextExclusionReason.DUPLICATE
    )
    assert duplicate.replacement_item_id in window.included_item_ids

    tampered = window.model_dump(mode="python")
    tampered["transforms"][0]["source_content_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="digest"):
        ContextWindowSnapshot.model_validate(tampered)


def test_prompt_replaces_budgeted_memory_history_with_unverified_summary() -> None:
    state = make_state(
        working_memory=tuple(f"hypothesis-{index}-" + ("x" * 360) for index in range(4))
    )
    policy = ContextBudgetPolicy(
        max_estimated_tokens=2_000,
        control_tokens=1_000,
        task_tokens=1_000,
        plan_tokens=1_000,
        retrieval_tokens=1_000,
        tool_tokens=1_000,
        memory_tokens=350,
        max_inline_tool_tokens=500,
        summary_target_tokens=100,
    )

    composition = compose_agent_prompt(
        ProviderRequest(
            request_id=REQUEST_ID,
            run_id=RUN_ID,
            state=state,
            requested_at=REQUESTED_AT,
        ),
        max_chars=50_000,
        context_policy=policy,
    )
    summaries = [
        item for item in composition.context_window.items if item.source_id.startswith("summary:")
    ]

    assert '"context_summaries":' in composition.prompt
    assert len(summaries) == 1
    assert summaries[0].trust_level is ContextTrustLevel.UNVERIFIED_MEMORY
    assert any(
        transform.kind is ContextTransformKind.EXTRACTIVE_SUMMARY
        for transform in composition.context_window.transforms
    )


def test_prompt_offloads_large_tool_preview_to_existing_artifact_identity() -> None:
    call_id = UUID("99999999-9999-4999-8999-999999999999")
    result = ToolExecutionResult(
        call_id=call_id,
        tool_name="read_file",
        status=ToolResultStatus.SUCCESS,
        payload={
            "preview": "sensitive-large-preview-" + ("x" * 7_900),
            "artifact": {
                "relative_path": "artifacts/run/read.txt",
                "sha256": "b" * 64,
                "size_bytes": 9_000,
                "media_type": "text/plain",
            },
        },
        output_bytes=9_000,
        truncated=True,
    )
    state = make_state(
        run_id=RUN_ID,
        consumed_call_ids=(call_id,),
        tool_results=(result,),
    )

    composition = compose_agent_prompt(
        ProviderRequest(
            request_id=REQUEST_ID,
            run_id=RUN_ID,
            state=state,
            requested_at=REQUESTED_AT,
        ),
        max_chars=50_000,
    )

    assert "sensitive-large-preview" not in composition.prompt
    assert "artifacts/run/read.txt" in composition.prompt
    assert any(
        exclusion.reason is ContextExclusionReason.ARTIFACT_OFFLOAD
        for exclusion in composition.context_window.exclusions
    )
    assert any(
        transform.kind is ContextTransformKind.ARTIFACT_PROJECTION
        for transform in composition.context_window.transforms
    )
