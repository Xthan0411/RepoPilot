"""Real repository tools stay usable through the next Fake-provider context window."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest
from test_context_accounting import REQUEST_ID, REQUESTED_AT, RUN_ID, make_state

from repopilot.config import RepoPilotSettings
from repopilot.context import ContextBudgetPolicy, estimate_utf8_tokens
from repopilot.enums import ToolErrorCode, ToolResultStatus
from repopilot.providers import (
    FakeProvider,
    FinishAction,
    ProviderResponse,
    ProviderUsageDelta,
    build_agent_prompt,
    build_context_aware_provider_request,
)
from repopilot.sandbox.artifacts import SandboxArtifactStore
from repopilot.schemas import ArtifactRef, ToolCall, ToolExecutionResult
from repopilot.tools import ReadFileTool, ReadOnlyRepository, SearchCodeTool, ToolRegistry
from repopilot.tools.output_projection import RepositoryToolOutputProjector


def _fixture(tmp_path: Path, source: str) -> tuple[ToolRegistry, SandboxArtifactStore]:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "sample.py").write_text(source, encoding="utf-8")
    store = SandboxArtifactStore(tmp_path / "output" / "artifacts")
    store.prepare("tool-output-test", repository_source=repository)
    settings = RepoPilotSettings()
    reader = ReadOnlyRepository.from_settings(repository, settings)
    registry = ToolRegistry.from_settings(
        settings,
        output_projector=RepositoryToolOutputProjector(store),
    )
    registry.register(ReadFileTool(reader))
    registry.register(SearchCodeTool.from_settings(reader, settings))
    return registry, store


def _call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(
        call_id=uuid4(),
        tool_name=name,
        validated_arguments=arguments,
        requested_at=REQUESTED_AT,
    )


def _next_fake_prompt(result: ToolExecutionResult) -> str:
    state = make_state(run_id=RUN_ID, consumed_call_ids=(result.call_id,), tool_results=(result,))
    request = build_context_aware_provider_request(
        request_id=REQUEST_ID,
        run_id=RUN_ID,
        state=state,
        allowed_tools=("read_file", "apply_patch"),
        requested_at=REQUESTED_AT,
        max_chars=50_000,
    )
    prompt = build_agent_prompt(request, max_chars=50_000)
    provider = FakeProvider(
        (
            ProviderResponse(
                request_id=REQUEST_ID,
                action=FinishAction(summary="Offline context checked"),
                usage=ProviderUsageDelta(
                    input_tokens=1,
                    output_tokens=1,
                    total_tokens=2,
                    estimated_cost_usd="0",
                    latency_ms=0,
                ),
            ),
        )
    )
    assert provider.generate(request).request_id == REQUEST_ID
    assert len(provider.requests) == 1
    return prompt


def _ref(payload: dict) -> ArtifactRef:
    return ArtifactRef.model_validate(payload["output_projection"]["artifact"])


def test_real_large_read_preserves_unicode_lines_and_next_fake_context(tmp_path: Path) -> None:
    lines = [
        f"value_{index} = '中文 source marker {index} " + "x" * 55 + "'\n" for index in range(250)
    ]
    registry, store = _fixture(tmp_path, "".join(lines))
    call = _call("read_file", path="sample.py", start_line=1, end_line=250)
    result = registry.invoke(call)
    assert result.status is ToolResultStatus.SUCCESS and result.payload is not None
    payload = result.payload
    assert result.truncated and payload["truncated"] is True
    returned_end = int(payload["end_line"])
    assert 0 < returned_end < 250
    assert payload["content"] == "".join(lines[:returned_end])
    assert payload["content_bytes"] == len(str(payload["content"]).encode("utf-8"))
    assert payload["next_start_line"] == returned_end + 1
    reference = _ref(payload)
    original_bytes = store.read(reference)
    original = json.loads(original_bytes)
    assert original["content"] == "".join(lines)
    assert hashlib.sha256(original_bytes).hexdigest() == reference.sha256
    assert reference.size_bytes == len(original_bytes)
    assert (store.root.parent / reference.relative_path).stat().st_mode & 0o777 == 0o600
    assert "中文 source marker 0" in _next_fake_prompt(result)
    assert (
        estimate_utf8_tokens(result.model_dump_json())
        < ContextBudgetPolicy().max_inline_tool_tokens
    )
    continuation = registry.invoke(
        _call(
            "read_file",
            path="sample.py",
            start_line=returned_end + 1,
            end_line=returned_end + 2,
        )
    )
    assert continuation.payload is not None
    assert continuation.payload["content"] == "".join(lines[returned_end : returned_end + 2])
    assert "output_projection" not in continuation.payload
    assert registry.invoke(call).error_code is ToolErrorCode.DUPLICATE_CALL
    assert len(store.collect()) == 1


def test_real_large_search_retains_actionable_matches_and_fake_context(tmp_path: Path) -> None:
    source = "".join(f"target_{index} = '{'a' * 100}'\n" for index in range(100))
    registry, store = _fixture(tmp_path, source)
    result = registry.invoke(_call("search_code", query="target_", path="sample.py"))
    assert result.status is ToolResultStatus.SUCCESS and result.payload is not None
    payload = result.payload
    matches = payload["matches"]
    assert isinstance(matches, list) and 0 < len(matches) < 100
    assert matches[0]["relative_path"] == "sample.py"
    assert matches[0]["line"] == 1
    assert payload["truncated"] is True
    original = json.loads(store.read(_ref(payload)))
    assert len(original["matches"]) == 100
    assert payload["output_projection"]["omitted_match_count"] == 100 - len(matches)
    assert "target_0" in _next_fake_prompt(result)


def test_long_unrepresentable_source_line_is_not_reported_as_complete(tmp_path: Path) -> None:
    registry, store = _fixture(tmp_path, "name = '" + "你" * 4000 + "'\n")
    result = registry.invoke(_call("read_file", path="sample.py"))
    assert result.status is ToolResultStatus.ERROR
    assert result.error_code is ToolErrorCode.OUTPUT_TOO_LARGE
    assert result.payload is None
    assert len(store.collect()) == 1
    assert json.loads(store.read(store.collect()[0]))["content"].startswith("name = '")


def test_projection_storage_failure_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, store = _fixture(
        tmp_path, "".join("value = '" + "x" * 100 + "'\n" for _ in range(200))
    )

    def fail_store(*args: object, **kwargs: object) -> ArtifactRef:
        raise OSError("private host path and credentials")

    monkeypatch.setattr(store, "persist", fail_store)
    result = registry.invoke(_call("read_file", path="sample.py"))
    assert result.status is ToolResultStatus.ERROR
    assert result.error_code is ToolErrorCode.TOOL_EXECUTION_ERROR
    assert "private" not in result.model_dump_json()
    assert len(store.collect()) == 0


def test_original_registry_hard_cap_is_not_bypassed_by_offloading(tmp_path: Path) -> None:
    _, store = _fixture(tmp_path, "value = '" + "x" * 100 + "'\n")
    registry = ToolRegistry(
        max_output_bytes=100,
        output_projector=RepositoryToolOutputProjector(store),
    )
    registry.register(
        ReadFileTool(
            ReadOnlyRepository.from_settings(
                tmp_path / "repository",
                RepoPilotSettings(),
            )
        )
    )
    result = registry.invoke(_call("read_file", path="sample.py"))
    assert result.status is ToolResultStatus.ERROR
    assert result.error_code is ToolErrorCode.OUTPUT_TOO_LARGE
    assert store.collect() == ()


def test_projected_read_never_skips_a_partially_returned_long_line(tmp_path: Path) -> None:
    _, store = _fixture(tmp_path, "a = 1\r\nb = 2\r\n" + "# " + "你" * 6000 + "\r\n")
    registry = ToolRegistry(output_projector=RepositoryToolOutputProjector(store))
    registry.register(
        ReadFileTool(
            ReadOnlyRepository.from_settings(
                tmp_path / "repository",
                RepoPilotSettings(repository_max_read_bytes=9000),
            )
        )
    )
    result = registry.invoke(_call("read_file", path="sample.py"))
    assert result.status is ToolResultStatus.SUCCESS and result.payload is not None
    assert result.payload["content"] == "a = 1\r\nb = 2\r\n"
    assert result.payload["end_line"] == 2
    assert result.payload["next_start_line"] == 3
    assert result.payload["output_projection"]["source_partial_line_omitted"] is True
    assert "a = 1" in _next_fake_prompt(result)


@pytest.mark.parametrize("target_after_preview", (False, True))
def test_complete_workflow_collects_tool_artifact_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_after_preview: bool,
) -> None:
    import test_isolated_repair_workflow as fixture

    from repopilot.agent import repair_decision_request_id
    from repopilot.evals.isolated_tests import IsolatedTestSandbox
    from repopilot.providers import ToolAction
    from repopilot.repair_workflow import RepairRunRequest, run_repair_request
    from repopilot.replay import replay_trace

    original_source = fixture.SOURCE
    source = (
        "# harmless comment\n" * 650 + original_source
        if target_after_preview
        else original_source + "# source detail 中文 " * 10 + "\n" + "# harmless comment\n" * 650
    )
    monkeypatch.setattr(fixture, "SOURCE", source)
    original_patch = fixture._candidate_patch

    def candidate_patch(*, correct_patch: bool) -> str:
        if target_after_preview:
            return original_patch(correct_patch=correct_patch).replace(
                "@@ -1,2 +1,2 @@", "@@ -651,2 +651,2 @@"
            )
        return (
            original_patch(correct_patch=correct_patch).replace(
                "@@ -1,2 +1,2 @@", "@@ -1,3 +1,3 @@"
            )
            + " "
            + fixture.SOURCE.splitlines()[2]
            + "\n"
        )

    monkeypatch.setattr(fixture, "_candidate_patch", candidate_patch)
    task = fixture._fixture_task(tmp_path)
    responses = list(fixture._responses(correct_patch=True))
    if target_after_preview:
        responses.insert(
            3,
            ProviderResponse(
                request_id=uuid4(),
                action=ToolAction(
                    tool_call=_call(
                        "read_file", path="calculator.py", start_line=651, end_line=652
                    ),
                    considered_tools=("read_file",),
                    selection_reason="Inspect the actual target region.",
                ),
                usage=responses[0].usage,
            ),
        )
        responses = [
            response.model_copy(
                update={
                    "request_id": repair_decision_request_id(fixture.RUN_ID, index),
                }
            )
            for index, response in enumerate(responses)
        ]

    def sandbox_factory(root: Path):
        return IsolatedTestSandbox(
            task=task,
            regression_patch=fixture.REGRESSION_PATCH,
            artifact_root=root / "artifacts",
            agent_factory=lambda path: fixture._TrustedFixtureSandbox(path, tmp_path / "agent"),
            evaluator_factory=lambda path: fixture._TrustedFixtureSandbox(path, tmp_path / "eval"),
        )

    artifacts = run_repair_request(
        RepairRunRequest(
            thread_id="projected-source-workflow",
            run_id=fixture.RUN_ID,
            started_at=fixture.STARTED_AT,
            task=task,
            sandbox_image_reference="python@sha256:" + "a" * 64,
            provider_responses=tuple(responses),
        ),
        output_dir=tmp_path / "output",
        settings=RepoPilotSettings(max_tokens=20_000),
        sandbox_factory=sandbox_factory,
    )
    assert artifacts.result.task_metrics.task_success is True
    replay_trace(artifacts.trace)
    refs = [
        ref for ref in artifacts.artifact_manifest.artifacts if "/tool-output-" in ref.relative_path
    ]
    assert len(refs) == 1
    data = (tmp_path / "output" / refs[0].relative_path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == refs[0].sha256
    assert ("return value - 1" in json.loads(data)["content"]) is not target_after_preview
    reads = [
        item.payload for item in artifacts.final_state.tool_results if item.tool_name == "read_file"
    ]
    if target_after_preview:
        assert len(reads) == 2
        assert reads[0] is not None and "return value - 1" not in str(reads[0]["content"])
        assert reads[1] is not None and reads[1]["content"] == original_source
