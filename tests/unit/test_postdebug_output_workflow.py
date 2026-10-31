"""Exercise real tools, context construction and isolated verification without an API."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from uuid import uuid5

import pytest
from pydantic import JsonValue
from test_isolated_repair_workflow import (
    PRIVATE_MARKER,
    REGRESSION_PATCH,
    RUN_ID,
    SOURCE,
    STARTED_AT,
    _candidate_patch,
    _fixture_task,
    _TrustedFixtureSandbox,
)

from repopilot.agent import repair_decision_request_id
from repopilot.config import RepoPilotSettings
from repopilot.enums import RepairContextMode, RunStatus
from repopilot.evals.isolated_tests import IsolatedTestSandbox
from repopilot.providers import (
    FakeProvider,
    FinishAction,
    ProviderResponse,
    ProviderUsageDelta,
    ToolAction,
)
from repopilot.providers.prompts import build_agent_prompt
from repopilot.repair_workflow import RepairRunRequest, RepairSandbox, run_repair_request
from repopilot.replay import replay_trace
from repopilot.schemas import ToolCall


def test_large_search_and_read_continue_to_real_verified_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _fixture_task(tmp_path)
    repository = Path(str(task.repository_path))
    large_source = "".join(
        f"# increment reference {index:03d}: " + "bounded evidence " * 7 + "\n"
        for index in range(150)
    )
    (repository / "references.py").write_text(large_source)
    for argv in (
        ("git", "add", "references.py"),
        (
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "Large tool fixture",
        ),
    ):
        subprocess.run(argv, cwd=repository, check=True, capture_output=True)
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    task = task.model_copy(update={"base_commit": revision})

    steps: tuple[tuple[str, dict[str, JsonValue]], ...] = (
        ("run_tests", {"test_index": 0}),
        ("search_code", {"query": "increment", "path": "references.py", "max_matches": 100}),
        ("read_file", {"path": "references.py", "start_line": 1, "end_line": 150}),
        ("read_file", {"path": "calculator.py"}),
        ("apply_patch", {"diff": _candidate_patch(correct_patch=True)}),
        ("run_tests", {"test_index": 0}),
    )
    usage = ProviderUsageDelta(
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        estimated_cost_usd=Decimal(0),
        latency_ms=1,
    )
    tool_responses = tuple(
        ProviderResponse(
            request_id=repair_decision_request_id(RUN_ID, index),
            action=ToolAction(
                tool_call=ToolCall(
                    call_id=uuid5(RUN_ID, f"large-output:{index}"),
                    tool_name=name,
                    validated_arguments=arguments,
                    requested_at=STARTED_AT,
                ),
                considered_tools=(name,),
                selection_reason="Read bounded source evidence.",
            ),
            usage=usage,
        )
        for index, (name, arguments) in enumerate(steps)
    )
    responses = (
        *tool_responses,
        ProviderResponse(
            request_id=repair_decision_request_id(RUN_ID, len(steps)),
            action=FinishAction(summary="Independent regression validation passed."),
            usage=usage,
        ),
    )
    providers: list[FakeProvider] = []

    def recording_provider(
        script: Iterable[ProviderResponse], *, provider_name: str
    ) -> FakeProvider:
        provider = FakeProvider(script, provider_name=provider_name)
        providers.append(provider)
        return provider

    def sandbox_factory(root: Path) -> RepairSandbox:
        return IsolatedTestSandbox(
            task=task,
            regression_patch=REGRESSION_PATCH,
            artifact_root=root / "artifacts",
            agent_factory=lambda path: _TrustedFixtureSandbox(path, tmp_path / "agent"),
            evaluator_factory=lambda path: _TrustedFixtureSandbox(path, tmp_path / "evaluator"),
        )

    monkeypatch.setattr("repopilot.repair_workflow.FakeProvider", recording_provider)
    output = tmp_path / "output"
    settings = RepoPilotSettings(max_steps=20, max_tokens=20_000, timeout_seconds=60)
    artifacts = run_repair_request(
        RepairRunRequest(
            thread_id="large-tool-output-e2e",
            run_id=RUN_ID,
            started_at=STARTED_AT,
            task=task,
            sandbox_image_reference=f"python@sha256:{'a' * 64}",
            context_mode=RepairContextMode.NONE,
            provider_responses=responses,
        ),
        output_dir=output,
        settings=settings,
        sandbox_factory=sandbox_factory,
    )

    assert artifacts.result.status is RunStatus.SUCCESS
    assert artifacts.result.usage.estimated_cost_usd == 0
    assert artifacts.final_state.test_state.commands_run == 2
    assert replay_trace(artifacts.trace).final_state == artifacts.final_state
    assert len(providers) == 1 and len(providers[0].requests) == len(responses)
    # Build the exact next-request prompts, not just the projection helper's output.
    prompts = [
        build_agent_prompt(request, max_chars=settings.openai_prompt_max_chars)
        for request in providers[0].requests
    ]
    assert "increment reference 000" in prompts[2]
    assert "increment reference 000" in prompts[3]
    assert "return value - 1" in prompts[4]
    assert PRIVATE_MARKER not in artifacts.trace.model_dump_json()
    assert all(PRIVATE_MARKER not in prompt for prompt in prompts)
    assert (repository / "calculator.py").read_text() == SOURCE
    assert (repository / "references.py").read_text() == large_source

    manifest = json.loads((output / "artifacts.json").read_text())
    raw_tool_results = []
    for reference in manifest["artifacts"]:
        raw = (output / reference["relative_path"]).read_bytes()
        assert len(raw) == reference["size_bytes"]
        assert hashlib.sha256(raw).hexdigest() == reference["sha256"]
        assert PRIVATE_MARKER.encode() not in raw
        if reference["relative_path"].endswith(".read.txt"):
            raw_tool_results.append(json.loads(raw))
    assert any(result.get("content") == large_source for result in raw_tool_results)
    assert any(len(result.get("matches", [])) == 100 for result in raw_tool_results)
