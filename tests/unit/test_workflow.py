"""Full deterministic P1 fixture workflow and artifact tests."""

from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest
from pydantic import SecretStr, ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    ModelProvider,
    RunStatus,
    SandboxNetworkMode,
    TerminationReason,
    TraceNode,
)
from repopilot.replay import replay_trace
from repopilot.schemas import FinalResult
from repopilot.trace import AgentTrace, read_trace
from repopilot.workflow import (
    AgentRunInputError,
    AgentRunIOError,
    AgentRunManifest,
    load_agent_run_manifest,
    run_agent_fixture,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
MANIFEST_PATH = FIXTURES / "p1_agent_manifest.json"


def load_payload() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
    )


def test_agent_run_manifest_round_trip_and_gold_isolation() -> None:
    manifest = load_agent_run_manifest(MANIFEST_PATH)
    rendered = manifest.model_dump_json()

    assert AgentRunManifest.model_validate_json(rendered) == manifest
    assert "expected_patch" not in rendered
    assert "gold_files" not in rendered


def test_agent_fixture_writes_deterministic_result_and_replayable_trace(
    tmp_path: Path,
) -> None:
    settings = RepoPilotSettings(
        openai_api_key=SecretStr("sensitive-workflow-openai"),
        langsmith_api_key=SecretStr("sensitive-workflow-langsmith"),
        langsmith_tracing=True,
    )
    output_dir = tmp_path / "output"

    first = run_agent_fixture(MANIFEST_PATH, output_dir, settings)
    result_bytes = first.result_path.read_bytes()
    trace_bytes = first.trace_path.read_bytes()
    second = run_agent_fixture(MANIFEST_PATH, output_dir, settings)

    assert second.result_path.read_bytes() == result_bytes
    assert second.trace_path.read_bytes() == trace_bytes
    assert FinalResult.model_validate_json(result_bytes) == first.result
    assert AgentTrace.model_validate_json(trace_bytes) == first.trace
    assert read_trace(first.trace_path) == first.trace
    assert replay_trace(first.trace).final_state == first.trace.final_state
    assert first.result.status is RunStatus.STOPPED
    assert first.result.termination_reason is TerminationReason.AGENT_STOPPED
    assert first.result.task_metrics.tool_steps == 3
    assert first.result.task_metrics.tests_passed is True
    assert first.result.task_metrics.task_success is False
    assert first.result.usage.total_tokens == 48
    assert first.result.trace_path == "trace.json"
    assert len(first.trace.events) == 12
    assert first.trace.events[-1].node is TraceNode.FINISH
    assert "sensitive-workflow-openai" not in trace_bytes.decode()
    assert "sensitive-workflow-langsmith" not in trace_bytes.decode()
    assert first.trace.settings["openai_api_key"] == "<redacted>"
    assert first.trace.settings["langsmith_api_key"] == "<redacted>"


def test_agent_fixture_uses_no_network_or_host_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("P1 Fake workflow attempted an external capability")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)

    artifacts = run_agent_fixture(
        MANIFEST_PATH,
        tmp_path / "output",
        RepoPilotSettings(),
    )

    assert artifacts.result.termination_reason is TerminationReason.AGENT_STOPPED


@pytest.mark.parametrize(
    "settings",
    [
        RepoPilotSettings(
            model_provider=ModelProvider.OPENAI,
            model="gpt-test",
            openai_api_key=SecretStr("provider-test-value"),
        ),
        RepoPilotSettings(model="another-fake"),
        RepoPilotSettings(sandbox_network=SandboxNetworkMode.SETUP_ONLY),
    ],
)
def test_agent_fixture_rejects_non_fake_or_networked_settings(
    tmp_path: Path,
    settings: RepoPilotSettings,
) -> None:
    with pytest.raises(AgentRunInputError):
        run_agent_fixture(MANIFEST_PATH, tmp_path / "output", settings)


def test_agent_manifest_rejects_mismatched_provider_or_sandbox_identity() -> None:
    provider_payload = load_payload()
    provider_payload["provider_responses"][0]["request_id"] = "ffffffff-ffff-4fff-8fff-ffffffffffff"
    sandbox_payload = load_payload()
    sandbox_payload["sandbox_handle"]["task_id"] = "another-task"

    with pytest.raises(ValidationError, match="decision order"):
        AgentRunManifest.model_validate(provider_payload)
    with pytest.raises(ValidationError, match="sandbox handle"):
        AgentRunManifest.model_validate(sandbox_payload)


def test_agent_manifest_and_repository_io_errors_are_safe(tmp_path: Path) -> None:
    with pytest.raises(AgentRunIOError, match="unable to read"):
        load_agent_run_manifest(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    secret = "sensitive-invalid-workflow"
    invalid.write_text(f'{{"secret": "{secret}"', encoding="utf-8")
    with pytest.raises(AgentRunInputError) as captured:
        load_agent_run_manifest(invalid)
    assert secret not in str(captured.value)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(MANIFEST_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(AgentRunIOError, match="repository"):
        run_agent_fixture(manifest, tmp_path / "output", RepoPilotSettings())
