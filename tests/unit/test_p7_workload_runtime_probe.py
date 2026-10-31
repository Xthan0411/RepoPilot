"""Real SDK/schema/prompt construction behind a network-forbidden transport."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from repopilot.config import RepoPilotSettings
from repopilot.enums import ModelProvider, SandboxLifecycle, SandboxNetworkMode
from repopilot.providers.openai import OpenAIProvider, OpenAISDKTransport
from repopilot.repair_workflow import RepairRetrievalContext, RepairWorkflowError
from repopilot.sandbox import FakeSandbox, SandboxHandle
from repopilot.schemas import CommandSpec, TaskSpec

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from scripts import p7_workload_runtime_probe as probe  # noqa: E402
from scripts.p7_holdout_control import HoldoutEnvironmentManifest  # noqa: E402


def _inputs(tmp_path: Path) -> dict[str, Any]:
    binding = HoldoutEnvironmentManifest.model_validate_json(
        (ROOT / "results/p7-holdout/environments-v1.json").read_bytes()
    ).bindings[0]
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "example.py").write_text("answer = 42\n")
    task = TaskSpec(
        task_id=binding.task_id,
        repository_path=str(repository),
        base_commit=binding.base_commit,
        issue_title="Fix example",
        issue_body="The example has a regression.",
        test_commands=(CommandSpec(argv=("python", "-m", "unittest")),),
        timeout_seconds=900,
    )
    return {
        "root": tmp_path,
        "task": task,
        "settings": RepoPilotSettings(
            model_provider=ModelProvider.OPENAI,
            model="gpt-5.6-terra",
            openai_api_key=SecretStr("REAL_SECRET_MUST_NOT_REACH_SDK"),
            sandbox_network=SandboxNetworkMode.NONE,
        ),
        "binding": binding,
        "regression_patch": "REGRESSION_TEST_ONLY_CANARY",
        "retrieval": RepairRetrievalContext(
            repository_id="example",
            repository_content_sha256="a" * 64,
            repository_summary="SAFE_REPOSITORY_TEXT_NOT_IN_METRICS",
            hits=(),
            indexing_latency_ms=0,
            retrieval_latency_ms=0,
        ),
    }


def _sandbox(inputs: dict[str, Any]) -> FakeSandbox:
    return FakeSandbox(
        handle=SandboxHandle(
            sandbox_id="runtime-probe",
            task_id=inputs["task"].task_id,
            repository_commit=inputs["task"].base_commit,
            prepared_at=datetime.now(UTC),
        ),
        steps=(),
    )


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("model request or Agent execution is forbidden")


def test_probe_builds_real_payload_and_cleans_without_provider_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    sandbox = _sandbox(inputs)
    directories: list[Path] = []

    def factory(**kwargs: object) -> object:
        assert kwargs["regression_patch"] == "REGRESSION_TEST_ONLY_CANARY"

        def build(path: Path) -> FakeSandbox:
            directories.append(path)
            return sandbox

        return build

    monkeypatch.setattr(probe, "build_isolated_test_sandbox_factory", factory)
    monkeypatch.setattr(OpenAIProvider, "generate", _forbidden)
    monkeypatch.setattr(OpenAISDKTransport, "create", _forbidden)
    original_init = OpenAISDKTransport.__init__
    keys: list[str] = []

    def checked_init(self: OpenAISDKTransport, *, api_key: str, timeout_seconds: int) -> None:
        keys.append(api_key)
        original_init(self, api_key=api_key, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(OpenAISDKTransport, "__init__", checked_init)
    result = probe.probe_runtime(**inputs)

    assert result["registered_tool_count"] == 7
    assert result["first_decision_allowed_tools"] == ["run_tests"]
    assert result["provider_requests_made"] == 0
    assert result["agent_constructed"] is False
    assert result["temporary_sandboxes_destroyed"] is True
    assert sandbox.lifecycle is SandboxLifecycle.DESTROYED
    assert directories and not directories[0].exists()
    assert keys == [probe._DUMMY_KEY]
    serialized = json.dumps(result)
    for canary in ("REAL_SECRET", "REGRESSION_TEST_ONLY", "SAFE_REPOSITORY_TEXT"):
        assert canary not in serialized


def test_probe_failure_still_destroys_sandbox_and_redacts_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    sandbox = _sandbox(inputs)
    monkeypatch.setattr(
        probe, "build_isolated_test_sandbox_factory", lambda **kwargs: lambda path: sandbox
    )

    def broken(**kwargs: object) -> None:
        raise RuntimeError("sk-SECRET postgresql://user:password@private source text")

    monkeypatch.setattr(probe, "_build_repair_registry", broken)
    with pytest.raises(RepairWorkflowError) as error:
        probe.probe_runtime(**inputs)
    assert error.value.detail.stage == "provider_configuration"
    assert error.value.detail.cause_code == "unexpected_exception"
    assert "SECRET" not in str(error.value)
    assert "password" not in error.value.detail.model_dump_json()
    assert sandbox.lifecycle is SandboxLifecycle.DESTROYED


def test_probe_rejects_networked_setup_before_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path)
    inputs["settings"] = inputs["settings"].model_copy(
        update={"sandbox_network": SandboxNetworkMode.SETUP_ONLY}
    )
    monkeypatch.setattr(probe, "build_isolated_test_sandbox_factory", _forbidden)
    with pytest.raises(RepairWorkflowError) as error:
        probe.probe_runtime(**inputs)
    assert error.value.detail.stage == "input"


def test_transport_blocks_even_an_accidental_create() -> None:
    transport = probe._BlockedSDKTransport(api_key=probe._DUMMY_KEY, timeout_seconds=1)
    try:
        with pytest.raises(RuntimeError, match="provider_requests_forbidden"):
            transport.create({})
    finally:
        transport.close()
