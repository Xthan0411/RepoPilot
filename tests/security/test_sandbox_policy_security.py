"""Host-free hostile-input proofs for the frozen Sandbox policy boundary."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    ErrorType,
    PolicyDecision,
    SandboxCommandPhase,
    SandboxNetworkMode,
    SandboxPathOperation,
    SandboxPolicyErrorCode,
)
from repopilot.sandbox import ReadFileRequest, RunRequest, SandboxPolicy
from repopilot.schemas import CommandSpec, TaskSpec

pytestmark = pytest.mark.security


def _task(
    *commands: CommandSpec,
    setup_commands: tuple[CommandSpec, ...] = (),
) -> TaskSpec:
    configured_commands = commands or (CommandSpec(argv=("python", "-c", "pass")),)
    return TaskSpec(
        task_id="sandbox-hostile-policy",
        repository_path="fixture/repository",
        base_commit="a" * 40,
        issue_title="Hostile sandbox policy inputs",
        issue_body="Prove commands and paths remain capability bounded.",
        setup_commands=setup_commands,
        test_commands=configured_commands,
        timeout_seconds=30,
    )


def _policy(
    *,
    network: SandboxNetworkMode = SandboxNetworkMode.NONE,
) -> SandboxPolicy:
    return SandboxPolicy.from_settings(RepoPilotSettings(sandbox_network=network))


@pytest.mark.parametrize(
    "argv",
    [
        ("sh", "-c", "touch /tmp/repopilot-host-pwned"),
        ("/bin/sh", "-c", "id"),
        ("docker", "run", "--privileged", "alpine"),
        ("sudo", "cat", "/etc/shadow"),
        ("curl", "https://example.com"),
        ("/usr/bin/python3", "-c", "print('absolute executable')"),
    ],
)
def test_task_owned_escape_executables_are_still_denied(
    argv: tuple[str, ...],
) -> None:
    command = CommandSpec(argv=argv)
    decision = _policy().evaluate_command(
        _task(command),
        command,
        phase=SandboxCommandPhase.TEST,
    )

    assert decision.decision is PolicyDecision.DENIED
    assert decision.error_type is ErrorType.POLICY_DENIED
    assert decision.error_code is SandboxPolicyErrorCode.EXECUTABLE_DENIED


def test_unconfigured_allowlisted_command_is_denied() -> None:
    configured = CommandSpec(argv=("python", "-c", "print('configured')"))
    injected = CommandSpec(argv=("python", "-c", "print('different')"))

    decision = _policy().evaluate_command(
        _task(configured),
        injected,
        phase=SandboxCommandPhase.TEST,
    )

    assert decision.decision is PolicyDecision.DENIED
    assert decision.error_code is SandboxPolicyErrorCode.COMMAND_NOT_CONFIGURED


def test_shell_metacharacters_are_opaque_argv_data_not_policy_syntax() -> None:
    canary = "$(touch /tmp/repopilot-host-pwned); `id` && echo injected"
    command = CommandSpec(argv=("python", "security_probe.py", canary))

    decision = _policy().evaluate_command(
        _task(command),
        command,
        phase=SandboxCommandPhase.TEST,
    )

    assert decision.decision is PolicyDecision.ALLOWED
    assert decision.executable == "python"
    assert canary not in decision.model_dump_json()


def test_network_request_is_denied_for_test_phase_even_in_setup_only_mode() -> None:
    command = CommandSpec(argv=("python", "security_probe.py"))
    decision = _policy(network=SandboxNetworkMode.SETUP_ONLY).evaluate_command(
        _task(command),
        command,
        phase=SandboxCommandPhase.TEST,
        network_requested=True,
    )

    assert decision.decision is PolicyDecision.DENIED
    assert decision.error_code is SandboxPolicyErrorCode.NETWORK_DENIED
    assert decision.network_enabled is False


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../host-secret",
        "src/../../host-secret",
        r"src\..\host-secret",
        ".git/config",
        ".env",
        ".env/OPENAI_API_KEY",
        ".ssh/id_ed25519",
        ".aws/credentials",
        ".docker/config.json",
        "artifacts",
        "artifacts/stdout.txt",
    ],
)
def test_repository_read_policy_denies_escape_and_sensitive_roots(path: str) -> None:
    decision = _policy().evaluate_path(
        _task(),
        operation=SandboxPathOperation.READ_REPOSITORY,
        relative_path=path,
    )

    assert decision.decision is PolicyDecision.DENIED
    assert decision.error_type is ErrorType.POLICY_DENIED
    assert decision.error_code is SandboxPolicyErrorCode.PATH_DENIED


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            ReadFileRequest,
            {
                "call_id": "00000000-0000-0000-0000-000000000001",
                "relative_path": "/etc/passwd",
            },
        ),
        (
            ReadFileRequest,
            {
                "call_id": "00000000-0000-0000-0000-000000000002",
                "relative_path": "../outside",
            },
        ),
        (
            RunRequest,
            {
                "call_id": "00000000-0000-0000-0000-000000000003",
                "command": {"argv": ["python", "-c", "print(1)"]},
                "network_requested": 1,
            },
        ),
        (
            RunRequest,
            {
                "call_id": "00000000-0000-0000-0000-000000000004",
                "command": {"argv": ["python", "-c", "print(1)"]},
                "network_requested": "false",
            },
        ),
    ],
)
def test_request_schemas_reject_path_and_boolean_type_confusion(
    model: type[ReadFileRequest] | type[RunRequest],
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)
