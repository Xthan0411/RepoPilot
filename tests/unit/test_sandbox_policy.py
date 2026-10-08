"""Fail-closed Sandbox policy, resource, command, path, and audit tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    ErrorType,
    PolicyDecision,
    SandboxCommandPhase,
    SandboxNetworkMode,
    SandboxPathOperation,
    SandboxPolicyErrorCode,
    SandboxPolicyOperation,
)
from repopilot.sandbox import (
    SANDBOX_ALLOWED_ENVIRONMENT_NAMES,
    RequestedSandboxResources,
    SandboxPolicy,
    SandboxPolicyConfiguration,
    SandboxPolicyEvaluation,
    SandboxPreparationRequest,
)
from repopilot.schemas import CommandSpec, TaskSpec

TASK_ID = "sandbox-policy-task"
COMMIT = "a" * 40
SAFE_IMAGE = f"repopilot-sandbox@sha256:{'b' * 64}"


def command(
    *argv: str,
    cwd: str | None = None,
    timeout_seconds: int | None = 300,
) -> CommandSpec:
    return CommandSpec(
        argv=argv,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )


def task(
    *,
    setup_commands: tuple[CommandSpec, ...] | None = None,
    test_commands: tuple[CommandSpec, ...] | None = None,
) -> TaskSpec:
    return TaskSpec(
        task_id=TASK_ID,
        repository_path="fixture/repository",
        base_commit=COMMIT,
        issue_title="Sandbox policy fixture",
        issue_body="Run only evaluator-configured commands.",
        setup_commands=(
            (command("uv", "sync", "--locked"),) if setup_commands is None else setup_commands
        ),
        test_commands=(
            (command("python", "-m", "pytest", "-q"),) if test_commands is None else test_commands
        ),
        timeout_seconds=900,
    )


def policy(**overrides: object) -> SandboxPolicy:
    return SandboxPolicy.from_settings(RepoPilotSettings.model_validate(overrides))


def assert_denied(
    decision: SandboxPolicyEvaluation,
    code: SandboxPolicyErrorCode,
) -> None:
    assert decision.decision is PolicyDecision.DENIED
    assert decision.error_type is ErrorType.POLICY_DENIED
    assert decision.error_code is code
    assert decision.network_enabled is False


def test_default_preparation_is_non_root_minimal_and_allowed() -> None:
    sandbox_policy = policy()
    request = sandbox_policy.default_preparation_request(
        TASK_ID,
        image_reference=SAFE_IMAGE,
    )

    decision = sandbox_policy.evaluate_preparation(
        request,
        expected_task_id=TASK_ID,
    )

    assert decision.decision is PolicyDecision.ALLOWED
    assert decision.operation is SandboxPolicyOperation.PREPARE
    assert decision.error_type is None
    assert decision.error_code is None
    assert request.run_as_uid > 0
    assert request.run_as_gid > 0
    assert request.privileged is False
    assert request.read_only_rootfs is True
    assert request.no_new_privileges is True
    assert request.capabilities == ()
    assert request.host_bind_mounts == ()
    assert request.mount_docker_socket is False
    assert request.network_mode is SandboxNetworkMode.NONE
    assert request.environment_names == SANDBOX_ALLOWED_ENVIRONMENT_NAMES
    assert len(decision.request_sha256) == 64


@pytest.mark.parametrize(
    ("mutate", "expected_task_id", "code"),
    [
        (
            lambda request: request.model_copy(update={"task_id": "other-task"}),
            TASK_ID,
            SandboxPolicyErrorCode.TASK_MISMATCH,
        ),
        (
            lambda request: request.model_copy(update={"image_reference": "python:3.12-slim"}),
            TASK_ID,
            SandboxPolicyErrorCode.IMAGE_DENIED,
        ),
        (
            lambda request: request.model_copy(
                update={"network_mode": SandboxNetworkMode.SETUP_ONLY}
            ),
            TASK_ID,
            SandboxPolicyErrorCode.NETWORK_DENIED,
        ),
        (
            lambda request: request.model_copy(update={"mount_docker_socket": True}),
            TASK_ID,
            SandboxPolicyErrorCode.DOCKER_SOCKET_DENIED,
        ),
        (
            lambda request: request.model_copy(update={"host_bind_mounts": ("/Users/developer",)}),
            TASK_ID,
            SandboxPolicyErrorCode.HOST_MOUNT_DENIED,
        ),
        (
            lambda request: request.model_copy(update={"privileged": True}),
            TASK_ID,
            SandboxPolicyErrorCode.PRIVILEGED_DENIED,
        ),
        (
            lambda request: request.model_copy(update={"run_as_uid": 0}),
            TASK_ID,
            SandboxPolicyErrorCode.ROOT_USER_DENIED,
        ),
        (
            lambda request: request.model_copy(update={"run_as_gid": 0}),
            TASK_ID,
            SandboxPolicyErrorCode.ROOT_USER_DENIED,
        ),
        (
            lambda request: request.model_copy(update={"capabilities": ("CAP_SYS_ADMIN",)}),
            TASK_ID,
            SandboxPolicyErrorCode.CAPABILITY_DENIED,
        ),
        (
            lambda request: request.model_copy(update={"read_only_rootfs": False}),
            TASK_ID,
            SandboxPolicyErrorCode.ROOTFS_WRITE_DENIED,
        ),
        (
            lambda request: request.model_copy(update={"no_new_privileges": False}),
            TASK_ID,
            SandboxPolicyErrorCode.NO_NEW_PRIVILEGES_REQUIRED,
        ),
        (
            lambda request: request.model_copy(update={"environment_names": ("OPENAI_API_KEY",)}),
            TASK_ID,
            SandboxPolicyErrorCode.ENVIRONMENT_DENIED,
        ),
        (
            lambda request: request.model_copy(
                update={"resources": request.resources.model_copy(update={"cpu_limit": 2.0})}
            ),
            TASK_ID,
            SandboxPolicyErrorCode.RESOURCE_LIMIT_DENIED,
        ),
    ],
)
def test_preparation_denies_each_host_escape_or_resource_expansion(
    mutate: Callable[[SandboxPreparationRequest], SandboxPreparationRequest],
    expected_task_id: str,
    code: SandboxPolicyErrorCode,
) -> None:
    sandbox_policy = policy()
    request = mutate(
        sandbox_policy.default_preparation_request(
            TASK_ID,
            image_reference=SAFE_IMAGE,
        )
    )

    decision = sandbox_policy.evaluate_preparation(
        request,
        expected_task_id=expected_task_id,
    )

    assert_denied(decision, code)


def test_preparation_audit_does_not_echo_host_paths_or_secret_values() -> None:
    sandbox_policy = policy()
    host_path = "/Users/developer/.ssh"
    request = sandbox_policy.default_preparation_request(
        TASK_ID,
        image_reference=SAFE_IMAGE,
    ).model_copy(
        update={
            "host_bind_mounts": (host_path,),
            "environment_names": ("OPENAI_API_KEY",),
        }
    )

    decision = sandbox_policy.evaluate_preparation(
        request,
        expected_task_id=TASK_ID,
    )
    rendered = decision.model_dump_json()

    assert host_path not in rendered
    assert "OPENAI_API_KEY" not in rendered
    assert_denied(decision, SandboxPolicyErrorCode.HOST_MOUNT_DENIED)


def test_image_and_capability_allowlists_cannot_be_expanded_or_leaked() -> None:
    sandbox_policy = policy()
    image_canary = "SECRET_MUTABLE_IMAGE_CANARY"
    request = sandbox_policy.default_preparation_request(
        TASK_ID,
        image_reference=image_canary,
    )

    decision = sandbox_policy.evaluate_preparation(
        request,
        expected_task_id=TASK_ID,
    )

    assert_denied(decision, SandboxPolicyErrorCode.IMAGE_DENIED)
    assert image_canary not in decision.model_dump_json()
    with pytest.raises(ValueError, match="cannot be expanded"):
        SandboxPolicyConfiguration(
            resources=sandbox_policy.configuration.resources,
            allowed_executables=(*sandbox_policy.configuration.allowed_executables, "sh"),
        )


def test_exact_task_owned_command_is_allowed_without_a_shell() -> None:
    sandbox_policy = policy()
    fixture_task = task()

    first = sandbox_policy.evaluate_command(
        fixture_task,
        fixture_task.test_commands[0],
        phase=SandboxCommandPhase.TEST,
    )
    second = sandbox_policy.evaluate_command(
        fixture_task,
        fixture_task.test_commands[0],
        phase=SandboxCommandPhase.TEST,
    )

    assert first.decision is PolicyDecision.ALLOWED
    assert first.operation is SandboxPolicyOperation.COMMAND
    assert first.executable == "python"
    assert first.effective_timeout_seconds == 300
    assert first.network_enabled is False
    assert first.request_sha256 == second.request_sha256


def test_unconfigured_or_shell_command_is_denied_even_if_schema_valid() -> None:
    sandbox_policy = policy()
    fixture_task = task()
    unconfigured = sandbox_policy.evaluate_command(
        fixture_task,
        command("python", "-m", "pytest", "-x"),
        phase=SandboxCommandPhase.TEST,
    )
    shell = command("sh", "-c", "pytest")
    shell_task = task(setup_commands=(shell,))
    denied_shell = sandbox_policy.evaluate_command(
        shell_task,
        shell,
        phase=SandboxCommandPhase.SETUP,
    )

    assert_denied(
        unconfigured,
        SandboxPolicyErrorCode.COMMAND_NOT_CONFIGURED,
    )
    assert_denied(denied_shell, SandboxPolicyErrorCode.EXECUTABLE_DENIED)


def test_command_timeout_and_network_are_phase_bounded() -> None:
    oversized = command("python", "-m", "pytest", timeout_seconds=301)
    timeout_task = task(test_commands=(oversized,))
    denied_timeout = policy().evaluate_command(
        timeout_task,
        oversized,
        phase=SandboxCommandPhase.TEST,
    )
    setup_only_policy = policy(sandbox_network=SandboxNetworkMode.SETUP_ONLY)
    fixture_task = task()
    allowed_setup_network = setup_only_policy.evaluate_command(
        fixture_task,
        fixture_task.setup_commands[0],
        phase=SandboxCommandPhase.SETUP,
        network_requested=True,
    )
    denied_test_network = setup_only_policy.evaluate_command(
        fixture_task,
        fixture_task.test_commands[0],
        phase=SandboxCommandPhase.TEST,
        network_requested=True,
    )

    assert_denied(denied_timeout, SandboxPolicyErrorCode.TIMEOUT_DENIED)
    assert allowed_setup_network.decision is PolicyDecision.ALLOWED
    assert allowed_setup_network.network_enabled is True
    assert_denied(denied_test_network, SandboxPolicyErrorCode.NETWORK_DENIED)


def test_command_audit_hashes_arguments_instead_of_echoing_them() -> None:
    canary = "SECRET_COMMAND_ARGUMENT_CANARY"
    configured = command("python", "-m", "pytest", canary)
    fixture_task = task(test_commands=(configured,))

    decision = policy().evaluate_command(
        fixture_task,
        configured,
        phase=SandboxCommandPhase.TEST,
    )

    assert decision.decision is PolicyDecision.ALLOWED
    assert canary not in decision.model_dump_json()


def test_command_rejects_runtime_type_confusion_as_invalid_request() -> None:
    fixture_task = task()

    decision = policy().evaluate_command(
        fixture_task,
        fixture_task.test_commands[0],
        phase=cast(SandboxCommandPhase, "test"),
    )

    assert_denied(decision, SandboxPolicyErrorCode.INVALID_REQUEST)


@pytest.mark.parametrize(
    ("operation", "path"),
    [
        (SandboxPathOperation.READ_REPOSITORY, "/etc/passwd"),
        (SandboxPathOperation.READ_REPOSITORY, "../outside"),
        (SandboxPathOperation.READ_REPOSITORY, "src\\outside.py"),
        (SandboxPathOperation.READ_REPOSITORY, ".git/config"),
        (SandboxPathOperation.READ_REPOSITORY, ".env"),
        (SandboxPathOperation.READ_REPOSITORY, ".env/secret"),
        (SandboxPathOperation.READ_REPOSITORY, "artifacts"),
        (SandboxPathOperation.READ_REPOSITORY, "artifacts/stdout.txt"),
        (SandboxPathOperation.WRITE_REPOSITORY, "artifacts/diff.patch"),
        (SandboxPathOperation.WRITE_ARTIFACT, "src/output.txt"),
    ],
)
def test_path_capabilities_reject_escape_sensitive_and_cross_root_paths(
    operation: SandboxPathOperation,
    path: str,
) -> None:
    decision = policy().evaluate_path(
        task(),
        operation=operation,
        relative_path=path,
    )

    assert_denied(decision, SandboxPolicyErrorCode.PATH_DENIED)


def test_repository_and_artifact_paths_are_separate_allowed_capabilities() -> None:
    sandbox_policy = policy()
    fixture_task = task()

    repository_read = sandbox_policy.evaluate_path(
        fixture_task,
        operation=SandboxPathOperation.READ_REPOSITORY,
        relative_path="src/package/module.py",
    )
    repository_write = sandbox_policy.evaluate_path(
        fixture_task,
        operation=SandboxPathOperation.WRITE_REPOSITORY,
        relative_path="tests/test_module.py",
    )
    artifact_write = sandbox_policy.evaluate_path(
        fixture_task,
        operation=SandboxPathOperation.WRITE_ARTIFACT,
        relative_path="artifacts/stdout.txt",
    )

    assert repository_read.decision is PolicyDecision.ALLOWED
    assert repository_write.decision is PolicyDecision.ALLOWED
    assert artifact_write.decision is PolicyDecision.ALLOWED


@pytest.mark.parametrize(
    "resources",
    [
        {"cpu_limit": True},
        {"memory_limit_mb": False},
        {"pids_limit": True},
        {"tmpfs_limit_mb": False},
        {"max_output_bytes": True},
        {"max_artifact_bytes": False},
    ],
)
def test_requested_resources_reject_boolean_numbers(
    resources: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "cpu_limit": 1.0,
        "memory_limit_mb": 1_024,
        "pids_limit": 256,
        "tmpfs_limit_mb": 256,
        "max_output_bytes": 1_000_000,
        "max_artifact_bytes": 20_000_000,
    }
    values.update(resources)

    with pytest.raises(ValueError):
        RequestedSandboxResources.model_validate(values)
