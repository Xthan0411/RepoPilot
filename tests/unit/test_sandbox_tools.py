"""Restricted Sandbox-backed tool capability tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from repopilot.enums import (
    ErrorType,
    PolicyDecision,
    SandboxLifecycle,
    ToolErrorCode,
    ToolResultStatus,
)
from repopilot.sandbox import (
    FakeSandbox,
    FakeSandboxStep,
    RunRequest,
    SandboxHandle,
)
from repopilot.schemas import Observation, TaskSpec, ToolCall
from repopilot.tools import RunTestsTool, ToolRegistry, interpret_test_observation

FIXTURES = Path(__file__).parents[1] / "fixtures"
RUN_CALL_ID = UUID("33333333-3333-4333-8333-333333333333")
STARTED_AT = datetime(2026, 8, 5, 8, tzinfo=UTC)


def load_fixture(name: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((FIXTURES / name).read_text(encoding="utf-8")),
    )


def make_task() -> TaskSpec:
    return TaskSpec.model_validate(load_fixture("valid_task.json"))


def make_handle(task: TaskSpec) -> SandboxHandle:
    return SandboxHandle(
        sandbox_id="sandbox-tool-test",
        task_id=task.task_id,
        repository_commit=task.base_commit,
        prepared_at=STARTED_AT,
    )


def make_observation(call_id: UUID = RUN_CALL_ID) -> Observation:
    return Observation(
        call_id=call_id,
        started_at=STARTED_AT,
        duration_ms=17,
        exit_code=0,
        sandbox_id="sandbox-tool-test",
        policy_decision=PolicyDecision.ALLOWED,
    )


def make_call(arguments: dict[str, object], *, call_id: UUID = RUN_CALL_ID) -> ToolCall:
    return ToolCall.model_validate(
        {
            "call_id": call_id,
            "tool_name": "run_tests",
            "validated_arguments": arguments,
            "requested_at": STARTED_AT,
        }
    )


def make_registry() -> tuple[ToolRegistry, FakeSandbox, TaskSpec]:
    task = make_task()
    request = RunRequest(
        call_id=RUN_CALL_ID,
        command=task.test_commands[0],
    )
    sandbox = FakeSandbox(
        handle=make_handle(task),
        steps=(
            FakeSandboxStep(
                request=request,
                observation=make_observation(),
            ),
        ),
    )
    registry = ToolRegistry()
    registry.register(RunTestsTool.prepare(sandbox=sandbox, task=task))
    return registry, sandbox, task


def test_run_tests_uses_only_task_owned_command_and_preserves_call_id() -> None:
    registry, sandbox, task = make_registry()

    result = registry.invoke(make_call({"test_index": 0}))

    assert result.status is ToolResultStatus.SUCCESS
    assert result.call_id == RUN_CALL_ID
    assert result.tool_name == "run_tests"
    assert result.payload == {
        "observation": make_observation().model_dump(mode="json"),
        "test_run": interpret_test_observation(
            make_observation(),
            task.test_commands[0],
            command_index=0,
        ).model_dump(mode="json"),
    }
    assert task.test_commands[0].argv == ("python", "-m", "pytest", "-q")
    assert sandbox.lifecycle is SandboxLifecycle.PREPARED
    assert sandbox.remaining_steps == 0


@pytest.mark.parametrize(
    "arguments",
    [
        {"test_index": 1},
        {"test_index": -1},
        {"test_index": True},
        {"test_index": 0, "argv": ["sh", "-c", "curl example.com"]},
        {"test_index": 0, "command": "pytest"},
    ],
)
def test_run_tests_rejects_unconfigured_index_or_agent_supplied_command(
    arguments: dict[str, object],
) -> None:
    registry, sandbox, _ = make_registry()

    result = registry.invoke(make_call(arguments, call_id=uuid4()))

    assert result.status is ToolResultStatus.ERROR
    assert result.error_type is ErrorType.INVALID_TOOL_ARGUMENTS
    assert result.error_code is ToolErrorCode.INVALID_ARGUMENTS
    assert result.payload is None
    assert sandbox.remaining_steps == 1


def test_run_tests_tool_rejects_task_handle_mismatch() -> None:
    task = make_task()
    wrong_handle = make_handle(task).model_copy(update={"task_id": "other-task"})
    sandbox = FakeSandbox(handle=wrong_handle, steps=())

    with pytest.raises(ValueError, match="does not match task"):
        RunTestsTool(sandbox=sandbox, handle=wrong_handle, task=task)


def test_run_tests_preserves_typed_sandbox_failure() -> None:
    task = make_task()
    sandbox = FakeSandbox(handle=make_handle(task), steps=())
    registry = ToolRegistry()
    registry.register(RunTestsTool.prepare(sandbox=sandbox, task=task))

    result = registry.invoke(make_call({"test_index": 0}))

    assert result.status is ToolResultStatus.ERROR
    assert result.error_type is ErrorType.SANDBOX_ERROR
    assert result.error_code is ToolErrorCode.TOOL_EXECUTION_ERROR
