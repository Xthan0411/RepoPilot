"""Restricted tools backed by an already prepared Sandbox adapter."""

from __future__ import annotations

from pydantic import BaseModel, Field

from repopilot.sandbox.base import RunRequest, SandboxAdapter, SandboxHandle
from repopilot.schemas import (
    Observation,
    StrictModel,
    TaskSpec,
    TestRunResult,
    ToolCall,
)
from repopilot.tools.registry import ToolArgumentError
from repopilot.tools.tests import interpret_test_observation


class RunTestsArguments(StrictModel):
    """Select one task-owned test command without accepting an argv from the Agent."""

    test_index: int = Field(default=0, ge=0, strict=True)


class RunTestsResult(StrictModel):
    """Artifact-backed Sandbox observation for one configured test command."""

    observation: Observation
    test_run: TestRunResult


class RunTestsTool:
    """Run only a command already present in ``TaskSpec.test_commands``."""

    name = "run_tests"
    description = "Run one task-configured test command in the prepared Sandbox."
    arguments_model: type[BaseModel] = RunTestsArguments

    def __init__(
        self,
        *,
        sandbox: SandboxAdapter,
        handle: SandboxHandle,
        task: TaskSpec,
    ) -> None:
        if handle.task_id != task.task_id or handle.repository_commit != task.base_commit:
            raise ValueError("sandbox handle does not match task")
        self._sandbox = sandbox
        self._handle = handle
        self._task = task

    @classmethod
    def prepare(
        cls,
        *,
        sandbox: SandboxAdapter,
        task: TaskSpec,
    ) -> RunTestsTool:
        """Prepare the injected Sandbox once and bind it to one task."""

        return cls(
            sandbox=sandbox,
            handle=sandbox.prepare(task),
            task=task,
        )

    def execute(self, arguments: BaseModel, *, call: ToolCall) -> BaseModel:
        if not isinstance(arguments, RunTestsArguments):
            raise TypeError("unexpected argument model")
        if arguments.test_index >= len(self._task.test_commands):
            raise ToolArgumentError("test_index exceeds configured commands")
        request = RunRequest(
            call_id=call.call_id,
            command=self._task.test_commands[arguments.test_index],
        )
        observation = self._sandbox.run(self._handle, request)
        return RunTestsResult(
            observation=observation,
            test_run=interpret_test_observation(
                observation,
                request.command,
                command_index=arguments.test_index,
            ),
        )
