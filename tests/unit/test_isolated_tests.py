"""Regression isolation, redaction, command identity and fail-closed mirror tests."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from repopilot.enums import ErrorType, PolicyDecision, SandboxCommandPhase, SandboxLifecycle
from repopilot.enums import TestStatus as ResultStatus
from repopilot.evals.isolated_tests import IsolatedTestSandbox, IsolatedTestSandboxError
from repopilot.repair_workflow import RepairSandbox
from repopilot.sandbox.artifacts import SandboxArtifactStore
from repopilot.sandbox.base import ApplyPatchRequest, ReadFileRequest, RunRequest, SandboxHandle
from repopilot.schemas import ArtifactRef, CommandSpec, Observation, TaskSpec
from repopilot.tools.tests import interpret_test_observation

HIDDEN = "PRIVATE_REGRESSION_SOURCE_CANARY"
REGRESSION = (
    "diff --git a/tests/test_bug.py b/tests/test_bug.py\n"
    "--- a/tests/test_bug.py\n"
    "+++ b/tests/test_bug.py\n"
    "@@ -1 +1,2 @@\n"
    " pass\n"
    f'+assert False, "{HIDDEN}"\n'
)
CANDIDATE = (
    b"diff --git a/code.py b/code.py\n"
    b"--- a/code.py\n"
    b"+++ b/code.py\n"
    b"@@ -1 +1 @@\n"
    b"-answer = 0\n"
    b"+answer = 1\n"
)
RAW_FAILURE = (
    f'assert False, "{HIDDEN}"\nTraceback {HIDDEN}\n'
    "=== short test summary info ===\n"
    f"FAILED tests/test_bug.py::test_bug[secret-{HIDDEN}] - AssertionError: {HIDDEN}\n"
    "=== 1 failed in 0.01s ===\n"
)


class RecordingSandbox:
    def __init__(self, root: Path) -> None:
        self.store = SandboxArtifactStore(root)
        self.root = root
        self.task: TaskSpec | None = None
        self.handle: SandboxHandle | None = None
        self.commands: list[RunRequest] = []
        self.reads: list[ReadFileRequest] = []
        self.patches: list[bytes] = []
        self.applied = 0
        self.output = RAW_FAILURE
        self.exit_code = 1
        self.fail_prepare = False
        self.fail_apply = False
        self.fail_setup = False
        self._lifecycle = SandboxLifecycle.NEW

    @property
    def lifecycle(self) -> SandboxLifecycle:
        return self._lifecycle

    def prepare(self, task: TaskSpec) -> SandboxHandle:
        self.task = task
        if self.fail_prepare:
            raise RuntimeError(HIDDEN)
        self.handle = SandboxHandle(
            sandbox_id="recording-" + uuid4().hex,
            task_id=task.task_id,
            repository_commit=task.base_commit,
            prepared_at=datetime.now(UTC),
        )
        self.store.prepare(
            self.handle.sandbox_id, repository_source=Path(task.repository_path or "")
        )
        self._lifecycle = SandboxLifecycle.PREPARED
        return self.handle

    def run(self, handle: SandboxHandle, request: RunRequest) -> Observation:
        assert handle == self.handle
        self.commands.append(request)
        if request.phase is SandboxCommandPhase.SETUP:
            return self.observation(request.call_id, HIDDEN, 1 if self.fail_setup else 0)
        return self.observation(request.call_id, self.output, self.exit_code)

    def read_file(self, handle: SandboxHandle, request: ReadFileRequest) -> Observation:
        assert handle == self.handle
        self.reads.append(request)
        return self.observation(request.call_id, "clean source", 0)

    def stage_patch(self, handle: SandboxHandle, call_id: UUID, content: bytes) -> ArtifactRef:
        assert handle == self.handle
        self.patches.append(content)
        return self.store.persist_patch(call_id, content)

    def apply_patch(self, handle: SandboxHandle, request: ApplyPatchRequest) -> Observation:
        assert handle == self.handle
        self.applied += 1
        return self.observation(
            request.call_id, HIDDEN if self.fail_apply else "applied", 1 if self.fail_apply else 0
        )

    def collect_artifacts(self, handle: SandboxHandle) -> tuple[ArtifactRef, ...]:
        assert handle == self.handle
        return self.store.collect()

    def destroy(self, handle: SandboxHandle) -> None:
        assert handle == self.handle
        self._lifecycle = SandboxLifecycle.DESTROYED

    def observation(self, call_id: UUID, output: str, code: int) -> Observation:
        assert self.handle is not None
        return Observation(
            call_id=call_id,
            started_at=datetime.now(UTC),
            duration_ms=1,
            exit_code=code,
            stdout_ref=self.store.persist(call_id, "stdout", output.encode()),
            stderr_ref=self.store.persist(call_id, "stderr", b""),
            stdout_preview=output,
            stderr_preview="",
            sandbox_id=self.handle.sandbox_id,
            policy_decision=PolicyDecision.ALLOWED,
        )


@pytest.fixture
def task(tmp_path: Path) -> TaskSpec:
    source = tmp_path / "source"
    source.mkdir()
    (source / "tests").mkdir()
    (source / "tests/test_bug.py").write_text("pass\n")
    (source / "code.py").write_bytes(b"answer = 0\n")
    for args in (
        ("init", "--quiet"),
        ("add", "."),
        (
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "Fixture",
        ),
    ):
        subprocess.run(("git", "-C", str(source), *args), check=True, capture_output=True)
    commit = subprocess.run(
        ("git", "-C", str(source), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return TaskSpec(
        task_id="isolated-unit",
        repository_path=str(source),
        base_commit=commit,
        issue_title="Repair fixture",
        issue_body="Fix the source without exposing regression implementation.",
        setup_commands=(CommandSpec(argv=("python", "-m", "pip", "list")),),
        test_commands=(CommandSpec(argv=("python", "-m", "pytest", "-q", "tests/test_bug.py")),),
    )


def build(
    task: TaskSpec,
    tmp_path: Path,
    *,
    regression: str = REGRESSION,
    evaluator_failure: str | None = None,
) -> tuple[IsolatedTestSandbox, list[RecordingSandbox], list[RecordingSandbox]]:
    agents: list[RecordingSandbox] = []
    evaluators: list[RecordingSandbox] = []

    def agent_factory(root: Path) -> RepairSandbox:
        sandbox = RecordingSandbox(root)
        agents.append(sandbox)
        return sandbox

    def evaluator_factory(root: Path) -> RepairSandbox:
        sandbox = RecordingSandbox(root)
        if evaluator_failure == "prepare":
            sandbox.fail_prepare = True
        evaluators.append(sandbox)
        return sandbox

    return (
        IsolatedTestSandbox(
            task=task,
            regression_patch=regression,
            artifact_root=tmp_path / "public/artifacts",
            agent_factory=agent_factory,
            evaluator_factory=evaluator_factory,
        ),
        agents,
        evaluators,
    )


def test_separate_checkouts_only_agent_reads_and_cleanup(task: TaskSpec, tmp_path: Path) -> None:
    wrapper, agents, evaluators = build(task, tmp_path)
    handle = wrapper.prepare(task)
    assert agents[0].task == task
    assert evaluators[0].task is not None
    private_checkout = Path(evaluators[0].task.repository_path or "")
    assert HIDDEN in (private_checkout / "tests/test_bug.py").read_text()
    assert HIDDEN not in Path(task.repository_path or "").joinpath("tests/test_bug.py").read_text()
    request = ReadFileRequest(call_id=uuid4(), relative_path="tests/test_bug.py")
    assert wrapper.read_file(handle, request).stdout_preview == "clean source"
    assert agents[0].reads == [request] and evaluators[0].reads == []
    wrapper.destroy(handle)
    wrapper.destroy(handle)
    assert not private_checkout.parent.exists()
    assert wrapper.lifecycle is SandboxLifecycle.DESTROYED
    assert agents[0].lifecycle is evaluators[0].lifecycle is SandboxLifecycle.DESTROYED


def test_sanitized_result_is_parseable_and_raw_is_not_public(
    task: TaskSpec, tmp_path: Path
) -> None:
    wrapper, agents, evaluators = build(task, tmp_path)
    handle = wrapper.prepare(task)
    result = wrapper.run(handle, RunRequest(call_id=uuid4(), command=task.test_commands[0]))
    assert result.sandbox_id == handle.sandbox_id
    assert HIDDEN not in result.model_dump_json()
    assert "Traceback" not in result.model_dump_json()
    parsed = interpret_test_observation(result, task.test_commands[0], command_index=0)
    assert parsed.status is ResultStatus.FAILED and parsed.failures == 1
    assert parsed.failing_tests[0].startswith("tests/test_bug.py::test_bug[case-")
    assert not agents[0].commands
    assert len(evaluators[0].commands) == 1
    for reference in wrapper.collect_artifacts(handle):
        content = (tmp_path / "public" / reference.relative_path).read_bytes()
        assert HIDDEN.encode() not in content
        assert hashlib.sha256(content).hexdigest() == reference.sha256
    wrapper.destroy(handle)


@pytest.mark.parametrize("kind", ("argv", "timeout", "cwd", "network"))
def test_only_exact_frozen_test_command_allowed(task: TaskSpec, tmp_path: Path, kind: str) -> None:
    wrapper, agents, evaluators = build(task, tmp_path)
    handle = wrapper.prepare(task)
    command = task.test_commands[0]
    if kind == "argv":
        command = command.model_copy(update={"argv": ("python", "-c", "print('secret')")})
    elif kind == "timeout":
        command = command.model_copy(update={"timeout_seconds": 2})
    elif kind == "cwd":
        command = command.model_copy(update={"cwd": "tests"})
    result = wrapper.run(
        handle, RunRequest(call_id=uuid4(), command=command, network_requested=kind == "network")
    )
    assert result.policy_decision is PolicyDecision.DENIED
    assert not agents[0].commands and not evaluators[0].commands
    wrapper.destroy(handle)


def test_candidate_mirrors_only_after_agent_success(task: TaskSpec, tmp_path: Path) -> None:
    wrapper, agents, evaluators = build(task, tmp_path)
    handle = wrapper.prepare(task)
    call_id = uuid4()
    reference = wrapper.stage_patch(handle, call_id, CANDIDATE)
    result = wrapper.apply_patch(handle, ApplyPatchRequest(call_id=call_id, patch_ref=reference))
    assert result.exit_code == 0 and result.error_type is None
    assert agents[0].patches == evaluators[0].patches == [CANDIDATE]
    assert agents[0].applied == evaluators[0].applied == 1
    wrapper.destroy(handle)


def test_candidate_cannot_overwrite_hidden_regression(task: TaskSpec, tmp_path: Path) -> None:
    wrapper, agents, evaluators = build(task, tmp_path)
    handle = wrapper.prepare(task)
    candidate = REGRESSION.replace(HIDDEN, "candidate-only").encode()
    call_id = uuid4()
    reference = wrapper.stage_patch(handle, call_id, candidate)
    result = wrapper.apply_patch(handle, ApplyPatchRequest(call_id=call_id, patch_ref=reference))
    assert result.error_type is ErrorType.PATCH_ERROR
    assert result.exit_code == 44
    assert result.stderr_preview == "repopilot_patch_error:protected_test_path\n"
    assert HIDDEN not in result.model_dump_json()
    assert "tests/test_bug.py" not in result.model_dump_json()
    assert not agents[0].applied and not evaluators[0].applied
    assert not evaluators[0].patches

    # The rejected whole candidate leaves both source workspaces untouched and usable.
    corrected_call_id = uuid4()
    corrected = wrapper.stage_patch(handle, corrected_call_id, CANDIDATE)
    accepted = wrapper.apply_patch(
        handle, ApplyPatchRequest(call_id=corrected_call_id, patch_ref=corrected)
    )
    assert accepted.exit_code == 0 and accepted.error_type is None
    assert agents[0].applied == evaluators[0].applied == 1
    assert evaluators[0].patches == [CANDIDATE]
    wrapper.destroy(handle)


def test_mirror_failure_poisons_session_and_redacts_evaluator(
    task: TaskSpec, tmp_path: Path
) -> None:
    wrapper, _, evaluators = build(task, tmp_path)
    handle = wrapper.prepare(task)
    evaluators[0].fail_apply = True
    call_id = uuid4()
    reference = wrapper.stage_patch(handle, call_id, CANDIDATE)
    result = wrapper.apply_patch(handle, ApplyPatchRequest(call_id=call_id, patch_ref=reference))
    assert result.error_type is ErrorType.SANDBOX_ERROR
    assert HIDDEN not in result.model_dump_json()
    with pytest.raises(IsolatedTestSandboxError):
        wrapper.run(handle, RunRequest(call_id=uuid4(), command=task.test_commands[0]))
    wrapper.destroy(handle)


def test_failed_agent_patch_is_never_mirrored(task: TaskSpec, tmp_path: Path) -> None:
    wrapper, agents, evaluators = build(task, tmp_path)
    handle = wrapper.prepare(task)
    agents[0].fail_apply = True
    call_id = uuid4()
    reference = wrapper.stage_patch(handle, call_id, CANDIDATE)
    result = wrapper.apply_patch(handle, ApplyPatchRequest(call_id=call_id, patch_ref=reference))
    assert result.exit_code == 1
    assert not evaluators[0].patches and not evaluators[0].applied
    wrapper.destroy(handle)


def test_evaluator_setup_failure_returns_no_private_output(task: TaskSpec, tmp_path: Path) -> None:
    wrapper, agents, evaluators = build(task, tmp_path)
    handle = wrapper.prepare(task)
    evaluators[0].fail_setup = True
    result = wrapper.run(
        handle,
        RunRequest(
            call_id=uuid4(), command=task.setup_commands[0], phase=SandboxCommandPhase.SETUP
        ),
    )
    assert result.error_type is ErrorType.SANDBOX_ERROR
    assert HIDDEN not in result.model_dump_json()
    assert len(agents[0].commands) == len(evaluators[0].commands) == 1
    wrapper.destroy(handle)


def test_prepare_failure_cleans_agent_and_temporary_root(task: TaskSpec, tmp_path: Path) -> None:
    wrapper, agents, evaluators = build(task, tmp_path, evaluator_failure="prepare")
    with pytest.raises(IsolatedTestSandboxError) as error:
        wrapper.prepare(task)
    assert HIDDEN not in str(error.value)
    assert agents[0].lifecycle is SandboxLifecycle.DESTROYED
    assert not evaluators[0].root.parent.exists()


def test_dirty_checkout_rejected_before_sandbox_creation(task: TaskSpec, tmp_path: Path) -> None:
    wrapper, agents, evaluators = build(task, tmp_path)
    Path(task.repository_path or "").joinpath("code.py").write_text("answer = 2\n")
    with pytest.raises(IsolatedTestSandboxError):
        wrapper.prepare(task)
    assert not agents and not evaluators


@pytest.mark.parametrize("path", (".", "../outside", ".git/config", "/etc/file", "a/../b", "a\\b"))
def test_unsafe_regression_paths_are_rejected(task: TaskSpec, tmp_path: Path, path: str) -> None:
    with pytest.raises(IsolatedTestSandboxError):
        build(task, tmp_path, regression=REGRESSION.replace("tests/test_bug.py", path))


def test_import_failure_is_not_recast_as_reproduction(task: TaskSpec, tmp_path: Path) -> None:
    wrapper, _, evaluators = build(task, tmp_path)
    handle = wrapper.prepare(task)
    evaluators[0].output = (
        f"ImportError: {HIDDEN}\n"
        "=== short test summary info ===\n"
        "ERROR tests/test_bug.py\n"
        "=== 1 error in 0.01s ===\n"
    )
    evaluators[0].exit_code = 2
    result = wrapper.run(handle, RunRequest(call_id=uuid4(), command=task.test_commands[0]))
    parsed = interpret_test_observation(result, task.test_commands[0], command_index=0)
    assert parsed.status is ResultStatus.ERROR
    assert result.error_type is ErrorType.TEST_ERROR
    assert HIDDEN not in result.model_dump_json()
    wrapper.destroy(handle)


def test_unittest_summary_survives_without_source(task: TaskSpec, tmp_path: Path) -> None:
    task = task.model_copy(
        update={"test_commands": (CommandSpec(argv=("python", "-m", "unittest")),)}
    )
    wrapper, _, evaluators = build(task, tmp_path)
    handle = wrapper.prepare(task)
    evaluators[0].output = (
        f"FAIL: test_bug (tests.test_bug.TestBug)\nTraceback: {HIDDEN}\n"
        "Ran 1 test in 0.01s\nFAILED (failures=1)\n"
    )
    result = wrapper.run(handle, RunRequest(call_id=uuid4(), command=task.test_commands[0]))
    parsed = interpret_test_observation(result, task.test_commands[0], command_index=0)
    assert parsed.status is ResultStatus.FAILED and parsed.tests_run == 1
    assert HIDDEN not in result.model_dump_json()
    wrapper.destroy(handle)


def test_summary_tampering_fails_closed(
    task: TaskSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper, _, evaluators = build(task, tmp_path)
    handle = wrapper.prepare(task)
    original = evaluators[0].observation

    def tampered(call_id: UUID, output: str, code: int) -> Observation:
        observation = original(call_id, output, code)
        assert observation.stdout_ref is not None
        path = evaluators[0].root.parent / observation.stdout_ref.relative_path
        path.write_text("=== 1 passed in 0.01s ===\n")
        return observation

    monkeypatch.setattr(evaluators[0], "observation", tampered)
    result = wrapper.run(handle, RunRequest(call_id=uuid4(), command=task.test_commands[0]))
    assert result.error_type is ErrorType.SANDBOX_ERROR
    assert result.stdout_ref is None
    wrapper.destroy(handle)


def test_replayed_call_and_wrong_handle_are_rejected(task: TaskSpec, tmp_path: Path) -> None:
    wrapper, _, _ = build(task, tmp_path)
    handle = wrapper.prepare(task)
    request = ReadFileRequest(call_id=uuid4(), relative_path="code.py")
    wrapper.read_file(handle, request)
    with pytest.raises(IsolatedTestSandboxError):
        wrapper.read_file(handle, request)
    with pytest.raises(IsolatedTestSandboxError):
        wrapper.collect_artifacts(handle.model_copy(update={"task_id": "not-the-task"}))
    wrapper.destroy(handle)


def test_prepare_task_identity_cannot_change(task: TaskSpec, tmp_path: Path) -> None:
    wrapper, agents, evaluators = build(task, tmp_path)
    with pytest.raises(IsolatedTestSandboxError):
        wrapper.prepare(task.model_copy(update={"test_commands": (CommandSpec(argv=("true",)),)}))
    assert not agents and not evaluators


def test_regression_symlink_mode_rejected(task: TaskSpec, tmp_path: Path) -> None:
    patch = REGRESSION.replace("--- a/", "new file mode 120000\n--- a/", 1)
    with pytest.raises(IsolatedTestSandboxError):
        build(task, tmp_path, regression=patch)


def test_no_host_hook_runs_when_preparing_regression(task: TaskSpec, tmp_path: Path) -> None:
    source = Path(task.repository_path or "")
    marker = tmp_path / "hook-was-run"
    hook = source / ".git/hooks/post-checkout"
    hook.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    hook.chmod(0o700)
    subprocess.run(
        ("git", "-C", str(source), "config", "core.hooksPath", str(hook.parent)),
        check=True,
        capture_output=True,
    )
    wrapper, _, _ = build(task, tmp_path)
    handle = wrapper.prepare(task)
    assert not marker.exists()
    wrapper.destroy(handle)


@pytest.mark.parametrize(
    ("rows", "summary", "expected_failures", "expected_errors"),
    (
        (f"FAILED tests/test_bug.py::test_bug[{HIDDEN} alpha - beta]", "1 failed", 1, 0),
        (
            "FAILED tests/test_bug.py::test_bug\nERROR tests/test_bug.py::test_bug",
            "1 failed, 1 error",
            1,
            1,
        ),
        ("FAILED tests/☃.py::test_bug", "1 failed", 1, 0),
        (
            "FAILED tests/test_bug.py::test_bug[first]\nFAILED tests/test_bug.py::test_bug[second]",
            "2 failed",
            2,
            0,
        ),
    ),
)
def test_pytest_projection_preserves_outcomes_without_parameter_values(
    task: TaskSpec,
    tmp_path: Path,
    rows: str,
    summary: str,
    expected_failures: int,
    expected_errors: int,
) -> None:
    task = task.model_copy(
        update={"test_commands": (CommandSpec(argv=("python", "-B", "-m", "pytest", "-q")),)}
    )
    wrapper, _, evaluators = build(task, tmp_path)
    handle = wrapper.prepare(task)
    evaluators[0].output = (
        f"assert False, '{HIDDEN}'\nFAILED fake.py::test_fake - {HIDDEN}\n"
        "\x1b[31m=== short test summary info ===\x1b[0m\n"
        f"{rows}\n=== {summary} in 0.1s ===\n"
    )
    result = wrapper.run(handle, RunRequest(call_id=uuid4(), command=task.test_commands[0]))
    parsed = interpret_test_observation(result, task.test_commands[0], command_index=0)
    assert parsed.status is ResultStatus.FAILED
    assert parsed.failures == expected_failures and parsed.errors == expected_errors
    assert HIDDEN not in result.model_dump_json()
    assert "fake.py" not in result.model_dump_json()
    assert "alpha" not in result.model_dump_json()
    assert all("::" in name for name in parsed.failing_tests)
    wrapper.destroy(handle)


@pytest.mark.parametrize("reason", ("truncated", "typed_error"))
def test_projection_does_not_turn_infrastructure_errors_into_test_failures(
    task: TaskSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    wrapper, _, evaluators = build(task, tmp_path)
    handle = wrapper.prepare(task)
    original = evaluators[0].observation

    def failed(call_id: UUID, output: str, code: int) -> Observation:
        observation = original(call_id, output, code)
        return observation.model_copy(
            update={"truncated": True}
            if reason == "truncated"
            else {"error_type": ErrorType.TEST_ERROR}
        )

    monkeypatch.setattr(evaluators[0], "observation", failed)
    result = wrapper.run(handle, RunRequest(call_id=uuid4(), command=task.test_commands[0]))
    parsed = interpret_test_observation(result, task.test_commands[0], command_index=0)
    assert parsed.status is ResultStatus.ERROR
    assert HIDDEN not in result.model_dump_json()
    wrapper.destroy(handle)


@pytest.mark.parametrize(
    "exception", ("TypeError", "ValueError", "AttributeError", HIDDEN + "Error")
)
def test_only_allowlisted_exception_type_crosses_isolation_boundary(
    task: TaskSpec, tmp_path: Path, exception: str
) -> None:
    task = task.model_copy(update={"test_commands": (CommandSpec(argv=("pytest", "-q")),)})
    wrapper, _, evaluators = build(task, tmp_path)
    handle = wrapper.prepare(task)
    evaluators[0].output = (
        f"private traceback {HIDDEN}\n"
        "=== short test summary info ===\n"
        f"FAILED tests/test_bug.py::test_bug[{HIDDEN} - TypeError] - {exception}: {HIDDEN}\n"
        f"ERROR tests/test_bug.py::test_teardown - KeyError: {HIDDEN}\n"
        "=== 1 failed, 1 error in 0.1s ===\n"
    )
    result = wrapper.run(handle, RunRequest(call_id=uuid4(), command=task.test_commands[0]))
    public = result.model_dump_json()
    assert HIDDEN not in public
    assert " - KeyError" in public
    assert (f" - {exception}" in public) is (not exception.startswith(HIDDEN))
    parsed = interpret_test_observation(result, task.test_commands[0], command_index=0)
    assert parsed.status is ResultStatus.FAILED
    assert (parsed.tests_run, parsed.failures, parsed.errors) == (2, 1, 1)
    wrapper.destroy(handle)


def test_exit_zero_without_executed_tests_is_not_success(task: TaskSpec, tmp_path: Path) -> None:
    task = task.model_copy(
        update={"test_commands": (CommandSpec(argv=("python", "-m", "unittest")),)}
    )
    wrapper, _, evaluators = build(task, tmp_path)
    handle = wrapper.prepare(task)
    evaluators[0].output = "OK\n"
    evaluators[0].exit_code = 0
    result = wrapper.run(handle, RunRequest(call_id=uuid4(), command=task.test_commands[0]))
    assert result.error_type is ErrorType.TEST_ERROR
    wrapper.destroy(handle)
