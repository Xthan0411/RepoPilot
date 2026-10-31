"""Offline production repair graph + verifier tests with trusted local fixtures.

The subprocess adapter below is test-only, accepts one hard-coded fixture command,
and must never be used to execute real benchmark repositories on the host.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4, uuid5

import pytest
from pydantic import JsonValue

from repopilot.agent import repair_decision_request_id
from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    PolicyDecision,
    ReproductionStatus,
    RunStatus,
    SandboxLifecycle,
)
from repopilot.enums import TestRunCategory as RunTestCategory
from repopilot.enums import TestStatus as RunTestStatus
from repopilot.evals.isolated_tests import IsolatedTestSandbox
from repopilot.providers import (
    FakeProvider,
    FinishAction,
    OpenAIProvider,
    ProviderResponse,
    ProviderUsageDelta,
    ToolAction,
)
from repopilot.repair_workflow import RepairRunRequest, RepairSandbox, run_repair_request
from repopilot.sandbox import (
    ApplyPatchRequest,
    ReadFileRequest,
    RunRequest,
    SandboxArtifactStore,
    SandboxHandle,
)
from repopilot.schemas import ArtifactRef, CommandSpec, Observation, TaskSpec, ToolCall

STARTED_AT = datetime(2026, 10, 28, 12, tzinfo=UTC)
RUN_ID = UUID("d27836a7-b84e-45d8-b0b2-100dc1d94b10")
FIXTURE_ID = "isolated-pytest-fixture"
PRIVATE_MARKER = "EVALUATOR_PRIVATE_ASSERTION_MUST_NOT_REACH_AGENT"
TEST_ARGV = ("python", "-m", "pytest", "-q", "tests/test_regression.py")
SOURCE = "def increment(value):\n    return value - 1\n"
REGRESSION_PATCH = (
    "diff --git a/tests/test_regression.py b/tests/test_regression.py\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/tests/test_regression.py\n"
    "@@ -0,0 +1,4 @@\n"
    "+from calculator import increment\n"
    "+\n"
    "+def test_increment():\n"
    f'+    assert increment(1) == 2, "{PRIVATE_MARKER}"\n'
)


class _TrustedFixtureSandbox:
    """Run only the tiny test-owned fixture, with no inherited secrets or plugins."""

    def __init__(self, artifact_root: Path, workspace_root: Path) -> None:
        self._store = SandboxArtifactStore(artifact_root)
        self.workspace = workspace_root / uuid4().hex
        self._lifecycle = SandboxLifecycle.NEW
        self._handle: SandboxHandle | None = None
        self.source_files: tuple[str, ...] = ()
        self.raw_test_outputs: list[str] = []

    @property
    def lifecycle(self) -> SandboxLifecycle:
        return self._lifecycle

    def prepare(self, task: TaskSpec) -> SandboxHandle:
        assert task.task_id == FIXTURE_ID
        assert task.repository_path is not None
        source = Path(task.repository_path)
        assert source.joinpath("calculator.py").read_text() == SOURCE
        shutil.copytree(source, self.workspace, ignore=shutil.ignore_patterns(".git"))
        self.source_files = tuple(
            sorted(str(path.relative_to(self.workspace)) for path in self.workspace.rglob("*"))
        )
        self._handle = SandboxHandle(
            sandbox_id=f"trusted-fixture-{uuid4().hex}",
            task_id=task.task_id,
            repository_commit=task.base_commit,
            prepared_at=STARTED_AT,
        )
        self._store.prepare(self._handle.sandbox_id, repository_source=source)
        self._lifecycle = SandboxLifecycle.PREPARED
        return self._handle

    def run(self, handle: SandboxHandle, request: RunRequest) -> Observation:
        assert handle == self._handle
        assert request.command.argv == TEST_ARGV
        assert not request.network_requested
        completed = subprocess.run(
            (sys.executable, "-B", *request.command.argv[1:]),
            cwd=self.workspace,
            env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            timeout=15,
            check=False,
        )
        self.raw_test_outputs.append((completed.stdout + completed.stderr).decode())
        return self._observation(
            request.call_id, completed.returncode, completed.stdout, completed.stderr
        )

    def read_file(self, handle: SandboxHandle, request: ReadFileRequest) -> Observation:
        assert handle == self._handle
        path = (self.workspace / request.relative_path).resolve(strict=True)
        assert path.is_relative_to(self.workspace)
        assert not path.is_symlink()
        lines = path.read_bytes().splitlines(keepends=True)
        content = b"".join(lines[request.start_line - 1 : request.end_line])
        return self._observation(request.call_id, 0, content, b"")

    def stage_patch(self, handle: SandboxHandle, call_id: UUID, content: bytes) -> ArtifactRef:
        assert handle == self._handle
        return self._store.persist_patch(call_id, content)

    def apply_patch(self, handle: SandboxHandle, request: ApplyPatchRequest) -> Observation:
        assert handle == self._handle
        completed = subprocess.run(
            ("git", "apply", "-"),
            input=self._store.read(request.patch_ref),
            cwd=self.workspace,
            capture_output=True,
            timeout=10,
            check=False,
        )
        return self._observation(
            request.call_id, completed.returncode, completed.stdout, completed.stderr
        )

    def collect_artifacts(self, handle: SandboxHandle) -> tuple[ArtifactRef, ...]:
        assert handle == self._handle
        return self._store.collect()

    def destroy(self, handle: SandboxHandle) -> None:
        assert handle == self._handle
        self._lifecycle = SandboxLifecycle.DESTROYED

    def _observation(
        self, call_id: UUID, exit_code: int, stdout: bytes, stderr: bytes
    ) -> Observation:
        assert self._handle is not None
        return Observation(
            call_id=call_id,
            started_at=STARTED_AT,
            duration_ms=1,
            exit_code=exit_code,
            stdout_ref=self._store.persist(call_id, "stdout", stdout),
            stderr_ref=self._store.persist(call_id, "stderr", stderr),
            stdout_preview=stdout.decode(),
            stderr_preview=stderr.decode(),
            sandbox_id=self._handle.sandbox_id,
            policy_decision=PolicyDecision.ALLOWED,
        )


def _fixture_task(tmp_path: Path) -> TaskSpec:
    repository = tmp_path / "source"
    repository.mkdir()
    repository.joinpath("calculator.py").write_text(SOURCE)
    for argv in (
        ("git", "init", "--quiet"),
        ("git", "add", "calculator.py"),
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
            "Trusted fixture",
        ),
    ):
        subprocess.run(argv, cwd=repository, capture_output=True, check=True)
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repository, capture_output=True, text=True, check=True
    ).stdout.strip()
    return TaskSpec(
        task_id=FIXTURE_ID,
        repository_path=str(repository),
        base_commit=revision,
        issue_title="Increment subtracts instead of adding",
        issue_body="calculator.increment(1) should return 2.",
        test_commands=(CommandSpec(argv=TEST_ARGV, timeout_seconds=15),),
        timeout_seconds=60,
    )


def _candidate_patch(*, correct_patch: bool) -> str:
    return (
        "diff --git a/calculator.py b/calculator.py\n"
        "--- a/calculator.py\n"
        "+++ b/calculator.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def increment(value):\n"
        "-    return value - 1\n"
        f"+    return value + {1 if correct_patch else 0}\n"
    )


def _responses(*, correct_patch: bool) -> tuple[ProviderResponse, ...]:
    patch = _candidate_patch(correct_patch=correct_patch)
    usage = ProviderUsageDelta(
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        estimated_cost_usd=Decimal("0"),
        latency_ms=1,
    )
    steps: tuple[tuple[str, dict[str, JsonValue], tuple[str, ...]], ...] = (
        ("run_tests", {"test_index": 0}, ("run_tests",)),
        ("search_code", {"query": "increment"}, ("search_code",)),
        ("read_file", {"path": "calculator.py"}, ("inspect_symbol", "read_file")),
        ("apply_patch", {"diff": patch}, ("apply_patch",)),
        ("run_tests", {"test_index": 0}, ("run_tests",)),
    )
    responses = tuple(
        ProviderResponse(
            request_id=repair_decision_request_id(RUN_ID, index),
            action=ToolAction(
                tool_call=ToolCall(
                    call_id=uuid5(RUN_ID, f"tool:{index}"),
                    tool_name=name,
                    validated_arguments=arguments,
                    requested_at=STARTED_AT,
                ),
                considered_tools=considered,
                selection_reason=f"Collect {name} evidence for the fixture.",
            ),
            usage=usage,
        )
        for index, (name, arguments, considered) in enumerate(steps)
    )
    if not correct_patch:
        return responses
    return (
        *responses,
        ProviderResponse(
            request_id=repair_decision_request_id(RUN_ID, len(steps)),
            action=FinishAction(summary="Candidate is ready for independent verification."),
            usage=usage,
        ),
    )


@pytest.mark.parametrize("correct_patch", (True, False), ids=("fixed", "still-fails"))
def test_real_repair_graph_uses_isolated_regressions_without_leaking_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, correct_patch: bool
) -> None:
    task = _fixture_task(tmp_path)
    output = tmp_path / "result"
    providers: list[FakeProvider] = []
    agent_sandboxes: list[_TrustedFixtureSandbox] = []
    evaluator_sandboxes: list[_TrustedFixtureSandbox] = []

    def recording_provider(
        responses: Iterable[ProviderResponse], *, provider_name: str
    ) -> FakeProvider:
        provider = FakeProvider(responses, provider_name=provider_name)
        providers.append(provider)
        return provider

    def agent_factory(root: Path) -> RepairSandbox:
        sandbox = _TrustedFixtureSandbox(root, tmp_path / "agent-workspaces")
        agent_sandboxes.append(sandbox)
        return sandbox

    def evaluator_factory(root: Path) -> RepairSandbox:
        sandbox = _TrustedFixtureSandbox(root, tmp_path / "evaluator-workspaces")
        evaluator_sandboxes.append(sandbox)
        return sandbox

    def isolated_factory(root: Path) -> RepairSandbox:
        return IsolatedTestSandbox(
            task=task,
            regression_patch=REGRESSION_PATCH,
            artifact_root=root / "artifacts",
            agent_factory=agent_factory,
            evaluator_factory=evaluator_factory,
        )

    monkeypatch.setattr("repopilot.repair_workflow.FakeProvider", recording_provider)
    artifacts = run_repair_request(
        RepairRunRequest(
            thread_id="isolated-repair-fixture",
            run_id=RUN_ID,
            started_at=STARTED_AT,
            task=task,
            sandbox_image_reference=f"python@sha256:{'a' * 64}",
            provider_responses=_responses(correct_patch=correct_patch),
        ),
        output_dir=output,
        settings=RepoPilotSettings(
            max_steps=20 if correct_patch else 5, max_tokens=20_000, timeout_seconds=60
        ),
        sandbox_factory=isolated_factory,
    )

    baseline = artifacts.final_state.baseline_test_result
    final_test = artifacts.final_state.test_state.last_result
    assert baseline is not None and final_test is not None
    assert baseline.status is RunTestStatus.FAILED
    assert baseline.category is RunTestCategory.ASSERTION_FAILURE
    assert baseline.tests_run == 1 and baseline.failures == 1
    assert baseline.failing_tests == ("tests/test_regression.py::test_increment",)
    assert artifacts.result.reproduction_result.status is ReproductionStatus.REPRODUCED
    assert artifacts.final_state.test_state.commands_run == 2
    assert final_test.status is (RunTestStatus.PASSED if correct_patch else RunTestStatus.FAILED)
    assert final_test.tests_run == 1
    assert (artifacts.result.status is RunStatus.SUCCESS) is correct_patch
    assert artifacts.result.usage.estimated_cost_usd == Decimal("0")

    assert len(agent_sandboxes) == len(evaluator_sandboxes) == len(providers) == 1
    agent = agent_sandboxes[0]
    evaluator = evaluator_sandboxes[0]
    assert "tests/test_regression.py" not in agent.source_files
    assert "tests/test_regression.py" in evaluator.source_files
    assert not agent.workspace.joinpath("tests").exists()
    assert not agent.raw_test_outputs
    assert len(evaluator.raw_test_outputs) == 2
    assert PRIVATE_MARKER in evaluator.raw_test_outputs[0]
    assert all(sandbox.lifecycle is SandboxLifecycle.DESTROYED for sandbox in (agent, evaluator))
    assert Path(str(task.repository_path), "calculator.py").read_text() == SOURCE
    assert not Path(str(task.repository_path), "tests").exists()
    assert PRIVATE_MARKER not in artifacts.trace.model_dump_json()
    assert len(providers[0].requests) == (6 if correct_patch else 5)
    for request in providers[0].requests:
        assert PRIVATE_MARKER not in request.model_dump_json()
        assert "assert increment(1) == 2" not in request.model_dump_json()
    for path in output.rglob("*"):
        if path.is_file():
            assert PRIVATE_MARKER.encode() not in path.read_bytes(), path.relative_to(output)


def test_clean_checkout_without_evaluator_tests_is_not_issue_reproduction(tmp_path: Path) -> None:
    """Reproduce the original missing-test defect through the actual graph."""

    task = _fixture_task(tmp_path)

    def clean_factory(root: Path) -> RepairSandbox:
        return _TrustedFixtureSandbox(root / "artifacts", tmp_path / "clean-workspaces")

    artifacts = run_repair_request(
        RepairRunRequest(
            thread_id="missing-regression-fixture",
            run_id=RUN_ID,
            started_at=STARTED_AT,
            task=task,
            sandbox_image_reference=f"python@sha256:{'a' * 64}",
            provider_responses=_responses(correct_patch=True),
        ),
        output_dir=tmp_path / "result",
        settings=RepoPilotSettings(max_steps=20, max_tokens=20_000, timeout_seconds=60),
        sandbox_factory=clean_factory,
    )

    baseline = artifacts.final_state.baseline_test_result
    assert baseline is not None
    assert baseline.status is RunTestStatus.ERROR
    assert baseline.tests_run == 0
    assert artifacts.result.reproduction_result.status is ReproductionStatus.ERROR
    assert artifacts.result.status is not RunStatus.SUCCESS
    assert artifacts.final_state.test_state.commands_run == 1


@pytest.mark.parametrize("correction_passes", (True, False), ids=("corrected", "still-fails"))
def test_validation_correction_reads_current_source_and_exports_cumulative_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, correction_passes: bool
) -> None:
    task = _fixture_task(tmp_path)
    output = tmp_path / "result"
    sandboxes: list[_TrustedFixtureSandbox] = []
    providers: list[FakeProvider] = []

    def factory(root: Path) -> RepairSandbox:
        def build(artifact_root: Path) -> RepairSandbox:
            sandbox = _TrustedFixtureSandbox(artifact_root, tmp_path / "workspaces")
            sandboxes.append(sandbox)
            return sandbox

        return IsolatedTestSandbox(
            task=task,
            regression_patch=REGRESSION_PATCH,
            artifact_root=root / "artifacts",
            agent_factory=build,
            evaluator_factory=build,
        )

    def provider(responses: Iterable[ProviderResponse], *, provider_name: str) -> FakeProvider:
        instance = FakeProvider(responses, provider_name=provider_name)
        providers.append(instance)
        return instance

    monkeypatch.setattr("repopilot.repair_workflow.FakeProvider", provider)
    incremental = (
        "diff --git a/calculator.py b/calculator.py\n"
        "--- a/calculator.py\n+++ b/calculator.py\n"
        "@@ -1,2 +1,2 @@\n def increment(value):\n"
        "-    return value + 0\n"
        f"+    return value + {1 if correction_passes else 2}\n"
    )
    responses = list(_responses(correct_patch=False))
    for name, arguments in (
        ("read_file", {"path": "calculator.py"}),
        ("apply_patch", {"diff": incremental}),
        ("run_tests", {"test_index": 0}),
    ):
        index = len(responses)
        responses.append(
            ProviderResponse(
                request_id=repair_decision_request_id(RUN_ID, index),
                action=ToolAction(
                    tool_call=ToolCall(
                        call_id=uuid5(RUN_ID, f"tool:{index}"),
                        tool_name=name,
                        validated_arguments=arguments,
                        requested_at=STARTED_AT,
                    ),
                    considered_tools=(name,),
                    selection_reason="Inspect or validate the correction.",
                ),
                usage=responses[0].usage,
            )
        )
    if correction_passes:
        responses.append(
            ProviderResponse(
                request_id=repair_decision_request_id(RUN_ID, len(responses)),
                action=FinishAction(
                    summary="The corrected candidate passed the frozen validation."
                ),
                usage=responses[0].usage,
            )
        )
    artifacts = run_repair_request(
        RepairRunRequest(
            thread_id="isolated-validation-correction",
            run_id=RUN_ID,
            started_at=STARTED_AT,
            task=task,
            sandbox_image_reference=f"python@sha256:{'a' * 64}",
            provider_responses=tuple(responses),
        ),
        output_dir=output,
        settings=RepoPilotSettings(max_steps=20, max_tokens=30_000, timeout_seconds=60),
        sandbox_factory=factory,
    )
    assert artifacts.result.task_metrics.task_success is correction_passes
    assert artifacts.final_state.patch_state.revision == 2
    assert artifacts.final_state.test_state.commands_run == 3
    assert len(providers[0].requests) == (9 if correction_passes else 8)
    reads = [
        event.tool_result.payload
        for event in artifacts.trace.events
        if event.node.value == "execute_tool" and event.tool_call.tool_name == "read_file"
    ]
    assert reads[0]["content"] == SOURCE
    assert reads[1]["content"] == SOURCE.replace("value - 1", "value + 0")
    reference = artifacts.final_state.patch_state.diff_ref
    assert reference is not None
    cumulative = (output / reference.relative_path).read_bytes()
    assert b"-    return value - 1" in cumulative
    assert b"-    return value + 0" not in cumulative
    fresh = tmp_path / "fresh"
    shutil.copytree(Path(str(task.repository_path)), fresh)
    subprocess.run(("git", "apply", "-"), input=cumulative, cwd=fresh, check=True)
    assert (fresh / "calculator.py").read_bytes() == (
        sandboxes[0].workspace / "calculator.py"
    ).read_bytes()
    assert (sandboxes[0].workspace / "calculator.py").read_bytes() == (
        sandboxes[1].workspace / "calculator.py"
    ).read_bytes()
    assert artifacts.result.usage.estimated_cost_usd == Decimal("0")
    for path in output.rglob("*"):
        if path.is_file():
            assert PRIVATE_MARKER.encode() not in path.read_bytes()


@pytest.mark.parametrize("case", ("valid", "already-passes", "collection-error"))
def test_zero_provider_canary_uses_production_test_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    from scripts.p7_test_protocol_canary import qualify_task

    task = _fixture_task(tmp_path)
    regression_patch = REGRESSION_PATCH
    if case == "already-passes":
        regression_patch = regression_patch.replace("increment(1) == 2", "increment(1) == 0")
    elif case == "collection-error":
        regression_patch = regression_patch.replace(
            "from calculator import increment", "from nonexistent_fixture_module import increment"
        )

    def forbid_provider(*args: object, **kwargs: object) -> None:
        raise AssertionError("the environment canary must not construct any Provider")

    monkeypatch.setattr(FakeProvider, "__init__", forbid_provider)
    monkeypatch.setattr(OpenAIProvider, "__init__", forbid_provider)

    def factory(root: Path) -> RepairSandbox:
        return IsolatedTestSandbox(
            task=task,
            regression_patch=regression_patch,
            artifact_root=root / "artifacts",
            agent_factory=lambda path: _TrustedFixtureSandbox(path, tmp_path / "agent"),
            evaluator_factory=lambda path: _TrustedFixtureSandbox(path, tmp_path / "evaluator"),
        )

    def qualify() -> object:
        return qualify_task(
            task=task,
            regression_patch=regression_patch,
            gold_patch=_candidate_patch(correct_patch=True),
            settings=RepoPilotSettings(),
            sandbox_image_reference=f"python@sha256:{'a' * 64}",
            sandbox_factory=factory,
        )

    if case != "valid":
        with pytest.raises(ValueError, match="protocol_baseline_not_reproduced"):
            qualify()
        return

    evidence = qualify_task(
        task=task,
        regression_patch=regression_patch,
        gold_patch=_candidate_patch(correct_patch=True),
        settings=RepoPilotSettings(),
        sandbox_image_reference=f"python@sha256:{'a' * 64}",
        sandbox_factory=factory,
    )
    assert evidence.baseline_reproduced and evidence.gold_verified
    assert evidence.baseline[0].status is RunTestStatus.FAILED
    assert evidence.baseline[0].category is RunTestCategory.ASSERTION_FAILURE
    assert evidence.gold[0].status is RunTestStatus.PASSED
    assert all(item.verifier_accepts_evidence for item in (*evidence.baseline, *evidence.gold))
    assert PRIVATE_MARKER not in evidence.model_dump_json()
