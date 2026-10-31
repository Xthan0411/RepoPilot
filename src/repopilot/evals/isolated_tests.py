"""Run frozen regression tests outside the Agent-visible repository and artifacts.

The Agent sandbox contains only the clean task checkout and candidate edits. A second
ephemeral sandbox receives an evaluator-only regression patch. Only sanitized framework
results cross back into the Agent trace; evaluator files and raw logs never do.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast
from uuid import UUID, uuid4

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    ErrorType,
    PolicyDecision,
    SandboxCommandPhase,
    SandboxLifecycle,
    TestStatus,
)
from repopilot.repair_workflow import RepairSandbox, RepairSandboxFactory
from repopilot.sandbox.artifacts import SandboxArtifactStore
from repopilot.sandbox.base import (
    ApplyPatchRequest,
    ReadFileRequest,
    RunRequest,
    SandboxError,
    SandboxHandle,
)
from repopilot.sandbox.docker import DockerSandbox
from repopilot.sandbox.policy import SANDBOX_DENIED_PATH_ROOTS, SandboxPolicy
from repopilot.schemas import ArtifactRef, CommandSpec, Observation, TaskSpec
from repopilot.tools.patch import PatchLimits, PatchValidationError, parse_unified_diff
from repopilot.tools.tests import (
    interpret_test_observation,
    is_pytest_command,
    pytest_failure_rows,
    pytest_failure_type_hints,
)

_MAX_BYTES = 10_000_000
_PATCH_LIMITS = PatchLimits(
    max_bytes=1_000_000, max_files=100, max_hunks=1_000, max_changed_lines=10_000
)
_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)
_SAFE_ID = re.compile(r"[A-Za-z0-9_./:-]{1,400}")
SandboxArtifactFactory = Callable[[Path], RepairSandbox]


class IsolatedTestSandboxError(SandboxError):
    """An evaluator failure that never includes hidden source, output, or paths."""

    def __init__(self) -> None:
        super().__init__("isolated test sandbox failed closed", error_type=ErrorType.SANDBOX_ERROR)


def build_isolated_test_sandbox_factory(
    *,
    settings: RepoPilotSettings,
    sandbox_image_reference: str,
    task: TaskSpec,
    regression_patch: str,
) -> RepairSandboxFactory:
    """Bind one frozen task and regression patch to the workflow's output-root factory."""

    policy = SandboxPolicy.from_settings(settings)

    def docker(artifact_root: Path) -> RepairSandbox:
        return cast(
            RepairSandbox,
            DockerSandbox(
                policy=policy,
                image_reference=sandbox_image_reference,
                artifact_root=artifact_root,
            ),
        )

    def build(output_root: Path) -> RepairSandbox:
        return IsolatedTestSandbox(
            task=task,
            regression_patch=regression_patch,
            artifact_root=output_root / "artifacts",
            agent_factory=docker,
            evaluator_factory=docker,
            preview_bytes=settings.tool_max_output_bytes,
        )

    return build


class IsolatedTestSandbox:
    """Single-task dual sandbox; the two injected factories take artifact-root paths."""

    def __init__(
        self,
        *,
        task: TaskSpec,
        regression_patch: str,
        artifact_root: Path,
        agent_factory: SandboxArtifactFactory,
        evaluator_factory: SandboxArtifactFactory,
        preview_bytes: int = 16_000,
    ) -> None:
        if not 1 <= preview_bytes <= _MAX_BYTES:
            raise IsolatedTestSandboxError()
        self._test_paths = _regression_paths(regression_patch)
        self._regression_patch = regression_patch.encode("utf-8")
        self._task = task
        self._artifact_root = artifact_root
        self._agent_factory = agent_factory
        self._evaluator_factory = evaluator_factory
        self._preview_bytes = preview_bytes
        self._agent: RepairSandbox | None = None
        self._evaluator: RepairSandbox | None = None
        self._handle: SandboxHandle | None = None
        self._evaluator_handle: SandboxHandle | None = None
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._private_root: Path | None = None
        self._public_store: SandboxArtifactStore | None = None
        self._lifecycle = SandboxLifecycle.NEW
        self._poisoned = False
        self._calls: set[UUID] = set()
        self._patches: dict[UUID, tuple[ArtifactRef, bytes, tuple[str, ...]]] = {}

    @property
    def lifecycle(self) -> SandboxLifecycle:
        return self._lifecycle

    def prepare(self, task: TaskSpec) -> SandboxHandle:
        if self._lifecycle is not SandboxLifecycle.NEW or task != self._task:
            raise IsolatedTestSandboxError()
        try:
            self._temporary = tempfile.TemporaryDirectory(prefix="repopilot-isolated-tests-")
            self._private_root = Path(self._temporary.name).resolve(strict=True)
            checkout = _prepare_checkout(task, self._private_root, self._regression_patch)
            self._public_store = SandboxArtifactStore(self._artifact_root)
            self._public_store.prepare(
                f"isolated-tests-{uuid4().hex}", repository_source=Path(task.repository_path or "")
            )
            self._agent = self._agent_factory(self._artifact_root)
            self._handle = self._agent.prepare(task)
            self._evaluator = self._evaluator_factory(self._private_root / "artifacts")
            evaluator_task = task.model_copy(
                update={"repository_path": str(checkout), "repository_url": None}
            )
            self._evaluator_handle = self._evaluator.prepare(evaluator_task)
            self._lifecycle = SandboxLifecycle.PREPARED
            return self._handle
        except Exception:
            self._poisoned = True
            self._cleanup()
            raise IsolatedTestSandboxError() from None

    def run(self, handle: SandboxHandle, request: RunRequest) -> Observation:
        self._require_usable(handle, request.call_id)
        assert self._agent is not None
        assert self._evaluator is not None and self._evaluator_handle is not None
        allowed = (
            self._task.test_commands
            if request.phase is SandboxCommandPhase.TEST
            else self._task.setup_commands
        )
        if request.network_requested or request.command not in allowed:
            return _denied(handle, request.call_id)
        if request.phase is SandboxCommandPhase.SETUP:
            agent_result = self._agent.run(handle, request)
            if not _succeeded(agent_result):
                self._poisoned = True
                return agent_result
            try:
                evaluator_result = self._evaluator.run(self._evaluator_handle, request)
            except Exception:
                self._poisoned = True
                return self._failure(handle, request.call_id)
            if not _succeeded(evaluator_result):
                self._poisoned = True
                return self._failure(handle, request.call_id)
            return agent_result
        try:
            raw = self._evaluator.run(self._evaluator_handle, request)
            if raw.policy_decision is PolicyDecision.DENIED:
                return _denied(handle, request.call_id)
            stdout = self._read_private(raw.stdout_ref)
            stderr = self._read_private(raw.stderr_ref)
            complete = raw.model_copy(
                update={
                    "stdout_preview": stdout.decode("utf-8", errors="replace")
                    if raw.stdout_ref is not None
                    else None,
                    "stderr_preview": stderr.decode("utf-8", errors="replace")
                    if raw.stderr_ref is not None
                    else None,
                }
            )
            output, error_type = _sanitize_test_output(complete, request.command)
            result = self._persist_result(handle, raw, output, b"", error_type=error_type)
            if self._evaluator.lifecycle is not SandboxLifecycle.PREPARED:
                self._poisoned = True
            return result
        except Exception:
            self._poisoned = True
            return self._failure(handle, request.call_id)

    def read_file(self, handle: SandboxHandle, request: ReadFileRequest) -> Observation:
        self._require_usable(handle, request.call_id)
        assert self._agent is not None
        return self._agent.read_file(handle, request)

    def stage_patch(self, handle: SandboxHandle, call_id: UUID, content: bytes) -> ArtifactRef:
        self._require_usable(handle)
        if call_id in self._patches or call_id in self._calls:
            raise IsolatedTestSandboxError()
        try:
            parsed = parse_unified_diff(content.decode("utf-8"), _PATCH_LIMITS)
        except (UnicodeDecodeError, PatchValidationError):
            raise IsolatedTestSandboxError() from None
        assert self._agent is not None
        reference = self._agent.stage_patch(handle, call_id, content)
        if reference.sha256 != hashlib.sha256(content).hexdigest() or reference.size_bytes != len(
            content
        ):
            self._poisoned = True
            raise IsolatedTestSandboxError()
        self._patches[call_id] = (reference, content, parsed.summary.files_changed)
        return reference

    def apply_patch(self, handle: SandboxHandle, request: ApplyPatchRequest) -> Observation:
        self._require_usable(handle, request.call_id)
        staged = self._patches.pop(request.call_id, None)
        if staged is None or staged[0] != request.patch_ref:
            return self._patch_rejected(handle, request.call_id, "artifact_mismatch")
        if any(_overlaps(path, hidden) for path in staged[2] for hidden in self._test_paths):
            # Reject the whole candidate before either workspace applies any source changes.
            # The public code communicates the rule, never the private regression path list.
            return self._patch_rejected(handle, request.call_id, "protected_test_path")
        assert self._agent is not None
        assert self._evaluator is not None and self._evaluator_handle is not None
        result = self._agent.apply_patch(handle, request)
        if not _succeeded(result):
            return result
        try:
            reference = self._evaluator.stage_patch(
                self._evaluator_handle, request.call_id, staged[1]
            )
            if (
                reference.sha256 != request.patch_ref.sha256
                or reference.size_bytes != request.patch_ref.size_bytes
            ):
                raise IsolatedTestSandboxError()
            mirrored = self._evaluator.apply_patch(
                self._evaluator_handle,
                ApplyPatchRequest(call_id=request.call_id, patch_ref=reference),
            )
            if not _succeeded(mirrored):
                raise IsolatedTestSandboxError()
        except Exception:
            self._poisoned = True
            return self._failure(handle, request.call_id)
        return result

    def collect_artifacts(self, handle: SandboxHandle) -> tuple[ArtifactRef, ...]:
        self._require_handle(handle)
        assert self._agent is not None and self._public_store is not None
        return (*self._agent.collect_artifacts(handle), *self._public_store.collect())

    def destroy(self, handle: SandboxHandle) -> None:
        self._require_handle(handle)
        if self._lifecycle is SandboxLifecycle.DESTROYED:
            return
        if not self._cleanup():
            raise IsolatedTestSandboxError()

    def _cleanup(self) -> bool:
        succeeded = True
        for sandbox, handle in (
            (self._evaluator, self._evaluator_handle),
            (self._agent, self._handle),
        ):
            if sandbox is not None and handle is not None:
                try:
                    sandbox.destroy(handle)
                except Exception:
                    succeeded = False
        if self._temporary is not None:
            try:
                self._temporary.cleanup()
            except OSError:
                succeeded = False
        self._patches.clear()
        self._regression_patch = b""
        self._lifecycle = SandboxLifecycle.DESTROYED
        return succeeded

    def _require_handle(self, handle: SandboxHandle) -> None:
        if self._handle is None or handle != self._handle:
            raise IsolatedTestSandboxError()

    def _require_usable(self, handle: SandboxHandle, call_id: UUID | None = None) -> None:
        self._require_handle(handle)
        if self._lifecycle is not SandboxLifecycle.PREPARED or self._poisoned:
            raise IsolatedTestSandboxError()
        if call_id is not None:
            if call_id in self._calls:
                raise IsolatedTestSandboxError()
            self._calls.add(call_id)

    def _read_private(self, reference: ArtifactRef | None) -> bytes:
        if reference is None:
            return b""
        assert self._private_root is not None
        assert self._evaluator is not None and self._evaluator_handle is not None
        if reference not in self._evaluator.collect_artifacts(self._evaluator_handle):
            raise IsolatedTestSandboxError()
        path = self._private_root / reference.relative_path
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise IsolatedTestSandboxError()
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(self._private_root) or reference.size_bytes > _MAX_BYTES:
            raise IsolatedTestSandboxError()
        descriptor = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise IsolatedTestSandboxError()
            content = stream.read(_MAX_BYTES + 1)
        if (
            len(content) != reference.size_bytes
            or hashlib.sha256(content).hexdigest() != reference.sha256
        ):
            raise IsolatedTestSandboxError()
        return content

    def _persist_result(
        self,
        handle: SandboxHandle,
        original: Observation,
        stdout: bytes,
        stderr: bytes,
        *,
        error_type: ErrorType | None,
    ) -> Observation:
        assert self._public_store is not None
        stdout_ref = self._public_store.persist(original.call_id, "stdout", stdout)
        stderr_ref = self._public_store.persist(original.call_id, "stderr", stderr)
        return original.model_copy(
            update={
                "sandbox_id": handle.sandbox_id,
                "stdout_ref": stdout_ref,
                "stderr_ref": stderr_ref,
                "stdout_preview": stdout[: self._preview_bytes].decode("utf-8", errors="replace"),
                "stderr_preview": stderr[: self._preview_bytes].decode("utf-8", errors="replace"),
                "truncated": original.truncated
                or len(stdout) > self._preview_bytes
                or len(stderr) > self._preview_bytes,
                "error_type": error_type,
            }
        )

    def _failure(self, handle: SandboxHandle, call_id: UUID) -> Observation:
        return Observation(
            call_id=call_id,
            started_at=datetime.now(UTC),
            duration_ms=0,
            sandbox_id=handle.sandbox_id,
            policy_decision=PolicyDecision.ALLOWED,
            error_type=ErrorType.SANDBOX_ERROR,
        )

    def _patch_rejected(self, handle: SandboxHandle, call_id: UUID, code: str) -> Observation:
        original = Observation(
            call_id=call_id,
            started_at=datetime.now(UTC),
            duration_ms=0,
            exit_code=44,
            sandbox_id=handle.sandbox_id,
            policy_decision=PolicyDecision.ALLOWED,
            error_type=ErrorType.PATCH_ERROR,
        )
        return self._persist_result(
            handle,
            original,
            b"",
            f"repopilot_patch_error:{code}\n".encode(),
            error_type=ErrorType.PATCH_ERROR,
        )


def _denied(handle: SandboxHandle, call_id: UUID) -> Observation:
    return Observation(
        call_id=call_id,
        started_at=datetime.now(UTC),
        duration_ms=0,
        sandbox_id=handle.sandbox_id,
        policy_decision=PolicyDecision.DENIED,
        error_type=ErrorType.POLICY_DENIED,
    )


def _succeeded(observation: Observation) -> bool:
    return (
        observation.policy_decision is PolicyDecision.ALLOWED
        and observation.exit_code == 0
        and observation.error_type is None
    )


def _overlaps(first: str, second: str) -> bool:
    return first == second or first.startswith(second + "/") or second.startswith(first + "/")


def _regression_paths(patch: str) -> tuple[str, ...]:
    try:
        encoded = patch.encode("utf-8")
    except UnicodeEncodeError:
        raise IsolatedTestSandboxError() from None
    if not encoded or len(encoded) > 1_000_000 or "\x00" in patch or "\r" in patch:
        raise IsolatedTestSandboxError()
    paths: list[str] = []
    for old, new in _DIFF_HEADER.findall(patch):
        path = PurePosixPath(old)
        if (
            old != new
            or not path.parts
            or path.is_absolute()
            or old != path.as_posix()
            or any(part in {".", "..", ".git", ".env"} for part in path.parts)
            or path.parts[0] in SANDBOX_DENIED_PATH_ROOTS
            or "\\" in old
            or old in paths
        ):
            raise IsolatedTestSandboxError()
        paths.append(old)
    if not paths or any(
        line.startswith(("GIT binary patch", "Binary files", "rename ", "copy "))
        or re.match(
            r"^(?:old mode|new mode|new file mode|deleted file mode) (?!100644$|100755$)",
            line,
        )
        for line in patch.splitlines()
    ):
        raise IsolatedTestSandboxError()
    return tuple(paths)


def _git(*arguments: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.autocrlf=false",
            *arguments,
        ],
        input=input_bytes,
        capture_output=True,
        check=False,
        timeout=120,
        env={
            "PATH": os.defpath,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_LFS_SKIP_SMUDGE": "1",
        },
    )
    if result.returncode != 0:
        raise IsolatedTestSandboxError()
    return result.stdout


def _prepare_checkout(task: TaskSpec, private_root: Path, regression_patch: bytes) -> Path:
    if task.repository_path is None:
        raise IsolatedTestSandboxError()
    source = Path(task.repository_path)
    if source.is_symlink() or not source.is_dir():
        raise IsolatedTestSandboxError()
    source = source.resolve(strict=True)
    if _git("-C", str(source), "rev-parse", "HEAD").decode().strip() != task.base_commit:
        raise IsolatedTestSandboxError()
    checkout = private_root / "checkout"
    _git("clone", "--local", "--no-hardlinks", "--no-checkout", "--", str(source), str(checkout))
    _git("-C", str(checkout), "checkout", "--detach", task.base_commit)
    # Inspect the source with the trusted clone's config/index: source-local filters,
    # hooks and fsmonitor must never execute on the host.
    if _git(
        "--git-dir=" + str(checkout / ".git"),
        "--work-tree=" + str(source),
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=all",
    ).strip():
        raise IsolatedTestSandboxError()
    declared_paths = _regression_paths(regression_patch.decode("utf-8"))
    actual_paths = tuple(
        line.split("\t", 2)[-1]
        for line in _git(
            "-C", str(checkout), "apply", "--numstat", "-", input_bytes=regression_patch
        )
        .decode("utf-8")
        .splitlines()
    )
    if actual_paths != declared_paths:
        raise IsolatedTestSandboxError()
    for relative_path in declared_paths:
        target = checkout / relative_path
        if target.is_symlink() or any(
            parent.is_symlink() for parent in target.parents if parent.is_relative_to(checkout)
        ):
            raise IsolatedTestSandboxError()
    _git(
        "-C",
        str(checkout),
        "apply",
        "--check",
        "--whitespace=nowarn",
        "-",
        input_bytes=regression_patch,
    )
    _git("-C", str(checkout), "apply", "--whitespace=nowarn", "-", input_bytes=regression_patch)
    return checkout


def _safe_test_id(value: str) -> str:
    base = value.split("[", 1)[0]
    if _SAFE_ID.fullmatch(base) is None or ".." in PurePosixPath(base).parts:
        return "redacted.py::test_" + hashlib.sha256(value.encode()).hexdigest()[:20]
    if "[" in value:
        return base + "[case-" + hashlib.sha256(value.encode()).hexdigest()[:16] + "]"
    return base


def _sanitize_test_output(
    observation: Observation, command: CommandSpec
) -> tuple[bytes, ErrorType | None]:
    parsed = interpret_test_observation(observation, command, command_index=0)
    if parsed.status is TestStatus.ERROR or parsed.tests_run <= 0:
        return (
            f"isolated_test_result: {parsed.category.value}\n".encode(),
            parsed.error_type or ErrorType.TEST_ERROR,
        )
    lines: list[str] = []
    if is_pytest_command(command):
        if parsed.failures or parsed.errors:
            lines.append("=== short test summary info ===")
            for (kind, name), hint in zip(
                pytest_failure_rows(observation, command),
                pytest_failure_type_hints(observation, command),
                strict=True,
            ):
                suffix = f" - {hint}" if hint is not None else ""
                lines.append(f"{kind} {_safe_test_id(name)}{suffix}")
        counts: list[str] = []
        for count, label in (
            (parsed.failures, "failed"),
            (parsed.errors, "error"),
            (max(0, parsed.tests_run - parsed.failures - parsed.errors), "passed"),
        ):
            if count:
                counts.append(f"{count} {label}")
        lines.append("=== " + ", ".join(counts) + " in 0.00s ===")
    else:
        for index, name in enumerate(parsed.failing_tests):
            prefix = "FAIL" if index < parsed.failures else "ERROR"
            lines.append(f"{prefix}: {_safe_test_id(name)}")
        lines.append(f"Ran {parsed.tests_run} tests in 0.00s")
        lines.append(
            f"FAILED (failures={parsed.failures}, errors={parsed.errors})"
            if parsed.failures or parsed.errors
            else "OK"
        )
    return ("\n".join(lines) + "\n").encode(), None
