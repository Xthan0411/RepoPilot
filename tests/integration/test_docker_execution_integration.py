"""Real-Docker P4-S3 command, read, artifact, network, and timeout proof."""

from __future__ import annotations

import os
from pathlib import Path
from time import monotonic
from uuid import uuid4

import pytest

from repopilot.config import RepoPilotSettings
from repopilot.enums import (
    ErrorType,
    SandboxCommandPhase,
    SandboxLifecycle,
    SandboxNetworkMode,
)
from repopilot.sandbox import (
    DockerCliClient,
    DockerClientError,
    DockerSandbox,
    ReadFileRequest,
    RunRequest,
    SandboxPolicy,
)
from repopilot.schemas import CommandSpec, TaskSpec

pytestmark = pytest.mark.integration


def test_real_docker_run_read_artifact_network_and_timeout(tmp_path: Path) -> None:
    image = os.environ.get("REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE")
    if image is None:
        pytest.skip("set REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE to a local digest-pinned Python image")

    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "module.txt").write_text(
        "first\nsecond\nthird\nfourth\n",
        encoding="utf-8",
    )
    (repository / "emit.py").write_text(
        "import sys\n"
        "sys.stdout.write('x' * 64 + '\\n')\n"
        "sys.stderr.write('stderr-evidence\\n')\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    (repository / "make_link.py").write_text(
        "from pathlib import Path\nPath('module-link.txt').symlink_to('module.txt')\n",
        encoding="utf-8",
    )
    (repository / "slow.py").write_text(
        "import time\ntime.sleep(10)\n",
        encoding="utf-8",
    )
    setup_command = CommandSpec(argv=("python", "-c", "print('setup-network')"))
    emit_command = CommandSpec(argv=("python", "emit.py"))
    link_command = CommandSpec(argv=("python", "make_link.py"))
    slow_command = CommandSpec(argv=("python", "slow.py"), timeout_seconds=1)
    task = TaskSpec(
        task_id="docker-execution-integration",
        repository_path=str(repository),
        base_commit="b" * 40,
        issue_title="Real Docker execution",
        issue_body="Prove bounded command output and no-follow reads.",
        setup_commands=(setup_command,),
        test_commands=(emit_command, link_command, slow_command),
        timeout_seconds=20,
    )
    policy = SandboxPolicy.from_settings(
        RepoPilotSettings(
            sandbox_network=SandboxNetworkMode.SETUP_ONLY,
            sandbox_cpu_limit=0.25,
            sandbox_memory_limit_mb=128,
            sandbox_pids_limit=32,
            sandbox_tmpfs_limit_mb=16,
            sandbox_command_timeout_seconds=3,
            sandbox_max_output_bytes=16,
            sandbox_max_artifact_bytes=512,
            patch_max_bytes=512,
        )
    )
    client = DockerCliClient()
    artifact_root = tmp_path / "artifacts"
    sandbox = DockerSandbox(
        policy=policy,
        image_reference=image,
        artifact_root=artifact_root,
        client=client,
    )
    handle = sandbox.prepare(task)
    assert sandbox.preparation is not None
    container_id = sandbox.preparation.container_id

    try:
        setup = sandbox.run(
            handle,
            RunRequest(
                call_id=uuid4(),
                command=setup_command,
                phase=SandboxCommandPhase.SETUP,
                network_requested=True,
            ),
        )
        assert setup.exit_code == 0
        assert setup.stdout_preview == "setup-network\n"
        assert client.inspect_container(container_id).network_names == ("none",)

        emit_call = uuid4()
        emitted = sandbox.run(
            handle,
            RunRequest(call_id=emit_call, command=emit_command),
        )
        assert emitted.exit_code == 7
        assert emitted.error_type is None
        assert emitted.stdout_preview == "x" * 16
        assert emitted.stderr_preview == "stderr-evidence\n"
        assert emitted.truncated is True
        assert emitted.stdout_ref is not None
        assert emitted.stdout_ref.size_bytes == 65
        assert (
            artifact_root / handle.sandbox_id / f"{emit_call}.stdout.txt"
        ).read_bytes() == b"x" * 64 + b"\n"

        read = sandbox.read_file(
            handle,
            ReadFileRequest(
                call_id=uuid4(),
                relative_path="module.txt",
                start_line=2,
                end_line=3,
            ),
        )
        assert read.exit_code == 0
        assert read.stdout_preview == "second\nthird\n"

        linked = sandbox.run(
            handle,
            RunRequest(call_id=uuid4(), command=link_command),
        )
        assert linked.exit_code == 0
        denied_link = sandbox.read_file(
            handle,
            ReadFileRequest(call_id=uuid4(), relative_path="module-link.txt"),
        )
        assert denied_link.exit_code == 40
        assert denied_link.error_type is ErrorType.PATH_VIOLATION

        started = monotonic()
        timed_out = sandbox.run(
            handle,
            RunRequest(call_id=uuid4(), command=slow_command),
        )
        elapsed = monotonic() - started
        assert timed_out.exit_code is None
        assert timed_out.error_type is ErrorType.TOOL_TIMEOUT
        assert elapsed < 5
        assert sandbox.lifecycle is SandboxLifecycle.DESTROYED
        assert len(sandbox.collect_artifacts(handle)) == 12
    finally:
        sandbox.destroy(handle)

    with pytest.raises(DockerClientError):
        client.inspect_container(container_id)
