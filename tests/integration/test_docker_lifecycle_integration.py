"""Real-Docker lifecycle proof for the P4-S2 Sandbox adapter."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from repopilot.config import RepoPilotSettings
from repopilot.enums import SandboxLifecycle
from repopilot.sandbox import (
    DOCKER_WORKSPACE_ROOT,
    DockerCliClient,
    DockerClientError,
    DockerSandbox,
    SandboxPolicy,
)
from repopilot.schemas import CommandSpec, TaskSpec

pytestmark = pytest.mark.integration


def test_real_docker_prepare_inspect_and_destroy_are_isolated(tmp_path: Path) -> None:
    image = os.environ.get("REPOPILOT_TEST_SANDBOX_IMAGE")
    if image is None:
        pytest.skip("set REPOPILOT_TEST_SANDBOX_IMAGE to an already-local digest-pinned image")

    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=host-canary", encoding="utf-8")
    (tmp_path / "ignored-link").symlink_to(tmp_path / "module.py")
    task = TaskSpec(
        task_id="docker-integration-task",
        repository_path=str(tmp_path),
        base_commit="a" * 40,
        issue_title="Real Docker lifecycle",
        issue_body="Prove isolation and cleanup without running repository commands.",
        test_commands=(CommandSpec(argv=("python", "-m", "pytest", "-q")),),
    )
    settings = RepoPilotSettings(
        sandbox_cpu_limit=0.25,
        sandbox_memory_limit_mb=128,
        sandbox_pids_limit=32,
        sandbox_tmpfs_limit_mb=16,
    )
    policy = SandboxPolicy.from_settings(settings)
    client = DockerCliClient()
    artifact_root = tmp_path.parent / f"{tmp_path.name}-artifacts"
    first = DockerSandbox(
        policy=policy,
        image_reference=image,
        artifact_root=artifact_root / "first",
        client=client,
    )
    second = DockerSandbox(
        policy=policy,
        image_reference=image,
        artifact_root=artifact_root / "second",
        client=client,
    )
    handles = []

    try:
        handles = [first.prepare(task), second.prepare(task)]
        assert handles[0].sandbox_id != handles[1].sandbox_id
        assert first.lifecycle is SandboxLifecycle.PREPARED
        assert second.lifecycle is SandboxLifecycle.PREPARED
        assert first.preparation is not None
        assert second.preparation is not None
        assert first.preparation.container_id != second.preparation.container_id
        for sandbox in (first, second):
            assert sandbox.preparation is not None
            record = sandbox.preparation
            inspection = record.inspection
            assert inspection.running is True
            assert inspection.user == "65532:65532"
            assert inspection.network_mode == "none"
            assert inspection.network_names == ("none",)
            assert inspection.read_only_rootfs is True
            assert inspection.privileged is False
            assert inspection.cap_drop == ("ALL",)
            assert inspection.security_options == ("no-new-privileges",)
            assert inspection.nano_cpus == 250_000_000
            assert inspection.memory_bytes == 128 * 1024 * 1024
            assert inspection.pids_limit == 32
            assert set(inspection.tmpfs) == {DOCKER_WORKSPACE_ROOT}
            assert all(mount.mount_type == "tmpfs" for mount in inspection.mounts)
            assert record.workspace_copy.file_count == 1
            assert record.workspace_copy.skipped_denied_count == 1
            assert record.workspace_copy.skipped_symlink_count == 1
    finally:
        for sandbox, handle in zip((first, second), handles, strict=False):
            sandbox.destroy(handle)

    for sandbox in (first, second):
        assert sandbox.lifecycle is SandboxLifecycle.DESTROYED
        assert sandbox.preparation is not None
        with pytest.raises(DockerClientError):
            client.inspect_container(sandbox.preparation.container_id)
