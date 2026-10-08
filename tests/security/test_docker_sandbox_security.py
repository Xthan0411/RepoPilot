"""Real-Docker proofs against command, host, secret, path, and network escape."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from repopilot.config import RepoPilotSettings
from repopilot.enums import ErrorType, PolicyDecision, SandboxNetworkMode
from repopilot.sandbox import (
    DockerCliClient,
    DockerClientError,
    DockerSandbox,
    ReadFileRequest,
    RunRequest,
    SandboxPolicy,
)
from repopilot.schemas import CommandSpec, TaskSpec

pytestmark = [pytest.mark.security, pytest.mark.integration]

_SENSITIVE_ENVIRONMENT = {
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AZURE_OPENAI_API_KEY",
    "DOCKER_AUTH_CONFIG",
    "GITHUB_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "OPENAI_API_KEY",
    "SSH_AUTH_SOCK",
}

_SECURITY_PROBE = """\
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

host_canary, host_target, payload, host_home = sys.argv[1:]
forbidden_environment = {
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AZURE_OPENAI_API_KEY",
    "DOCKER_AUTH_CONFIG",
    "GITHUB_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "OPENAI_API_KEY",
    "SSH_AUTH_SOCK",
}

try:
    dns_probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import socket; socket.getaddrinfo('example.com', 443)",
        ],
        capture_output=True,
        check=False,
        timeout=1,
    )
except subprocess.TimeoutExpired:
    dns_blocked = True
else:
    dns_blocked = dns_probe.returncode != 0

try:
    connection = socket.create_connection(("1.1.1.1", 53), timeout=0.25)
except OSError:
    tcp_blocked = True
else:
    connection.close()
    tcp_blocked = False

checks = {
    "docker_socket_absent": not Path("/var/run/docker.sock").exists(),
    "dns_blocked": dns_blocked,
    "environment_clean": forbidden_environment.isdisjoint(os.environ),
    "home_not_forwarded": os.environ.get("HOME") != host_home,
    "host_canary_absent": not Path(host_canary).exists(),
    "host_target_absent": not Path(host_target).exists(),
    "payload_is_one_argument": len(sys.argv) == 5 and "$(touch " in payload,
    "repository_injection_absent": not Path("container-pwned").exists(),
    "tcp_blocked": tcp_blocked,
}
print(json.dumps(checks, sort_keys=True))
raise SystemExit(0 if all(checks.values()) else 91)
"""

_MUTATE_PATHS = """\
from pathlib import Path

Path("safe.txt").write_text("safe\\n", encoding="utf-8")
Path("runtime-link.txt").symlink_to("safe.txt")
Path("real-dir").mkdir()
Path("runtime-dir-link").symlink_to("real-dir", target_is_directory=True)
"""


def _write_attack_repository(repository: Path) -> None:
    repository.mkdir()
    (repository / "security_probe.py").write_text(_SECURITY_PROBE, encoding="utf-8")
    (repository / "mutate_paths.py").write_text(_MUTATE_PATHS, encoding="utf-8")
    (repository / ".env").write_text(
        "OPENAI_API_KEY=repository-env-canary\n",
        encoding="utf-8",
    )
    git_directory = repository / ".git"
    git_directory.mkdir()
    (git_directory / "config").write_text(
        "credential=repository-git-canary\n",
        encoding="utf-8",
    )


def test_real_container_blocks_host_command_secret_path_and_network_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = os.environ.get("REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE")
    if image is None:
        pytest.skip("set REPOPILOT_TEST_SANDBOX_PYTHON_IMAGE to a local digest-pinned Python image")

    repository = tmp_path / "repository"
    _write_attack_repository(repository)
    host_canary = tmp_path / "host-canary.txt"
    host_canary_value = "p4s4-host-file-canary"
    host_canary.write_text(host_canary_value, encoding="utf-8")
    host_injection_target = tmp_path / "host-command-injection-created"
    host_home = str(Path.home())
    environment_canaries = {
        "OPENAI_API_KEY": "p4s4-openai-env-canary",
        "AWS_SECRET_ACCESS_KEY": "p4s4-aws-env-canary",
        "GITHUB_TOKEN": "p4s4-github-env-canary",
        "SSH_AUTH_SOCK": str(tmp_path / "fake-ssh-agent.sock"),
    }
    for name, value in environment_canaries.items():
        monkeypatch.setenv(name, value)

    payload = (
        f"$(touch {host_injection_target}); "
        "touch /workspace/repository/container-pwned && echo injected"
    )
    probe_command = CommandSpec(
        argv=(
            "python",
            "security_probe.py",
            str(host_canary),
            str(host_injection_target),
            payload,
            host_home,
        )
    )
    mutate_command = CommandSpec(argv=("python", "mutate_paths.py"))
    symlink_cwd_command = CommandSpec(
        argv=("python", "-c", "from pathlib import Path; Path('cwd-executed').touch()"),
        cwd="runtime-dir-link",
    )
    task = TaskSpec(
        task_id="docker-core-security",
        repository_path=str(repository),
        base_commit="c" * 40,
        issue_title="Core Docker security attacks",
        issue_body="Attempt command, host, secret, path, socket, and network escapes.",
        test_commands=(probe_command, mutate_command, symlink_cwd_command),
        timeout_seconds=20,
    )
    policy = SandboxPolicy.from_settings(
        RepoPilotSettings(
            sandbox_network=SandboxNetworkMode.NONE,
            sandbox_cpu_limit=0.25,
            sandbox_memory_limit_mb=128,
            sandbox_pids_limit=32,
            sandbox_tmpfs_limit_mb=16,
            sandbox_command_timeout_seconds=5,
            sandbox_max_output_bytes=4_096,
            sandbox_max_artifact_bytes=8_192,
            patch_max_bytes=8_192,
        )
    )
    client = DockerCliClient()
    artifact_root = tmp_path / "trusted-artifacts"
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
        inspection = sandbox.preparation.inspection
        assert inspection.network_names == ("none",)
        assert inspection.user == "65532:65532"
        assert inspection.privileged is False
        assert inspection.read_only_rootfs is True
        assert inspection.cap_drop == ("ALL",)
        assert inspection.security_options == ("no-new-privileges",)
        assert not any(mount.mount_type == "bind" for mount in inspection.mounts)
        assert not any(mount.destination == "/var/run/docker.sock" for mount in inspection.mounts)
        assert _SENSITIVE_ENVIRONMENT.isdisjoint(inspection.environment_names)

        denied_network = sandbox.run(
            handle,
            RunRequest(
                call_id=uuid4(),
                command=probe_command,
                network_requested=True,
            ),
        )
        assert denied_network.policy_decision is PolicyDecision.DENIED
        assert denied_network.error_type is ErrorType.POLICY_DENIED
        assert sandbox.collect_artifacts(handle) == ()

        probe = sandbox.run(
            handle,
            RunRequest(call_id=uuid4(), command=probe_command),
        )
        assert probe.policy_decision is PolicyDecision.ALLOWED
        assert probe.error_type is None
        assert probe.exit_code == 0
        assert probe.stdout_preview is not None
        checks = json.loads(probe.stdout_preview)
        assert checks
        assert all(checks.values())
        assert probe.stderr_preview == ""
        assert host_canary.read_text(encoding="utf-8") == host_canary_value
        assert not host_injection_target.exists()

        artifact_count = len(sandbox.collect_artifacts(handle))
        for command in (
            CommandSpec(argv=("sh", "-c", "touch /tmp/repopilot-host-pwned")),
            CommandSpec(argv=("python", "-c", "print('unconfigured')")),
            CommandSpec(argv=("docker", "ps")),
        ):
            denied_command = sandbox.run(
                handle,
                RunRequest(call_id=uuid4(), command=command),
            )
            assert denied_command.policy_decision is PolicyDecision.DENIED
            assert denied_command.error_type is ErrorType.POLICY_DENIED
        assert len(sandbox.collect_artifacts(handle)) == artifact_count

        for denied_path in (".env", ".git/config", "artifacts/stdout.txt"):
            denied_read = sandbox.read_file(
                handle,
                ReadFileRequest(call_id=uuid4(), relative_path=denied_path),
            )
            assert denied_read.policy_decision is PolicyDecision.DENIED
            assert denied_read.error_type is ErrorType.POLICY_DENIED
        assert len(sandbox.collect_artifacts(handle)) == artifact_count

        mutation = sandbox.run(
            handle,
            RunRequest(call_id=uuid4(), command=mutate_command),
        )
        assert mutation.exit_code == 0
        for runtime_escape in ("runtime-link.txt", "runtime-dir-link"):
            refused = sandbox.read_file(
                handle,
                ReadFileRequest(call_id=uuid4(), relative_path=runtime_escape),
            )
            assert refused.exit_code == 40
            assert refused.error_type is ErrorType.PATH_VIOLATION

        symlink_cwd = sandbox.run(
            handle,
            RunRequest(call_id=uuid4(), command=symlink_cwd_command),
        )
        assert symlink_cwd.exit_code == 125
        created_via_symlink = sandbox.read_file(
            handle,
            ReadFileRequest(call_id=uuid4(), relative_path="real-dir/cwd-executed"),
        )
        assert created_via_symlink.exit_code == 40
        assert created_via_symlink.error_type is ErrorType.PATH_VIOLATION

        for artifact in sandbox.collect_artifacts(handle):
            artifact_path = artifact_root / handle.sandbox_id / Path(artifact.relative_path).name
            artifact_bytes = artifact_path.read_bytes()
            assert host_canary_value.encode() not in artifact_bytes
            assert str(host_canary).encode() not in artifact_bytes
            assert payload.encode() not in artifact_bytes
            for secret in environment_canaries.values():
                assert secret.encode() not in artifact_bytes
    finally:
        sandbox.destroy(handle)

    assert host_canary.read_text(encoding="utf-8") == host_canary_value
    assert not host_injection_target.exists()
    with pytest.raises(DockerClientError):
        client.inspect_container(container_id)
