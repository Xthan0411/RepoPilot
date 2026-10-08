"""Run a Gold-isolated Django canary through RepoPilot's real Docker Sandbox.

The canary adapts one pinned SWE-bench image locally, extracts an exact clean base
checkout, applies the evaluator test patch only to a temporary Sandbox source, runs
offline setup, proves baseline failure, applies evaluator Gold, and proves the focused
test passes.  It never calls a model Provider and does not declare the formal holdout
ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NoReturn, Self
from uuid import UUID, uuid5

from pydantic import AwareDatetime, Field, TypeAdapter, model_validator
from pydantic_core import to_jsonable_python

from repopilot.enums import PolicyDecision, SandboxCommandPhase, SandboxLifecycle
from repopilot.evals import load_experiment_matrix, select_experiment_arm
from repopilot.evals.dataset import SweBenchSourceRow, load_frozen_dataset
from repopilot.sandbox import (
    ApplyPatchRequest,
    DockerCliClient,
    DockerClient,
    DockerContainerInspection,
    DockerContainerSpec,
    DockerExecResult,
    DockerExecSpec,
    DockerImageIdentity,
    DockerPreparationRecord,
    DockerSandbox,
    RunRequest,
    SandboxHandle,
    SandboxPolicy,
    WorkspaceCopyResult,
)
from repopilot.schemas import CommandSpec, GitSha, Observation, Sha256, StrictModel, TaskSpec

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.p7_swebench_environment import (
    QUALIFICATION_PATH,
    SweBenchUpstreamQualification,
)

SCHEMA_VERSION: Literal[1] = 1
CANARY_VERSION: Literal["p7-django-sandbox-canary-v1"] = "p7-django-sandbox-canary-v1"
TASK_ID: Literal["django__django-15902"] = "django__django-15902"
MATRIX_PATH = Path("configs/experiments/p7-v2.yaml")
DATASET_MANIFEST_PATH = Path("data/manifests/p7-v1/manifest.json")
SOURCE_ROWS_PATH = Path("data/manifests/p7-v1/source-rows.json")
CHECKOUT_PATH = Path(f"artifacts/p7-holdout/checkouts/{TASK_ID}")
RESULT_PATH = Path("results/p7-holdout/django-sandbox-canary-v1.json")
COMMAND_TIMEOUT_CAP_SECONDS: Literal[300] = 300
_CANARY_NAMESPACE = UUID("4638d7e4-5bb0-4463-9f8d-3f1bf1cd31c8")


class CanaryError(RuntimeError):
    """Stable, secret-free canary failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class CanaryCommandEvidence(StrictModel):
    """One output-free setup/test outcome."""

    phase: Literal["setup", "baseline", "gold"]
    command_index: int = Field(ge=0, strict=True)
    declared_timeout_seconds: int = Field(gt=0, strict=True)
    effective_timeout_seconds: int = Field(gt=0, le=300, strict=True)
    network_requested: Literal[False] = False
    policy_decision: Literal[PolicyDecision.ALLOWED] = PolicyDecision.ALLOWED
    exit_code: int


class DjangoSandboxCanary(StrictModel):
    """Gold-free result proving one adapted image works with the real Sandbox."""

    schema_version: Literal[1] = SCHEMA_VERSION
    canary_version: Literal["p7-django-sandbox-canary-v1"] = CANARY_VERSION
    controller_commit: GitSha
    collected_at: AwareDatetime
    task_id: Literal["django__django-15902"] = TASK_ID
    base_commit: GitSha
    upstream_qualification_sha256: Sha256
    upstream_image_reference: str
    adapted_local_image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    adapted_policy_reference: str = Field(
        pattern=r"^repopilot\.local/[a-z0-9._/-]+@sha256:[0-9a-f]{64}$"
    )
    checkout_relative_path: Literal["artifacts/p7-holdout/checkouts/django__django-15902"] = (
        "artifacts/p7-holdout/checkouts/django__django-15902"
    )
    checkout_commit: GitSha
    checkout_clean: Literal[True]
    observed_python_version: str = Field(pattern=r"^3\.9\.[0-9]+$")
    sandbox_preparation: DockerPreparationRecord
    setup_commands: tuple[CanaryCommandEvidence, ...]
    baseline_test: CanaryCommandEvidence
    gold_test: CanaryCommandEvidence
    test_patch_persisted_outside_temporary_evaluator: Literal[False] = False
    gold_persisted_outside_temporary_evaluator: Literal[False] = False
    gold_visible_to_agent: Literal[False] = False
    paid_requests_made: Literal[0] = 0
    compatibility_canary_passed: Literal[True]
    formal_environment_ready: Literal[False]
    formal_blockers: tuple[
        Literal[
            "registry_digest_required",
            "task_command_timeout_contract_mismatch",
            "remaining_tasks_unverified",
        ],
        ...,
    ]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_canary(self) -> Self:
        if self.checkout_commit != self.base_commit:
            raise ValueError("canary checkout is not at the exact base commit")
        if any(item.exit_code != 0 for item in self.setup_commands):
            raise ValueError("canary setup did not pass")
        if self.baseline_test.exit_code == 0 or self.gold_test.exit_code != 0:
            raise ValueError("canary baseline/gold outcomes are invalid")
        expected_blockers = (
            "registry_digest_required",
            "task_command_timeout_contract_mismatch",
            "remaining_tasks_unverified",
        )
        if self.formal_blockers != expected_blockers:
            raise ValueError("canary formal blockers are incomplete")
        _validate_content_digest(self)
        return self


class LocalImageIdDockerClient:
    """Translate one policy-shaped digest to an exact local Docker config digest."""

    def __init__(
        self,
        *,
        delegate: DockerClient,
        policy_reference: str,
        image_id: str,
    ) -> None:
        self._delegate = delegate
        self._policy_reference = policy_reference
        self._image_id = image_id

    def ping(self) -> None:
        self._delegate.ping()

    def resolve_image(self, image_reference: str) -> DockerImageIdentity:
        if image_reference != self._policy_reference:
            raise CanaryError("image_identity_mismatch")
        identity = self._delegate.resolve_image(self._image_id)
        if identity.image_id != self._image_id:
            raise CanaryError("image_identity_mismatch")
        return identity

    def create_container(self, spec: DockerContainerSpec) -> str:
        if spec.image_reference != self._policy_reference:
            raise CanaryError("image_identity_mismatch")
        return self._delegate.create_container(
            spec.model_copy(update={"image_reference": self._image_id})
        )

    def start_container(self, container_id: str) -> None:
        self._delegate.start_container(container_id)

    def copy_directory(
        self,
        container_id: str,
        source: Path,
        *,
        max_bytes: int,
    ) -> WorkspaceCopyResult:
        return self._delegate.copy_directory(container_id, source, max_bytes=max_bytes)

    def copy_artifact(
        self,
        container_id: str,
        name: str,
        content: bytes,
        *,
        max_bytes: int,
    ) -> None:
        self._delegate.copy_artifact(container_id, name, content, max_bytes=max_bytes)

    def inspect_container(self, container_id: str) -> DockerContainerInspection:
        return self._delegate.inspect_container(container_id)

    def execute(
        self,
        container_id: str,
        specification: DockerExecSpec,
    ) -> DockerExecResult:
        return self._delegate.execute(container_id, specification)

    def connect_setup_network(self, container_id: str) -> None:
        self._delegate.connect_setup_network(container_id)

    def disconnect_setup_network(self, container_id: str) -> None:
        self._delegate.disconnect_setup_network(container_id)

    def remove_container(self, container_id: str) -> None:
        self._delegate.remove_container(container_id)


def cap_command_timeouts(task: TaskSpec) -> TaskSpec:
    """Create the explicit canary-only projection required by the frozen 300s policy."""

    def cap(command: CommandSpec) -> CommandSpec:
        declared = command.timeout_seconds or COMMAND_TIMEOUT_CAP_SECONDS
        return command.model_copy(
            update={"timeout_seconds": min(declared, COMMAND_TIMEOUT_CAP_SECONDS)}
        )

    return task.model_copy(
        update={
            "setup_commands": tuple(cap(command) for command in task.setup_commands),
            "test_commands": tuple(cap(command) for command in task.test_commands),
        }
    )


def build_dockerfile(
    upstream_image_reference: str,
    *,
    distribution_name: str = "Django",
) -> str:
    """Build a Gold-free adapter layer with offline, writable user-site pip."""

    if not distribution_name or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in distribution_name
    ):
        raise CanaryError("distribution_name_invalid")

    return f"""FROM {upstream_image_reference}
USER root
SHELL [\"/bin/bash\", \"-lc\"]
RUN source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && \\
    python -m pip uninstall -y {distribution_name}
RUN getent group 65532 >/dev/null || printf 'repopilot:x:65532:\\n' >> /etc/group; \\
    getent passwd 65532 >/dev/null || \\
    printf 'repopilot:x:65532:65532:RepoPilot:/workspace:/usr/sbin/nologin\\n' >> /etc/passwd
RUN printf '%s\\n' \\
    '[global]' \\
    'no-index = true' \\
    'user = true' \\
    'disable-pip-version-check = true' \\
    'no-build-isolation = false' > /etc/pip.conf
ENV HOME=\"/workspace\"
ENV PYTHONUSERBASE=\"/workspace/.local\"
ENV TMPDIR=\"/workspace\"
ENV PATH=\"/workspace/.local/bin:/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:${{PATH}}\"
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve(strict=True)
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise CanaryError("host_platform_mismatch")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise CanaryError("controller_source_dirty")
    if (root / RESULT_PATH).exists():
        raise CanaryError("output_conflict")
    controller_commit = _git(root, "rev-parse", "HEAD")
    qualification = SweBenchUpstreamQualification.model_validate_json(
        (root / QUALIFICATION_PATH).read_bytes()
    )
    _verify_ref(root, qualification.baseline_report_ref)
    _verify_ref(root, qualification.gold_report_ref)
    upstream_task = next(item for item in qualification.task_images if item.task_id == TASK_ID)
    if not upstream_task.python_matches_declaration:
        raise CanaryError("django_python_mismatch")
    bundle = load_frozen_dataset(root / DATASET_MANIFEST_PATH, repository_root=root)
    evaluation = next(
        item for item in bundle.external_evaluation_tasks if item.task.task_id == TASK_ID
    )
    source_row = _source_row(root)
    checkout = _prepare_clean_checkout(
        root,
        image_reference=upstream_task.source_image_reference,
        base_commit=evaluation.task.base_commit,
    )
    local_tag, image_id, policy_reference = _build_adapter_image(
        upstream_task.source_image_reference
    )
    del local_tag
    matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
    settings = select_experiment_arm(matrix, "no-semantic-rag-v1").settings(
        openai_api_key="canary-provider-not-used",
        database_url="postgresql://repopilot:repopilot@localhost:55433/repopilot",
    )
    with tempfile.TemporaryDirectory(prefix="repopilot-p7-django-evaluator-") as temporary:
        temporary_root = Path(temporary)
        evaluator_checkout = temporary_root / "repository"
        _run(
            "git",
            "clone",
            "--quiet",
            "--no-hardlinks",
            str(checkout),
            str(evaluator_checkout),
            timeout=120,
        )
        _run_input(
            ("git", "-C", str(evaluator_checkout), "apply", "--whitespace=nowarn", "-"),
            source_row.test_patch.encode("utf-8"),
            timeout=30,
        )
        task = cap_command_timeouts(
            evaluation.task.model_copy(
                update={
                    "repository_path": str(evaluator_checkout),
                    "repository_url": None,
                }
            )
        )
        artifact_root = temporary_root / "sandbox-artifacts"
        client = LocalImageIdDockerClient(
            delegate=DockerCliClient(),
            policy_reference=policy_reference,
            image_id=image_id,
        )
        sandbox = DockerSandbox(
            policy=SandboxPolicy.from_settings(settings),
            image_reference=policy_reference,
            artifact_root=artifact_root,
            client=client,
            id_factory=lambda: "p7-django-canary-v1",
        )
        handle: SandboxHandle | None = None
        try:
            handle = sandbox.prepare(task)
            setup_evidence = _run_setup(
                sandbox,
                handle,
                task,
                declared_task=evaluation.task,
            )
            baseline = _run_test(sandbox, handle, task, phase="baseline", call_suffix="baseline")
            if baseline.exit_code == 0:
                raise CanaryError("baseline_unexpectedly_passed")
            patch_call = uuid5(_CANARY_NAMESPACE, f"{TASK_ID}:gold-patch")
            patch_ref = sandbox.stage_patch(
                handle,
                patch_call,
                evaluation.expected_patch.encode("utf-8"),
            )
            patch_observation = sandbox.apply_patch(
                handle,
                ApplyPatchRequest(call_id=patch_call, patch_ref=patch_ref),
            )
            if patch_observation.exit_code != 0 or patch_observation.error_type is not None:
                raise CanaryError("gold_patch_not_applied")
            gold = _run_test(sandbox, handle, task, phase="gold", call_suffix="gold")
            if gold.exit_code != 0:
                raise CanaryError("gold_test_failed")
            preparation = sandbox.preparation
            if preparation is None:
                raise CanaryError("sandbox_preparation_missing")
        finally:
            if handle is not None and sandbox.lifecycle is SandboxLifecycle.PREPARED:
                sandbox.destroy(handle)
    observed_python = _probe_python(image_id)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "canary_version": CANARY_VERSION,
        "controller_commit": controller_commit,
        "collected_at": datetime.now(UTC),
        "task_id": TASK_ID,
        "base_commit": evaluation.task.base_commit,
        "upstream_qualification_sha256": qualification.content_sha256,
        "upstream_image_reference": upstream_task.source_image_reference,
        "adapted_local_image_id": image_id,
        "adapted_policy_reference": policy_reference,
        "checkout_relative_path": CHECKOUT_PATH.as_posix(),
        "checkout_commit": _git(checkout, "rev-parse", "HEAD"),
        "checkout_clean": not bool(_git(checkout, "status", "--porcelain=v1")),
        "observed_python_version": observed_python,
        "sandbox_preparation": preparation,
        "setup_commands": setup_evidence,
        "baseline_test": baseline,
        "gold_test": gold,
        "test_patch_persisted_outside_temporary_evaluator": False,
        "gold_persisted_outside_temporary_evaluator": False,
        "gold_visible_to_agent": False,
        "paid_requests_made": 0,
        "compatibility_canary_passed": True,
        "formal_environment_ready": False,
        "formal_blockers": (
            "registry_digest_required",
            "task_command_timeout_contract_mismatch",
            "remaining_tasks_unverified",
        ),
    }
    result = DjangoSandboxCanary.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )
    _write_private_create_only(
        root / RESULT_PATH,
        result.model_dump_json(indent=2).encode("utf-8") + b"\n",
    )
    print(
        "django_canary_passed=true setup=offline baseline_failed=true gold_passed=true "
        "paid_requests=0 formal_ready=false blockers=3"
    )
    return 0


def _source_row(root: Path) -> SweBenchSourceRow:
    rows = TypeAdapter(tuple[SweBenchSourceRow, ...]).validate_json(
        (root / SOURCE_ROWS_PATH).read_bytes()
    )
    return next(item for item in rows if item.instance_id == TASK_ID)


def _prepare_clean_checkout(
    root: Path,
    *,
    image_reference: str,
    base_commit: str,
) -> Path:
    target = root / CHECKOUT_PATH
    if target.exists():
        if (
            target.is_dir()
            and _git(target, "rev-parse", "HEAD") == base_commit
            and not _git(target, "status", "--porcelain=v1")
        ):
            return target
        raise CanaryError("checkout_conflict")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{TASK_ID}-", dir=target.parent))
    container_id = ""
    try:
        container_id = _run(
            "docker",
            "container",
            "create",
            "--entrypoint",
            "/bin/true",
            image_reference,
            timeout=30,
        ).strip()
        _run(
            "docker",
            "container",
            "cp",
            f"{container_id}:/testbed/.",
            str(staging),
            timeout=120,
        )
        _run("git", "-C", str(staging), "checkout", "--detach", base_commit, timeout=60)
        _run("git", "-C", str(staging), "reset", "--hard", base_commit, timeout=60)
        _run("git", "-C", str(staging), "clean", "-ffdqx", timeout=60)
        if _git(staging, "rev-parse", "HEAD") != base_commit or _git(
            staging, "status", "--porcelain=v1"
        ):
            raise CanaryError("checkout_identity_mismatch")
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if container_id:
            subprocess.run(
                ("docker", "container", "rm", "--force", container_id),
                check=False,
                capture_output=True,
                timeout=30,
            )
    return target


def _build_adapter_image(upstream_image_reference: str) -> tuple[str, str, str]:
    local_tag = "repopilot.local/p7/django-django-15902:canary-v1"
    with tempfile.TemporaryDirectory(prefix="repopilot-p7-django-image-") as temporary:
        dockerfile = Path(temporary) / "Dockerfile"
        dockerfile.write_text(build_dockerfile(upstream_image_reference), encoding="utf-8")
        _run(
            "docker",
            "build",
            "--network",
            "none",
            "--tag",
            local_tag,
            "--file",
            str(dockerfile),
            temporary,
            timeout=600,
        )
    inspected = json.loads(_run("docker", "image", "inspect", local_tag, timeout=30))[0]
    image_id = inspected["Id"]
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise CanaryError("adapted_image_invalid")
    policy_reference = "repopilot.local/p7/django-django-15902@sha256:" + image_id.removeprefix(
        "sha256:"
    )
    return local_tag, image_id, policy_reference


def _run_setup(
    sandbox: DockerSandbox,
    handle: SandboxHandle,
    task: TaskSpec,
    *,
    declared_task: TaskSpec,
) -> tuple[CanaryCommandEvidence, ...]:
    evidence: list[CanaryCommandEvidence] = []
    for index, command in enumerate(task.setup_commands):
        observation = sandbox.run(
            handle,
            RunRequest(
                call_id=uuid5(_CANARY_NAMESPACE, f"{TASK_ID}:setup:{index}"),
                command=command,
                phase=SandboxCommandPhase.SETUP,
                network_requested=False,
            ),
        )
        if (
            observation.exit_code != 0
            or observation.error_type is not None
            or observation.policy_decision is not PolicyDecision.ALLOWED
        ):
            _raise_command_failure(f"setup_{index}_failed", observation)
        declared = declared_task.setup_commands[index].timeout_seconds
        if declared is None or command.timeout_seconds is None:
            raise CanaryError("setup_timeout_missing")
        evidence.append(
            CanaryCommandEvidence(
                phase="setup",
                command_index=index,
                declared_timeout_seconds=declared,
                effective_timeout_seconds=command.timeout_seconds,
                exit_code=observation.exit_code,
            )
        )
    return tuple(evidence)


def _raise_command_failure(code: str, observation: Observation) -> NoReturn:
    """Emit bounded public-task diagnostics before raising the stable error code."""

    diagnostic = {
        "code": code,
        "exit_code": observation.exit_code,
        "error_type": observation.error_type,
        "policy_decision": observation.policy_decision,
        "stdout_tail": (observation.stdout_preview or "")[-4_000:],
        "stderr_tail": (observation.stderr_preview or "")[-4_000:],
    }
    print(
        "canary_command_failure="
        + json.dumps(to_jsonable_python(diagnostic), ensure_ascii=False, sort_keys=True),
        file=sys.stderr,
    )
    raise CanaryError(code)


def _run_test(
    sandbox: DockerSandbox,
    handle: SandboxHandle,
    task: TaskSpec,
    *,
    phase: Literal["baseline", "gold"],
    call_suffix: str,
) -> CanaryCommandEvidence:
    command = task.test_commands[0]
    observation = sandbox.run(
        handle,
        RunRequest(
            call_id=uuid5(_CANARY_NAMESPACE, f"{TASK_ID}:test:{call_suffix}"),
            command=command,
            phase=SandboxCommandPhase.TEST,
            network_requested=False,
        ),
    )
    if (
        observation.exit_code is None
        or observation.error_type is not None
        or observation.policy_decision is not PolicyDecision.ALLOWED
        or command.timeout_seconds is None
    ):
        raise CanaryError(f"{phase}_execution_failed")
    return CanaryCommandEvidence(
        phase=phase,
        command_index=0,
        declared_timeout_seconds=600,
        effective_timeout_seconds=command.timeout_seconds,
        exit_code=observation.exit_code,
    )


def _probe_python(image_id: str) -> str:
    return _run(
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--entrypoint",
        "python",
        image_id,
        "-c",
        "import platform; print(platform.python_version())",
        timeout=30,
    ).strip()


def _verify_ref(root: Path, reference: object) -> None:
    relative_path = getattr(reference, "relative_path", None)
    expected_sha256 = getattr(reference, "sha256", None)
    expected_size = getattr(reference, "size_bytes", None)
    if not isinstance(relative_path, str):
        raise CanaryError("qualification_reference_invalid")
    path = root / relative_path
    content = path.read_bytes()
    if len(content) != expected_size or hashlib.sha256(content).hexdigest() != expected_sha256:
        raise CanaryError("qualification_reference_invalid")


def _write_private_create_only(path: Path, content: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
    except OSError:
        raise CanaryError("output_conflict") from None


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_content_digest(value: StrictModel) -> None:
    payload = value.model_dump(mode="python", exclude={"content_sha256"})
    if _canonical_sha256(payload) != getattr(value, "content_sha256", None):
        raise ValueError("canary content digest does not match")


def _git(root: Path, *arguments: str) -> str:
    return _run("git", "-C", str(root), *arguments, timeout=60).strip()


def _run(*arguments: str, timeout: int) -> str:
    completed = subprocess.run(
        arguments,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout


def _run_input(arguments: tuple[str, ...], content: bytes, *, timeout: int) -> None:
    subprocess.run(
        arguments,
        input=content,
        check=True,
        capture_output=True,
        timeout=timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
