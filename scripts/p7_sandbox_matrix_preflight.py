"""Qualify all frozen P7 holdout tasks through RepoPilot's real Docker Sandbox.

This evaluator-only preflight adapts the pinned task environments locally, extracts exact
clean base checkouts, applies each test patch only inside a temporary evaluator copy,
runs setup offline, proves baseline failure, applies evaluator Gold, and proves the
focused tests pass. It never calls a model Provider and never exposes Gold to an Agent.
Local image IDs are compatibility evidence, not a formal registry manifest.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self
from uuid import UUID, uuid5

from pydantic import AwareDatetime, Field, TypeAdapter, model_validator

from repopilot.config import RepoPilotSettings
from repopilot.enums import PolicyDecision, SandboxCommandPhase, SandboxLifecycle
from repopilot.evals import load_experiment_matrix, select_experiment_arm
from repopilot.evals.dataset import SweBenchSourceRow, load_frozen_dataset
from repopilot.sandbox import (
    ApplyPatchRequest,
    DockerCliClient,
    DockerPreparationRecord,
    DockerSandbox,
    RunRequest,
    SandboxHandle,
    SandboxPolicy,
)
from repopilot.schemas import EvaluationTaskSpec, GitSha, Sha256, StrictModel, TaskSpec

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.p7_django_sandbox_canary import (
    CanaryCommandEvidence,
    CanaryError,
    LocalImageIdDockerClient,
    _canonical_sha256,
    _git,
    _probe_python,
    _raise_command_failure,
    _run,
    _run_input,
    _verify_ref,
    _write_private_create_only,
    build_dockerfile,
    cap_command_timeouts,
)
from scripts.p7_swebench_environment import (
    EXPECTED_TASK_IDS,
    HARNESS_REVISION,
    QUALIFICATION_PATH,
    SweBenchUpstreamQualification,
    python_version_matches,
)

SCHEMA_VERSION: Literal[1] = 1
PREFLIGHT_VERSION: Literal["p7-sandbox-matrix-preflight-v1"] = "p7-sandbox-matrix-preflight-v1"
MATRIX_PATH = Path("configs/experiments/p7-v2.yaml")
DATASET_MANIFEST_PATH = Path("data/manifests/p7-v1/manifest.json")
SOURCE_ROWS_PATH = Path("data/manifests/p7-v1/source-rows.json")
CHECKOUT_ROOT = Path("artifacts/p7-holdout/checkouts")
RESULT_PATH = Path("results/p7-holdout/sandbox-matrix-preflight-v1.json")
_PREFLIGHT_NAMESPACE = UUID("3509ea47-f190-4b2d-a296-41db69b6f6b0")
_DISTRIBUTIONS = {
    "django__django-15902": "Django",
    "mwaskom__seaborn-3190": "seaborn",
    "pallets__flask-4992": "Flask",
    "pydata__xarray-4094": "xarray",
    "pydata__xarray-5131": "xarray",
    "pytest-dev__pytest-11143": "pytest",
    "pytest-dev__pytest-11148": "pytest",
    "sympy__sympy-20590": "sympy",
}
_FLASK_TASK_ID = "pallets__flask-4992"
_FLASK_BUILD_TOOLS = ("wheel==0.45.1",)
_SCM_BUILD_TOOLS = ("setuptools-scm==7.1.0",)
_SYMPY_TEST_TOOLS = (
    "iniconfig==2.0.0",
    "packaging==23.1",
    "pluggy==1.3.0",
    "exceptiongroup==1.1.3",
    "tomli==2.0.1",
    "pytest==7.4.0",
)
_TASK_BUILD_TOOLS = {
    # Exact base metadata uses setup_requires/setuptools_scm for xarray 4094,
    # PEP 517 setuptools_scm for xarray 5131 and both pytest tasks, and the
    # legacy archive plugin only for xarray 5131. These are evaluator build
    # prerequisites; task setup remains network-disabled inside the Sandbox.
    "pydata__xarray-4094": _SCM_BUILD_TOOLS,
    "pydata__xarray-5131": (
        *_SCM_BUILD_TOOLS,
        "setuptools-scm-git-archive==1.4.1",
    ),
    "pytest-dev__pytest-11143": _SCM_BUILD_TOOLS,
    "pytest-dev__pytest-11148": _SCM_BUILD_TOOLS,
    "sympy__sympy-20590": _SYMPY_TEST_TOOLS,
}
_TASK_SCM_PRETEND_VERSIONS = {
    # RepoPilot deliberately excludes .git from the Sandbox workspace archive. Preserve
    # the exact versions already installed in the digest-pinned upstream task images.
    "pydata__xarray-4094": "0.15.2.dev110+ga64cf2d54",
    "pytest-dev__pytest-11143": "8.0.0.dev53+g6995257cf",
    "pytest-dev__pytest-11148": "8.0.0.dev46+g2f7415cfb",
}
_PINNED_PACKAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*==[A-Za-z0-9][A-Za-z0-9._+-]*$")
_SCM_PRETEND_VERSION = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+-]*$")


class SandboxTaskPreflightEvidence(StrictModel):
    """Gold-free durable evidence for one temporary evaluator qualification."""

    task_id: str
    base_commit: GitSha
    declared_python_version: str
    observed_python_version: str
    python_matches_declaration: bool
    upstream_image_reference: str
    adapter_base_image_reference: str
    adapter_build_network: Literal["none", "trusted_dependency_bootstrap"]
    adapter_bootstrap_packages: tuple[str, ...]
    adapter_build_tools: tuple[str, ...]
    adapter_scm_pretend_version: str | None
    adapted_local_image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    adapted_policy_reference: str = Field(
        pattern=r"^repopilot\.local/[a-z0-9._/-]+@sha256:[0-9a-f]{64}$"
    )
    platform: Literal["linux/amd64"] = "linux/amd64"
    checkout_relative_path: str
    checkout_commit: GitSha
    checkout_clean: Literal[True]
    sandbox_preparation: DockerPreparationRecord
    setup_commands: tuple[CanaryCommandEvidence, ...]
    baseline_tests: tuple[CanaryCommandEvidence, ...]
    gold_tests: tuple[CanaryCommandEvidence, ...]
    setup_offline: Literal[True] = True
    baseline_failed: Literal[True] = True
    gold_passed: Literal[True] = True
    test_patch_persisted_outside_temporary_evaluator: Literal[False] = False
    gold_persisted_outside_temporary_evaluator: Literal[False] = False
    gold_visible_to_agent: Literal[False] = False

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.checkout_commit != self.base_commit:
            raise ValueError("preflight checkout is not at the exact base commit")
        if self.python_matches_declaration != python_version_matches(
            self.declared_python_version,
            self.observed_python_version,
        ):
            raise ValueError("preflight Python match decision is inconsistent")
        if any(item.exit_code != 0 for item in self.setup_commands):
            raise ValueError("preflight setup did not pass")
        if not self.baseline_tests or all(item.exit_code == 0 for item in self.baseline_tests):
            raise ValueError("preflight baseline did not fail")
        if not self.gold_tests or any(item.exit_code != 0 for item in self.gold_tests):
            raise ValueError("preflight Gold did not pass")
        expected_path = f"artifacts/p7-holdout/checkouts/{self.task_id}"
        if self.checkout_relative_path != expected_path:
            raise ValueError("preflight checkout path does not match task identity")
        if not self.adapted_policy_reference.endswith(
            "@sha256:" + self.adapted_local_image_id.removeprefix("sha256:")
        ):
            raise ValueError("preflight policy reference does not match local image ID")
        if (self.adapter_build_network == "trusted_dependency_bootstrap") != bool(
            self.adapter_bootstrap_packages or self.adapter_build_tools
        ):
            raise ValueError("adapter bootstrap network and package evidence disagree")
        if self.adapter_scm_pretend_version != _TASK_SCM_PRETEND_VERSIONS.get(self.task_id):
            raise ValueError("adapter SCM version evidence does not match task identity")
        return self


class SandboxMatrixPreflight(StrictModel):
    """All-eight Sandbox evidence with unresolved formal blockers kept explicit."""

    schema_version: Literal[1] = SCHEMA_VERSION
    preflight_version: Literal["p7-sandbox-matrix-preflight-v1"] = PREFLIGHT_VERSION
    controller_commit: GitSha
    collected_at: AwareDatetime
    upstream_qualification_sha256: Sha256
    harness_revision: GitSha
    host_platform: Literal["linux/amd64"] = "linux/amd64"
    task_evidence: tuple[SandboxTaskPreflightEvidence, ...]
    adapter_verified_task_count: Literal[8]
    required_task_count: Literal[8] = 8
    python_mismatch_task_ids: tuple[str, ...]
    paid_requests_made: Literal[0] = 0
    gold_visible_to_agent: Literal[False] = False
    compatibility_preflight_passed: Literal[True]
    formal_environment_ready: Literal[False]
    formal_blockers: tuple[
        Literal[
            "registry_digest_required",
            "task_command_timeout_contract_mismatch",
            "task_python_mismatch",
        ],
        ...,
    ]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_preflight(self) -> Self:
        if tuple(item.task_id for item in self.task_evidence) != EXPECTED_TASK_IDS:
            raise ValueError("matrix preflight must cover the exact frozen task set")
        mismatches = tuple(
            item.task_id for item in self.task_evidence if not item.python_matches_declaration
        )
        if self.python_mismatch_task_ids != mismatches:
            raise ValueError("matrix preflight Python mismatches are inconsistent")
        expected_blockers = (
            "registry_digest_required",
            "task_command_timeout_contract_mismatch",
            *(("task_python_mismatch",) if mismatches else ()),
        )
        if self.formal_blockers != expected_blockers:
            raise ValueError("matrix preflight formal blockers are inconsistent")
        payload = self.model_dump(mode="python", exclude={"content_sha256"})
        if _canonical_sha256(payload) != self.content_sha256:
            raise ValueError("matrix preflight content digest does not match")
        return self


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--task-id",
        action="append",
        choices=EXPECTED_TASK_IDS,
        help=(
            "run only the selected task during diagnosis; repeat this option for "
            "multiple tasks (a subset never writes the formal all-task artifact)"
        ),
    )
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve(strict=True)
    task_ids = _selected_task_ids(arguments.task_id)
    diagnostic_only = task_ids != EXPECTED_TASK_IDS
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise CanaryError("host_platform_mismatch")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise CanaryError("controller_source_dirty")
    if not diagnostic_only and (root / RESULT_PATH).exists():
        raise CanaryError("output_conflict")

    controller_commit = _git(root, "rev-parse", "HEAD")
    qualification = SweBenchUpstreamQualification.model_validate_json(
        (root / QUALIFICATION_PATH).read_bytes()
    )
    _verify_ref(root, qualification.baseline_report_ref)
    _verify_ref(root, qualification.gold_report_ref)
    if qualification.harness_revision != HARNESS_REVISION:
        raise CanaryError("harness_revision_mismatch")

    bundle = load_frozen_dataset(root / DATASET_MANIFEST_PATH, repository_root=root)
    evaluations = {item.task.task_id: item for item in bundle.external_evaluation_tasks}
    source_rows = {
        item.instance_id: item
        for item in TypeAdapter(tuple[SweBenchSourceRow, ...]).validate_json(
            (root / SOURCE_ROWS_PATH).read_bytes()
        )
    }
    upstream_images = {item.task_id: item for item in qualification.task_images}
    if (
        tuple(evaluations) != EXPECTED_TASK_IDS
        or tuple(source_rows) != EXPECTED_TASK_IDS
        or tuple(upstream_images) != EXPECTED_TASK_IDS
    ):
        raise CanaryError("task_set_mismatch")

    matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
    settings = select_experiment_arm(matrix, "no-semantic-rag-v1").settings(
        openai_api_key="preflight-provider-not-used",
        database_url="postgresql://repopilot:repopilot@localhost:55433/repopilot",
    )
    evidence: list[SandboxTaskPreflightEvidence] = []
    failures: list[dict[str, str]] = []
    for task_id in task_ids:
        print(f"sandbox_task_started={task_id}", flush=True)
        evaluation = evaluations[task_id]
        adapter_base_image_reference = upstream_images[task_id].source_image_reference
        adapter_bootstrap_packages: tuple[str, ...] = ()
        adapter_build_tools = _TASK_BUILD_TOOLS.get(task_id, ())
        adapter_scm_pretend_version = _TASK_SCM_PRETEND_VERSIONS.get(task_id)
        if task_id == _FLASK_TASK_ID:
            adapter_base_image_reference = matrix.sandbox_image_reference
            adapter_bootstrap_packages = _dependency_packages(evaluation.task)
            adapter_build_tools = _FLASK_BUILD_TOOLS
        try:
            item = _qualify_task(
                root=root,
                evaluation=evaluation,
                source_row=source_rows[task_id],
                upstream_image_reference=upstream_images[task_id].source_image_reference,
                adapter_base_image_reference=adapter_base_image_reference,
                adapter_bootstrap_packages=adapter_bootstrap_packages,
                adapter_build_tools=adapter_build_tools,
                adapter_scm_pretend_version=adapter_scm_pretend_version,
                declared_python_version=upstream_images[task_id].declared_python_version,
                settings=settings,
            )
        except CanaryError as error:
            failures.append({"task_id": task_id, "code": error.code})
            print(
                f"sandbox_task_failed={task_id} code={error.code}",
                file=sys.stderr,
                flush=True,
            )
            continue
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            code = (
                "preflight_subprocess_timeout"
                if isinstance(error, subprocess.TimeoutExpired)
                else "preflight_subprocess_failed"
            )
            failures.append({"task_id": task_id, "code": code})
            print(
                f"sandbox_task_failed={task_id} code={code}",
                file=sys.stderr,
                flush=True,
            )
            continue
        evidence.append(item)
        print(
            f"sandbox_task_passed={task_id} "
            f"python_match={str(item.python_matches_declaration).lower()}",
            flush=True,
        )

    if failures:
        print(
            "sandbox_matrix_failures=" + json.dumps(failures, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        print(
            f"sandbox_matrix_preflight_passed=false attempted={len(task_ids)} "
            f"passed={len(evidence)} failed={len(failures)} paid_requests=0",
            flush=True,
        )
        return 1
    if diagnostic_only:
        print(
            f"sandbox_matrix_diagnostic_passed=true tasks={len(evidence)} "
            "artifact_written=false paid_requests=0",
            flush=True,
        )
        return 0

    task_evidence = tuple(evidence)
    mismatches = tuple(
        item.task_id for item in task_evidence if not item.python_matches_declaration
    )
    blockers = (
        "registry_digest_required",
        "task_command_timeout_contract_mismatch",
        *(("task_python_mismatch",) if mismatches else ()),
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "preflight_version": PREFLIGHT_VERSION,
        "controller_commit": controller_commit,
        "collected_at": datetime.now(UTC),
        "upstream_qualification_sha256": qualification.content_sha256,
        "harness_revision": qualification.harness_revision,
        "host_platform": "linux/amd64",
        "task_evidence": task_evidence,
        "adapter_verified_task_count": 8,
        "required_task_count": 8,
        "python_mismatch_task_ids": mismatches,
        "paid_requests_made": 0,
        "gold_visible_to_agent": False,
        "compatibility_preflight_passed": True,
        "formal_environment_ready": False,
        "formal_blockers": blockers,
    }
    result = SandboxMatrixPreflight.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )
    _write_private_create_only(
        root / RESULT_PATH,
        result.model_dump_json(indent=2).encode("utf-8") + b"\n",
    )
    print(
        "sandbox_matrix_preflight_passed=true tasks=8 setup=offline "
        f"python_mismatches={len(mismatches)} paid_requests=0 "
        f"formal_ready=false blockers={len(blockers)}"
    )
    return 0


def _qualify_task(
    *,
    root: Path,
    evaluation: EvaluationTaskSpec,
    source_row: SweBenchSourceRow,
    upstream_image_reference: str,
    adapter_base_image_reference: str,
    adapter_bootstrap_packages: tuple[str, ...],
    adapter_build_tools: tuple[str, ...],
    adapter_scm_pretend_version: str | None,
    declared_python_version: str,
    settings: RepoPilotSettings,
) -> SandboxTaskPreflightEvidence:
    task_id = evaluation.task.task_id
    checkout = _prepare_clean_checkout(
        root,
        task_id=task_id,
        image_reference=upstream_image_reference,
        base_commit=evaluation.task.base_commit,
    )
    image_id, policy_reference = _build_adapter_image(
        adapter_base_image_reference,
        task_id=task_id,
        distribution_name=_DISTRIBUTIONS[task_id],
        bootstrap_packages=adapter_bootstrap_packages,
        build_tools=adapter_build_tools,
        scm_pretend_version=adapter_scm_pretend_version,
    )
    with tempfile.TemporaryDirectory(prefix=f"repopilot-p7-{_safe_name(task_id)}-") as temporary:
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
                update={"repository_path": str(evaluator_checkout), "repository_url": None}
            )
        )
        client = LocalImageIdDockerClient(
            delegate=DockerCliClient(),
            policy_reference=policy_reference,
            image_id=image_id,
        )
        sandbox = DockerSandbox(
            policy=SandboxPolicy.from_settings(settings),
            image_reference=policy_reference,
            artifact_root=temporary_root / "sandbox-artifacts",
            client=client,
            id_factory=lambda: f"p7-{_safe_name(task_id)}",
        )
        handle: SandboxHandle | None = None
        try:
            handle = sandbox.prepare(task)
            setup = _run_setup(sandbox, handle, task, declared_task=evaluation.task)
            baseline = _run_tests(
                sandbox,
                handle,
                task,
                declared_task=evaluation.task,
                phase="baseline",
            )
            if all(item.exit_code == 0 for item in baseline):
                raise CanaryError("baseline_unexpectedly_passed")
            patch_call = uuid5(_PREFLIGHT_NAMESPACE, f"{task_id}:gold-patch")
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
                _raise_command_failure("gold_patch_not_applied", patch_observation)
            gold = _run_tests(
                sandbox,
                handle,
                task,
                declared_task=evaluation.task,
                phase="gold",
            )
            failed_gold = next((item for item in gold if item.exit_code != 0), None)
            if failed_gold is not None:
                raise CanaryError("gold_test_failed")
            preparation = sandbox.preparation
            if preparation is None:
                raise CanaryError("sandbox_preparation_missing")
        finally:
            if handle is not None and sandbox.lifecycle is SandboxLifecycle.PREPARED:
                sandbox.destroy(handle)

    observed_python = _probe_python(image_id)
    relative_checkout = CHECKOUT_ROOT / task_id
    checkout_commit = _git(checkout, "rev-parse", "HEAD")
    if checkout_commit != evaluation.task.base_commit or _git(checkout, "status", "--porcelain=v1"):
        raise CanaryError("checkout_identity_mismatch")
    return SandboxTaskPreflightEvidence(
        task_id=task_id,
        base_commit=evaluation.task.base_commit,
        declared_python_version=declared_python_version,
        observed_python_version=observed_python,
        python_matches_declaration=python_version_matches(
            declared_python_version,
            observed_python,
        ),
        upstream_image_reference=upstream_image_reference,
        adapter_base_image_reference=adapter_base_image_reference,
        adapter_build_network=(
            "trusted_dependency_bootstrap"
            if adapter_bootstrap_packages or adapter_build_tools
            else "none"
        ),
        adapter_bootstrap_packages=adapter_bootstrap_packages,
        adapter_build_tools=adapter_build_tools,
        adapter_scm_pretend_version=adapter_scm_pretend_version,
        adapted_local_image_id=image_id,
        adapted_policy_reference=policy_reference,
        checkout_relative_path=relative_checkout.as_posix(),
        checkout_commit=checkout_commit,
        checkout_clean=True,
        sandbox_preparation=preparation,
        setup_commands=setup,
        baseline_tests=baseline,
        gold_tests=gold,
    )


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
                call_id=uuid5(
                    _PREFLIGHT_NAMESPACE,
                    f"{task.task_id}:setup:{index}",
                ),
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
            _raise_command_failure(f"{task.task_id}:setup_{index}_failed", observation)
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


def _run_tests(
    sandbox: DockerSandbox,
    handle: SandboxHandle,
    task: TaskSpec,
    *,
    declared_task: TaskSpec,
    phase: Literal["baseline", "gold"],
) -> tuple[CanaryCommandEvidence, ...]:
    evidence: list[CanaryCommandEvidence] = []
    for index, command in enumerate(task.test_commands):
        observation = sandbox.run(
            handle,
            RunRequest(
                call_id=uuid5(
                    _PREFLIGHT_NAMESPACE,
                    f"{task.task_id}:{phase}:{index}",
                ),
                command=command,
                phase=SandboxCommandPhase.TEST,
                network_requested=False,
            ),
        )
        if (
            observation.exit_code is None
            or observation.error_type is not None
            or observation.policy_decision is not PolicyDecision.ALLOWED
        ):
            _raise_command_failure(f"{task.task_id}:{phase}_{index}_failed", observation)
        declared = declared_task.test_commands[index].timeout_seconds
        if declared is None or command.timeout_seconds is None:
            raise CanaryError("test_timeout_missing")
        if phase == "gold" and observation.exit_code != 0:
            _raise_command_failure(f"{task.task_id}:gold_{index}_failed", observation)
        evidence.append(
            CanaryCommandEvidence(
                phase=phase,
                command_index=index,
                declared_timeout_seconds=declared,
                effective_timeout_seconds=command.timeout_seconds,
                exit_code=observation.exit_code,
            )
        )
    return tuple(evidence)


def _prepare_clean_checkout(
    root: Path,
    *,
    task_id: str,
    image_reference: str,
    base_commit: str,
) -> Path:
    target = root / CHECKOUT_ROOT / task_id
    if target.exists():
        if (
            target.is_dir()
            and _git(target, "rev-parse", "HEAD") == base_commit
            and not _git(target, "status", "--porcelain=v1")
        ):
            return target
        raise CanaryError("checkout_conflict")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{_safe_name(task_id)}-", dir=target.parent))
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


def _build_adapter_image(
    adapter_base_image_reference: str,
    *,
    task_id: str,
    distribution_name: str,
    bootstrap_packages: tuple[str, ...],
    build_tools: tuple[str, ...],
    scm_pretend_version: str | None,
) -> tuple[str, str]:
    safe_name = _safe_name(task_id)
    local_tag = f"repopilot.local/p7/{safe_name}:sandbox-preflight-v1"
    with tempfile.TemporaryDirectory(prefix=f"repopilot-p7-image-{safe_name}-") as temporary:
        dockerfile = Path(temporary) / "Dockerfile"
        dockerfile_content = (
            build_python_adapter_dockerfile(
                adapter_base_image_reference,
                bootstrap_packages=bootstrap_packages,
                build_tools=build_tools,
            )
            if bootstrap_packages
            else build_conda_adapter_dockerfile(
                adapter_base_image_reference,
                distribution_name=distribution_name,
                build_tools=build_tools,
                scm_pretend_version=scm_pretend_version,
            )
        )
        dockerfile.write_text(dockerfile_content, encoding="utf-8")
        _run_docker_build(
            "docker",
            "build",
            "--network",
            "default" if bootstrap_packages or build_tools else "none",
            "--tag",
            local_tag,
            "--file",
            str(dockerfile),
            temporary,
        )
    inspected = json.loads(_run("docker", "image", "inspect", local_tag, timeout=30))[0]
    image_id = inspected["Id"]
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise CanaryError("adapted_image_invalid")
    policy_reference = f"repopilot.local/p7/{safe_name}@sha256:" + image_id.removeprefix("sha256:")
    return image_id, policy_reference


def build_conda_adapter_dockerfile(
    base_image_reference: str,
    *,
    distribution_name: str,
    build_tools: tuple[str, ...],
    scm_pretend_version: str | None = None,
) -> str:
    """Add pinned build tools before locking the upstream Conda image offline."""

    base_dockerfile = build_dockerfile(
        base_image_reference,
        distribution_name=distribution_name,
    )
    if any(_PINNED_PACKAGE.fullmatch(package) is None for package in build_tools):
        raise CanaryError("build_tools_invalid")
    if scm_pretend_version is not None and (
        _SCM_PRETEND_VERSION.fullmatch(scm_pretend_version) is None
        or not any(package.startswith("setuptools-scm==") for package in build_tools)
    ):
        raise CanaryError("scm_pretend_version_invalid")
    result = base_dockerfile
    if build_tools:
        install = (
            "RUN source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && \\\n"
            "    python -m pip install --disable-pip-version-check --no-cache-dir --no-deps "
            + " ".join(build_tools)
            + "\n"
        )
        uninstall_marker = "RUN source /opt/miniconda3/etc/profile.d/conda.sh"
        result = result.replace(uninstall_marker, install + uninstall_marker, 1)
    if scm_pretend_version is not None:
        scm_distribution = re.sub(r"[-_.]+", "_", distribution_name).upper()
        result += (
            f'ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_{scm_distribution}="{scm_pretend_version}"\n'
        )
    return result


def build_python_adapter_dockerfile(
    base_image_reference: str,
    *,
    bootstrap_packages: tuple[str, ...],
    build_tools: tuple[str, ...],
) -> str:
    """Build the frozen Python 3.12 Flask environment before offline Sandbox setup."""

    all_packages = (*bootstrap_packages, *build_tools)
    if not bootstrap_packages or any(
        _PINNED_PACKAGE.fullmatch(package) is None for package in all_packages
    ):
        raise CanaryError("bootstrap_packages_invalid")
    package_arguments = " ".join(all_packages)
    return f"""FROM {base_image_reference}
USER root
SHELL [\"/bin/sh\", \"-c\"]
RUN python -m pip install --disable-pip-version-check --no-cache-dir {package_arguments}
RUN getent group 65532 >/dev/null || printf 'repopilot:x:65532:\\n' >> /etc/group; \\
    getent passwd 65532 >/dev/null || \\
    printf 'repopilot:x:65532:65532:RepoPilot:/workspace:/sbin/nologin\\n' >> /etc/passwd
RUN printf '%s\\n' \\
    '[global]' \\
    'no-index = true' \\
    'user = true' \\
    'disable-pip-version-check = true' \\
    'no-build-isolation = false' > /etc/pip.conf
ENV HOME=\"/workspace\"
ENV PYTHONUSERBASE=\"/workspace/.local\"
ENV TMPDIR=\"/workspace\"
ENV PATH=\"/workspace/.local/bin:${{PATH}}\"
"""


def _selected_task_ids(requested: list[str] | None) -> tuple[str, ...]:
    """Return a stable, de-duplicated task subset in frozen matrix order."""

    if requested is None:
        return EXPECTED_TASK_IDS
    requested_set = set(requested)
    if not requested_set or not requested_set.issubset(EXPECTED_TASK_IDS):
        raise CanaryError("task_selection_invalid")
    return tuple(task_id for task_id in EXPECTED_TASK_IDS if task_id in requested_set)


def _dependency_packages(task: TaskSpec) -> tuple[str, ...]:
    if not task.setup_commands:
        raise CanaryError("dependency_setup_missing")
    command = task.setup_commands[0].argv
    if command[:4] != ("python", "-m", "pip", "install"):
        raise CanaryError("dependency_setup_invalid")
    packages = command[4:]
    if not packages or any(_PINNED_PACKAGE.fullmatch(package) is None for package in packages):
        raise CanaryError("dependency_setup_invalid")
    return packages


def _run_docker_build(*arguments: str) -> None:
    """Run a public, Gold-free adapter build with bounded failure diagnostics."""

    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as error:
        diagnostic: dict[str, object] = {
            "code": "adapter_image_build_timeout",
            "stdout_tail": (error.stdout or "")[-4_000:],
            "stderr_tail": (error.stderr or "")[-4_000:],
        }
        print(
            "adapter_build_failure=" + json.dumps(diagnostic, ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
        raise CanaryError("adapter_image_build_timeout") from None
    if completed.returncode == 0:
        return
    diagnostic = {
        "code": "adapter_image_build_failed",
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-4_000:],
        "stderr_tail": completed.stderr[-4_000:],
    }
    print(
        "adapter_build_failure=" + json.dumps(diagnostic, ensure_ascii=False, sort_keys=True),
        file=sys.stderr,
    )
    raise CanaryError("adapter_image_build_failed")


def _safe_name(task_id: str) -> str:
    safe = task_id.lower().replace("__", "-").replace("_", "-")
    if not safe or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for character in safe
    ):
        raise CanaryError("task_id_invalid")
    return safe


if __name__ == "__main__":
    raise SystemExit(main())
