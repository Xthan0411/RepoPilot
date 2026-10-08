"""Bind P7-S6 registry, checkout, and command-timeout evidence for formal execution.

The command consumes the committed 8/8 Sandbox preflight and immutable registry
manifest. It does not build or pull images, run task commands, load source rows or
evaluator Gold patches, invoke a model Provider, or contact Artifact Registry. It
preserves the frozen task and matrix while projecting each declared setup/test command
timeout through the already frozen 300-second Sandbox ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
from datetime import UTC, datetime
from enum import StrEnum, unique
from pathlib import Path

from pydantic import ValidationError

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repopilot.evals import load_experiment_matrix
from repopilot.evals.dataset import FrozenDatasetManifest
from repopilot.schemas import TaskSpec
from scripts.p7_holdout_control import (
    COMMAND_TIMEOUT_CAP_SECONDS,
    ENVIRONMENT_PATH,
    EXPECTED_TASK_IDS,
    MATRIX_PATH,
    TIMEOUT_CONTRACT_VERSION,
    HoldoutCommandTimeoutBinding,
    HoldoutControlError,
    HoldoutEnvironmentBinding,
    HoldoutEnvironmentManifest,
    HoldoutErrorCode,
    _canonical_sha256,
    build_holdout_timeout_bindings,
    holdout_command_contract_sha256,
    holdout_timeout_projection_sha256,
    project_holdout_task_timeouts,
    write_control_artifact,
)
from scripts.p7_publish_registry_images import (
    EXPECTED_PREFLIGHT_RAW_SHA256,
    RegistryManifest,
)
from scripts.p7_publish_registry_images import (
    RESULT_PATH as REGISTRY_MANIFEST_PATH,
)
from scripts.p7_sandbox_matrix_preflight import (
    DATASET_MANIFEST_PATH,
    SandboxMatrixPreflight,
    SandboxTaskPreflightEvidence,
)
from scripts.p7_sandbox_matrix_preflight import (
    RESULT_PATH as SANDBOX_PREFLIGHT_PATH,
)

EXPECTED_REGISTRY_MANIFEST_RAW_SHA256 = (
    "edc0b94b7ec8247e00e2f36c80d5f930e1dc7141602da9e6514f55338841fef9"
)


@unique
class EnvironmentBindingErrorCode(StrEnum):
    """Stable, secret-free failures before any paid or Agent execution."""

    INVALID_CONFIGURATION = "invalid_configuration"
    HOST_PLATFORM_MISMATCH = "host_platform_mismatch"
    CONTROLLER_SOURCE_DIRTY = "controller_source_dirty"
    SOURCE_ARTIFACT_INVALID = "source_artifact_invalid"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    CHECKOUT_UNAVAILABLE = "checkout_unavailable"
    CHECKOUT_IDENTITY_MISMATCH = "checkout_identity_mismatch"
    TIMEOUT_CONTRACT_MISMATCH = "timeout_contract_mismatch"
    OUTPUT_CONFLICT = "output_conflict"


class EnvironmentBindingError(RuntimeError):
    """One bounded environment-binding failure."""

    def __init__(self, code: EnvironmentBindingErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    try:
        root = arguments.repository_root.resolve(strict=True)
        _validate_controller(root)
        matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
        dataset_manifest = FrozenDatasetManifest.model_validate_json(
            (root / DATASET_MANIFEST_PATH).read_bytes()
        )
        preflight_bytes = (root / SANDBOX_PREFLIGHT_PATH).read_bytes()
        registry_bytes = (root / REGISTRY_MANIFEST_PATH).read_bytes()
        preflight = SandboxMatrixPreflight.model_validate_json(preflight_bytes)
        registry = RegistryManifest.model_validate_json(registry_bytes)
        manifest = build_environment_manifest(
            root=root,
            controller_commit=_git(root, "rev-parse", "HEAD"),
            experiment_id=matrix.experiment_id,
            matrix_sha256=matrix.content_sha256,
            matrix_dataset_manifest_sha256=matrix.dataset.manifest_content_sha256,
            matrix_sandbox_command_timeout_seconds=matrix.shared_settings.values.get(
                "sandbox_command_timeout_seconds"
            ),
            matrix_sandbox_network=matrix.shared_settings.values.get("sandbox_network"),
            dataset_manifest_sha256=dataset_manifest.content_sha256,
            dataset_manifest=dataset_manifest,
            preflight=preflight,
            preflight_bytes=preflight_bytes,
            registry=registry,
            registry_bytes=registry_bytes,
            created_at=datetime.now(UTC),
        )
        write_control_artifact(root / ENVIRONMENT_PATH, manifest)
    except EnvironmentBindingError as error:
        print(
            f"holdout_environment_binding_failed=true code={error.code.value} paid_requests=0",
            file=sys.stderr,
        )
        return 1
    except HoldoutControlError as error:
        code = (
            EnvironmentBindingErrorCode.OUTPUT_CONFLICT
            if error.code is HoldoutErrorCode.OUTPUT_CONFLICT
            else EnvironmentBindingErrorCode.SOURCE_ARTIFACT_INVALID
        )
        print(
            f"holdout_environment_binding_failed=true code={code.value} paid_requests=0",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValidationError, ValueError):
        print(
            "holdout_environment_binding_failed=true code=source_artifact_invalid paid_requests=0",
            file=sys.stderr,
        )
        return 1
    print(
        "holdout_environment_binding_passed=true tasks=8 registry_digests=8 "
        "timeout_cap_seconds=300 paid_requests=0 formal_environment_ready=true blockers=0"
    )
    return 0


def build_environment_manifest(
    *,
    root: Path,
    controller_commit: str,
    experiment_id: str,
    matrix_sha256: str,
    matrix_dataset_manifest_sha256: str,
    matrix_sandbox_command_timeout_seconds: object,
    matrix_sandbox_network: object,
    dataset_manifest_sha256: str,
    dataset_manifest: FrozenDatasetManifest,
    preflight: SandboxMatrixPreflight,
    preflight_bytes: bytes,
    registry: RegistryManifest,
    registry_bytes: bytes,
    created_at: datetime,
) -> HoldoutEnvironmentManifest:
    """Create the complete Gold-free formal environment binding after exact checks."""

    if (
        experiment_id != "repopilot-p7-v2"
        or matrix_dataset_manifest_sha256 != dataset_manifest_sha256
        or matrix_sandbox_command_timeout_seconds != COMMAND_TIMEOUT_CAP_SECONDS
        or matrix_sandbox_network != "none"
    ):
        raise EnvironmentBindingError(EnvironmentBindingErrorCode.SOURCE_IDENTITY_MISMATCH)
    _validate_sources(
        dataset_manifest=dataset_manifest,
        dataset_manifest_sha256=dataset_manifest_sha256,
        preflight=preflight,
        preflight_bytes=preflight_bytes,
        registry=registry,
        registry_bytes=registry_bytes,
    )
    cases = {item.case_id: item for item in dataset_manifest.external_cases}
    preflight_items = {item.task_id: item for item in preflight.task_evidence}
    registry_items = {item.task_id: item for item in registry.images}
    source_tasks: list[TaskSpec] = []
    runtime_tasks: list[TaskSpec] = []
    bindings: list[HoldoutEnvironmentBinding] = []
    for task_id in EXPECTED_TASK_IDS:
        case = cases[task_id]
        task = case.agent_task("Gold-free formal environment command binding")
        compatibility = preflight_items[task_id]
        image = registry_items[task_id]
        _verify_checkout(
            root,
            compatibility.checkout_relative_path,
            base_commit=case.base_commit,
        )
        setup_timeouts, test_timeouts = build_holdout_timeout_bindings(task)
        _verify_timeout_evidence(
            compatibility=compatibility,
            setup_timeouts=setup_timeouts,
            test_timeouts=test_timeouts,
        )
        if (
            compatibility.base_commit != case.base_commit
            or compatibility.declared_python_version != case.python_version
            or compatibility.adapted_local_image_id != image.local_image_id
            or compatibility.adapted_policy_reference != image.source_policy_reference
            or not image.remote_pull_verified
        ):
            raise EnvironmentBindingError(EnvironmentBindingErrorCode.SOURCE_IDENTITY_MISMATCH)
        runtime_task = project_holdout_task_timeouts(task)
        source_tasks.append(task)
        runtime_tasks.append(runtime_task)
        bindings.append(
            HoldoutEnvironmentBinding(
                task_id=task_id,
                base_commit=case.base_commit,
                python_version=case.python_version,
                checkout_relative_path=compatibility.checkout_relative_path,
                checkout_commit=compatibility.checkout_commit,
                sandbox_image_reference=image.registry_digest_reference,
                platform=compatibility.platform,
                task_timeout_seconds=task.timeout_seconds,
                setup_command_timeouts=setup_timeouts,
                test_command_timeouts=test_timeouts,
                timeout_projection_sha256=holdout_timeout_projection_sha256(task),
                setup_commands_passed_offline=True,
                baseline_failed=True,
                gold_passed=True,
            )
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "environment_version": "holdout-environments-v1",
        "experiment_id": experiment_id,
        "matrix_sha256": matrix_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "execution_harness": "swe-bench",
        "execution_harness_revision": preflight.harness_revision,
        "controller_commit": controller_commit,
        "created_at": created_at,
        "compatibility_preflight_raw_sha256": hashlib.sha256(preflight_bytes).hexdigest(),
        "compatibility_preflight_content_sha256": preflight.content_sha256,
        "registry_manifest_raw_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "registry_manifest_content_sha256": registry.content_sha256,
        "timeout_contract_version": TIMEOUT_CONTRACT_VERSION,
        "sandbox_command_timeout_seconds": COMMAND_TIMEOUT_CAP_SECONDS,
        "source_command_contract_sha256": holdout_command_contract_sha256(tuple(source_tasks)),
        "runtime_command_contract_sha256": holdout_command_contract_sha256(tuple(runtime_tasks)),
        "bindings": tuple(bindings),
        "paid_requests_made": 0,
        "gold_visible_to_agent": False,
        "formal_environment_ready": True,
        "formal_blockers": (),
    }
    return HoldoutEnvironmentManifest.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def _validate_controller(root: Path) -> None:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise EnvironmentBindingError(EnvironmentBindingErrorCode.HOST_PLATFORM_MISMATCH)
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise EnvironmentBindingError(EnvironmentBindingErrorCode.CONTROLLER_SOURCE_DIRTY)
    if (root / ENVIRONMENT_PATH).exists():
        raise EnvironmentBindingError(EnvironmentBindingErrorCode.OUTPUT_CONFLICT)


def _validate_sources(
    *,
    dataset_manifest: FrozenDatasetManifest,
    dataset_manifest_sha256: str,
    preflight: SandboxMatrixPreflight,
    preflight_bytes: bytes,
    registry: RegistryManifest,
    registry_bytes: bytes,
) -> None:
    if (
        dataset_manifest_sha256 != dataset_manifest.content_sha256
        or tuple(item.case_id for item in dataset_manifest.external_cases) != EXPECTED_TASK_IDS
        or tuple(item.task_id for item in preflight.task_evidence) != EXPECTED_TASK_IDS
        or tuple(item.task_id for item in registry.images) != EXPECTED_TASK_IDS
        or hashlib.sha256(preflight_bytes).hexdigest() != EXPECTED_PREFLIGHT_RAW_SHA256
        or hashlib.sha256(registry_bytes).hexdigest() != EXPECTED_REGISTRY_MANIFEST_RAW_SHA256
        or registry.source_preflight_raw_sha256 != EXPECTED_PREFLIGHT_RAW_SHA256
        or registry.source_preflight_content_sha256 != preflight.content_sha256
        or registry.formal_blockers != ("task_command_timeout_contract_mismatch",)
    ):
        raise EnvironmentBindingError(EnvironmentBindingErrorCode.SOURCE_ARTIFACT_INVALID)


def _verify_checkout(root: Path, relative_path: str, *, base_commit: str) -> Path:
    try:
        checkout = (root / relative_path).resolve(strict=True)
    except OSError:
        raise EnvironmentBindingError(EnvironmentBindingErrorCode.CHECKOUT_UNAVAILABLE) from None
    if not checkout.is_relative_to(root):
        raise EnvironmentBindingError(EnvironmentBindingErrorCode.CHECKOUT_IDENTITY_MISMATCH)
    try:
        valid = _git(checkout, "rev-parse", "HEAD") == base_commit and not _git(
            checkout, "status", "--porcelain=v1"
        )
    except (OSError, subprocess.CalledProcessError):
        raise EnvironmentBindingError(EnvironmentBindingErrorCode.CHECKOUT_UNAVAILABLE) from None
    if not valid:
        raise EnvironmentBindingError(EnvironmentBindingErrorCode.CHECKOUT_IDENTITY_MISMATCH)
    return checkout


def _verify_timeout_evidence(
    *,
    compatibility: SandboxTaskPreflightEvidence,
    setup_timeouts: tuple[HoldoutCommandTimeoutBinding, ...],
    test_timeouts: tuple[HoldoutCommandTimeoutBinding, ...],
) -> None:
    setup_commands = getattr(compatibility, "setup_commands", ())
    baseline_tests = getattr(compatibility, "baseline_tests", ())
    gold_tests = getattr(compatibility, "gold_tests", ())
    if (
        len(setup_commands) != len(setup_timeouts)
        or len(baseline_tests) != len(test_timeouts)
        or len(gold_tests) != len(test_timeouts)
    ):
        raise EnvironmentBindingError(EnvironmentBindingErrorCode.TIMEOUT_CONTRACT_MISMATCH)

    def matches(observed: object, expected: HoldoutCommandTimeoutBinding) -> bool:
        return (
            getattr(observed, "command_index", None) == expected.command_index
            and getattr(observed, "declared_timeout_seconds", None)
            == expected.declared_timeout_seconds
            and getattr(observed, "effective_timeout_seconds", None)
            == expected.effective_timeout_seconds
            and getattr(observed, "network_requested", None) is False
        )

    if any(
        not matches(observed, expected)
        for observed, expected in zip(setup_commands, setup_timeouts, strict=True)
    ) or any(
        not matches(observed, expected)
        for commands in (baseline_tests, gold_tests)
        for observed, expected in zip(commands, test_timeouts, strict=True)
    ):
        raise EnvironmentBindingError(EnvironmentBindingErrorCode.TIMEOUT_CONTRACT_MISMATCH)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
