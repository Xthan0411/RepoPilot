"""Qualify the reauthorized v2 Holdout without constructing an Agent or Provider.

Only the exact Jina cache preparation may use network access. Docker inspection is
local, and pgvector initialization is confined to the configured project database.
All failed gates are reported without credentials, subprocess output, or host paths.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from repopilot.config import DEFAULT_DATABASE_URL
from repopilot.evals import load_experiment_matrix, select_experiment_arm
from scripts import p7_holdout_readiness as legacy
from scripts.p7_holdout_control import (
    MATRIX_PATH,
    HoldoutControlError,
    HoldoutEnvironmentManifest,
    write_control_artifact,
)
from scripts.p7_holdout_requalification import _read
from scripts.p7_holdout_v2 import (
    READINESS_V2_PATH,
    V2Controls,
    build_readiness,
    load_controls,
    verify_execution_controller,
    verify_runtime_environment,
)
from scripts.p7_publish_registry_images import RegistryManifest

REGISTRY_PATH = Path("results/p7-holdout/registry-manifest-v1.json")
REQUIRED_DEPENDENCIES = ("openai", "psycopg", "sentence_transformers", "torch", "transformers")


def _safe_path(root: Path, relative: str | Path) -> Path:
    relative = Path(relative)
    path = root / relative
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe_runtime_path")
    if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root)):
        raise ValueError("unsafe_runtime_path")
    if not path.resolve().is_relative_to(root):
        raise ValueError("unsafe_runtime_path")
    return path


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()


def _checkouts(root: Path, environment: HoldoutEnvironmentManifest) -> tuple[str, ...]:
    """Reject incorrect repositories, dirty files, and all untracked/ignored additions."""

    failures: list[str] = []
    for binding in environment.bindings:
        try:
            checkout = _safe_path(root, binding.checkout_relative_path)
            if (
                not checkout.is_dir()
                or Path(_git(checkout, "rev-parse", "--show-toplevel")).resolve() != checkout
                or _git(checkout, "rev-parse", "HEAD") != binding.base_commit
                or _git(checkout, "status", "--porcelain=v1", "--untracked-files=all", "--ignored")
            ):
                raise ValueError("checkout_mismatch")
        except (OSError, ValueError, subprocess.SubprocessError):
            failures.append(binding.task_id)
    return tuple(failures)


def _resolve_images(
    root: Path, environment: HoldoutEnvironmentManifest
) -> tuple[tuple[legacy.ReadinessImageEvidence, ...], tuple[str, ...]]:
    """Inspect exact local images, including platform and published image/config identity."""

    raw = _read(root, REGISTRY_PATH)
    registry = RegistryManifest.model_validate_json(raw)
    if (
        hashlib.sha256(raw).hexdigest() != environment.registry_manifest_raw_sha256
        or registry.content_sha256 != environment.registry_manifest_content_sha256
    ):
        raise ValueError("registry_identity_mismatch")
    records = {item.task_id: item for item in registry.images}
    evidence: list[legacy.ReadinessImageEvidence] = []
    failures: list[str] = []
    for binding in environment.bindings:
        expected = records[binding.task_id]
        try:
            completed = subprocess.run(
                ("docker", "image", "inspect", binding.sandbox_image_reference),
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            payload = json.loads(completed.stdout)
            if not isinstance(payload, list) or len(payload) != 1:
                raise ValueError("invalid_image_inspection")
            item = payload[0]
            if (
                expected.registry_digest_reference != binding.sandbox_image_reference
                or expected.platform != binding.platform
                or binding.platform != "linux/amd64"
                or item["Id"] != expected.local_image_id
                or item["Os"] != "linux"
                or item["Architecture"] != "amd64"
                or binding.sandbox_image_reference not in (item.get("RepoDigests") or [])
            ):
                raise ValueError("image_identity_mismatch")
            evidence.append(
                legacy.ReadinessImageEvidence(
                    task_id=binding.task_id,
                    registry_digest_reference=binding.sandbox_image_reference,
                    local_image_id=item["Id"],
                )
            )
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            failures.append(binding.task_id)
    return tuple(evidence), tuple(failures)


def _output_conflicts(root: Path, paths: tuple[str, ...]) -> tuple[int, ...]:
    failures: list[int] = []
    for index, relative in enumerate(paths):
        try:
            path = _safe_path(root, relative)
            if path.exists() or any(
                p.exists() and not p.is_dir() for p in path.parents if p.is_relative_to(root)
            ):
                raise ValueError("output_conflict")
        except (OSError, ValueError):
            failures.append(index)
    return tuple(failures)


def _missing_dependencies() -> tuple[str, ...]:
    missing: list[str] = []
    for name in REQUIRED_DEPENDENCIES:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return tuple(missing)


def _blocked(codes: tuple[str, ...], **details: object) -> int:
    diagnostic = {
        "codes": sorted(codes),
        "paid_requests_made": 0,
        "provider_api_requests_made": 0,
        **details,
    }
    print(
        "holdout_runtime_readiness_v2_passed=false diagnostic="
        + json.dumps(diagnostic, separators=(",", ":"), sort_keys=True)
    )
    return 1


def _local_gates(
    root: Path, controls: V2Controls
) -> tuple[tuple[legacy.ReadinessImageEvidence, ...], dict[str, object]]:
    images, missing_images = _resolve_images(root, controls.environment)
    checkout_failures = _checkouts(root, controls.environment)
    conflicts = _output_conflicts(
        root,
        (str(READINESS_V2_PATH), *(slot.output_path for slot in controls.plan.run_slots)),
    )
    return images, {
        "missing_image_task_ids": missing_images,
        "invalid_checkout_task_ids": checkout_failures,
        "conflicting_output_indices": conflicts,
    }


def qualify(root: Path, *, prepare_embedding_cache: bool = False) -> int:
    """Run all zero-paid gates and create one private readiness artifact on success."""

    verify_runtime_environment()
    controls = load_controls(root)
    controller_objects = verify_execution_controller(root)
    images, diagnostics = _local_gates(root, controls)
    missing_dependencies = _missing_dependencies()
    provider_available = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    diagnostics.update(
        missing_local_dependencies=missing_dependencies,
        provider_credential_available=provider_available,
        embedding_cache_preparation_requested=prepare_embedding_cache,
    )
    gates = {
        "sandbox_image_unavailable": diagnostics["missing_image_task_ids"],
        "checkout_unavailable": diagnostics["invalid_checkout_task_ids"],
        "output_conflict": diagnostics["conflicting_output_indices"],
        "local_dependency_unavailable": missing_dependencies,
        "provider_credential_required": not provider_available,
    }
    if codes := tuple(code for code, failed in gates.items() if failed):
        return _blocked(codes, **diagnostics)
    try:
        matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
        settings = select_experiment_arm(matrix, "dense-only-v1").settings(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            database_url=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
        )
    except (ValueError, OSError):
        return _blocked(("invalid_configuration",))

    embedding: legacy.EmbeddingReadinessEvidence | None = None
    vector: legacy.VectorReadinessEvidence | None = None
    embedding_stage: str | None = None
    if prepare_embedding_cache:
        try:
            legacy._probe_embedding(root, settings, offline=False)
        except legacy.ReadinessError:
            embedding_stage = "cache_preparation"
    if embedding_stage is None:
        try:
            embedding = legacy._probe_embedding(root, settings, offline=True).model_copy(
                update={"cache_preparation_network_used": prepare_embedding_cache}
            )
        except legacy.ReadinessError:
            embedding_stage = "final_offline_probe"
    with suppress(legacy.ReadinessError):
        vector = legacy._probe_vector_backend(settings)
    codes = tuple(
        code
        for code, failed in (
            ("embedding_snapshot_unavailable", embedding is None),
            ("vector_backend_unavailable", vector is None),
        )
        if failed
    )
    if embedding is None or vector is None:
        return _blocked(codes, embedding_failure_stage=embedding_stage)

    # The cache probe can take minutes: do not certify a source or asset changed meanwhile.
    current = load_controls(root)
    current_objects = verify_execution_controller(root)
    if current != controls or current_objects != controller_objects:
        return _blocked(("runtime_changed_during_readiness",))
    final_images, final_diagnostics = _local_gates(root, current)
    if final_images != images or any(final_diagnostics.values()):
        return _blocked(("runtime_changed_during_readiness",), **final_diagnostics)
    readiness = build_readiness(
        controls=controls,
        checked_at=datetime.now(UTC),
        execution_controller_objects=controller_objects,
        embedding=embedding,
        vector_backend=vector,
        images=images,
    )
    try:
        write_control_artifact(_safe_path(root, READINESS_V2_PATH), readiness)
    except (OSError, ValueError, HoldoutControlError):
        return _blocked(("output_conflict",))
    print(
        "holdout_runtime_readiness_v2_passed=true tasks=8 images=8 "
        "embedding_dimension=768 vector_dimension=768 provider_requests=0 "
        "paid_requests=0 ready_for_paid_holdout=true blockers=0"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--prepare-embedding-cache", action="store_true")
    arguments = parser.parse_args()
    try:
        return qualify(
            arguments.repository_root.resolve(strict=True),
            prepare_embedding_cache=arguments.prepare_embedding_cache,
        )
    except legacy.ReadinessError as error:
        return _blocked((error.code.value,))
    except (OSError, ValueError, subprocess.SubprocessError):
        return _blocked(("control_identity_or_authorization_invalid",))


if __name__ == "__main__":
    raise SystemExit(main())
