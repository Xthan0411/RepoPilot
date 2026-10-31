"""Fresh zero-paid readiness for the frozen GPU-cache execution wrapper.

Validate the already qualified cache once, then run one short offline CPU probe
in a disposable interpreter. This does not rebuild corpora, repeat the numeric
comparison, run retrieval, construct an Agent, or send a Provider request.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from contextlib import redirect_stdout, suppress
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from repopilot.config import DEFAULT_DATABASE_URL, RepoPilotSettings
from repopilot.embeddings import (
    EmbeddingInput,
    EmbeddingRequest,
    embedding_identity_from_settings,
    embedding_request_id,
)
from repopilot.evals import load_experiment_matrix, select_experiment_arm
from repopilot.schemas import StrictModel
from scripts import p7_holdout_readiness as legacy
from scripts import p7_holdout_readiness_v2 as v2
from scripts.p7_gpu_cache_acceptance import (
    CacheAcceptanceError,
    _pinned_documents,
    verify_pinned_cache,
)
from scripts.p7_gpu_cache_store import CacheStoreError
from scripts.p7_gpu_cpu_probe import (
    PINNED_IDENTITY,
    CpuProbeError,
    _vector_norm,
    create_cpu_provider,
    no_network,
)
from scripts.p7_gpu_execution_profile import verify_cpu_runtime
from scripts.p7_holdout_control import MATRIX_PATH, HoldoutControlError, write_control_artifact
from scripts.p7_holdout_gpu import (
    READINESS_V3_PATH,
    V3Controls,
    build_readiness,
    load_controls,
    verify_execution_controller,
    verify_runtime_environment,
)

CACHE_PATH = Path("artifacts/p7-gpu-cache-v1")
CPU_PROBE_TIMEOUT_SECONDS = 900
MAX_PROBE_OUTPUT_BYTES = 64 * 1024
_ENVIRONMENT_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")


class CpuReadinessObservation(StrictModel):
    embedding: legacy.EmbeddingReadinessEvidence
    cpu_profile: dict[str, JsonValue]


def _blocked(codes: tuple[str, ...], **details: object) -> int:
    diagnostic = {
        "codes": sorted(codes),
        "paid_requests_made": 0,
        "provider_api_requests_made": 0,
        **details,
    }
    print(
        "holdout_runtime_readiness_v3_passed=false diagnostic="
        + json.dumps(diagnostic, separators=(",", ":"), sort_keys=True)
    )
    return 1


def _worker_environment() -> dict[str, str]:
    """Only OS necessities reach the worker, never credentials or model overrides."""
    environment = {key: os.environ[key] for key in _ENVIRONMENT_ALLOWLIST if key in os.environ}
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONWARNINGS="ignore",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        CUDA_VISIBLE_DEVICES="",
    )
    return environment


def _cpu_probe() -> CpuReadinessObservation:
    """Load CPU/FP32/batch1/threads8 and embed one fixed short probe offline."""
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or any(
            key in {"OPENAI_API_KEY", "DATABASE_URL"} or key.startswith("REPOPILOT_")
            for key in os.environ
        )
    ):
        raise CpuProbeError("cpu_probe_environment_invalid")
    with no_network():
        provider, profile = create_cpu_provider(PINNED_IDENTITY)
        if provider.identity != PINNED_IDENTITY:
            raise CpuProbeError("embedding_identity_mismatch")
        item = EmbeddingInput(
            content_hash=legacy.PROBE_CONTENT_SHA256, content=legacy.PROBE_CONTENT
        )
        request = EmbeddingRequest(
            request_id=embedding_request_id(PINNED_IDENTITY, (item,), 1),
            identity=PINNED_IDENTITY,
            inputs=(item,),
            attempt=1,
        )
        response = provider.embed(request)
        if (
            response.identity != PINNED_IDENTITY
            or response.request_id != request.request_id
            or len(response.vectors) != 1
            or response.vectors[0].content_hash != item.content_hash
        ):
            raise CpuProbeError("embedding_response_mismatch")
        _vector_norm(response.vectors[0].values, PINNED_IDENTITY.dimension)
    return CpuReadinessObservation(
        embedding=legacy.EmbeddingReadinessEvidence(
            model=PINNED_IDENTITY.model,
            revision=PINNED_IDENTITY.revision,
            code_revision=PINNED_IDENTITY.code_revision,
            identity_sha256=PINNED_IDENTITY.fingerprint,
            cache_preparation_network_used=False,
        ),
        cpu_profile=profile,
    )


def _probe_cpu(root: Path, settings: RepoPilotSettings) -> CpuReadinessObservation:
    """The worker exits before readiness is written, releasing its model memory."""
    try:
        with subprocess.Popen(
            (
                sys.executable,
                "-m",
                "scripts.p7_holdout_readiness_v3",
                "--repository-root",
                str(root),
                "--cpu-probe",
            ),
            cwd=root,
            env=_worker_environment(),
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as child:
            try:
                output, _ = child.communicate(timeout=CPU_PROBE_TIMEOUT_SECONDS)
            except BaseException:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.communicate()
                raise
            if child.returncode != 0 or len(output) > MAX_PROBE_OUTPUT_BYTES:
                raise ValueError("cpu_probe_failed")
        observation = CpuReadinessObservation.model_validate_json(output)
        identity = embedding_identity_from_settings(settings)
        embedding = observation.embedding
        if (
            identity != PINNED_IDENTITY
            or embedding.identity_sha256 != identity.fingerprint
            or embedding.model != identity.model
            or embedding.revision != identity.revision
            or embedding.code_revision != identity.code_revision
            or embedding.cache_preparation_network_used
        ):
            raise ValueError("cpu_probe_identity_mismatch")
        verify_cpu_runtime(root, observation.cpu_profile)
        return observation
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        raise legacy.ReadinessError(
            legacy.ReadinessErrorCode.EMBEDDING_SNAPSHOT_UNAVAILABLE
        ) from None


def _local_gates(
    root: Path, controls: V3Controls
) -> tuple[tuple[legacy.ReadinessImageEvidence, ...], dict[str, object]]:
    images, missing_images = v2._resolve_images(root, controls.environment)
    checkout_failures = v2._checkouts(root, controls.environment)
    conflicts = v2._output_conflicts(
        root,
        (str(READINESS_V3_PATH), *(slot.output_path for slot in controls.plan.run_slots)),
    )
    return images, {
        "missing_image_task_ids": missing_images,
        "invalid_checkout_task_ids": checkout_failures,
        "conflicting_output_indices": conflicts,
    }


def _cache_metadata_unchanged(root: Path, report: dict[str, JsonValue]) -> bool:
    documents = _pinned_documents(v2._safe_path(root, CACHE_PATH))
    return tuple(document["sha256"] for document in documents) == (
        report.get("job_sha256"),
        report.get("inputs_sha256"),
        report.get("complete_sha256"),
    )


def qualify(root: Path) -> int:
    """An exact authorization is required before any potentially slow local probe."""
    verify_runtime_environment()
    controls = load_controls(root)
    controller_objects = verify_execution_controller(root)
    images, diagnostics = _local_gates(root, controls)
    missing_dependencies = v2._missing_dependencies()
    provider_available = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    diagnostics.update(
        missing_local_dependencies=missing_dependencies,
        provider_credential_available=provider_available,
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
            openai_api_key="readiness-provider-not-constructed",
            database_url=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
        )
    except (ValueError, OSError):
        return _blocked(("invalid_configuration",))

    try:
        accepted = verify_pinned_cache(
            v2._safe_path(root, CACHE_PATH), embedding_identity_from_settings(settings)
        )
    except (CacheAcceptanceError, CacheStoreError, OSError, ValueError):
        return _blocked(("gpu_cache_unavailable",))

    observation: CpuReadinessObservation | None = None
    vector: legacy.VectorReadinessEvidence | None = None
    with suppress(legacy.ReadinessError):
        observation = _probe_cpu(root, settings)
    with suppress(legacy.ReadinessError):
        vector = legacy._probe_vector_backend(settings)
    if observation is None or vector is None:
        return _blocked(
            tuple(
                code
                for code, failed in (
                    ("cpu_embedding_profile_unavailable", observation is None),
                    ("vector_backend_unavailable", vector is None),
                )
                if failed
            )
        )

    current = load_controls(root)
    current_objects = verify_execution_controller(root)
    if current != controls or current_objects != controller_objects:
        return _blocked(("runtime_changed_during_readiness",))
    final_images, final_diagnostics = _local_gates(root, current)
    if final_images != images or any(final_diagnostics.values()):
        return _blocked(("runtime_changed_during_readiness",), **final_diagnostics)
    try:
        if not _cache_metadata_unchanged(root, accepted.report):
            return _blocked(("gpu_cache_changed_during_readiness",))
    except (CacheAcceptanceError, CacheStoreError, OSError, ValueError):
        return _blocked(("gpu_cache_changed_during_readiness",))
    readiness = build_readiness(
        controls=controls,
        checked_at=datetime.now(UTC),
        execution_controller_objects=controller_objects,
        embedding=observation.embedding,
        vector_backend=vector,
        images=images,
        cpu_profile=observation.cpu_profile,
        gpu_cache=accepted.report,
    )
    try:
        write_control_artifact(v2._safe_path(root, READINESS_V3_PATH), readiness)
    except (OSError, ValueError, HoldoutControlError):
        return _blocked(("output_conflict",))
    print(
        "holdout_runtime_readiness_v3_passed=true tasks=8 images=8 "
        "embedding_dimension=768 vector_dimension=768 gpu_cache_verified=true "
        "cpu_probe_count=1 provider_requests=0 paid_requests=0 "
        "ready_for_paid_holdout=true blockers=0"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--cpu-probe", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.cpu_probe:
        try:
            # Optional libraries may emit informational stdout; never mix it with evidence.
            with redirect_stdout(sys.stderr):
                observation = _cpu_probe()
            print(observation.model_dump_json())
            return 0
        except Exception:
            print('{"cpu_probe_passed":false,"code":"cpu_probe_failed"}')
            return 1
    try:
        return qualify(arguments.repository_root.resolve(strict=True))
    except legacy.ReadinessError as error:
        return _blocked((error.code.value,))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        return _blocked(("control_identity_or_authorization_invalid",))


if __name__ == "__main__":
    raise SystemExit(main())
