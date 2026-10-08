"""Qualify P7-S6 runtime dependencies without making an OpenAI request.

The command consumes the frozen plan, explicit USD 32 authorization, and the accepted
x86 environment artifact. It inspects already-local task images, probes the pinned Jina
snapshot, initializes the isolated pgvector schema, and writes one private create-only
readiness artifact only after every formal preflight check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from enum import StrEnum, unique
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, ValidationError, model_validator
from pydantic_core import to_jsonable_python

from repopilot.config import DEFAULT_DATABASE_URL, RepoPilotSettings
from repopilot.embeddings import embedding_identity_from_settings
from repopilot.evals import (
    ExperimentMatrix,
    FormalFreezeMarker,
    load_experiment_matrix,
    select_experiment_arm,
)
from repopilot.evals.dataset import DatasetPreflightReport
from repopilot.sandbox import DockerCliClient, DockerClientError
from repopilot.schemas import GitSha, Identifier, Sha256, StrictModel, TaskId
from repopilot.storage import PostgresVectorStore, VectorStoreError
from scripts.p7_holdout_control import (
    DATASET_PREFLIGHT_PATH,
    ENVIRONMENT_PATH,
    EXPECTED_TASK_IDS,
    FREEZE_PATH,
    MATRIX_PATH,
    PLAN_PATH,
    HoldoutApplicationPlan,
    HoldoutEnvironmentManifest,
    HoldoutPreflight,
    PaidHoldoutAuthorization,
    build_holdout_preflight,
    write_control_artifact,
)

AUTHORIZATION_PATH = Path("results/p7-holdout/authorization-v1.json")
READINESS_PATH = Path("results/p7-holdout/runtime-readiness-v1.json")
READINESS_VERSION: Literal["holdout-runtime-readiness-v1"] = "holdout-runtime-readiness-v1"
ALLOWED_SECRET_ENVIRONMENT = ("DATABASE_URL", "OPENAI_API_KEY")
PROBE_CONTENT = "def formal_holdout_embedding_probe() -> int:\n    return 1\n"
PROBE_CONTENT_SHA256: Literal[
    "611508dd77d83428b7522372c8828b4916839a9b3eac325c3175b6acc0770575"
] = "611508dd77d83428b7522372c8828b4916839a9b3eac325c3175b6acc0770575"
if hashlib.sha256(PROBE_CONTENT.encode()).hexdigest() != PROBE_CONTENT_SHA256:
    raise RuntimeError("embedding readiness probe identity changed")
EMBEDDING_PROBE_TIMEOUT_SECONDS = 900
EMBEDDING_CHILD_ENVIRONMENT_ALLOWLIST = (
    "HOME",
    "HUGGINGFACE_HUB_CACHE",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "TMPDIR",
    "TORCH_HOME",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
)
FROZEN_RUNTIME_PATHS = (
    "src/repopilot",
    "pyproject.toml",
    "uv.lock",
    "configs",
    "data/memory",
    "tests/fixtures/p5_repair_tasks",
)
AUDITED_COMPATIBILITY_PATH: Literal["src/repopilot/sandbox/docker.py"] = (
    "src/repopilot/sandbox/docker.py"
)
AUDITED_COMPATIBILITY_BASE_BLOB: Literal["ac0f4c4050350555644fb81e354a658d0d1de73f"] = (
    "ac0f4c4050350555644fb81e354a658d0d1de73f"
)
AUDITED_COMPATIBILITY_RESULT_BLOB: Literal["67c326a0ec8408c5dcd3661d099ec730e8986721"] = (
    "67c326a0ec8408c5dcd3661d099ec730e8986721"
)
AUDITED_COMPATIBILITY_PATCH_SHA256: Literal[
    "eab26db5938e04385aa770b203bf736ae142171d6979b3395ca19c363fa0faf1"
] = "eab26db5938e04385aa770b203bf736ae142171d6979b3395ca19c363fa0faf1"


@unique
class ReadinessErrorCode(StrEnum):
    """Stable secret-free reasons that prevent runtime readiness."""

    INVALID_CONFIGURATION = "invalid_configuration"
    HOST_PLATFORM_MISMATCH = "host_platform_mismatch"
    CONTROLLER_SOURCE_DIRTY = "controller_source_dirty"
    SOURCE_ARTIFACT_INVALID = "source_artifact_invalid"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    AUTHORIZATION_MISMATCH = "authorization_mismatch"
    UNAPPROVED_ENVIRONMENT_OVERRIDE = "unapproved_environment_override"
    LOCAL_DEPENDENCY_UNAVAILABLE = "local_dependency_unavailable"
    PROVIDER_CREDENTIAL_REQUIRED = "provider_credential_required"
    SANDBOX_IMAGE_UNAVAILABLE = "sandbox_image_unavailable"
    EMBEDDING_SNAPSHOT_UNAVAILABLE = "embedding_snapshot_unavailable"
    VECTOR_BACKEND_UNAVAILABLE = "vector_backend_unavailable"
    READINESS_BLOCKED = "readiness_blocked"
    OUTPUT_CONFLICT = "output_conflict"


class ReadinessError(RuntimeError):
    """Typed readiness failure that never carries a credential or DSN."""

    def __init__(
        self,
        code: ReadinessErrorCode,
        *,
        diagnostic: dict[str, object] | None = None,
    ) -> None:
        message = code.value
        if diagnostic is not None:
            message += " diagnostic=" + json.dumps(
                diagnostic,
                separators=(",", ":"),
                sort_keys=True,
            )
        super().__init__(message)
        self.code = code
        self.diagnostic = diagnostic


class ProviderCredentialEvidence(StrictModel):
    """Presence-only credential evidence; validity is learned only by a paid run."""

    validation: Literal["non_blank_presence_only"] = "non_blank_presence_only"
    available: Literal[True] = True
    api_requests_made: Literal[0] = 0
    secret_persisted: Literal[False] = False


class EmbeddingReadinessEvidence(StrictModel):
    """Pinned Jina identity and one bounded final offline vector observation."""

    provider: Literal["jina_local"] = "jina_local"
    model: str
    revision: GitSha
    code_revision: GitSha
    identity_sha256: Sha256
    normalized: Literal[True] = True
    maximum_length: Literal[2048] = 2048
    declared_dimension: Literal[768] = 768
    probe_content_sha256: Literal[
        "611508dd77d83428b7522372c8828b4916839a9b3eac325c3175b6acc0770575"
    ] = PROBE_CONTENT_SHA256
    observed_vector_count: Literal[1] = 1
    observed_dimension: Literal[768] = 768
    final_probe_network: Literal["offline"] = "offline"
    cache_preparation_network_used: bool


class EmbeddingProbeObservation(StrictModel):
    """Secret-free result emitted by one fresh embedding-probe interpreter."""

    model: str
    revision: GitSha
    code_revision: GitSha
    identity_sha256: Sha256
    normalized: Literal[True] = True
    maximum_length: Literal[2048] = 2048
    declared_dimension: Literal[768] = 768
    probe_content_sha256: Literal[
        "611508dd77d83428b7522372c8828b4916839a9b3eac325c3175b6acc0770575"
    ] = PROBE_CONTENT_SHA256
    observed_vector_count: Literal[1] = 1
    observed_dimension: Literal[768] = 768


class VectorReadinessEvidence(StrictModel):
    """Secret-free pgvector schema identity returned by initialization."""

    backend: Literal["pgvector"] = "pgvector"
    schema_name: Identifier
    table_name: Identifier
    extension_version: str
    dimension: Literal[768] = 768
    database_url_persisted: Literal[False] = False


class ReadinessImageEvidence(StrictModel):
    """One already-local immutable task image resolved without pulling."""

    task_id: TaskId
    registry_digest_reference: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/:~-]{0,439}@sha256:[0-9a-f]{64}$"
    )
    local_image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class FrozenRuntimeEvidence(StrictModel):
    """Exact post-freeze environment compatibility change accepted by readiness."""

    validation: Literal["frozen_source_plus_audited_compatibility_patch"] = (
        "frozen_source_plus_audited_compatibility_patch"
    )
    frozen_source_commit: GitSha
    environment_controller_commit: GitSha
    changed_paths: tuple[Literal["src/repopilot/sandbox/docker.py"], ...]
    base_blob_git_sha1: Literal["ac0f4c4050350555644fb81e354a658d0d1de73f"]
    result_blob_git_sha1: Literal["67c326a0ec8408c5dcd3661d099ec730e8986721"]
    patch_sha256: Literal["eab26db5938e04385aa770b203bf736ae142171d6979b3395ca19c363fa0faf1"]
    compatibility_scope: Literal["sandbox_workspace_archive_metadata_only"] = (
        "sandbox_workspace_archive_metadata_only"
    )
    changes_model_prompt_retrieval_task_or_scoring: Literal[False] = False


class HoldoutRuntimeReadiness(StrictModel):
    """All zero-paid gates needed immediately before formal Holdout execution."""

    schema_version: Literal[1] = 1
    readiness_version: Literal["holdout-runtime-readiness-v1"] = READINESS_VERSION
    experiment_id: Identifier
    controller_commit: GitSha
    checked_at: AwareDatetime
    host_platform: Literal["linux/amd64"] = "linux/amd64"
    plan_raw_sha256: Sha256
    plan_content_sha256: Sha256
    authorization_raw_sha256: Sha256
    authorization_content_sha256: Sha256
    environment_raw_sha256: Sha256
    environment_content_sha256: Sha256
    frozen_runtime: FrozenRuntimeEvidence
    allowed_secret_environment: tuple[str, ...]
    provider: ProviderCredentialEvidence
    embedding: EmbeddingReadinessEvidence
    vector_backend: VectorReadinessEvidence
    images: tuple[ReadinessImageEvidence, ...]
    preflight: HoldoutPreflight
    paid_requests_made: Literal[0] = 0
    gold_visible_to_agent: Literal[False] = False
    ready_for_paid_holdout: Literal[True] = True
    formal_blockers: tuple[()] = ()
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_readiness(self) -> Self:
        if self.allowed_secret_environment != ALLOWED_SECRET_ENVIRONMENT:
            raise ValueError("readiness secret environment changed")
        if tuple(item.task_id for item in self.images) != EXPECTED_TASK_IDS:
            raise ValueError("readiness images do not cover the exact task set")
        if not self.preflight.ready_for_paid_holdout:
            raise ValueError("readiness embeds a blocked preflight")
        if self.preflight.controller_commit != self.controller_commit:
            raise ValueError("readiness controller identity does not match preflight")
        _validate_content_digest(self)
        return self


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--prepare-embedding-cache",
        action="store_true",
        help=(
            "allow one exact-revision Hugging Face cache preparation before the mandatory "
            "offline embedding probe"
        ),
    )
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve(strict=True)
    _verify_host_and_environment()
    controller_commit = _git(root, "rev-parse", "HEAD")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ReadinessError(ReadinessErrorCode.CONTROLLER_SOURCE_DIRTY)
    if (root / READINESS_PATH).exists():
        raise ReadinessError(ReadinessErrorCode.OUTPUT_CONFLICT)

    artifacts = _load_control_artifacts(root)
    matrix, marker, dataset_preflight, plan, authorization, environment = artifacts.models
    _verify_source_identities(plan, authorization, environment, marker)
    frozen_runtime = _verify_frozen_runtime(
        root=root,
        source_commit=marker.source_commit,
        environment_commit=environment.controller_commit,
        controller_commit=controller_commit,
    )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    provider_credential_available = bool(api_key.strip())
    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    try:
        settings = select_experiment_arm(matrix, "dense-only-v1").settings(
            openai_api_key=(api_key if provider_credential_available else "readiness-placeholder"),
            database_url=database_url,
        )
    except ValidationError:
        raise ReadinessError(ReadinessErrorCode.INVALID_CONFIGURATION) from None
    local_dependencies_available = all(
        importlib.util.find_spec(name) is not None
        for name in ("openai", "psycopg", "sentence_transformers")
    )

    images, missing_image_task_ids = _resolve_images(environment)
    cache_prepared = bool(arguments.prepare_embedding_cache)
    embedding: EmbeddingReadinessEvidence | None = None
    embedding_available = False
    embedding_failure_stage: str | None = None
    if local_dependencies_available:
        if cache_prepared:
            try:
                _probe_embedding(root, settings, offline=False)
            except ReadinessError:
                embedding_failure_stage = "cache_preparation"
        if embedding_failure_stage is None:
            try:
                embedding = _probe_embedding(root, settings, offline=True).model_copy(
                    update={"cache_preparation_network_used": cache_prepared}
                )
                embedding_available = True
            except ReadinessError:
                embedding_failure_stage = "final_offline_probe"
    else:
        embedding_failure_stage = "local_dependencies"
    vector_backend: VectorReadinessEvidence | None = None
    vector_backend_available = False
    try:
        vector_backend = _probe_vector_backend(settings)
        vector_backend_available = True
    except ReadinessError:
        pass

    blockers: set[ReadinessErrorCode] = set()
    if not provider_credential_available:
        blockers.add(ReadinessErrorCode.PROVIDER_CREDENTIAL_REQUIRED)
    if not local_dependencies_available:
        blockers.add(ReadinessErrorCode.LOCAL_DEPENDENCY_UNAVAILABLE)
    if missing_image_task_ids:
        blockers.add(ReadinessErrorCode.SANDBOX_IMAGE_UNAVAILABLE)
    if not embedding_available:
        blockers.add(ReadinessErrorCode.EMBEDDING_SNAPSHOT_UNAVAILABLE)
    if not vector_backend_available:
        blockers.add(ReadinessErrorCode.VECTOR_BACKEND_UNAVAILABLE)
    if blockers:
        diagnostic = {
            "codes": sorted(item.value for item in blockers),
            "embedding_cache_preparation_requested": cache_prepared,
            "embedding_failure_stage": embedding_failure_stage,
            "embedding_snapshot_available": embedding_available,
            "local_dependencies_available": local_dependencies_available,
            "missing_image_task_ids": missing_image_task_ids,
            "paid_requests_made": 0,
            "provider_api_requests_made": 0,
            "provider_credential_available": provider_credential_available,
            "vector_backend_available": vector_backend_available,
        }
        print(
            "holdout_runtime_readiness_passed=false diagnostic="
            + json.dumps(diagnostic, separators=(",", ":"), sort_keys=True)
        )
        return 1
    if embedding is None or vector_backend is None:
        raise ReadinessError(ReadinessErrorCode.READINESS_BLOCKED)
    preflight = build_holdout_preflight(
        root=root,
        plan=plan,
        marker=marker,
        dataset_preflight=dataset_preflight,
        checked_at=datetime.now(UTC),
        controller_source_clean=True,
        frozen_runtime_matches=True,
        authorization=authorization,
        environment_manifest=environment,
        provider_credential_available=provider_credential_available,
        vector_backend_available=vector_backend_available,
        embedding_snapshot_available=embedding_available,
        controller_commit=controller_commit,
    )
    if not preflight.ready_for_paid_holdout:
        raise ReadinessError(ReadinessErrorCode.READINESS_BLOCKED)

    readiness = build_readiness_manifest(
        experiment_id=plan.experiment_id,
        controller_commit=controller_commit,
        checked_at=preflight.checked_at,
        artifacts=artifacts,
        frozen_runtime=frozen_runtime,
        embedding=embedding,
        vector_backend=vector_backend,
        images=images,
        preflight=preflight,
    )
    try:
        write_control_artifact(root / READINESS_PATH, readiness)
    except Exception:
        raise ReadinessError(ReadinessErrorCode.OUTPUT_CONFLICT) from None
    print(
        "holdout_runtime_readiness_passed=true tasks=8 images=8 "
        "embedding_dimension=768 vector_dimension=768 provider_requests=0 "
        "paid_requests=0 ready_for_paid_holdout=true blockers=0"
    )
    return 0


class _ControlArtifacts:
    """Raw bytes and typed control inputs kept together for exact binding."""

    def __init__(
        self,
        *,
        plan_raw: bytes,
        authorization_raw: bytes,
        environment_raw: bytes,
        models: tuple[
            ExperimentMatrix,
            FormalFreezeMarker,
            DatasetPreflightReport,
            HoldoutApplicationPlan,
            PaidHoldoutAuthorization,
            HoldoutEnvironmentManifest,
        ],
    ) -> None:
        self.plan_raw = plan_raw
        self.authorization_raw = authorization_raw
        self.environment_raw = environment_raw
        self.models = models


def _load_control_artifacts(root: Path) -> _ControlArtifacts:
    try:
        plan_raw = (root / PLAN_PATH).read_bytes()
        authorization_raw = (root / AUTHORIZATION_PATH).read_bytes()
        environment_raw = (root / ENVIRONMENT_PATH).read_bytes()
        matrix = load_experiment_matrix(root / MATRIX_PATH, repository_root=root)
        marker = FormalFreezeMarker.model_validate_json((root / FREEZE_PATH).read_bytes())
        dataset_preflight = DatasetPreflightReport.model_validate_json(
            (root / DATASET_PREFLIGHT_PATH).read_bytes()
        )
        plan = HoldoutApplicationPlan.model_validate_json(plan_raw)
        authorization = PaidHoldoutAuthorization.model_validate_json(authorization_raw)
        environment = HoldoutEnvironmentManifest.model_validate_json(environment_raw)
    except (OSError, ValidationError, ValueError):
        raise ReadinessError(ReadinessErrorCode.SOURCE_ARTIFACT_INVALID) from None
    return _ControlArtifacts(
        plan_raw=plan_raw,
        authorization_raw=authorization_raw,
        environment_raw=environment_raw,
        models=(matrix, marker, dataset_preflight, plan, authorization, environment),
    )


def _verify_host_and_environment() -> None:
    machine = platform.machine().lower()
    if platform.system().lower() != "linux" or machine not in {"amd64", "x86_64"}:
        raise ReadinessError(ReadinessErrorCode.HOST_PLATFORM_MISMATCH)
    if any(name.startswith("REPOPILOT_") for name in os.environ):
        raise ReadinessError(ReadinessErrorCode.UNAPPROVED_ENVIRONMENT_OVERRIDE)


def _verify_source_identities(
    plan: HoldoutApplicationPlan,
    authorization: PaidHoldoutAuthorization,
    environment: HoldoutEnvironmentManifest,
    marker: FormalFreezeMarker,
) -> None:
    if (
        authorization.experiment_id != plan.experiment_id
        or authorization.matrix_sha256 != plan.matrix_sha256
        or authorization.freeze_marker_sha256 != plan.freeze_marker_sha256
        or authorization.plan_sha256 != plan.content_sha256
        or not authorization.paid_execution_approved
        or not authorization.final_in_flight_response_overshoot_acknowledged
    ):
        raise ReadinessError(ReadinessErrorCode.AUTHORIZATION_MISMATCH)
    if (
        marker.content_sha256 != plan.freeze_marker_sha256
        or environment.experiment_id != plan.experiment_id
        or environment.matrix_sha256 != plan.matrix_sha256
        or environment.dataset_manifest_sha256 != plan.dataset_manifest_sha256
        or tuple(item.task_id for item in environment.bindings) != plan.task_ids
    ):
        raise ReadinessError(ReadinessErrorCode.SOURCE_IDENTITY_MISMATCH)


def _resolve_images(
    environment: HoldoutEnvironmentManifest,
) -> tuple[tuple[ReadinessImageEvidence, ...], tuple[TaskId, ...]]:
    client = DockerCliClient()
    evidence: list[ReadinessImageEvidence] = []
    missing: list[TaskId] = []
    for binding in environment.bindings:
        try:
            identity = client.resolve_image(binding.sandbox_image_reference)
            evidence.append(
                ReadinessImageEvidence(
                    task_id=binding.task_id,
                    registry_digest_reference=binding.sandbox_image_reference,
                    local_image_id=identity.image_id,
                )
            )
        except DockerClientError:
            missing.append(binding.task_id)
    return tuple(evidence), tuple(missing)


def _probe_embedding(
    root: Path,
    settings: RepoPilotSettings,
    *,
    offline: bool,
) -> EmbeddingReadinessEvidence:
    try:
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                "scripts.p7_embedding_readiness_probe",
                "--repository-root",
                str(root),
                "--network-mode",
                "offline" if offline else "online",
            ),
            cwd=root,
            env=_embedding_probe_environment(offline=offline),
            check=False,
            capture_output=True,
            text=True,
            timeout=EMBEDDING_PROBE_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            raise ValueError
        observation = EmbeddingProbeObservation.model_validate_json(completed.stdout)
        identity = embedding_identity_from_settings(settings)
        if (
            observation.model != identity.model
            or observation.revision != identity.revision
            or observation.code_revision != identity.code_revision
            or observation.identity_sha256 != identity.fingerprint
            or observation.declared_dimension != identity.dimension
            or observation.normalized != identity.normalize
            or observation.maximum_length != identity.max_length
        ):
            raise ValueError
        return EmbeddingReadinessEvidence(
            model=observation.model,
            revision=observation.revision,
            code_revision=observation.code_revision,
            identity_sha256=observation.identity_sha256,
            cache_preparation_network_used=False,
        )
    except (OSError, subprocess.SubprocessError, ValidationError, ValueError):
        raise ReadinessError(ReadinessErrorCode.EMBEDDING_SNAPSHOT_UNAVAILABLE) from None


def _probe_vector_backend(settings: RepoPilotSettings) -> VectorReadinessEvidence:
    try:
        info = PostgresVectorStore(settings).initialize()
        if (
            info.backend != "pgvector"
            or info.schema_name is None
            or info.table_name is None
            or info.extension_version is None
            or info.dimension != 768
        ):
            raise ValueError
        return VectorReadinessEvidence(
            backend="pgvector",
            schema_name=info.schema_name,
            table_name=info.table_name,
            extension_version=info.extension_version,
            dimension=768,
        )
    except (VectorStoreError, ValidationError, TypeError, ValueError):
        raise ReadinessError(ReadinessErrorCode.VECTOR_BACKEND_UNAVAILABLE) from None


def build_readiness_manifest(
    *,
    experiment_id: str,
    controller_commit: str,
    checked_at: datetime,
    artifacts: _ControlArtifacts,
    frozen_runtime: FrozenRuntimeEvidence,
    embedding: EmbeddingReadinessEvidence,
    vector_backend: VectorReadinessEvidence,
    images: tuple[ReadinessImageEvidence, ...],
    preflight: HoldoutPreflight,
) -> HoldoutRuntimeReadiness:
    plan = artifacts.models[3]
    authorization = artifacts.models[4]
    environment = artifacts.models[5]
    marker = artifacts.models[1]
    if not isinstance(plan, HoldoutApplicationPlan):
        raise ReadinessError(ReadinessErrorCode.INVALID_CONFIGURATION)
    if not isinstance(authorization, PaidHoldoutAuthorization):
        raise ReadinessError(ReadinessErrorCode.INVALID_CONFIGURATION)
    if not isinstance(environment, HoldoutEnvironmentManifest):
        raise ReadinessError(ReadinessErrorCode.INVALID_CONFIGURATION)
    if not isinstance(marker, FormalFreezeMarker):
        raise ReadinessError(ReadinessErrorCode.INVALID_CONFIGURATION)
    if (
        frozen_runtime.frozen_source_commit != marker.source_commit
        or frozen_runtime.environment_controller_commit != environment.controller_commit
    ):
        raise ReadinessError(ReadinessErrorCode.SOURCE_IDENTITY_MISMATCH)
    expected_images = tuple(
        (item.task_id, item.sandbox_image_reference) for item in environment.bindings
    )
    observed_images = tuple((item.task_id, item.registry_digest_reference) for item in images)
    if observed_images != expected_images:
        raise ReadinessError(ReadinessErrorCode.SOURCE_IDENTITY_MISMATCH)
    if (
        preflight.plan_sha256 != plan.content_sha256
        or preflight.paid_requests_made != 0
        or preflight.gold_visible_to_agent
    ):
        raise ReadinessError(ReadinessErrorCode.SOURCE_IDENTITY_MISMATCH)
    payload: dict[str, object] = {
        "schema_version": 1,
        "readiness_version": READINESS_VERSION,
        "experiment_id": experiment_id,
        "controller_commit": controller_commit,
        "checked_at": checked_at,
        "host_platform": "linux/amd64",
        "plan_raw_sha256": hashlib.sha256(artifacts.plan_raw).hexdigest(),
        "plan_content_sha256": plan.content_sha256,
        "authorization_raw_sha256": hashlib.sha256(artifacts.authorization_raw).hexdigest(),
        "authorization_content_sha256": authorization.content_sha256,
        "environment_raw_sha256": hashlib.sha256(artifacts.environment_raw).hexdigest(),
        "environment_content_sha256": environment.content_sha256,
        "frozen_runtime": frozen_runtime,
        "allowed_secret_environment": ALLOWED_SECRET_ENVIRONMENT,
        "provider": ProviderCredentialEvidence(),
        "embedding": embedding,
        "vector_backend": vector_backend,
        "images": images,
        "preflight": preflight,
        "paid_requests_made": 0,
        "gold_visible_to_agent": False,
        "ready_for_paid_holdout": True,
        "formal_blockers": (),
    }
    return HoldoutRuntimeReadiness.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def _embedding_probe_environment(*, offline: bool) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in EMBEDDING_CHILD_ENVIRONMENT_ALLOWLIST
    }
    environment["HF_HUB_OFFLINE"] = "1" if offline else "0"
    environment["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return environment


def _verify_frozen_runtime(
    *,
    root: Path,
    source_commit: str,
    environment_commit: str,
    controller_commit: str,
) -> FrozenRuntimeEvidence:
    """Accept only the audited archive-mtime patch after the formal source freeze."""

    if not _git_is_ancestor(root, source_commit, environment_commit) or not _git_is_ancestor(
        root, environment_commit, controller_commit
    ):
        raise ReadinessError(
            ReadinessErrorCode.SOURCE_IDENTITY_MISMATCH,
            diagnostic={"stage": "frozen_runtime_ancestry"},
        )
    changed_paths = _git_changed_paths(root, source_commit, controller_commit)
    if changed_paths != (AUDITED_COMPATIBILITY_PATH,):
        raise ReadinessError(
            ReadinessErrorCode.SOURCE_IDENTITY_MISMATCH,
            diagnostic={
                "changed_paths": changed_paths,
                "stage": "frozen_runtime_paths",
            },
        )
    try:
        base_blob = _git(root, "rev-parse", f"{source_commit}:{AUDITED_COMPATIBILITY_PATH}")
        environment_blob = _git(
            root,
            "rev-parse",
            f"{environment_commit}:{AUDITED_COMPATIBILITY_PATH}",
        )
        controller_blob = _git(
            root,
            "rev-parse",
            f"{controller_commit}:{AUDITED_COMPATIBILITY_PATH}",
        )
        patch_sha256 = hashlib.sha256(
            _git_diff_bytes(
                root,
                source_commit,
                environment_commit,
                AUDITED_COMPATIBILITY_PATH,
            )
        ).hexdigest()
    except (OSError, subprocess.SubprocessError):
        raise ReadinessError(
            ReadinessErrorCode.SOURCE_IDENTITY_MISMATCH,
            diagnostic={"stage": "frozen_runtime_git_inspection"},
        ) from None
    if (
        base_blob != AUDITED_COMPATIBILITY_BASE_BLOB
        or environment_blob != AUDITED_COMPATIBILITY_RESULT_BLOB
        or controller_blob != AUDITED_COMPATIBILITY_RESULT_BLOB
        or patch_sha256 != AUDITED_COMPATIBILITY_PATCH_SHA256
    ):
        raise ReadinessError(
            ReadinessErrorCode.SOURCE_IDENTITY_MISMATCH,
            diagnostic={
                "base_blob_matches": base_blob == AUDITED_COMPATIBILITY_BASE_BLOB,
                "controller_blob_matches": (controller_blob == AUDITED_COMPATIBILITY_RESULT_BLOB),
                "environment_blob_matches": (environment_blob == AUDITED_COMPATIBILITY_RESULT_BLOB),
                "patch_sha256_matches": (patch_sha256 == AUDITED_COMPATIBILITY_PATCH_SHA256),
                "stage": "frozen_runtime_patch_identity",
            },
        )
    return FrozenRuntimeEvidence(
        frozen_source_commit=source_commit,
        environment_controller_commit=environment_commit,
        changed_paths=(AUDITED_COMPATIBILITY_PATH,),
        base_blob_git_sha1=AUDITED_COMPATIBILITY_BASE_BLOB,
        result_blob_git_sha1=AUDITED_COMPATIBILITY_RESULT_BLOB,
        patch_sha256=AUDITED_COMPATIBILITY_PATCH_SHA256,
    )


def _git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=root,
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def _git_changed_paths(root: Path, source_commit: str, controller_commit: str) -> tuple[str, ...]:
    output = _git(
        root,
        "diff",
        "--name-only",
        source_commit,
        controller_commit,
        "--",
        *FROZEN_RUNTIME_PATHS,
    )
    return tuple(line for line in output.splitlines() if line)


def _git_diff_bytes(root: Path, base: str, target: str, path: str) -> bytes:
    completed = subprocess.run(
        (
            "git",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "--full-index",
            "--no-color",
            base,
            target,
            "--",
            path,
        ),
        cwd=root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_content_digest(value: StrictModel) -> None:
    payload = value.model_dump(mode="python", exclude={"content_sha256"})
    if _canonical_sha256(payload) != getattr(value, "content_sha256", None):
        raise ValueError("runtime readiness content digest does not match")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReadinessError, ValidationError) as error:
        raise SystemExit(str(error)) from None
