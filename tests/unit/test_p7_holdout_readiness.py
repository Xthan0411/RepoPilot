"""P7-S6 explicit authorization and zero-paid runtime-readiness tests."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from repopilot.evals import FormalFreezeMarker, load_experiment_matrix
from repopilot.evals.dataset import DatasetPreflightReport
from repopilot.sandbox import DockerClientError, DockerImageIdentity

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.p7_embedding_readiness_probe as embedding_probe  # noqa: E402
import scripts.p7_holdout_control as control  # noqa: E402
import scripts.p7_holdout_readiness as readiness  # noqa: E402
from scripts.p7_holdout_control import (  # noqa: E402
    HoldoutApplicationPlan,
    HoldoutEnvironmentManifest,
    PaidHoldoutAuthorization,
    build_holdout_preflight,
)

AUTHORIZATION_PATH = PROJECT_ROOT / "results/p7-holdout/authorization-v1.json"
AUTHORIZATION_RAW_SHA256 = "37df6c2210f71c3046cfc9c79101b4911b00e4143cc9a47999276fd7009aec58"
CONTROLLER_COMMIT = "c" * 40
NOW = datetime(2026, 8, 8, 19, tzinfo=UTC)


def _artifacts() -> readiness._ControlArtifacts:
    plan_raw = PROJECT_ROOT.joinpath(control.PLAN_PATH).read_bytes()
    authorization_raw = AUTHORIZATION_PATH.read_bytes()
    environment_raw = PROJECT_ROOT.joinpath(control.ENVIRONMENT_PATH).read_bytes()
    matrix = load_experiment_matrix(
        PROJECT_ROOT / control.MATRIX_PATH,
        repository_root=PROJECT_ROOT,
    )
    marker = FormalFreezeMarker.model_validate_json(
        PROJECT_ROOT.joinpath(control.FREEZE_PATH).read_bytes()
    )
    dataset_preflight = DatasetPreflightReport.model_validate_json(
        PROJECT_ROOT.joinpath(control.DATASET_PREFLIGHT_PATH).read_bytes()
    )
    return readiness._ControlArtifacts(
        plan_raw=plan_raw,
        authorization_raw=authorization_raw,
        environment_raw=environment_raw,
        models=(
            matrix,
            marker,
            dataset_preflight,
            HoldoutApplicationPlan.model_validate_json(plan_raw),
            PaidHoldoutAuthorization.model_validate_json(authorization_raw),
            HoldoutEnvironmentManifest.model_validate_json(environment_raw),
        ),
    )


def _ready_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> control.HoldoutPreflight:
    artifacts = _artifacts()
    _, marker, dataset_preflight, plan, authorization, environment = artifacts.models
    monkeypatch.setattr(control, "_checkouts_match", lambda root, manifest: True)
    return build_holdout_preflight(
        root=tmp_path,
        plan=plan,
        marker=marker,
        dataset_preflight=dataset_preflight,
        checked_at=NOW,
        controller_source_clean=True,
        frozen_runtime_matches=True,
        authorization=authorization,
        environment_manifest=environment,
        provider_credential_available=True,
        vector_backend_available=True,
        embedding_snapshot_available=True,
        controller_commit=CONTROLLER_COMMIT,
    )


def _embedding() -> readiness.EmbeddingReadinessEvidence:
    matrix = _artifacts().models[0]
    values = matrix.shared_settings.values
    return readiness.EmbeddingReadinessEvidence(
        model=str(values["embedding_model"]),
        revision=str(values["embedding_revision"]),
        code_revision=str(values["embedding_code_revision"]),
        identity_sha256="d" * 64,
        cache_preparation_network_used=False,
    )


def _vector() -> readiness.VectorReadinessEvidence:
    return readiness.VectorReadinessEvidence(
        schema_name="repopilot",
        table_name="vector_embeddings",
        extension_version="0.8.1",
    )


def _images() -> tuple[readiness.ReadinessImageEvidence, ...]:
    environment = _artifacts().models[5]
    return tuple(
        readiness.ReadinessImageEvidence(
            task_id=item.task_id,
            registry_digest_reference=item.sandbox_image_reference,
            local_image_id="sha256:" + item.sandbox_image_reference.rsplit("sha256:", 1)[1],
        )
        for item in environment.bindings
    )


def _frozen_runtime() -> readiness.FrozenRuntimeEvidence:
    _, marker, _, _, _, environment = _artifacts().models
    return readiness.FrozenRuntimeEvidence(
        frozen_source_commit=marker.source_commit,
        environment_controller_commit=environment.controller_commit,
        changed_paths=(readiness.AUDITED_COMPATIBILITY_PATH,),
        base_blob_git_sha1=readiness.AUDITED_COMPATIBILITY_BASE_BLOB,
        result_blob_git_sha1=readiness.AUDITED_COMPATIBILITY_RESULT_BLOB,
        patch_sha256=readiness.AUDITED_COMPATIBILITY_PATCH_SHA256,
    )


def test_committed_authorization_exactly_binds_user_approved_plan() -> None:
    raw = AUTHORIZATION_PATH.read_bytes()
    authorization = PaidHoldoutAuthorization.model_validate_json(raw)
    plan = HoldoutApplicationPlan.model_validate_json(
        PROJECT_ROOT.joinpath(control.PLAN_PATH).read_bytes()
    )

    assert hashlib.sha256(raw).hexdigest() == AUTHORIZATION_RAW_SHA256
    assert authorization.authorization_id == "p7-holdout-user-approved-v1"
    assert authorization.approved_by == "ethan"
    assert authorization.plan_sha256 == plan.content_sha256
    assert str(authorization.max_total_cost_usd) == "32.00"
    assert authorization.paid_execution_approved
    assert authorization.final_in_flight_response_overshoot_acknowledged


def test_readiness_manifest_binds_all_zero_paid_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _artifacts()
    manifest = readiness.build_readiness_manifest(
        experiment_id="repopilot-p7-v2",
        controller_commit=CONTROLLER_COMMIT,
        checked_at=NOW,
        artifacts=artifacts,
        frozen_runtime=_frozen_runtime(),
        embedding=_embedding(),
        vector_backend=_vector(),
        images=_images(),
        preflight=_ready_preflight(tmp_path, monkeypatch),
    )
    serialized = manifest.model_dump_json()

    assert manifest.ready_for_paid_holdout
    assert manifest.preflight.ready_for_paid_holdout
    assert manifest.preflight.controller_commit == CONTROLLER_COMMIT
    assert manifest.allowed_secret_environment == ("DATABASE_URL", "OPENAI_API_KEY")
    assert manifest.frozen_runtime.changed_paths == ("src/repopilot/sandbox/docker.py",)
    assert not manifest.frozen_runtime.changes_model_prompt_retrieval_task_or_scoring
    assert manifest.provider.api_requests_made == 0
    assert manifest.paid_requests_made == 0
    assert manifest.embedding.final_probe_network == "offline"
    assert manifest.embedding.observed_dimension == 768
    assert manifest.vector_backend.dimension == 768
    assert len(manifest.images) == 8
    assert not manifest.gold_visible_to_agent
    assert "expected_patch" not in serialized
    assert "OPENAI_API_KEY=" not in serialized
    assert "postgresql://" not in serialized


def test_readiness_rejects_incomplete_image_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _artifacts()
    with pytest.raises(readiness.ReadinessError) as captured:
        readiness.build_readiness_manifest(
            experiment_id="repopilot-p7-v2",
            controller_commit=CONTROLLER_COMMIT,
            checked_at=NOW,
            artifacts=artifacts,
            frozen_runtime=_frozen_runtime(),
            embedding=_embedding(),
            vector_backend=_vector(),
            images=_images()[:-1],
            preflight=_ready_preflight(tmp_path, monkeypatch),
        )
    assert captured.value.code is readiness.ReadinessErrorCode.SOURCE_IDENTITY_MISMATCH


def test_unapproved_repopilot_override_fails_before_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scripts.p7_holdout_readiness.platform.system", lambda: "Linux")
    monkeypatch.setattr("scripts.p7_holdout_readiness.platform.machine", lambda: "x86_64")
    monkeypatch.setenv("REPOPILOT_MODEL", "different-model")

    with pytest.raises(readiness.ReadinessError) as captured:
        readiness._verify_host_and_environment()

    assert captured.value.code is readiness.ReadinessErrorCode.UNAPPROVED_ENVIRONMENT_OVERRIDE


def test_image_diagnostic_attempts_all_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _artifacts().models[5]
    missing = {environment.bindings[1].task_id, environment.bindings[6].task_id}

    class FakeDockerClient:
        def resolve_image(self, reference: str) -> DockerImageIdentity:
            task_id = next(
                item.task_id
                for item in environment.bindings
                if item.sandbox_image_reference == reference
            )
            if task_id in missing:
                raise DockerClientError("image inspection")
            digest = reference.rsplit("sha256:", 1)[1]
            return DockerImageIdentity(
                image_id=f"sha256:{digest}",
                repository_digests=(reference,),
            )

    monkeypatch.setattr(readiness, "DockerCliClient", FakeDockerClient)
    images, missing_ids = readiness._resolve_images(environment)

    assert len(images) == 6
    assert set(missing_ids) == missing


def test_exact_environment_compatibility_patch_is_accepted(tmp_path: Path) -> None:
    """Verify exact historical bytes using isolated commits, including in a fresh clone."""

    def git(*arguments: str) -> str:
        return subprocess.run(
            ("git", *arguments),
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    current = PROJECT_ROOT.joinpath(readiness.AUDITED_COMPATIBILITY_PATH).read_text()
    added_constant = (
        "# Deterministic 1980-01-02 UTC: accepted by ZIP-based build backends even after\n"
        "# a worst-case timezone conversion, unlike TarInfo's default Unix epoch.\n"
        "DOCKER_ARCHIVE_MTIME_SECONDS = 315_619_200\n"
    )
    added_assignment = "    info.mtime = DOCKER_ARCHIVE_MTIME_SECONDS\n"
    assert current.count(added_constant) == 1
    assert current.count(added_assignment) == 2
    frozen = current.replace(added_constant, "").replace(added_assignment, "")
    path = tmp_path / readiness.AUDITED_COMPATIBILITY_PATH
    path.parent.mkdir(parents=True)
    path.write_text(frozen)
    git("init", "--quiet")
    git("config", "user.name", "RepoPilot test")
    git("config", "user.email", "test@example.invalid")
    git("config", "commit.gpgsign", "false")
    git("add", readiness.AUDITED_COMPATIBILITY_PATH)
    git("commit", "--quiet", "-m", "Frozen sandbox fixture")
    source_commit = git("rev-parse", "HEAD")
    path.write_text(current)
    git("add", readiness.AUDITED_COMPATIBILITY_PATH)
    git("commit", "--quiet", "-m", "Audited archive metadata compatibility patch")
    environment_commit = git("rev-parse", "HEAD")
    git("commit", "--quiet", "--allow-empty", "-m", "Controller fixture")
    controller_commit = git("rev-parse", "HEAD")

    evidence = readiness._verify_frozen_runtime(
        root=tmp_path,
        source_commit=source_commit,
        environment_commit=environment_commit,
        controller_commit=controller_commit,
    )

    assert evidence.changed_paths == (readiness.AUDITED_COMPATIBILITY_PATH,)
    assert evidence.base_blob_git_sha1 == readiness.AUDITED_COMPATIBILITY_BASE_BLOB
    assert evidence.result_blob_git_sha1 == readiness.AUDITED_COMPATIBILITY_RESULT_BLOB
    assert evidence.patch_sha256 == readiness.AUDITED_COMPATIBILITY_PATCH_SHA256


def test_any_additional_frozen_runtime_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, marker, _, _, _, environment = _artifacts().models
    monkeypatch.setattr(readiness, "_git_is_ancestor", lambda *args: True)
    monkeypatch.setattr(
        readiness,
        "_git_changed_paths",
        lambda *args: (readiness.AUDITED_COMPATIBILITY_PATH, "pyproject.toml"),
    )

    with pytest.raises(readiness.ReadinessError) as captured:
        readiness._verify_frozen_runtime(
            root=PROJECT_ROOT,
            source_commit=marker.source_commit,
            environment_commit=environment.controller_commit,
            controller_commit="f" * 40,
        )

    assert captured.value.code is readiness.ReadinessErrorCode.SOURCE_IDENTITY_MISMATCH
    assert captured.value.diagnostic == {
        "changed_paths": (
            readiness.AUDITED_COMPATIBILITY_PATH,
            "pyproject.toml",
        ),
        "stage": "frozen_runtime_paths",
    }


def test_embedding_subprocess_environment_is_offline_and_secret_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("DATABASE_URL", "must-not-leak")
    monkeypatch.setenv("REPOPILOT_MODEL", "must-not-leak")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-leak")

    environment = readiness._embedding_probe_environment(offline=True)

    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert "OPENAI_API_KEY" not in environment
    assert "DATABASE_URL" not in environment
    assert "REPOPILOT_MODEL" not in environment
    assert "UNRELATED_SECRET" not in environment


def test_embedding_child_rejects_secret_even_with_offline_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for name in tuple(os.environ):
        if name.startswith("REPOPILOT_"):
            monkeypatch.delenv(name, raising=False)

    embedding_probe._validate_network_mode("offline")
    monkeypatch.setenv("OPENAI_API_KEY", "must-be-rejected")

    with pytest.raises(SystemExit, match="embedding_probe_secret_exposure"):
        embedding_probe._validate_network_mode("offline")


def test_module_help_has_no_side_effect() -> None:
    completed = subprocess.run(
        (sys.executable, "-m", "scripts.p7_holdout_readiness", "--help"),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--prepare-embedding-cache" in completed.stdout
