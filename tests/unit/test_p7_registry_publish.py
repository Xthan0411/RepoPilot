"""P7-S6 immutable Artifact Registry publication contract tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.p7_publish_registry_images as registry  # noqa: E402
from scripts.p7_django_sandbox_canary import _canonical_sha256  # noqa: E402
from scripts.p7_sandbox_matrix_preflight import (  # noqa: E402
    RESULT_PATH as SANDBOX_PREFLIGHT_PATH,
)
from scripts.p7_sandbox_matrix_preflight import (  # noqa: E402
    SandboxMatrixPreflight,
    _safe_name,
)
from scripts.p7_swebench_environment import EXPECTED_TASK_IDS  # noqa: E402

COMMITTED_MANIFEST_RAW_SHA256 = "edc0b94b7ec8247e00e2f36c80d5f930e1dc7141602da9e6514f55338841fef9"


def _completed(
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(("command",), returncode, stdout, stderr)


def _preflight() -> SandboxMatrixPreflight:
    return SandboxMatrixPreflight.model_validate_json(
        PROJECT_ROOT.joinpath(SANDBOX_PREFLIGHT_PATH).read_bytes()
    )


def _registry_image(
    task_id: str,
    local_image_id: str,
    source_policy_reference: str,
    *,
    index: int,
) -> registry.RegistryImageEvidence:
    image_name = _safe_name(task_id)
    digest = f"sha256:{index:064x}"
    return registry.RegistryImageEvidence(
        task_id=task_id,
        source_policy_reference=source_policy_reference,
        local_image_id=local_image_id,
        local_size_bytes=1_000 + index,
        registry_tag_reference=(f"{registry.REGISTRY_ROOT}/{image_name}:{registry.IMMUTABLE_TAG}"),
        registry_manifest_digest=digest,
        registry_digest_reference=f"{registry.REGISTRY_ROOT}/{image_name}@{digest}",
        remote_config_image_id=local_image_id,
        publication_outcome="published",
        remote_pull_verified=True,
    )


def _manifest() -> registry.RegistryManifest:
    preflight = _preflight()
    images = tuple(
        _registry_image(
            source.task_id,
            source.adapted_local_image_id,
            source.adapted_policy_reference,
            index=index,
        )
        for index, source in enumerate(preflight.task_evidence, start=1)
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "manifest_version": registry.MANIFEST_VERSION,
        "publisher_commit": "a" * 40,
        "collected_at": datetime(2026, 8, 8, tzinfo=UTC),
        "source_preflight_relative_path": SANDBOX_PREFLIGHT_PATH.as_posix(),
        "source_preflight_raw_sha256": registry.EXPECTED_PREFLIGHT_RAW_SHA256,
        "source_preflight_content_sha256": preflight.content_sha256,
        "project_id": registry.PROJECT_ID,
        "location": registry.LOCATION,
        "repository": registry.REPOSITORY,
        "repository_resource_name": (
            "projects/repopilot-504912/locations/us-west1/repositories/repopilot-p7-s6"
        ),
        "registry_host": registry.REGISTRY_HOST,
        "registry_root": registry.REGISTRY_ROOT,
        "repository_format": "DOCKER",
        "repository_mode": "STANDARD_REPOSITORY",
        "immutable_tags": True,
        "vulnerability_scanning_disabled": True,
        "active_principal": registry.SERVICE_ACCOUNT,
        "credential_source": "compute_engine_metadata",
        "service_account_key_used": False,
        "host_platform": "linux/amd64",
        "images": images,
        "published_task_count": 8,
        "required_task_count": 8,
        "paid_requests_made": 0,
        "provider_requests_made": 0,
        "gold_visible_to_agent": False,
        "registry_qualification_passed": True,
        "formal_environment_ready": False,
        "formal_blockers": ("task_command_timeout_contract_mismatch",),
    }
    return registry.RegistryManifest.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def test_committed_preflight_has_exact_expected_raw_identity() -> None:
    content = PROJECT_ROOT.joinpath(SANDBOX_PREFLIGHT_PATH).read_bytes()

    preflight = registry._load_preflight(content)

    assert hashlib.sha256(content).hexdigest() == registry.EXPECTED_PREFLIGHT_RAW_SHA256
    assert len(preflight.task_evidence) == 8
    with pytest.raises(registry.RegistryPublishError, match="source_preflight_invalid"):
        registry._load_preflight(content + b"\n")


def test_committed_registry_manifest_is_exact_complete_external_evidence() -> None:
    content = PROJECT_ROOT.joinpath(registry.RESULT_PATH).read_bytes()

    manifest = registry.RegistryManifest.model_validate_json(content)

    assert hashlib.sha256(content).hexdigest() == COMMITTED_MANIFEST_RAW_SHA256
    assert manifest.publisher_commit == "a5c43fdc9e3f2c33db0ab0e6f7af3a5cd46dd170"
    assert tuple(item.task_id for item in manifest.images) == EXPECTED_TASK_IDS
    assert all(item.publication_outcome == "published" for item in manifest.images)
    assert all(item.remote_pull_verified for item in manifest.images)
    assert manifest.published_task_count == manifest.required_task_count == 8
    assert manifest.formal_blockers == ("task_command_timeout_contract_mismatch",)
    assert manifest.paid_requests_made == manifest.provider_requests_made == 0
    assert not manifest.gold_visible_to_agent


def test_registry_manifest_binds_all_eight_images_and_remaining_blocker() -> None:
    manifest = _manifest()

    assert tuple(item.task_id for item in manifest.images) == EXPECTED_TASK_IDS
    assert len({item.registry_manifest_digest for item in manifest.images}) == 8
    assert manifest.formal_blockers == ("task_command_timeout_contract_mismatch",)
    assert manifest.paid_requests_made == 0
    assert manifest.provider_requests_made == 0
    assert not manifest.gold_visible_to_agent


def test_registry_image_rejects_task_digest_or_config_drift() -> None:
    manifest = _manifest()
    item = manifest.images[0]
    raw = item.model_dump(mode="json")
    raw["registry_digest_reference"] = raw["registry_digest_reference"].replace(
        item.registry_manifest_digest,
        "sha256:" + "f" * 64,
    )
    with pytest.raises(ValidationError, match="digest reference is inconsistent"):
        registry.RegistryImageEvidence.model_validate(raw)

    raw = item.model_dump(mode="json")
    raw["remote_config_image_id"] = "sha256:" + "f" * 64
    with pytest.raises(ValidationError, match="remote image config"):
        registry.RegistryImageEvidence.model_validate(raw)


def test_repository_validation_requires_exact_private_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {
        "name": ("projects/repopilot-504912/locations/us-west1/repositories/repopilot-p7-s6"),
        "format": "DOCKER",
        "mode": "STANDARD_REPOSITORY",
        "dockerConfig": {"immutableTags": True},
        "vulnerabilityScanningConfig": {"enablementState": "SCANNING_DISABLED"},
    }
    monkeypatch.setattr(
        registry,
        "_run_command",
        lambda *args, **kwargs: _completed(stdout=json.dumps(record)),
    )

    assert registry._validate_repository() == record["name"]

    record["dockerConfig"] = {"immutableTags": False}
    with pytest.raises(registry.RegistryPublishError, match="repository_invalid"):
        registry._validate_repository()


def test_identity_requires_matching_metadata_and_active_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "_metadata_text", lambda path: registry.SERVICE_ACCOUNT)
    monkeypatch.setattr(
        registry,
        "_run_command",
        lambda *args, **kwargs: _completed(stdout=registry.SERVICE_ACCOUNT + "\n"),
    )

    assert registry._validate_identity() == registry.SERVICE_ACCOUNT

    monkeypatch.setattr(registry, "_metadata_text", lambda path: "other@example.invalid")
    with pytest.raises(registry.RegistryPublishError, match="identity_mismatch"):
        registry._validate_identity()


def test_local_image_inspection_requires_exact_id_and_amd64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_id = "sha256:" + "a" * 64
    record: dict[str, object] = {
        "Id": image_id,
        "Architecture": "amd64",
        "Os": "linux",
        "Size": 123,
        "RepoDigests": [],
    }
    monkeypatch.setattr(
        registry,
        "_run_command",
        lambda *args, **kwargs: _completed(stdout=json.dumps([record])),
    )

    inspected = registry._inspect_local_image(image_id)

    assert inspected["image_id"] == image_id
    assert inspected["size_bytes"] == 123

    record["Architecture"] = "arm64"
    with pytest.raises(registry.RegistryPublishError, match="local_image_platform_mismatch"):
        registry._inspect_local_image(image_id)


def test_remote_probe_is_resumable_but_rejects_wrong_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_id = "sha256:" + "a" * 64
    digest = "sha256:" + "b" * 64
    registry_image = f"{registry.REGISTRY_ROOT}/task"
    tag = f"{registry_image}:{registry.IMMUTABLE_TAG}"
    monkeypatch.setattr(
        registry,
        "_run_command",
        lambda *args, **kwargs: _completed(),
    )
    monkeypatch.setattr(
        registry,
        "_inspect_local_image",
        lambda *args, **kwargs: {
            "image_id": image_id,
            "size_bytes": 123,
            "repository_digests": (f"{registry_image}@{digest}",),
        },
    )

    assert registry._pull_and_verify(
        tag,
        registry_image=registry_image,
        expected_image_id=image_id,
        docker_config="/tmp/docker-config",
        missing_allowed=False,
    ) == (digest, image_id)

    monkeypatch.setattr(
        registry,
        "_inspect_local_image",
        lambda *args, **kwargs: {
            "image_id": "sha256:" + "c" * 64,
            "size_bytes": 123,
            "repository_digests": (f"{registry_image}@{digest}",),
        },
    )
    with pytest.raises(registry.RegistryPublishError, match="remote_tag_conflict"):
        registry._pull_and_verify(
            tag,
            registry_image=registry_image,
            expected_image_id=image_id,
            docker_config="/tmp/docker-config",
            missing_allowed=False,
        )


def test_remote_probe_accepts_only_explicit_missing_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registry,
        "_run_command",
        lambda *args, **kwargs: _completed(
            returncode=1,
            stderr="manifest unknown: requested tag was not found",
        ),
    )

    assert (
        registry._pull_and_verify(
            f"{registry.REGISTRY_ROOT}/task:{registry.IMMUTABLE_TAG}",
            registry_image=f"{registry.REGISTRY_ROOT}/task",
            expected_image_id="sha256:" + "a" * 64,
            docker_config="/tmp/docker-config",
            missing_allowed=True,
        )
        is None
    )

    monkeypatch.setattr(
        registry,
        "_run_command",
        lambda *args, **kwargs: _completed(returncode=1, stderr="permission denied"),
    )
    with pytest.raises(registry.RegistryPublishError, match="remote_probe_failed"):
        registry._pull_and_verify(
            f"{registry.REGISTRY_ROOT}/task:{registry.IMMUTABLE_TAG}",
            registry_image=f"{registry.REGISTRY_ROOT}/task",
            expected_image_id="sha256:" + "a" * 64,
            docker_config="/tmp/docker-config",
            missing_allowed=True,
        )


def test_existing_remote_tag_skips_mutating_docker_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _preflight().task_evidence[0]
    digest = "sha256:" + "b" * 64
    monkeypatch.setattr(
        registry,
        "_pull_and_verify",
        lambda *args, **kwargs: (digest, source.adapted_local_image_id),
    )

    def reject_mutation(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("existing remote tag must not be republished")

    monkeypatch.setattr(registry, "_run_command", reject_mutation)

    evidence = registry._publish_one(
        source,
        {"image_id": source.adapted_local_image_id, "size_bytes": 123},
        docker_config="/tmp/docker-config",
    )

    assert evidence.publication_outcome == "already_present"
    assert evidence.registry_manifest_digest == digest


def test_metadata_token_is_bounded_and_command_environment_removes_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "token-" + "x" * 64
    monkeypatch.setattr(
        registry,
        "_metadata_text",
        lambda path: json.dumps({"access_token": token, "expires_in": 3_599}),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-propagate")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/key.json")
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret")
    monkeypatch.setenv("SOME_ACCESS_TOKEN", "must-not-propagate")

    assert registry._metadata_token() == token
    environment = registry._command_environment("/tmp/isolated-docker")
    assert "OPENAI_API_KEY" not in environment
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in environment
    assert "DATABASE_URL" not in environment
    assert "SOME_ACCESS_TOKEN" not in environment
    assert environment["DOCKER_CONFIG"] == "/tmp/isolated-docker"


def test_publish_failure_still_logs_out_isolated_docker_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _preflight()
    calls: list[tuple[tuple[str, ...], str | None, str | None]] = []

    def run_command(
        argv: tuple[str, ...],
        *,
        input_text: str | None = None,
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((argv, input_text, None if env is None else env.get("DOCKER_CONFIG")))
        return _completed()

    monkeypatch.setattr(registry, "_metadata_token", lambda: "short-lived-token")
    monkeypatch.setattr(registry, "_run_command", run_command)
    monkeypatch.setattr(
        registry,
        "_publish_one",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            registry.RegistryPublishError(registry.RegistryPublishErrorCode.PUBLISH_FAILED)
        ),
    )

    with pytest.raises(registry.RegistryPublishError, match="publish_failed"):
        registry._publish_all(preflight, tuple({} for _ in preflight.task_evidence))

    assert calls[0][0][:2] == ("docker", "login")
    assert calls[0][1] == "short-lived-token\n"
    assert calls[-1][0] == ("docker", "logout", registry.REGISTRY_HOST)
    assert calls[0][2]
    assert calls[0][2] == calls[-1][2]


def test_execute_requires_exact_registry_confirmation(tmp_path: Path) -> None:
    with pytest.raises(registry.RegistryPublishError, match="invalid_confirmation"):
        registry._validate_controller(tmp_path, execute=True, confirmation=None)
    with pytest.raises(registry.RegistryPublishError, match="invalid_confirmation"):
        registry._validate_controller(
            tmp_path,
            execute=True,
            confirmation="another-registry.invalid/project/repository",
        )


def test_direct_file_help_invocation_has_no_external_side_effect() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            str(PROJECT_ROOT / "scripts/p7_publish_registry_images.py"),
            "--help",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--confirm-registry" in completed.stdout
    assert "--execute" in completed.stdout
