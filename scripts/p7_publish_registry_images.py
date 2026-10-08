"""Publish qualified P7-S6 images and retain an immutable registry manifest.

The command consumes the committed all-task Sandbox preflight, verifies the exact local
image IDs and the dedicated Artifact Registry boundary, and publishes one immutable tag
per frozen task. Authentication uses a short-lived token from the attached Compute
Engine service account and an isolated temporary Docker configuration. No Provider,
Agent, task setup, test, or evaluator Gold path is invoked.

Without ``--execute`` the command is read-only. Mutating execution additionally requires
an exact ``--confirm-registry`` value. Partial remote publication is resumable: an
existing immutable tag is accepted only when pulling it resolves to the expected local
Docker config image ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import UTC, datetime
from enum import StrEnum, unique
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import AwareDatetime, Field, model_validator

from repopilot.schemas import GitSha, Sha256, StrictModel, TaskId

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.p7_django_sandbox_canary import (
    _canonical_sha256,
    _git,
    _write_private_create_only,
)
from scripts.p7_sandbox_matrix_preflight import (
    RESULT_PATH as SANDBOX_PREFLIGHT_PATH,
)
from scripts.p7_sandbox_matrix_preflight import (
    SandboxMatrixPreflight,
    SandboxTaskPreflightEvidence,
    _safe_name,
)
from scripts.p7_swebench_environment import EXPECTED_TASK_IDS

SCHEMA_VERSION: Literal[1] = 1
MANIFEST_VERSION: Literal["p7-registry-manifest-v1"] = "p7-registry-manifest-v1"
PROJECT_ID: Literal["repopilot-504912"] = "repopilot-504912"
LOCATION: Literal["us-west1"] = "us-west1"
REPOSITORY: Literal["repopilot-p7-s6"] = "repopilot-p7-s6"
SERVICE_ACCOUNT: Literal["repopilot-eval-publisher@repopilot-504912.iam.gserviceaccount.com"] = (
    "repopilot-eval-publisher@repopilot-504912.iam.gserviceaccount.com"
)
REGISTRY_HOST: Literal["us-west1-docker.pkg.dev"] = "us-west1-docker.pkg.dev"
REGISTRY_ROOT: Literal["us-west1-docker.pkg.dev/repopilot-504912/repopilot-p7-s6"] = (
    "us-west1-docker.pkg.dev/repopilot-504912/repopilot-p7-s6"
)
IMMUTABLE_TAG: Literal["p7-s6-v1"] = "p7-s6-v1"
RESULT_PATH = Path("results/p7-holdout/registry-manifest-v1.json")
EXPECTED_PREFLIGHT_RAW_SHA256: Sha256 = (
    "9504aed7904436f807d890077f927dc3a4c0a6cfa890d383811535c2c8889c1c"
)
METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_NOT_FOUND_MARKERS = ("manifest unknown", "name unknown", "not found", "not_found")
_SECRET_ENV_KEYS = {
    "CLOUDSDK_AUTH_ACCESS_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "JINA_API_KEY",
    "OPENAI_API_KEY",
    "REPOPILOT_JINA_API_KEY",
    "REPOPILOT_OPENAI_API_KEY",
}


@unique
class RegistryPublishErrorCode(StrEnum):
    """Stable fail-closed outcomes for the registry publication boundary."""

    INVALID_CONFIRMATION = "invalid_confirmation"
    HOST_PLATFORM_MISMATCH = "host_platform_mismatch"
    CONTROLLER_SOURCE_DIRTY = "controller_source_dirty"
    SOURCE_PREFLIGHT_INVALID = "source_preflight_invalid"
    OUTPUT_CONFLICT = "output_conflict"
    METADATA_UNAVAILABLE = "metadata_unavailable"
    IDENTITY_MISMATCH = "identity_mismatch"
    REPOSITORY_INVALID = "repository_invalid"
    LOCAL_IMAGE_UNAVAILABLE = "local_image_unavailable"
    LOCAL_IMAGE_IDENTITY_MISMATCH = "local_image_identity_mismatch"
    LOCAL_IMAGE_PLATFORM_MISMATCH = "local_image_platform_mismatch"
    AUTHENTICATION_FAILED = "authentication_failed"
    REMOTE_PROBE_FAILED = "remote_probe_failed"
    REMOTE_TAG_CONFLICT = "remote_tag_conflict"
    PUBLISH_FAILED = "publish_failed"
    REMOTE_DIGEST_INVALID = "remote_digest_invalid"
    LOGOUT_FAILED = "logout_failed"
    COMMAND_UNAVAILABLE = "command_unavailable"


class RegistryPublishError(RuntimeError):
    """One secret-free registry publication failure."""

    def __init__(self, code: RegistryPublishErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


class RegistryImageEvidence(StrictModel):
    """One local-to-registry identity proof without credentials or push logs."""

    task_id: TaskId
    source_policy_reference: str = Field(
        pattern=r"^repopilot\.local/[a-z0-9._/-]+@sha256:[0-9a-f]{64}$"
    )
    local_image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    local_size_bytes: int = Field(gt=0, strict=True)
    platform: Literal["linux/amd64"] = "linux/amd64"
    immutable_tag: Literal["p7-s6-v1"] = IMMUTABLE_TAG
    registry_tag_reference: str = Field(
        pattern=(
            r"^us-west1-docker\.pkg\.dev/repopilot-504912/repopilot-p7-s6/"
            r"[a-z0-9._-]+:p7-s6-v1$"
        )
    )
    registry_manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    registry_digest_reference: str = Field(
        pattern=(
            r"^us-west1-docker\.pkg\.dev/repopilot-504912/repopilot-p7-s6/"
            r"[a-z0-9._-]+@sha256:[0-9a-f]{64}$"
        )
    )
    remote_config_image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    publication_outcome: Literal["published", "already_present"]
    remote_pull_verified: Literal[True] = True

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        image_name = _safe_name(self.task_id)
        expected_tag = f"{REGISTRY_ROOT}/{image_name}:{IMMUTABLE_TAG}"
        expected_digest = f"{REGISTRY_ROOT}/{image_name}@{self.registry_manifest_digest}"
        if self.registry_tag_reference != expected_tag:
            raise ValueError("registry tag does not match task identity")
        if self.registry_digest_reference != expected_digest:
            raise ValueError("registry digest reference is inconsistent")
        if self.remote_config_image_id != self.local_image_id:
            raise ValueError("remote image config does not match the qualified local image")
        if not self.source_policy_reference.endswith("@" + self.local_image_id):
            raise ValueError("source policy reference does not match local image ID")
        return self


class RegistryManifest(StrictModel):
    """All-eight immutable registry evidence with the remaining blocker explicit."""

    schema_version: Literal[1] = SCHEMA_VERSION
    manifest_version: Literal["p7-registry-manifest-v1"] = MANIFEST_VERSION
    publisher_commit: GitSha
    collected_at: AwareDatetime
    source_preflight_relative_path: Literal[
        "results/p7-holdout/sandbox-matrix-preflight-v1.json"
    ] = "results/p7-holdout/sandbox-matrix-preflight-v1.json"
    source_preflight_raw_sha256: Sha256
    source_preflight_content_sha256: Sha256
    project_id: Literal["repopilot-504912"] = PROJECT_ID
    location: Literal["us-west1"] = LOCATION
    repository: Literal["repopilot-p7-s6"] = REPOSITORY
    repository_resource_name: Literal[
        "projects/repopilot-504912/locations/us-west1/repositories/repopilot-p7-s6"
    ]
    registry_host: Literal["us-west1-docker.pkg.dev"] = REGISTRY_HOST
    registry_root: Literal["us-west1-docker.pkg.dev/repopilot-504912/repopilot-p7-s6"] = (
        REGISTRY_ROOT
    )
    repository_format: Literal["DOCKER"] = "DOCKER"
    repository_mode: Literal["STANDARD_REPOSITORY"] = "STANDARD_REPOSITORY"
    immutable_tags: Literal[True] = True
    vulnerability_scanning_disabled: Literal[True] = True
    active_principal: Literal[
        "repopilot-eval-publisher@repopilot-504912.iam.gserviceaccount.com"
    ] = SERVICE_ACCOUNT
    credential_source: Literal["compute_engine_metadata"] = "compute_engine_metadata"
    service_account_key_used: Literal[False] = False
    host_platform: Literal["linux/amd64"] = "linux/amd64"
    images: tuple[RegistryImageEvidence, ...]
    published_task_count: Literal[8]
    required_task_count: Literal[8] = 8
    paid_requests_made: Literal[0] = 0
    provider_requests_made: Literal[0] = 0
    gold_visible_to_agent: Literal[False] = False
    registry_qualification_passed: Literal[True]
    formal_environment_ready: Literal[False]
    formal_blockers: tuple[Literal["task_command_timeout_contract_mismatch"], ...]
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if tuple(item.task_id for item in self.images) != EXPECTED_TASK_IDS:
            raise ValueError("registry manifest must cover the exact frozen task set")
        if len({item.local_image_id for item in self.images}) != 8:
            raise ValueError("registry manifest local image IDs must be unique")
        if len({item.registry_manifest_digest for item in self.images}) != 8:
            raise ValueError("registry manifest digests must be unique")
        if self.source_preflight_raw_sha256 != EXPECTED_PREFLIGHT_RAW_SHA256:
            raise ValueError("registry manifest source preflight identity changed")
        if self.formal_blockers != ("task_command_timeout_contract_mismatch",):
            raise ValueError("registry manifest blockers are inconsistent")
        payload = self.model_dump(mode="python", exclude={"content_sha256"})
        if _canonical_sha256(payload) != self.content_sha256:
            raise ValueError("registry manifest content digest does not match")
        return self


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--execute",
        action="store_true",
        help="publish images and create the manifest (default is a read-only preflight)",
    )
    parser.add_argument(
        "--confirm-registry",
        help=f"required with --execute; must equal {REGISTRY_ROOT}",
    )
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve(strict=True)

    try:
        _validate_controller(
            root,
            execute=arguments.execute,
            confirmation=arguments.confirm_registry,
        )
        try:
            preflight_bytes = (root / SANDBOX_PREFLIGHT_PATH).read_bytes()
        except OSError:
            raise RegistryPublishError(RegistryPublishErrorCode.SOURCE_PREFLIGHT_INVALID) from None
        preflight = _load_preflight(preflight_bytes)
        principal = _validate_identity()
        repository_resource_name = _validate_repository()
        local_images = tuple(
            _inspect_local_image(item.adapted_local_image_id) for item in preflight.task_evidence
        )
        if not arguments.execute:
            print(
                "registry_publish_preflight_passed=true images=8 "
                "execute=false external_mutations=0 paid_requests=0"
            )
            return 0

        publisher_commit = _git(root, "rev-parse", "HEAD")
        images = _publish_all(preflight, local_images)
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "manifest_version": MANIFEST_VERSION,
            "publisher_commit": publisher_commit,
            "collected_at": datetime.now(UTC),
            "source_preflight_relative_path": SANDBOX_PREFLIGHT_PATH.as_posix(),
            "source_preflight_raw_sha256": hashlib.sha256(preflight_bytes).hexdigest(),
            "source_preflight_content_sha256": preflight.content_sha256,
            "project_id": PROJECT_ID,
            "location": LOCATION,
            "repository": REPOSITORY,
            "repository_resource_name": repository_resource_name,
            "registry_host": REGISTRY_HOST,
            "registry_root": REGISTRY_ROOT,
            "repository_format": "DOCKER",
            "repository_mode": "STANDARD_REPOSITORY",
            "immutable_tags": True,
            "vulnerability_scanning_disabled": True,
            "active_principal": principal,
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
        manifest = RegistryManifest.model_validate(
            {**payload, "content_sha256": _canonical_sha256(payload)}
        )
        _write_private_create_only(
            root / RESULT_PATH,
            manifest.model_dump_json(indent=2).encode("utf-8") + b"\n",
        )
        print(
            "registry_publish_passed=true images=8 immutable=true "
            "credential=compute_engine_metadata paid_requests=0 "
            "formal_ready=false blockers=1"
        )
        return 0
    except RegistryPublishError as error:
        print(
            f"registry_publish_failed=true code={error.code.value} paid_requests=0",
            file=sys.stderr,
        )
        return 1


def _validate_controller(root: Path, *, execute: bool, confirmation: str | None) -> None:
    if execute and confirmation != REGISTRY_ROOT:
        raise RegistryPublishError(RegistryPublishErrorCode.INVALID_CONFIRMATION)
    if not execute and confirmation is not None:
        raise RegistryPublishError(RegistryPublishErrorCode.INVALID_CONFIRMATION)
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise RegistryPublishError(RegistryPublishErrorCode.HOST_PLATFORM_MISMATCH)
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise RegistryPublishError(RegistryPublishErrorCode.CONTROLLER_SOURCE_DIRTY)
    if (root / RESULT_PATH).exists():
        raise RegistryPublishError(RegistryPublishErrorCode.OUTPUT_CONFLICT)


def _load_preflight(content: bytes) -> SandboxMatrixPreflight:
    try:
        if hashlib.sha256(content).hexdigest() != EXPECTED_PREFLIGHT_RAW_SHA256:
            raise ValueError
        preflight = SandboxMatrixPreflight.model_validate_json(content)
        if (
            not preflight.compatibility_preflight_passed
            or preflight.adapter_verified_task_count != 8
            or preflight.python_mismatch_task_ids
            or preflight.paid_requests_made != 0
            or preflight.gold_visible_to_agent
        ):
            raise ValueError
        return preflight
    except (ValueError, TypeError):
        raise RegistryPublishError(RegistryPublishErrorCode.SOURCE_PREFLIGHT_INVALID) from None


def _validate_identity() -> str:
    metadata_identity = _metadata_text("instance/service-accounts/default/email")
    completed = _run_command(
        (
            "gcloud",
            "auth",
            "list",
            "--filter=status:ACTIVE",
            "--format=value(account)",
        ),
        check=False,
    )
    active_accounts = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if (
        metadata_identity != SERVICE_ACCOUNT
        or completed.returncode != 0
        or active_accounts != (SERVICE_ACCOUNT,)
    ):
        raise RegistryPublishError(RegistryPublishErrorCode.IDENTITY_MISMATCH)
    return SERVICE_ACCOUNT


def _validate_repository() -> str:
    completed = _run_command(
        (
            "gcloud",
            "artifacts",
            "repositories",
            "describe",
            REPOSITORY,
            f"--project={PROJECT_ID}",
            f"--location={LOCATION}",
            "--format=json",
        ),
        check=False,
    )
    try:
        record = cast(dict[str, object], json.loads(completed.stdout))
        docker_config = cast(dict[str, object], record["dockerConfig"])
        scanning = cast(dict[str, object], record["vulnerabilityScanningConfig"])
        resource_name = cast(str, record["name"])
        valid = (
            completed.returncode == 0
            and resource_name
            == f"projects/{PROJECT_ID}/locations/{LOCATION}/repositories/{REPOSITORY}"
            and record["format"] == "DOCKER"
            and record["mode"] == "STANDARD_REPOSITORY"
            and docker_config["immutableTags"] is True
            and scanning["enablementState"] == "SCANNING_DISABLED"
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        valid = False
        resource_name = ""
    if not valid:
        raise RegistryPublishError(RegistryPublishErrorCode.REPOSITORY_INVALID)
    return resource_name


def _inspect_local_image(
    image_reference: str,
    *,
    docker_config: str | None = None,
) -> dict[str, object]:
    env = _command_environment(docker_config)
    completed = _run_command(
        ("docker", "image", "inspect", image_reference),
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise RegistryPublishError(RegistryPublishErrorCode.LOCAL_IMAGE_UNAVAILABLE)
    try:
        records = cast(list[dict[str, object]], json.loads(completed.stdout))
        if len(records) != 1:
            raise ValueError
        record = records[0]
        image_id = cast(str, record["Id"])
        architecture = cast(str, record["Architecture"])
        operating_system = cast(str, record["Os"])
        size_bytes = cast(int, record["Size"])
        repository_digests = tuple(sorted(cast(list[str], record.get("RepoDigests") or [])))
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
            raise ValueError
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise RegistryPublishError(RegistryPublishErrorCode.LOCAL_IMAGE_UNAVAILABLE) from None
    if image_reference.startswith("sha256:") and image_id != image_reference:
        raise RegistryPublishError(RegistryPublishErrorCode.LOCAL_IMAGE_IDENTITY_MISMATCH)
    if architecture != "amd64" or operating_system != "linux":
        raise RegistryPublishError(RegistryPublishErrorCode.LOCAL_IMAGE_PLATFORM_MISMATCH)
    return {
        "image_id": image_id,
        "size_bytes": size_bytes,
        "repository_digests": repository_digests,
    }


def _publish_all(
    preflight: SandboxMatrixPreflight,
    local_images: tuple[dict[str, object], ...],
) -> tuple[RegistryImageEvidence, ...]:
    token = _metadata_token()
    with tempfile.TemporaryDirectory(prefix="repopilot-p7-docker-config-") as docker_config:
        env = _command_environment(docker_config)
        login = _run_command(
            (
                "docker",
                "login",
                "--username",
                "oauth2accesstoken",
                "--password-stdin",
                f"https://{REGISTRY_HOST}",
            ),
            input_text=token + "\n",
            check=False,
            env=env,
        )
        token = ""
        if login.returncode != 0:
            raise RegistryPublishError(RegistryPublishErrorCode.AUTHENTICATION_FAILED)
        images: list[RegistryImageEvidence] = []
        try:
            for source, local in zip(preflight.task_evidence, local_images, strict=True):
                print(f"registry_image_started={source.task_id}", flush=True)
                evidence = _publish_one(source, local, docker_config=docker_config)
                images.append(evidence)
                print(
                    f"registry_image_passed={source.task_id} "
                    f"outcome={evidence.publication_outcome} "
                    f"digest={evidence.registry_manifest_digest}",
                    flush=True,
                )
        finally:
            logout = _run_command(
                ("docker", "logout", REGISTRY_HOST),
                check=False,
                env=env,
            )
            if logout.returncode != 0:
                raise RegistryPublishError(RegistryPublishErrorCode.LOGOUT_FAILED)
        return tuple(images)


def _publish_one(
    source: SandboxTaskPreflightEvidence,
    local: dict[str, object],
    *,
    docker_config: str,
) -> RegistryImageEvidence:
    task_id = source.task_id
    source_policy_reference = source.adapted_policy_reference
    expected_image_id = source.adapted_local_image_id
    if local["image_id"] != expected_image_id:
        raise RegistryPublishError(RegistryPublishErrorCode.LOCAL_IMAGE_IDENTITY_MISMATCH)
    image_name = _safe_name(task_id)
    registry_image = f"{REGISTRY_ROOT}/{image_name}"
    tag_reference = f"{registry_image}:{IMMUTABLE_TAG}"
    existing = _pull_and_verify(
        tag_reference,
        registry_image=registry_image,
        expected_image_id=expected_image_id,
        docker_config=docker_config,
        missing_allowed=True,
    )
    if existing is not None:
        digest, remote_id = existing
        outcome: Literal["published", "already_present"] = "already_present"
    else:
        env = _command_environment(docker_config)
        tagged = _run_command(
            ("docker", "tag", expected_image_id, tag_reference),
            check=False,
            env=env,
        )
        if tagged.returncode != 0:
            raise RegistryPublishError(RegistryPublishErrorCode.PUBLISH_FAILED)
        pushed = _run_command(
            ("docker", "push", tag_reference),
            check=False,
            env=env,
        )
        if pushed.returncode != 0:
            raise RegistryPublishError(RegistryPublishErrorCode.PUBLISH_FAILED)
        verified = _pull_and_verify(
            tag_reference,
            registry_image=registry_image,
            expected_image_id=expected_image_id,
            docker_config=docker_config,
            missing_allowed=False,
        )
        if verified is None:
            raise RegistryPublishError(RegistryPublishErrorCode.REMOTE_PROBE_FAILED)
        digest, remote_id = verified
        outcome = "published"
    return RegistryImageEvidence(
        task_id=task_id,
        source_policy_reference=source_policy_reference,
        local_image_id=expected_image_id,
        local_size_bytes=cast(int, local["size_bytes"]),
        registry_tag_reference=tag_reference,
        registry_manifest_digest=digest,
        registry_digest_reference=f"{registry_image}@{digest}",
        remote_config_image_id=remote_id,
        publication_outcome=outcome,
        remote_pull_verified=True,
    )


def _pull_and_verify(
    tag_reference: str,
    *,
    registry_image: str,
    expected_image_id: str,
    docker_config: str,
    missing_allowed: bool,
) -> tuple[str, str] | None:
    env = _command_environment(docker_config)
    pulled = _run_command(
        ("docker", "pull", tag_reference),
        check=False,
        env=env,
    )
    if pulled.returncode != 0:
        combined = (pulled.stdout + "\n" + pulled.stderr).lower()
        if missing_allowed and any(marker in combined for marker in _NOT_FOUND_MARKERS):
            return None
        raise RegistryPublishError(RegistryPublishErrorCode.REMOTE_PROBE_FAILED)
    remote = _inspect_local_image(tag_reference, docker_config=docker_config)
    remote_id = cast(str, remote["image_id"])
    if remote_id != expected_image_id:
        raise RegistryPublishError(RegistryPublishErrorCode.REMOTE_TAG_CONFLICT)
    prefix = registry_image + "@"
    candidates = tuple(
        item.removeprefix(prefix)
        for item in cast(tuple[str, ...], remote["repository_digests"])
        if item.startswith(prefix)
    )
    if len(candidates) != 1 or _DIGEST.fullmatch(candidates[0]) is None:
        raise RegistryPublishError(RegistryPublishErrorCode.REMOTE_DIGEST_INVALID)
    return candidates[0], remote_id


def _metadata_text(relative_path: str) -> str:
    request = urllib.request.Request(
        f"{METADATA_ROOT}/{relative_path}",
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            content = cast(bytes, response.read(8_193))
            if len(content) > 8_192:
                raise ValueError
        value = content.decode("utf-8").strip()
        if not value:
            raise ValueError
        return value
    except (OSError, UnicodeError, ValueError, urllib.error.URLError):
        raise RegistryPublishError(RegistryPublishErrorCode.METADATA_UNAVAILABLE) from None


def _metadata_token() -> str:
    try:
        payload = cast(
            dict[str, object],
            json.loads(_metadata_text("instance/service-accounts/default/token")),
        )
        token = cast(str, payload["access_token"])
        expires_in = cast(int, payload["expires_in"])
        if (
            not isinstance(token, str)
            or not 20 <= len(token) <= 8_192
            or not isinstance(expires_in, int)
            or isinstance(expires_in, bool)
            or expires_in < 300
        ):
            raise ValueError
        return token
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise RegistryPublishError(RegistryPublishErrorCode.METADATA_UNAVAILABLE) from None


def _command_environment(docker_config: str | None = None) -> dict[str, str]:
    secret_fragments = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in _SECRET_ENV_KEYS
        and key != "DATABASE_URL"
        and not any(fragment in key.upper() for fragment in secret_fragments)
    }
    if docker_config is not None:
        environment["DOCKER_CONFIG"] = docker_config
    return environment


def _run_command(
    argv: tuple[str, ...],
    *,
    input_text: str | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            input=input_text,
            check=check,
            capture_output=True,
            text=True,
            timeout=3_600,
            env=env,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        code = RegistryPublishErrorCode.COMMAND_UNAVAILABLE
        raise RegistryPublishError(code) from error


if __name__ == "__main__":
    raise SystemExit(main())
