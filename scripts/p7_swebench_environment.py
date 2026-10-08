"""Collect Gold-free P7-S6 evidence from a pinned SWE-bench evaluator run.

This collector reads only aggregate evaluator reports and Docker image metadata.  It
does not expose solution patches, call a model Provider, or declare the images ready
for RepoPilot's stricter Sandbox adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from datetime import UTC, datetime
from enum import StrEnum, unique
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import AwareDatetime, Field, ValidationError, model_validator
from pydantic_core import to_jsonable_python

from repopilot.evals.dataset import FrozenDatasetManifest
from repopilot.schemas import ArtifactRef, GitSha, Sha256, StrictModel, TaskId

SCHEMA_VERSION: Literal[1] = 1
QUALIFICATION_VERSION: Literal["swebench-upstream-qualification-v1"] = (
    "swebench-upstream-qualification-v1"
)
HARNESS_REVISION: GitSha = "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
HARNESS_VERSION: Literal["v4.1.0"] = "v4.1.0"
PINNED_IMAGE_TAG: Literal["repopilot-p7-s6-v1"] = "repopilot-p7-s6-v1"
MANIFEST_PATH = Path("data/manifests/p7-v1/manifest.json")
BASELINE_REPORT_NAME = "environment-baseline.p7-s6-all-baseline-v1.json"
GOLD_REPORT_NAME = "gold.p7-s6-all-gold-v1.json"
EVIDENCE_ROOT = Path("results/p7-holdout/upstream-evaluator")
BASELINE_EVIDENCE_PATH = EVIDENCE_ROOT / "baseline-v1.json"
GOLD_EVIDENCE_PATH = EVIDENCE_ROOT / "gold-v1.json"
QUALIFICATION_PATH = Path("results/p7-holdout/upstream-qualification-v1.json")
EXPECTED_TASK_IDS = (
    "django__django-15902",
    "mwaskom__seaborn-3190",
    "pallets__flask-4992",
    "pydata__xarray-4094",
    "pydata__xarray-5131",
    "pytest-dev__pytest-11143",
    "pytest-dev__pytest-11148",
    "sympy__sympy-20590",
)
_MAX_REPORT_BYTES = 1_000_000
_DIGEST_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:~-]{0,439}@sha256:[0-9a-f]{64}$")


@unique
class EnvironmentQualificationErrorCode(StrEnum):
    """Stable reasons that block evidence collection or Sandbox adaptation."""

    INVALID_CONFIGURATION = "invalid_configuration"
    HARNESS_REVISION_MISMATCH = "harness_revision_mismatch"
    HARNESS_SOURCE_DIRTY = "harness_source_dirty"
    HOST_PLATFORM_MISMATCH = "host_platform_mismatch"
    REPORT_INVALID = "report_invalid"
    REPORT_OUTCOME_MISMATCH = "report_outcome_mismatch"
    IMAGE_UNAVAILABLE = "image_unavailable"
    IMAGE_UNPINNED = "image_unpinned"
    IMAGE_PLATFORM_MISMATCH = "image_platform_mismatch"
    BASE_COMMIT_UNREACHABLE = "base_commit_unreachable"
    TASK_PYTHON_MISMATCH = "task_python_mismatch"
    SANDBOX_ADAPTER_UNVERIFIED = "sandbox_adapter_unverified"
    OUTPUT_CONFLICT = "output_conflict"


class EnvironmentQualificationError(RuntimeError):
    """Secret-free collector failure."""

    def __init__(self, code: EnvironmentQualificationErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


class SweBenchAggregateReport(StrictModel):
    """Bounded schema emitted by the pinned SWE-bench v4.1.0 reporter."""

    total_instances: Literal[8]
    submitted_instances: Literal[8]
    completed_instances: Literal[8]
    resolved_instances: int = Field(ge=0, le=8, strict=True)
    unresolved_instances: int = Field(ge=0, le=8, strict=True)
    empty_patch_instances: Literal[0]
    error_instances: Literal[0]
    completed_ids: tuple[TaskId, ...]
    incomplete_ids: tuple[TaskId, ...]
    empty_patch_ids: tuple[TaskId, ...]
    submitted_ids: tuple[TaskId, ...]
    resolved_ids: tuple[TaskId, ...]
    unresolved_ids: tuple[TaskId, ...]
    error_ids: tuple[TaskId, ...]
    schema_version: Literal[2]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if (
            self.completed_ids != EXPECTED_TASK_IDS
            or self.submitted_ids != EXPECTED_TASK_IDS
            or self.incomplete_ids
            or self.empty_patch_ids
            or self.error_ids
            or self.resolved_instances + self.unresolved_instances != 8
        ):
            raise ValueError("SWE-bench report does not cover the exact frozen task set")
        return self


class UpstreamTaskImageEvidence(StrictModel):
    """One official evaluator image, before RepoPilot-specific adaptation."""

    task_id: TaskId
    base_commit: GitSha
    declared_python_version: str = Field(pattern=r"^3\.(?:9|10|11|12)(?:\.[0-9]+)?$")
    observed_python_version: str = Field(pattern=r"^3\.(?:9|10|11|12)(?:\.[0-9]+)?$")
    python_matches_declaration: bool
    local_image_tag: str
    source_image_reference: str
    platform: Literal["linux/amd64"]
    size_bytes: int = Field(gt=0, strict=True)
    base_commit_reachable: Literal[True]
    baseline_failed: Literal[True]
    gold_passed: Literal[True]
    gold_visible_to_agent: Literal[False] = False

    @model_validator(mode="after")
    def validate_image(self) -> Self:
        if _DIGEST_IMAGE.fullmatch(self.source_image_reference) is None:
            raise ValueError("upstream image evidence requires an immutable digest")
        matches = python_version_matches(
            self.declared_python_version,
            self.observed_python_version,
        )
        if matches != self.python_matches_declaration:
            raise ValueError("Python match decision does not match observed evidence")
        return self


class SweBenchUpstreamQualification(StrictModel):
    """Gold-free summary proving upstream evaluator behavior, not Agent readiness."""

    schema_version: Literal[1] = SCHEMA_VERSION
    qualification_version: Literal["swebench-upstream-qualification-v1"] = QUALIFICATION_VERSION
    collector_commit: GitSha
    harness_repository: Literal["https://github.com/SWE-bench/SWE-bench.git"] = (
        "https://github.com/SWE-bench/SWE-bench.git"
    )
    harness_version: Literal["v4.1.0"] = HARNESS_VERSION
    harness_revision: GitSha = HARNESS_REVISION
    host_platform: Literal["linux/amd64"]
    collected_at: AwareDatetime
    dataset_manifest_sha256: Sha256
    baseline_report_ref: ArtifactRef
    gold_report_ref: ArtifactRef
    task_images: tuple[UpstreamTaskImageEvidence, ...]
    upstream_baseline_failed_task_count: Literal[8]
    upstream_gold_passed_task_count: Literal[8]
    upstream_environment_qualified: Literal[True]
    repopilot_sandbox_adapter_ready: Literal[False]
    adapter_blockers: tuple[EnvironmentQualificationErrorCode, ...]
    paid_requests_made: Literal[0] = 0
    gold_visible_to_agent: Literal[False] = False
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_qualification(self) -> Self:
        if tuple(item.task_id for item in self.task_images) != EXPECTED_TASK_IDS:
            raise ValueError("qualification must cover the exact frozen task set")
        expected_blockers = [EnvironmentQualificationErrorCode.SANDBOX_ADAPTER_UNVERIFIED]
        if any(not item.python_matches_declaration for item in self.task_images):
            expected_blockers.append(EnvironmentQualificationErrorCode.TASK_PYTHON_MISMATCH)
        if self.adapter_blockers != tuple(expected_blockers):
            raise ValueError("adapter blockers do not match collected evidence")
        _validate_content_digest(self)
        return self


def python_version_matches(declared: str, observed: str) -> bool:
    """Match a minor declaration to a patch release, or require an exact patch."""

    declared_parts = declared.split(".")
    observed_parts = observed.split(".")
    if len(declared_parts) == 2:
        return observed_parts[:2] == declared_parts
    return observed_parts == declared_parts


def build_upstream_qualification(
    *,
    collector_commit: str,
    collected_at: datetime,
    dataset_manifest: FrozenDatasetManifest,
    baseline_ref: ArtifactRef,
    gold_ref: ArtifactRef,
    task_images: tuple[UpstreamTaskImageEvidence, ...],
) -> SweBenchUpstreamQualification:
    """Build the content-addressed, explicitly adapter-blocked evidence record."""

    blockers = [EnvironmentQualificationErrorCode.SANDBOX_ADAPTER_UNVERIFIED]
    if any(not item.python_matches_declaration for item in task_images):
        blockers.append(EnvironmentQualificationErrorCode.TASK_PYTHON_MISMATCH)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "qualification_version": QUALIFICATION_VERSION,
        "collector_commit": collector_commit,
        "harness_repository": "https://github.com/SWE-bench/SWE-bench.git",
        "harness_version": HARNESS_VERSION,
        "harness_revision": HARNESS_REVISION,
        "host_platform": "linux/amd64",
        "collected_at": collected_at,
        "dataset_manifest_sha256": dataset_manifest.content_sha256,
        "baseline_report_ref": baseline_ref,
        "gold_report_ref": gold_ref,
        "task_images": task_images,
        "upstream_baseline_failed_task_count": 8,
        "upstream_gold_passed_task_count": 8,
        "upstream_environment_qualified": True,
        "repopilot_sandbox_adapter_ready": False,
        "adapter_blockers": tuple(blockers),
        "paid_requests_made": 0,
        "gold_visible_to_agent": False,
    }
    return SweBenchUpstreamQualification.model_validate(
        {**payload, "content_sha256": _canonical_sha256(payload)}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--harness-root", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve(strict=True)
    harness_root = arguments.harness_root.resolve(strict=True)
    _validate_host_and_harness(harness_root)
    collector_commit = _git(root, "rev-parse", "HEAD")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise EnvironmentQualificationError(EnvironmentQualificationErrorCode.INVALID_CONFIGURATION)
    manifest = FrozenDatasetManifest.model_validate_json((root / MANIFEST_PATH).read_bytes())
    baseline_bytes, baseline = _load_report(harness_root / BASELINE_REPORT_NAME)
    gold_bytes, gold = _load_report(harness_root / GOLD_REPORT_NAME)
    _validate_report_outcomes(baseline, gold)
    cases = {item.case_id: item for item in manifest.external_cases}
    task_images = tuple(
        _inspect_task_image(
            task_id,
            base_commit=cases[task_id].base_commit,
            declared_python_version=cases[task_id].python_version,
        )
        for task_id in EXPECTED_TASK_IDS
    )
    output_paths = (
        root / BASELINE_EVIDENCE_PATH,
        root / GOLD_EVIDENCE_PATH,
        root / QUALIFICATION_PATH,
    )
    if any(path.exists() for path in output_paths):
        raise EnvironmentQualificationError(EnvironmentQualificationErrorCode.OUTPUT_CONFLICT)
    baseline_ref = _artifact_ref(BASELINE_EVIDENCE_PATH, baseline_bytes)
    gold_ref = _artifact_ref(GOLD_EVIDENCE_PATH, gold_bytes)
    qualification = build_upstream_qualification(
        collector_commit=collector_commit,
        collected_at=datetime.now(UTC),
        dataset_manifest=manifest,
        baseline_ref=baseline_ref,
        gold_ref=gold_ref,
        task_images=task_images,
    )
    _write_private_create_only(root / BASELINE_EVIDENCE_PATH, baseline_bytes)
    _write_private_create_only(root / GOLD_EVIDENCE_PATH, gold_bytes)
    _write_private_create_only(
        root / QUALIFICATION_PATH,
        qualification.model_dump_json(indent=2).encode("utf-8") + b"\n",
    )
    mismatches = sum(not item.python_matches_declaration for item in task_images)
    print(
        "upstream_qualified=true "
        f"tasks={len(task_images)} python_mismatches={mismatches} "
        "repopilot_adapter_ready=false paid_requests=0"
    )
    return 0


def _validate_host_and_harness(harness_root: Path) -> None:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise EnvironmentQualificationError(
            EnvironmentQualificationErrorCode.HOST_PLATFORM_MISMATCH
        )
    if _git(harness_root, "rev-parse", "HEAD") != HARNESS_REVISION:
        raise EnvironmentQualificationError(
            EnvironmentQualificationErrorCode.HARNESS_REVISION_MISMATCH
        )
    if _git(harness_root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise EnvironmentQualificationError(EnvironmentQualificationErrorCode.HARNESS_SOURCE_DIRTY)


def _load_report(path: Path) -> tuple[bytes, SweBenchAggregateReport]:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError
        content = path.read_bytes()
        if not content or len(content) > _MAX_REPORT_BYTES:
            raise OSError
        report = SweBenchAggregateReport.model_validate_json(content)
    except (OSError, UnicodeError, ValidationError):
        raise EnvironmentQualificationError(
            EnvironmentQualificationErrorCode.REPORT_INVALID
        ) from None
    return content, report


def _validate_report_outcomes(
    baseline: SweBenchAggregateReport,
    gold: SweBenchAggregateReport,
) -> None:
    if (
        baseline.resolved_instances != 0
        or baseline.unresolved_instances != 8
        or baseline.resolved_ids
        or baseline.unresolved_ids != EXPECTED_TASK_IDS
        or gold.resolved_instances != 8
        or gold.unresolved_instances != 0
        or gold.resolved_ids != EXPECTED_TASK_IDS
        or gold.unresolved_ids
    ):
        raise EnvironmentQualificationError(
            EnvironmentQualificationErrorCode.REPORT_OUTCOME_MISMATCH
        )


def _inspect_task_image(
    task_id: str,
    *,
    base_commit: str,
    declared_python_version: str,
) -> UpstreamTaskImageEvidence:
    image_base = f"swebench/sweb.eval.x86_64.{task_id.replace('__', '_1776_')}"
    local_tag = f"{image_base}:{PINNED_IMAGE_TAG}"
    try:
        inspected = json.loads(_run("docker", "image", "inspect", local_tag))
        record = cast(dict[str, object], cast(list[object], inspected)[0])
        repository_digests = cast(list[str], record.get("RepoDigests") or [])
        source_reference = next(
            item for item in repository_digests if item.startswith(f"{image_base}@sha256:")
        )
        architecture = cast(str, record["Architecture"])
        operating_system = cast(str, record["Os"])
        size_bytes = cast(int, record["Size"])
    except (IndexError, KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError):
        raise EnvironmentQualificationError(
            EnvironmentQualificationErrorCode.IMAGE_UNAVAILABLE
        ) from None
    if _DIGEST_IMAGE.fullmatch(source_reference) is None:
        raise EnvironmentQualificationError(EnvironmentQualificationErrorCode.IMAGE_UNPINNED)
    if architecture != "amd64" or operating_system != "linux":
        raise EnvironmentQualificationError(
            EnvironmentQualificationErrorCode.IMAGE_PLATFORM_MISMATCH
        )
    probe = (
        "source /opt/miniconda3/etc/profile.d/conda.sh && "
        "conda activate testbed && "
        "python -c 'import platform; print(platform.python_version())' && "
        f"git -C /testbed cat-file -e '{base_commit}^{{commit}}'"
    )
    try:
        observed_python = _run(
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "/bin/bash",
            local_tag,
            "-lc",
            probe,
        ).strip()
    except subprocess.CalledProcessError:
        raise EnvironmentQualificationError(
            EnvironmentQualificationErrorCode.BASE_COMMIT_UNREACHABLE
        ) from None
    return UpstreamTaskImageEvidence(
        task_id=task_id,
        base_commit=base_commit,
        declared_python_version=declared_python_version,
        observed_python_version=observed_python,
        python_matches_declaration=python_version_matches(
            declared_python_version,
            observed_python,
        ),
        local_image_tag=local_tag,
        source_image_reference=source_reference,
        platform="linux/amd64",
        size_bytes=size_bytes,
        base_commit_reachable=True,
        baseline_failed=True,
        gold_passed=True,
    )


def _artifact_ref(relative_path: Path, content: bytes) -> ArtifactRef:
    return ArtifactRef(
        relative_path=relative_path.as_posix(),
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        media_type="application/json",
    )


def _write_private_create_only(path: Path, content: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
    except OSError:
        raise EnvironmentQualificationError(
            EnvironmentQualificationErrorCode.OUTPUT_CONFLICT
        ) from None


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
        raise ValueError("qualification content digest does not match")


def _git(root: Path, *arguments: str) -> str:
    return _run("git", "-C", str(root), *arguments).strip()


def _run(*arguments: str) -> str:
    completed = subprocess.run(
        arguments,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed.stdout


if __name__ == "__main__":
    raise SystemExit(main())
