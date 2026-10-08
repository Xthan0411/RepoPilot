"""Frozen dataset contracts, provenance validation, and gold-isolated loading."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    Field,
    HttpUrl,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from repopilot.enums import (
    DatasetCaseKind,
    DatasetExclusionReason,
    DatasetManifestErrorCode,
    DatasetPreflightStatus,
    DatasetSelectionDecision,
    DatasetSplit,
)
from repopilot.evals.repair_fixtures import (
    RepairFixtureError,
    RepairFixtureManifest,
    load_repair_fixture_manifest,
)
from repopilot.schemas import (
    ArtifactRef,
    CommandSpec,
    EvaluationTaskSpec,
    GitSha,
    Identifier,
    NonEmptyText,
    RelativePath,
    Sha256,
    StrictModel,
    TaskId,
    TaskSpec,
    VersionIdentifier,
)

FROZEN_DATASET_SCHEMA_VERSION: Literal[1] = 1
CANDIDATE_DECISION_SCHEMA_VERSION: Literal[1] = 1
DATASET_PREFLIGHT_SCHEMA_VERSION: Literal[1] = 1
FROZEN_DATASET_LOADER_VERSION: Literal["dataset-loader-v1"] = "dataset-loader-v1"
_MAX_MANIFEST_BYTES = 1_000_000
_MAX_SOURCE_ROWS_BYTES = 2_000_000
_MAX_AUDIT_BYTES = 1_000_000
_MAX_PATCH_BYTES = 262_144
_DIGEST_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:~-]{0,439}@sha256:[0-9a-f]{64}$")
_REPOSITORY_NAME = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SOURCE_INSTANCE = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+$")
_PYTHON_VERSION = re.compile(r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?$")
_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)
RepositoryName = Annotated[str, StringConstraints(pattern=_REPOSITORY_NAME.pattern)]
SourceInstanceId = Annotated[str, StringConstraints(pattern=_SOURCE_INSTANCE.pattern)]
PythonVersion = Annotated[str, StringConstraints(pattern=_PYTHON_VERSION.pattern)]


_ERROR_MESSAGES = {
    DatasetManifestErrorCode.MANIFEST_UNAVAILABLE: "dataset manifest is unavailable",
    DatasetManifestErrorCode.MANIFEST_INVALID: "dataset manifest is invalid",
    DatasetManifestErrorCode.UNSUPPORTED_VERSION: "dataset schema version is unsupported",
    DatasetManifestErrorCode.INTEGRITY_MISMATCH: "dataset content integrity check failed",
    DatasetManifestErrorCode.IDENTITY_MISMATCH: "dataset source identity does not match",
    DatasetManifestErrorCode.DUPLICATE_CASE: "dataset contains duplicate case identity",
    DatasetManifestErrorCode.SOURCE_UNVERIFIED: "dataset source provenance is unverified",
    DatasetManifestErrorCode.ATTRIBUTION_MISSING: "dataset attribution is incomplete",
    DatasetManifestErrorCode.LICENSE_MISSING: "dataset repository license is unverified",
    DatasetManifestErrorCode.SPLIT_VIOLATION: "dataset split policy was violated",
    DatasetManifestErrorCode.GOLD_UNAVAILABLE: "dataset evaluator gold is unavailable",
    DatasetManifestErrorCode.PREFLIGHT_INVALID: "dataset preflight evidence is invalid",
    DatasetManifestErrorCode.PATH_UNSAFE: "dataset path is unsafe",
    DatasetManifestErrorCode.INPUT_IO_ERROR: "dataset input could not be read",
}


class DatasetManifestError(RuntimeError):
    """Safe typed dataset failure without paths, issue text, patches, or command output."""

    def __init__(self, code: DatasetManifestErrorCode) -> None:
        super().__init__(_ERROR_MESSAGES[code])
        self.code = code


class DatasetSourceAttribution(StrictModel):
    """Immutable upstream dataset identity and human-readable citation."""

    source_id: Identifier
    dataset_name: NonEmptyText
    dataset_url: HttpUrl
    dataset_revision: GitSha
    dataset_card_sha256: Sha256
    project_url: HttpUrl
    project_license: VersionIdentifier
    project_license_url: HttpUrl
    citation: NonEmptyText


class RepositoryLicenseEvidence(StrictModel):
    """License evidence read at the exact task base commit."""

    spdx_identifier: VersionIdentifier
    license_path: RelativePath
    license_blob_sha: GitSha
    license_url: HttpUrl
    copyright_notice: NonEmptyText


class SweBenchSourceRow(StrictModel):
    """Exact selected row shape from the immutable SWE-bench Lite snapshot."""

    repo: RepositoryName
    instance_id: SourceInstanceId
    base_commit: GitSha
    patch: str = Field(min_length=1, max_length=_MAX_PATCH_BYTES, repr=False)
    test_patch: str = Field(min_length=1, max_length=_MAX_PATCH_BYTES, repr=False)
    problem_statement: NonEmptyText
    hints_text: str
    created_at: AwareDatetime
    version: VersionIdentifier
    fail_to_pass_json: str = Field(alias="FAIL_TO_PASS", min_length=2)
    pass_to_pass_json: str = Field(alias="PASS_TO_PASS", min_length=2, repr=False)
    environment_setup_commit: GitSha

    @model_validator(mode="after")
    def validate_test_lists(self) -> Self:
        _parse_test_list(self.fail_to_pass_json)
        _parse_test_list(self.pass_to_pass_json, allow_empty=True)
        return self


class FrozenExternalCase(StrictModel):
    """One selected public Issue with frozen execution and attribution metadata."""

    case_id: TaskId
    source_instance_id: SourceInstanceId
    source_row_sha256: Sha256
    repository: RepositoryName
    repository_url: HttpUrl
    pull_request_url: HttpUrl
    issue_urls: tuple[HttpUrl, ...] = Field(min_length=1, max_length=4)
    base_commit: GitSha
    environment_setup_commit: GitSha
    upstream_version: VersionIdentifier
    python_version: PythonVersion
    issue_title: NonEmptyText
    setup_commands: tuple[CommandSpec, ...] = Field(min_length=1, max_length=8)
    test_commands: tuple[CommandSpec, ...] = Field(min_length=1, max_length=4)
    timeout_seconds: int = Field(gt=0, le=3600, strict=True)
    split: Literal[DatasetSplit.HOLDOUT] = DatasetSplit.HOLDOUT
    gold_files: tuple[RelativePath, ...] = Field(min_length=1, max_length=20)
    fail_to_pass: tuple[NonEmptyText, ...] = Field(min_length=1, max_length=32)
    repository_license: RepositoryLicenseEvidence
    preflight_status: DatasetPreflightStatus

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        expected_repository_url = f"https://github.com/{self.repository}.git"
        expected_pr_url = f"https://github.com/{self.repository}/pull/"
        if str(self.repository_url).rstrip("/") != expected_repository_url:
            raise ValueError("repository URL must be the exact public Git clone URL")
        if not str(self.pull_request_url).startswith(expected_pr_url):
            raise ValueError("pull request URL must belong to the frozen repository")
        issue_prefix = f"https://github.com/{self.repository}/issues/"
        if any(not str(item).startswith(issue_prefix) for item in self.issue_urls):
            raise ValueError("issue URLs must belong to the frozen repository")
        if self.gold_files != tuple(sorted(self.gold_files)) or len(set(self.gold_files)) != len(
            self.gold_files
        ):
            raise ValueError("gold files must be sorted and unique")
        if self.fail_to_pass != tuple(sorted(self.fail_to_pass)) or len(
            set(self.fail_to_pass)
        ) != len(self.fail_to_pass):
            raise ValueError("FAIL_TO_PASS tests must be sorted and unique")
        for command in self.setup_commands:
            _validate_setup_command(command)
        for command in self.test_commands:
            _validate_test_command(command)
        for command in (*self.setup_commands, *self.test_commands):
            if command.cwd is not None:
                raise ValueError("dataset commands must run at the repository root")
        return self

    def agent_task(self, issue_body: str, repository_path: Path | None = None) -> TaskSpec:
        """Project only Agent-visible fields, optionally into a prepared local checkout."""

        if repository_path is None:
            return TaskSpec(
                task_id=self.case_id,
                repository_url=self.repository_url,
                base_commit=self.base_commit,
                issue_title=self.issue_title,
                issue_body=issue_body,
                setup_commands=self.setup_commands,
                test_commands=self.test_commands,
                timeout_seconds=self.timeout_seconds,
            )
        return TaskSpec(
            task_id=self.case_id,
            repository_path=str(repository_path),
            base_commit=self.base_commit,
            issue_title=self.issue_title,
            issue_body=issue_body,
            setup_commands=self.setup_commands,
            test_commands=self.test_commands,
            timeout_seconds=self.timeout_seconds,
        )


class FrozenDatasetManifest(StrictModel):
    """Versioned dataset selection with source, fixture, and public-Issue provenance."""

    schema_version: Literal[1] = FROZEN_DATASET_SCHEMA_VERSION
    loader_version: Literal["dataset-loader-v1"] = FROZEN_DATASET_LOADER_VERSION
    dataset_id: Identifier
    created_at: AwareDatetime
    source: DatasetSourceAttribution
    source_rows_ref: ArtifactRef
    fixture_manifest_ref: ArtifactRef
    fixture_task_ids: tuple[TaskId, ...] = Field(min_length=2, max_length=3)
    external_cases: tuple[FrozenExternalCase, ...] = Field(min_length=8, max_length=15)
    dev_task_count: int = Field(ge=2, le=3, strict=True)
    holdout_task_count: int = Field(ge=8, le=15, strict=True)
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.fixture_task_ids != tuple(sorted(self.fixture_task_ids)) or len(
            set(self.fixture_task_ids)
        ) != len(self.fixture_task_ids):
            raise ValueError("fixture task IDs must be sorted and unique")
        case_ids = tuple(item.case_id for item in self.external_cases)
        source_ids = tuple(item.source_instance_id for item in self.external_cases)
        if case_ids != tuple(sorted(case_ids)):
            raise ValueError("external dataset cases must be sorted")
        if len(set(case_ids)) != len(case_ids) or len(set(source_ids)) != len(source_ids):
            raise ValueError("external dataset case identities must be unique")
        if set(case_ids) & set(self.fixture_task_ids):
            raise ValueError("fixture and external task identities must be disjoint")
        if self.dev_task_count != len(self.fixture_task_ids):
            raise ValueError("development task count does not match fixture selection")
        if self.holdout_task_count != len(self.external_cases):
            raise ValueError("holdout task count does not match public Issue selection")
        if self.source_rows_ref.media_type != "application/json":
            raise ValueError("source-row snapshot must be JSON")
        if self.fixture_manifest_ref.media_type != "application/json":
            raise ValueError("fixture manifest must be JSON")
        payload = self.model_dump(mode="python", exclude={"content_sha256"})
        if dataset_content_sha256(payload) != self.content_sha256:
            raise ValueError("dataset manifest digest does not match")
        return self


class CandidateDecision(StrictModel):
    """One retained inclusion/exclusion decision from public-task review."""

    source_instance_id: SourceInstanceId
    repository: RepositoryName
    pull_request_url: HttpUrl
    base_commit: GitSha
    decision: DatasetSelectionDecision
    reasons: tuple[DatasetExclusionReason, ...] = Field(max_length=4)
    rationale: NonEmptyText

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.decision is DatasetSelectionDecision.INCLUDED and self.reasons:
            raise ValueError("included candidates cannot carry exclusion reasons")
        if self.decision is DatasetSelectionDecision.EXCLUDED and not self.reasons:
            raise ValueError("excluded candidates require a stable reason")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("candidate exclusion reasons must be unique")
        return self


class CandidateDecisionLog(StrictModel):
    """Integrity-protected candidate selection and exclusion audit."""

    schema_version: Literal[1] = CANDIDATE_DECISION_SCHEMA_VERSION
    dataset_id: Identifier
    source_revision: GitSha
    decisions: tuple[CandidateDecision, ...] = Field(min_length=8, max_length=100)
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_log(self) -> Self:
        identities = tuple(item.source_instance_id for item in self.decisions)
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(identities):
            raise ValueError("candidate decisions must be sorted and unique")
        if not any(item.decision is DatasetSelectionDecision.EXCLUDED for item in self.decisions):
            raise ValueError("candidate audit must retain at least one exclusion")
        payload = self.model_dump(mode="python", exclude={"content_sha256"})
        if dataset_content_sha256(payload) != self.content_sha256:
            raise ValueError("candidate decision digest does not match")
        return self


class DatasetPreflightEvidence(StrictModel):
    """Bounded claim about metadata-only or baseline-fail/gold-pass validation."""

    case_id: TaskId
    kind: DatasetCaseKind
    status: DatasetPreflightStatus
    checked_at: AwareDatetime
    python_version: PythonVersion
    sandbox_image_reference: str | None = Field(default=None, max_length=512)
    setup_log_ref: ArtifactRef | None = None
    baseline_log_ref: ArtifactRef | None = None
    gold_log_ref: ArtifactRef | None = None
    baseline_exit_code: int | None = Field(default=None, ge=0, le=255, strict=True)
    gold_exit_code: int | None = Field(default=None, ge=0, le=255, strict=True)
    observed_failure_markers: tuple[NonEmptyText, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        runtime_values = (
            self.sandbox_image_reference,
            self.setup_log_ref,
            self.baseline_log_ref,
            self.gold_log_ref,
            self.baseline_exit_code,
            self.gold_exit_code,
        )
        if self.status is DatasetPreflightStatus.METADATA_VERIFIED:
            if any(item is not None for item in runtime_values) or self.observed_failure_markers:
                raise ValueError("metadata-only preflight cannot claim execution evidence")
        else:
            if any(item is None for item in runtime_values):
                raise ValueError("gold-verified preflight requires complete runtime evidence")
            if (
                self.sandbox_image_reference is None
                or _DIGEST_IMAGE.fullmatch(self.sandbox_image_reference) is None
            ):
                raise ValueError("preflight Sandbox image must use an immutable digest")
            if self.baseline_exit_code == 0 or self.gold_exit_code != 0:
                raise ValueError("preflight must prove baseline failure and gold success")
            if not self.observed_failure_markers:
                raise ValueError("preflight must retain a bounded failure marker")
        return self


class DatasetPreflightReport(StrictModel):
    """Integrity-protected evidence depth for every frozen task."""

    schema_version: Literal[1] = DATASET_PREFLIGHT_SCHEMA_VERSION
    dataset_id: Identifier
    generated_at: AwareDatetime
    entries: tuple[DatasetPreflightEvidence, ...] = Field(min_length=10, max_length=18)
    content_sha256: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        case_ids = tuple(item.case_id for item in self.entries)
        if case_ids != tuple(sorted(case_ids)) or len(set(case_ids)) != len(case_ids):
            raise ValueError("preflight entries must be sorted and unique")
        if not any(
            item.kind is DatasetCaseKind.GITHUB_ISSUE
            and item.status is DatasetPreflightStatus.BASELINE_FAILED_GOLD_PASSED
            for item in self.entries
        ):
            raise ValueError("at least one public Issue requires gold runtime validation")
        payload = self.model_dump(mode="python", exclude={"content_sha256"})
        if dataset_content_sha256(payload) != self.content_sha256:
            raise ValueError("dataset preflight digest does not match")
        return self


@dataclass(frozen=True)
class FrozenDatasetBundle:
    """Validated evaluator view; Agent consumers receive only each inner TaskSpec."""

    manifest: FrozenDatasetManifest
    fixture_manifest: RepairFixtureManifest
    external_evaluation_tasks: tuple[EvaluationTaskSpec, ...]

    def evaluation_tasks(self) -> tuple[EvaluationTaskSpec, ...]:
        """Return all evaluator wrappers in stable task-ID order."""

        fixture_tasks = tuple(case.evaluation_task for case in self.fixture_manifest.cases)
        return tuple(
            sorted(
                (*fixture_tasks, *self.external_evaluation_tasks),
                key=lambda item: item.task.task_id,
            )
        )

    def agent_tasks(self) -> tuple[TaskSpec, ...]:
        """Return the gold-free projection used by runners and Providers."""

        return tuple(item.task for item in self.evaluation_tasks())


def load_frozen_dataset(
    manifest_path: Path,
    *,
    repository_root: Path,
) -> FrozenDatasetBundle:
    """Load a frozen manifest, all source/gold bytes, and the checked-in dev fixtures."""

    root = _validated_root(repository_root)
    manifest_bytes = _read_contained_file(root, manifest_path, _MAX_MANIFEST_BYTES)
    manifest = _parse_versioned_model(
        manifest_bytes,
        FrozenDatasetManifest,
        FROZEN_DATASET_SCHEMA_VERSION,
    )
    source_bytes = _read_artifact(root, manifest.source_rows_ref, _MAX_SOURCE_ROWS_BYTES)
    try:
        raw_rows = json.loads(source_bytes)
        rows = TypeAdapter(tuple[SweBenchSourceRow, ...]).validate_python(raw_rows)
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError, TypeError):
        raise DatasetManifestError(DatasetManifestErrorCode.SOURCE_UNVERIFIED) from None
    if tuple(item.instance_id for item in rows) != tuple(
        sorted(item.instance_id for item in rows)
    ) or len({item.instance_id for item in rows}) != len(rows):
        raise DatasetManifestError(DatasetManifestErrorCode.SOURCE_UNVERIFIED)
    by_id = {item.instance_id: item for item in rows}
    if set(by_id) != {item.source_instance_id for item in manifest.external_cases}:
        raise DatasetManifestError(DatasetManifestErrorCode.IDENTITY_MISMATCH)

    fixture_path = _artifact_path(root, manifest.fixture_manifest_ref)
    _read_artifact(root, manifest.fixture_manifest_ref, _MAX_MANIFEST_BYTES)
    try:
        fixture_manifest = load_repair_fixture_manifest(fixture_path)
    except RepairFixtureError:
        raise DatasetManifestError(DatasetManifestErrorCode.SOURCE_UNVERIFIED) from None
    fixture_ids = tuple(case.evaluation_task.task.task_id for case in fixture_manifest.cases)
    if fixture_ids != manifest.fixture_task_ids:
        raise DatasetManifestError(DatasetManifestErrorCode.IDENTITY_MISMATCH)

    evaluation_tasks: list[EvaluationTaskSpec] = []
    for case in manifest.external_cases:
        row = by_id[case.source_instance_id]
        _validate_source_case(case, row)
        evaluation_tasks.append(
            EvaluationTaskSpec(
                task=case.agent_task(row.problem_statement),
                expected_patch=row.patch,
                gold_files=case.gold_files,
                split=DatasetSplit.HOLDOUT,
            )
        )
    return FrozenDatasetBundle(
        manifest=manifest,
        fixture_manifest=fixture_manifest,
        external_evaluation_tasks=tuple(evaluation_tasks),
    )


def load_candidate_decision_log(
    path: Path,
    *,
    repository_root: Path,
    manifest: FrozenDatasetManifest,
) -> CandidateDecisionLog:
    """Load and cross-check the complete inclusion/exclusion audit."""

    root = _validated_root(repository_root)
    payload = _read_contained_file(root, path, _MAX_AUDIT_BYTES)
    log = _parse_versioned_model(
        payload,
        CandidateDecisionLog,
        CANDIDATE_DECISION_SCHEMA_VERSION,
    )
    if log.dataset_id != manifest.dataset_id or log.source_revision != (
        manifest.source.dataset_revision
    ):
        raise DatasetManifestError(DatasetManifestErrorCode.IDENTITY_MISMATCH)
    included = {
        item.source_instance_id
        for item in log.decisions
        if item.decision is DatasetSelectionDecision.INCLUDED
    }
    expected = {item.source_instance_id for item in manifest.external_cases}
    if included != expected:
        raise DatasetManifestError(DatasetManifestErrorCode.IDENTITY_MISMATCH)
    return log


def load_dataset_preflight_report(
    path: Path,
    *,
    repository_root: Path,
    manifest: FrozenDatasetManifest,
) -> DatasetPreflightReport:
    """Load preflight claims and verify every referenced log plus case identity."""

    root = _validated_root(repository_root)
    payload = _read_contained_file(root, path, _MAX_AUDIT_BYTES)
    report = _parse_versioned_model(
        payload,
        DatasetPreflightReport,
        DATASET_PREFLIGHT_SCHEMA_VERSION,
    )
    if report.dataset_id != manifest.dataset_id:
        raise DatasetManifestError(DatasetManifestErrorCode.IDENTITY_MISMATCH)
    expected_cases = set(manifest.fixture_task_ids) | {
        item.case_id for item in manifest.external_cases
    }
    if {item.case_id for item in report.entries} != expected_cases:
        raise DatasetManifestError(DatasetManifestErrorCode.IDENTITY_MISMATCH)
    external_status = {item.case_id: item.preflight_status for item in manifest.external_cases}
    for entry in report.entries:
        if entry.case_id in external_status:
            if (
                entry.kind is not DatasetCaseKind.GITHUB_ISSUE
                or entry.status is not external_status[entry.case_id]
            ):
                raise DatasetManifestError(DatasetManifestErrorCode.PREFLIGHT_INVALID)
        elif entry.kind is not DatasetCaseKind.FIXTURE:
            raise DatasetManifestError(DatasetManifestErrorCode.PREFLIGHT_INVALID)
        for reference in (
            entry.setup_log_ref,
            entry.baseline_log_ref,
            entry.gold_log_ref,
        ):
            if reference is not None:
                _read_artifact(root, reference, _MAX_AUDIT_BYTES)
    return report


def build_dataset_artifact_ref(
    repository_root: Path,
    relative_path: str,
    media_type: str,
) -> ArtifactRef:
    """Build an ArtifactRef from actual contained bytes for freeze tooling."""

    root = _validated_root(repository_root)
    path = _contained_path(root, Path(relative_path))
    content = _read_regular_file(path, _MAX_SOURCE_ROWS_BYTES)
    return ArtifactRef(
        relative_path=relative_path,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        media_type=media_type,
    )


def dataset_content_sha256(value: object) -> str:
    """Hash JSON-like content with the same canonical form used by all P7-S2 files."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def source_row_sha256(row: SweBenchSourceRow) -> str:
    """Hash one exact upstream row, including evaluator-only gold."""

    return dataset_content_sha256(row.model_dump(mode="python", by_alias=True))


def _validate_source_case(case: FrozenExternalCase, row: SweBenchSourceRow) -> None:
    if (
        row.instance_id != case.source_instance_id
        or row.repo != case.repository
        or row.base_commit != case.base_commit
        or row.environment_setup_commit != case.environment_setup_commit
        or row.version != case.upstream_version
        or source_row_sha256(row) != case.source_row_sha256
    ):
        raise DatasetManifestError(DatasetManifestErrorCode.IDENTITY_MISMATCH)
    if _declared_patch_paths(row.patch) != case.gold_files:
        raise DatasetManifestError(DatasetManifestErrorCode.GOLD_UNAVAILABLE)
    _declared_patch_paths(row.test_patch, allow_new_files=True)
    if tuple(sorted(_parse_test_list(row.fail_to_pass_json))) != case.fail_to_pass:
        raise DatasetManifestError(DatasetManifestErrorCode.IDENTITY_MISMATCH)


def _declared_patch_paths(patch: str, *, allow_new_files: bool = False) -> tuple[str, ...]:
    if len(patch.encode("utf-8")) > _MAX_PATCH_BYTES or "\x00" in patch or "@@" not in patch:
        raise DatasetManifestError(DatasetManifestErrorCode.GOLD_UNAVAILABLE)
    matches = _DIFF_HEADER.findall(patch)
    if not matches:
        raise DatasetManifestError(DatasetManifestErrorCode.GOLD_UNAVAILABLE)
    paths: list[str] = []
    for before, after in matches:
        if before != after:
            raise DatasetManifestError(DatasetManifestErrorCode.GOLD_UNAVAILABLE)
        pure_path = PurePosixPath(before)
        if (
            pure_path.is_absolute()
            or not pure_path.parts
            or ".." in pure_path.parts
            or "\\" in before
            or pure_path.as_posix() != before
        ):
            raise DatasetManifestError(DatasetManifestErrorCode.GOLD_UNAVAILABLE)
        if not allow_new_files and f"--- a/{before}\n" not in patch:
            raise DatasetManifestError(DatasetManifestErrorCode.GOLD_UNAVAILABLE)
        if f"+++ b/{after}\n" not in patch:
            raise DatasetManifestError(DatasetManifestErrorCode.GOLD_UNAVAILABLE)
        paths.append(before)
    if len(set(paths)) != len(paths):
        raise DatasetManifestError(DatasetManifestErrorCode.GOLD_UNAVAILABLE)
    return tuple(sorted(paths))


def _parse_test_list(value: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        raise ValueError("SWE-bench test list must be valid JSON") from None
    if (
        not isinstance(parsed, list)
        or (not parsed and not allow_empty)
        or len(parsed) > 10_000
        or any(
            not isinstance(item, str) or not item.strip() or "\x00" in item or len(item) > 4096
            for item in parsed
        )
        or len(set(parsed)) != len(parsed)
    ):
        raise ValueError("SWE-bench test list is invalid")
    return tuple(parsed)


def _validate_setup_command(command: CommandSpec) -> None:
    argv = command.argv
    if (
        len(argv) < 4
        or argv[:3]
        not in {
            ("python", "-m", "pip"),
            ("python3", "-m", "pip"),
        }
        or argv[3] != "install"
    ):
        raise ValueError("dataset setup commands must use python -m pip install")
    denied_options = {
        "--extra-index-url",
        "--find-links",
        "--index-url",
        "--trusted-host",
    }
    for argument in argv[4:]:
        lowered = argument.lower()
        if (
            lowered.startswith(("git+", "http://", "https://"))
            or lowered in denied_options
            or any(lowered.startswith(f"{option}=") for option in denied_options)
        ):
            raise ValueError("dataset setup commands cannot select a remote source")
    for flag in ("-r", "--requirement"):
        if flag in argv:
            index = argv.index(flag)
            if index + 1 >= len(argv):
                raise ValueError("dataset requirement flag requires a path")
            requirement = PurePosixPath(argv[index + 1])
            if requirement.is_absolute() or ".." in requirement.parts or "\\" in argv[index + 1]:
                raise ValueError("dataset requirement path must stay in the repository")


def _validate_test_command(command: CommandSpec) -> None:
    argv = command.argv
    if argv[0] == "pytest":
        return
    if len(argv) >= 3 and argv[:3] in {
        ("python", "-m", "pytest"),
        ("python3", "-m", "pytest"),
    }:
        return
    if len(argv) >= 2 and argv[0] in {"python", "python3"} and argv[1] == ("tests/runtests.py"):
        return
    raise ValueError("dataset tests must use pytest or the fixed Django test runner")


def _parse_versioned_model[ModelT: StrictModel](
    payload: bytes,
    model_type: type[ModelT],
    expected_version: int,
) -> ModelT:
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise DatasetManifestError(DatasetManifestErrorCode.MANIFEST_INVALID) from None
    if not isinstance(raw, dict):
        raise DatasetManifestError(DatasetManifestErrorCode.MANIFEST_INVALID)
    if raw.get("schema_version") != expected_version:
        raise DatasetManifestError(DatasetManifestErrorCode.UNSUPPORTED_VERSION)
    try:
        return model_type.model_validate(raw)
    except ValidationError:
        raise DatasetManifestError(DatasetManifestErrorCode.MANIFEST_INVALID) from None


def _validated_root(root: Path) -> Path:
    if not root.is_absolute() or root.is_symlink():
        raise DatasetManifestError(DatasetManifestErrorCode.PATH_UNSAFE)
    try:
        resolved = root.resolve(strict=True)
    except OSError:
        raise DatasetManifestError(DatasetManifestErrorCode.PATH_UNSAFE) from None
    if resolved != root or not root.is_dir():
        raise DatasetManifestError(DatasetManifestErrorCode.PATH_UNSAFE)
    return root


def _read_contained_file(root: Path, path: Path, max_bytes: int) -> bytes:
    candidate = path if path.is_absolute() else root / path
    contained = _contained_path(root, candidate)
    return _read_regular_file(contained, max_bytes)


def _artifact_path(root: Path, reference: ArtifactRef) -> Path:
    return _contained_path(root, root / Path(reference.relative_path))


def _read_artifact(root: Path, reference: ArtifactRef, max_bytes: int) -> bytes:
    content = _read_regular_file(_artifact_path(root, reference), max_bytes)
    if len(content) != reference.size_bytes or hashlib.sha256(content).hexdigest() != (
        reference.sha256
    ):
        raise DatasetManifestError(DatasetManifestErrorCode.INTEGRITY_MISMATCH)
    return content


def _contained_path(root: Path, candidate: Path) -> Path:
    try:
        if candidate.is_symlink():
            raise OSError
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        raise DatasetManifestError(DatasetManifestErrorCode.PATH_UNSAFE) from None
    return resolved


def _read_regular_file(path: Path, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError
            content = stream.read(max_bytes + 1)
    except OSError:
        raise DatasetManifestError(DatasetManifestErrorCode.INPUT_IO_ERROR) from None
    if len(content) > max_bytes:
        raise DatasetManifestError(DatasetManifestErrorCode.INPUT_IO_ERROR)
    return content


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, HttpUrl):
        return str(value)
    raise TypeError(f"cannot canonicalize {type(value).__name__}")
