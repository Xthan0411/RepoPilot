"""Deterministic, gold-isolated repair fixtures for the P5 Agent loop."""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, ValidationError, model_validator

from repopilot.enums import DatasetSplit
from repopilot.schemas import (
    EvaluationTaskSpec,
    GitSha,
    Identifier,
    NonEmptyText,
    RelativePath,
    Sha256,
    StrictModel,
    TaskSpec,
)

REPAIR_FIXTURE_SCHEMA_VERSION: Literal[1] = 1
REPAIR_FIXTURE_TEST_ARGV = (
    "python",
    "-m",
    "unittest",
    "discover",
    "-s",
    "checks",
    "-p",
    "check_*.py",
    "-q",
)
_MAX_MANIFEST_BYTES = 1_000_000
_MAX_REPOSITORY_FILES = 100
_MAX_REPOSITORY_BYTES = 2_000_000
_MAX_PATCH_BYTES = 100_000
_DENIED_REPOSITORY_NAMES = frozenset(
    {".aws", ".codex", ".config", ".docker", ".env", ".git", ".ssh", "artifacts"}
)
_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)


class RepairFixtureErrorCode(StrEnum):
    """Stable failures for fixture loading and snapshot verification."""

    MANIFEST_UNAVAILABLE = "manifest_unavailable"
    MANIFEST_INVALID = "manifest_invalid"
    PATH_VIOLATION = "path_violation"
    REPOSITORY_UNAVAILABLE = "repository_unavailable"
    SNAPSHOT_MISMATCH = "snapshot_mismatch"
    BASE_COMMIT_MISMATCH = "base_commit_mismatch"


_ERROR_MESSAGES = {
    RepairFixtureErrorCode.MANIFEST_UNAVAILABLE: "repair fixture manifest is unavailable",
    RepairFixtureErrorCode.MANIFEST_INVALID: "repair fixture manifest is invalid",
    RepairFixtureErrorCode.PATH_VIOLATION: "repair fixture repository path is invalid",
    RepairFixtureErrorCode.REPOSITORY_UNAVAILABLE: "repair fixture repository is unavailable",
    RepairFixtureErrorCode.SNAPSHOT_MISMATCH: "repair fixture snapshot does not match",
    RepairFixtureErrorCode.BASE_COMMIT_MISMATCH: (
        "repair fixture base commit does not match its snapshot"
    ),
}


class RepairFixtureError(RuntimeError):
    """Safe typed error that never includes manifest, path, patch, or source content."""

    def __init__(self, code: RepairFixtureErrorCode) -> None:
        super().__init__(_ERROR_MESSAGES[code])
        self.code = code


class RepositoryFixtureSnapshot(StrictModel):
    """Canonical content identity for one checked-in baseline repository."""

    base_commit: GitSha
    sha256: Sha256
    file_count: int = Field(gt=0, le=_MAX_REPOSITORY_FILES)
    byte_count: int = Field(gt=0, le=_MAX_REPOSITORY_BYTES)


class RepairFixtureCase(StrictModel):
    """Evaluator-owned task, expected failure, and repository snapshot."""

    evaluation_task: EvaluationTaskSpec
    repository_sha256: Sha256
    expected_base_exit_code: int = Field(default=1, ge=1, le=255, strict=True)
    expected_failure_markers: tuple[NonEmptyText, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        task = self.evaluation_task.task
        if (
            task.repository_url is not None
            or task.repository_path is None
            or PurePosixPath(task.repository_path).is_absolute()
            or ".." in PurePosixPath(task.repository_path).parts
        ):
            raise ValueError("repair fixture repositories must use contained relative paths")
        if (
            task.setup_commands
            or len(task.test_commands) != 1
            or task.test_commands[0].argv != REPAIR_FIXTURE_TEST_ARGV
            or task.test_commands[0].cwd is not None
            or task.test_commands[0].timeout_seconds != 30
            or task.timeout_seconds != 60
        ):
            raise ValueError("repair fixture commands must preserve the hermetic test contract")
        if self.evaluation_task.split is not DatasetSplit.DEV:
            raise ValueError("P5 repair fixtures are development tasks, not holdout evidence")
        if tuple(sorted(self.evaluation_task.gold_files)) != self.evaluation_task.gold_files:
            raise ValueError("repair fixture gold files must be sorted")
        if len(set(self.expected_failure_markers)) != len(self.expected_failure_markers):
            raise ValueError("repair fixture failure markers must be unique")
        patch = self.evaluation_task.expected_patch
        if len(patch.encode("utf-8")) > _MAX_PATCH_BYTES:
            raise ValueError("repair fixture patch exceeds its size limit")
        if _declared_patch_paths(patch) != self.evaluation_task.gold_files:
            raise ValueError("repair fixture patch paths must exactly match gold files")
        return self


class RepairFixtureManifest(StrictModel):
    """Versioned suite whose gold wrapper is never passed to the Agent."""

    schema_version: Literal[1] = REPAIR_FIXTURE_SCHEMA_VERSION
    suite_id: Identifier
    cases: tuple[RepairFixtureCase, ...] = Field(min_length=2, max_length=20)

    @model_validator(mode="after")
    def validate_cases(self) -> Self:
        task_ids = tuple(case.evaluation_task.task.task_id for case in self.cases)
        repository_paths = tuple(case.evaluation_task.task.repository_path for case in self.cases)
        if task_ids != tuple(sorted(task_ids)) or len(set(task_ids)) != len(task_ids):
            raise ValueError("repair fixture task IDs must be sorted and unique")
        if len(set(repository_paths)) != len(repository_paths):
            raise ValueError("repair fixture repository paths must be unique")
        return self

    def agent_tasks(self) -> tuple[TaskSpec, ...]:
        """Return only the strict Agent-visible inner tasks."""

        return tuple(case.evaluation_task.task for case in self.cases)


def load_repair_fixture_manifest(path: Path) -> RepairFixtureManifest:
    """Load, contain, hash, and absolutize one evaluator-owned fixture suite."""

    try:
        if path.is_symlink():
            raise OSError
        payload = path.read_bytes()
    except OSError as error:
        raise RepairFixtureError(RepairFixtureErrorCode.MANIFEST_UNAVAILABLE) from error
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise RepairFixtureError(RepairFixtureErrorCode.MANIFEST_UNAVAILABLE)
    try:
        manifest = RepairFixtureManifest.model_validate_json(payload)
    except (ValidationError, UnicodeDecodeError, ValueError) as error:
        raise RepairFixtureError(RepairFixtureErrorCode.MANIFEST_INVALID) from error

    try:
        fixture_root = path.parent.resolve(strict=True)
    except OSError as error:
        raise RepairFixtureError(RepairFixtureErrorCode.MANIFEST_UNAVAILABLE) from error

    loaded_cases: list[RepairFixtureCase] = []
    for case in manifest.cases:
        raw_repository = case.evaluation_task.task.repository_path
        if raw_repository is None:
            raise RepairFixtureError(RepairFixtureErrorCode.MANIFEST_INVALID)
        repository = _resolve_repository(fixture_root, raw_repository)
        snapshot = snapshot_repair_repository(repository)
        if snapshot.sha256 != case.repository_sha256:
            raise RepairFixtureError(RepairFixtureErrorCode.SNAPSHOT_MISMATCH)
        if snapshot.base_commit != case.evaluation_task.task.base_commit:
            raise RepairFixtureError(RepairFixtureErrorCode.BASE_COMMIT_MISMATCH)
        loaded_task = case.evaluation_task.task.model_copy(
            update={"repository_path": str(repository)}
        )
        loaded_evaluation = case.evaluation_task.model_copy(update={"task": loaded_task})
        loaded_cases.append(case.model_copy(update={"evaluation_task": loaded_evaluation}))
    return manifest.model_copy(update={"cases": tuple(loaded_cases)})


def snapshot_repair_repository(repository: Path) -> RepositoryFixtureSnapshot:
    """Hash regular fixture files with canonical path/length framing."""

    try:
        root = repository.resolve(strict=True)
    except OSError as error:
        raise RepairFixtureError(RepairFixtureErrorCode.REPOSITORY_UNAVAILABLE) from error
    if not root.is_dir():
        raise RepairFixtureError(RepairFixtureErrorCode.REPOSITORY_UNAVAILABLE)

    sha1 = hashlib.sha1(usedforsecurity=False)
    sha256 = hashlib.sha256()
    file_count = 0
    byte_count = 0
    try:
        paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        for candidate in paths:
            relative = candidate.relative_to(root)
            if candidate.is_symlink() or any(
                part in _DENIED_REPOSITORY_NAMES for part in relative.parts
            ):
                raise RepairFixtureError(RepairFixtureErrorCode.REPOSITORY_UNAVAILABLE)
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise RepairFixtureError(RepairFixtureErrorCode.REPOSITORY_UNAVAILABLE)
            content = candidate.read_bytes()
            file_count += 1
            byte_count += len(content)
            if file_count > _MAX_REPOSITORY_FILES or byte_count > _MAX_REPOSITORY_BYTES:
                raise RepairFixtureError(RepairFixtureErrorCode.REPOSITORY_UNAVAILABLE)
            path_bytes = relative.as_posix().encode("utf-8")
            frame = (
                len(path_bytes).to_bytes(4, "big")
                + path_bytes
                + len(content).to_bytes(8, "big")
                + content
            )
            sha1.update(frame)
            sha256.update(frame)
    except OSError as error:
        raise RepairFixtureError(RepairFixtureErrorCode.REPOSITORY_UNAVAILABLE) from error
    if file_count == 0 or byte_count == 0:
        raise RepairFixtureError(RepairFixtureErrorCode.REPOSITORY_UNAVAILABLE)
    return RepositoryFixtureSnapshot(
        base_commit=sha1.hexdigest(),
        sha256=sha256.hexdigest(),
        file_count=file_count,
        byte_count=byte_count,
    )


def _resolve_repository(fixture_root: Path, relative_path: str) -> Path:
    pure_path = PurePosixPath(relative_path)
    if (
        pure_path.is_absolute()
        or not pure_path.parts
        or ".." in pure_path.parts
        or "\\" in relative_path
    ):
        raise RepairFixtureError(RepairFixtureErrorCode.PATH_VIOLATION)
    try:
        resolved = (fixture_root / Path(*pure_path.parts)).resolve(strict=True)
        resolved.relative_to(fixture_root)
    except (OSError, ValueError) as error:
        raise RepairFixtureError(RepairFixtureErrorCode.PATH_VIOLATION) from error
    if not resolved.is_dir():
        raise RepairFixtureError(RepairFixtureErrorCode.REPOSITORY_UNAVAILABLE)
    return resolved


def _declared_patch_paths(patch: str) -> tuple[str, ...]:
    matches = _DIFF_HEADER.findall(patch)
    if not matches or "@@" not in patch or "\x00" in patch:
        raise ValueError("repair fixture patch must be a bounded unified diff")
    declared: list[str] = []
    for before, after in matches:
        if before != after:
            raise ValueError("repair fixture patches cannot rename files")
        try:
            canonical = _PatchPath(relative_path=before).relative_path
        except ValidationError as error:
            raise ValueError("repair fixture patch path is invalid") from error
        if f"--- a/{canonical}\n" not in patch or f"+++ b/{canonical}\n" not in patch:
            raise ValueError("repair fixture patch headers are incomplete")
        declared.append(canonical)
    if len(set(declared)) != len(declared):
        raise ValueError("repair fixture patch paths must be unique")
    return tuple(sorted(declared))


class _PatchPath(StrictModel):
    relative_path: RelativePath
