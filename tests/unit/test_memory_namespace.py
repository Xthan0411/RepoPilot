"""Experiment/snapshot/split namespace and access-control contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from repopilot.enums import (
    DatasetSplit,
    MemoryAccessMode,
    MemoryNamespaceErrorCode,
    RepositoryMemoryLookupStatus,
)
from repopilot.memory import (
    FileMemoryNamespaceManager,
    MemoryNamespaceError,
    build_repository_map_record,
)
from repopilot.retrieval import (
    RepositoryIgnorePolicy,
    RepositoryManifest,
    RepositorySource,
    source_id_for,
)
from repopilot.schemas import MemoryNamespace

NOW = datetime(2026, 8, 7, 18, 0, tzinfo=UTC)
REPOSITORY_ID = "repo_namespace_fixture"
COMMIT = "a" * 40


def namespace(
    *,
    snapshot_id: str = "snapshot-v1",
    split: DatasetSplit = DatasetSplit.DEV,
    access_mode: MemoryAccessMode = MemoryAccessMode.READ_WRITE,
) -> MemoryNamespace:
    return MemoryNamespace(
        experiment_id="memory-isolation",
        snapshot_id=snapshot_id,
        split=split,
        access_mode=access_mode,
    )


def manifest() -> RepositoryManifest:
    path = "pkg/parser.py"
    return RepositoryManifest.build(
        repository_id=REPOSITORY_ID,
        commit=COMMIT,
        ignore_policy=RepositoryIgnorePolicy(),
        files=(
            RepositorySource(
                source_id=source_id_for(REPOSITORY_ID, COMMIT, path),
                relative_path=path,
                size_bytes=100,
                content_hash="1" * 64,
            ),
        ),
    )


def test_namespace_manifest_is_private_integrity_protected_and_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "memory"
    manager = FileMemoryNamespaceManager(root)
    requested = namespace()

    first = manager.open(requested, created_at=NOW)
    second = manager.open(requested, created_at=NOW.replace(hour=19))
    path = (
        root
        / requested.experiment_id
        / requested.snapshot_id
        / requested.split.value
        / "namespace.json"
    )

    assert first.manifest == second.manifest
    assert first.manifest.created_at == NOW
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert json.loads(path.read_text(encoding="utf-8"))["namespace"] == {
        "access_mode": "read_write",
        "experiment_id": "memory-isolation",
        "snapshot_id": "snapshot-v1",
        "split": "dev",
    }


def test_dev_and_holdout_are_disjoint_and_holdout_cannot_write(
    tmp_path: Path,
) -> None:
    manager = FileMemoryNamespaceManager(tmp_path / "memory")
    dev = manager.open(namespace(), created_at=NOW)
    next_snapshot = manager.open(namespace(snapshot_id="snapshot-v2"), created_at=NOW)
    holdout_namespace = namespace(
        split=DatasetSplit.HOLDOUT,
        access_mode=MemoryAccessMode.READ_ONLY,
    )
    holdout = manager.open(holdout_namespace, created_at=NOW)
    current_manifest = manifest()
    record = build_repository_map_record(current_manifest, created_at=NOW)

    assert dev.repository.put(record, manifest=current_manifest)
    assert dev.repository.lookup(current_manifest).status is RepositoryMemoryLookupStatus.CURRENT
    assert (
        next_snapshot.repository.lookup(current_manifest).status
        is RepositoryMemoryLookupStatus.EMPTY
    )
    assert holdout.repository.lookup(current_manifest).status is RepositoryMemoryLookupStatus.EMPTY
    with pytest.raises(MemoryNamespaceError) as captured:
        holdout.repository.put(record, manifest=current_manifest)
    assert captured.value.code is MemoryNamespaceErrorCode.ACCESS_DENIED


def test_namespace_rejects_mutable_snapshot_holdout_write_and_relative_root() -> None:
    with pytest.raises(ValidationError, match="frozen snapshot"):
        namespace(snapshot_id="latest")
    with pytest.raises(ValidationError, match="read-only"):
        namespace(
            split=DatasetSplit.HOLDOUT,
            access_mode=MemoryAccessMode.READ_WRITE,
        )
    with pytest.raises(MemoryNamespaceError) as captured:
        FileMemoryNamespaceManager(Path("relative-memory"))
    assert captured.value.code is MemoryNamespaceErrorCode.PATH_UNSAFE


def test_namespace_revalidates_copied_models_and_freezes_access_mode(
    tmp_path: Path,
) -> None:
    manager = FileMemoryNamespaceManager(tmp_path / "memory")
    requested = namespace()
    manager.open(requested, created_at=NOW)

    with pytest.raises(MemoryNamespaceError) as access:
        manager.open(
            requested.model_copy(update={"access_mode": MemoryAccessMode.READ_ONLY}),
            created_at=NOW,
        )
    assert access.value.code is MemoryNamespaceErrorCode.ACCESS_DENIED

    invalid_holdout = requested.model_copy(update={"split": DatasetSplit.HOLDOUT})
    with pytest.raises(MemoryNamespaceError) as invalid:
        manager.open(invalid_holdout, created_at=NOW)
    assert invalid.value.code is MemoryNamespaceErrorCode.INVALID_NAMESPACE


def test_namespace_detects_manifest_tampering_and_symlink_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "memory"
    manager = FileMemoryNamespaceManager(root)
    requested = namespace()
    manager.open(requested, created_at=NOW)
    path = (
        root
        / requested.experiment_id
        / requested.snapshot_id
        / requested.split.value
        / "namespace.json"
    )
    payload = path.read_text(encoding="utf-8").replace("snapshot-v1", "snapshot-v2")
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(MemoryNamespaceError) as tampered:
        manager.open(requested, created_at=NOW)
    assert tampered.value.code is MemoryNamespaceErrorCode.INTEGRITY_MISMATCH

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked-memory"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(MemoryNamespaceError) as symlink:
        FileMemoryNamespaceManager(link).open(namespace(), created_at=NOW)
    assert symlink.value.code is MemoryNamespaceErrorCode.PATH_UNSAFE
