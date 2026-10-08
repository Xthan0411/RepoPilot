"""Persistent Sandbox artifact namespace, integrity, and overwrite tests."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path
from uuid import UUID

import pytest

from repopilot.sandbox import SandboxArtifactError, SandboxArtifactStore

CALL_ID = UUID("11111111-1111-1111-1111-111111111111")
OTHER_CALL_ID = UUID("22222222-2222-2222-2222-222222222222")


def test_artifact_store_writes_immutable_hashed_files_in_stable_order(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    root = tmp_path / "artifacts"
    store = SandboxArtifactStore(root)
    store.prepare("sbx-artifact-test", repository_source=repository)

    stderr = store.persist(CALL_ID, "stderr", b"failure\n")
    stdout = store.persist(CALL_ID, "stdout", b"ok\n")

    assert stdout.relative_path == (
        "artifacts/sbx-artifact-test/11111111-1111-1111-1111-111111111111.stdout.txt"
    )
    assert stdout.sha256 == hashlib.sha256(b"ok\n").hexdigest()
    assert stdout.size_bytes == 3
    assert store.collect() == (stderr, stdout)
    destination = root / "sbx-artifact-test" / "11111111-1111-1111-1111-111111111111.stdout.txt"
    assert destination.read_bytes() == b"ok\n"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600

    with pytest.raises(SandboxArtifactError, match="already exists"):
        store.persist(CALL_ID, "stdout", b"replacement")
    assert destination.read_bytes() == b"ok\n"

    externally_created = (
        root / "sbx-artifact-test" / "22222222-2222-2222-2222-222222222222.stdout.txt"
    )
    externally_created.write_bytes(b"external-canary")
    with pytest.raises(SandboxArtifactError, match="write failed"):
        store.persist(OTHER_CALL_ID, "stdout", b"must-not-overwrite")
    assert externally_created.read_bytes() == b"external-canary"


def test_artifact_store_rejects_nested_source_symlink_and_reprepare(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    nested = SandboxArtifactStore(repository / "artifacts")
    with pytest.raises(SandboxArtifactError, match="separate"):
        nested.prepare("sbx-nested", repository_source=repository)

    link = tmp_path / "artifact-link"
    link.symlink_to(tmp_path / "target", target_is_directory=True)
    with pytest.raises(SandboxArtifactError, match="symlink"):
        SandboxArtifactStore(link)

    store = SandboxArtifactStore(tmp_path / "separate-artifacts")
    store.prepare("sbx-once", repository_source=repository)
    with pytest.raises(SandboxArtifactError, match="already prepared"):
        store.prepare("sbx-twice", repository_source=repository)


def test_patch_artifact_is_immutable_and_reverified_before_use(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    root = tmp_path / "artifacts"
    store = SandboxArtifactStore(root)
    store.prepare("sbx-patch", repository_source=repository)
    content = b"diff --git a/a.py b/a.py\n"

    artifact = store.persist_patch(CALL_ID, content)

    assert artifact.relative_path.endswith(f"{CALL_ID}.patch.diff")
    assert artifact.media_type == "text/x-diff; charset=utf-8"
    assert store.read(artifact) == content

    destination = root / "sbx-patch" / f"{CALL_ID}.patch.diff"
    destination.write_bytes(b"tampered")
    with pytest.raises(SandboxArtifactError, match="integrity"):
        store.read(artifact)


@pytest.mark.parametrize("stream_name", ["", "diff", "STDOUT"])
def test_artifact_store_rejects_unrecognized_stream(
    tmp_path: Path,
    stream_name: str,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    store = SandboxArtifactStore(tmp_path / "artifacts")
    store.prepare("sbx-stream", repository_source=repository)

    with pytest.raises(SandboxArtifactError, match="stream"):
        store.persist(CALL_ID, stream_name, b"content")
