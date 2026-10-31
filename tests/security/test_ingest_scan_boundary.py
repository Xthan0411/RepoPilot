"""The wider internal inventory scan retains the repository security boundaries."""

from pathlib import Path

import pytest

from repopilot.enums import RepositoryErrorCode, RepositoryIngestErrorCode
from repopilot.retrieval import (
    RepositoryIngestError,
    RepositoryIngestScanPolicy,
    ingest_repository,
)
from repopilot.tools import ReadOnlyRepository, RepositoryAccessPolicy, RepositoryTextFile

pytestmark = pytest.mark.security


def test_internal_limit_fails_before_reading_any_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("value = 1\n", encoding="utf-8")

    def forbidden_read(self: ReadOnlyRepository, path: str) -> RepositoryTextFile:
        pytest.fail("truncated inventories must be rejected before any source reads")

    monkeypatch.setattr(ReadOnlyRepository, "read_text_file", forbidden_read)
    with pytest.raises(RepositoryIngestError) as captured:
        ingest_repository(
            ReadOnlyRepository(tmp_path),
            repository_id="security.repo",
            commit="a" * 40,
            scan_policy=RepositoryIngestScanPolicy(max_list_entries=2),
        )
    assert captured.value.code is RepositoryIngestErrorCode.LISTING_TRUNCATED


def test_internal_scan_does_not_traverse_ignored_or_custom_denied_directories(
    tmp_path: Path,
) -> None:
    (tmp_path / "visible.py").write_text("visible = 1\n", encoding="utf-8")
    for name in (".git", ".env", ".venv", "__pycache__", "private"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "hidden.py").write_text("secret_marker = True\n", encoding="utf-8")
    repository = ReadOnlyRepository(
        tmp_path,
        policy=RepositoryAccessPolicy(
            max_list_entries=1,
            denied_names=(".env", ".git", ".hg", ".svn", "private"),
        ),
    )
    manifest = ingest_repository(
        repository,
        repository_id="security.repo",
        commit="a" * 40,
        scan_policy=RepositoryIngestScanPolicy(max_list_entries=1),
    )

    assert tuple(source.relative_path for source in manifest.files) == ("visible.py",)
    assert "hidden.py" not in manifest.model_dump_json()
    assert "secret_marker" not in manifest.model_dump_json()
    assert repository.policy.max_list_entries == 1


def test_internal_scan_never_follows_external_file_or_directory_symlinks(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("outside_secret = True\n", encoding="utf-8")
    (repository_root / "linked-file.py").symlink_to(outside / "secret.py")
    (repository_root / "linked-directory").symlink_to(outside, target_is_directory=True)
    (repository_root / "safe.py").write_text("safe = 1\n", encoding="utf-8")

    manifest = ingest_repository(
        ReadOnlyRepository(repository_root), repository_id="security.repo", commit="a" * 40
    )

    assert tuple(source.relative_path for source in manifest.files) == ("safe.py",)
    assert tuple(source.relative_path for source in manifest.skipped_sources) == (
        "linked-directory",
        "linked-file.py",
    )
    assert all(
        source.reason is RepositoryErrorCode.SYMLINK_FORBIDDEN
        for source in manifest.skipped_sources
    )
    assert "secret.py" not in manifest.model_dump_json()
    assert "outside_secret" not in manifest.model_dump_json()


def test_independent_scan_keeps_caller_file_size_limit(tmp_path: Path) -> None:
    (tmp_path / "small.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "oversized.py").write_text("x" * 9, encoding="utf-8")
    repository = ReadOnlyRepository(
        tmp_path,
        policy=RepositoryAccessPolicy(max_list_entries=1, max_file_bytes=8, max_read_bytes=8),
    )
    manifest = ingest_repository(repository, repository_id="security.repo", commit="a" * 40)

    assert tuple(source.relative_path for source in manifest.files) == ("small.py",)
    assert len(manifest.skipped_sources) == 1
    assert manifest.skipped_sources[0].relative_path == "oversized.py"
    assert manifest.skipped_sources[0].reason is RepositoryErrorCode.FILE_TOO_LARGE
    assert repository.policy.max_list_entries == 1
    assert repository.policy.max_file_bytes == 8
